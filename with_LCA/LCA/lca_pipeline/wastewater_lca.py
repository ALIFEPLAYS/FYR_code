from __future__ import annotations

from .human_health_daly import add_human_health_method
from .ecosystem_quality import add_ecosystem_method
from .lcia_methods import (
    core_lcia_methods,
    regionalise_foreground_utility_water,
)


def run_wastewater_lca(superpro_outputs, settings):
    """Run the existing wastewater discharge/reuse route-factor logic in memory."""
    import numpy as np
    import pandas as pd
    import bw2data as bd
    import bw2calc as bc

    display = lambda *args, **kwargs: None
    raw = superpro_outputs.copy()
    ECOINVENT_VERSION = settings["brightway"]["ecoinvent_version"]
    ECOINVENT_SYSTEM_MODEL = settings["brightway"]["system_model"]
    from datetime import datetime
    DATE_TAG = datetime.now().strftime("%Y%m%d")

    # ---------------------------------------------------------------------
    # Required fields and numeric normalisation
    # ---------------------------------------------------------------------

    REQUIRED_COLS = [
        "inventory_id",
        "source_report",
        "source_table",
        "broad_process_area",
        "superpro_section",
        "unit_operation",
        "source_procedure",
        "destination",
        "flow_name_superpro",
        "component",
        "amount_per_batch",
        "unit_per_batch",
        "amount_per_year",
        "unit_per_year",
        "stream_volume_L_per_batch",
        "stream_volume_L_per_year",
        "stream_component_sum_kg_per_batch",
        "stream_total_minus_component_sum_kg_per_batch",
        "source_row",
        "source_column",
    ]

    missing_cols = [col for col in REQUIRED_COLS if col not in raw.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {SHEET_NAME}: {missing_cols}")

    df = raw.copy()

    for col in [
        "amount_per_batch",
        "amount_per_year",
        "amount_per_kg_mp",
        "stream_volume_L_per_batch",
        "stream_volume_L_per_year",
        "stream_component_sum_kg_per_batch",
        "stream_total_minus_component_sum_kg_per_batch",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in [
        "source_report",
        "source_table",
        "broad_process_area",
        "superpro_section",
        "unit_operation",
        "source_procedure",
        "destination",
        "flow_name_superpro",
        "component",
        "unit_per_batch",
        "unit_per_year",
    ]:
        df[col] = df[col].astype(str).str.strip()

    stream_details = df[
        df["source_report"].eq("SR")
        & df["source_table"].eq("3. STREAM DETAILS")
        & df["destination"].eq("OUTPUT")
    ].copy()

    print(f"Output stream-detail rows: {len(stream_details):,}")
    print(f"Output streams: {stream_details['flow_name_superpro'].nunique():,}")


    # ---------------------------------------------------------------------
    # Build one row per SuperPro output stream with component metadata
    # ---------------------------------------------------------------------

    TOTAL_COMPONENT = "TOTAL"
    WATER_COMPONENT = "WFI"
    GAS_COMPONENTS = {"Nitrogen", "Oxygen", "Carb. Dioxide", "Carbon Dioxide", "CO2"}
    PRODUCT_FLOW_NAMES = {"Product"}

    stream_rows = []
    component_rows = []

    for flow_name, g in stream_details.groupby("flow_name_superpro", dropna=False):
        total_rows = g[g["component"].eq(TOTAL_COMPONENT)]
        if len(total_rows) != 1:
            display(g)
            raise ValueError(f"Expected exactly one TOTAL row for stream {flow_name}, found {len(total_rows)}")

        total = total_rows.iloc[0]
        comps = g[~g["component"].eq(TOTAL_COMPONENT)].copy()

        component_names = set(comps["component"].dropna().astype(str))
        water_kg = comps.loc[comps["component"].eq(WATER_COMPONENT), "amount_per_batch"].sum()
        nonwater_component_kg = comps.loc[~comps["component"].eq(WATER_COMPONENT), "amount_per_batch"].sum()
        gas_component_kg = comps.loc[comps["component"].isin(GAS_COMPONENTS), "amount_per_batch"].sum()
        mAb_kg = comps.loc[comps["component"].eq("mAb"), "amount_per_batch"].sum()
        biomass_kg = comps.loc[comps["component"].eq("Biomass"), "amount_per_batch"].sum()

        total_kg = total["amount_per_batch"]
        volume_L = total["stream_volume_L_per_batch"]
        volume_m3 = volume_L / 1000 if pd.notna(volume_L) else np.nan

        is_product = (str(flow_name) in PRODUCT_FLOW_NAMES) or ("Product" in str(total["unit_operation"]))
        is_named_vent = str(flow_name).lower().startswith("vent")
        is_gas_only = (water_kg == 0) and component_names.issubset(GAS_COMPONENTS) and len(component_names) > 0
        is_gas = is_named_vent or is_gas_only

        is_aqueous_output = (water_kg > 0) and (not is_product) and (not is_gas)

        if is_product:
            exclusion_reason = "product_stream"
        elif is_gas:
            exclusion_reason = "gas_or_vent_stream"
        elif water_kg <= 0:
            exclusion_reason = "no_WFI_component_detected"
        else:
            exclusion_reason = "included_aqueous_output"

        # Water quality class is recorded for later constraints. Reuse quantity is not fixed here.
        contaminant_components = sorted([c for c in component_names if c != WATER_COMPONENT])
        if not is_aqueous_output:
            water_quality_class = "not_wastewater_inventory"
            default_reuse_fraction = 0.0
        elif set(contaminant_components) == set():
            water_quality_class = "wfi_only_low_contaminant_process_water"
            default_reuse_fraction = np.nan
        elif biomass_kg > 0 or "Media" in component_names:
            water_quality_class = "biological_biomass_or_media_containing_wastewater"
            default_reuse_fraction = np.nan
        elif any(c in component_names for c in ["Sodium Hydroxid", "Sodium Hydroxide", "Acetic-Acid", "TRIS Base", "TRIS HCl", "Sodium Chloride", "Na2HPO4", "Sodium Phosphat", "HEPES"]):
            water_quality_class = "buffer_salt_or_cleaning_wastewater"
            default_reuse_fraction = np.nan
        elif mAb_kg > 0:
            water_quality_class = "product_residue_containing_wastewater"
            default_reuse_fraction = np.nan
        else:
            water_quality_class = "aqueous_output_unclassified_review"
            default_reuse_fraction = np.nan

        stream_rows.append({
            "flow_name_superpro": flow_name,
            "inventory_id_total": total["inventory_id"],
            "source_report": total["source_report"],
            "source_table": total["source_table"],
            "broad_process_area": total["broad_process_area"],
            "superpro_section": total["superpro_section"],
            "unit_operation": total["unit_operation"],
            "source_procedure": total["source_procedure"],
            "destination": total["destination"],
            "total_kg_per_batch": total_kg,
            "water_kg_per_batch": water_kg,
            "nonwater_component_kg_per_batch": nonwater_component_kg,
            "gas_component_kg_per_batch": gas_component_kg,
            "mAb_kg_per_batch": mAb_kg,
            "biomass_kg_per_batch": biomass_kg,
            "stream_volume_L_per_batch": volume_L,
            "stream_volume_m3_per_batch": volume_m3,
            "water_fraction_of_total_mass": water_kg / total_kg if total_kg and pd.notna(total_kg) else np.nan,
            "nonwater_fraction_of_total_mass": (total_kg - water_kg) / total_kg if total_kg and pd.notna(total_kg) else np.nan,
            "density_kg_per_L_from_total_and_volume": total_kg / volume_L if volume_L and pd.notna(volume_L) else np.nan,
            "component_list": "; ".join(sorted(component_names)),
            "contaminant_component_list": "; ".join(contaminant_components),
            "is_product": is_product,
            "is_gas": is_gas,
            "is_aqueous_output": is_aqueous_output,
            "exclusion_reason": exclusion_reason,
            "water_quality_class": water_quality_class,
            "default_reuse_fraction_for_reuse_route": default_reuse_fraction,
            "mapping_note": "Selected if WFI component > 0 and stream is not product/gas. Component list retained; no COD/BOD/pH inferred.",
            "source_row_total": total["source_row"],
            "source_column_total": total["source_column"],
        })

        for _, c in comps.iterrows():
            component_rows.append({
                "flow_name_superpro": flow_name,
                "inventory_id_component": c["inventory_id"],
                "component": c["component"],
                "component_kg_per_batch": c["amount_per_batch"],
                "source_row": c["source_row"],
                "source_column": c["source_column"],
            })

    output_stream_classification = pd.DataFrame(stream_rows)
    wastewater_components = pd.DataFrame(component_rows)

    wastewater_inventory = output_stream_classification[
        output_stream_classification["is_aqueous_output"].eq(True)
    ].copy()

    excluded_output_streams = output_stream_classification[
        output_stream_classification["is_aqueous_output"].eq(False)
    ].copy()

    print(f"Aqueous output streams included: {len(wastewater_inventory):,}")
    print(f"Excluded output streams: {len(excluded_output_streams):,}")
    print(f"Total aqueous output mass: {wastewater_inventory['total_kg_per_batch'].sum():,.2f} kg/batch")
    print(f"Total WFI/water in aqueous outputs: {wastewater_inventory['water_kg_per_batch'].sum():,.2f} kg/batch")
    print(f"Total aqueous output volume: {wastewater_inventory['stream_volume_m3_per_batch'].sum():,.3f} m3/batch")

    display(wastewater_inventory.sort_values("total_kg_per_batch", ascending=False))
    display(excluded_output_streams[["flow_name_superpro", "unit_operation", "total_kg_per_batch", "stream_volume_L_per_batch", "component_list", "exclusion_reason"]].sort_values("exclusion_reason"))


    # ---------------------------------------------------------------------
    # Validation checks
    # ---------------------------------------------------------------------
    # These checks catch the most likely mapping mistakes:
    #   - accidentally including gas vents as wastewater;
    #   - accidentally including product as wastewater;
    #   - including rows without water/WFI;
    #   - unrealistic total-minus-component imbalances.

    if wastewater_inventory["flow_name_superpro"].str.lower().str.startswith("vent").any():
        raise ValueError("A vent stream has entered the wastewater inventory. Review stream classification.")

    if wastewater_inventory["flow_name_superpro"].isin(PRODUCT_FLOW_NAMES).any():
        raise ValueError("The product stream has entered the wastewater inventory. Review stream classification.")

    if (wastewater_inventory["water_kg_per_batch"] <= 0).any():
        display(wastewater_inventory[wastewater_inventory["water_kg_per_batch"] <= 0])
        raise ValueError("At least one selected wastewater stream has no WFI/water component.")

    # Component sum check using the original TOTAL rows.
    component_balance_check = stream_details[stream_details["component"].eq(TOTAL_COMPONENT)][
        [
            "flow_name_superpro",
            "amount_per_batch",
            "stream_component_sum_kg_per_batch",
            "stream_total_minus_component_sum_kg_per_batch",
        ]
    ].rename(columns={"amount_per_batch": "total_kg_per_batch"})

    component_balance_check = component_balance_check.merge(
        wastewater_inventory[["flow_name_superpro", "is_aqueous_output"]],
        on="flow_name_superpro",
        how="left",
    )
    component_balance_check["is_aqueous_output"] = component_balance_check["is_aqueous_output"].fillna(False)
    component_balance_check["absolute_balance_gap_kg_per_batch"] = component_balance_check["stream_total_minus_component_sum_kg_per_batch"].abs()

    bad_balances = component_balance_check[
        component_balance_check["is_aqueous_output"]
        & (component_balance_check["absolute_balance_gap_kg_per_batch"] > 0.05)
    ]

    if len(bad_balances) > 0:
        display(bad_balances)
        raise ValueError("One or more selected wastewater streams has a material balance gap > 0.05 kg/batch.")

    print("Validation passed.")
    display(component_balance_check[component_balance_check["is_aqueous_output"]].sort_values("absolute_balance_gap_kg_per_batch", ascending=False))


    # ---------------------------------------------------------------------
    # Brightway project and databases
    # ---------------------------------------------------------------------
    # Keep these names aligned with the local Brightway project used for the other LCA notebooks.

    import bw2data as bd
    import bw2calc as bc

    ECOINVENT_VERSION = settings["brightway"]["ecoinvent_version"]
    ECOINVENT_SYSTEM_MODEL = settings["brightway"]["system_model"]

    BRIGHTWAY_PROJECT = settings["brightway"]["project"]
    BACKGROUND_DB = settings["brightway"]["background_database"]
    BIOSPHERE_DB = settings["brightway"]["biosphere_database"]

    bd.projects.set_current(BRIGHTWAY_PROJECT)

    if not (BACKGROUND_DB in bd.databases):
        raise AssertionError(f'Missing background database: {BACKGROUND_DB}')
    if not (BIOSPHERE_DB in bd.databases):
        raise AssertionError(f'Missing biosphere database: {BIOSPHERE_DB}')

    print(f"Project: {bd.projects.current}")
    print(f"Background database: {BACKGROUND_DB}")
    print(f"Biosphere database: {BIOSPHERE_DB}")

    # ---------------------------------------------------------------------
    # LCIA methods — ecoinvent 3.12 hardcoded methods
    # ---------------------------------------------------------------------
    # Three midpoint categories used for the LCA factor table:
    #   1. GWP
    #   2. fossil resource use
    #   3. water scarcity
    #
    # ecoinvent 3.12 keeps ReCiPe 2016 v1.03 available.
    # IMPACT World+ footprint version is updated from v2.0.1 to v2.1 in ecoinvent 3.12.

    LCIA_METHODS = core_lcia_methods()
    LCIA_METHODS = add_human_health_method(LCIA_METHODS)
    LCIA_METHODS = add_ecosystem_method(LCIA_METHODS)

    missing_methods = {
        label: method
        for label, method in LCIA_METHODS.items()
        if method not in bd.methods
    }

    if missing_methods:
        display(pd.DataFrame(
            [{"impact_category": k, "method": v} for k, v in missing_methods.items()]
        ))
        raise ValueError("One or more LCIA methods are not available in this Brightway project.")

    for label, method in LCIA_METHODS.items():
        print(f"{label}: {method}")

    # ---------------------------------------------------------------------
    # Brightway activity helpers: exact metadata mapping, no ranked searches
    # ---------------------------------------------------------------------


    def activity_metadata(activity, mapping_type, activity_search, assumption_note=None):
        return {
            "activity_name": activity.get("name"),
            "reference_product": activity.get("reference product"),
            "location": activity.get("location"),
            "unit": activity.get("unit"),
            "database": activity.key[0],
            "code": activity.key[1],
            "mapping_type": mapping_type,
            "activity_search": activity_search,
            "assumption_note": assumption_note,
        }


    def exact_activity_table(database, *, name=None, reference_product=None, location=None, unit=None):
        """Return activities matching the provided metadata exactly.

        This deliberately avoids broad free-text ranking. It is used here because the wastewater
        mapping should be fixed to the reviewed generic wastewater treatment activity, not whichever
        industry-specific candidate happens to rank first.
        """
        rows = []
        for act in bd.Database(database):
            if name is not None and act.get("name") != name:
                continue
            if reference_product is not None and act.get("reference product") != reference_product:
                continue
            if location is not None and act.get("location") != location:
                continue
            if unit is not None and act.get("unit") != unit:
                continue
            rows.append(activity_metadata(act, "exact_metadata_candidate", "exact metadata lookup"))
        return pd.DataFrame(rows)


    def get_activity_by_exact_metadata(database, *, name, reference_product, location, unit):
        matches = exact_activity_table(
            database,
            name=name,
            reference_product=reference_product,
            location=location,
            unit=unit,
        )
        if len(matches) != 1:
            print("Exact metadata lookup did not return exactly one activity.")
            print("Requested:")
            print({
                "database": database,
                "name": name,
                "reference_product": reference_product,
                "location": location,
                "unit": unit,
            })
            if len(matches) > 0:
                display(matches)
            raise ValueError("Fill/review WASTEWATER_TREATMENT_EXACT_SPEC so it resolves to one ecoinvent activity.")
        row = matches.iloc[0]
        return bd.get_activity((row["database"], row["code"]))


    # ---------------------------------------------------------------------
    # Wastewater treatment activity mapping
    # ---------------------------------------------------------------------
    # Fixed exact mapping to a generic average wastewater treatment activity.
    # This avoids accidental selection of industry-specific wastewater datasets.

    WASTEWATER_TREATMENT_EXACT_SPEC = {
        "database": BACKGROUND_DB,
        "name": "treatment of wastewater, average, wastewater treatment",
        "reference_product": "wastewater, average",
        "location": "Europe without Switzerland",
        "unit": "cubic meter",
    }

    wastewater_treatment_activity = get_activity_by_exact_metadata(**WASTEWATER_TREATMENT_EXACT_SPEC)

    wastewater_mapping = pd.DataFrame([
        activity_metadata(
            wastewater_treatment_activity,
            mapping_type="fixed_exact_metadata_reviewed",
            activity_search="exact activity name + exact reference product + exact location + exact unit",
            assumption_note=(
                "Generic average wastewater treatment proxy for actual aqueous output. "
                "Selected by exact reviewed metadata to avoid accidental industry-specific wastewater activities."
            ),
        )
    ])

    print("Selected wastewater treatment activity:")
    display(wastewater_mapping)

    # ---------------------------------------------------------------------
    # Wastewater treatment impact factors
    # ---------------------------------------------------------------------
    # The selected wastewater treatment activity has a waste reference product:
    #   reference product = "wastewater, average"
    #
    # In ecoinvent/Brightway, waste-treatment activities can have reversed
    # signs because the activity represents accepting a waste flow. To model
    # treatment of +1 m3 physical wastewater, we demand -1 m3 of the
    # waste-treatment activity.
    #
    # This makes wastewater treatment a positive burden per m3 treated.
    # The raw +1 Brightway score is retained for transparency, but the
    # route/optimiser-facing factor uses the corrected -1 orientation.

    WASTEWATER_TREATMENT_DEMAND_AMOUNT = -1.0

    wastewater_treatment_factors = []

    for impact_category, method in LCIA_METHODS.items():
        # Raw +1 score retained only for diagnostics/transparency.
        lca_raw = bc.LCA({wastewater_treatment_activity: 1.0}, method)
        lca_raw.lci()
        lca_raw.lcia()
        raw_score_plus_1 = float(lca_raw.score)

        # Correct physical treatment score: treatment of +1 m3 wastewater.
        lca_corrected = bc.LCA({wastewater_treatment_activity: WASTEWATER_TREATMENT_DEMAND_AMOUNT}, method)
        lca_corrected.lci()
        lca_corrected.lcia()
        corrected_score = float(lca_corrected.score)

        wastewater_treatment_factors.append({
            "route_factor_name": "wastewater_treatment",
            "life_cycle_stage": "wastewater_treatment",
            "impact_category": impact_category,

            # Raw Brightway +1 score is kept for audit trail.
            "impact_per_activity_unit_raw": raw_score_plus_1,

            # Model-facing factor: positive burden per m3 physical wastewater treated.
            "impact_per_activity_unit": corrected_score,

            "activity_demand_amount": WASTEWATER_TREATMENT_DEMAND_AMOUNT,
            "activity_amount_unit": wastewater_treatment_activity.get("unit"),
            "sign_convention_adjusted_for_optimisation": True,
            "sign_convention_note": (
                "Wastewater treatment has waste reference product. "
                "Treatment of +1 m3 physical wastewater is modelled by demanding -1 m3 "
                "of the ecoinvent waste-treatment activity. Raw +1 Brightway score is retained separately."
            ),
            **activity_metadata(
                wastewater_treatment_activity,
                mapping_type=wastewater_mapping.iloc[0]["mapping_type"],
                activity_search=wastewater_mapping.iloc[0]["activity_search"],
                assumption_note=wastewater_mapping.iloc[0]["assumption_note"],
            ),
        })

    wastewater_treatment_factors = pd.DataFrame(wastewater_treatment_factors)

    display(wastewater_treatment_factors)

    # ---------------------------------------------------------------------
    # Utility-water offset activity mapping
    # ---------------------------------------------------------------------
    # The reuse route can replace non-contact utility water. This is modelled as
    # a negative burden applied only to the reused volume. WFI is deliberately not
    # used as the displaced product.
    #
    # Selection is restricted to reviewed exact candidate names/reference products.
    # If none exist in the local database, the notebook continues without a credit
    # and the missing mapping is visible in the outputs.

    RUN_UTILITY_WATER_OFFSET_CREDIT = bool(settings["wastewater"].get("run_utility_water_offset_credit", True))

    UTILITY_WATER_SUPPLY_CANDIDATE_SPECS = [
        {
            "database": BACKGROUND_DB,
            "name": "market for tap water",
            "reference_product": "tap water",
            "preferred_locations": ["Europe without Switzerland", "RER", "GLO", "RoW", "GB"],
            "accepted_units": ["cubic meter", "kilogram"],
        },
        {
            "database": BACKGROUND_DB,
            "name": "tap water production, conventional treatment",
            "reference_product": "tap water",
            "preferred_locations": ["Europe without Switzerland", "RER", "GLO", "RoW", "GB"],
            "accepted_units": ["cubic meter", "kilogram"],
        },
        {
            "database": BACKGROUND_DB,
            "name": "market for water, deionised",
            "reference_product": "water, deionised",
            "preferred_locations": ["Europe without Switzerland", "RER", "GLO", "RoW", "GB"],
            "accepted_units": ["cubic meter", "kilogram"],
        },
    ]


    def find_first_exact_candidate(specs):
        """Return the first available exact candidate activity from a reviewed list."""
        for spec in specs:
            candidates = []
            for act in bd.Database(spec["database"]):
                if act.get("name") != spec["name"]:
                    continue
                if act.get("reference product") != spec["reference_product"]:
                    continue
                if act.get("unit") not in spec["accepted_units"]:
                    continue
                candidates.append(act)

            if not candidates:
                continue

            for location in spec["preferred_locations"]:
                loc_matches = [act for act in candidates if act.get("location") == location]
                if loc_matches:
                    return loc_matches[0], spec

            candidates = sorted(candidates, key=lambda a: (str(a.get("location", "")), str(a.key[1])))
            return candidates[0], spec

        return None, None


    utility_water_supply_activity = None
    utility_water_supply_mapping = pd.DataFrame(columns=[
        "activity_name", "reference_product", "location", "unit", "database", "code",
        "mapping_type", "activity_search", "assumption_note",
    ])
    utility_water_offset_factors = pd.DataFrame(columns=[
        "route_factor_name", "life_cycle_stage", "impact_category",
        "impact_per_activity_unit_raw", "impact_per_activity_unit",
        "activity_amount_unit", "activity_unit_to_m3_multiplier",
        "activity_name", "reference_product", "location", "unit", "database", "code",
        "mapping_type", "activity_search", "assumption_note",
    ])

    if RUN_UTILITY_WATER_OFFSET_CREDIT:
        utility_water_supply_activity, selected_spec = find_first_exact_candidate(
            UTILITY_WATER_SUPPLY_CANDIDATE_SPECS
        )

        if utility_water_supply_activity is None:
            print(
                "Warning: no reviewed utility-water supply activity was found. "
                "The reuse route will expose reusable water as a secondary output but no avoided-water LCIA credit is applied."
            )
        else:
            # Demand amount required to represent 1 m3 of avoided utility water.
            # If the activity unit is kg, use 1000 kg/m3. If it is m3, use 1.
            unit = utility_water_supply_activity.get("unit")
            activity_unit_to_m3_multiplier = 1000.0 if unit == "kilogram" else 1.0

            utility_water_supply_mapping = pd.DataFrame([
                activity_metadata(
                    utility_water_supply_activity,
                    mapping_type="fixed_exact_metadata_reviewed_candidate_list",
                    activity_search="reviewed exact candidate list for non-contact utility water",
                    assumption_note=(
                        "Negative burden for displaced non-contact utility-water supply. "
                        "Credit is applied only to the reused volume to avoid double counting."
                    ),
                )
            ])

            rows = []
            for impact_category, method in LCIA_METHODS.items():
                lca = bc.LCA({utility_water_supply_activity: activity_unit_to_m3_multiplier}, method)
                lca.lci()
                lca.lcia()
                raw_score_per_m3 = float(lca.score)
                regionalisation_fields = {
                    "water_scarcity_regionalisation_status": "not_applicable",
                    "water_scarcity_native_world_cf": np.nan,
                    "water_scarcity_background_factor_per_m3": np.nan,
                    "water_scarcity_foreground_consumption_m3_per_m3": np.nan,
                }
                corrected_score_per_m3 = raw_score_per_m3
                if impact_category == "water_scarcity":
                    configured_water = settings["lcia"]["water_scarcity_regionalisation"]
                    site_factors = {
                        site: float(values["factor_m3_world_eq_per_m3"])
                        for site, values in configured_water["sites"].items()
                    }
                    regionalised = regionalise_foreground_utility_water(
                        lca=lca,
                        utility_water_activity=utility_water_supply_activity,
                        site_factors=site_factors,
                        country_factor=float(
                            configured_water["generic_scenario_factor_m3_world_eq_per_m3"]
                        ),
                    )
                    corrected_score_per_m3 = float(
                        regionalised["corrected_scores"]["UK_country_fallback"]
                    )
                    regionalisation_fields = {
                        "water_scarcity_regionalisation_status": (
                            "world_generic_brightway_foreground_replaced_with_AWARE20_site_factor"
                        ),
                        "water_scarcity_native_world_cf": regionalised["native_world_cf"],
                        "water_scarcity_background_factor_per_m3": -regionalised["background_score"],
                        "water_scarcity_foreground_consumption_m3_per_m3": (
                            -regionalised["foreground_consumption_m3"]
                        ),
                        "water_scarcity_provider_activity_ids": " | ".join(
                            map(str, regionalised["provider_activity_ids"])
                        ),
                    }
                    for site, score in regionalised["corrected_scores"].items():
                        if site != "UK_country_fallback":
                            regionalisation_fields[f"water_scarcity_factor__{site}"] = -float(score)

                rows.append({
                    "route_factor_name": "avoided_utility_water_supply",
                    "life_cycle_stage": "avoided_utility_water_supply",
                    "impact_category": impact_category,
                    "impact_per_activity_unit_raw": raw_score_per_m3,
                    "impact_per_activity_unit": -corrected_score_per_m3,
                    "activity_amount_unit": "m3 reused utility water",
                    "activity_unit_to_m3_multiplier": activity_unit_to_m3_multiplier,
                    **regionalisation_fields,
                    **activity_metadata(
                        utility_water_supply_activity,
                        mapping_type="fixed_exact_metadata_reviewed_candidate_list",
                        activity_search="reviewed exact candidate list for non-contact utility water",
                        assumption_note=(
                            "Avoided non-contact utility-water supply credit. Stored as a negative optimiser factor; "
                            "AWARE site correction applies only to the traced foreground tap-water chain."
                        ),
                    ),
                })
            utility_water_offset_factors = pd.DataFrame(rows)

    print("Utility-water offset credit enabled:", RUN_UTILITY_WATER_OFFSET_CREDIT)
    print("Utility-water offset mapped:", utility_water_supply_activity is not None)
    if len(utility_water_supply_mapping) > 0:
        display(utility_water_supply_mapping)
    if len(utility_water_offset_factors) > 0:
        display(utility_water_offset_factors)


    # ---------------------------------------------------------------------
    # Wastewater route options
    # ---------------------------------------------------------------------
    # Only two route options are defined here. This is not a scenario layer.
    #
    # Route 1: treatment and discharge.
    # Route 2: treatment for non-contact utility-water reuse.
    #
    # Route factors are per 1 m3 routed. The optimiser later chooses how much
    # wastewater goes to each route and must cap reuse by utility-water demand,
    # quality acceptance and treatment capacity.

    route_settings = settings["wastewater"].get("routes", {})
    discharge_settings = route_settings.get("treatment_and_discharge", {})
    reuse_settings = route_settings.get("treatment_for_utility_reuse", {})

    wastewater_route_options = pd.DataFrame([
        {
            "route": "treatment_and_discharge",
            "route_kind": "disposal",
            "reuse_fraction_per_m3_routed": float(discharge_settings.get("reuse_fraction_per_m3_routed", 0.0)),
            "discharge_fraction_per_m3_routed": float(discharge_settings.get("discharge_fraction_per_m3_routed", 1.0)),
            "treatment_fraction_per_m3_routed": float(discharge_settings.get("treatment_fraction_per_m3_routed", 1.0)),
            "include_utility_water_offset": False,
            "description": "Treat aqueous output to discharge standard, then discharge to sewer.",
        },
        {
            "route": "treatment_for_utility_reuse",
            "route_kind": "reuse",
            "reuse_fraction_per_m3_routed": float(reuse_settings.get("reuse_fraction_per_m3_routed", 1.0)),
            "discharge_fraction_per_m3_routed": float(reuse_settings.get("discharge_fraction_per_m3_routed", 0.0)),
            "treatment_fraction_per_m3_routed": float(reuse_settings.get("treatment_fraction_per_m3_routed", 1.0)),
            "include_utility_water_offset": len(utility_water_offset_factors) > 0,
            "description": "Treat routed aqueous output for non-contact utility-water reuse.",
        },
    ])

    display(wastewater_route_options)


    # ---------------------------------------------------------------------
    # Build detailed LCA rows for the two wastewater route options
    # ---------------------------------------------------------------------
    # Amount basis:
    #   - each route factor is expressed per m3 routed to that route;
    #   - treatment burden is applied to the routed m3;
    #   - avoided utility-water supply is applied only to the reuse route;
    #   - the physical reusable-water output is listed in secondary_outputs.
    #
    # The notebook does not decide route split. Per-batch values show the result
    # if the full eligible stream were routed to that option; the optimiser later
    # chooses the actual routed volumes.

    ACTIVITY_AMOUNT_BASIS = "m3"

    base = wastewater_inventory.copy()
    base["inventory_id"] = base["inventory_id_total"]
    base["components_present"] = base["component_list"]
    base["activity_amount_per_batch"] = base["stream_volume_m3_per_batch"]

    if base["activity_amount_per_batch"].isna().any():
        display(base[base["activity_amount_per_batch"].isna()])
        raise ValueError("Missing m3 activity amount for at least one wastewater stream.")

    lca_parts = []
    route_mass_rows = []

    for _, route in wastewater_route_options.iterrows():
        route_base = base.copy()

        reuse_fraction = float(route["reuse_fraction_per_m3_routed"])
        discharge_fraction = float(route["discharge_fraction_per_m3_routed"])
        treatment_fraction = float(route["treatment_fraction_per_m3_routed"])

        route_base["reuse_fraction"] = reuse_fraction
        route_base["discharge_fraction"] = discharge_fraction
        route_base["treated_fraction"] = treatment_fraction
        route_base["treated_activity_amount_per_batch"] = route_base["activity_amount_per_batch"] * treatment_fraction
        route_base["reused_activity_amount_per_batch"] = route_base["activity_amount_per_batch"] * reuse_fraction
        route_base["discharged_activity_amount_per_batch"] = route_base["activity_amount_per_batch"] * discharge_fraction

        treatment_part = route_base.merge(wastewater_treatment_factors, how="cross")
        for col, value in route.items():
            treatment_part[col] = value
        treatment_part["life_cycle_stage"] = "wastewater_treatment"
        treatment_part["routed_fraction"] = treatment_part["treated_fraction"]
        treatment_part["routed_activity_amount_per_batch"] = treatment_part["treated_activity_amount_per_batch"]
        treatment_part["impact_per_batch"] = treatment_part["routed_activity_amount_per_batch"] * treatment_part["impact_per_activity_unit"]
        treatment_part["routed_total_kg_per_batch"] = treatment_part["total_kg_per_batch"] * treatment_part["routed_fraction"]
        treatment_part["routed_water_kg_per_batch"] = treatment_part["water_kg_per_batch"] * treatment_part["routed_fraction"]
        treatment_part["routed_volume_m3_per_batch"] = treatment_part["stream_volume_m3_per_batch"] * treatment_part["routed_fraction"]
        treatment_part["is_credit"] = False
        treatment_part["credit_applied_to_secondary_output"] = False
        treatment_part["route_assumption_note"] = "Treatment burden applied per m3 routed."
        lca_parts.append(treatment_part)

        if bool(route["include_utility_water_offset"]):
            offset_part = route_base.merge(utility_water_offset_factors, how="cross")
            for col, value in route.items():
                offset_part[col] = value
            offset_part["life_cycle_stage"] = "avoided_utility_water_supply"
            offset_part["routed_fraction"] = offset_part["reuse_fraction"]
            offset_part["routed_activity_amount_per_batch"] = offset_part["reused_activity_amount_per_batch"]
            offset_part["impact_per_batch"] = offset_part["routed_activity_amount_per_batch"] * offset_part["impact_per_activity_unit"]
            offset_part["routed_total_kg_per_batch"] = offset_part["total_kg_per_batch"] * offset_part["routed_fraction"]
            offset_part["routed_water_kg_per_batch"] = offset_part["water_kg_per_batch"] * offset_part["routed_fraction"]
            offset_part["routed_volume_m3_per_batch"] = offset_part["stream_volume_m3_per_batch"] * offset_part["routed_fraction"]
            offset_part["is_credit"] = True
            offset_part["credit_applied_to_secondary_output"] = True
            offset_part["route_assumption_note"] = (
                "Negative burden for non-contact utility-water supply displaced by the reuse route. "
                "Do not add a second avoided-water credit from secondary_outputs."
            )
            lca_parts.append(offset_part)

        mass = route_base[[
            "inventory_id",
            "broad_process_area",
            "superpro_section",
            "unit_operation",
            "source_procedure",
            "destination",
            "flow_name_superpro",
            "components_present",
            "water_quality_class",
            "total_kg_per_batch",
            "water_kg_per_batch",
            "nonwater_component_kg_per_batch",
            "stream_volume_m3_per_batch",
            "activity_amount_per_batch",
            "reuse_fraction",
            "discharge_fraction",
            "treated_activity_amount_per_batch",
            "reused_activity_amount_per_batch",
            "discharged_activity_amount_per_batch",
        ]].copy()
        for col, value in route.items():
            mass[col] = value
        mass["treated_volume_m3_per_batch"] = mass["treated_activity_amount_per_batch"]
        mass["reused_volume_m3_per_batch"] = mass["reused_activity_amount_per_batch"]
        mass["discharged_volume_m3_per_batch"] = mass["discharged_activity_amount_per_batch"]
        mass["utility_water_credit_in_route_factor"] = bool(route["include_utility_water_offset"])
        route_mass_rows.append(mass)

    if len(lca_parts) == 0:
        raise ValueError("No wastewater LCA rows were generated.")

    wastewater_lca_detailed = pd.concat(lca_parts, ignore_index=True, sort=False)
    wastewater_route_mass_inventory = pd.concat(route_mass_rows, ignore_index=True, sort=False)

    print(f"Wastewater LCA detailed rows: {len(wastewater_lca_detailed):,}")
    print(f"Wastewater route mass inventory rows: {len(wastewater_route_mass_inventory):,}")
    display(wastewater_lca_detailed.head(20))
    display(wastewater_route_mass_inventory.head(20))


    # ---------------------------------------------------------------------
    # Secondary outputs from wastewater route options
    # ---------------------------------------------------------------------
    # This table exposes the physical outputs of the two wastewater routes for
    # later optimisation. It does not choose a route.

    secondary_rows = []

    for _, row in wastewater_route_mass_inventory.iterrows():
        common = {
            "source_module": "wastewater",
            "parent_stream_category": "plant_wastewater",
            "parent_route": row["route"],
            "parent_eol_sink": row["route_kind"],
            "parent_polymer": None,
            "inventory_id": row["inventory_id"],
            "broad_process_area": row["broad_process_area"],
            "superpro_section": row["superpro_section"],
            "unit_operation": row["unit_operation"],
            "source_procedure": row["source_procedure"],
            "flow_name_superpro": row["flow_name_superpro"],
            "water_quality_class": row["water_quality_class"],
            "parent_unit": "m3 wastewater routed",
            "embedded_in_lca_activity": False,
            "source_of_quantity": "foreground_route_mass_balance",
            "utility_water_credit_in_route_factor": bool(row.get("utility_water_credit_in_route_factor", False)),
        }

        # Treated effluent discharged after treatment.
        secondary_rows.append({
            **common,
            "secondary_stream_category": "treated_wastewater_discharge",
            "secondary_stream_name": "treated_effluent_to_sewer_or_discharge",
            "stream_direction": "treated_output_to_discharge",
            "quantity_per_parent_unit": row["discharge_fraction"],
            "quantity_per_batch": row["discharged_volume_m3_per_batch"],
            "quantity_unit": "m3",
            "export_to_wastewater_pool": False,
            "routable_in_master": False,
            "credit_already_in_route_factor": False,
            "assumption_note": "Discharged treated effluent after wastewater treatment.",
        })

        # Reused water available as non-contact utility water.
        if row["reused_volume_m3_per_batch"] > 0:
            secondary_rows.append({
                **common,
                "secondary_stream_category": "reused_utility_water",
                "secondary_stream_name": "treated_water_available_for_non_contact_utility_use",
                "stream_direction": "secondary_product_output",
                "quantity_per_parent_unit": row["reuse_fraction"],
                "quantity_per_batch": row["reused_volume_m3_per_batch"],
                "quantity_unit": "m3",
                "export_to_wastewater_pool": False,
                "routable_in_master": True,
                "credit_already_in_route_factor": bool(row.get("utility_water_credit_in_route_factor", False)),
                "assumption_note": (
                    "Reusable treated water output. Later optimisation should limit this by utility-water demand, "
                    "reuse quality acceptance and treatment capacity. If credit_already_in_route_factor=True, "
                    "do not add a second avoided-water credit for this same volume."
                ),
            })

    wastewater_secondary_outputs = pd.DataFrame(secondary_rows)

    secondary_output_cols = [
        "source_module",
        "parent_stream_category",
        "parent_route",
        "parent_eol_sink",
        "parent_polymer",
        "secondary_stream_category",
        "secondary_stream_name",
        "stream_direction",
        "inventory_id",
        "broad_process_area",
        "superpro_section",
        "unit_operation",
        "source_procedure",
        "flow_name_superpro",
        "water_quality_class",
        "quantity_per_parent_unit",
        "parent_unit",
        "quantity_per_batch",
        "quantity_unit",
        "embedded_in_lca_activity",
        "export_to_wastewater_pool",
        "routable_in_master",
        "source_of_quantity",
        "utility_water_credit_in_route_factor",
        "credit_already_in_route_factor",
        "assumption_note",
    ]
    secondary_output_cols = [c for c in secondary_output_cols if c in wastewater_secondary_outputs.columns]
    wastewater_secondary_outputs = wastewater_secondary_outputs[secondary_output_cols].copy()

    print(f"Wastewater secondary output rows: {len(wastewater_secondary_outputs):,}")
    display(wastewater_secondary_outputs.head(20))
    display(
        wastewater_secondary_outputs
        .groupby(["parent_route", "secondary_stream_category", "quantity_unit"], dropna=False, as_index=False)
        .agg(quantity_per_batch=("quantity_per_batch", "sum"))
    )


    # ---------------------------------------------------------------------
    # Summaries
    # ---------------------------------------------------------------------

    group_cols = ["route", "life_cycle_stage", "impact_category"]
    summary_by_route_stage_category = (
        wastewater_lca_detailed
        .groupby(group_cols, dropna=False, as_index=False)
        .agg(
            routed_total_kg_per_batch=("routed_total_kg_per_batch", "sum"),
            routed_water_kg_per_batch=("routed_water_kg_per_batch", "sum"),
            routed_volume_m3_per_batch=("routed_volume_m3_per_batch", "sum"),
            impact_per_batch=("impact_per_batch", "sum"),
        )
    )

    summary_by_section = (
        wastewater_lca_detailed
        .groupby(["route", "impact_category", "broad_process_area", "superpro_section"], dropna=False, as_index=False)
        .agg(
            routed_total_kg_per_batch=("routed_total_kg_per_batch", "sum"),
            routed_water_kg_per_batch=("routed_water_kg_per_batch", "sum"),
            routed_volume_m3_per_batch=("routed_volume_m3_per_batch", "sum"),
            impact_per_batch=("impact_per_batch", "sum"),
        )
    )

    summary_by_quality = (
        wastewater_lca_detailed
        .groupby(["route", "impact_category", "water_quality_class"], dropna=False, as_index=False)
        .agg(
            routed_total_kg_per_batch=("routed_total_kg_per_batch", "sum"),
            routed_water_kg_per_batch=("routed_water_kg_per_batch", "sum"),
            routed_volume_m3_per_batch=("routed_volume_m3_per_batch", "sum"),
            impact_per_batch=("impact_per_batch", "sum"),
        )
    )

    summary_by_stream = (
        wastewater_lca_detailed
        .groupby(["route", "impact_category", "flow_name_superpro", "superpro_section", "unit_operation", "water_quality_class"], dropna=False, as_index=False)
        .agg(
            routed_total_kg_per_batch=("routed_total_kg_per_batch", "sum"),
            routed_water_kg_per_batch=("routed_water_kg_per_batch", "sum"),
            routed_volume_m3_per_batch=("routed_volume_m3_per_batch", "sum"),
            impact_per_batch=("impact_per_batch", "sum"),
        )
    )

    display(summary_by_route_stage_category)


    wastewater_main_detailed_table = wastewater_lca_detailed.copy()
    wastewater_main_detailed_table["source_module"] = "wastewater"
    wastewater_main_detailed_table["stream_category"] = "plant_wastewater"
    wastewater_main_detailed_table["functional_unit"] = "per_batch"
    wastewater_main_detailed_table["eol_pathway"] = wastewater_main_detailed_table["route"]
    wastewater_main_detailed_table["eol_sink"] = wastewater_main_detailed_table["route_kind"]
    wastewater_main_detailed_table["impact_per_unit"] = wastewater_main_detailed_table["impact_per_activity_unit"]
    wastewater_main_detailed_table["impact_per_unit_raw"] = wastewater_main_detailed_table.get(
        "impact_per_activity_unit_raw",
        wastewater_main_detailed_table["impact_per_activity_unit"],
    )
    wastewater_main_detailed_table["activity_amount_unit"] = ACTIVITY_AMOUNT_BASIS
    wastewater_main_detailed_table["quantity_unit"] = "m3 wastewater routed"
    wastewater_main_detailed_table["impact_factor_unit"] = wastewater_main_detailed_table["impact_category"] + " per m3 routed"
    wastewater_main_detailed_table["is_production"] = False
    wastewater_main_detailed_table["is_eol_treatment"] = wastewater_main_detailed_table["life_cycle_stage"].eq("wastewater_treatment")
    wastewater_main_detailed_table["secondary_streams_embedded_in_activity"] = False

    preferred_first_cols = [
        "source_module",
        "stream_category",
        "functional_unit",
        "inventory_id",
        "broad_process_area",
        "superpro_section",
        "unit_operation",
        "source_procedure",
        "flow_name_superpro",
        "water_quality_class",
        "total_kg_per_batch",
        "water_kg_per_batch",
        "stream_volume_m3_per_batch",
        "eol_pathway",
        "eol_sink",
        "route",
        "route_kind",
        "life_cycle_stage",
        "impact_category",
        "impact_per_unit",
        "impact_per_unit_raw",
        "impact_factor_unit",
        "impact_per_batch",
        "activity_amount_unit",
        "routed_activity_amount_per_batch",
        "reuse_fraction",
        "discharge_fraction",
        "is_eol_treatment",
        "is_credit",
        "credit_applied_to_secondary_output",
        "secondary_streams_embedded_in_activity",
        "sign_convention_adjusted_for_optimisation",
        "sign_convention_note",
    ]
    preferred_first_cols = [c for c in preferred_first_cols if c in wastewater_main_detailed_table.columns]
    remaining_cols = [c for c in wastewater_main_detailed_table.columns if c not in preferred_first_cols]
    wastewater_main_detailed_table = wastewater_main_detailed_table[preferred_first_cols + remaining_cols].copy()

    # Route factor table: independent of individual streams.
    route_group_cols = [
        "source_module",
        "stream_category",
        "route",
        "eol_pathway",
        "eol_sink",
        "route_kind",
        "life_cycle_stage",
        "impact_category",
        "impact_factor_unit",
        "quantity_unit",
        "is_eol_treatment",
        "is_credit",
        "credit_applied_to_secondary_output",
        "activity_name",
        "reference_product",
        "location",
        "unit",
        "database",
        "code",
        "mapping_type",
        "activity_search",
        "assumption_note",
        "water_scarcity_regionalisation_status",
        "water_scarcity_native_world_cf",
        "water_scarcity_background_factor_per_m3",
        "water_scarcity_foreground_consumption_m3_per_m3",
        "water_scarcity_provider_activity_ids",
        "water_scarcity_factor__Oxford",
        "water_scarcity_factor__Cambridge",
        "water_scarcity_factor__Slough",
    ]
    route_group_cols = [c for c in route_group_cols if c in wastewater_main_detailed_table.columns]

    wastewater_route_factor_table = (
        wastewater_main_detailed_table
        .groupby(route_group_cols, dropna=False, as_index=False)
        .agg(
            impact_per_m3_for_optimisation=("impact_per_unit", "mean"),
            raw_brightway_impact_per_m3=("impact_per_unit_raw", "mean"),
            min_impact_per_m3=("impact_per_unit", "min"),
            max_impact_per_m3=("impact_per_unit", "max"),
            represented_volume_m3_per_batch=("routed_activity_amount_per_batch", "sum"),
            represented_impact_per_batch=("impact_per_batch", "sum"),
            contributing_stream_rows=("inventory_id", "nunique"),
        )
    )

    wastewater_route_factor_table["impact_factor_variability_abs"] = (
        wastewater_route_factor_table["max_impact_per_m3"]
        - wastewater_route_factor_table["min_impact_per_m3"]
    )
    wastewater_route_factor_table["route_factor_status"] = np.where(
        wastewater_route_factor_table["impact_factor_variability_abs"].abs() <= 1e-12,
        "single_consistent_factor",
        "mixed_factor_check",
    )

    route_first_cols = [
        "source_module",
        "stream_category",
        "route",
        "eol_pathway",
        "eol_sink",
        "life_cycle_stage",
        "impact_category",
        "impact_per_m3_for_optimisation",
        "raw_brightway_impact_per_m3",
        "impact_factor_unit",
        "is_eol_treatment",
        "is_credit",
        "credit_applied_to_secondary_output",
        "represented_volume_m3_per_batch",
        "represented_impact_per_batch",
        "route_factor_status",
    ]
    route_first_cols = [c for c in route_first_cols if c in wastewater_route_factor_table.columns]
    route_remaining_cols = [c for c in wastewater_route_factor_table.columns if c not in route_first_cols]
    wastewater_route_factor_table = wastewater_route_factor_table[route_first_cols + route_remaining_cols].copy()

    if "wastewater_secondary_outputs" not in locals():
        wastewater_secondary_outputs = pd.DataFrame()

    files = {
        "main_detailed_table": wastewater_main_detailed_table,
        "route_factor_table": wastewater_route_factor_table,
        "secondary_outputs": wastewater_secondary_outputs,
    }
    return {
        "main_detailed_table": wastewater_main_detailed_table,
        "route_factor_table": wastewater_route_factor_table,
        "secondary_outputs": wastewater_secondary_outputs,
        "inventory": wastewater_inventory,
        "route_summary": summary_by_route_stage_category,
        "route_mass_summary": wastewater_route_mass_inventory,
        "treatment_mapping": wastewater_mapping,
    }
