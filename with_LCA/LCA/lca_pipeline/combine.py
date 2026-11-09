from __future__ import annotations


def combine_lca_results(plastics, wastewater, manufacturing, biomass, settings):
    """Harmonise the LCA modules into the optimiser-facing route-factor matrix and input/output table."""
    import numpy as np
    import pandas as pd

    display = lambda *args, **kwargs: None
    route_tables = {
        "wastewater": wastewater["route_factor_table"],
        "sut_plastics": plastics["route_factor_table"],
        "manufacturing": manufacturing["route_factor_table"],
        "biomass": biomass["route_factor_table"],
    }
    main_tables = {
        "wastewater": wastewater["main_detailed_table"],
        "sut_plastics": plastics["main_detailed_table"],
        "manufacturing": manufacturing["main_detailed_table"],
        "biomass": biomass["main_detailed_table"],
    }
    secondary_tables = {
        "wastewater": wastewater["secondary_outputs"],
        "sut_plastics": plastics["secondary_outputs"],
        "manufacturing": manufacturing["secondary_outputs"],
        "biomass": biomass["secondary_outputs"],
    }
    optional_tables = {
        "wastewater": {},
        "sut_plastics": {
            "route_content_summary": plastics.get("route_content_summary", pd.DataFrame()),
            "polymer_route_mapping": plastics.get("polymer_route_mapping", pd.DataFrame()),
        },
        "manufacturing": {
            "physical_inventory": manufacturing.get("physical_inventory", pd.DataFrame()),
        },
        "biomass": {
            "inventory": biomass.get("inventory", pd.DataFrame()),
            "pathway_quantities": biomass.get("pathway_quantities", pd.DataFrame()),
            "activity_mapping": biomass.get("activity_mapping", pd.DataFrame()),
        },
    }
    UTILITY_WATER_REUSE_CONFIG = {
        "reuse_pathway": "treatment_for_utility_reuse",
        "credit_stage": "avoided_utility_water_supply",
        "recovery_fraction": float(settings["wastewater"].get("utility_water_reuse_efficiency", 1.0)),
        "utility_water_demand_m3_per_batch": settings["wastewater"].get("utility_water_reuse_demand_limit_m3_per_batch"),
        "assume_demand_equals_recoverable_when_missing": True,
    }
    # ---------------------------------------------------------------------
    # Imports
    # ---------------------------------------------------------------------
    import numpy as np
    import pandas as pd

    pd.set_option("display.max_columns", 250)
    pd.set_option("display.width", 200)




    # ---------------------------------------------------------------------
    # Helper functions
    # ---------------------------------------------------------------------
    def get_series(df, column, default=np.nan):
        """Return a DataFrame column if present, otherwise a same-length default Series."""
        if column in df.columns:
            return df[column]
        return pd.Series([default] * len(df), index=df.index)


    def first_existing_series(df, columns, default=np.nan, required=False, label=None):
        """Return the first available column from a list of candidate names."""
        for column in columns:
            if column in df.columns:
                return df[column]
        if required:
            raise ValueError(
                f"Missing required column for {label or 'value'}. Tried: {columns}. "
                f"Available columns: {list(df.columns)}"
            )
        return pd.Series([default] * len(df), index=df.index)


    def first_existing_column_name(df, columns, required=False, label=None):
        """Return the first available column name from candidates."""
        for column in columns:
            if column in df.columns:
                return column
        if required:
            raise ValueError(
                f"Missing required column for {label or 'value'}. Tried: {columns}. "
                f"Available columns: {list(df.columns)}"
            )
        return None


    def join_unique(values):
        """Readable pipe-separated unique values, used for aggregated metadata."""
        cleaned = []
        for value in values:
            if pd.isna(value):
                continue
            text = str(value).strip()
            if text and text.lower() != "nan" and text not in cleaned:
                cleaned.append(text)
        return " | ".join(cleaned)


    def parse_bool(value, default=False):
        """Parse bools safely from bool/string/numeric values."""
        if pd.isna(value):
            return default
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n"}:
            return False
        return default


    def as_bool_series(series, default=False):
        """Convert mixed bool/object columns to bool safely."""
        if not isinstance(series, pd.Series):
            return pd.Series([default])
        return series.map(lambda value: parse_bool(value, default=default))


    # ---------------------------------------------------------------------
    # Harmonise route-factor tables into one long table
    # ---------------------------------------------------------------------
    def harmonise_route_factors(route_tables):
        harmonised = []

        # Wastewater: factors are per m3 routed wastewater.
        df = route_tables["wastewater"].copy()
        wastewater = pd.DataFrame({
            "source_module": get_series(df, "source_module", "wastewater_lca"),
            "stream_category": get_series(df, "stream_category", "plant_wastewater"),
            "optimisation_flow": "plant_wastewater",
            "material_or_service": "plant_wastewater",
            "pathway": first_existing_series(df, ["route", "pathway"], required=True, label="wastewater route/pathway"),
            "eol_pathway": get_series(df, "eol_pathway", np.nan),
            "eol_sink": get_series(df, "eol_sink", np.nan),
            "life_cycle_stage": get_series(df, "life_cycle_stage", np.nan),
            "impact_category": first_existing_series(df, ["impact_category"], required=True, label="wastewater impact_category"),
            "impact_factor": first_existing_series(
                df,
                ["impact_per_m3_for_optimisation", "impact_per_activity_unit", "impact_factor"],
                required=True,
                label="wastewater impact factor",
            ),
            "optimisation_quantity_unit": "m3",
            "impact_factor_unit": get_series(df, "impact_factor_unit", np.nan),
            "represented_quantity_per_batch": first_existing_series(
                df,
                ["represented_volume_m3_per_batch", "activity_amount_per_batch", "routed_volume_m3_per_batch"],
                default=np.nan,
            ),
            "represented_impact_per_batch": first_existing_series(
                df,
                ["represented_impact_per_batch", "impact_per_batch"],
                default=np.nan,
            ),
            "route_kind": get_series(df, "route_kind", np.nan),
            "is_production": False,
            "is_eol_treatment": get_series(df, "is_eol_treatment", True),
            "is_credit": get_series(df, "is_credit", False),
            "recycling_recovery_fraction": get_series(
                df, "recycling_recovery_fraction", 0.0
            ),
            "virgin_material_substitution_ratio": get_series(
                df, "virgin_material_substitution_ratio", 0.0
            ),
            "activity_name": get_series(df, "activity_name", np.nan),
            "reference_product": get_series(df, "reference_product", np.nan),
            "location": get_series(df, "location", np.nan),
            "unit": get_series(df, "unit", np.nan),
            "database": get_series(df, "database", np.nan),
            "code": get_series(df, "code", np.nan),
            "mapping_type": get_series(df, "mapping_type", np.nan),
            "assumption_note": get_series(df, "assumption_note", np.nan),
            "water_scarcity_regionalisation_status": get_series(
                df, "water_scarcity_regionalisation_status", "not_applicable"
            ),
            "water_scarcity_native_world_cf": get_series(
                df, "water_scarcity_native_world_cf", np.nan
            ),
            "water_scarcity_background_factor_per_m3": get_series(
                df, "water_scarcity_background_factor_per_m3", np.nan
            ),
            "water_scarcity_foreground_consumption_m3_per_m3": get_series(
                df, "water_scarcity_foreground_consumption_m3_per_m3", np.nan
            ),
            "water_scarcity_provider_activity_ids": get_series(
                df, "water_scarcity_provider_activity_ids", np.nan
            ),
            "water_scarcity_factor__Oxford": get_series(
                df, "water_scarcity_factor__Oxford", np.nan
            ),
            "water_scarcity_factor__Cambridge": get_series(
                df, "water_scarcity_factor__Cambridge", np.nan
            ),
            "water_scarcity_factor__Slough": get_series(
                df, "water_scarcity_factor__Slough", np.nan
            ),
        })
        harmonised.append(wastewater)

        # SUT plastics: factors are per kg polymer routed through a selected EOL pathway.
        df = route_tables["sut_plastics"].copy()
        polymer_col = first_existing_column_name(df, ["polymer", "Polymer"], required=True, label="SUT polymer")
        plastics_note = get_series(df, "demand_orientation_note", "").fillna("").astype(str)
        if "mapping_assumption_level" in df.columns:
            plastics_note = (plastics_note + " " + df["mapping_assumption_level"].fillna("").astype(str)).str.strip()
        plastics = pd.DataFrame({
            "source_module": get_series(df, "source_module", "sut_plastics"),
            "stream_category": get_series(df, "stream_category", "sut_plastic_waste"),
            "optimisation_flow": "sut_plastic__" + df[polymer_col].astype(str),
            "material_or_service": df[polymer_col].astype(str),
            "pathway": first_existing_series(df, ["eol_pathway", "pathway"], required=True, label="SUT eol_pathway"),
            "eol_pathway": first_existing_series(df, ["eol_pathway", "pathway"], default=np.nan),
            "eol_sink": get_series(df, "eol_sink", np.nan),
            "life_cycle_stage": get_series(df, "life_cycle_stage", np.nan),
            "impact_category": first_existing_series(df, ["impact_category"], required=True, label="SUT impact_category"),
            "impact_factor": first_existing_series(
                df,
                ["impact_per_kg_for_optimisation", "impact_per_kg_mean", "impact_per_kg", "impact_factor"],
                required=True,
                label="SUT impact factor",
            ),
            "optimisation_quantity_unit": "kg",
            "impact_factor_unit": get_series(df, "impact_factor_unit", np.nan),
            "represented_quantity_per_batch": first_existing_series(
                df,
                ["represented_mass_kg_per_batch", "polymer_kg_per_batch", "quantity_per_batch"],
                default=np.nan,
            ),
            "represented_impact_per_batch": first_existing_series(
                df,
                ["represented_impact_per_batch", "impact_per_batch"],
                default=np.nan,
            ),
            "route_kind": get_series(df, "route_factor_status", np.nan),
            "is_production": get_series(df, "is_production", False),
            "is_eol_treatment": get_series(df, "is_eol_treatment", False),
            "is_credit": get_series(df, "is_credit", False),
            "recycling_recovery_fraction": get_series(df, "recycling_recovery_fraction", 0.0),
            "virgin_material_substitution_ratio": get_series(df, "virgin_material_substitution_ratio", 0.0),
            "activity_name": get_series(df, "activity_name", np.nan),
            "reference_product": get_series(df, "reference_product", np.nan),
            "location": get_series(df, "location", np.nan),
            "unit": get_series(df, "unit", np.nan),
            "database": get_series(df, "database", np.nan),
            "code": get_series(df, "code", np.nan),
            "mapping_type": get_series(df, "mapping_type", np.nan),
            "assumption_note": plastics_note,
        })
        harmonised.append(plastics)

        # Manufacturing: factors are per native activity unit: kg, kWh or MJ.
        df = route_tables["manufacturing"].copy()
        flow_col = first_existing_column_name(
            df,
            ["manufacturing_flow", "lca_flow_key", "material_or_service", "route"],
            required=True,
            label="manufacturing flow identifier",
        )

        # Manufacturing route identity:
        # - electricity keeps its explicit supply_pathway route choice;
        # - fixed utilities remain baseline_supply.
        if "supply_pathway" in df.columns:
            pathway_series = df["supply_pathway"].fillna(get_series(df, "route", df[flow_col]))
        elif "route" in df.columns:
            pathway_series = df["route"].fillna(df[flow_col])
        elif "route_factor_scenario" in df.columns:
            pathway_series = df["route_factor_scenario"].fillna(df[flow_col])
        else:
            pathway_series = df[flow_col]

        manufacturing = pd.DataFrame({
            "source_module": get_series(df, "source_module", "manufacturing_footprint"),
            "stream_category": get_series(df, "stream_category", np.nan),
            "optimisation_flow": "manufacturing__" + df[flow_col].astype(str),
            "material_or_service": df[flow_col].astype(str),
            "pathway": pathway_series,
            "eol_pathway": np.nan,
            "eol_sink": np.nan,
            "life_cycle_stage": get_series(df, "life_cycle_stage", np.nan),
            "impact_category": first_existing_series(df, ["impact_category"], required=True, label="manufacturing impact_category"),
            "impact_factor": first_existing_series(
                df,
                ["impact_per_unit_for_optimisation", "impact_per_activity_unit", "impact_factor"],
                required=True,
                label="manufacturing impact factor",
            ),
            "optimisation_quantity_unit": first_existing_series(
                df,
                ["quantity_unit", "activity_unit", "unit"],
                default=np.nan,
            ),
            "impact_factor_unit": get_series(df, "impact_factor_unit", np.nan),
            "represented_quantity_per_batch": first_existing_series(
                df,
                ["represented_quantity_per_batch", "activity_demand_amount_per_batch", "source_quantity"],
                default=np.nan,
            ),
            "represented_impact_per_batch": first_existing_series(
                df,
                ["represented_impact_per_batch", "impact_per_batch"],
                default=np.nan,
            ),
            "route_kind": get_series(df, "route_kind", np.nan),
            "is_production": get_series(df, "is_production", True),
            "is_eol_treatment": get_series(df, "is_eol_treatment", False),
            "is_credit": get_series(df, "is_credit", False),
            "activity_name": get_series(df, "activity_name", np.nan),
            "reference_product": get_series(df, "reference_product", np.nan),
            "location": get_series(df, "location", np.nan),
            "unit": get_series(df, "unit", np.nan),
            "database": get_series(df, "database", np.nan),
            "code": get_series(df, "code", np.nan),
            "mapping_type": get_series(df, "mapping_type", np.nan),
            "assumption_note": get_series(df, "assumption_note", np.nan),
        })
        harmonised.append(manufacturing)

        # Biomass: net factors are per kg residual biomass routed to AD.
        df = route_tables["biomass"].copy()
        biomass_route = pd.DataFrame({
            "source_module": get_series(df, "source_module", "biomass"),
            "stream_category": get_series(df, "stream_category", "residual_biomass"),
            "optimisation_flow": "residual_biomass",
            "material_or_service": get_series(df, "material_or_service", "residual_biomass"),
            "pathway": first_existing_series(df, ["pathway", "route"], required=True, label="biomass pathway"),
            "eol_pathway": first_existing_series(df, ["eol_pathway", "pathway", "route"], default=np.nan),
            "eol_sink": get_series(df, "eol_sink", "anaerobic_digestion"),
            "life_cycle_stage": get_series(df, "life_cycle_stage", np.nan),
            "impact_category": first_existing_series(df, ["impact_category"], required=True, label="biomass impact_category"),
            "impact_factor": first_existing_series(
                df,
                ["impact_per_kg_for_optimisation", "impact_per_kg_biomass", "impact_factor"],
                required=True,
                label="biomass impact factor",
            ),
            "optimisation_quantity_unit": "kg",
            "impact_factor_unit": get_series(df, "impact_factor_unit", np.nan),
            "represented_quantity_per_batch": first_existing_series(
                df,
                ["represented_mass_kg_per_batch", "biomass_kg_per_batch", "quantity_per_batch"],
                default=np.nan,
            ),
            "represented_impact_per_batch": first_existing_series(
                df,
                ["represented_impact_per_batch", "impact_per_batch"],
                default=np.nan,
            ),
            "route_kind": get_series(df, "route_factor_status", "single_consistent_factor"),
            "is_production": get_series(df, "is_production", False),
            "is_eol_treatment": get_series(df, "is_eol_treatment", False),
            "is_credit": get_series(df, "is_credit", False),
            "activity_name": get_series(df, "activity_name", np.nan),
            "reference_product": get_series(df, "reference_product", np.nan),
            "location": get_series(df, "location", np.nan),
            "unit": get_series(df, "unit", np.nan),
            "database": get_series(df, "database", np.nan),
            "code": get_series(df, "code", np.nan),
            "mapping_type": get_series(df, "mapping_type", np.nan),
            "assumption_note": get_series(df, "assumption_note", np.nan),
        })
        harmonised.append(biomass_route)

        route_factor_long = pd.concat(harmonised, ignore_index=True)
        route_factor_long["impact_factor"] = pd.to_numeric(route_factor_long["impact_factor"], errors="coerce")
        route_factor_long["represented_quantity_per_batch"] = pd.to_numeric(route_factor_long["represented_quantity_per_batch"], errors="coerce")
        route_factor_long["represented_impact_per_batch"] = pd.to_numeric(route_factor_long["represented_impact_per_batch"], errors="coerce")
        route_factor_long["recycling_recovery_fraction"] = pd.to_numeric(
            get_series(route_factor_long, "recycling_recovery_fraction", 0.0),
            errors="coerce",
        ).fillna(0.0)
        route_factor_long["virgin_material_substitution_ratio"] = pd.to_numeric(
            get_series(route_factor_long, "virgin_material_substitution_ratio", 0.0),
            errors="coerce",
        ).fillna(0.0)
        route_factor_long["is_credit"] = as_bool_series(route_factor_long["is_credit"], default=False)
        route_factor_long["is_eol_treatment"] = as_bool_series(route_factor_long["is_eol_treatment"], default=False)
        route_factor_long["is_production"] = as_bool_series(route_factor_long["is_production"], default=False)

        route_factor_long["stage_type"] = np.select(
            [
                route_factor_long["is_credit"],
                route_factor_long["is_eol_treatment"],
                route_factor_long["is_production"],
            ],
            ["credit", "treatment", "production"],
            default="supply",
        )

        route_factor_long["factor_interpretation"] = (
            "If 1 "
            + route_factor_long["optimisation_quantity_unit"].astype(str)
            + " of "
            + route_factor_long["material_or_service"].astype(str)
            + " is assigned to pathway "
            + route_factor_long["pathway"].astype(str)
            + ", the LCIA burden is the route factor for that impact category."
        )

        return route_factor_long


    route_factor_long = harmonise_route_factors(route_tables)

    print(route_factor_long.shape)
    display(route_factor_long.head(10))

    # ---------------------------------------------------------------------
    # Add explicit utility-water supply and cap internal wastewater reuse
    # ---------------------------------------------------------------------
    def add_explicit_utility_water_accounting(route_factor_long, config):
        """Add the missing baseline utility-water supply and apply reuse consistently.

        The wastewater module already contains an avoided utility-water factor. The
        master model therefore creates the corresponding positive foreground supply
        row for all scenarios and scales the negative credit to the volume that is
        actually reusable. WFI rows are never modified.
        """
        result = route_factor_long.copy()

        source_text = result["source_module"].astype(str).str.lower()
        pathway_text = result["pathway"].astype(str)
        stage_text = result["life_cycle_stage"].astype(str).str.lower()

        wastewater_mask = source_text.str.contains("wastewater", na=False)
        reuse_pathway_mask = pathway_text.eq(config["reuse_pathway"])
        credit_mask = (
            wastewater_mask
            & reuse_pathway_mask
            & (
                result["is_credit"].fillna(False).astype(bool)
                | stage_text.eq(str(config["credit_stage"]).lower())
                | stage_text.str.contains("avoided_utility_water", na=False)
            )
        )
        treatment_mask = (
            wastewater_mask
            & reuse_pathway_mask
            & ~credit_mask
            & stage_text.str.contains("wastewater_treatment", na=False)
        )

        exported_credit_rows = result.loc[credit_mask].copy()
        if exported_credit_rows.empty:
            raise ValueError(
                "The utility-reuse wastewater pathway has no avoided utility-water "
                "credit rows. Rerun the wastewater LCA export first."
            )

        route_volume_candidates = pd.to_numeric(
            result.loc[treatment_mask, "represented_quantity_per_batch"],
            errors="coerce",
        ).dropna()
        if route_volume_candidates.empty:
            route_volume_candidates = pd.to_numeric(
                result.loc[wastewater_mask & reuse_pathway_mask, "represented_quantity_per_batch"],
                errors="coerce",
            ).dropna()
        if route_volume_candidates.empty:
            raise ValueError("Could not determine the wastewater volume available for utility reuse.")

        routed_wastewater_m3 = float(route_volume_candidates.max())
        recovery_fraction = float(config.get("recovery_fraction", 1.0))
        if not 0 <= recovery_fraction <= 1:
            raise ValueError("UTILITY_WATER_REUSE_CONFIG['recovery_fraction'] must be between 0 and 1.")

        recoverable_water_m3 = routed_wastewater_m3 * recovery_fraction
        configured_demand = config.get("utility_water_demand_m3_per_batch")

        if configured_demand is None:
            if not config.get("assume_demand_equals_recoverable_when_missing", False):
                raise ValueError(
                    "Set UTILITY_WATER_REUSE_CONFIG['utility_water_demand_m3_per_batch'] "
                    "to the non-contact utility-water demand."
                )
            utility_water_demand_m3 = recoverable_water_m3
            demand_basis = (
                "Temporary assumption: non-contact utility-water demand is at least "
                "equal to recoverable wastewater volume. Replace with the quantified "
                "SuperPro/plant utility-water demand."
            )
        else:
            utility_water_demand_m3 = float(configured_demand)
            if utility_water_demand_m3 < 0:
                raise ValueError("Utility-water demand cannot be negative.")
            demand_basis = "User-configured non-contact utility-water demand."

        reused_water_m3 = min(recoverable_water_m3, utility_water_demand_m3)
        excess_recovered_water_m3 = max(recoverable_water_m3 - reused_water_m3, 0.0)
        fresh_utility_after_reuse_m3 = max(utility_water_demand_m3 - reused_water_m3, 0.0)
        applied_credit_fraction = (
            reused_water_m3 / routed_wastewater_m3
            if routed_wastewater_m3 > 0
            else 0.0
        )

        # Scale the exported avoided-supply factor to the fraction of each routed m3
        # that is actually recovered and used. Keep the route quantity on the full
        # wastewater basis so the net route factor remains valid per m3 routed.
        original_credit_factors = pd.to_numeric(
            result.loc[credit_mask, "impact_factor"], errors="coerce"
        )
        result.loc[credit_mask, "impact_factor"] = (
            original_credit_factors * applied_credit_fraction
        )
        regionalised_factor_cols = [
            "water_scarcity_background_factor_per_m3",
            "water_scarcity_foreground_consumption_m3_per_m3",
            "water_scarcity_factor__Oxford",
            "water_scarcity_factor__Cambridge",
            "water_scarcity_factor__Slough",
        ]
        for column in regionalised_factor_cols:
            if column in result.columns:
                result.loc[credit_mask, column] = (
                    pd.to_numeric(result.loc[credit_mask, column], errors="coerce")
                    * applied_credit_fraction
                )
        result.loc[credit_mask, "represented_quantity_per_batch"] = routed_wastewater_m3
        result.loc[credit_mask, "represented_impact_per_batch"] = (
            pd.to_numeric(result.loc[credit_mask, "impact_factor"], errors="coerce")
            * routed_wastewater_m3
        )
        result.loc[credit_mask, "assumption_note"] = (
            result.loc[credit_mask, "assumption_note"].fillna("").astype(str)
            + " Applied only to the internally reused fraction; WFI is not displaced."
        ).str.strip()

        # Create the matching positive foreground utility-water supply. Factors are
        # the sign-reversed exported avoided-supply factors, before recovery scaling.
        utility_supply_rows = exported_credit_rows.copy()
        utility_supply_rows["source_module"] = "utility_water_supply"
        utility_supply_rows["stream_category"] = "non_contact_utility_water"
        utility_supply_rows["optimisation_flow"] = "fresh_utility_water"
        utility_supply_rows["material_or_service"] = "utility_water"
        utility_supply_rows["pathway"] = "fresh_utility_water_supply"
        utility_supply_rows["eol_pathway"] = np.nan
        utility_supply_rows["eol_sink"] = np.nan
        utility_supply_rows["life_cycle_stage"] = "utility_water_supply"
        utility_supply_rows["impact_factor"] = -pd.to_numeric(
            exported_credit_rows["impact_factor"], errors="coerce"
        )
        for column in regionalised_factor_cols:
            if column in utility_supply_rows.columns:
                utility_supply_rows[column] = -pd.to_numeric(
                    exported_credit_rows[column], errors="coerce"
                )
        utility_supply_rows["represented_quantity_per_batch"] = utility_water_demand_m3
        utility_supply_rows["represented_impact_per_batch"] = (
            utility_supply_rows["impact_factor"] * utility_water_demand_m3
        )
        utility_supply_rows["route_kind"] = "foreground_supply"
        utility_supply_rows["is_production"] = True
        utility_supply_rows["is_eol_treatment"] = False
        utility_supply_rows["is_credit"] = False
        utility_supply_rows["stage_type"] = "production"
        utility_supply_rows["mapping_type"] = "derived_from_exported_avoided_utility_water_factor"
        utility_supply_rows["assumption_note"] = (
            "Explicit baseline non-contact utility-water supply. " + demand_basis
        )
        utility_supply_rows["factor_interpretation"] = (
            "Positive burden per m3 of fresh non-contact utility water supplied."
        )

        result = pd.concat([result, utility_supply_rows], ignore_index=True)

        accounting = pd.DataFrame([
            {
                "wastewater_routed_m3_per_batch": routed_wastewater_m3,
                "reuse_recovery_fraction": recovery_fraction,
                "recoverable_water_m3_per_batch": recoverable_water_m3,
                "utility_water_demand_m3_per_batch": utility_water_demand_m3,
                "reused_water_m3_per_batch": reused_water_m3,
                "fresh_utility_water_after_reuse_m3_per_batch": fresh_utility_after_reuse_m3,
                "excess_recovered_water_m3_per_batch": excess_recovered_water_m3,
                "credit_fraction_per_m3_routed": applied_credit_fraction,
                "wfi_displaced_m3_per_batch": 0.0,
                "utility_water_demand_basis": demand_basis,
            }
        ])

        return result, accounting


    route_factor_long, utility_water_accounting_table = add_explicit_utility_water_accounting(
        route_factor_long,
        UTILITY_WATER_REUSE_CONFIG,
    )

    print("Utility-water reuse accounting:")
    display(utility_water_accounting_table)


    # ---------------------------------------------------------------------
    # Build the matrix-style MINLP route factor table
    # ---------------------------------------------------------------------
    index_cols = [
        "source_module",
        "stream_category",
        "optimisation_flow",
        "material_or_service",
        "pathway",
        "eol_pathway",
        "optimisation_quantity_unit",
    ]

    # Important: manufacturing rows have no real EOL pathway, so eol_pathway/eol_sink
    # can be NaN. Pandas pivot_table drops rows with NaN in index columns, which was
    # removing the manufacturing electricity routes from route_factor_matrix. Fill
    # model-index blanks before building the matrix.
    route_factor_long[index_cols] = route_factor_long[index_cols].fillna("not_applicable")
    route_factor_long["eol_sink"] = route_factor_long["eol_sink"].fillna("not_applicable")

    # Net route factor = sum of all life-cycle-stage factors for the route.
    # For plastics this gives production + EOL + any credits per kg polymer/pathway.
    # For wastewater this gives treatment + any reuse credit per m3.
    # For manufacturing this gives the supply burden per native input/utility unit.
    route_factor_net_long = (
        route_factor_long
        .groupby(index_cols + ["impact_category"], dropna=False, as_index=False)
        .agg(
            net_impact_factor=("impact_factor", "sum"),
            represented_quantity_per_batch=("represented_quantity_per_batch", "max"),
            represented_impact_per_batch=("represented_impact_per_batch", "sum"),
            life_cycle_stages=("life_cycle_stage", join_unique),
            eol_sinks=("eol_sink", join_unique),
            activity_names=("activity_name", join_unique),
            reference_products=("reference_product", join_unique),
            locations=("location", join_unique),
            databases=("database", join_unique),
            activity_codes=("code", join_unique),
            assumption_notes=("assumption_note", join_unique),
        )
    )

    # Method-specific stage provenance may differ (for example the climate-only
    # SUT proxy has no full LCI). Keep one row per physical route and aggregate
    # provenance across methods instead of using it as a pivot key.
    route_metadata = (
        route_factor_net_long.groupby(index_cols, dropna=False, as_index=False)
        .agg(
            represented_quantity_per_batch=("represented_quantity_per_batch", "max"),
            life_cycle_stages=("life_cycle_stages", join_unique),
            eol_sinks=("eol_sinks", join_unique),
            activity_names=("activity_names", join_unique),
            reference_products=("reference_products", join_unique),
            locations=("locations", join_unique),
            databases=("databases", join_unique),
            activity_codes=("activity_codes", join_unique),
            assumption_notes=("assumption_notes", join_unique),
        )
    )
    route_factor_matrix = (
        route_factor_net_long
        .pivot_table(index=index_cols, columns="impact_category",
                     values="net_impact_factor", aggfunc="first")
        .reset_index()
        .merge(route_metadata, on=index_cols, how="left", validate="one_to_one")
    )
    route_factor_matrix.columns.name = None

    # Keep physical recovery and quality-adjusted substitution separate at the
    # optimiser interface. Taking the maximum across route stages retrieves the
    # explicit mechanical-recycling parameter while production/disposal rows are 0.
    plastics_route_parameters = (
        route_factor_long
        .groupby(index_cols, dropna=False, as_index=False)
        .agg(
            recycling_recovery_fraction=("recycling_recovery_fraction", "max"),
            virgin_material_substitution_ratio=(
                "virgin_material_substitution_ratio",
                "max",
            ),
        )
    )
    route_factor_matrix = route_factor_matrix.merge(
        plastics_route_parameters,
        on=index_cols,
        how="left",
        validate="one_to_one",
    )

    for impact_category in route_factor_long["impact_category"].dropna().unique():
        if impact_category in route_factor_matrix.columns:
            route_factor_matrix = route_factor_matrix.rename(
                columns={impact_category: f"net_{impact_category}_factor_per_optimisation_unit"}
            )

    # The ecoinvent footprint implementation uses world-generic AWARE factors.
    # Replace only the explicitly traced foreground tap-water component; retain
    # every other inventory column at its native Brightway geography.
    regionalised_water_rows = route_factor_long[
        route_factor_long["impact_category"].astype(str).eq("water_scarcity")
        & route_factor_long["water_scarcity_regionalisation_status"].astype(str).eq(
            "world_generic_brightway_foreground_replaced_with_AWARE20_site_factor"
        )
    ].copy()
    if regionalised_water_rows.empty:
        raise ValueError(
            "No foreground utility-water rows carry the required AWARE 2.0 "
            "regionalisation decomposition."
        )
    water_components = (
        regionalised_water_rows
        .groupby(index_cols, dropna=False, as_index=False)
        .agg(
            water_scarcity_generic_component=("impact_factor", "sum"),
            water_scarcity_background_component=(
                "water_scarcity_background_factor_per_m3", "sum"
            ),
            water_scarcity_foreground_consumption_m3_per_optimisation_unit=(
                "water_scarcity_foreground_consumption_m3_per_m3", "sum"
            ),
            water_scarcity_native_world_cf=("water_scarcity_native_world_cf", "max"),
            water_scarcity_provider_activity_ids=(
                "water_scarcity_provider_activity_ids", join_unique
            ),
        )
    )
    route_factor_matrix = route_factor_matrix.merge(
        water_components,
        on=index_cols,
        how="left",
        validate="one_to_one",
    )
    for column in (
        "water_scarcity_generic_component",
        "water_scarcity_background_component",
        "water_scarcity_foreground_consumption_m3_per_optimisation_unit",
    ):
        route_factor_matrix[column] = pd.to_numeric(
            route_factor_matrix[column], errors="coerce"
        ).fillna(0.0)
    generic_net_water = pd.to_numeric(
        route_factor_matrix["net_water_scarcity_factor_per_optimisation_unit"],
        errors="raise",
    )
    for site, specification in settings["lcia"]["water_scarcity_regionalisation"]["sites"].items():
        site_cf = float(specification["factor_m3_world_eq_per_m3"])
        route_factor_matrix[f"net_water_scarcity_factor__{site}"] = (
            generic_net_water
            - route_factor_matrix["water_scarcity_generic_component"]
            + route_factor_matrix["water_scarcity_background_component"]
            + route_factor_matrix[
                "water_scarcity_foreground_consumption_m3_per_optimisation_unit"
            ] * site_cf
        )

    # Stage breakdown columns: useful for seeing production vs treatment vs credit.
    stage_breakdown_long = (
        route_factor_long
        .groupby(index_cols + ["impact_category", "stage_type"], dropna=False, as_index=False)
        .agg(stage_impact_factor=("impact_factor", "sum"))
    )
    stage_breakdown_wide = (
        stage_breakdown_long
        .pivot_table(
            index=index_cols,
            columns=["impact_category", "stage_type"],
            values="stage_impact_factor",
            aggfunc="first",
            fill_value=0.0,
        )
        .reset_index()
    )
    stage_breakdown_wide.columns = [
        "__".join(map(str, col)).strip("_") if isinstance(col, tuple) else col
        for col in stage_breakdown_wide.columns
    ]

    route_factor_matrix = route_factor_matrix.merge(
        stage_breakdown_wide,
        on=index_cols,
        how="left",
        validate="one_to_one",
    )

    # Keep the broad production/treatment/credit columns used by the optimiser,
    # and also expose each named life-cycle stage independently. This makes the
    # ecoinvent virgin-polymer burden and the additional pharmaceutical-grade SUT
    # component-manufacturing GWP directly distinguishable in reported results.
    named_stage_breakdown_long = (
        route_factor_long
        .dropna(subset=["life_cycle_stage"])
        .groupby(
            index_cols + ["impact_category", "life_cycle_stage"],
            dropna=False,
            as_index=False,
        )
        .agg(stage_impact_factor=("impact_factor", "sum"))
    )
    named_stage_breakdown_wide = (
        named_stage_breakdown_long
        .pivot_table(
            index=index_cols,
            columns=["impact_category", "life_cycle_stage"],
            values="stage_impact_factor",
            aggfunc="first",
            fill_value=0.0,
        )
        .reset_index()
    )
    named_stage_breakdown_wide.columns = [
        "__".join(map(str, col)).strip("_") if isinstance(col, tuple) else col
        for col in named_stage_breakdown_wide.columns
    ]
    route_factor_matrix = route_factor_matrix.merge(
        named_stage_breakdown_wide,
        on=index_cols,
        how="left",
        validate="one_to_one",
    )

    plastics_rows = route_factor_matrix["optimisation_flow"].astype(str).str.startswith(
        "sut_plastic__"
    )
    required_production_columns = {
        "gwp__virgin_polymer_production",
        "gwp__sut_component_manufacturing",
        "human_health_daly__virgin_polymer_production",
        "human_health_daly__sut_component_manufacturing",
        "ecosystem_quality__virgin_polymer_production",
        "ecosystem_quality__sut_component_manufacturing",
    }
    missing_production_columns = required_production_columns - set(route_factor_matrix.columns)
    if missing_production_columns:
        raise ValueError(
            "SUT production stages are not separately reportable in the route-factor matrix: "
            f"{sorted(missing_production_columns)}"
        )
    if plastics_rows.any():
        manufacturing_gwp = pd.to_numeric(
            route_factor_matrix.loc[plastics_rows, "gwp__sut_component_manufacturing"],
            errors="raise",
        )
        expected_manufacturing_gwp = float(
            settings["plastics"]["sut_manufacturing_gwp_adder_kgco2e_per_kg"]
        )
        if not np.allclose(manufacturing_gwp, expected_manufacturing_gwp, atol=1e-12):
            raise ValueError(
                "The separately reported SUT component-manufacturing GWP column does "
                "not reconcile to the configured per-kilogram adder."
            )
        manufacturing_daly = pd.to_numeric(
            route_factor_matrix.loc[
                plastics_rows, "human_health_daly__sut_component_manufacturing"
            ],
            errors="raise",
        )
        expected_manufacturing_daly = (
            expected_manufacturing_gwp
            * float(
                settings["human_health_daly"]
                ["climate_change_damage_factor_daly_per_kg_co2e"]
            )
        )
        if not np.allclose(manufacturing_daly, expected_manufacturing_daly, atol=1e-15):
            raise ValueError(
                "The manual SUT DALY stage must equal the 1.9 kg CO2e/kg proxy "
                "times the ReCiPe (H) climate-to-human-health factor exactly once."
            )
        from .ecosystem_quality import sut_manufacturing_climate_ecosystem_per_kg
        manufacturing_ecosystem = pd.to_numeric(
            route_factor_matrix.loc[plastics_rows, "ecosystem_quality__sut_component_manufacturing"],
            errors="raise",
        )
        expected_ecosystem = sut_manufacturing_climate_ecosystem_per_kg(settings)
        if not np.allclose(manufacturing_ecosystem, expected_ecosystem, atol=1e-18):
            raise ValueError("The SUT ecosystem proxy does not reconcile to the installed climate CF.")

    # The avoided-material credit is deliberately limited to displaced virgin
    # polymer production. It must never credit back the foreground 1.9 kg
    # CO2e/kg pharmaceutical-grade SUT component-manufacturing stage.
    credit_scope_columns = {
        "gwp__production",
        "gwp__virgin_polymer_production",
        "gwp__sut_component_manufacturing",
        "gwp__avoided_virgin_material_credit",
    }
    missing_credit_scope_columns = credit_scope_columns - set(route_factor_matrix.columns)
    if missing_credit_scope_columns:
        raise ValueError(
            "SUT avoided-credit boundary cannot be validated; missing "
            f"{sorted(missing_credit_scope_columns)}"
        )
    if plastics_rows.any():
        plastic_matrix = route_factor_matrix.loc[plastics_rows]
        if not np.allclose(
            pd.to_numeric(plastic_matrix["gwp__production"], errors="raise"),
            pd.to_numeric(
                plastic_matrix["gwp__virgin_polymer_production"], errors="raise"
            )
            + pd.to_numeric(
                plastic_matrix["gwp__sut_component_manufacturing"], errors="raise"
            ),
            atol=1e-10,
        ):
            raise ValueError(
                "Total SUT production GWP must equal virgin-polymer production plus "
                "the separate component-manufacturing stage."
            )
        recovery = pd.to_numeric(
            plastic_matrix["recycling_recovery_fraction"], errors="raise"
        )
        substitution = pd.to_numeric(
            plastic_matrix["virgin_material_substitution_ratio"], errors="raise"
        )
        mechanical_rows = recovery > 0
        expected_credit = (
            -pd.to_numeric(
                plastic_matrix.loc[
                    mechanical_rows, "gwp__virgin_polymer_production"
                ],
                errors="raise",
            )
            * recovery.loc[mechanical_rows]
            * substitution.loc[mechanical_rows]
        )
        actual_credit = pd.to_numeric(
            plastic_matrix.loc[
                mechanical_rows, "gwp__avoided_virgin_material_credit"
            ],
            errors="raise",
        )
        if not np.allclose(actual_credit, expected_credit, atol=1e-10):
            raise ValueError(
                "Avoided-material GWP must credit virgin-polymer production only; "
                "the SUT component-manufacturing stage is not substitutable."
            )

        human_health_columns = {
            "human_health_daly__production",
            "human_health_daly__virgin_polymer_production",
            "human_health_daly__sut_component_manufacturing",
            "human_health_daly__avoided_virgin_material_credit",
        }
        missing_human_health_columns = human_health_columns - set(plastic_matrix.columns)
        if missing_human_health_columns:
            raise ValueError(
                "Human-health endpoint stages are incomplete: "
                f"{sorted(missing_human_health_columns)}"
            )
        if not np.allclose(
            pd.to_numeric(plastic_matrix["human_health_daly__production"], errors="raise"),
            pd.to_numeric(
                plastic_matrix["human_health_daly__virgin_polymer_production"],
                errors="raise",
            )
            + pd.to_numeric(
                plastic_matrix["human_health_daly__sut_component_manufacturing"],
                errors="raise",
            ),
            atol=1e-15,
        ):
            raise ValueError(
                "Human-health production DALY must equal direct virgin-polymer "
                "endpoint damage plus the manual climate-health SUT stage."
            )
        expected_human_health_credit = (
            -pd.to_numeric(
                plastic_matrix.loc[
                    mechanical_rows,
                    "human_health_daly__virgin_polymer_production",
                ],
                errors="raise",
            )
            * recovery.loc[mechanical_rows]
            * substitution.loc[mechanical_rows]
        )
        actual_human_health_credit = pd.to_numeric(
            plastic_matrix.loc[
                mechanical_rows,
                "human_health_daly__avoided_virgin_material_credit",
            ],
            errors="raise",
        )
        if not np.allclose(
            actual_human_health_credit,
            expected_human_health_credit,
            atol=1e-15,
        ):
            raise ValueError(
                "Avoided human-health damage may credit virgin-polymer LCI only, "
                "not the manual SUT manufacturing proxy."
            )
        ecosystem_columns = {
            "ecosystem_quality__production",
            "ecosystem_quality__virgin_polymer_production",
            "ecosystem_quality__sut_component_manufacturing",
            "ecosystem_quality__avoided_virgin_material_credit",
        }
        if ecosystem_columns - set(plastic_matrix.columns):
            raise ValueError("Ecosystem-quality polymer stages are incomplete.")
        if not np.allclose(
            pd.to_numeric(plastic_matrix["ecosystem_quality__production"], errors="raise"),
            pd.to_numeric(plastic_matrix["ecosystem_quality__virgin_polymer_production"], errors="raise")
            + pd.to_numeric(plastic_matrix["ecosystem_quality__sut_component_manufacturing"], errors="raise"),
            atol=1e-18,
        ):
            raise ValueError("Ecosystem production must reconcile to polymer LCI plus the climate-only proxy.")
        expected_ecosystem_credit = (
            -pd.to_numeric(plastic_matrix.loc[mechanical_rows, "ecosystem_quality__virgin_polymer_production"], errors="raise")
            * recovery.loc[mechanical_rows] * substitution.loc[mechanical_rows]
        )
        actual_ecosystem_credit = pd.to_numeric(
            plastic_matrix.loc[mechanical_rows, "ecosystem_quality__avoided_virgin_material_credit"],
            errors="raise",
        )
        if not np.allclose(actual_ecosystem_credit, expected_ecosystem_credit, atol=1e-18):
            raise ValueError("Ecosystem avoided-material credit must displace virgin polymer LCI only.")

    factor_cols = [c for c in route_factor_matrix.columns if c.startswith("net_") and c.endswith("_factor_per_optimisation_unit")]
    route_factor_matrix["route_factor_basis"] = "net factor per optimisation unit"
    route_factor_matrix["factor_sign_convention"] = "positive = burden; negative = avoided-burden credit"
    route_factor_matrix["double_counting_warning"] = (
        "Use net factor columns for route decisions. Stage-breakdown columns are explanatory and should not be added again."
    )
    route_factor_matrix["ecoinvent_version"] = settings["brightway"]["ecoinvent_version"]
    route_factor_matrix["ecoinvent_system_model"] = settings["brightway"]["system_model"]
    route_factor_matrix["brightway_project"] = settings["brightway"]["project"]
    route_factor_matrix["gwp_method"] = " | ".join(settings["lcia"]["methods"]["gwp"])
    route_factor_matrix["gwp_method_label"] = settings["lcia"]["gwp_method_label"]
    route_factor_matrix["fossil_resource_method"] = " | ".join(
        settings["lcia"]["methods"]["fossil_resource_use"]
    )
    route_factor_matrix["water_scarcity_method"] = " | ".join(
        settings["lcia"]["methods"]["water_scarcity"]
    )
    water_settings = settings["lcia"]["water_scarcity_regionalisation"]
    route_factor_matrix["water_scarcity_regionalisation_audit"] = (
        "ecoinvent reduced-footprint method audited as world-generic; only traced "
        "foreground tap-water columns are replaced with site AWARE 2.0 factors"
    )
    route_factor_matrix["water_scarcity_regionalisation_equation"] = water_settings[
        "correction_equation"
    ]
    route_factor_matrix["water_scarcity_background_scope"] = water_settings[
        "background_scope"
    ]
    route_factor_matrix["water_scarcity_source_references"] = " | ".join(
        water_settings["source_references"]
    )
    for site, specification in water_settings["sites"].items():
        route_factor_matrix[f"aware20_site_cf__{site}"] = float(
            specification["factor_m3_world_eq_per_m3"]
        )
        route_factor_matrix[f"aware20_basin_id__{site}"] = int(specification["basin_id"])
    route_factor_matrix["sut_manufacturing_gwp_adder_kgco2e_per_kg"] = float(
        settings["plastics"]["sut_manufacturing_gwp_adder_kgco2e_per_kg"]
    )
    route_factor_matrix["sut_manufacturing_gwp_adder_value_qualifier"] = settings[
        "plastics"
    ]["sut_manufacturing_gwp_adder_value_qualifier"]
    route_factor_matrix["sut_manufacturing_gwp_adder_source"] = settings["plastics"][
        "sut_manufacturing_gwp_adder_source"
    ]
    route_factor_matrix["sut_manufacturing_gwp_adder_source_references"] = " | ".join(
        settings["plastics"]["sut_manufacturing_gwp_adder_source_references"]
    )
    route_factor_matrix["sut_manufacturing_gwp_adder_boundary"] = settings["plastics"][
        "sut_manufacturing_gwp_adder_boundary"
    ]
    route_factor_matrix["human_health_daly_method"] = " | ".join(
        settings["human_health_daly"]["method"]
    )
    route_factor_matrix["human_health_daly_unit"] = "DALY"
    route_factor_matrix["human_health_daly_equation"] = settings[
        "human_health_daly"
    ]["equation"]
    route_factor_matrix["human_health_daly_contributing_midpoints"] = " | ".join(
        settings["human_health_daly"]["contributing_midpoint_pathways"]
    )
    route_factor_matrix["human_health_daly_direct_endpoint_validation_method"] = (
        " | ".join(
            settings["human_health_daly"]["direct_endpoint_validation_method"]
        )
    )
    route_factor_matrix["human_health_daly_direct_endpoint_role"] = settings[
        "human_health_daly"
    ]["direct_endpoint_role"]
    route_factor_matrix["human_health_daly_water_method_note"] = settings[
        "human_health_daly"
    ]["reported_water_scarcity_relationship"]
    route_factor_matrix[
        "sut_manufacturing_climate_health_daly_per_kg_co2e"
    ] = float(
        settings["human_health_daly"]
        ["climate_change_damage_factor_daly_per_kg_co2e"]
    )
    route_factor_matrix["human_health_daly_reporting_only"] = True
    route_factor_matrix["ecosystem_quality_method"] = " | ".join(
        settings["ecosystem_quality"]["method"]
    )
    route_factor_matrix["ecosystem_quality_unit"] = "species.yr"
    route_factor_matrix["ecosystem_quality_reporting_only"] = True
    route_factor_matrix["ecosystem_quality_manual_sut_proxy_scope"] = settings[
        "ecosystem_quality"
    ]["manual_sut_proxy_scope"]
    route_factor_matrix["packaging_uplift_rate"] = float(
        settings["plastics"]["packaging_uplift_rate"]
    )

    # Sort for readability.
    module_order = {
        "manufacturing_footprint": 1,
        "utility_water_supply": 2,
        "sut_plastics": 3,
        "wastewater": 4,
        "wastewater_lca": 4,
        "biomass": 5,
        "biomass_lca": 5,
    }
    route_factor_matrix["_module_order"] = route_factor_matrix["source_module"].map(module_order).fillna(99)
    route_factor_matrix = (
        route_factor_matrix
        .sort_values(["_module_order", "stream_category", "material_or_service", "pathway"])
        .drop(columns="_module_order")
        .reset_index(drop=True)
    )

    print("Route factor matrix shape:", route_factor_matrix.shape)
    display(route_factor_matrix.head(20))


    # ---------------------------------------------------------------------
    # Build input/output table for the optimisation model
    # ---------------------------------------------------------------------
    input_output_rows = []


    # ---------------------------------------------------------------------
    # Manufacturing foreground inputs
    # ---------------------------------------------------------------------
    manufacturing_main = main_tables["manufacturing"].copy()
    required_manufacturing_main_cols = {
        "lca_flow_key",
        "inventory_group",
        "source_basis",
        "source_unit",
        "activity_unit",
        "source_quantity",
        "activity_demand_amount_per_batch",
    }

    if required_manufacturing_main_cols.issubset(manufacturing_main.columns):
        group_cols = [
            "lca_flow_key",
            "inventory_group",
            "source_basis",
            "source_unit",
            "activity_unit",
        ]
        if "conversion_note" in manufacturing_main.columns:
            group_cols.append("conversion_note")
        else:
            manufacturing_main["conversion_note"] = "derived from manufacturing main_detailed_table"
            group_cols.append("conversion_note")

        manufacturing_input_summary = (
            manufacturing_main
            .groupby(group_cols, dropna=False, as_index=False)
            .agg(
                quantity_per_batch=("source_quantity", "first"),
                optimisation_quantity_per_batch=("activity_demand_amount_per_batch", "first"),
            )
        )

        for _, row in manufacturing_input_summary.iterrows():
            input_output_rows.append({
                "source_module": "manufacturing_footprint",
                "io_direction": "input",
                "io_table_role": "foreground_process_requirement",
                "stream_category": row["inventory_group"],
                "material_or_service": row["lca_flow_key"],
                "pathway": row["lca_flow_key"],
                "quantity_per_batch": row["quantity_per_batch"],
                "quantity_unit": row["source_unit"],
                "optimisation_quantity_per_batch": row["optimisation_quantity_per_batch"],
                "optimisation_quantity_unit": row["activity_unit"],
                "included_in_lca": True,
                "routable_in_minlp": False,
                "fixed_process_requirement": True,
                "source_basis": row["source_basis"],
                "notes": row["conversion_note"],
            })
    else:
        missing = sorted(required_manufacturing_main_cols.difference(manufacturing_main.columns))
        raise ValueError(
            "Manufacturing detailed LCA table is missing required foreground columns: "
            + ", ".join(missing)
        )


    # ---------------------------------------------------------------------
    # Wastewater outputs: route alternatives for the same generated volume
    # ---------------------------------------------------------------------
    wastewater_routes = route_tables["wastewater"].copy()
    if "route" not in wastewater_routes.columns and "pathway" in wastewater_routes.columns:
        wastewater_routes["route"] = wastewater_routes["pathway"]
    if "quantity_unit" not in wastewater_routes.columns:
        wastewater_routes["quantity_unit"] = "m3"
    if "eol_pathway" not in wastewater_routes.columns:
        wastewater_routes["eol_pathway"] = wastewater_routes["route"]
    if "eol_sink" not in wastewater_routes.columns:
        wastewater_routes["eol_sink"] = "not_applicable"

    volume_col = first_existing_column_name(
        wastewater_routes,
        ["represented_volume_m3_per_batch", "activity_amount_per_batch", "routed_volume_m3_per_batch"],
        required=True,
        label="wastewater represented volume",
    )

    wastewater_io = (
        wastewater_routes
        .groupby(["route", "eol_pathway", "eol_sink", "quantity_unit"], dropna=False, as_index=False)
        .agg(quantity_per_batch=(volume_col, "max"))
    )
    for _, row in wastewater_io.iterrows():
        quantity_unit = str(row["quantity_unit"])
        input_output_rows.append({
            "source_module": "wastewater_lca",
            "io_direction": "output",
            "io_table_role": "routable_waste_stream",
            "stream_category": "plant_wastewater",
            "material_or_service": "plant_wastewater",
            "pathway": row["route"],
            "quantity_per_batch": row["quantity_per_batch"],
            "quantity_unit": f"{quantity_unit}/batch" if not quantity_unit.endswith("/batch") else quantity_unit,
            "optimisation_quantity_per_batch": row["quantity_per_batch"],
            "optimisation_quantity_unit": quantity_unit.replace("/batch", ""),
            "included_in_lca": True,
            "routable_in_minlp": True,
            "fixed_process_requirement": False,
            "source_basis": "wastewater route_factor_table represented volume",
            "notes": f"Alternative pathway; eol_sink={row['eol_sink']}",
        })


    # ---------------------------------------------------------------------
    # SUT plastics outputs: reconstruct route-content summary if workbook sheet is absent
    # ---------------------------------------------------------------------
    route_content_summary = optional_tables.get("sut_plastics", {}).get("route_content_summary")

    if route_content_summary is None:
        sut_route_table = route_tables["sut_plastics"].copy()
        polymer_col = first_existing_column_name(sut_route_table, ["polymer", "Polymer"], required=True, label="SUT polymer")
        mass_col = first_existing_column_name(
            sut_route_table,
            ["represented_mass_kg_per_batch", "polymer_kg_per_batch", "quantity_per_batch"],
            required=True,
            label="SUT represented mass",
        )

        if "eol_pathway" not in sut_route_table.columns and "pathway" in sut_route_table.columns:
            sut_route_table["eol_pathway"] = sut_route_table["pathway"]
        if "eol_sink" not in sut_route_table.columns:
            sut_route_table["eol_sink"] = "not_applicable"
        if "mechanically_recycled" not in sut_route_table.columns:
            sut_route_table["mechanically_recycled"] = False
        if "mapping_assumption_level" not in sut_route_table.columns:
            sut_route_table["mapping_assumption_level"] = np.nan
        if "routing_note" not in sut_route_table.columns:
            sut_route_table["routing_note"] = np.nan

        # Use EOL treatment rows where available to avoid counting production/credit rows.
        eol_mask = pd.Series([True] * len(sut_route_table), index=sut_route_table.index)
        if "life_cycle_stage" in sut_route_table.columns:
            eol_mask = sut_route_table["life_cycle_stage"].astype(str).eq("plastic_end_of_life")
        elif "is_eol_treatment" in sut_route_table.columns:
            eol_mask = as_bool_series(sut_route_table["is_eol_treatment"], default=False)

        sut_route_content_base = sut_route_table[eol_mask].copy()
        if sut_route_content_base.empty:
            sut_route_content_base = sut_route_table.copy()

        route_content_summary = (
            sut_route_content_base
            .groupby(
                ["eol_pathway", "eol_sink", polymer_col, "mechanically_recycled", "mapping_assumption_level", "routing_note"],
                dropna=False,
                as_index=False,
            )
            .agg(polymer_kg_per_batch=(mass_col, "max"))
            .rename(columns={polymer_col: "polymer"})
        )
        route_content_summary["share_of_pathway_mass"] = route_content_summary.groupby("eol_pathway")["polymer_kg_per_batch"].transform(
            lambda x: x / x.sum() if x.sum() else 0
        )

    for _, row in route_content_summary.iterrows():
        input_output_rows.append({
            "source_module": "sut_plastics_lca",
            "io_direction": "output",
            "io_table_role": "routable_waste_stream",
            "stream_category": "sut_plastic_waste",
            "material_or_service": row["polymer"],
            "pathway": row["eol_pathway"],
            "quantity_per_batch": row["polymer_kg_per_batch"],
            "quantity_unit": "kg/batch",
            "optimisation_quantity_per_batch": row["polymer_kg_per_batch"],
            "optimisation_quantity_unit": "kg",
            "included_in_lca": True,
            "routable_in_minlp": True,
            "fixed_process_requirement": False,
            "source_basis": "SUT plastics route_content_summary reconstructed from route_factor_table if workbook sheet absent",
            "notes": f"eol_sink={row.get('eol_sink', np.nan)}; mechanically_recycled={row.get('mechanically_recycled', np.nan)}; share={row.get('share_of_pathway_mass', np.nan)}",
        })


    # ---------------------------------------------------------------------
    # Explicit utility-water demand and recovered-water substitution
    # ---------------------------------------------------------------------
    if "utility_water_accounting_table" not in locals() or utility_water_accounting_table.empty:
        raise ValueError("Utility-water accounting was not created before the input/output table.")

    utility_accounting = utility_water_accounting_table.iloc[0]

    input_output_rows.append({
        "source_module": "utility_water_supply",
        "io_direction": "input",
        "io_table_role": "fixed_utility_demand_balance",
        "stream_category": "non_contact_utility_water",
        "material_or_service": "utility_water",
        "pathway": "fresh_utility_water_supply",
        "quantity_per_batch": utility_accounting["utility_water_demand_m3_per_batch"],
        "quantity_unit": "m3/batch",
        "optimisation_quantity_per_batch": utility_accounting["utility_water_demand_m3_per_batch"],
        "optimisation_quantity_unit": "m3",
        "included_in_lca": True,
        "routable_in_minlp": True,
        "fixed_process_requirement": True,
        "source_basis": utility_accounting["utility_water_demand_basis"],
        "notes": (
            "Demand balance required in optimisation: fresh utility water + internally "
            "reused water = utility-water demand. This is non-contact utility water, not WFI."
        ),
    })

    input_output_rows.append({
        "source_module": "wastewater_lca",
        "io_direction": "secondary_output",
        "io_table_role": "recoverable_internal_utility_water",
        "stream_category": "treated_wastewater",
        "material_or_service": "recovered_utility_water",
        "pathway": UTILITY_WATER_REUSE_CONFIG["reuse_pathway"],
        "quantity_per_batch": utility_accounting["recoverable_water_m3_per_batch"],
        "quantity_unit": "m3/batch",
        "optimisation_quantity_per_batch": utility_accounting["recoverable_water_m3_per_batch"],
        "optimisation_quantity_unit": "m3",
        "included_in_lca": True,
        "routable_in_minlp": True,
        "fixed_process_requirement": False,
        "source_basis": "Wastewater routed volume multiplied by configured recovery fraction.",
        "notes": (
            "Can replace non-contact utility water only. Constraint required: reused water "
            "<= recoverable water and <= utility-water demand. WFI displacement is zero."
        ),
    })


    # ---------------------------------------------------------------------
    # Residual biomass output and fixed AD valorisation pathway
    # ---------------------------------------------------------------------
    biomass_inventory = optional_tables.get("biomass", {}).get("inventory", pd.DataFrame())
    if biomass_inventory is None or biomass_inventory.empty:
        raise ValueError("The biomass module did not provide its SuperPro inventory row.")
    biomass_quantity = float(pd.to_numeric(
        biomass_inventory.iloc[0]["biomass_kg_per_batch"], errors="raise"
    ))
    biomass_pathway = str(settings["biomass"]["pathway"])
    input_output_rows.append({
        "source_module": "biomass",
        "io_direction": "output",
        "io_table_role": "routable_waste_stream",
        "stream_category": "residual_biomass",
        "material_or_service": "residual_biomass",
        "pathway": biomass_pathway,
        "quantity_per_batch": biomass_quantity,
        "quantity_unit": "kg/batch",
        "optimisation_quantity_per_batch": biomass_quantity,
        "optimisation_quantity_unit": "kg",
        "included_in_lca": True,
        "routable_in_minlp": True,
        "fixed_process_requirement": False,
        "source_basis": "Overall SuperPro component balance output",
        "notes": (
            "Residual biomass routed to anaerobic digestion with biomethane upgrading "
            "and agricultural digestate recovery. CHP credits are not included."
        ),
    })

    # ---------------------------------------------------------------------
    # Secondary outputs and traceability rows from all modules
    # ---------------------------------------------------------------------
    for module_name, secondary_df in secondary_tables.items():
        if secondary_df is None or len(secondary_df) == 0:
            continue
        for _, row in secondary_df.iterrows():
            input_output_rows.append({
                "source_module": module_name,
                "io_direction": row.get("stream_direction", "secondary"),
                "io_table_role": "secondary_output_or_traceability",
                "stream_category": row.get("secondary_stream_category", np.nan),
                "material_or_service": row.get("secondary_stream_name", np.nan),
                "pathway": row.get("parent_route", np.nan),
                "quantity_per_batch": row.get("quantity_per_batch", np.nan),
                "quantity_unit": row.get("quantity_unit", np.nan),
                "optimisation_quantity_per_batch": np.nan,
                "optimisation_quantity_unit": row.get("quantity_unit", np.nan),
                "included_in_lca": parse_bool(row.get("embedded_in_lca_activity", False), default=False),
                "routable_in_minlp": parse_bool(row.get("routable_in_master", False), default=False),
                "fixed_process_requirement": False,
                "source_basis": row.get("source_of_quantity", np.nan),
                "notes": row.get("assumption_note", np.nan),
            })

    input_output_table = pd.DataFrame(input_output_rows)

    print("Input/output table shape:", input_output_table.shape)
    display(input_output_table.head(30))


    # ---------------------------------------------------------------------
    # Validation checks for model-facing tables
    # ---------------------------------------------------------------------
    required_route_cols = [
        "source_module",
        "stream_category",
        "material_or_service",
        "pathway",
        "optimisation_quantity_unit",
    ]
    missing_route_cols = set(required_route_cols) - set(route_factor_matrix.columns)
    if missing_route_cols:
        raise ValueError(f"route_factor_matrix missing required columns: {sorted(missing_route_cols)}")

    factor_cols = [c for c in route_factor_matrix.columns if c.startswith("net_") and c.endswith("_factor_per_optimisation_unit")]
    if not factor_cols:
        raise ValueError("No net impact factor columns were created.")

    if route_factor_matrix[factor_cols].isna().any().any():
        bad = route_factor_matrix[route_factor_matrix[factor_cols].isna().any(axis=1)]
        display(bad)
        raise ValueError("At least one net route factor is missing.")

    required_io_cols = ["source_module", "io_direction", "io_table_role", "stream_category", "quantity_per_batch", "quantity_unit"]
    missing_io_cols = set(required_io_cols) - set(input_output_table.columns)
    if missing_io_cols:
        raise ValueError(f"input_output_table missing required columns: {sorted(missing_io_cols)}")

    print("Validation passed.")
    print(f"Route-factor rows: {len(route_factor_matrix):,}")
    print(f"Input/output rows: {len(input_output_table):,}")
    print("Factor columns:", factor_cols)

    utility_supply_rows_check = route_factor_matrix[
        route_factor_matrix["source_module"].astype(str).eq("utility_water_supply")
    ]
    if utility_supply_rows_check.empty:
        raise ValueError("No explicit fresh utility-water supply route was created.")

    reuse_credit_check = route_factor_long[
        route_factor_long["source_module"].astype(str).str.contains("wastewater", case=False, na=False)
        & route_factor_long["pathway"].astype(str).eq(UTILITY_WATER_REUSE_CONFIG["reuse_pathway"])
        & route_factor_long["is_credit"].fillna(False).astype(bool)
    ]
    if reuse_credit_check.empty:
        raise ValueError("No wastewater utility-reuse credit remains after harmonisation.")

    credit_activity_text = reuse_credit_check["activity_name"].fillna("").astype(str).str.lower()
    if credit_activity_text.str.contains("ultrapure|wfi", regex=True).any():
        raise ValueError("The utility-water reuse credit is incorrectly displacing WFI/ultrapure water.")

    utility_accounting_check = utility_water_accounting_table.iloc[0]
    if utility_accounting_check["reused_water_m3_per_batch"] > utility_accounting_check["utility_water_demand_m3_per_batch"] + 1e-9:
        raise ValueError("Reused water exceeds non-contact utility-water demand.")
    if utility_accounting_check["wfi_displaced_m3_per_batch"] != 0:
        raise ValueError("WFI displacement must remain zero for this reuse case.")

    print("Utility-water substitution validation passed: WFI unchanged and reuse capped by demand.")


    # ---------------------------------------------------------------------
    # Scenario comparison data table
    # ---------------------------------------------------------------------
    # Each scenario contains:
    #   - fixed manufacturing inputs, with one selected electricity pathway;
    #   - explicit fresh non-contact utility-water demand;
    #   - one plastics pathway;
    #   - one wastewater pathway;
    #   - the fixed biomass AD-to-biomethane pathway.
    #
    # In the reuse case, the wastewater route contains a negative avoided utility-
    # water factor capped by the recoverable volume and utility-water demand. WFI is
    # retained unchanged in every scenario.

    factor_cols = [
        c for c in route_factor_matrix.columns
        if c.startswith("net_") and c.endswith("_factor_per_optimisation_unit")
    ]
    impact_categories = [
        c.replace("net_", "").replace("_factor_per_optimisation_unit", "")
        for c in factor_cols
    ]

    SCENARIO_DEFINITIONS = settings["scenarios"]

    scenario_rows = []


    def add_scenario_component(scenario_name, module_label, component_df):
        for _, row in component_df.iterrows():
            qty = pd.to_numeric(row["represented_quantity_per_batch"], errors="coerce")
            for impact_category in impact_categories:
                factor_col = f"net_{impact_category}_factor_per_optimisation_unit"
                factor = pd.to_numeric(row[factor_col], errors="coerce")
                scenario_rows.append({
                    "scenario": scenario_name,
                    "scenario_note": SCENARIO_DEFINITIONS[scenario_name].get("scenario_note", ""),
                    "source_module": module_label,
                    "stream_category": row["stream_category"],
                    "material_or_service": row["material_or_service"],
                    "pathway": row["pathway"],
                    "impact_category": impact_category,
                    "quantity_per_batch": qty,
                    "quantity_unit": row["optimisation_quantity_unit"],
                    "net_factor": factor,
                    "impact_per_batch": factor * qty,
                })


    manufacturing_components = route_factor_matrix[
        route_factor_matrix["source_module"].astype(str).str.contains("manufacturing", case=False, na=False)
    ].copy()
    utility_water_components = route_factor_matrix[
        route_factor_matrix["source_module"].astype(str).eq("utility_water_supply")
    ].copy()
    plastics_components = route_factor_matrix[
        route_factor_matrix["source_module"].astype(str).str.contains("sut", case=False, na=False)
    ].copy()
    wastewater_components = route_factor_matrix[
        route_factor_matrix["source_module"].astype(str).str.contains("wastewater", case=False, na=False)
    ].copy()
    biomass_components = route_factor_matrix[
        route_factor_matrix["source_module"].astype(str).str.contains("biomass", case=False, na=False)
    ].copy()

    if utility_water_components.empty:
        raise ValueError("Explicit utility-water supply rows are missing from route_factor_matrix.")


    def lower_text(column):
        if column in manufacturing_components.columns:
            return manufacturing_components[column].astype(str).str.lower()
        return pd.Series([""] * len(manufacturing_components), index=manufacturing_components.index)


    manufacturing_is_electricity = (
        lower_text("material_or_service").eq("electricity")
        | lower_text("material_or_service").str.contains("electricity", na=False)
        | lower_text("optimisation_flow").str.contains("electricity", na=False)
        | lower_text("pathway").str.startswith("electricity_")
        | lower_text("activity_names").str.contains("electricity", na=False)
    )

    expected_electricity_pathways = {
        "electricity_grid_uk_bau",
        "electricity_renewable_onshore_wind",
    }
    electricity_pathways_detected = set(
        manufacturing_components.loc[
            manufacturing_is_electricity,
            "pathway",
        ].dropna().astype(str).unique()
    )
    missing_expected = expected_electricity_pathways - electricity_pathways_detected
    if missing_expected:
        raise ValueError(
            "Missing expected electricity pathways: " + ", ".join(sorted(missing_expected))
        )

    for scenario_name, choices in SCENARIO_DEFINITIONS.items():
        electricity_pathway = choices["electricity_pathway"]

        manufacturing_fixed = manufacturing_components[~manufacturing_is_electricity].copy()
        manufacturing_electricity_choice = manufacturing_components[
            manufacturing_is_electricity
            & manufacturing_components["pathway"].astype(str).eq(electricity_pathway)
        ].copy()

        if manufacturing_electricity_choice.empty:
            available = sorted(electricity_pathways_detected)
            raise ValueError(
                f"Electricity pathway not found for scenario {scenario_name}: {electricity_pathway}\n"
                f"Available electricity pathways: {available}"
            )

        plastics_choice = plastics_components[
            plastics_components["pathway"].astype(str).eq(choices["plastic_pathway"])
        ].copy()
        wastewater_choice = wastewater_components[
            wastewater_components["pathway"].astype(str).eq(choices["wastewater_pathway"])
        ].copy()
        biomass_choice = biomass_components[
            biomass_components["pathway"].astype(str).eq(choices["biomass_pathway"])
        ].copy()

        if plastics_choice.empty:
            available = sorted(plastics_components["pathway"].dropna().astype(str).unique())
            raise ValueError(
                f"Plastic pathway not found for scenario {scenario_name}: {choices['plastic_pathway']}\n"
                f"Available plastic pathways: {available}"
            )
        if wastewater_choice.empty:
            available = sorted(wastewater_components["pathway"].dropna().astype(str).unique())
            raise ValueError(
                f"Wastewater pathway not found for scenario {scenario_name}: {choices['wastewater_pathway']}\n"
                f"Available wastewater pathways: {available}"
            )
        if biomass_choice.empty:
            available = sorted(biomass_components["pathway"].dropna().astype(str).unique())
            raise ValueError(
                f"Biomass pathway not found for scenario {scenario_name}: {choices['biomass_pathway']}\n"
                f"Available biomass pathways: {available}"
            )

        add_scenario_component(
            scenario_name,
            "manufacturing",
            pd.concat([manufacturing_fixed, manufacturing_electricity_choice], ignore_index=True),
        )
        add_scenario_component(scenario_name, "utility_water_supply", utility_water_components)
        add_scenario_component(scenario_name, "sut_plastics", plastics_choice)
        add_scenario_component(scenario_name, "wastewater", wastewater_choice)
        add_scenario_component(scenario_name, "biomass", biomass_choice)

    scenario_impact_table = pd.DataFrame(scenario_rows)
    scenario_totals = (
        scenario_impact_table
        .groupby(["scenario", "scenario_note", "impact_category"], as_index=False)
        .agg(net_impact_per_batch=("impact_per_batch", "sum"))
    )
    scenario_module_breakdown = (
        scenario_impact_table
        .groupby(["scenario", "impact_category", "source_module"], as_index=False)
        .agg(impact_per_batch=("impact_per_batch", "sum"))
    )

    # Physical water-volume accounting. This is deliberately separate from LCIA.
    accounting = utility_water_accounting_table.iloc[0]
    wfi_rows = manufacturing_components[
        manufacturing_components["material_or_service"].astype(str).str.lower().eq("wfi")
    ].copy()
    if wfi_rows.empty:
        wfi_volume_m3_per_batch = 0.0
    else:
        wfi_quantity = float(pd.to_numeric(wfi_rows["represented_quantity_per_batch"], errors="coerce").max())
        wfi_unit = str(wfi_rows["optimisation_quantity_unit"].iloc[0]).strip().lower()
        mass_units = {"kg", "kilogram", "kilograms"}
        wfi_volume_m3_per_batch = wfi_quantity / 1000.0 if wfi_unit in mass_units else wfi_quantity

    water_substitution_rows = []
    for scenario_name, choices in SCENARIO_DEFINITIONS.items():
        reuse_selected = choices["wastewater_pathway"] == UTILITY_WATER_REUSE_CONFIG["reuse_pathway"]
        reused_m3 = float(accounting["reused_water_m3_per_batch"]) if reuse_selected else 0.0
        utility_demand_m3 = float(accounting["utility_water_demand_m3_per_batch"])
        fresh_utility_m3 = utility_demand_m3 - reused_m3
        water_substitution_rows.append({
            "scenario": scenario_name,
            "wfi_m3_per_batch": wfi_volume_m3_per_batch,
            "utility_water_demand_m3_per_batch": utility_demand_m3,
            "recycled_utility_water_m3_per_batch": reused_m3,
            "fresh_utility_water_m3_per_batch": fresh_utility_m3,
            "net_freshwater_withdrawal_m3_per_batch": wfi_volume_m3_per_batch + fresh_utility_m3,
            "wfi_displaced_m3_per_batch": 0.0,
        })
    water_substitution_summary = pd.DataFrame(water_substitution_rows)

    print("Scenario totals:")
    display(scenario_totals)
    print("Physical utility-water substitution summary (not an LCIA result):")
    display(water_substitution_summary)


    excluded_scenarios = set()
    scenario_order = list(SCENARIO_DEFINITIONS)

    # ---------------------------------------------------------------------
    # Physical freshwater-withdrawal balance for utility-water substitution
    # ---------------------------------------------------------------------
    # This is a foreground water-volume check in m3/batch, not an LCIA result.
    # It makes the internal substitution visible without conflating water volume
    # with the selected water-scarcity characterisation method.

    water_plot_df = water_substitution_summary[
        ~water_substitution_summary["scenario"].isin(excluded_scenarios)
    ].copy()

    ordered_water_scenarios = [
        scenario for scenario in scenario_order
        if scenario in set(water_plot_df["scenario"])
    ]
    return {
        "route_factor_matrix": route_factor_matrix,
        "input_output_table": input_output_table,
        "route_factor_long": route_factor_long,
        "scenario_impact_table": scenario_impact_table,
        "scenario_totals": scenario_totals,
        "utility_water_accounting_table": utility_water_accounting_table,
        "freshwater_balance_by_scenario": water_substitution_summary,
    }
