from __future__ import annotations

from .human_health_daly import add_human_health_method
from .ecosystem_quality import add_ecosystem_method
from .lcia_methods import core_lcia_methods


def run_manufacturing_lca(manufacturing_inputs, itemised_costs, settings):
    """Run the existing manufacturing input and utility footprint logic in memory."""
    from datetime import datetime
    import math
    import warnings
    import numpy as np
    import pandas as pd
    import bw2data as bd
    import bw2calc as bc

    display = lambda *args, **kwargs: None
    manufacturing_inputs = manufacturing_inputs.copy()
    itemised_costs = itemised_costs.copy()
    RUN_DATE = datetime.now().strftime("%Y%m%d")
    ECOINVENT_VERSION = settings["brightway"]["ecoinvent_version"]
    ECOINVENT_SYSTEM_MODEL = settings["brightway"]["system_model"]
    BW_PROJECT = settings["brightway"]["project"]
    BACKGROUND_DB = settings["brightway"]["background_database"]
    BIOSPHERE_DB = settings["brightway"]["biosphere_database"]
    ELECTRICITY_COST_USD_PER_KWH = float(settings["manufacturing"].get("electricity_cost_usd_per_kwh", 0.10))
    WATER_CP_KJ_PER_KG_C = float(settings["manufacturing"].get("water_cp_kj_per_kg_c", 4.184))
    CHILLED_WATER_DELTA_T_C = float(settings["manufacturing"].get("chilled_water_delta_t_c", 5.0))
    COOLING_WATER_DELTA_T_C = float(settings["manufacturing"].get("cooling_water_delta_t_c", 10.0))
    STEAM_MJ_PER_MT = float(settings["manufacturing"].get("steam_mj_per_mt", 2257.0))
    KG_PER_MT = float(settings["manufacturing"].get("kg_per_mt", 1000.0))
    # ---------------------------------------------------------------------
    # Imports
    # ---------------------------------------------------------------------
    import math
    import warnings

    import numpy as np
    import pandas as pd

    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 180)

    # ---------------------------------------------------------------------
    # WFI / process water from overall component balance
    # ---------------------------------------------------------------------
    component_clean = manufacturing_inputs["component"].astype(str).str.strip().str.upper()

    # Traceability rows only: visible WFI in explicit input stream components.
    wfi_stream_detail_rows = manufacturing_inputs[
        manufacturing_inputs["record_type"].eq("input_stream_component")
        & component_clean.eq("WFI")
    ].copy()

    # Source-of-truth row for LCA: total WFI component entering the process.
    wfi_overall_component_rows = manufacturing_inputs[
        manufacturing_inputs["record_type"].eq("overall_component_input")
        & component_clean.eq("WFI")
    ].copy()

    if wfi_overall_component_rows.empty:
        raise ValueError("No overall_component_input row found for component == WFI.")

    # Validation only. These rows are not separately added to the LCA physical inventory.
    wfi_validation_by_record_type = (
        manufacturing_inputs[component_clean.eq("WFI")]
        .groupby("record_type", dropna=False, as_index=False)
        .agg(
            amount_per_batch_kg=("amount_per_batch", "sum"),
            row_count=("inventory_id", "count"),
        )
        .sort_values("record_type")
    )

    wfi_by_section = (
        wfi_stream_detail_rows
        .groupby(["superpro_section", "unit_operation"], dropna=False, as_index=False)
        .agg(
            amount_per_batch_kg=("amount_per_batch", "sum"),
            row_count=("inventory_id", "count"),
        )
        .sort_values("amount_per_batch_kg", ascending=False)
    )

    stream_detail_wfi_kg = float(wfi_stream_detail_rows["amount_per_batch"].sum())
    overall_wfi_kg = float(wfi_overall_component_rows["amount_per_batch"].sum())

    # This is the LCA WFI quantity.
    wfi_kg_per_batch = overall_wfi_kg

    residual_wfi_kg = float(
        manufacturing_inputs[
            component_clean.eq("WFI")
            & manufacturing_inputs["record_type"].eq("overall_minus_stream_input_residual")
        ]["amount_per_batch"].sum()
    )

    wfi_reconciliation = pd.DataFrame([
        {
            "check": "WFI overall component input used for LCA",
            "amount_kg_per_batch": overall_wfi_kg,
            "used_in_lca": True,
            "note": "Source of truth for WFI LCA inventory: overall_component_input row where component == WFI.",
        },
        {
            "check": "WFI explicit stream-detail component sum",
            "amount_kg_per_batch": stream_detail_wfi_kg,
            "used_in_lca": False,
            "note": "Traceability only; not added on top of overall component balance.",
        },
        {
            "check": "WFI residual required to close totalised balance",
            "amount_kg_per_batch": residual_wfi_kg,
            "used_in_lca": False,
            "note": "Validation only; shows WFI not visible in explicit Stream Details component rows.",
        },
        {
            "check": "Overall minus stream-detail WFI",
            "amount_kg_per_batch": overall_wfi_kg - stream_detail_wfi_kg,
            "used_in_lca": False,
            "note": "Should match the residual if the workbook reconciliation is internally consistent.",
        },
    ])

    print(f"WFI used for LCA from overall component balance: {wfi_kg_per_batch:,.2f} kg/batch")
    print(f"Explicit stream-detail WFI shown for traceability only: {stream_detail_wfi_kg:,.2f} kg/batch")
    print(f"Residual WFI not visible in explicit stream details: {residual_wfi_kg:,.2f} kg/batch")

    display(wfi_validation_by_record_type)
    display(wfi_reconciliation)
    display(wfi_by_section)
    display(wfi_overall_component_rows)

    # ---------------------------------------------------------------------
    # Utility quantities from itemised costs only, with brine excluded
    # ---------------------------------------------------------------------
    UTILITY_ITEM_TO_FLOW_KEY = {
        "Std Power": "electricity",
        "Steam": "steam",
        "Chilled Water": "chilled_water",
        "Cooling Water": "cooling_water",
    }

    utility_source_rows = itemised_costs[
        itemised_costs["cost_group"].eq("utility")
        & itemised_costs["item_name"].isin(UTILITY_ITEM_TO_FLOW_KEY.keys())
    ].copy()

    excluded_utility_rows = itemised_costs[
        itemised_costs["cost_group"].eq("utility")
        & ~itemised_costs["item_name"].isin(UTILITY_ITEM_TO_FLOW_KEY.keys())
    ].copy()

    utility_source_rows["lca_flow_key"] = utility_source_rows["item_name"].map(UTILITY_ITEM_TO_FLOW_KEY)

    utility_quantity_by_item = (
        utility_source_rows
        .groupby(["lca_flow_key", "item_name", "amount_unit_per_batch"], dropna=False, as_index=False)
        .agg(
            source_amount_per_batch=("amount_per_batch", "sum"),
            source_cost_per_batch_usd=("cost_per_batch_usd", "sum"),
            row_count=("cost_id", "count"),
        )
        .sort_values("lca_flow_key")
    )

    print("Included utility rows:")
    display(utility_quantity_by_item)

    print("Excluded utility rows, including CaCl2 Brine:")
    display(excluded_utility_rows[["cost_id", "item_name", "procedure_or_equipment", "amount_per_batch", "amount_unit_per_batch", "cost_per_batch_usd"]])

    # ---------------------------------------------------------------------
    # Build physical LCA inventory
    # ---------------------------------------------------------------------
    physical_inventory_rows = []

    # WFI: overall component-balance kg directly maps to kg of selected water activity.
    physical_inventory_rows.append({
        "lca_flow_key": "wfi",
        "inventory_group": "process_water",
        "source_basis": "SR overall component balance: overall_component_input row where component == WFI",
        "source_quantity": wfi_kg_per_batch,
        "source_unit": "kg/batch",
        "activity_demand_amount_per_batch": wfi_kg_per_batch,
        "activity_unit": "kilogram",
        "conversion_note": "No conversion: WFI overall component balance is kg/batch and selected activity unit is kg.",
    })

    # Utilities: source quantities from itemised cost rows.
    for _, row in utility_quantity_by_item.iterrows():
        key = row["lca_flow_key"]
        qty = float(row["source_amount_per_batch"])
        src_unit = row["amount_unit_per_batch"]

        if key == "electricity":
            activity_amount = qty
            activity_unit = "kilowatt hour"
            note = "No conversion: SuperPro Std Power is kW-h/batch and selected activity unit is kWh."
        elif key == "steam":
            activity_amount = qty * STEAM_MJ_PER_MT
            activity_unit = "megajoule"
            note = f"Converted from MT steam to MJ heat using STEAM_MJ_PER_MT={STEAM_MJ_PER_MT:g}."
        elif key == "chilled_water":
            activity_amount = qty * KG_PER_MT * WATER_CP_KJ_PER_KG_C * CHILLED_WATER_DELTA_T_C / 1000.0
            activity_unit = "megajoule"
            note = (
                "Converted from MT chilled water to MJ cooling energy using "
                f"cp={WATER_CP_KJ_PER_KG_C:g} kJ/kg/C and deltaT={CHILLED_WATER_DELTA_T_C:g} C."
            )
        elif key == "cooling_water":
            activity_amount = qty * KG_PER_MT * WATER_CP_KJ_PER_KG_C * COOLING_WATER_DELTA_T_C / 1000.0
            activity_unit = "megajoule"
            note = (
                "Converted from MT cooling water to MJ cooling energy using "
                f"cp={WATER_CP_KJ_PER_KG_C:g} kJ/kg/C and deltaT={COOLING_WATER_DELTA_T_C:g} C."
            )
        else:
            raise ValueError(f"Unexpected utility key: {key}")

        physical_inventory_rows.append({
            "lca_flow_key": key,
            "inventory_group": "utility",
            "source_basis": "ICR itemised utility cost rows only",
            "source_quantity": qty,
            "source_unit": src_unit,
            "activity_demand_amount_per_batch": activity_amount,
            "activity_unit": activity_unit,
            "conversion_note": note,
        })

    physical_inventory = pd.DataFrame(physical_inventory_rows)

    display(physical_inventory)

    # ---------------------------------------------------------------------
    # Cost analysis
    # ---------------------------------------------------------------------
    cost_analysis = itemised_costs.copy()

    # Preserve original SuperPro cost.
    cost_analysis["superpro_cost_per_batch_usd"] = cost_analysis["cost_per_batch_usd"]
    cost_analysis["adjusted_cost_per_batch_usd"] = cost_analysis["cost_per_batch_usd"]
    cost_analysis["cost_adjustment_note"] = "SuperPro itemised cost retained."

    # Recalculate electricity cost from editable parameter.
    electricity_mask = cost_analysis["cost_group"].eq("utility") & cost_analysis["item_name"].eq("Std Power")
    cost_analysis.loc[electricity_mask, "adjusted_cost_per_batch_usd"] = (
        cost_analysis.loc[electricity_mask, "amount_per_batch"].astype(float) * ELECTRICITY_COST_USD_PER_KWH
    )
    cost_analysis.loc[electricity_mask, "cost_adjustment_note"] = (
        f"Recalculated as kWh * ELECTRICITY_COST_USD_PER_KWH={ELECTRICITY_COST_USD_PER_KWH:g}."
    )

    cost_summary_by_group = (
        cost_analysis
        .groupby("cost_group", dropna=False, as_index=False)
        .agg(
            superpro_cost_per_batch_usd=("superpro_cost_per_batch_usd", "sum"),
            adjusted_cost_per_batch_usd=("adjusted_cost_per_batch_usd", "sum"),
            row_count=("cost_id", "count"),
        )
        .sort_values("adjusted_cost_per_batch_usd", ascending=False)
    )

    cost_summary_by_item = (
        cost_analysis
        .groupby(["cost_group", "item_name"], dropna=False, as_index=False)
        .agg(
            source_amount_per_batch=("amount_per_batch", "sum"),
            source_unit_examples=("amount_unit_per_batch", lambda s: "; ".join(sorted(set(map(str, s.dropna()))))),
            adjusted_cost_per_batch_usd=("adjusted_cost_per_batch_usd", "sum"),
            row_count=("cost_id", "count"),
        )
        .sort_values("adjusted_cost_per_batch_usd", ascending=False)
    )

    cost_summary_by_section = (
        cost_analysis
        .groupby(["process_section", "cost_group"], dropna=False, as_index=False)
        .agg(
            adjusted_cost_per_batch_usd=("adjusted_cost_per_batch_usd", "sum"),
            row_count=("cost_id", "count"),
        )
        .sort_values("adjusted_cost_per_batch_usd", ascending=False)
    )

    display(cost_summary_by_group)
    display(cost_summary_by_item.head(30))
    display(cost_summary_by_section.head(30))

    # ---------------------------------------------------------------------
    # Final selected LCA mappings: hardcoded activities and supply routes
    # ---------------------------------------------------------------------
    # ecoinvent version:
    #   Updated from ecoinvent 3.10.1 to ecoinvent 3.12.
    #
    # The mapping table is intentionally self-contained.
    # Electricity has two supply routes using the same SuperPro quantity:
    #   1) UK grid BAU
    #   2) GB onshore wind renewable electricity
    #
    # All activity codes below were resolved in the local Brightway
    # ecoinvent-3.12-cutoff database by unique exact metadata match.

    BACKGROUND_DB = settings["brightway"]["background_database"]

    SELECTED_ACTIVITY_SPECS = {
        "wfi_baseline_supply": {
            "lca_flow_key": "wfi",
            "supply_pathway": "baseline_supply",
            "database": BACKGROUND_DB,
            "activity_code": "e6ccca2b1b884187e432d7098714fc0b",
            "activity_name": "market for water, ultrapure",
            "reference_product": "water, ultrapure",
            "location": "RER",
            "unit": "kilogram",
            "selection_rationale": (
                "Closest available proxy to pharmaceutical WFI/purified water; "
                "use market rather than production where possible."
            ),
        },

        "electricity_grid_uk_bau": {
            "lca_flow_key": "electricity",
            "supply_pathway": "electricity_grid_uk_bau",
            "database": BACKGROUND_DB,
            "activity_code": "24a01b36c7f5264a3fa9bf91fc216ddf",
            "activity_name": "market for electricity, medium voltage",
            "reference_product": "electricity, medium voltage",
            "location": "GB",
            "unit": "kilowatt hour",
            "selection_rationale": (
                "Business-as-usual UK grid electricity route. SuperPro Std Power remains "
                "the electricity quantity source; only the background electricity activity changes."
            ),
        },

        "electricity_renewable_onshore_wind": {
            "lca_flow_key": "electricity",
            "supply_pathway": "electricity_renewable_onshore_wind",
            "database": BACKGROUND_DB,
            "activity_code": "74cb3a668cea4668548fd992b0074da2",
            "activity_name": "electricity production, wind, 1-3MW turbine, onshore",
            "reference_product": "electricity, high voltage",
            "location": "GB",
            "unit": "kilowatt hour",
            "selection_rationale": (
                "Renewable onshore wind electricity route used to align with the renewable-electricity "
                "scenario assumption. SuperPro Std Power remains the electricity quantity source; "
                "only the background electricity activity changes."
            ),
        },

        "steam_baseline_supply": {
            "lca_flow_key": "steam",
            "supply_pathway": "baseline_supply",
            "database": BACKGROUND_DB,
            "activity_code": "789db97030a6f74a8c789da43db8e20e",
            "activity_name": "market for heat, from steam, in chemical industry",
            "reference_product": "heat, from steam, in chemical industry",
            "location": "RER",
            "unit": "megajoule",
            "selection_rationale": (
                "Matches steam as useful heat in chemical industry and uses MJ, "
                "consistent with conversion from MT steam to heat duty."
            ),
        },

        "chilled_water_baseline_supply": {
            "lca_flow_key": "chilled_water",
            "supply_pathway": "baseline_supply",
            "database": BACKGROUND_DB,
            "activity_code": "b45c799aa2ddf9ba8912e0f138c2d756",
            "activity_name": "market for cooling energy, at -15 °C",
            "reference_product": "cooling energy, at -15 °C",
            "location": "GLO",
            "unit": "megajoule",
            "selection_rationale": (
                "Chilled-water utility represented as cooling duty; -15 °C cooling-energy "
                "market is the closest chilled-service candidate."
            ),
        },

        "cooling_water_baseline_supply": {
            "lca_flow_key": "cooling_water",
            "supply_pathway": "baseline_supply",
            "database": BACKGROUND_DB,
            "activity_code": "ed1f0cecabd4822e91039f8caa3fe382",
            "activity_name": "market for cooling energy",
            "reference_product": "cooling energy",
            "location": "GLO",
            "unit": "megajoule",
            "selection_rationale": (
                "Generic cooling-energy market used as a transparent proxy for cooling-water duty "
                "because no clean cooling-water supply market is selected."
            ),
        },
    }

    selected_lca_mappings = (
        pd.DataFrame.from_dict(SELECTED_ACTIVITY_SPECS, orient="index")
        .rename_axis("mapping_id")
        .reset_index()
    )

    selected_lca_mappings["selected_for_lca"] = True

    # ---------------------------------------------------------------------
    # Validate selected activity units match converted physical inventory units
    # ---------------------------------------------------------------------

    unit_check = physical_inventory[["lca_flow_key", "activity_unit"]].merge(
        selected_lca_mappings[
            [
                "mapping_id",
                "lca_flow_key",
                "supply_pathway",
                "unit",
                "activity_name",
                "reference_product",
                "location",
                "database",
                "activity_code",
            ]
        ],
        on="lca_flow_key",
        how="left",
    )

    unit_check["mapping_present"] = unit_check["unit"].notna()
    unit_check["unit_match"] = unit_check["activity_unit"].eq(unit_check["unit"])

    if not unit_check["mapping_present"].all():
        display(unit_check)
        raise ValueError("At least one physical-inventory flow is missing a selected activity mapping.")

    if not unit_check["unit_match"].all():
        display(unit_check)
        raise ValueError("At least one selected activity unit does not match the converted physical-inventory activity unit.")

    display(
        selected_lca_mappings[
            [
                "mapping_id",
                "lca_flow_key",
                "supply_pathway",
                "activity_name",
                "reference_product",
                "location",
                "unit",
                "database",
                "activity_code",
                "selection_rationale",
            ]
        ]
    )

    display(unit_check)

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

    for label, method in LCIA_METHODS.items():
        print(f"{label}: {method}")

    # ---------------------------------------------------------------------
    # Brightway setup and method validation
    # ---------------------------------------------------------------------
    try:
        import bw2data as bd
        import bw2calc as bc
    except ImportError as exc:
        raise ImportError(
            "Brightway is not installed in this environment. Run this notebook in your Brightway environment."
        ) from exc

    bd.projects.set_current(BW_PROJECT)

    print("LCIA methods found:")
    for label, method in LCIA_METHODS.items():
        print(f"  {label}: {method}")

    def find_activity_by_exact_metadata(row):
        """Find a unique Brightway activity by exact reviewed metadata."""
        db_name = str(row["database"]).strip()
        expected = {
            "name": str(row["activity_name"]).strip(),
            "reference product": str(row["reference_product"]).strip(),
            "location": str(row["location"]).strip(),
            "unit": str(row["unit"]).strip(),
        }

        matches = []
        for act in bd.Database(db_name):
            if all(str(act.get(k, "")).strip() == v for k, v in expected.items()):
                matches.append(act)

        if len(matches) != 1:
            preview = [
                {
                    "name": m.get("name"),
                    "reference_product": m.get("reference product"),
                    "location": m.get("location"),
                    "unit": m.get("unit"),
                    "code": m.key[1],
                }
                for m in matches[:10]
            ]
            raise LookupError(
                f"Expected exactly one activity for {row['mapping_id']} by exact metadata, found {len(matches)}.\n"
                f"Expected metadata: {expected}\n"
                f"Preview matches: {preview}"
            )

        return matches[0]


    def get_activity_from_mapping_row(row):
        """Return the selected Brightway activity from hardcoded code or exact metadata."""
        db_name = str(row["database"]).strip()
        activity_code = str(row["activity_code"]).strip()

        if activity_code == "LOOKUP_BY_EXACT_METADATA":
            activity = find_activity_by_exact_metadata(row)
        else:
            if activity_code in {"", "None", "nan"}:
                raise ValueError(f"Missing activity_code for selected LCA mapping: {row['mapping_id']}")

            try:
                activity = bd.get_activity((db_name, activity_code))
            except Exception as exc:
                # Reimporting an unchanged ecoinvent release can alter Brightway
                # internal codes. Resolve only an exact, unique reviewed dataset;
                # the resolved code is exported in the activity-mapping audit.
                matches = [
                    candidate for candidate in bd.Database(db_name)
                    if candidate.get("name") == str(row["activity_name"])
                    and candidate.get("reference product") == str(row["reference_product"])
                    and candidate.get("location") == str(row["location"])
                    and candidate.get("unit") == str(row["unit"])
                ]
                if len(matches) != 1:
                    raise LookupError(
                        f"Code {activity_code!r} is unavailable for {row['mapping_id']}; "
                        f"exact reviewed metadata resolved to {len(matches)} activities."
                    ) from exc
                activity = matches[0]

        # Strict metadata check.
        expected_metadata = {
            "activity_name": row["activity_name"],
            "reference_product": row["reference_product"],
            "location": row["location"],
            "unit": row["unit"],
        }

        resolved_metadata = {
            "activity_name": activity.get("name"),
            "reference_product": activity.get("reference product"),
            "location": activity.get("location"),
            "unit": activity.get("unit"),
        }

        mismatches = []
        for field, expected_value in expected_metadata.items():
            resolved_value = resolved_metadata[field]
            if str(expected_value) != str(resolved_value):
                mismatches.append((field, expected_value, resolved_value))

        if mismatches:
            mismatch_text = "; ".join(
                f"{field}: expected {expected_value!r}, resolved {resolved_value!r}"
                for field, expected_value, resolved_value in mismatches
            )
            raise ValueError(
                f"Activity resolved for {row['mapping_id']}, but metadata does not match the reviewed row. "
                f"Mismatches: {mismatch_text}"
            )

        return activity


    def activity_metadata(activity):
        """Extract standard Brightway activity metadata for exported mapping tables."""
        return {
            "resolved_activity_name": activity.get("name"),
            "resolved_reference_product": activity.get("reference product"),
            "resolved_location": activity.get("location"),
            "resolved_unit": activity.get("unit"),
            "resolved_database": activity.key[0],
            "resolved_activity_code": activity.key[1],
        }


    def score_activity(activity, method, demand_amount=1.0):
        """Calculate the LCIA score for one Brightway activity and method."""
        lca = bc.LCA({activity: demand_amount}, method)
        lca.lci()
        lca.lcia()
        return float(lca.score)


    # ---------------------------------------------------------------------
    # Activity-level LCIA factors
    # ---------------------------------------------------------------------
    impact_factor_rows = []

    for _, mapping_row in selected_lca_mappings.iterrows():
        flow_key = mapping_row["lca_flow_key"]
        activity = get_activity_from_mapping_row(mapping_row)
        meta = activity_metadata(activity)

        for impact_category, method in LCIA_METHODS.items():
            score_per_activity_unit = score_activity(activity, method, demand_amount=1.0)

            resolved_unit = activity.get("unit")
            expected_unit = mapping_row["unit"]
            if str(resolved_unit) != str(expected_unit):
                raise ValueError(
                    f"Resolved activity for {mapping_row['mapping_id']} has unit {resolved_unit}, "
                    f"but selected mapping expects {expected_unit}."
                )

            impact_factor_rows.append({
                "mapping_id": mapping_row["mapping_id"],
                "lca_flow_key": flow_key,
                "supply_pathway": mapping_row["supply_pathway"],
                "impact_category": impact_category,
                "impact_per_activity_unit": score_per_activity_unit,
                "activity_demand_amount_for_factor": 1.0,
                "method": method,
                "selection_rationale": mapping_row["selection_rationale"],
                **meta,
            })

    impact_factor_table = pd.DataFrame(impact_factor_rows)

    display(impact_factor_table)


    # ---------------------------------------------------------------------
    # Detailed LCA results per batch
    # ---------------------------------------------------------------------
    main_detailed_table = physical_inventory.merge(
        impact_factor_table,
        on="lca_flow_key",
        how="left",
        validate="one_to_many",
    )

    main_detailed_table["impact_per_batch"] = (
        main_detailed_table["activity_demand_amount_per_batch"].astype(float)
        * main_detailed_table["impact_per_activity_unit"].astype(float)
    )

    summary_by_flow = (
        main_detailed_table
        .groupby(["lca_flow_key", "inventory_group", "impact_category"], dropna=False, as_index=False)
        .agg(
            source_quantity=("source_quantity", "first"),
            source_unit=("source_unit", "first"),
            activity_demand_amount_per_batch=("activity_demand_amount_per_batch", "first"),
            activity_unit=("activity_unit", "first"),
            impact_per_batch=("impact_per_batch", "sum"),
        )
        .sort_values(["impact_category", "impact_per_batch"], ascending=[True, False])
    )

    summary_by_category = (
        main_detailed_table
        .groupby("impact_category", dropna=False, as_index=False)
        .agg(impact_per_batch=("impact_per_batch", "sum"))
        .sort_values("impact_category")
    )

    display(main_detailed_table)
    display(summary_by_flow)
    display(summary_by_category)

    # ---------------------------------------------------------------------
    # Validation checks
    # ---------------------------------------------------------------------
    validation_checks = []

    def add_check(check_name, value, expected, tolerance, units, passed, note):
        validation_checks.append({
            "check_name": check_name,
            "value": value,
            "expected": expected,
            "absolute_difference": abs(value - expected) if pd.notna(value) and pd.notna(expected) else np.nan,
            "tolerance": tolerance,
            "units": units,
            "passed": bool(passed),
            "note": note,
        })

    add_check(
        "WFI LCA inventory equals overall component-balance WFI input",
        float(physical_inventory.loc[physical_inventory["lca_flow_key"].eq("wfi"), "source_quantity"].iloc[0]),
        overall_wfi_kg,
        1e-9,
        "kg/batch",
        math.isclose(
            float(physical_inventory.loc[physical_inventory["lca_flow_key"].eq("wfi"), "source_quantity"].iloc[0]),
            overall_wfi_kg,
            rel_tol=0,
            abs_tol=1e-9,
        ),
        "Confirms WFI LCA inventory uses the SuperPro Overall Component Balance INPUT value.",
    )

    add_check(
        "WFI stream-detail plus residual reconciles to overall input",
        stream_detail_wfi_kg + residual_wfi_kg,
        overall_wfi_kg,
        1e-6,
        "kg/batch",
        math.isclose(
            stream_detail_wfi_kg + residual_wfi_kg,
            overall_wfi_kg,
            rel_tol=0,
            abs_tol=1e-6,
        ),
        "Confirms explicit Stream Details WFI plus residual reconciles to the Overall Component Balance WFI input.",
    )

    for key in ["electricity", "steam", "chilled_water", "cooling_water"]:
        source_total = float(utility_quantity_by_item.loc[utility_quantity_by_item["lca_flow_key"].eq(key), "source_amount_per_batch"].sum())
        inventory_total = float(physical_inventory.loc[physical_inventory["lca_flow_key"].eq(key), "source_quantity"].sum())
        add_check(
            f"{key} utility source quantity retained",
            inventory_total,
            source_total,
            1e-9,
            str(utility_quantity_by_item.loc[utility_quantity_by_item["lca_flow_key"].eq(key), "amount_unit_per_batch"].iloc[0]),
            math.isclose(inventory_total, source_total, rel_tol=0, abs_tol=1e-9),
            "Quantity comes from ICR itemised utility rows only.",
        )

    brine_in_inventory = "cacl2_brine" in set(physical_inventory["lca_flow_key"])
    add_check(
        "CaCl2 brine excluded from LCA inventory",
        0 if not brine_in_inventory else 1,
        0,
        0,
        "boolean flag",
        not brine_in_inventory,
        "Brine is present in ICR utility rows but deliberately excluded from this final notebook.",
    )

    validation_checks = pd.DataFrame(validation_checks)

    display(validation_checks)

    if not validation_checks["passed"].all():
        raise ValueError("At least one validation check failed. Review validation_checks before using outputs.")

    manufacturing_main_detailed_table = main_detailed_table.copy()

    manufacturing_main_detailed_table.insert(0, "source_module", "manufacturing_footprint")
    manufacturing_main_detailed_table.insert(1, "stream_category", manufacturing_main_detailed_table["inventory_group"])
    manufacturing_main_detailed_table.insert(2, "functional_unit", "per_batch")

    # Add route-style fields so the table looks and behaves like wastewater/plastics.
    manufacturing_main_detailed_table["route"] = manufacturing_main_detailed_table["supply_pathway"]
    manufacturing_main_detailed_table["route_kind"] = np.where(
        manufacturing_main_detailed_table["lca_flow_key"].eq("electricity"),
        "electricity_supply_choice",
        "fixed_supply",
    )
    manufacturing_main_detailed_table["life_cycle_stage"] = np.where(
        manufacturing_main_detailed_table["inventory_group"].eq("utility"),
        "manufacturing_utility_supply",
        "manufacturing_process_water_supply",
    )
    manufacturing_main_detailed_table["is_eol_treatment"] = False
    manufacturing_main_detailed_table["is_credit"] = False
    manufacturing_main_detailed_table["secondary_streams_embedded_in_activity"] = False
    manufacturing_main_detailed_table["model_scope"] = "manufacturing_inputs_and_utilities_only"
    manufacturing_main_detailed_table["export_date"] = RUN_DATE


    # 2) Route factor table: one optimisation factor per flow, supply pathway and impact category.

    manufacturing_route_factor_table = (
        manufacturing_main_detailed_table
        .groupby(
            [
                "source_module",
                "stream_category",
                "lca_flow_key",
                "supply_pathway",
                "route",
                "route_kind",
                "life_cycle_stage",
                "impact_category",
                "activity_unit",
                "resolved_activity_name",
                "resolved_reference_product",
                "resolved_location",
                "resolved_unit",
                "resolved_database",
                "resolved_activity_code",
                "selection_rationale",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            impact_per_activity_unit=("impact_per_activity_unit", "first"),
            impact_factor_unit=("impact_category", "first"),
            represented_quantity_per_batch=("activity_demand_amount_per_batch", "sum"),
            represented_impact_per_batch=("impact_per_batch", "sum"),
            contributing_inventory_rows=("lca_flow_key", "count"),
        )
    )

    manufacturing_route_factor_table["impact_factor_unit"] = (
        manufacturing_route_factor_table["impact_category"]
        + " per "
        + manufacturing_route_factor_table["activity_unit"].astype(str)
    )

    manufacturing_route_factor_table["route_factor_status"] = np.where(
        manufacturing_route_factor_table["route_kind"].eq("electricity_supply_choice"),
        "alternative_supply_route",
        "single_consistent_factor",
    )
    manufacturing_route_factor_table["route_factor_scenario"] = manufacturing_route_factor_table["supply_pathway"]
    manufacturing_route_factor_table["quantity_unit"] = manufacturing_route_factor_table["activity_unit"]
    manufacturing_route_factor_table["is_production"] = True
    manufacturing_route_factor_table["is_eol_treatment"] = False
    manufacturing_route_factor_table["is_credit"] = False
    manufacturing_route_factor_table["secondary_streams_embedded_in_activity"] = False
    manufacturing_route_factor_table["mapping_type"] = "fixed_exact_metadata_reviewed"
    manufacturing_route_factor_table["activity_demand_amount"] = 1.0
    manufacturing_route_factor_table["demand_orientation_note"] = (
        "Positive demand of +1 activity unit represents supply of the manufacturing input/utility. "
        "Electricity supply pathways are mutually exclusive alternatives."
    )
    manufacturing_route_factor_table["min_impact_per_activity_unit"] = manufacturing_route_factor_table["impact_per_activity_unit"]
    manufacturing_route_factor_table["max_impact_per_activity_unit"] = manufacturing_route_factor_table["impact_per_activity_unit"]
    manufacturing_route_factor_table["impact_factor_variability_abs"] = 0.0
    manufacturing_route_factor_table["export_date"] = RUN_DATE

    manufacturing_route_factor_table = manufacturing_route_factor_table.rename(
        columns={
            "lca_flow_key": "manufacturing_flow",
            "impact_per_activity_unit": "impact_per_unit_for_optimisation",
            "resolved_activity_name": "activity_name",
            "resolved_reference_product": "reference_product",
            "resolved_location": "location",
            "resolved_unit": "unit",
            "resolved_database": "database",
            "resolved_activity_code": "code",
            "selection_rationale": "assumption_note",
        }
    )


    # 3) Secondary outputs: traceability and reconciliation only.

    secondary_output_rows = []

    secondary_output_rows.append({
        "source_module": "manufacturing_footprint",
        "parent_stream_category": "process_water",
        "parent_route": "wfi",
        "parent_eol_sink": None,
        "secondary_stream_category": "wfi_traceability",
        "secondary_stream_name": "explicit_stream_detail_wfi_component_sum",
        "stream_direction": "input_traceability",
        "quantity_per_batch": stream_detail_wfi_kg,
        "quantity_unit": "kg",
        "embedded_in_lca_activity": False,
        "export_to_wastewater_pool": False,
        "routable_in_master": False,
        "source_of_quantity": "SR Stream Details input_stream_component rows where component == WFI",
        "assumption_note": "Traceability only. Not added to LCA because WFI LCA quantity uses the Overall Component Balance input.",
    })

    secondary_output_rows.append({
        "source_module": "manufacturing_footprint",
        "parent_stream_category": "process_water",
        "parent_route": "wfi",
        "parent_eol_sink": None,
        "secondary_stream_category": "wfi_reconciliation",
        "secondary_stream_name": "overall_minus_stream_detail_wfi_residual",
        "stream_direction": "input_reconciliation",
        "quantity_per_batch": residual_wfi_kg,
        "quantity_unit": "kg",
        "embedded_in_lca_activity": False,
        "export_to_wastewater_pool": False,
        "routable_in_master": False,
        "source_of_quantity": "SR overall_minus_stream_input_residual row where component == WFI",
        "assumption_note": "Validation only. Shows WFI embedded in aqueous inputs that is not visible in explicit Stream Details component rows.",
    })

    secondary_output_rows.append({
        "source_module": "manufacturing_footprint",
        "parent_stream_category": "process_water",
        "parent_route": "wfi",
        "parent_eol_sink": None,
        "secondary_stream_category": "wfi_lca_source",
        "secondary_stream_name": "overall_component_balance_wfi_input",
        "stream_direction": "input_inventory_source",
        "quantity_per_batch": overall_wfi_kg,
        "quantity_unit": "kg",
        "embedded_in_lca_activity": True,
        "export_to_wastewater_pool": False,
        "routable_in_master": False,
        "source_of_quantity": "SR Overall Component Balance INPUT row where component == WFI",
        "assumption_note": "This is the WFI quantity used in the manufacturing LCA physical inventory.",
    })

    for _, row in utility_quantity_by_item.iterrows():
        secondary_output_rows.append({
            "source_module": "manufacturing_footprint",
            "parent_stream_category": "utility",
            "parent_route": row["lca_flow_key"],
            "parent_eol_sink": None,
            "secondary_stream_category": "utility_source_quantity",
            "secondary_stream_name": row["item_name"],
            "stream_direction": "input_inventory_source",
            "quantity_per_batch": row["source_amount_per_batch"],
            "quantity_unit": row["amount_unit_per_batch"],
            "embedded_in_lca_activity": True,
            "export_to_wastewater_pool": False,
            "routable_in_master": False,
            "source_of_quantity": "ICR itemised utility cost rows",
            "assumption_note": "Utility source quantity used for manufacturing LCA after unit conversion where required.",
        })

    secondary_outputs = pd.DataFrame(secondary_output_rows)
    secondary_outputs["export_date"] = RUN_DATE


    # 4) Compact method/mapping appendix for traceability.

    method_mapping_reference = selected_lca_mappings.merge(
        impact_factor_table,
        on=["mapping_id", "lca_flow_key", "supply_pathway"],
        how="left",
        validate="one_to_many",
    )
    method_mapping_reference["export_date"] = RUN_DATE
    return {
        "main_detailed_table": manufacturing_main_detailed_table,
        "route_factor_table": manufacturing_route_factor_table,
        "secondary_outputs": secondary_outputs,
        "physical_inventory": physical_inventory,
        "cost_analysis": cost_analysis,
        "method_mapping_reference": method_mapping_reference,
    }
