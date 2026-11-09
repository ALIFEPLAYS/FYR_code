"""
The SuperPro biomass value was interpreted as biomass solids rather than total wet residual-stream mass because it was
 obtained from the overall component balance. In the absence of measured solids and volatile-solids content, the biomass
  was provisionally treated as dry organic solids, with the assumption tested through sensitivity analysis.
"""
from __future__ import annotations

from .human_health_daly import add_human_health_method
from .ecosystem_quality import add_ecosystem_method
from .lcia_methods import core_lcia_methods


def extract_biomass_inventory(superpro_inventory, settings):
    """Extract the residual biomass output from the processed SuperPro table."""
    import pandas as pd

    config = settings.get("biomass", {})
    stream_name = str(config.get("stream_name", "Biomass")).strip()
    source_table_contains = str(
        config.get("source_table_contains", "OVERALL COMPONENT BALANCE")
    ).strip()

    required = {
        "inventory_id",
        "source_table",
        "destination",
        "flow_name_superpro",
        "component",
        "amount_per_batch",
        "unit_per_batch",
        "amount_per_year",
        "unit_per_year",
    }
    missing = required.difference(superpro_inventory.columns)
    if missing:
        raise ValueError(
            "SuperPro inventory is missing columns required for the biomass module: "
            + ", ".join(sorted(missing))
        )

    source_mask = superpro_inventory["source_table"].astype(str).str.contains(
        source_table_contains,
        case=False,
        na=False,
        regex=False,
    )
    output_mask = superpro_inventory["destination"].astype(str).str.upper().eq("OUTPUT")
    biomass_mask = (
        superpro_inventory["component"].astype(str).str.strip().str.casefold().eq(stream_name.casefold())
        | superpro_inventory["flow_name_superpro"].astype(str).str.strip().str.casefold().eq(stream_name.casefold())
    )

    biomass = superpro_inventory[source_mask & output_mask & biomass_mask].copy()
    if len(biomass) != 1:
        raise ValueError(
            f"Expected one overall SuperPro biomass output row for {stream_name!r}; "
            f"found {len(biomass)}."
        )

    biomass["amount_per_batch"] = pd.to_numeric(
        biomass["amount_per_batch"], errors="coerce"
    )
    biomass["amount_per_year"] = pd.to_numeric(
        biomass["amount_per_year"], errors="coerce"
    )
    if biomass["amount_per_batch"].isna().any():
        raise ValueError("Biomass amount_per_batch is missing or non-numeric.")
    if (biomass["amount_per_batch"] <= 0).any():
        raise ValueError("Biomass amount_per_batch must be positive.")

    unit = str(biomass.iloc[0]["unit_per_batch"]).strip().lower()
    if unit not in {"kg/batch", "kg per batch"}:
        raise ValueError(
            "The biomass module expects the overall biomass output in kg/batch; "
            f"found {biomass.iloc[0]['unit_per_batch']!r}."
        )

    biomass = biomass.rename(
        columns={
            "amount_per_batch": "biomass_kg_per_batch",
            "amount_per_year": "biomass_kg_per_year",
        }
    )
    biomass["source_module"] = "biomass"
    biomass["stream_category"] = "residual_biomass"
    biomass["material_or_service"] = "residual_biomass"
    return biomass.reset_index(drop=True)


def run_biomass_lca(superpro_inventory, settings):
    """Model residual biomass valorisation through AD, biomethane and digestate recovery."""
    import numpy as np
    import pandas as pd
    import bw2calc as bc
    import bw2data as bd

    biomass_inventory = extract_biomass_inventory(superpro_inventory, settings)
    config = settings.get("biomass", {})
    pathway = str(config.get("pathway", "ad_biomethane_digestate"))
    pathway_label = str(
        config.get(
            "pathway_label",
            "Anaerobic digestion with biomethane and digestate recovery",
        )
    )

    biomass_kg_per_batch = float(biomass_inventory.iloc[0]["biomass_kg_per_batch"])
    biomass_kg_per_year = float(biomass_inventory.iloc[0]["biomass_kg_per_year"])

    _validate_biomass_settings(config)

    project = settings["brightway"]["project"]
    background_db = settings["brightway"]["background_database"]
    biosphere_db = settings["brightway"]["biosphere_database"]
    bd.projects.set_current(project)

    if background_db not in bd.databases:
        raise ValueError(
            f"Missing Brightway background database {background_db!r}."
        )
    if biosphere_db not in bd.databases:
        raise ValueError(
            f"Missing Brightway biosphere database {biosphere_db!r}."
        )

    methods = core_lcia_methods()
    methods = add_human_health_method(methods)
    methods = add_ecosystem_method(methods)
    missing_methods = [method for method in methods.values() if method not in bd.methods]
    if missing_methods:
        raise ValueError(
            "The biomass module could not find all configured LCIA methods in the "
            f"current Brightway project: {missing_methods}"
        )

    activity_specs = _reviewed_activity_specs(
        background_db=background_db,
        configured_specs=config.get("activities", {}),
    )
    required_activity_keys = (
        "anaerobic_digestion",
        "upgrading_electricity",
        "avoided_natural_gas",
        "avoided_nitrogen_fertiliser",
        "avoided_phosphate_fertiliser",
        "avoided_potassium_fertiliser",
    )

    activities = {
        key: _resolve_activity(bd, background_db, activity_specs[key])
        for key in required_activity_keys
    }

    ad_reference_product = activities["anaerobic_digestion"].get("reference product")
    if not _is_waste_reference_product(ad_reference_product):
        raise ValueError(
            "The configured anaerobic-digestion activity does not have a waste "
            f"reference product: {ad_reference_product!r}. The -1 treatment-demand "
            "orientation is valid only for an activity that accepts residual biomass "
            "or biowaste as its reference product."
        )

    # Foreground biomass yields are explicit study assumptions. Missing values
    # must stop the run instead of changing the model's physical basis silently.
    biogas_yield = float(config["biogas_yield_m3_per_kg_biomass"])
    digestate_yield = float(config["digestate_yield_kg_per_kg_biomass"])
    yield_source = "declared foreground JSON assumption"
    digestate_source = "declared foreground JSON assumption"

    recovery = float(config["upgrading_recovery_fraction"])
    raw_biogas_ncv = float(config["raw_biogas_ncv_mj_per_m3"])
    biomethane_ncv = float(config["biomethane_ncv_mj_per_m3"])
    biomethane_m3_per_kg = biogas_yield * raw_biogas_ncv / biomethane_ncv * recovery
    upgrading_kwh_per_m3 = float(
        config["upgrading_electricity_kwh_per_m3_biomethane"]
    )
    upgrading_kwh_per_kg = biomethane_m3_per_kg * upgrading_kwh_per_m3

    digestate_recovery = float(config["digestate_recovery_fraction"])
    digestate_kg_per_kg = digestate_yield * digestate_recovery
    nutrients = config["digestate_available_nutrients_kg_per_tonne"]
    substitution = config["fertiliser_substitution_efficiency"]

    available_nutrients = {
        "nitrogen_as_n": (
            digestate_kg_per_kg
            / 1000.0
            * float(nutrients["N"])
            * float(substitution["N"])
        ),
        "phosphorus_as_p2o5": (
            digestate_kg_per_kg
            / 1000.0
            * float(nutrients["P2O5"])
            * float(substitution["P2O5"])
        ),
        "potassium_as_k2o": (
            digestate_kg_per_kg
            / 1000.0
            * float(nutrients["K2O"])
            * float(substitution["K2O"])
        ),
    }

    ammonium_nitrate_n_fraction = float(
        config.get("ammonium_nitrate_n_mass_fraction", 0.35)
    )
    avoided_fertiliser_products = {
        "ammonium_nitrate": (
            available_nutrients["nitrogen_as_n"]
            / ammonium_nitrate_n_fraction
        ),
        "phosphorus_fertiliser_as_p2o5": (
            available_nutrients["phosphorus_as_p2o5"]
        ),
        "potassium_fertiliser_as_k2o": (
            available_nutrients["potassium_as_k2o"]
        ),
    }

    stage_specs = [
        {
            "life_cycle_stage": "anaerobic_digestion",
            "stage_type": "treatment",
            "activity_key": "anaerobic_digestion",
            "activity_demand_per_kg_biomass": -1.0,
            "is_credit": False,
            "is_eol_treatment": True,
            "quantity_basis": "1 kg residual biomass accepted for anaerobic digestion",
            "demand_orientation_note": (
                "The ecoinvent waste-treatment activity is demanded at -1 activity unit "
                "per kg of physical residual biomass treated."
            ),
        },
        {
            "life_cycle_stage": "biogas_upgrading",
            "stage_type": "processing",
            "activity_key": "upgrading_electricity",
            "activity_demand_per_kg_biomass": upgrading_kwh_per_kg,
            "is_credit": False,
            "is_eol_treatment": False,
            "quantity_basis": "electricity demand for biomethane upgrading",
            "demand_orientation_note": "Positive market demand for upgrading electricity.",
        },
        {
            "life_cycle_stage": "avoided_natural_gas",
            "stage_type": "credit",
            "activity_key": "avoided_natural_gas",
            "activity_demand_per_kg_biomass": -biomethane_m3_per_kg,
            "is_credit": True,
            "is_eol_treatment": False,
            "quantity_basis": "biomethane supplied to the gas grid, displacing fossil natural gas",
            "demand_orientation_note": "Negative market demand represents avoided fossil natural-gas supply.",
        },
        {
            "life_cycle_stage": "avoided_nitrogen_fertiliser",
            "stage_type": "credit",
            "activity_key": "avoided_nitrogen_fertiliser",
            "activity_demand_per_kg_biomass": -avoided_fertiliser_products["ammonium_nitrate"],
            "is_credit": True,
            "is_eol_treatment": False,
            "quantity_basis": (
                "ammonium nitrate displaced; available kg N divided by "
                f"{ammonium_nitrate_n_fraction:.3f} kg N per kg ammonium nitrate"
            ),
            "demand_orientation_note": "Negative market demand represents avoided mineral-fertiliser supply.",
        },
        {
            "life_cycle_stage": "avoided_phosphate_fertiliser",
            "stage_type": "credit",
            "activity_key": "avoided_phosphate_fertiliser",
            "activity_demand_per_kg_biomass": -avoided_fertiliser_products["phosphorus_fertiliser_as_p2o5"],
            "is_credit": True,
            "is_eol_treatment": False,
            "quantity_basis": "available digestate phosphorus expressed directly as kg P2O5",
            "demand_orientation_note": "Negative market demand represents avoided mineral-fertiliser supply.",
        },
        {
            "life_cycle_stage": "avoided_potassium_fertiliser",
            "stage_type": "credit",
            "activity_key": "avoided_potassium_fertiliser",
            "activity_demand_per_kg_biomass": -avoided_fertiliser_products["potassium_fertiliser_as_k2o"],
            "is_credit": True,
            "is_eol_treatment": False,
            "quantity_basis": "available digestate potassium expressed directly as kg K2O",
            "demand_orientation_note": "Negative market demand represents avoided mineral-fertiliser supply.",
        },
    ]

    detailed_rows = []
    mapping_rows = []
    for key, activity in activities.items():
        mapping_rows.append(_activity_metadata(key, activity, activity_specs[key]))

    for stage in stage_specs:
        activity = activities[stage["activity_key"]]
        demand = float(stage["activity_demand_per_kg_biomass"])
        for impact_category, method in methods.items():
            lca = bc.LCA({activity: demand}, method)
            lca.lci()
            lca.lcia()
            factor = float(lca.score)
            metadata = _activity_metadata(
                stage["activity_key"],
                activity,
                activity_specs[stage["activity_key"]],
            )
            detailed_rows.append({
                "inventory_id": biomass_inventory.iloc[0]["inventory_id"],
                "source_module": "biomass",
                "stream_category": "residual_biomass",
                "material_or_service": "residual_biomass",
                "pathway": pathway,
                "route": pathway,
                "pathway_label": pathway_label,
                "eol_pathway": pathway,
                "eol_sink": "anaerobic_digestion_biomethane_and_digestate",
                "life_cycle_stage": stage["life_cycle_stage"],
                "stage_type": stage["stage_type"],
                "impact_category": impact_category,
                "impact_per_kg_biomass": factor,
                "impact_per_batch": factor * biomass_kg_per_batch,
                "impact_factor_unit": f"impact per kg residual biomass ({impact_category})",
                "quantity_unit": "kg",
                "biomass_kg_per_batch": biomass_kg_per_batch,
                "biomass_kg_per_year": biomass_kg_per_year,
                "represented_mass_kg_per_batch": biomass_kg_per_batch,
                "activity_demand_per_kg_biomass": demand,
                "activity_demand_basis": stage["quantity_basis"],
                "is_production": False,
                "is_eol_treatment": stage["is_eol_treatment"],
                "is_credit": stage["is_credit"],
                "secondary_streams_embedded_in_activity": False,
                "biogas_yield_m3_per_kg_biomass": biogas_yield,
                "biomethane_m3_per_kg_biomass": biomethane_m3_per_kg,
                "digestate_kg_per_kg_biomass": digestate_kg_per_kg,
                "available_n_kg_per_kg_biomass": available_nutrients["nitrogen_as_n"],
                "available_p2o5_kg_per_kg_biomass": available_nutrients["phosphorus_as_p2o5"],
                "available_k2o_kg_per_kg_biomass": available_nutrients["potassium_as_k2o"],
                "ammonium_nitrate_n_mass_fraction": ammonium_nitrate_n_fraction,
                "yield_source": yield_source,
                "digestate_yield_source": digestate_source,
                "assumption_note": (
                    "Foreground avoided-burden route: AD and upgrading burdens minus fossil natural-gas "
                    "and mineral-fertiliser supply credits. Electricity and heat credits are not included. "
                    + stage["demand_orientation_note"]
                ),
                **metadata,
            })

    main_detailed = pd.DataFrame(detailed_rows)
    route_group_cols = [
        "source_module",
        "stream_category",
        "material_or_service",
        "pathway",
        "route",
        "pathway_label",
        "eol_pathway",
        "eol_sink",
        "life_cycle_stage",
        "stage_type",
        "impact_category",
        "impact_factor_unit",
        "quantity_unit",
        "is_production",
        "is_eol_treatment",
        "is_credit",
        "activity_name",
        "reference_product",
        "location",
        "unit",
        "database",
        "code",
        "mapping_type",
        "activity_demand_per_kg_biomass",
        "activity_demand_basis",
        "assumption_note",
    ]
    route_factor_table = (
        main_detailed.groupby(route_group_cols, dropna=False, as_index=False)
        .agg(
            impact_per_kg_for_optimisation=("impact_per_kg_biomass", "mean"),
            represented_mass_kg_per_batch=("represented_mass_kg_per_batch", "max"),
            represented_impact_per_batch=("impact_per_batch", "sum"),
            contributing_inventory_rows=("inventory_id", "nunique"),
        )
    )
    route_factor_table["route_factor_status"] = "single_consistent_factor"

    biomethane_m3_per_batch = biomethane_m3_per_kg * biomass_kg_per_batch
    digestate_kg_per_batch = digestate_kg_per_kg * biomass_kg_per_batch
    nutrient_per_batch = {
        key: value * biomass_kg_per_batch
        for key, value in available_nutrients.items()
    }
    avoided_fertiliser_per_batch = {
        key: value * biomass_kg_per_batch
        for key, value in avoided_fertiliser_products.items()
    }
    pathway_quantities = pd.DataFrame([{
        "source_module": "biomass",
        "pathway": pathway,
        "pathway_label": pathway_label,
        "biomass_kg_per_batch": biomass_kg_per_batch,
        "biomass_kg_per_year": biomass_kg_per_year,
        "biogas_m3_per_batch": biogas_yield * biomass_kg_per_batch,
        "biomethane_m3_per_batch": biomethane_m3_per_batch,
        "upgrading_electricity_kwh_per_batch": upgrading_kwh_per_kg * biomass_kg_per_batch,
        "digestate_kg_per_batch": digestate_kg_per_batch,
        # Nutrient-equivalent outputs retained in the data tables.
        "avoided_nitrogen_fertiliser_kg_per_batch": nutrient_per_batch["nitrogen_as_n"],
        "avoided_phosphate_fertiliser_kg_per_batch": nutrient_per_batch["phosphorus_as_p2o5"],
        "avoided_potassium_fertiliser_kg_per_batch": nutrient_per_batch["potassium_as_k2o"],
        # Actual ecoinvent product quantities used in the avoided burdens.
        "avoided_ammonium_nitrate_kg_per_batch": avoided_fertiliser_per_batch["ammonium_nitrate"],
        "avoided_inorganic_phosphorus_fertiliser_as_p2o5_kg_per_batch": avoided_fertiliser_per_batch["phosphorus_fertiliser_as_p2o5"],
        "avoided_inorganic_potassium_fertiliser_as_k2o_kg_per_batch": avoided_fertiliser_per_batch["potassium_fertiliser_as_k2o"],
        "ammonium_nitrate_n_mass_fraction": ammonium_nitrate_n_fraction,
        "biogas_yield_source": yield_source,
        "digestate_yield_source": digestate_source,
    }])

    secondary_outputs = pd.DataFrame([
        {
            "source_module": "biomass",
            "parent_stream_category": "residual_biomass",
            "parent_route": pathway,
            "secondary_stream_category": "biomethane",
            "secondary_stream_name": "biomethane_to_gas_grid",
            "stream_direction": "recovered_output",
            "quantity_per_batch": biomethane_m3_per_batch,
            "quantity_unit": "m3",
            "embedded_in_lca_activity": True,
            "export_to_wastewater_pool": False,
            "routable_in_master": False,
            "source_of_quantity": yield_source,
            "assumption_note": "Biomethane is represented in the route LCA by an avoided high-pressure natural-gas supply credit.",
        },
        {
            "source_module": "biomass",
            "parent_stream_category": "residual_biomass",
            "parent_route": pathway,
            "secondary_stream_category": "digestate",
            "secondary_stream_name": "agricultural_digestate",
            "stream_direction": "recovered_output",
            "quantity_per_batch": digestate_kg_per_batch,
            "quantity_unit": "kg",
            "embedded_in_lca_activity": True,
            "export_to_wastewater_pool": False,
            "routable_in_master": False,
            "source_of_quantity": digestate_source,
            "assumption_note": "Digestate recovery is represented through the available N, P2O5 and K2O fertiliser-substitution credits.",
        },
        {
            "source_module": "biomass",
            "parent_stream_category": "residual_biomass",
            "parent_route": pathway,
            "secondary_stream_category": "fertiliser_substitution",
            "secondary_stream_name": "available_nitrogen",
            "stream_direction": "avoided_product",
            "quantity_per_batch": nutrient_per_batch["nitrogen_as_n"],
            "quantity_unit": "kg N",
            "embedded_in_lca_activity": True,
            "export_to_wastewater_pool": False,
            "routable_in_master": False,
            "source_of_quantity": "digestate nutrient parameters",
            "assumption_note": (
                "Available digestate nitrogen is reported as kg N. The LCIA "
                "credit uses ammonium nitrate at 0.35 kg N per kg product."
            ),
        },
        {
            "source_module": "biomass",
            "parent_stream_category": "residual_biomass",
            "parent_route": pathway,
            "secondary_stream_category": "fertiliser_substitution",
            "secondary_stream_name": "available_phosphate",
            "stream_direction": "avoided_product",
            "quantity_per_batch": nutrient_per_batch["phosphorus_as_p2o5"],
            "quantity_unit": "kg P2O5",
            "embedded_in_lca_activity": True,
            "export_to_wastewater_pool": False,
            "routable_in_master": False,
            "source_of_quantity": "digestate nutrient parameters",
            "assumption_note": (
                "Available digestate phosphorus is expressed as kg P2O5 and "
                "directly substitutes inorganic phosphorus fertiliser, as P2O5."
            ),
        },
        {
            "source_module": "biomass",
            "parent_stream_category": "residual_biomass",
            "parent_route": pathway,
            "secondary_stream_category": "fertiliser_substitution",
            "secondary_stream_name": "available_potash",
            "stream_direction": "avoided_product",
            "quantity_per_batch": nutrient_per_batch["potassium_as_k2o"],
            "quantity_unit": "kg K2O",
            "embedded_in_lca_activity": True,
            "export_to_wastewater_pool": False,
            "routable_in_master": False,
            "source_of_quantity": "digestate nutrient parameters",
            "assumption_note": (
                "Available digestate potassium is expressed as kg K2O and "
                "directly substitutes inorganic potassium fertiliser, as K2O."
            ),
        },
    ])

    route_summary = (
        main_detailed.groupby(
            ["pathway", "impact_category", "life_cycle_stage", "stage_type"],
            as_index=False,
        )["impact_per_batch"].sum()
    )

    return {
        "main_detailed_table": main_detailed,
        "route_factor_table": route_factor_table,
        "secondary_outputs": secondary_outputs,
        "inventory": biomass_inventory,
        "pathway_quantities": pathway_quantities,
        "route_summary": route_summary,
        "activity_mapping": pd.DataFrame(mapping_rows),
    }


def _validate_biomass_settings(config):
    required = [
        "upgrading_recovery_fraction",
        "raw_biogas_ncv_mj_per_m3",
        "biomethane_ncv_mj_per_m3",
        "upgrading_electricity_kwh_per_m3_biomethane",
        "digestate_recovery_fraction",
        "digestate_available_nutrients_kg_per_tonne",
        "fertiliser_substitution_efficiency",
        "biogas_yield_m3_per_kg_biomass",
        "digestate_yield_kg_per_kg_biomass",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError("Biomass settings are missing: " + ", ".join(missing))

    nutrient_keys = {"N", "P2O5", "K2O"}
    missing_nutrients = nutrient_keys.difference(
        config["digestate_available_nutrients_kg_per_tonne"]
    )
    missing_substitution = nutrient_keys.difference(
        config["fertiliser_substitution_efficiency"]
    )
    if missing_nutrients:
        raise ValueError(
            "digestate_available_nutrients_kg_per_tonne is missing: "
            + ", ".join(sorted(missing_nutrients))
        )
    if missing_substitution:
        raise ValueError(
            "fertiliser_substitution_efficiency is missing: "
            + ", ".join(sorted(missing_substitution))
        )

    fractions = {
        "upgrading_recovery_fraction": config["upgrading_recovery_fraction"],
        "digestate_recovery_fraction": config["digestate_recovery_fraction"],
        **{
            f"fertiliser_substitution_efficiency.{key}":
                config["fertiliser_substitution_efficiency"][key]
            for key in sorted(nutrient_keys)
        },
    }
    for name, value in fractions.items():
        value = float(value)
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1; found {value}.")

    positive = {
        "raw_biogas_ncv_mj_per_m3": config["raw_biogas_ncv_mj_per_m3"],
        "biomethane_ncv_mj_per_m3": config["biomethane_ncv_mj_per_m3"],
        "biogas_yield_m3_per_kg_biomass": config["biogas_yield_m3_per_kg_biomass"],
        "digestate_yield_kg_per_kg_biomass": config["digestate_yield_kg_per_kg_biomass"],
    }

    for name, value in positive.items():
        if float(value) <= 0:
            raise ValueError(f"{name} must be positive; found {value}.")

    if float(config["upgrading_electricity_kwh_per_m3_biomethane"]) < 0:
        raise ValueError(
            "upgrading_electricity_kwh_per_m3_biomethane cannot be negative."
        )

    ammonium_nitrate_n_fraction = float(
        config.get("ammonium_nitrate_n_mass_fraction", 0.35)
    )
    if not 0 < ammonium_nitrate_n_fraction <= 1:
        raise ValueError(
            "ammonium_nitrate_n_mass_fraction must be greater than 0 and "
            f"no greater than 1; found {ammonium_nitrate_n_fraction}."
        )

    for nutrient in sorted(nutrient_keys):
        value = float(
            config["digestate_available_nutrients_kg_per_tonne"][nutrient]
        )
        if value < 0:
            raise ValueError(f"Digestate nutrient {nutrient} cannot be negative.")


def _is_waste_reference_product(reference_product):
    text = str(reference_product or "").strip().lower()
    return "waste" in text or "residual biomass" in text



def _reviewed_activity_specs(background_db, configured_specs=None):
    """Return the exact ecoinvent mappings reviewed for the biomass route."""
    reviewed = {
        "anaerobic_digestion": {
            "mapping_key": "anaerobic_digestion_CH",
            "candidate_names": [
                "treatment of biowaste by anaerobic digestion",
            ],
            "candidate_reference_products": [
                "biowaste",
            ],
            "unit": "kilogram",
            "preferred_locations": ["CH"],
            "require_preferred_location": True,
            "assumption_note": (
                "Swiss anaerobic-digestion treatment is used as a European "
                "technology proxy because no UK or regional European dataset "
                "was identified."
            ),
        },
        "upgrading_electricity": {
            "mapping_key": "upgrading_electricity_GB",
            "candidate_names": [
                "market for electricity, medium voltage",
            ],
            "candidate_reference_products": [
                "electricity, medium voltage",
            ],
            "unit": "kilowatt hour",
            "preferred_locations": ["GB"],
            "require_preferred_location": True,
            "assumption_note": (
                "GB medium-voltage electricity supplies the foreground "
                "biogas-upgrading electricity demand."
            ),
        },
        "avoided_natural_gas": {
            "mapping_key": "avoided_natural_gas_high_pressure_GB",
            "candidate_names": [
                "market for natural gas, high pressure",
            ],
            "candidate_reference_products": [
                "natural gas, high pressure",
            ],
            "unit": "cubic meter",
            "preferred_locations": ["GB"],
            "require_preferred_location": True,
            "assumption_note": (
                "Grid-injected biomethane displaces GB high-pressure fossil "
                "natural-gas supply on a cubic-metre basis."
            ),
        },
        "avoided_nitrogen_fertiliser": {
            "mapping_key": "avoided_ammonium_nitrate_RER",
            "candidate_names": [
                "market for ammonium nitrate",
            ],
            "candidate_reference_products": [
                "ammonium nitrate",
            ],
            "unit": "kilogram",
            "preferred_locations": ["RER"],
            "require_preferred_location": True,
            "assumption_note": (
                "Available digestate nitrogen displaces European ammonium "
                "nitrate. One kilogram of ammonium nitrate represents "
                "0.35 kg nitrogen as N."
            ),
        },
        "avoided_phosphate_fertiliser": {
            "mapping_key": "avoided_inorganic_phosphorus_fertiliser_GB",
            "candidate_names": [
                "market for inorganic phosphorus fertiliser, as P2O5",
            ],
            "candidate_reference_products": [
                "inorganic phosphorus fertiliser, as P2O5",
            ],
            "unit": "kilogram",
            "preferred_locations": ["GB"],
            "require_preferred_location": True,
            "assumption_note": (
                "Available digestate phosphorus, expressed as P2O5, displaces "
                "GB inorganic phosphorus fertiliser on the same basis."
            ),
        },
        "avoided_potassium_fertiliser": {
            "mapping_key": "avoided_inorganic_potassium_fertiliser_GB",
            "candidate_names": [
                "market for inorganic potassium fertiliser, as K2O",
            ],
            "candidate_reference_products": [
                "inorganic potassium fertiliser, as K2O",
            ],
            "unit": "kilogram",
            "preferred_locations": ["GB"],
            "require_preferred_location": True,
            "assumption_note": (
                "Available digestate potassium, expressed as K2O, displaces "
                "GB inorganic potassium fertiliser on the same basis."
            ),
        },
    }

    configured_specs = configured_specs or {}
    for key, spec in reviewed.items():
        configured = configured_specs.get(key, {})
        spec["database"] = configured.get("database", background_db)

    return reviewed


def _resolve_activity(bd, background_db, spec):
    database = spec.get("database", background_db)
    names = spec.get("candidate_names") or [spec.get("name")]
    reference_products = spec.get("candidate_reference_products") or [
        spec.get("reference_product")
    ]
    names = [value for value in names if value]
    reference_products = [value for value in reference_products if value]
    preferred_locations = spec.get("preferred_locations", [])
    expected_unit = spec.get("unit")

    candidates = []
    for activity in bd.Database(database):
        if activity.get("name") not in names:
            continue
        if reference_products and activity.get("reference product") not in reference_products:
            continue
        if expected_unit and activity.get("unit") != expected_unit:
            continue
        candidates.append(activity)

    if not candidates:
        required_tokens = [str(token).lower() for token in spec.get("required_name_tokens", [])]
        for activity in bd.Database(database):
            name = str(activity.get("name", "")).lower()
            reference = str(activity.get("reference product", "")).lower()
            if required_tokens and not all(token in name for token in required_tokens):
                continue
            if reference_products and reference not in {
                str(value).lower() for value in reference_products
            }:
                continue
            if expected_unit and activity.get("unit") != expected_unit:
                continue
            candidates.append(activity)

    if not candidates:
        raise ValueError(
            "No ecoinvent activity matched biomass mapping "
            f"{spec.get('mapping_key', 'unnamed')!r}. Names={names}, "
            f"reference_products={reference_products}, unit={expected_unit!r}."
        )

    for location in preferred_locations:
        matches = [
            activity
            for activity in candidates
            if activity.get("location") == location
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            exact_name = [
                activity
                for activity in matches
                if activity.get("name") == names[0]
            ]
            if len(exact_name) == 1:
                return exact_name[0]

    if preferred_locations and spec.get("require_preferred_location", False):
        options = [
            (
                activity.get("name"),
                activity.get("reference product"),
                activity.get("location"),
                activity.get("unit"),
                activity.key,
            )
            for activity in candidates
        ]
        raise ValueError(
            "No candidate matched the reviewed geography for biomass mapping "
            f"{spec.get('mapping_key', 'unnamed')!r}. Required locations="
            f"{preferred_locations}; candidates={options}"
        )

    if len(candidates) == 1:
        return candidates[0]

    options = [
        (
            activity.get("name"),
            activity.get("reference product"),
            activity.get("location"),
            activity.get("unit"),
            activity.key,
        )
        for activity in candidates
    ]
    raise ValueError(
        "Biomass activity mapping is ambiguous. Add or reorder preferred_locations "
        f"for {spec.get('mapping_key', 'unnamed')!r}. Candidates: {options}"
    )


def _activity_metadata(mapping_key, activity, spec):
    return {
        "mapping_key": mapping_key,
        "activity_name": activity.get("name"),
        "reference_product": activity.get("reference product"),
        "location": activity.get("location"),
        "unit": activity.get("unit"),
        "database": activity.key[0],
        "code": activity.key[1],
        "mapping_type": "reviewed_exact_metadata_with_location_preference",
        "activity_search": (
            "candidate activity names + candidate reference products + unit + "
            "ordered preferred locations"
        ),
        "mapping_assumption": spec.get("assumption_note", ""),
    }
