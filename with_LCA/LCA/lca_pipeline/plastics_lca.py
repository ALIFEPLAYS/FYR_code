from __future__ import annotations

from .scientific_parameters import (
    SUBSTITUTION_RATIO,
    SUT_MANUFACTURING_GWP_ADDER_KGCO2E_PER_KG,
)
from .human_health_daly import (
    IMPACT_CATEGORY as HUMAN_HEALTH_DALY,
    add_human_health_method,
    sut_manufacturing_human_health_daly_per_kg,
)
from .ecosystem_quality import add_ecosystem_method, sut_manufacturing_climate_ecosystem_per_kg
from .lcia_methods import core_lcia_methods


def run_plastics_lca(sut_inventory, settings, project_root):
    """Run the existing SUT production/EOL route-factor logic with the explicit autoclave addition."""
    from pathlib import Path
    from datetime import datetime
    import numpy as np
    import pandas as pd
    import bw2data as bd
    import bw2calc as bc

    display = lambda *args, **kwargs: None
    DATE_TAG = datetime.now().strftime("%Y%m%d")
    sut_polymer_inventory = sut_inventory.copy()
    SUT_DATASET_DIR = Path(project_root)
    ECOINVENT_VERSION = settings["brightway"]["ecoinvent_version"]
    ECOINVENT_SYSTEM_MODEL = settings["brightway"]["system_model"]
    SUT_MANUFACTURING_GWP_ADDER = float(
        settings["plastics"].get(
            "sut_manufacturing_gwp_adder_kgco2e_per_kg",
            SUT_MANUFACTURING_GWP_ADDER_KGCO2E_PER_KG,
        )
    )
    SUT_MANUFACTURING_DALY_ADDER = sut_manufacturing_human_health_daly_per_kg(settings)
    VIRGIN_MATERIAL_SUBSTITUTION_RATIO = float(
        settings["plastics"].get(
            "mechanical_recycling_substitution_ratio",
            SUBSTITUTION_RATIO,
        )
    )
    if SUT_MANUFACTURING_GWP_ADDER < 0:
        raise ValueError("The SUT manufacturing GWP adder must be non-negative.")
    if not 0 <= VIRGIN_MATERIAL_SUBSTITUTION_RATIO <= 1:
        raise ValueError("The mechanical-recycling substitution ratio must be between 0 and 1.")
    from pathlib import Path
    from datetime import datetime

    import numpy as np
    import pandas as pd

    import bw2data as bd
    import bw2calc as bc


    pd.set_option("display.max_columns", 200)
    pd.set_option("display.max_colwidth", 180)

    DATE_TAG = datetime.now().strftime("%Y%m%d")

    # ---------------------------------------------------------------------
    # Required detailed fields
    # ---------------------------------------------------------------------

    PROCESS_AREA_COL = "process_area"
    SUPERPRO_SECTION_COL = "superpro_section"
    UNIT_OPERATION_COL = "unit_operation"
    SUT_FLOW_COL = "flow_name_superpro"
    SUT_CATEGORY_TYPE_COL = "sut_subcomponent"
    POLYMER_COL = "polymer"
    MASS_COL = "polymer_kg_per_batch"

    REQUIRED_COLS = [
        "inventory_id",
        PROCESS_AREA_COL,
        SUPERPRO_SECTION_COL,
        UNIT_OPERATION_COL,
        SUT_FLOW_COL,
        SUT_CATEGORY_TYPE_COL,
        POLYMER_COL,
        MASS_COL,
    ]

    missing_cols = [col for col in REQUIRED_COLS if col not in sut_polymer_inventory.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in sut_polymer_inventory.csv: {missing_cols}")

    sut_polymer_inventory = sut_polymer_inventory.copy()
    sut_polymer_inventory[POLYMER_COL] = sut_polymer_inventory[POLYMER_COL].astype(str).str.strip()
    sut_polymer_inventory[MASS_COL] = pd.to_numeric(sut_polymer_inventory[MASS_COL], errors="coerce")

    if sut_polymer_inventory[MASS_COL].isna().any():
        display(sut_polymer_inventory[sut_polymer_inventory[MASS_COL].isna()])
        raise ValueError("Some polymer_kg_per_batch values are missing or non-numeric.")

    if (sut_polymer_inventory[MASS_COL] < 0).any():
        display(sut_polymer_inventory[sut_polymer_inventory[MASS_COL] < 0])
        raise ValueError("Negative polymer masses found. Check inventory before LCA.")

    detail_cols = [
        "inventory_id",
        PROCESS_AREA_COL,
        SUPERPRO_SECTION_COL,
        UNIT_OPERATION_COL,
        SUT_FLOW_COL,
        SUT_CATEGORY_TYPE_COL,
        POLYMER_COL,
        MASS_COL,
    ]

    print("Detailed LCA fields retained:")
    for col in detail_cols:
        print(f"- {col}")

    # ---------------------------------------------------------------------
    # Inventory mass checks
    # ---------------------------------------------------------------------

    polymer_demand = (
        sut_polymer_inventory
        .groupby(POLYMER_COL, dropna=False, as_index=False)
        .agg(polymer_kg_per_batch=(MASS_COL, "sum"))
        .sort_values("polymer_kg_per_batch", ascending=False)
    )

    display(polymer_demand)



    # ---------------------------------------------------------------------
    # Brightway project and databases — ecoinvent 3.12 cutoff
    # ---------------------------------------------------------------------
    # Run the separate ecoinvent 3.12 setup/import notebook before this notebook.

    ECOINVENT_VERSION = settings["brightway"]["ecoinvent_version"]
    ECOINVENT_SYSTEM_MODEL = settings["brightway"]["system_model"]

    BRIGHTWAY_PROJECT = settings["brightway"]["project"]
    BACKGROUND_DB = settings["brightway"]["background_database"]
    BIOSPHERE_DB = settings["brightway"]["biosphere_database"]

    bd.projects.set_current(BRIGHTWAY_PROJECT)
    SUT_MANUFACTURING_ECOSYSTEM_ADDER = sut_manufacturing_climate_ecosystem_per_kg(settings)

    if not (BACKGROUND_DB in bd.databases):
        raise AssertionError(f'Missing background database: {BACKGROUND_DB}. Run the ecoinvent 3.12 setup/import notebook first.')
    if not (BIOSPHERE_DB in bd.databases):
        raise AssertionError(f'Missing biosphere database: {BIOSPHERE_DB}. Run the ecoinvent 3.12 setup/import notebook first.')

    print(f"Project: {bd.projects.current}")
    print(f"Background database: {BACKGROUND_DB}")
    print(f"Biosphere database: {BIOSPHERE_DB}")


    # ---------------------------------------------------------------------
    # LCIA methods — one canonical mapping shared by every LCA module.
    # ---------------------------------------------------------------------
    # Three midpoint categories used for the LCA factor table:
    #   1. GWP
    #   2. fossil resource use
    #   3. water scarcity
    #
    # GWP uses IMPACT World+ v2.1 short-term carbon footprint (IPCC AR6).
    # The ReCiPe fossil-fuel method is deliberately unchanged.
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
    # Explicit Brightway activity mapping helpers
    # ---------------------------------------------------------------------
    # The notebook avoids broad activity searches for mapped activities.
    # Where ecoinvent codes are known, the code is used directly and validated.
    # Where a code is not yet known, name + reference-product matching is used.

    def get_activity_by_code(database, code, expected_name=None, expected_reference_product=None):
        activity = bd.get_activity((database, code))

        if expected_name is not None and activity.get("name") != expected_name:
            raise ValueError(
                f"Hardcoded activity name mismatch for {code}.\n"
                f"Expected: {expected_name}\n"
                f"Found:    {activity.get('name')}"
            )

        if expected_reference_product is not None and activity.get("reference product") != expected_reference_product:
            raise ValueError(
                f"Hardcoded activity reference-product mismatch for {code}.\n"
                f"Expected: {expected_reference_product}\n"
                f"Found:    {activity.get('reference product')}"
            )

        return activity


    def find_exact_activity(name, reference_product, preferred_locations, background_db=BACKGROUND_DB):
        db = bd.Database(background_db)

        candidates = [
            act for act in db
            if act.get("name") == name
            and act.get("reference product") == reference_product
        ]

        if not candidates:
            raise ValueError(
                "No activity found.\n"
                f"name={name}\n"
                f"reference_product={reference_product}"
            )

        candidates_df = pd.DataFrame(
            [
                {
                    "activity": act,
                    "name": act.get("name"),
                    "reference_product": act.get("reference product"),
                    "location": act.get("location"),
                    "unit": act.get("unit"),
                    "database": act.key[0],
                    "code": act.key[1],
                }
                for act in candidates
            ]
        )

        for location in preferred_locations:
            matches = candidates_df[candidates_df["location"].eq(location)]
            if len(matches) > 0:
                return matches.iloc[0]["activity"], candidates_df.drop(columns=["activity"])

        return candidates_df.iloc[0]["activity"], candidates_df.drop(columns=["activity"])


    def activity_metadata(activity, mapping_type, activity_search=None):
        return {
            "activity_name": activity.get("name"),
            "reference_product": activity.get("reference product"),
            "location": activity.get("location"),
            "unit": activity.get("unit"),
            "database": activity.key[0],
            "code": activity.key[1],
            "mapping_type": mapping_type,
            "activity_search": activity_search,
        }


    # ---------------------------------------------------------------------
    # Explicit polymer-to-ecoinvent production mapping — ecoinvent 3.12
    # ---------------------------------------------------------------------
    # Production activities are resolved by activity name + reference product
    # in the ecoinvent 3.12 cutoff database. This avoids carrying over 3.10.1
    # activity codes, which are not guaranteed to be valid in 3.12.

    PRODUCTION_ACTIVITY_MAP = {
        "LLDPE": {
            "name": "polyethylene production, linear low density, granulate",
            "reference_product": "polyethylene, linear low density, granulate",
            "preferred_locations": ["RER", "Europe without Switzerland", "GLO", "RoW"],
            "mapping_type": "direct",
        },
        "LDPE": {
            "name": "polyethylene production, low density, granulate",
            "reference_product": "polyethylene, low density, granulate",
            "preferred_locations": ["RER", "Europe without Switzerland", "GLO", "RoW"],
            "mapping_type": "direct",
        },
        "HDPE": {
            "name": "polyethylene production, high density, granulate",
            "reference_product": "polyethylene, high density, granulate",
            "preferred_locations": ["RER", "Europe without Switzerland", "GLO", "RoW"],
            "mapping_type": "direct",
        },
        "PP": {
            "name": "polypropylene production, granulate",
            "reference_product": "polypropylene, granulate",
            "preferred_locations": ["RER", "Europe without Switzerland", "GLO", "RoW"],
            "mapping_type": "direct",
        },
        "TPE": {
            "name": "polyurethane production, flexible foam, TDI-based, low density",
            "reference_product": "polyurethane, flexible foam",
            "preferred_locations": ["RER", "GLO", "RoW"],
            "mapping_type": "proxy",
        },
        "Silicone": {
            "name": "silicone product production",
            "reference_product": "silicone product",
            "preferred_locations": ["RER", "GLO", "RoW"],
            "mapping_type": "sensitivity",
        },
        "EVOH": {
            "name": "ethylene vinyl acetate copolymer production",
            "reference_product": "ethylene vinyl acetate copolymer",
            "preferred_locations": ["RER", "GLO", "RoW"],
            "mapping_type": "proxy",
        },
    }

    polymer_lca_mapping = pd.DataFrame(
        [
            {
                "polymer": polymer,
                "name": spec["name"],
                "reference_product": spec["reference_product"],
                "preferred_locations": "; ".join(spec.get("preferred_locations", [])),
                "mapping_type": spec["mapping_type"],
            }
            for polymer, spec in PRODUCTION_ACTIVITY_MAP.items()
        ]
    )

    display(polymer_lca_mapping)


    # ---------------------------------------------------------------------
    # Resolve explicitly mapped polymer production activities
    # ---------------------------------------------------------------------
    # Only polymers actually present in the SUT inventory are resolved here.
    # `Other` is handled later as a weighted-average proxy.

    present_polymers = set(polymer_demand[POLYMER_COL].dropna().astype(str))
    required_production_polymers = sorted(
        polymer for polymer in present_polymers
        if polymer in PRODUCTION_ACTIVITY_MAP
    )

    unmapped_non_other_polymers = sorted(
        polymer for polymer in present_polymers
        if polymer not in PRODUCTION_ACTIVITY_MAP and polymer not in {"Other", "Non-plastic"}
    )
    if unmapped_non_other_polymers:
        raise ValueError(
            "The following inventory polymers need explicit production mappings before LCA: "
            f"{unmapped_non_other_polymers}"
        )

    mapped_activity_rows = []

    for polymer in required_production_polymers:
        spec = PRODUCTION_ACTIVITY_MAP[polymer]

        if "code" in spec:
            activity = get_activity_by_code(
                database=spec.get("database", BACKGROUND_DB),
                code=spec["code"],
                expected_name=spec.get("expected_name", spec.get("name")),
                expected_reference_product=spec.get("expected_reference_product", spec.get("reference_product")),
            )
            candidates = pd.DataFrame([activity_metadata(activity, "fixed_code_reviewed", spec["code"])])
            activity_search = spec["code"]
            mapping_type = "fixed_code_reviewed"
        else:
            activity, candidates = find_exact_activity(
                name=spec["name"],
                reference_product=spec["reference_product"],
                preferred_locations=spec.get("preferred_locations", ["RER", "GLO", "RoW"]),
                background_db=BACKGROUND_DB,
            )
            activity_search = f"{spec['name']} | {spec['reference_product']}"
            mapping_type = spec["mapping_type"]

        metadata = activity_metadata(
            activity=activity,
            mapping_type=mapping_type,
            activity_search=activity_search,
        )
        metadata["polymer"] = polymer
        metadata["candidate_count"] = len(candidates)
        mapped_activity_rows.append(metadata)

    mapped_activities = pd.DataFrame(mapped_activity_rows)[
        [
            "polymer",
            "activity_name",
            "reference_product",
            "location",
            "unit",
            "database",
            "code",
            "mapping_type",
            "activity_search",
            "candidate_count",
        ]
    ]

    display(mapped_activities)


    # ---------------------------------------------------------------------
    # Virgin polymer production factors by LCIA category
    # Virgin polymer production is an ordinary product activity, so it should be demanded as bc.LCA({activity: 1.0}, method)
    # ---------------------------------------------------------------------

    production_factor_rows = []

    for _, row in mapped_activities.iterrows():
        activity = bd.get_activity((row["database"], row["code"]))

        for impact_category, method in LCIA_METHODS.items():
            lca = bc.LCA({activity: 1.0}, method)
            lca.lci()
            lca.lcia()

            production_factor_rows.append(
                {
                    "polymer": row["polymer"],
                    "life_cycle_stage": "virgin_polymer_production",
                    "impact_category": impact_category,
                    "impact_per_kg": float(lca.score),
                    "activity_name": row["activity_name"],
                    "reference_product": row["reference_product"],
                    "location": row["location"],
                    "unit": row["unit"],
                    "database": row["database"],
                    "code": row["code"],
                    "mapping_type": row["mapping_type"],
                }
            )

    polymer_production_factors = pd.DataFrame(production_factor_rows)
    display(polymer_production_factors)

    # ---------------------------------------------------------------------
    # Add derived production factor for 'Other'
    # ---------------------------------------------------------------------

    known_polymer_mass = polymer_demand[
        polymer_demand[POLYMER_COL].isin(polymer_production_factors["polymer"])
    ].copy()

    if polymer_demand[POLYMER_COL].eq("Other").any():
        other_factor_rows = []

        for impact_category in polymer_production_factors["impact_category"].unique():
            factors_i = polymer_production_factors[
                polymer_production_factors["impact_category"].eq(impact_category)
            ][["polymer", "impact_per_kg"]]

            weighted_i = known_polymer_mass.merge(
                factors_i,
                left_on=POLYMER_COL,
                right_on="polymer",
                how="inner",
            )

            other_factor = np.average(
                weighted_i["impact_per_kg"],
                weights=weighted_i["polymer_kg_per_batch"],
            )

            other_factor_rows.append(
                {
                    "polymer": "Other",
                    "life_cycle_stage": "virgin_polymer_production",
                    "impact_category": impact_category,
                    "impact_per_kg": float(other_factor),
                    "activity_name": "derived weighted average of mapped polymers",
                    "reference_product": "generic plastic proxy",
                    "location": "derived",
                    "unit": "kilogram",
                    "database": "derived",
                    "code": "derived",
                    "mapping_type": "derived_proxy",
                }
            )

        polymer_production_factors = pd.concat(
            [polymer_production_factors, pd.DataFrame(other_factor_rows)],
            ignore_index=True,
        )

    other_mass = polymer_demand.loc[polymer_demand[POLYMER_COL].eq("Other"), "polymer_kg_per_batch"].sum()
    total_mass = polymer_demand["polymer_kg_per_batch"].sum()
    print(f"Other mass: {other_mass:,.3f} kg/batch ({100 * other_mass / total_mass:,.2f}% of total polymer mass)")

    display(polymer_production_factors)

    # ---------------------------------------------------------------------
    # Virgin polymer production impacts - detailed row level
    # ---------------------------------------------------------------------

    sut_inventory_lca = sut_polymer_inventory[
        ~sut_polymer_inventory[POLYMER_COL].eq("Non-plastic")
    ].copy()

    production_impacts_detailed = sut_inventory_lca.merge(
        polymer_production_factors,
        left_on=POLYMER_COL,
        right_on="polymer",
        how="left",
        suffixes=("", "_factor"),
    )

    missing = production_impacts_detailed[production_impacts_detailed["impact_per_kg"].isna()]
    if len(missing) > 0:
        display(missing)
        raise ValueError("Missing virgin polymer production factor for at least one inventory row.")

    production_impacts_detailed["impact_per_batch"] = (
        production_impacts_detailed[MASS_COL] * production_impacts_detailed["impact_per_kg"]
    )
    production_impacts_detailed["eol_sink"] = "not_applicable"

    # Keep canonical polymer column only.
    if "polymer_factor" in production_impacts_detailed.columns:
        production_impacts_detailed = production_impacts_detailed.drop(columns=["polymer_factor"])

    display(production_impacts_detailed.head())

    # ---------------------------------------------------------------------
    # Additional downstream SUT component-manufacturing burden (GWP only)
    # ---------------------------------------------------------------------
    # This supplier/component-manufacturing proxy is additional to ecoinvent
    # polymer production. There is no evidence-backed fossil-resource or water-
    # scarcity proxy, so no values are fabricated for those categories.
    sut_manufacturing_impacts_detailed = sut_inventory_lca.copy()
    sut_manufacturing_impacts_detailed["life_cycle_stage"] = "sut_component_manufacturing"
    sut_manufacturing_impacts_detailed["impact_category"] = "gwp"
    sut_manufacturing_impacts_detailed["impact_per_kg"] = SUT_MANUFACTURING_GWP_ADDER
    sut_manufacturing_impacts_detailed["impact_per_batch"] = (
        sut_manufacturing_impacts_detailed[MASS_COL] * SUT_MANUFACTURING_GWP_ADDER
    )
    sut_manufacturing_impacts_detailed["eol_sink"] = "not_applicable"
    sut_manufacturing_impacts_detailed["activity_name"] = (
        "additional pharmaceutical-grade SUT component-manufacturing GWP proxy"
    )
    sut_manufacturing_impacts_detailed["reference_product"] = "modelled SUT plastic component"
    sut_manufacturing_impacts_detailed["location"] = "supplier-stage proxy"
    sut_manufacturing_impacts_detailed["unit"] = "kilogram"
    sut_manufacturing_impacts_detailed["database"] = "foreground assumption"
    sut_manufacturing_impacts_detailed["code"] = "SUT_MANUFACTURING_GWP_ADDER"
    sut_manufacturing_impacts_detailed["mapping_type"] = "gwp_only_supplier_stage_proxy"
    sut_manufacturing_impacts_detailed["assumption_source"] = settings["plastics"][
        "sut_manufacturing_gwp_adder_source"
    ]
    sut_manufacturing_impacts_detailed["assumption_source_references"] = " | ".join(
        settings["plastics"]["sut_manufacturing_gwp_adder_source_references"]
    )
    sut_manufacturing_impacts_detailed["assumption_boundary"] = settings["plastics"][
        "sut_manufacturing_gwp_adder_boundary"
    ]
    sut_manufacturing_impacts_detailed["value_qualifier"] = settings["plastics"][
        "sut_manufacturing_gwp_adder_value_qualifier"
    ]
    sut_manufacturing_impacts_detailed["activity_demand_amount"] = 1.0
    sut_manufacturing_impacts_detailed["activity_demand_basis"] = "per_kg_modelled_SUT_plastic"
    sut_manufacturing_impacts_detailed["demand_orientation_note"] = (
        "Upper-bound bundled 1.9 kg CO2e/kg supplier-stage fabrication proxy from "
        "Reiners et al. (2026), citing Ramasamy (2018); applied to GWP only and "
        "kept separate from ecoinvent polymer-production burdens."
    )

    # The manual supplier-stage proxy has no full LCI. Add only its ReCiPe
    # climate-change human-health damage using the documented H factor; do not
    # invent toxicity, water or other endpoint pathways.
    sut_manufacturing_daly_detailed = sut_manufacturing_impacts_detailed.copy()
    sut_manufacturing_daly_detailed["impact_category"] = HUMAN_HEALTH_DALY
    sut_manufacturing_daly_detailed["impact_per_kg"] = SUT_MANUFACTURING_DALY_ADDER
    sut_manufacturing_daly_detailed["impact_per_batch"] = (
        sut_manufacturing_daly_detailed[MASS_COL] * SUT_MANUFACTURING_DALY_ADDER
    )
    sut_manufacturing_daly_detailed["code"] = "SUT_MANUFACTURING_CLIMATE_HEALTH_ADDER"
    sut_manufacturing_daly_detailed["mapping_type"] = "climate_only_recipe_endpoint_h_proxy"
    sut_manufacturing_daly_detailed["demand_orientation_note"] = (
        "Climate-change human-health damage only: 1.9 kg CO2e/kg multiplied by "
        "9.28e-7 DALY/kg CO2e. No other human-health pathways are fabricated."
    )
    sut_manufacturing_impacts_detailed = pd.concat(
        [sut_manufacturing_impacts_detailed, sut_manufacturing_daly_detailed],
        ignore_index=True,
        sort=False,
    )
    sut_manufacturing_ecosystem_detailed = sut_manufacturing_daly_detailed.copy()
    sut_manufacturing_ecosystem_detailed["impact_category"] = "ecosystem_quality"
    sut_manufacturing_ecosystem_detailed["impact_per_kg"] = SUT_MANUFACTURING_ECOSYSTEM_ADDER
    sut_manufacturing_ecosystem_detailed["impact_per_batch"] = (
        sut_manufacturing_ecosystem_detailed[MASS_COL] * SUT_MANUFACTURING_ECOSYSTEM_ADDER
    )
    sut_manufacturing_ecosystem_detailed["code"] = "SUT_MANUFACTURING_CLIMATE_ECOSYSTEM_ADDER"
    sut_manufacturing_ecosystem_detailed["mapping_type"] = "climate_only_recipe_ecosystem_proxy"
    sut_manufacturing_ecosystem_detailed["demand_orientation_note"] = (
        "Climate-only ecosystem damage from the existing 1.9 kg CO2e/kg SUT proxy, "
        "using the installed ReCiPe 2016 (H) fossil CO2 endpoint CF; no other "
        "ecosystem pathways or inventory exchanges are imputed."
    )
    sut_manufacturing_impacts_detailed = pd.concat(
        [sut_manufacturing_impacts_detailed, sut_manufacturing_ecosystem_detailed],
        ignore_index=True,
        sort=False,
    )

    # ---------------------------------------------------------------------
    # Explicit EOL pathway mapping and factors — ecoinvent 3.12 recycling update
    # ---------------------------------------------------------------------
    # This cell builds the route-factor backbone for plastics EOL.
    #
    # Pathways included:
    #   1. landfill_all
    #   2. incineration_all
    #   3. incineration_efw_heat_electricity
    #   4. mechanical_recycling_main_residual_incineration
    #
    # Key modelling choices:
    #   - The old PE-only recycling alternative is removed.
    #   - The main recycling route uses the reviewed ecoinvent 3.12 recycling activities:
    #       * LLDPE and LDPE -> treatment of waste polyethylene, low density, packaging, recycling;
    #         reference product: polyethylene, low density, pellets, recycled.
    #       * PP -> treatment of waste polypropylene, recycling;
    #         reference product: polypropylene, pellets, recycled.
    #       * TPE -> treatment of waste plastic, mixed, recycling;
    #         reference product: plastic, mixed, recycled.
    #   - EVOH, Other and unresolved non-target polymers remain residual and are routed
    #     to residual mixed-plastic incineration in the mechanical recycling pathway.
    #   - Incineration/landfill waste-treatment activities are demanded as -1 kg to
    #     represent treatment of +1 kg physical waste, because ecoinvent waste-treatment
    #     activities use waste-reference-product sign conventions.
    #   - Recycling activities have recycled material as the reference product. Their
    #     treatment burden is therefore calculated per kg incoming waste as:
    #         activity demand = recovery fraction × 1 kg recycled product activity.
    # ---------------------------------------------------------------------

    PE_FAMILY_POLYMERS = {"LLDPE", "LDPE", "HDPE"}
    LOW_DENSITY_PE_RECYCLING_POLYMERS = {"LLDPE", "LDPE"}
    MAIN_MECHANICAL_RECYCLING_POLYMERS = {"LLDPE", "LDPE", "PP", "TPE"}
    RESIDUAL_POLYMERS_FOR_MAIN_RECYCLING = {"EVOH", "Other", "Silicone"}

    EOL_PATHWAYS = [
        "landfill_all",
        "incineration_all",
        "incineration_efw_heat_electricity",
        "mechanical_recycling_main_residual_incineration",
    ]

    REMOVED_EOL_PATHWAYS = [
        "mechanical_recycling_pe_only_residual_incineration",
        "mechanical_recycling_separable_polymers_residual_incineration",
    ]

    # Recovery fractions are used for foreground circularity mass balance,
    # recycling treatment activity demand, and avoided-material credit calculation.
    # The 0.80/0.70 values represent 20%/30% material loss.
    RECYCLING_RECOVERY_FRACTIONS_BY_POLYMER = {
        "LLDPE": 0.80,
        "LDPE": 0.80,
        "PP": 0.80,
        "TPE": 0.70,
        "EVOH": 0.00,
        "Other": 0.00,
        "Silicone": 0.00,
    }

    RECYCLING_RECOVERY_FRACTIONS_BY_POLYMER.update(settings["plastics"].get("recycling_recovery_fraction", {}))

    # reviewed EOL activities.
    # Activity Browser displays geography as Europe (RER); Brightway normally stores
    # this as location code "RER".
    EOL_ACTIVITY_SPECS = {
        "incineration_pe": {
            "activity_key": "incineration_pe",
            "name": "treatment of waste polyethylene, municipal incineration",
            "reference_product": "waste polyethylene",
            "preferred_locations": ["Europe without Switzerland", "RER", "GLO", "RoW"],
            "activity_demand_amount": -1.0,
            "activity_demand_basis": "per_kg_input_waste",
            "demand_orientation_note": "Waste-treatment activity demanded as -1 kg to represent treatment of +1 kg physical waste.",
        },
        "incineration_pp": {
            "activity_key": "incineration_pp",
            "name": "treatment of waste polypropylene, municipal incineration",
            "reference_product": "waste polypropylene",
            "preferred_locations": ["Europe without Switzerland", "RER", "GLO", "RoW"],
            "activity_demand_amount": -1.0,
            "activity_demand_basis": "per_kg_input_waste",
            "demand_orientation_note": "Waste-treatment activity demanded as -1 kg to represent treatment of +1 kg physical waste.",
        },
        "incineration_mixed_plastic": {
            "activity_key": "incineration_mixed_plastic",
            "name": "treatment of waste plastic, mixture, municipal incineration",
            "reference_product": "waste plastic, mixture",
            "preferred_locations": ["Europe without Switzerland", "RER", "GLO", "RoW"],
            "activity_demand_amount": -1.0,
            "activity_demand_basis": "per_kg_input_waste",
            "demand_orientation_note": "Waste-treatment activity demanded as -1 kg to represent treatment of +1 kg physical waste.",
        },
        "landfill_pe": {
            "activity_key": "landfill_pe",
            "name": "treatment of waste polyethylene, sanitary landfill",
            "reference_product": "waste polyethylene",
            "preferred_locations": ["Europe without Switzerland", "RER", "GLO", "RoW"],
            "activity_demand_amount": -1.0,
            "activity_demand_basis": "per_kg_input_waste",
            "demand_orientation_note": "Waste-treatment activity demanded as -1 kg to represent treatment of +1 kg physical waste.",
        },
        "landfill_pp": {
            "activity_key": "landfill_pp",
            "name": "treatment of waste polypropylene, sanitary landfill",
            "reference_product": "waste polypropylene",
            "preferred_locations": ["Europe without Switzerland", "RER", "GLO", "RoW"],
            "activity_demand_amount": -1.0,
            "activity_demand_basis": "per_kg_input_waste",
            "demand_orientation_note": "Waste-treatment activity demanded as -1 kg to represent treatment of +1 kg physical waste.",
        },
        "landfill_mixed_plastic": {
            "activity_key": "landfill_mixed_plastic",
            "name": "treatment of waste plastic, mixture, sanitary landfill",
            "reference_product": "waste plastic, mixture",
            "preferred_locations": ["Europe without Switzerland", "RER", "GLO", "RoW"],
            "activity_demand_amount": -1.0,
            "activity_demand_basis": "per_kg_input_waste",
            "demand_orientation_note": "Waste-treatment activity demanded as -1 kg to represent treatment of +1 kg physical waste.",
        },
        "mechanical_recycling_low_density_pe_packaging_recycling": {
            "activity_key": "mechanical_recycling_low_density_pe_packaging_recycling",
            "name": "treatment of waste polyethylene, low density, packaging, recycling",
            "reference_product": "polyethylene, low density, pellets, recycled",
            "preferred_locations": ["RER", "Europe without Switzerland", "GLO", "RoW"],
            "activity_demand_amount": 1.0,
            "activity_demand_basis": "per_kg_recovered_recycled_output",
            "demand_orientation_note": "Recycling activity demanded as recovered kg recycled low-density polyethylene pellets per kg incoming LLDPE/LDPE waste.",
        },
        "mechanical_recycling_pp_recycling": {
            "activity_key": "mechanical_recycling_pp_recycling",
            "name": "treatment of waste polypropylene, recycling",
            "reference_product": "polypropylene, pellets, recycled",
            "preferred_locations": ["RER", "Europe without Switzerland", "GLO", "RoW"],
            "activity_demand_amount": 1.0,
            "activity_demand_basis": "per_kg_recovered_recycled_output",
            "demand_orientation_note": "Recycling activity demanded as recovered kg recycled polypropylene pellets per kg incoming PP waste.",
        },
        "mechanical_recycling_mixed_plastic_recycling": {
            "activity_key": "mechanical_recycling_mixed_plastic_recycling",
            "name": "treatment of waste plastic, mixed, recycling",
            "reference_product": "plastic, mixed, recycled",
            "preferred_locations": ["RER", "Europe without Switzerland", "GLO", "RoW"],
            "activity_demand_amount": 1.0,
            "activity_demand_basis": "per_kg_recovered_recycled_output",
            "demand_orientation_note": "Recycling activity demanded as recovered kg recycled mixed plastic per kg incoming TPE waste.",
        },
    }


    def find_eol_activity_from_spec(spec, background_db=BACKGROUND_DB):
        """Find an EOL activity either by fixed code or metadata."""
        if "code" in spec:
            activity = get_activity_by_code(
                spec.get("database", background_db),
                spec["code"],
                expected_name=spec.get("expected_name"),
                expected_reference_product=spec.get("expected_reference_product"),
            )
            candidates = pd.DataFrame([activity_metadata(activity, "fixed_code_reviewed", spec["code"])])
            return activity, candidates

        activity, candidates = find_exact_activity(
            name=spec["name"],
            reference_product=spec["reference_product"],
            preferred_locations=spec.get("preferred_locations", ["RER", "GLO", "RoW"]),
            background_db=background_db,
        )
        return activity, candidates


    EOL_ACTIVITY_LOOKUP = {}
    eol_mapping_rows = []

    for activity_key, spec in EOL_ACTIVITY_SPECS.items():
        activity, candidates = find_eol_activity_from_spec(spec)
        EOL_ACTIVITY_LOOKUP[activity_key] = activity

        if "code" in spec:
            activity_search = spec["code"]
            mapping_type = "fixed_code_reviewed"
        else:
            activity_search = f"{spec['name']} | {spec['reference_product']}"
            mapping_type = "fixed_exact_metadata_reviewed"

        row = activity_metadata(
            activity=activity,
            mapping_type=mapping_type,
            activity_search=activity_search,
        )
        row["activity_key"] = activity_key
        row["candidate_count"] = len(candidates)
        row["activity_demand_amount"] = spec["activity_demand_amount"]
        row["activity_demand_basis"] = spec["activity_demand_basis"]
        row["demand_orientation_note"] = spec["demand_orientation_note"]
        eol_mapping_rows.append(row)

    eol_mapped_activities = pd.DataFrame(eol_mapping_rows)[
        [
            "activity_key",
            "activity_name",
            "reference_product",
            "location",
            "unit",
            "database",
            "code",
            "mapping_type",
            "activity_search",
            "candidate_count",
            "activity_demand_amount",
            "activity_demand_basis",
            "demand_orientation_note",
        ]
    ]


    def get_explicit_eol_activity(activity_key):
        if activity_key not in EOL_ACTIVITY_LOOKUP:
            raise KeyError(f"Unknown EOL activity key: {activity_key}")
        return EOL_ACTIVITY_LOOKUP[activity_key]


    def get_eol_activity_demand_amount(activity_key, recycling_recovery_fraction=0.0):
        """Return activity demand per kg incoming SUT plastic waste."""
        spec = EOL_ACTIVITY_SPECS[activity_key]
        base_amount = float(spec["activity_demand_amount"])

        if spec.get("activity_demand_basis") == "per_kg_recovered_recycled_output":
            return base_amount * float(recycling_recovery_fraction)

        return base_amount


    def classify_direct_disposal_sink(polymer, disposal_route):
        """Return display sink, activity key and note for all-incineration/all-landfill routes."""
        polymer = str(polymer).strip()

        if disposal_route == "incineration":
            if polymer in PE_FAMILY_POLYMERS:
                return "incineration_pe", "incineration_pe", "PE-family polymer routed to polyethylene incineration."
            if polymer == "PP":
                return "incineration_pp", "incineration_pp", "PP routed to polypropylene incineration."
            return "incineration_mixed_plastic", "incineration_mixed_plastic", "Non-PE/non-PP polymer routed to mixed-plastic incineration."

        if disposal_route == "landfill":
            if polymer in PE_FAMILY_POLYMERS:
                return "landfill_pe", "landfill_pe", "PE-family polymer routed to polyethylene landfill."
            if polymer == "PP":
                return "landfill_pp", "landfill_pp", "PP routed to polypropylene landfill."
            return "landfill_mixed_plastic", "landfill_mixed_plastic", "Non-PE/non-PP polymer routed to mixed-plastic landfill."

        raise ValueError(f"Unknown disposal route: {disposal_route}")


    def classify_eol_sink(polymer, eol_pathway):
        polymer = str(polymer).strip()

        if eol_pathway == "incineration_all":
            sink, activity_key, note = classify_direct_disposal_sink(polymer, "incineration")
            return sink, activity_key, False, 0.0, note, "direct_disposal"

        if eol_pathway == "incineration_efw_heat_electricity":
            sink, activity_key, note = classify_direct_disposal_sink(polymer, "incineration")
            return sink, activity_key, False, 0.0, note + " EfW route adds separate heat/electricity credit rows.", "efw_disposal"

        if eol_pathway == "landfill_all":
            sink, activity_key, note = classify_direct_disposal_sink(polymer, "landfill")
            return sink, activity_key, False, 0.0, note, "direct_disposal"

        if eol_pathway == "mechanical_recycling_main_residual_incineration":
            if polymer in LOW_DENSITY_PE_RECYCLING_POLYMERS:
                recovery = RECYCLING_RECOVERY_FRACTIONS_BY_POLYMER[polymer]
                return (
                    "mechanical_recycling_low_density_pe_to_low_density_polyethylene_pellets",
                    "mechanical_recycling_low_density_pe_packaging_recycling",
                    True,
                    recovery,
                    f"{polymer} routed to ecoinvent 3.12 treatment of waste polyethylene, low density, packaging, recycling, producing recycled low-density polyethylene pellets.",
                    "low_density_PE_recycling_3_12_packaging_route",
                )
            if polymer == "PP":
                recovery = RECYCLING_RECOVERY_FRACTIONS_BY_POLYMER[polymer]
                return (
                    "mechanical_recycling_pp_to_polypropylene_pellets",
                    "mechanical_recycling_pp_recycling",
                    True,
                    recovery,
                    "PP routed to ecoinvent 3.12 treatment of waste polypropylene, recycling, producing recycled polypropylene pellets.",
                    "PP_recycling_3_12_direct_route",
                )
            if polymer == "TPE":
                recovery = RECYCLING_RECOVERY_FRACTIONS_BY_POLYMER[polymer]
                return (
                    "mechanical_recycling_tpe_mixed_plastic_recycling",
                    "mechanical_recycling_mixed_plastic_recycling",
                    True,
                    recovery,
                    "TPE routed to ecoinvent 3.12 mixed-plastic recycling as a mixed-plastic recycling proxy.",
                    "TPE_recycling_3_12_mixed_plastic_proxy",
                )
            return (
                "residual_incineration_mixed_plastic",
                "incineration_mixed_plastic",
                False,
                0.0,
                "EVOH/Other/HDPE/unresolved polymer routed to residual mixed-plastic incineration in the main recycling route. HDPE is not present in the current inventory and has no active recycling mapping in this route.",
                "residual_disposal",
            )

        raise ValueError(f"Unknown EOL pathway: {eol_pathway}")


    eol_assignment_parts = []

    for eol_pathway in EOL_PATHWAYS:
        tmp = sut_inventory_lca.copy()
        tmp["eol_pathway"] = eol_pathway

        classified = tmp[POLYMER_COL].apply(lambda polymer: classify_eol_sink(polymer, eol_pathway))
        tmp["eol_sink"] = classified.apply(lambda x: x[0])
        tmp["activity_key"] = classified.apply(lambda x: x[1])
        tmp["mechanically_recycled"] = classified.apply(lambda x: x[2])
        tmp["recycling_recovery_fraction"] = classified.apply(lambda x: x[3])
        tmp["virgin_material_substitution_ratio"] = np.where(
            tmp["mechanically_recycled"],
            VIRGIN_MATERIAL_SUBSTITUTION_RATIO,
            0.0,
        )
        tmp["routing_note"] = classified.apply(lambda x: x[4])
        tmp["mapping_assumption_level"] = classified.apply(lambda x: x[5])
        eol_assignment_parts.append(tmp)

    eol_pathway_assignments = pd.concat(eol_assignment_parts, ignore_index=True)

    # Factor rows are keyed by displayed eol_sink but use activity_key to fetch
    # the actual Brightway activity. Recycling activity demand is scaled by the
    # recovery fraction so impact_per_kg remains per kg incoming SUT plastic waste.
    eol_factor_key_rows = (
        eol_pathway_assignments[
            [
                "eol_pathway",
                "eol_sink",
                "activity_key",
                POLYMER_COL,
                "recycling_recovery_fraction",
                "mapping_assumption_level",
            ]
        ]
        .drop_duplicates()
        .rename(columns={POLYMER_COL: "polymer"})
    )

    eol_factor_rows = []

    for _, factor_row in eol_factor_key_rows.iterrows():
        activity = get_explicit_eol_activity(factor_row["activity_key"])
        activity_demand_amount = get_eol_activity_demand_amount(
            factor_row["activity_key"],
            recycling_recovery_fraction=factor_row["recycling_recovery_fraction"],
        )
        metadata = activity_metadata(
            activity=activity,
            mapping_type="fixed_reviewed_3_12_activity",
            activity_search=factor_row["activity_key"],
        )

        for impact_category, method in LCIA_METHODS.items():
            lca = bc.LCA({activity: activity_demand_amount}, method)
            lca.lci()
            lca.lcia()

            eol_factor_rows.append(
                {
                    "eol_pathway": factor_row["eol_pathway"],
                    "eol_sink": factor_row["eol_sink"],
                    "polymer": factor_row["polymer"],
                    "life_cycle_stage": "plastic_end_of_life",
                    "impact_category": impact_category,
                    "impact_per_kg": float(lca.score),
                    "activity_key": factor_row["activity_key"],
                    "activity_demand_amount": activity_demand_amount,
                    "activity_demand_basis": EOL_ACTIVITY_SPECS[factor_row["activity_key"]]["activity_demand_basis"],
                    "demand_orientation_note": EOL_ACTIVITY_SPECS[factor_row["activity_key"]]["demand_orientation_note"],
                    "recycling_recovery_fraction": factor_row["recycling_recovery_fraction"],
                    "mapping_assumption_level": factor_row["mapping_assumption_level"],
                    **metadata,
                }
            )

    eol_factors = pd.DataFrame(eol_factor_rows)

    # Main pathway-to-polymer mapping table for reporting/export.
    polymer_route_mapping_table = (
        eol_pathway_assignments[
            [
                "eol_pathway",
                POLYMER_COL,
                "eol_sink",
                "activity_key",
                "mechanically_recycled",
                "recycling_recovery_fraction",
                "virgin_material_substitution_ratio",
                "mapping_assumption_level",
                "routing_note",
            ]
        ]
        .drop_duplicates()
        .rename(columns={POLYMER_COL: "polymer"})
        .merge(
            eol_mapped_activities[
                [
                    "activity_key",
                    "activity_name",
                    "reference_product",
                    "location",
                    "unit",
                    "database",
                    "code",
                    "activity_demand_amount",
                    "activity_demand_basis",
                    "demand_orientation_note",
                ]
            ],
            on="activity_key",
            how="left",
        )
        .sort_values(["eol_pathway", "polymer", "eol_sink"])
        .reset_index(drop=True)
    )

    route_content_summary = (
        eol_pathway_assignments
        .groupby(
            [
                "eol_pathway",
                "eol_sink",
                "activity_key",
                POLYMER_COL,
                "mechanically_recycled",
                "mapping_assumption_level",
                "routing_note",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(polymer_kg_per_batch=(MASS_COL, "sum"))
        .rename(columns={POLYMER_COL: "polymer"})
        .sort_values(["eol_pathway", "eol_sink", "polymer_kg_per_batch"], ascending=[True, True, False])
    )
    route_content_summary["share_of_pathway_mass"] = route_content_summary.groupby("eol_pathway")["polymer_kg_per_batch"].transform(
        lambda x: x / x.sum() if x.sum() else 0
    )

    print("Mapped EOL activities:")
    display(eol_mapped_activities)

    print("Polymer-to-route mapping table:")
    display(polymer_route_mapping_table)

    print("Route content summary:")
    display(route_content_summary)

    print(f"EOL factor rows: {len(eol_factors):,}")
    display(eol_factors.head(50))


    # ---------------------------------------------------------------------
    # EfW explicit energy recovery assumptions
    # ---------------------------------------------------------------------
    # EfW route structure:
    #   corrected incineration burden
    #   - avoided electricity production
    #   - avoided industrial heat production
    #
    # Base case uses UK-average exported EfW energy, not theoretical full heat recovery.
    # Values are easy to change in the parameter block below.

    EFW_ELECTRICITY_KWH_PER_KG = 0.600   # kWh electricity exported per kg waste source: https://static1.squarespace.com/static/5eabdfbe067c0b62fc7b58b7/t/6800cd36c2408b1f7ad8f645/1744883006223/uk-energy-from-waste-statistics-2024-4812.pdf
    EFW_HEAT_MJ_PER_KG = 0.4176            # MJ useful heat exported per kg waste

    EFW_SOURCE_BASIS = (
        "UK-average EfW export basis from Tolvik UK EfW statistics. "
        "Electricity uses approximately 0.600 kWh/kg waste. "
        "Heat uses approximately 0.4176 MJ/kg waste, based on UK useful heat export values. "
        "Use as base case; sensitivity can test high heat-offtake cases."
    )

    EFW_ASSUMPTION_NOTE = (
        "EfW credits are applied as avoided UK medium-voltage electricity and avoided "
        "industrial/district natural-gas heat. Incineration burden remains the corrected "
        "ecoinvent municipal incineration treatment burden."
    )

    EFW_POLYMERS = ["LLDPE", "LDPE", "PP", "TPE", "EVOH", "Other"]

    EFW_ELECTRICITY_KWH_PER_KG = float(settings["plastics"].get("efw_electricity_kwh_per_kg", EFW_ELECTRICITY_KWH_PER_KG))
    EFW_HEAT_MJ_PER_KG = float(settings["plastics"].get("efw_heat_mj_per_kg", EFW_HEAT_MJ_PER_KG))

    EFW_ENERGY_RECOVERY_ASSUMPTIONS = pd.DataFrame(
        [
            {
                "polymer": polymer,
                "electricity_kwh_per_kg": EFW_ELECTRICITY_KWH_PER_KG,
                "heat_mj_per_kg": EFW_HEAT_MJ_PER_KG,
                "electricity_credit_mapping_key": "avoided_electricity_medium_voltage_GB",
                "heat_credit_mapping_key": "avoided_heat_district_industrial_natural_gas_GLO",
                "source_basis": EFW_SOURCE_BASIS,
                "assumption_note": EFW_ASSUMPTION_NOTE,
            }
            for polymer in EFW_POLYMERS
        ]
    )

    display(EFW_ENERGY_RECOVERY_ASSUMPTIONS)

    # ---------------------------------------------------------------------
    # EOL mapping validation checks
    # ---------------------------------------------------------------------

    required_pathways = set(EOL_PATHWAYS)
    actual_pathways = set(eol_pathway_assignments["eol_pathway"].dropna().unique())

    if required_pathways != actual_pathways:
        raise ValueError(f"Pathway mismatch. Expected {required_pathways}, found {actual_pathways}")

    removed_pathways_present = actual_pathways.intersection(set(REMOVED_EOL_PATHWAYS))
    if removed_pathways_present:
        raise ValueError(f"Removed/obsolete recycling pathways are still present: {removed_pathways_present}")

    # Every assignment must have a factor for every selected LCIA method.
    required_factor_keys = (
        eol_pathway_assignments[
            ["eol_pathway", "eol_sink", POLYMER_COL]
        ]
        .drop_duplicates()
        .rename(columns={POLYMER_COL: "polymer"})
        .merge(
            pd.DataFrame({"impact_category": list(LCIA_METHODS.keys())}),
            how="cross",
        )
    )

    factor_presence_check = required_factor_keys.merge(
        eol_factors[["eol_pathway", "eol_sink", "polymer", "impact_category", "impact_per_kg"]],
        on=["eol_pathway", "eol_sink", "polymer", "impact_category"],
        how="left",
    )

    missing_assignment_factors = factor_presence_check[factor_presence_check["impact_per_kg"].isna()]
    if len(missing_assignment_factors) > 0:
        display(missing_assignment_factors)
        raise ValueError("At least one EOL assignment has no LCIA factor.")

    gwp_eol_check_cols = [
        "eol_pathway",
        "eol_sink",
        "polymer",
        "impact_per_kg",
        "activity_key",
        "activity_demand_amount",
        "activity_demand_basis",
        "activity_name",
        "reference_product",
        "location",
        "code",
        "mapping_assumption_level",
    ]
    gwp_eol_check_cols = [col for col in gwp_eol_check_cols if col in eol_factors.columns]

    gwp_eol_check = (
        eol_factors[eol_factors["impact_category"].eq("gwp")][gwp_eol_check_cols]
        .sort_values(["eol_pathway", "eol_sink", "polymer"])
    )

    display(gwp_eol_check)

    # Sign-orientation sanity checks.
    production_gwp_bad = polymer_production_factors[
        polymer_production_factors["impact_category"].eq("gwp")
        & (polymer_production_factors["impact_per_kg"] < 0)
    ]
    if len(production_gwp_bad) > 0:
        display(production_gwp_bad)
        raise ValueError("Virgin polymer production GWP should not be negative.")

    recycling_treatment_gwp = gwp_eol_check[
        gwp_eol_check["activity_key"].astype(str).str.contains("mechanical_recycling", na=False)
    ]
    if len(recycling_treatment_gwp) > 0 and (recycling_treatment_gwp["impact_per_kg"] < 0).any():
        display(recycling_treatment_gwp)
        raise ValueError("Mechanical recycling treatment GWP is negative; check activity demand orientation and recovery scaling.")

    incineration_treatment_gwp = gwp_eol_check[
        gwp_eol_check["activity_key"].astype(str).str.contains("incineration", na=False)
    ]
    if len(incineration_treatment_gwp) > 0 and (incineration_treatment_gwp["impact_per_kg"] < 0).any():
        display(incineration_treatment_gwp)
        raise ValueError("Incineration GWP is negative after corrected waste-treatment orientation.")

    # Ensure the main mechanical recycling pathway routes polymers as intended.
    main_recycling_map = polymer_route_mapping_table[
        polymer_route_mapping_table["eol_pathway"].eq("mechanical_recycling_main_residual_incineration")
    ]

    should_recycle = main_recycling_map[
        main_recycling_map["polymer"].isin(sorted(MAIN_MECHANICAL_RECYCLING_POLYMERS))
    ]
    if len(should_recycle) > 0 and not should_recycle["mechanically_recycled"].all():
        display(should_recycle)
        raise ValueError("At least one intended main-route recycling polymer is not routed to recycling.")

    pp_map = main_recycling_map[main_recycling_map["polymer"].eq("PP")]
    if len(pp_map) > 0 and not pp_map["activity_key"].eq("mechanical_recycling_pp_recycling").all():
        display(pp_map)
        raise ValueError("PP should use the ecoinvent 3.12 treatment of waste polypropylene, recycling activity.")

    tpe_map = main_recycling_map[main_recycling_map["polymer"].eq("TPE")]
    if len(tpe_map) > 0 and not tpe_map["activity_key"].eq("mechanical_recycling_mixed_plastic_recycling").all():
        display(tpe_map)
        raise ValueError("TPE should use the ecoinvent 3.12 mixed-plastic recycling route.")

    low_density_pe_map = main_recycling_map[
        main_recycling_map["polymer"].isin(sorted(LOW_DENSITY_PE_RECYCLING_POLYMERS))
    ]
    if len(low_density_pe_map) > 0 and not low_density_pe_map["activity_key"].eq("mechanical_recycling_low_density_pe_packaging_recycling").all():
        display(low_density_pe_map)
        raise ValueError(
            "LLDPE and LDPE should use the ecoinvent 3.12 treatment of waste polyethylene, "
            "low density, packaging, recycling activity producing recycled low-density polyethylene pellets."
        )

    hdpe_map = main_recycling_map[main_recycling_map["polymer"].eq("HDPE")]
    if len(hdpe_map) > 0 and hdpe_map["mechanically_recycled"].any():
        display(hdpe_map)
        raise ValueError(
            "HDPE is not present in the current inventory and should not be actively recycled "
            "in this final main-route mapping unless a specific HDPE route is added."
        )

    should_residual = main_recycling_map[
        main_recycling_map["polymer"].isin(sorted(RESIDUAL_POLYMERS_FOR_MAIN_RECYCLING))
    ]
    if len(should_residual) > 0 and should_residual["mechanically_recycled"].any():
        display(should_residual)
        raise ValueError("EVOH/Other/unresolved residual polymers should remain residual in the main mechanical recycling pathway.")

    print("EOL mapping validation passed.")


    # ---------------------------------------------------------------------
    # End-of-life impacts - detailed row level expanded by EOL pathway
    # ---------------------------------------------------------------------

    eol_impacts_detailed = eol_pathway_assignments.merge(
        eol_factors,
        left_on=["eol_pathway", "eol_sink", POLYMER_COL],
        right_on=["eol_pathway", "eol_sink", "polymer"],
        how="left",
        suffixes=("", "_factor"),
    )

    missing = eol_impacts_detailed[eol_impacts_detailed["impact_per_kg"].isna()]
    if len(missing) > 0:
        display(missing)
        raise ValueError("Missing EOL factor for at least one detailed EOL pathway row.")

    eol_impacts_detailed["impact_per_batch"] = (
        eol_impacts_detailed[MASS_COL] * eol_impacts_detailed["impact_per_kg"]
    )

    if "polymer_factor" in eol_impacts_detailed.columns:
        eol_impacts_detailed = eol_impacts_detailed.drop(columns=["polymer_factor"])

    print("EOL pathways included:")
    for pathway in sorted(eol_impacts_detailed["eol_pathway"].unique()):
        print(f"- {pathway}")

    display(eol_impacts_detailed)


    # ---------------------------------------------------------------------
    # Avoided energy activity mapping — ecoinvent 3.12
    # ---------------------------------------------------------------------
    # These are hardcoded activity mappings by:
    #   activity name
    #   reference product
    #   location
    #
    # Do not use old 3.10.1 activity codes here because ecoinvent codes can change
    # between imported releases.

    AVOIDED_ENERGY_ACTIVITY_SPECS = {
        "avoided_electricity_medium_voltage_GB": {
            "database": BACKGROUND_DB,
            "activity_name": "market for electricity, medium voltage",
            "reference_product": "electricity, medium voltage",
            "location": "GB",
            "unit": "kilowatt hour",
            "note": (
                "Displaced Great Britain grid electricity, medium voltage. "
                "Used for explicit EfW electricity credit."
            ),
        },
        "avoided_heat_district_industrial_natural_gas_GLO": {
            "database": BACKGROUND_DB,
            "activity_name": "market group for heat, district or industrial, natural gas",
            "reference_product": "heat, district or industrial, natural gas",
            "location": "GLO",
            "unit": "megajoule",
            "note": (
                "Displaced industrial/district heat from natural gas. "
                "Used for explicit EfW heat credit."
            ),
        },
    }


    def get_activity_by_exact_metadata(
        database,
        activity_name,
        reference_product,
        location,
        unit=None,
    ):
        """
        Return one Brightway activity using ecoinvent metadata.

        This is intentionally not a broad search. It is a hardcoded mapping by
        name/reference product/location, with optional unit validation.
        """
        db = bd.Database(database)

        candidates = [
            act for act in db
            if act.get("name") == activity_name
            and act.get("reference product") == reference_product
            and act.get("location") == location
        ]

        if unit is not None:
            candidates = [
                act for act in candidates
                if act.get("unit") == unit
            ]

        if len(candidates) == 0:
            raise ValueError(
                "No ecoinvent activity found for avoided energy mapping.\n"
                f"database: {database}\n"
                f"activity_name: {activity_name}\n"
                f"reference_product: {reference_product}\n"
                f"location: {location}\n"
                f"unit: {unit}"
            )

        if len(candidates) > 1:
            display(pd.DataFrame([
                {
                    "activity_name": act.get("name"),
                    "reference_product": act.get("reference product"),
                    "location": act.get("location"),
                    "unit": act.get("unit"),
                    "database": act.key[0],
                    "code": act.key[1],
                }
                for act in candidates
            ]))
            raise ValueError(
                "Multiple avoided energy activities found. "
                "Add a stricter unit/location filter."
            )

        return candidates[0]


    AVOIDED_ENERGY_ACTIVITY_LOOKUP = {}
    avoided_energy_activity_mapping_rows = []

    for key, spec in AVOIDED_ENERGY_ACTIVITY_SPECS.items():
        activity = get_activity_by_exact_metadata(
            database=spec["database"],
            activity_name=spec["activity_name"],
            reference_product=spec["reference_product"],
            location=spec["location"],
            unit=spec["unit"],
        )

        AVOIDED_ENERGY_ACTIVITY_LOOKUP[key] = activity

        avoided_energy_activity_mapping_rows.append(
            {
                "mapping_key": key,
                "activity_name": activity.get("name"),
                "reference_product": activity.get("reference product"),
                "location": activity.get("location"),
                "unit": activity.get("unit"),
                "database": activity.key[0],
                "code": activity.key[1],
                "mapping_type": "exact_metadata_ecoinvent_3_12",
                "note": spec["note"],
            }
        )

    avoided_energy_activity_mapping_table = pd.DataFrame(avoided_energy_activity_mapping_rows)

    display(avoided_energy_activity_mapping_table)

    # ---------------------------------------------------------------------
    # Avoided material and EfW credit rows
    # ---------------------------------------------------------------------
    # This cell defines the optional credit tables used by the full detailed LCA table.
    # It prevents NameError failures later and keeps the credit logic explicit:
    #   - mechanical recycling receives an avoided virgin material credit;
    #   - EfW receives avoided electricity and heat credits using the mappings above.

    # ---------------------------------------------------------------------
    # Mechanical recycling: avoided virgin material credit
    # ---------------------------------------------------------------------

    recycling_credit_base = eol_pathway_assignments[
        eol_pathway_assignments["mechanically_recycled"].eq(True)
    ].copy()

    recycling_credits_detailed = pd.DataFrame()

    if len(recycling_credit_base) > 0:
        recycling_credits_detailed = recycling_credit_base.merge(
            polymer_production_factors[
                [
                    "polymer",
                    "impact_category",
                    "impact_per_kg",
                    "activity_name",
                    "reference_product",
                    "location",
                    "unit",
                    "database",
                    "code",
                    "mapping_type",
                ]
            ].rename(
                columns={
                    "impact_per_kg": "virgin_production_impact_per_kg",
                    "activity_name": "avoided_activity_name",
                    "reference_product": "avoided_reference_product",
                    "location": "avoided_location",
                    "unit": "avoided_unit",
                    "database": "avoided_database",
                    "code": "avoided_code",
                    "mapping_type": "avoided_mapping_type",
                }
            ),
            left_on=POLYMER_COL,
            right_on="polymer",
            how="left",
        )

        missing_recycling_credit_factors = recycling_credits_detailed[
            recycling_credits_detailed["virgin_production_impact_per_kg"].isna()
        ]
        if len(missing_recycling_credit_factors) > 0:
            display(missing_recycling_credit_factors)
            raise ValueError("Missing virgin production factor for at least one avoided-material recycling credit row.")

        recycling_credits_detailed["life_cycle_stage"] = "avoided_virgin_material_credit"
        recycling_credits_detailed["impact_per_kg"] = (
            -1.0
            * recycling_credits_detailed["recycling_recovery_fraction"]
            * recycling_credits_detailed["virgin_material_substitution_ratio"]
            * recycling_credits_detailed["virgin_production_impact_per_kg"]
        )
        recycling_credits_detailed["impact_per_batch"] = (
            recycling_credits_detailed[MASS_COL]
            * recycling_credits_detailed["impact_per_kg"]
        )

        recycling_credits_detailed["activity_name"] = (
            "avoided virgin material: " + recycling_credits_detailed["avoided_activity_name"].astype(str)
        )
        recycling_credits_detailed["reference_product"] = recycling_credits_detailed["avoided_reference_product"]
        recycling_credits_detailed["location"] = recycling_credits_detailed["avoided_location"]
        recycling_credits_detailed["unit"] = recycling_credits_detailed["avoided_unit"]
        recycling_credits_detailed["database"] = recycling_credits_detailed["avoided_database"]
        recycling_credits_detailed["code"] = recycling_credits_detailed["avoided_code"]
        recycling_credits_detailed["mapping_type"] = "avoided_virgin_material_credit_from_production_factor"
        recycling_credits_detailed["activity_demand_amount"] = (
            -1.0
            * recycling_credits_detailed["recycling_recovery_fraction"]
            * recycling_credits_detailed["virgin_material_substitution_ratio"]
        )
        recycling_credits_detailed["activity_demand_basis"] = "avoided_kg_virgin_material_per_kg_input_waste"
        recycling_credits_detailed["demand_orientation_note"] = (
            "Negative credit equal to physical recovery yield times the 0.81 quality/substitution ratio times the mapped virgin production factor."
        )
        recycling_credits_detailed["energy_quantity_per_kg_waste"] = np.nan
        recycling_credits_detailed["energy_quantity_unit"] = np.nan

        if "polymer_y" in recycling_credits_detailed.columns:
            recycling_credits_detailed = recycling_credits_detailed.drop(columns=["polymer_y"])
        if "polymer_x" in recycling_credits_detailed.columns:
            recycling_credits_detailed = recycling_credits_detailed.rename(columns={"polymer_x": POLYMER_COL})

    print(f"Recycling credit rows: {len(recycling_credits_detailed):,}")

    # ---------------------------------------------------------------------
    # EfW: avoided electricity and heat credits
    # ---------------------------------------------------------------------

    efw_credit_factor_rows = []
    efw_energy_credit_parts = []

    efw_credit_base = eol_pathway_assignments[
        eol_pathway_assignments["eol_pathway"].eq("incineration_efw_heat_electricity")
    ].copy()

    efw_energy_credits_detailed = pd.DataFrame()

    if len(efw_credit_base) > 0:
        efw_credit_base = efw_credit_base.merge(
            EFW_ENERGY_RECOVERY_ASSUMPTIONS,
            left_on=POLYMER_COL,
            right_on="polymer",
            how="left",
            suffixes=("", "_energy_assumption"),
        )

        missing_efw_assumptions = efw_credit_base[
            efw_credit_base["electricity_kwh_per_kg"].isna()
            | efw_credit_base["heat_mj_per_kg"].isna()
        ]
        if len(missing_efw_assumptions) > 0:
            display(missing_efw_assumptions)
            raise ValueError("Missing EfW energy recovery assumption for at least one polymer.")

        credit_specs = [
            {
                "life_cycle_stage": "efw_electricity_credit",
                "mapping_key_col": "electricity_credit_mapping_key",
                "quantity_col": "electricity_kwh_per_kg",
                "energy_quantity_unit": "kilowatt hour per kg waste",
            },
            {
                "life_cycle_stage": "efw_heat_credit",
                "mapping_key_col": "heat_credit_mapping_key",
                "quantity_col": "heat_mj_per_kg",
                "energy_quantity_unit": "megajoule per kg waste",
            },
        ]

        for credit_spec in credit_specs:
            for mapping_key, rows_for_mapping in efw_credit_base.groupby(credit_spec["mapping_key_col"], dropna=False):
                if mapping_key not in AVOIDED_ENERGY_ACTIVITY_LOOKUP:
                    raise KeyError(f"Missing avoided energy activity mapping for key: {mapping_key}")

                activity = AVOIDED_ENERGY_ACTIVITY_LOOKUP[mapping_key]
                metadata = activity_metadata(
                    activity=activity,
                    mapping_type="efw_avoided_energy_credit_ecoinvent_3_12",
                    activity_search=mapping_key,
                )

                for impact_category, method in LCIA_METHODS.items():
                    # Product-market activity. Negative demand gives avoided-burden credit.
                    lca = bc.LCA({activity: -1.0}, method)
                    lca.lci()
                    lca.lcia()
                    impact_per_energy_unit = float(lca.score)

                    detail = rows_for_mapping.copy()
                    detail["life_cycle_stage"] = credit_spec["life_cycle_stage"]
                    detail["impact_category"] = impact_category
                    detail["energy_quantity_per_kg_waste"] = pd.to_numeric(
                        detail[credit_spec["quantity_col"]],
                        errors="coerce",
                    )
                    detail["energy_quantity_unit"] = credit_spec["energy_quantity_unit"]
                    detail["impact_per_kg"] = (
                        detail["energy_quantity_per_kg_waste"]
                        * impact_per_energy_unit
                    )
                    detail["impact_per_batch"] = detail[MASS_COL] * detail["impact_per_kg"]
                    detail["activity_demand_amount"] = -1.0 * detail["energy_quantity_per_kg_waste"]
                    detail["activity_demand_basis"] = "avoided_energy_per_kg_input_waste"
                    detail["demand_orientation_note"] = (
                        "Negative demand of displaced energy activity per kg waste routed to EfW."
                    )

                    for col, value in metadata.items():
                        detail[col] = value

                    efw_energy_credit_parts.append(detail)

                    efw_credit_factor_rows.append(
                        {
                            "life_cycle_stage": credit_spec["life_cycle_stage"],
                            "mapping_key": mapping_key,
                            "impact_category": impact_category,
                            "impact_per_energy_unit": impact_per_energy_unit,
                            "energy_activity_unit": activity.get("unit"),
                            "energy_quantity_unit": credit_spec["energy_quantity_unit"],
                            **metadata,
                        }
                    )

    if len(efw_energy_credit_parts) > 0:
        efw_energy_credits_detailed = pd.concat(efw_energy_credit_parts, ignore_index=True, sort=False)

    efw_credit_factors = pd.DataFrame(efw_credit_factor_rows)

    print(f"EfW energy credit rows: {len(efw_energy_credits_detailed):,}")
    if len(efw_credit_factors) > 0:
        display(efw_credit_factors)



    # ---------------------------------------------------------------------
    # Autoclave/decontamination electricity before selected SUT EOL routes
    # ---------------------------------------------------------------------
    autoclave_cleaning_detailed = pd.DataFrame()
    autoclave_kwh_per_kg = float(settings["plastics"].get("autoclave_kwh_per_kg", 0.8))
    autoclave_low_kwh_per_kg = float(settings["plastics"].get("autoclave_low_kwh_per_kg", 0.2))
    autoclave_high_kwh_per_kg = float(settings["plastics"].get("autoclave_high_kwh_per_kg", 1.4))
    autoclave_routes = settings["plastics"].get("autoclave_routes")
    if autoclave_routes is None:
        autoclave_routes = [
            settings["plastics"].get(
                "autoclave_route",
                "mechanical_recycling_main_residual_incineration",
            )
        ]
    autoclave_routes = list(dict.fromkeys(str(route) for route in autoclave_routes))
    available_autoclave_routes = set(eol_pathway_assignments["eol_pathway"].astype(str).unique())
    unknown_autoclave_routes = sorted(set(autoclave_routes) - available_autoclave_routes)
    if unknown_autoclave_routes:
        raise ValueError(
            "Unknown plastics.autoclave_routes values: "
            + ", ".join(unknown_autoclave_routes)
        )
    if min(autoclave_low_kwh_per_kg, autoclave_kwh_per_kg, autoclave_high_kwh_per_kg) < 0:
        raise ValueError("Autoclave electricity assumptions must be non-negative.")
    if not (autoclave_low_kwh_per_kg <= autoclave_kwh_per_kg <= autoclave_high_kwh_per_kg):
        raise ValueError(
            "Expected autoclave_low_kwh_per_kg <= autoclave_kwh_per_kg "
            "<= autoclave_high_kwh_per_kg."
        )
    autoclave_mapping_key = settings["plastics"].get(
        "autoclave_activity_mapping_key",
        "avoided_electricity_medium_voltage_GB",
    )
    autoclave_base = eol_pathway_assignments[
        eol_pathway_assignments["eol_pathway"].isin(autoclave_routes)
    ].copy()
    if len(autoclave_base) > 0 and autoclave_kwh_per_kg != 0:
        activity = AVOIDED_ENERGY_ACTIVITY_LOOKUP[autoclave_mapping_key]
        metadata = activity_metadata(activity, "autoclave_electricity_exact_metadata", autoclave_mapping_key)
        pieces = []
        for impact_category, method in LCIA_METHODS.items():
            lca = bc.LCA({activity: 1.0}, method)
            lca.lci(); lca.lcia()
            impact_per_kwh = float(lca.score)
            detail = autoclave_base.copy()
            detail["life_cycle_stage"] = "autoclave_cleaning"
            detail["impact_category"] = impact_category
            detail["impact_per_kg"] = autoclave_kwh_per_kg * impact_per_kwh
            detail["impact_per_batch"] = detail[MASS_COL] * detail["impact_per_kg"]
            detail["activity_demand_amount"] = autoclave_kwh_per_kg
            detail["activity_demand_basis"] = "kWh_autoclave_electricity_per_kg_input_SUT_waste"
            detail["demand_orientation_note"] = (
                "Positive GB medium-voltage electricity demand for SUT "
                "autoclaving/decontamination before the selected end-of-life route."
            )
            detail["eol_sink"] = "autoclave_pre_treatment"
            detail["energy_quantity_per_kg_waste"] = autoclave_kwh_per_kg
            detail["energy_quantity_unit"] = "kilowatt hour per kg waste"
            detail["autoclave_kwh_per_kg"] = autoclave_kwh_per_kg
            detail["autoclave_low_kwh_per_kg"] = autoclave_low_kwh_per_kg
            detail["autoclave_high_kwh_per_kg"] = autoclave_high_kwh_per_kg
            detail["autoclave_impact_per_kg_low"] = autoclave_low_kwh_per_kg * impact_per_kwh
            detail["autoclave_impact_per_kg_high"] = autoclave_high_kwh_per_kg * impact_per_kwh
            for key, value in metadata.items(): detail[key] = value
            pieces.append(detail)
        autoclave_cleaning_detailed = pd.concat(pieces, ignore_index=True, sort=False)


    # ---------------------------------------------------------------------
    # Full detailed SUT plastics LCA table by EOL pathway
    # ---------------------------------------------------------------------

    production_expanded = []

    for eol_pathway in EOL_PATHWAYS:
        tmp = production_impacts_detailed.copy()
        tmp["eol_pathway"] = eol_pathway
        tmp["eol_sink"] = "not_applicable"
        production_expanded.append(tmp)

    production_expanded = pd.concat(production_expanded, ignore_index=True)

    manufacturing_expanded = []
    for eol_pathway in EOL_PATHWAYS:
        tmp = sut_manufacturing_impacts_detailed.copy()
        tmp["eol_pathway"] = eol_pathway
        tmp["eol_sink"] = "not_applicable"
        manufacturing_expanded.append(tmp)
    manufacturing_expanded = pd.concat(manufacturing_expanded, ignore_index=True)

    lca_parts = [production_expanded, manufacturing_expanded, eol_impacts_detailed]
    if len(autoclave_cleaning_detailed) > 0:
        lca_parts.append(autoclave_cleaning_detailed)
    if len(recycling_credits_detailed) > 0:
        lca_parts.append(recycling_credits_detailed)
    if "efw_energy_credits_detailed" in locals() and len(efw_energy_credits_detailed) > 0:
        lca_parts.append(efw_energy_credits_detailed)

    sut_plastics_lca_detailed = pd.concat(
        lca_parts,
        ignore_index=True,
    )

    print(f"EOL pathways included: {sorted(sut_plastics_lca_detailed['eol_pathway'].dropna().unique())}")
    print(f"Detailed full LCA rows: {len(sut_plastics_lca_detailed):,}")

    display(sut_plastics_lca_detailed.head())


    # ---------------------------------------------------------------------
    # Circularity indicators by detailed row and EOL pathway
    # ---------------------------------------------------------------------
    # Circularity is treated as a mass-flow KPI, not a Brightway LCIA method.

    circularity_indicators_detailed = eol_pathway_assignments[
        [
            "eol_pathway",
            "eol_sink",
            "inventory_id",
            PROCESS_AREA_COL,
            SUPERPRO_SECTION_COL,
            UNIT_OPERATION_COL,
            SUT_FLOW_COL,
            SUT_CATEGORY_TYPE_COL,
            POLYMER_COL,
            MASS_COL,
            "mechanically_recycled",
            "recycling_recovery_fraction",
            "virgin_material_substitution_ratio",
            "routing_note",
        ]
    ].copy()

    circularity_indicators_detailed["recovered_material_kg_per_batch"] = (
        circularity_indicators_detailed[MASS_COL]
        * circularity_indicators_detailed["recycling_recovery_fraction"]
    )

    circularity_indicators_detailed["avoided_virgin_material_kg_per_batch"] = (
        circularity_indicators_detailed["recovered_material_kg_per_batch"]
        * circularity_indicators_detailed["virgin_material_substitution_ratio"]
    )

    circularity_indicators_detailed["residual_disposal_kg_per_batch"] = (
        circularity_indicators_detailed[MASS_COL]
        - circularity_indicators_detailed["recovered_material_kg_per_batch"]
    )

    circularity_indicators_detailed["circularity_loss_kg_per_batch"] = (
        circularity_indicators_detailed[MASS_COL]
        - circularity_indicators_detailed["avoided_virgin_material_kg_per_batch"]
    )

    summary_circularity_by_pathway = (
        circularity_indicators_detailed
        .groupby("eol_pathway", as_index=False)
        .agg(
            polymer_kg_per_batch=(MASS_COL, "sum"),
            recovered_material_kg_per_batch=("recovered_material_kg_per_batch", "sum"),
            avoided_virgin_material_kg_per_batch=("avoided_virgin_material_kg_per_batch", "sum"),
            residual_disposal_kg_per_batch=("residual_disposal_kg_per_batch", "sum"),
            circularity_loss_kg_per_batch=("circularity_loss_kg_per_batch", "sum"),
        )
    )

    summary_circularity_by_pathway["circularity_rate"] = (
        summary_circularity_by_pathway["avoided_virgin_material_kg_per_batch"]
        / summary_circularity_by_pathway["polymer_kg_per_batch"]
    )

    # Backwards-compatible name for export/plots.
    summary_circularity_by_sink = summary_circularity_by_pathway.copy()

    display(circularity_indicators_detailed.head())
    display(summary_circularity_by_pathway)


    # ---------------------------------------------------------------------
    # Secondary streams from mechanical recycling route activities
    # ---------------------------------------------------------------------
    # This table contains only quantities that are explicitly available from:
    #   1. direct exchanges exposed by the selected background recycling activities;
    #   2. foreground mass balances calculated in this notebook.
    #
    # No unavailable quantities are created. If water, wastewater, electricity or
    # heat are not directly exposed by the selected background activity, they will
    # not appear as physical secondary outputs here. Their environmental burden
    # remains inside the route LCIA factor and should not be routed again.

    SECONDARY_KEYWORD_GROUPS = {
        "wastewater": ["wastewater", "sewage", "effluent"],
        "water_utility": ["water"],
        "electricity_utility": ["electricity"],
        "heat_or_steam_utility": ["heat", "steam"],
    }

    EXCLUDE_SECONDARY_TEXT = [
        "market group for electricity",
    ]


    def _safe_exchange_input_metadata(exchange):
        """Return metadata for the exchange input activity/flow where available."""
        try:
            input_obj = exchange.input
        except Exception:
            input_obj = None

        if input_obj is None:
            return {
                "exchange_input_name": None,
                "exchange_reference_product": None,
                "exchange_location": None,
                "exchange_database": None,
                "exchange_code": None,
                "exchange_input_unit": exchange.get("unit"),
            }

        return {
            "exchange_input_name": input_obj.get("name"),
            "exchange_reference_product": input_obj.get("reference product"),
            "exchange_location": input_obj.get("location"),
            "exchange_database": input_obj.key[0] if hasattr(input_obj, "key") else None,
            "exchange_code": input_obj.key[1] if hasattr(input_obj, "key") else None,
            "exchange_input_unit": input_obj.get("unit", exchange.get("unit")),
        }


    def classify_secondary_exchange(exchange_name, reference_product):
        """Classify visible water/wastewater/utility exchanges using conservative text rules."""
        text = f"{exchange_name or ''} {reference_product or ''}".lower()

        if any(term in text for term in EXCLUDE_SECONDARY_TEXT):
            return None

        if any(term in text for term in SECONDARY_KEYWORD_GROUPS["wastewater"]):
            return {
                "secondary_stream_category": "wastewater",
                "secondary_stream_name": "visible_recycling_wastewater_or_effluent_exchange",
                "stream_direction": "embedded_treatment_or_output",
                "routable_in_master": False,
            }

        if "water" in text:
            return {
                "secondary_stream_category": "water_utility",
                "secondary_stream_name": "visible_recycling_process_or_utility_water_exchange",
                "stream_direction": "utility_input",
                "routable_in_master": False,
            }

        if "electricity" in text:
            return {
                "secondary_stream_category": "electricity_utility",
                "secondary_stream_name": "visible_recycling_electricity_exchange",
                "stream_direction": "utility_input",
                "routable_in_master": False,
            }

        if "heat" in text or "steam" in text:
            return {
                "secondary_stream_category": "heat_or_steam_utility",
                "secondary_stream_name": "visible_recycling_heat_or_steam_exchange",
                "stream_direction": "utility_input",
                "routable_in_master": False,
            }

        return None


    def exchange_amount_to_parent_kg_factor(amount, exchange_unit, activity_unit):
        """
        Return exchange amount per kg plastic routed.

        The mapped recycling activity should be demanded per kg plastic. If the
        selected activity uses another unit, the raw exchange is still reported but
        flagged for review through quantity_status.
        """
        return float(amount)


    def extract_secondary_exchange_factors(activity, activity_key, eol_pathway, eol_sink, polymer):
        """Extract visible secondary exchange factors from one mapped recycling activity."""
        rows = []

        for exchange in activity.exchanges():
            meta = _safe_exchange_input_metadata(exchange)
            classification = classify_secondary_exchange(
                meta.get("exchange_input_name"),
                meta.get("exchange_reference_product"),
            )
            if classification is None:
                continue

            amount = exchange.get("amount")
            if amount is None:
                continue

            exchange_unit = meta.get("exchange_input_unit")
            activity_unit = activity.get("unit")
            quantity_per_parent_unit = exchange_amount_to_parent_kg_factor(
                amount=amount,
                exchange_unit=exchange_unit,
                activity_unit=activity_unit,
            )

            rows.append({
                "source_module": "sut_plastics",
                "parent_stream_category": "sut_plastic_waste",
                "parent_route": eol_pathway,
                "parent_eol_sink": eol_sink,
                "parent_polymer": polymer,
                "activity_key": activity_key,
                "activity_name": activity.get("name"),
                "activity_reference_product": activity.get("reference product"),
                "activity_location": activity.get("location"),
                "activity_unit": activity_unit,
                "exchange_type": exchange.get("type"),
                "exchange_amount_per_activity_unit": float(amount),
                "exchange_unit": exchange_unit,
                "quantity_per_parent_unit": quantity_per_parent_unit,
                "parent_unit": "kg plastic routed to this activity",
                "quantity_unit": exchange_unit,
                "embedded_in_lca_activity": True,
                "export_to_wastewater_pool": False,
                "routable_in_master": classification.get("routable_in_master", False),
                "source_of_quantity": "direct_background_activity_exchange",
                "quantity_status": "exposed_by_selected_background_activity",
                "assumption_note": (
                    "Visible direct exchange from the mapped background recycling/treatment activity. "
                    "It is reported for traceability only and remains embedded in the route LCIA factor. "
                    "Do not add it again to wastewater or utility LCIA unless recycling is rebuilt as a foreground process."
                ),
                **classification,
                **meta,
            })

        return rows


    # ---------------------------------------------------------------------
    # Direct utility/wastewater exchanges visible in selected background activities
    # ---------------------------------------------------------------------

    recycling_secondary_factor_rows = []

    recycling_factor_lookup = (
        eol_pathway_assignments.loc[
            eol_pathway_assignments["mechanically_recycled"].eq(True),
            ["eol_pathway", "eol_sink", POLYMER_COL, "activity_key"],
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    for _, row in recycling_factor_lookup.iterrows():
        activity = get_explicit_eol_activity(row["activity_key"])
        recycling_secondary_factor_rows.extend(
            extract_secondary_exchange_factors(
                activity=activity,
                activity_key=row["activity_key"],
                eol_pathway=row["eol_pathway"],
                eol_sink=row["eol_sink"],
                polymer=row[POLYMER_COL],
            )
        )

    recycling_secondary_factors = pd.DataFrame(recycling_secondary_factor_rows)

    if len(recycling_secondary_factors) > 0:
        recycling_secondary_streams = eol_pathway_assignments[
            eol_pathway_assignments["mechanically_recycled"].eq(True)
        ].merge(
            recycling_secondary_factors,
            left_on=["eol_pathway", "eol_sink", POLYMER_COL, "activity_key"],
            right_on=["parent_route", "parent_eol_sink", "parent_polymer", "activity_key"],
            how="inner",
        )

        recycling_secondary_streams["quantity_per_batch"] = (
            recycling_secondary_streams[MASS_COL]
            * recycling_secondary_streams["quantity_per_parent_unit"]
        )
    else:
        recycling_secondary_streams = pd.DataFrame()

    # ---------------------------------------------------------------------
    # Foreground recovered-material and residual-disposal streams
    # ---------------------------------------------------------------------

    material_secondary_streams = circularity_indicators_detailed.copy()
    material_secondary_streams["source_module"] = "sut_plastics"
    material_secondary_streams["parent_stream_category"] = "sut_plastic_waste"
    material_secondary_streams["parent_route"] = material_secondary_streams["eol_pathway"]
    material_secondary_streams["parent_eol_sink"] = material_secondary_streams["eol_sink"]
    material_secondary_streams["parent_polymer"] = material_secondary_streams[POLYMER_COL]
    material_secondary_streams["parent_unit"] = "kg plastic waste"
    material_secondary_streams["embedded_in_lca_activity"] = False
    material_secondary_streams["source_of_quantity"] = "foreground_mass_balance"
    material_secondary_streams["quantity_status"] = "calculated_foreground_mass_balance"

    recovered_streams = material_secondary_streams[
        material_secondary_streams["recovered_material_kg_per_batch"].gt(0)
    ].copy()
    recovered_streams["secondary_stream_category"] = "recovered_plastic_material"
    recovered_streams["secondary_stream_name"] = "recovered_downcycled_polymer"
    recovered_streams["stream_direction"] = "secondary_product_output"
    recovered_streams["quantity_per_batch"] = recovered_streams["recovered_material_kg_per_batch"]
    recovered_streams["quantity_unit"] = "kg"
    recovered_streams["quantity_per_parent_unit"] = recovered_streams["recycling_recovery_fraction"]
    recovered_streams["export_to_wastewater_pool"] = False
    recovered_streams["routable_in_master"] = True
    recovered_streams["assumption_note"] = "Recovered material output from mechanical recycling mass balance."

    residual_streams = material_secondary_streams[
        material_secondary_streams["residual_disposal_kg_per_batch"].gt(0)
        & material_secondary_streams["mechanically_recycled"].eq(True)
    ].copy()
    residual_streams["secondary_stream_category"] = "residual_plastic_waste"
    residual_streams["secondary_stream_name"] = "recycling_residual_to_disposal"
    residual_streams["stream_direction"] = "residual_output"
    residual_streams["quantity_per_batch"] = residual_streams["residual_disposal_kg_per_batch"]
    residual_streams["quantity_unit"] = "kg"
    residual_streams["quantity_per_parent_unit"] = 1.0 - residual_streams["recycling_recovery_fraction"]
    residual_streams["export_to_wastewater_pool"] = False
    residual_streams["routable_in_master"] = True
    residual_streams["assumption_note"] = (
        "Residual material from mechanical recycling mass balance. Route separately only if residual disposal is explicitly modelled outside the recycling route factor."
    )

    secondary_streams_database = pd.concat(
        [recycling_secondary_streams, recovered_streams, residual_streams],
        ignore_index=True,
        sort=False,
    )

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
        PROCESS_AREA_COL,
        SUPERPRO_SECTION_COL,
        UNIT_OPERATION_COL,
        SUT_FLOW_COL,
        SUT_CATEGORY_TYPE_COL,
        POLYMER_COL,
        MASS_COL,
        "quantity_per_parent_unit",
        "parent_unit",
        "quantity_per_batch",
        "quantity_unit",
        "embedded_in_lca_activity",
        "export_to_wastewater_pool",
        "routable_in_master",
        "source_of_quantity",
        "quantity_status",
        "assumption_note",
        "activity_key",
        "activity_name",
        "activity_reference_product",
        "activity_location",
        "activity_unit",
        "exchange_type",
        "exchange_amount_per_activity_unit",
        "exchange_unit",
        "exchange_input_name",
        "exchange_reference_product",
        "exchange_location",
        "exchange_database",
        "exchange_code",
    ]
    secondary_output_cols = [c for c in secondary_output_cols if c in secondary_streams_database.columns]
    secondary_streams_database = secondary_streams_database[secondary_output_cols].copy()

    print(f"Secondary stream rows: {len(secondary_streams_database):,}")
    display(secondary_streams_database.head(50))

    if len(secondary_streams_database) > 0:
        display(
            secondary_streams_database
            .groupby(["parent_route", "secondary_stream_category", "quantity_status"], dropna=False, as_index=False)
            .agg(
                rows=("secondary_stream_category", "count"),
                quantity_per_batch=("quantity_per_batch", "sum"),
            )
        )

    if not (
        len(secondary_streams_database) > 0
        and secondary_streams_database["secondary_stream_category"].isin(["water_utility", "wastewater"]).any()
    ):
        print(
            "No explicit recycling water or wastewater quantities were exposed by the selected background recycling activities. "
            "These burdens remain embedded in the LCIA route factors and are not exported as routable physical streams."
        )


    # ---------------------------------------------------------------------
    # Detailed summaries for downstream aggregation
    # ---------------------------------------------------------------------

    full_detail_group_cols = [
        "eol_pathway",
        "eol_sink",
        "life_cycle_stage",
        "impact_category",
        "inventory_id",
        PROCESS_AREA_COL,
        SUPERPRO_SECTION_COL,
        UNIT_OPERATION_COL,
        SUT_FLOW_COL,
        SUT_CATEGORY_TYPE_COL,
        POLYMER_COL,
    ]

    summary_full_detail = (
        sut_plastics_lca_detailed
        .groupby(full_detail_group_cols, dropna=False, as_index=False)
        .agg(
            polymer_kg_per_batch=(MASS_COL, "sum"),
            impact_per_batch=("impact_per_batch", "sum"),
        )
    )

    summary_by_sink_stage_category = (
        sut_plastics_lca_detailed
        .groupby(["eol_pathway", "eol_sink", "life_cycle_stage", "impact_category"], dropna=False, as_index=False)
        .agg(
            polymer_kg_per_batch=(MASS_COL, "sum"),
            impact_per_batch=("impact_per_batch", "sum"),
        )
    )

    summary_by_process_area = (
        sut_plastics_lca_detailed
        .groupby(["eol_pathway", "impact_category", PROCESS_AREA_COL, "life_cycle_stage"], dropna=False, as_index=False)
        .agg(impact_per_batch=("impact_per_batch", "sum"))
    )

    summary_by_sut_type = (
        sut_plastics_lca_detailed
        .groupby(["eol_pathway", "impact_category", SUT_CATEGORY_TYPE_COL, "life_cycle_stage"], dropna=False, as_index=False)
        .agg(impact_per_batch=("impact_per_batch", "sum"))
    )

    summary_by_polymer = (
        sut_plastics_lca_detailed
        .groupby(["eol_pathway", "impact_category", POLYMER_COL, "life_cycle_stage"], dropna=False, as_index=False)
        .agg(impact_per_batch=("impact_per_batch", "sum"))
    )

    summary_by_eol_pathway = (
        sut_plastics_lca_detailed
        .groupby(["eol_pathway", "impact_category"], dropna=False, as_index=False)
        .agg(impact_per_batch=("impact_per_batch", "sum"))
    )

    display(summary_full_detail.head())
    display(summary_by_sink_stage_category)
    display(summary_by_eol_pathway)


    base_cols = [
        "inventory_id",
        PROCESS_AREA_COL,
        SUPERPRO_SECTION_COL,
        UNIT_OPERATION_COL,
        SUT_FLOW_COL,
        SUT_CATEGORY_TYPE_COL,
        POLYMER_COL,
        MASS_COL,
        "eol_pathway",
        "eol_sink",
        "life_cycle_stage",
        "impact_category",
        "impact_per_kg",
        "impact_per_batch",
        "activity_name",
        "reference_product",
        "location",
        "unit",
        "database",
        "code",
        "mapping_type",
        "activity_demand_amount",
        "activity_demand_basis",
        "demand_orientation_note",
        "mapping_assumption_level",
        "energy_quantity_per_kg_waste",
        "energy_quantity_unit",
    ]

    available_base_cols = [col for col in base_cols if col in sut_plastics_lca_detailed.columns]
    optimisation_lca_database = sut_plastics_lca_detailed[available_base_cols].copy()

    circularity_cols = [
        "eol_pathway",
        "eol_sink",
        "inventory_id",
        PROCESS_AREA_COL,
        SUPERPRO_SECTION_COL,
        UNIT_OPERATION_COL,
        SUT_FLOW_COL,
        SUT_CATEGORY_TYPE_COL,
        POLYMER_COL,
        "mechanically_recycled",
        "recycling_recovery_fraction",
        "virgin_material_substitution_ratio",
        "recovered_material_kg_per_batch",
        "avoided_virgin_material_kg_per_batch",
        "residual_disposal_kg_per_batch",
        "circularity_loss_kg_per_batch",
        "routing_note",
    ]
    available_circularity_cols = [col for col in circularity_cols if col in circularity_indicators_detailed.columns]
    circularity_for_join = circularity_indicators_detailed[available_circularity_cols].drop_duplicates().copy()

    join_cols = [
        "eol_pathway",
        "eol_sink",
        "inventory_id",
        PROCESS_AREA_COL,
        SUPERPRO_SECTION_COL,
        UNIT_OPERATION_COL,
        SUT_FLOW_COL,
        SUT_CATEGORY_TYPE_COL,
        POLYMER_COL,
    ]
    join_cols = [
        col for col in join_cols
        if col in optimisation_lca_database.columns and col in circularity_for_join.columns
    ]

    optimisation_lca_database = optimisation_lca_database.merge(
        circularity_for_join,
        on=join_cols,
        how="left",
    )

    optimisation_lca_database["source_module"] = "sut_plastics"
    optimisation_lca_database["stream_category"] = "sut_plastic_waste"
    optimisation_lca_database["functional_unit"] = "per_batch"
    optimisation_lca_database["model_scope"] = "SUT_plastics_only"
    optimisation_lca_database["ecoinvent_version"] = ECOINVENT_VERSION
    optimisation_lca_database["ecoinvent_system_model"] = ECOINVENT_SYSTEM_MODEL
    optimisation_lca_database["brightway_project"] = BRIGHTWAY_PROJECT
    optimisation_lca_database["includes_packaging"] = True
    optimisation_lca_database["packaging_uplift_rate"] = 0.0
    optimisation_lca_database["includes_consumable_conversion"] = True
    optimisation_lca_database["sut_manufacturing_gwp_adder_kgco2e_per_kg"] = (
        SUT_MANUFACTURING_GWP_ADDER
    )
    optimisation_lca_database["includes_transport"] = False
    optimisation_lca_database["secondary_streams_embedded_in_activity"] = (
        optimisation_lca_database["eol_sink"].fillna("").str.contains("mechanical_recycling", case=False)
    )

    optimisation_lca_database["is_credit"] = optimisation_lca_database["life_cycle_stage"].isin([
        "avoided_virgin_material_credit",
        "efw_heat_credit",
        "efw_electricity_credit",
    ])
    optimisation_lca_database["is_eol_treatment"] = optimisation_lca_database["life_cycle_stage"].isin(["plastic_end_of_life", "autoclave_cleaning"])
    optimisation_lca_database["is_production"] = optimisation_lca_database[
        "life_cycle_stage"
    ].isin(["virgin_polymer_production", "sut_component_manufacturing"])

    zero_fill_cols = [
        "recycling_recovery_fraction",
        "virgin_material_substitution_ratio",
        "recovered_material_kg_per_batch",
        "avoided_virgin_material_kg_per_batch",
        "residual_disposal_kg_per_batch",
        "circularity_loss_kg_per_batch",
    ]
    for col in zero_fill_cols:
        if col in optimisation_lca_database.columns:
            optimisation_lca_database[col] = optimisation_lca_database[col].fillna(0.0)

    if "mechanically_recycled" in optimisation_lca_database.columns:
        optimisation_lca_database["mechanically_recycled"] = optimisation_lca_database["mechanically_recycled"].fillna(False).astype(bool)

    IMPACT_UNITS = {
        "gwp": "kg CO2-eq",
        "fossil_resource_use": "kg oil-eq",
        "water_scarcity": "water scarcity footprint",
        "human_health_daly": "DALY",
        "ecosystem_quality": "species.yr",
    }
    optimisation_lca_database["impact_unit"] = optimisation_lca_database["impact_category"].map(IMPACT_UNITS).fillna("impact units")
    optimisation_lca_database["quantity_unit"] = "kg plastic"
    optimisation_lca_database["impact_factor_unit"] = optimisation_lca_database["impact_unit"] + " per kg plastic"

    preferred_first_cols = [
        "source_module",
        "stream_category",
        "functional_unit",
        "ecoinvent_version",
        "ecoinvent_system_model",
        "brightway_project",
        "inventory_id",
        PROCESS_AREA_COL,
        SUPERPRO_SECTION_COL,
        UNIT_OPERATION_COL,
        SUT_FLOW_COL,
        SUT_CATEGORY_TYPE_COL,
        POLYMER_COL,
        MASS_COL,
        "quantity_unit",
        "eol_pathway",
        "eol_sink",
        "life_cycle_stage",
        "impact_category",
        "impact_per_kg",
        "impact_factor_unit",
        "impact_per_batch",
        "impact_unit",
        "is_production",
        "is_eol_treatment",
        "is_credit",
        "mechanically_recycled",
        "recycling_recovery_fraction",
        "virgin_material_substitution_ratio",
        "recovered_material_kg_per_batch",
        "avoided_virgin_material_kg_per_batch",
        "residual_disposal_kg_per_batch",
        "circularity_loss_kg_per_batch",
        "secondary_streams_embedded_in_activity",
    ]
    preferred_first_cols = [c for c in preferred_first_cols if c in optimisation_lca_database.columns]
    remaining_cols = [c for c in optimisation_lca_database.columns if c not in preferred_first_cols]
    optimisation_lca_database = optimisation_lca_database[preferred_first_cols + remaining_cols].copy()

    # Compact route-factor table. The optimiser should use this for route choices.
    # Fixed production rows and route-dependent EOL/credit rows are kept separate
    # to avoid counting virgin production once per alternative route.
    route_factor_group_cols = [
        "source_module",
        "stream_category",
        POLYMER_COL,
        "eol_pathway",
        "eol_sink",
        "life_cycle_stage",
        "impact_category",
        "impact_unit",
        "quantity_unit",
        "impact_factor_unit",
        "is_production",
        "is_eol_treatment",
        "is_credit",
        "mechanically_recycled",
        "recycling_recovery_fraction",
        "virgin_material_substitution_ratio",
        "secondary_streams_embedded_in_activity",
        "activity_name",
        "reference_product",
        "location",
        "unit",
        "database",
        "code",
        "mapping_type",
        "activity_demand_amount",
        "activity_demand_basis",
        "demand_orientation_note",
        "mapping_assumption_level",
        "energy_quantity_per_kg_waste",
        "energy_quantity_unit",
    ]
    route_factor_group_cols = [col for col in route_factor_group_cols if col in optimisation_lca_database.columns]

    route_factor_table = (
        optimisation_lca_database
        .groupby(route_factor_group_cols, dropna=False, as_index=False)
        .agg(
            impact_per_kg_mean=("impact_per_kg", "mean"),
            min_impact_per_kg=("impact_per_kg", "min"),
            max_impact_per_kg=("impact_per_kg", "max"),
            contributing_inventory_rows=("inventory_id", "nunique"),
            represented_mass_kg_per_batch=(MASS_COL, "sum"),
            represented_impact_per_batch=("impact_per_batch", "sum"),
        )
    )

    route_factor_table["impact_factor_variability_abs"] = route_factor_table["max_impact_per_kg"] - route_factor_table["min_impact_per_kg"]
    route_factor_table["route_factor_status"] = np.where(
        route_factor_table["impact_factor_variability_abs"].abs() <= 1e-9,
        "single_consistent_factor",
        "mass_weighted_or_mixed_factor_check",
    )

    weighted_rows = []
    for keys, group in optimisation_lca_database.groupby(route_factor_group_cols, dropna=False):
        group = group.copy()
        mass_sum = group[MASS_COL].sum()
        if mass_sum > 0:
            weighted_factor = (group["impact_per_kg"] * group[MASS_COL]).sum() / mass_sum
        else:
            weighted_factor = group["impact_per_kg"].mean()
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(route_factor_group_cols, keys))
        row["impact_per_kg_for_optimisation"] = weighted_factor
        weighted_rows.append(row)

    weighted_factors = pd.DataFrame(weighted_rows)
    route_factor_table = route_factor_table.merge(weighted_factors, on=route_factor_group_cols, how="left")

    route_preferred_cols = [
        "source_module",
        "stream_category",
        POLYMER_COL,
        "eol_pathway",
        "eol_sink",
        "life_cycle_stage",
        "impact_category",
        "impact_per_kg_for_optimisation",
        "impact_factor_unit",
        "is_production",
        "is_eol_treatment",
        "is_credit",
        "mechanically_recycled",
        "recycling_recovery_fraction",
        "virgin_material_substitution_ratio",
        "secondary_streams_embedded_in_activity",
        "route_factor_status",
        "represented_mass_kg_per_batch",
        "represented_impact_per_batch",
        "contributing_inventory_rows",
        "activity_name",
        "reference_product",
        "location",
        "unit",
        "database",
        "code",
        "mapping_type",
        "activity_demand_amount",
        "activity_demand_basis",
        "demand_orientation_note",
        "mapping_assumption_level",
        "energy_quantity_per_kg_waste",
        "energy_quantity_unit",
    ]
    route_preferred_cols = [c for c in route_preferred_cols if c in route_factor_table.columns]
    route_remaining_cols = [c for c in route_factor_table.columns if c not in route_preferred_cols]
    route_factor_table = route_factor_table[route_preferred_cols + route_remaining_cols].copy()

    if "secondary_streams_database" not in locals():
        secondary_streams_database = pd.DataFrame()

    # Validation checks
    grain_cols = ["inventory_id", "eol_pathway", "eol_sink", "life_cycle_stage", "impact_category", POLYMER_COL]
    duplicate_check_cols = [col for col in grain_cols if col in optimisation_lca_database.columns]
    duplicate_rows = optimisation_lca_database[
        optimisation_lca_database.duplicated(subset=duplicate_check_cols, keep=False)
    ]
    if len(duplicate_rows) > 0:
        print("Warning: duplicate rows found at nominal detailed optimisation grain.")
        display(duplicate_rows.head(20))

    missing_impacts = optimisation_lca_database[optimisation_lca_database["impact_per_batch"].isna()]
    if len(missing_impacts) > 0:
        display(missing_impacts)
        raise ValueError("Missing impact_per_batch values in optimisation_lca_database.")

    missing_route_factors = route_factor_table[route_factor_table["impact_per_kg_for_optimisation"].isna()]
    if len(missing_route_factors) > 0:
        display(missing_route_factors)
        raise ValueError("Missing impact_per_kg_for_optimisation values in route_factor_table.")

    print(f"Detailed LCA rows: {len(optimisation_lca_database):,}")
    print(f"Route-factor rows: {len(route_factor_table):,}")
    print(f"Secondary output rows: {len(secondary_streams_database):,}")
    display(optimisation_lca_database.head())
    display(route_factor_table.head())
    display(secondary_streams_database.head())
    return {
        "main_detailed_table": optimisation_lca_database,
        "route_factor_table": route_factor_table,
        "secondary_outputs": secondary_streams_database,
        "polymer_route_mapping": polymer_route_mapping_table,
        "route_content_summary": route_content_summary,
        "circularity_summary": summary_circularity_by_pathway,
        "circularity_detailed": circularity_indicators_detailed,
        "efw_mapping": avoided_energy_activity_mapping_table,
        "autoclave_detail": autoclave_cleaning_detailed,
    }
