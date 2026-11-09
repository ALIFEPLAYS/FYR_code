from __future__ import annotations

# POLICY-ONLY REPORTING BUILD: EOL_LABEL_FIX_v1_20260724
# Uses BAU + cost-plus-carbon core outputs, solves the EfW comparison,
# labels route-treatment LCIA correctly, and reports annual network cost.

"""
Post-processing comparison for the industrial-symbiosis MILP.

This script deliberately does NOT rerun the full objective/Pareto analysis.
It:
  1. reads BAU and OPT_COST_PLUS_CARBON_VALUE from the existing main-run CSVs;
  2. solves one additional constrained case in which every plastic polymer must use UK EfW;
  3. enumerates Oxford, Cambridge and Slough as the possible 205-batch node for that
     constrained EfW case and selects the lowest cost-plus-carbon solution;
  4. writes tabular results.

No incineration-without-energy-recovery case is created.
"""

import importlib.util
import json
import math
import os
import tempfile
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Project paths and controls
# -----------------------------------------------------------------------------
# This script is expected to sit beside the master optimisation module in
# MILP/code. It reads the main-run outputs and solves only the forced
# EfW case using the Pyomo/GAMS solver configured in the master module.
CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR.parents[2]))
from project_paths import LCA_DIR, SUPERPRO_DIR

PROJECT_ROOT = CODE_DIR.parent
INPUT_DIR = PROJECT_ROOT / 'inputs'
RESULT_ROOT = PROJECT_ROOT / 'results'

MAIN_MODULE = CODE_DIR / 'run_master_MILP.py'
MAIN_RESULT_DIR = Path(os.getenv('IS_MAIN_RESULT_DIR', str(RESULT_ROOT / 'ecosystem_master_20261015')))
OUTPUT_DIR = Path(os.getenv('IS_ROUTING_OUTPUT_DIR', str(RESULT_ROOT / 'ecosystem_routing_20261015')))
PARAMS_FILE = PROJECT_ROOT.parent / 'parameters/MILP_central_params.json'

INPUTS = {
    'supply_chain': Path(os.environ.get(
        'IS_SUPPLY_CHAIN_FILE',
        PROJECT_ROOT.parent / 'mAb_SupplyChain_model/mab_supply_chain_superpro_v2_results.csv',
    )),
    'eligibility': INPUT_DIR / 'polymer_route_eligibility_20260722.xlsx',
    'lca': Path(os.getenv(
        'IS_LCA_ROUTE_FACTOR_FILE',
        str(LCA_DIR / 'outputs/master_lca_route_factor_database_ecoinvent_3_12.xlsx'),
    )),
    'superpro_master': SUPERPRO_DIR / 'data/processed/SP_master_dataset.xlsx',
    'params': PARAMS_FILE,
}

EXISTING_CASES = {
    "BAU": "Base case",
    "OPT_COST_PLUS_CARBON_VALUE": "Optimal routing",
}
FORCED_EFW_CASE_ID = "FORCED_EFW_COST_PLUS_CARBON_VALUE"
FORCED_EFW_CASE_NAME = "Energy from Waste (EfW)"
EFW_ROUTE = "incineration_efw_heat_electricity"

# Transport has validated cost and GWP coefficients in the current MILP inputs.
# No transport fossil-resource or water-scarcity coefficients exist in the supplied
# route-factor database, so transport is included in cost/GWP only and explicitly
# flagged as not modelled for the other two LCIA categories.


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------
def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_csv_required(path: Path) -> pd.DataFrame:
    require(path.exists(), f"Missing required main-run output: {path}")
    return pd.read_csv(path)


def import_main_module():
    require(MAIN_MODULE.exists(), f"Main optimisation module not found: {MAIN_MODULE}")
    spec = importlib.util.spec_from_file_location("is_main_model_for_postprocess", MAIN_MODULE)
    require(spec is not None and spec.loader is not None, "Could not create import specification for main module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def single_row(df: pd.DataFrame, mask: pd.Series, label: str) -> pd.Series:
    rows = df.loc[mask]
    require(len(rows) == 1, f"Expected one row for {label}; found {len(rows)}.")
    return rows.iloc[0]


def engineering_formatter(value: float, _position: int) -> str:
    sign = "−" if value < 0 else ""
    x = abs(value)
    if x >= 1e9:
        return f"{sign}{x / 1e9:.1f}B"
    if x >= 1e6:
        return f"{sign}{x / 1e6:.1f}M"
    if x >= 1e3:
        return f"{sign}{x / 1e3:.1f}k"
    return f"{sign}{x:.0f}"


def pct_change(value: float, bau: float) -> float:
    return 100.0 * (value - bau) / abs(bau) if abs(bau) > 1e-12 else np.nan


# -----------------------------------------------------------------------------
# Read the existing main-run cases
# -----------------------------------------------------------------------------
def load_existing_outputs() -> Dict[str, pd.DataFrame]:
    files = {
        "summary": "case_summary.csv",
        "routes": "selected_case_polymer_routing.csv",
        "sites": "selected_case_site_decisions.csv",
        "mix": "selected_case_route_mix.csv",
    }
    return {key: read_csv_required(MAIN_RESULT_DIR / name) for key, name in files.items()}


def extract_existing_case(outputs: Mapping[str, pd.DataFrame], case_id: str, case_name: str) -> Dict[str, Any]:
    summary = outputs["summary"].loc[outputs["summary"]["case_id"].eq(case_id)].copy()
    require(len(summary) == 1, f"Case {case_id} not found exactly once in case_summary.csv.")
    row = summary.iloc[0]

    routes = outputs["routes"].loc[outputs["routes"]["case_id"].eq(case_id)].copy()
    sites = outputs["sites"].loc[outputs["sites"]["case_id"].eq(case_id)].copy()
    mix = outputs["mix"].loc[outputs["mix"]["case_id"].eq(case_id)].copy()
    require(not routes.empty and not sites.empty and not mix.empty, f"Decision outputs are incomplete for {case_id}.")

    return {
        "case_id": case_id,
        "case_name": case_name,
        "metrics": {
            "cost": float(row["cost"]),
            "gwp": float(row["gwp"]),
            "human_health_daly": float(row["human_health_daly"]),
            "ecosystem_quality": float(row["ecosystem_quality"]),
            "policy_cost": float(row["policy_cost"]),
            "freshwater": float(row["freshwater"]),
            "circularity_loss": float(row["circularity_loss"]),
            "residual": float(row["residual"]),
        },
        "routes": routes,
        "sites": sites,
        "mix": mix,
        "solver_backend": str(row.get("solver_backend", "existing main-run output")),
        "selection_method": str(row.get("selection_method", "existing main-run output")),
    }


# -----------------------------------------------------------------------------
# Solve only the forced-EfW case
# -----------------------------------------------------------------------------
def make_compatibility_params(original_params: Path, destination: Path) -> Dict[str, Any]:
    """Create a temporary read-only compatibility copy; never modify the user's input."""
    params = json.loads(original_params.read_text())
    renewable = params["industrial_renewable_electricity_cost"]
    if "value" not in renewable:
        require("central_value" in renewable, "Renewable electricity parameters lack both value and central_value.")
        renewable["value"] = renewable["central_value"]

    bev = params["road_transport_modes"]["battery_electric_articulated_hgv"]
    require(
        not bool(bev.get("additional_electricity_cost_in_objective", False)),
        "Battery-electric HGV charging electricity must not be added separately to transport cost.",
    )
    require(
        bool(bev.get("cost_includes_charging_electricity", True)),
        "Battery-electric HGV £/vehicle-km rate must be the all-in service cost.",
    )
    require(
        bool(bev.get("reference_gwp_must_not_be_added_to_dynamic_lca_result", True)),
        "The 0.236 kgCO2e/km reference must remain validation-only.",
    )
    destination.write_text(json.dumps(params, indent=2))
    return params


def enrich_forced_decisions(core, model, result, case_name: str):
    routes, sites, mix = core.solution_decisions(model, result)
    data = model.data

    # solution_decisions already derives annual batches and the low-batch flag from
    # the endogenous B[j] and low_batch[j] decision variables. Do not overwrite
    # them with the input CSV's original allocation.
    routes["case_name"] = case_name

    sites["fixed_wfi_mains_water_m3"] = sites["site"].map(data.fixed_mains_water_m3)
    sites["net_mains_freshwater_withdrawal_m3"] = (
        sites["fresh_utility_water_m3"] + sites["fixed_wfi_mains_water_m3"]
    )
    sites["case_name"] = case_name

    transport_rows = []
    for j in data.sites:
        for route in data.routes:
            for mode in data.transport_modes:
                key = ("n", j, route, mode)
                if key not in model.index:
                    continue
                journeys = float(result.x[model.index[key]])
                if journeys <= 1e-8:
                    continue
                vehicle_km = journeys * data.roundtrip_multiplier[j, route, mode] * data.distance_km[j, route]
                transport_rows.append({
                    "route": route,
                    "transport_mode": mode,
                    "journeys_per_year": journeys,
                    "vehicle_km_per_year": vehicle_km,
                    "transport_cost_gbp_per_year": vehicle_km * data.transport_cost_gbp_per_km[mode],
                    "transport_gwp_kgco2e_per_year": vehicle_km * data.transport_gwp_kg_per_km[mode],
                })
    transport = pd.DataFrame(transport_rows)
    require(not transport.empty, "Forced-EfW solution contains no transport journeys.")
    transport = transport.groupby(["route", "transport_mode"], as_index=False).sum(numeric_only=True)

    mix = mix.drop(columns=[c for c in [
        "journeys_per_year", "vehicle_km_per_year", "transport_cost_gbp_per_year",
        "transport_gwp_kgco2e_per_year"
    ] if c in mix.columns])
    mix = mix.merge(transport, on=["route", "transport_mode"], how="left", validate="one_to_one")
    mix["case_name"] = case_name

    return routes, sites, mix


def solve_forced_efw() -> Tuple[Dict[str, Any], pd.DataFrame, Any]:
    core = import_main_module()
    print(f'Forced-EfW solve uses solver: {"SciPy HiGHS" if os.getenv("IS_MILP_BACKEND", "gams").lower() == "scipy" else core.GAMS_SOLVER_NAME}')
    outputs = []

    supply = pd.read_csv(INPUTS["supply_chain"])
    site_map = pd.read_excel(INPUTS["eligibility"], sheet_name="site_mapping")
    location_by_site = dict(zip(site_map["manufacturing_site"].astype(str), site_map["site_location"].astype(str)))
    require(set(supply["manufacturing_site"].astype(str)) == set(location_by_site), "Supply-chain/site-map mismatch.")

    levels = sorted({int(round(v)) for v in supply["mab_batches_per_year"].astype(float)})
    require(len(levels) == 2, f"Expected exactly two retained batch levels; found {levels}.")
    low_batches, high_batches = levels[0], levels[1]

    with tempfile.TemporaryDirectory(prefix="forced_efw_postprocess_") as tmp:
        tmp = Path(tmp)
        compat_params = tmp / "MILP_params_compat.json"
        make_compatibility_params(INPUTS["params"], compat_params)

        # All non-temporary input paths remain the original files.
        core.INPUT_FILES.update({
            "eligibility": INPUTS["eligibility"],
            "lca": INPUTS["lca"],
            "superpro_master": INPUTS["superpro_master"],
            "params": compat_params,
        })

        for low_site in supply["manufacturing_site"].astype(str):
            candidate_supply = supply.copy()
            candidate_supply["mab_batches_per_year"] = np.where(
                candidate_supply["manufacturing_site"].astype(str).eq(low_site),
                low_batches,
                high_batches,
            )
            candidate_path = tmp / f"supply_low_{low_site}.csv"
            candidate_supply.to_csv(candidate_path, index=False)
            core.INPUT_FILES["supply_chain"] = candidate_path

            data = core.load_model_data()
            model = core.build_linear_model(data)

            # Explicitly fix the endogenous low-batch-location decision for this
            # audit solve. Merely rewriting the input allocation is insufficient
            # because the core model is intentionally free to relocate the low node.
            for j in data.sites:
                idx = model.index[("low_batch", j)]
                fixed_value = 1.0 if j == low_site else 0.0
                model.lower[idx] = fixed_value
                model.upper[idx] = fixed_value

            require(EFW_ROUTE in data.routes, f"Required EfW route {EFW_ROUTE!r} is not in the base model.")
            for key, idx in model.index.items():
                if key[0] == "x" and key[3] != EFW_ROUTE:
                    model.upper[idx] = 0.0

            result = core.solve_model(
                model,
                case_id=f"{FORCED_EFW_CASE_ID}__LOW_{low_site}",
                primary="policy_cost",
                selection_method="all plastics forced to EfW; minimise private cost plus lifecycle carbon value",
                metadata={"low_batch_site": low_site, "low_batch_location": location_by_site[low_site]},
            )
            require(result.success, f"Forced-EfW solve failed for low-batch site {low_site}: {result.message}")
            result.metrics["human_health_daly"] = core.evaluate_human_health_daly(model, result)
            result.metrics["ecosystem_quality"] = core.evaluate_ecosystem_quality(model, result)

            routes, sites, mix = enrich_forced_decisions(core, model, result, FORCED_EFW_CASE_NAME)
            outputs.append({
                "case_id": FORCED_EFW_CASE_ID,
                "case_name": FORCED_EFW_CASE_NAME,
                "metrics": dict(result.metrics),
                "routes": routes,
                "sites": sites,
                "mix": mix,
                "solver_backend": result.metadata.get("solver_backend", ""),
                "selection_method": result.selection_method,
                "low_site": low_site,
                "low_location": location_by_site[low_site],
                "model": model,
                "data": data,
                "result": result,
            })

    audit = pd.DataFrame([
        {
            "low_batch_site": item["low_site"],
            "low_batch_location": item["low_location"],
            "private_cost_gbp_per_year": item["metrics"]["cost"],
            "gwp_kgco2e_per_year": item["metrics"]["gwp"],
            "cost_plus_carbon_value_gbp_per_year": item["metrics"]["policy_cost"],
            "net_freshwater_m3_per_year": item["metrics"]["freshwater"],
            "solver_backend": item["solver_backend"],
        }
        for item in outputs
    ]).sort_values("cost_plus_carbon_value_gbp_per_year")

    chosen_low_site = str(audit.iloc[0]["low_batch_site"])
    chosen = next(item for item in outputs if item["low_site"] == chosen_low_site)
    chosen["result"].case_id = FORCED_EFW_CASE_ID
    chosen["routes"]["case_id"] = FORCED_EFW_CASE_ID
    chosen["sites"]["case_id"] = FORCED_EFW_CASE_ID
    chosen["mix"]["case_id"] = FORCED_EFW_CASE_ID
    return chosen, audit, core


# -----------------------------------------------------------------------------
# LCIA and cost post-processing
# -----------------------------------------------------------------------------
IMPACT_DEFINITIONS = {
    "gwp": {
        "net": "net_gwp_factor_per_optimisation_unit",
        "virgin_production": "gwp__virgin_polymer_production",
        "sut_component_manufacturing": "gwp__sut_component_manufacturing",
        "supply": "gwp__supply",
        "treatment": "gwp__treatment",
        "credit": "gwp__credit",
        "label": "Carbon footprint (IMPACT World+ v2.1, IPCC AR6)",
        "unit_annual": "kg CO₂e/year",
        "unit_batch": "kg CO₂e/batch",
    },
    "fossil_resource": {
        "net": "net_fossil_resource_use_factor_per_optimisation_unit",
        "virgin_production": "fossil_resource_use__virgin_polymer_production",
        "sut_component_manufacturing": None,
        "supply": "fossil_resource_use__supply",
        "treatment": "fossil_resource_use__treatment",
        "credit": "fossil_resource_use__credit",
        "label": "Fossil resource scarcity",
        "unit_annual": "kg oil-eq/year",
        "unit_batch": "kg oil-eq/batch",
    },
    "water_scarcity": {
        "net": "net_water_scarcity_factor_per_optimisation_unit",
        "virgin_production": "water_scarcity__virgin_polymer_production",
        "sut_component_manufacturing": None,
        "supply": "water_scarcity__supply",
        "treatment": "water_scarcity__treatment",
        "credit": "water_scarcity__credit",
        "label": "Water scarcity (AWARE 2.0, site-regionalised foreground)",
        "unit_annual": "m³ world-eq/year",
        "unit_batch": "m³ world-eq/batch",
    },
    "human_health_daly": {
        "net": "net_human_health_daly_factor_per_optimisation_unit",
        "virgin_production": "human_health_daly__virgin_polymer_production",
        "sut_component_manufacturing": "human_health_daly__sut_component_manufacturing",
        "supply": "human_health_daly__supply",
        "treatment": "human_health_daly__treatment",
        "credit": "human_health_daly__credit",
        "label": "Human-health damage",
        "unit_annual": "DALY/year",
        "unit_batch": "DALY/batch",
    },
    "ecosystem_quality": {
        "net": "net_ecosystem_quality_factor_per_optimisation_unit",
        "virgin_production": "ecosystem_quality__virgin_polymer_production",
        "sut_component_manufacturing": "ecosystem_quality__sut_component_manufacturing",
        "supply": "ecosystem_quality__supply",
        "treatment": "ecosystem_quality__treatment",
        "credit": "ecosystem_quality__credit",
        "label": "Ecosystem damage",
        "unit_annual": "species·year/year",
        "unit_batch": "species·year/batch",
    },
}


def route_factor_row(rf: pd.DataFrame, flow: str, pathway: str, material: str | None = None) -> pd.Series:
    mask = rf["optimisation_flow"].astype(str).eq(flow) & rf["pathway"].astype(str).eq(pathway)
    if material is not None:
        mask &= rf["material_or_service"].astype(str).eq(material)
    return single_row(rf, mask, f"route factor {flow}/{pathway}/{material}")


def calculate_impact_components(
    case: Mapping[str, Any],
    data,
    rf: pd.DataFrame,
    total_batches: float,
) -> Dict[str, Dict[str, float]]:
    """Exact high-level LCIA decomposition.

    The category totals reconcile exactly to the master MILP objectives. The
    route-factor workbook embeds autoclave electricity within the aggregate
    plastic treatment burden for landfill and recycling, rather than reliably
    exposing it in the ``__supply`` field. For reporting only, the known 0.8
    kWh/kg autoclave demand is valued with the same grid LCIA factors and removed
    from the aggregate process burden. EfW receives zero autoclave burden.
    Recycling material credits and EfW energy credits remain distinct.
    """
    routes = case["routes"]
    sites = case["sites"]
    mix = case["mix"]

    components: Dict[str, Dict[str, float]] = {}
    for metric, definition in IMPACT_DEFINITIONS.items():
        values = {
            "Other utilities": 0.0,
            "Electricity": 0.0,
            "Virgin polymer production": 0.0,
            "SUT component manufacturing": 0.0,
            "Autoclave": 0.0,
            "End-of-life treatment": 0.0,
            "Recycling credit": 0.0,
            "EfW energy credit": 0.0,
            "Water": 0.0,
            "Biomass": 0.0,
            "Diesel HGV": 0.0,
            "Electric HGV": 0.0,
        }

        # Fixed non-electric manufacturing foreground: WFI, chilled water,
        # cooling water and steam.
        for flow in [
            "manufacturing__wfi",
            "manufacturing__chilled_water",
            "manufacturing__cooling_water",
            "manufacturing__steam",
        ]:
            row = route_factor_row(rf, flow, "baseline_supply")
            values["Other utilities"] += (
                float(row["represented_quantity_per_batch"])
                * total_batches
                * float(row[definition["net"]])
            )

        # Manufacturing electricity is deliberately reported separately from IS.
        grid_row = route_factor_row(rf, "manufacturing__electricity", "electricity_grid_uk_bau")
        wind_row = route_factor_row(rf, "manufacturing__electricity", "electricity_renewable_onshore_wind")
        values["Electricity"] = (
            sites["grid_kwh"].sum() * float(grid_row[definition["net"]])
            + sites["wind_kwh"].sum() * float(wind_row[definition["net"]])
        )

        # High-level plastic decomposition. The net route factor remains the
        # source of truth. The exported explanatory columns currently place the
        # embedded autoclave and end-of-life burdens together in supply+treatment
        # for some pathways. Split that combined burden using the model's explicit
        # autoclave demand and the method-consistent grid LCIA factor.
        grid_cost_per_kwh = float(data.electricity_cost_gbp_per_kwh["grid"])
        require(grid_cost_per_kwh > 0.0, "Grid electricity cost must be positive.")
        autoclave_kwh_per_kg_values = {
            float(data.autoclave_cost_gbp_per_t[j]) / (1000.0 * grid_cost_per_kwh)
            for j in data.sites
        }
        require(
            len(autoclave_kwh_per_kg_values) == 1,
            f"Autoclave electricity demand differs by site: {autoclave_kwh_per_kg_values}",
        )
        autoclave_kwh_per_kg = autoclave_kwh_per_kg_values.pop()
        grid_factor_for_metric = float(grid_row[definition["net"]])
        autoclave_factor_per_kg = autoclave_kwh_per_kg * grid_factor_for_metric

        for row in routes.itertuples(index=False):
            factor = route_factor_row(
                rf,
                f"sut_plastic__{row.polymer}",
                str(row.lca_pathway),
                str(row.polymer),
            )
            polymer = str(row.polymer)
            route_name = str(row.route)
            kg = float(row.tonnes_per_year) * 1000.0
            combined_process_factor = (
                float(factor[definition["supply"]])
                + float(factor[definition["treatment"]])
            )
            requires_autoclave = bool(data.autoclave_required[polymer, route_name])
            autoclave_factor = autoclave_factor_per_kg if requires_autoclave else 0.0
            eol_factor = combined_process_factor - autoclave_factor
            require(
                eol_factor >= -1e-10,
                f"Derived end-of-life factor is negative for {polymer}/{route_name}/{metric}: "
                f"combined={combined_process_factor}, autoclave={autoclave_factor}.",
            )
            if abs(eol_factor) < 1e-10:
                eol_factor = 0.0

            values["Virgin polymer production"] += (
                kg * float(factor[definition["virgin_production"]])
            )
            component_column = definition["sut_component_manufacturing"]
            if component_column is not None:
                values["SUT component manufacturing"] += kg * float(factor[component_column])
            values["Autoclave"] += kg * autoclave_factor
            values["End-of-life treatment"] += kg * eol_factor

            credit = kg * float(factor[definition["credit"]])
            if "mechanical_recycling" in route_name:
                values["Recycling credit"] += credit
            elif "incineration_efw" in route_name:
                values["EfW energy credit"] += credit
            else:
                require(
                    abs(credit) <= 1e-8,
                    f"Unclassified non-zero plastic credit for route {route_name}: {credit}",
                )

        # Water formulation mirrors the master MILP net-factor convention:
        # eligible utility-water demand plus the selected discharge/reuse route.
        fresh_row = route_factor_row(rf, "fresh_utility_water", "fresh_utility_water_supply")
        discharge_row = route_factor_row(rf, "plant_wastewater", "treatment_and_discharge")
        reuse_row = route_factor_row(rf, "plant_wastewater", "treatment_for_utility_reuse")
        if metric == "water_scarcity":
            water_total = 0.0
            for site_row in sites.itertuples(index=False):
                site = str(site_row.site)
                location = str(data.site_location[site])
                factor_column = f"net_water_scarcity_factor__{location}"
                require(
                    factor_column in rf.columns,
                    f"Missing site-resolved AWARE 2.0 factor column {factor_column}.",
                )
                eligible_utility_demand = (
                    float(site_row.fresh_utility_water_m3)
                    + float(site_row.water_reused_m3)
                )
                water_total += (
                    eligible_utility_demand * float(fresh_row[factor_column])
                    + float(site_row.effluent_discharged_m3)
                    * float(discharge_row[factor_column])
                    + float(site_row.wastewater_to_reuse_m3)
                    * float(reuse_row[factor_column])
                )
            values["Water"] = water_total
        else:
            eligible_utility_demand = (
                sites["fresh_utility_water_m3"] + sites["water_reused_m3"]
            ).sum()
            q_reuse = sites["wastewater_to_reuse_m3"].sum()
            q_discharge = sites["effluent_discharged_m3"].sum()
            values["Water"] = (
                eligible_utility_demand * float(fresh_row[definition["net"]])
                + q_discharge * float(discharge_row[definition["net"]])
                + q_reuse * float(reuse_row[definition["net"]])
            )

        values["Biomass"] = sum(
            float(site.biomass_kg_per_year)
            * data.biomass_route_factors[str(site.biomass_route)][
                'fossil_resource_use' if metric == 'fossil_resource' else metric]
            for site in sites.itertuples(index=False)
        )

        # The current database has validated transport coefficients for GWP and
        # cost only. Diesel and battery-electric transport are displayed
        # separately whenever the optimiser selects them.
        if metric == "gwp":
            for row in mix.itertuples(index=False):
                mode = str(row.transport_mode)
                category = (
                    "Electric HGV"
                    if "battery_electric" in mode
                    else "Diesel HGV"
                )
                values[category] += float(row.transport_gwp_kgco2e_per_year)

        components[metric] = values

    calculated_gwp = sum(components["gwp"].values())
    expected_gwp = float(case["metrics"]["gwp"])
    tolerance = max(1e-4, 1e-8 * abs(expected_gwp))
    require(
        abs(calculated_gwp - expected_gwp) <= tolerance,
        f"GWP post-processing does not reconcile for {case['case_id']}: "
        f"calculated={calculated_gwp}, expected={expected_gwp}",
    )
    calculated_daly = sum(components["human_health_daly"].values())
    expected_daly = float(case["metrics"]["human_health_daly"])
    daly_tolerance = max(1e-12, 1e-8 * abs(expected_daly))
    require(
        abs(calculated_daly - expected_daly) <= daly_tolerance,
        f"DALY post-processing does not reconcile for {case['case_id']}: "
        f"calculated={calculated_daly}, expected={expected_daly}",
    )
    calculated_ecosystem = sum(components["ecosystem_quality"].values())
    expected_ecosystem = float(case["metrics"]["ecosystem_quality"])
    ecosystem_tolerance = max(1e-15, 1e-8 * abs(expected_ecosystem))
    require(
        abs(calculated_ecosystem - expected_ecosystem) <= ecosystem_tolerance,
        f"Ecosystem post-processing does not reconcile for {case['case_id']}: "
        f"calculated={calculated_ecosystem}, expected={expected_ecosystem}",
    )
    return components


def calculate_cost_components(case: Mapping[str, Any], data) -> Dict[str, float]:
    """Exact annual-cost decomposition, including autoclaving and transport mode."""
    routes = case["routes"]
    sites = case["sites"]
    mix = case["mix"]

    electricity = (
        sites["grid_kwh"].sum() * data.electricity_cost_gbp_per_kwh["grid"]
        + sites["wind_kwh"].sum() * data.electricity_cost_gbp_per_kwh["wind"]
    )

    landfill_fees = 0.0
    efw_fees = 0.0
    recycling_fees = 0.0
    autoclave_pre_treatment = 0.0
    recyclate_return = 0.0
    for row in routes.itertuples(index=False):
        p, r, j = str(row.polymer), str(row.route), str(row.site)
        tonnes = float(row.tonnes_per_year)
        if r == "incineration_efw_heat_electricity":
            require(
                int(data.autoclave_required[p, r]) == 0,
                f"Autoclaving must not be assigned to the EfW route {p}/{r}.",
            )
        elif r == "landfill_all" or r.startswith("mechanical_recycling"):
            require(
                int(data.autoclave_required[p, r]) == 1,
                f"Autoclaving is required for landfill and mechanical recycling: {p}/{r}.",
            )
        route_fee = tonnes * data.route_cost_gbp_per_t[p, r]
        if r == "landfill_all":
            landfill_fees += route_fee
        elif "incineration_efw" in r:
            efw_fees += route_fee
        elif "mechanical_recycling" in r:
            recycling_fees += route_fee
        else:
            raise RuntimeError(f"Unclassified plastic route fee for reporting: {r}")
        autoclave_pre_treatment += tonnes * (
            data.autoclave_required[p, r] * data.autoclave_cost_gbp_per_t[j]
        )
        recyclate_return -= tonnes * (
            data.value_capture[p, r]
            * data.recovered_value_gbp_per_t[p, r]
            * data.recovery_yield[p, r]
        )

    water_operating = sum(
        data.treatment_opex_gbp_per_m3[j] * data.wastewater_m3[j]
        for j in data.sites
    )
    water_operating += sum(
        float(row.fresh_utility_water_m3) * data.fresh_water_cost_gbp_per_m3[str(row.site)]
        + float(row.effluent_discharged_m3) * data.discharge_cost_gbp_per_m3[str(row.site)]
        for row in sites.itertuples(index=False)
    )
    annualised_wwt_capex = sum(
        data.treatment_fixed_cost_gbp[j] * data.site_open[j]
        for j in data.sites
    )

    diesel_transport = 0.0
    electric_transport = 0.0
    for row in mix.itertuples(index=False):
        value = float(row.transport_cost_gbp_per_year)
        if "battery_electric" in str(row.transport_mode):
            electric_transport += value
        else:
            diesel_transport += value

    values = {
        "Electricity": electricity,
        "Landfill fee": landfill_fees,
        "EfW fee": efw_fees,
        "Recycling fee": recycling_fees,
        "Autoclave": autoclave_pre_treatment,
        "Recyclate return": recyclate_return,
        "Water OPEX": water_operating,
        "WWT CAPEX": annualised_wwt_capex,
        "Biomass": data.biomass_cost_gbp,
        "Diesel HGV": diesel_transport,
        "Electric HGV": electric_transport,
    }
    calculated = sum(values.values())
    expected = float(case["metrics"]["cost"])
    tolerance = max(1e-4, 1e-8 * abs(expected))
    require(
        abs(calculated - expected) <= tolerance,
        f"Cost post-processing does not reconcile for {case['case_id']}: "
        f"calculated={calculated}, expected={expected}",
    )
    return values


def calculate_net_water_components(case: Mapping[str, Any]) -> Dict[str, float]:
    sites = case["sites"]
    fixed_wfi = float(sites["fixed_wfi_mains_water_m3"].sum())
    eligible = float((sites["fresh_utility_water_m3"] + sites["water_reused_m3"]).sum())
    reuse_credit = -float(sites["water_reused_m3"].sum())
    values = {
        "WFI": fixed_wfi,
        "Utility water": eligible,
        "Reuse credit": reuse_credit,
    }
    expected = float(case["metrics"]["freshwater"])
    require(abs(sum(values.values()) - expected) <= 1e-6, f"Net-water breakdown does not reconcile for {case['case_id']}.")
    return values


# -----------------------------------------------------------------------------
# Reporting tables
# -----------------------------------------------------------------------------
def electricity_text(sites: pd.DataFrame) -> str:
    total = float(sites["grid_kwh"].sum() + sites["wind_kwh"].sum())
    wind = float(sites["wind_kwh"].sum())
    renewable_pct = 100.0 * wind / total
    grid_pct = 100.0 - renewable_pct
    if renewable_pct <= 1e-6:
        return "Grid (100%)"
    if grid_pct <= 1e-6:
        return "Renewable cPPA (100%)"
    return f"Renewable {renewable_pct:.1f}%; grid {grid_pct:.1f}%"


def plastics_text(mix: pd.DataFrame) -> str:
    pretty = {
        "landfill_all": "Landfill",
        "incineration_efw_heat_electricity": "EfW",
        "mechanical_recycling_polyolefins": "Mechanical recycling – polyolefins",
        "mechanical_recycling_tpe_tubing": "Mechanical recycling – TPE tubing",
    }
    rows = mix.loc[mix["tonnes_per_year"].astype(float) > 1e-8].sort_values(
        "tonnes_per_year", ascending=False
    )
    total = float(rows["tonnes_per_year"].sum())
    return "; ".join(
        f"{pretty.get(str(r.route), str(r.route).replace('_', ' ').title())}: "
        f"{r.tonnes_per_year:.1f} t/y ({100.0 * r.tonnes_per_year / total:.0f}%)"
        for r in rows.itertuples(index=False)
    )


def water_text(sites: pd.DataFrame) -> str:
    eligible = float((sites["fresh_utility_water_m3"] + sites["water_reused_m3"]).sum())
    reused = float(sites["water_reused_m3"].sum())
    discharged = float(sites["effluent_discharged_m3"].sum())
    reuse_pct = 100.0 * reused / eligible
    if reuse_pct <= 1e-6:
        return f"No reuse; {discharged:,.0f} m³/y discharged"
    return f"Reuse {reuse_pct:.0f}%; {discharged:,.0f} m³/y discharged"


def transport_text(mix: pd.DataFrame) -> str:
    pretty = {
        "diesel_articulated_hgv": "Diesel HGV",
        "battery_electric_articulated_hgv": "Battery-electric HGV",
    }
    grouped = mix.groupby("transport_mode", as_index=False).agg(
        journeys=("journeys_per_year", "sum"),
        vehicle_km=("vehicle_km_per_year", "sum"),
    )
    grouped = grouped.loc[grouped["journeys"].astype(float) > 1e-8].copy()
    total_km = float(grouped["vehicle_km"].sum())
    rows = []
    for r in grouped.itertuples(index=False):
        label = pretty.get(str(r.transport_mode), str(r.transport_mode))
        text = f"{label}; {r.journeys:.0f} journeys/y; {r.vehicle_km:,.0f} km/y"
        if len(grouped) > 1:
            text += f" ({100.0 * r.vehicle_km / total_km:.0f}% of km)"
        rows.append(text)
    return "; ".join(rows)


def low_batch_location(sites: pd.DataFrame) -> str:
    if "is_low_batch_location" in sites.columns and sites["is_low_batch_location"].any():
        return str(sites.loc[sites["is_low_batch_location"], "location"].iloc[0])
    idx = sites["annual_batches"].astype(float).idxmin()
    return str(sites.loc[idx, "location"])


def build_tables(
    cases: Iterable[Mapping[str, Any]], impacts, costs, waters,
    total_batches: float, route_factors: pd.DataFrame,
):
    cases = list(cases)
    gwp_methods = route_factors["gwp_method"].dropna().astype(str).unique()
    water_methods = route_factors["water_scarcity_method"].dropna().astype(str).unique()
    require(len(gwp_methods) == 1, f"Expected one GWP method; found {gwp_methods}.")
    require(len(water_methods) == 1, f"Expected one water-scarcity method; found {water_methods}.")
    bau = cases[0]
    bau_impacts = {m: sum(impacts[bau["case_id"]][m].values()) for m in IMPACT_DEFINITIONS}
    bau_cost = float(bau["metrics"]["cost"])
    bau_policy = float(bau["metrics"]["policy_cost"])
    bau_water = float(bau["metrics"]["freshwater"])

    summary_rows = []
    decision_rows = []
    category_rows = []

    for case in cases:
        cid = case["case_id"]
        impact_totals = {m: sum(impacts[cid][m].values()) for m in IMPACT_DEFINITIONS}
        carbon_value_cost = float(case["metrics"]["policy_cost"] - case["metrics"]["cost"])
        summary_rows.append({
            "case_id": cid,
            "case_name": case["case_name"],
            "gwp_method": gwp_methods[0],
            "water_scarcity_method": water_methods[0],
            "water_scarcity_foreground_regionalisation": (
                "Oxford/Cambridge/Slough annual non-agricultural AWARE 2.0 basin factors; "
                "background inventory retains native Brightway geography"
            ),
            "low_batch_location": low_batch_location(case["sites"]),
            "private_annual_cost_gbp": float(case["metrics"]["cost"]),
            "pct_change_vs_bau__private_cost": pct_change(float(case["metrics"]["cost"]), bau_cost),
            "lifecycle_carbon_value_cost_gbp": carbon_value_cost,
            "cost_plus_carbon_value_gbp": float(case["metrics"]["policy_cost"]),
            "pct_change_vs_bau__cost_plus_carbon": pct_change(float(case["metrics"]["policy_cost"]), bau_policy),
            "gwp_kgco2e_per_year": impact_totals["gwp"],
            "gwp_kgco2e_per_batch": impact_totals["gwp"] / total_batches,
            "pct_change_vs_bau__gwp": pct_change(impact_totals["gwp"], bau_impacts["gwp"]),
            "fossil_resource_kg_oil_eq_per_year": impact_totals["fossil_resource"],
            "fossil_resource_kg_oil_eq_per_batch": impact_totals["fossil_resource"] / total_batches,
            "pct_change_vs_bau__fossil_resource": pct_change(impact_totals["fossil_resource"], bau_impacts["fossil_resource"]),
            "water_scarcity_m3_world_eq_per_year": impact_totals["water_scarcity"],
            "water_scarcity_m3_world_eq_per_batch": impact_totals["water_scarcity"] / total_batches,
            "pct_change_vs_bau__water_scarcity": pct_change(impact_totals["water_scarcity"], bau_impacts["water_scarcity"]),
            "human_health_daly_per_year": impact_totals["human_health_daly"],
            "human_health_daly_per_batch": impact_totals["human_health_daly"] / total_batches,
            "avoided_human_health_daly_vs_bau": bau_impacts["human_health_daly"] - impact_totals["human_health_daly"],
            "human_health_daly_reduction_vs_bau_pct": (
                100.0 * (bau_impacts["human_health_daly"] - impact_totals["human_health_daly"])
                / bau_impacts["human_health_daly"]
                if bau_impacts["human_health_daly"] != 0 else np.nan
            ),
            "ecosystem_quality_species_year_per_year": impact_totals["ecosystem_quality"],
            "ecosystem_quality_species_year_per_batch": impact_totals["ecosystem_quality"] / total_batches,
            "avoided_ecosystem_quality_vs_bau": bau_impacts["ecosystem_quality"] - impact_totals["ecosystem_quality"],
            "ecosystem_quality_reduction_vs_bau_pct": (
                100.0 * (bau_impacts["ecosystem_quality"] - impact_totals["ecosystem_quality"])
                / bau_impacts["ecosystem_quality"]
                if bau_impacts["ecosystem_quality"] != 0 else np.nan
            ),
            "net_freshwater_withdrawal_m3_per_year": float(case["metrics"]["freshwater"]),
            "pct_change_vs_bau__net_freshwater": pct_change(float(case["metrics"]["freshwater"]), bau_water),
            "residual_disposal_t_per_year": float(case["metrics"]["residual"]),
            "circularity_loss_t_per_year": float(case["metrics"]["circularity_loss"]),
        })
        decision_rows.append({
            "case_id": cid,
            "case_name": case["case_name"],
            "low_batch_location": low_batch_location(case["sites"]),
            "electricity_choice": electricity_text(case["sites"]),
            "plastics_routes_chosen": plastics_text(case["mix"]),
            "water_route_chosen": water_text(case["sites"]),
            "biomass_route_chosen": "; ".join(
                f"{route}: {group.biomass_kg_per_year.sum()/1000:.3f} t/y"
                for route, group in case["sites"].groupby("biomass_route")
            ),
            "transport_choice": transport_text(case["mix"]),
            "solver_backend": case["solver_backend"],
            "selection_method": case["selection_method"],
        })

        for metric, values in impacts[cid].items():
            bau_values = impacts[bau["case_id"]][metric]
            for category, value in values.items():
                category_rows.append({
                    "case_id": cid,
                    "case_name": case["case_name"],
                    "reporting_block": (
                        "Manufacturing electricity" if category == "Electricity"
                        else "Fixed manufacturing background" if category == "Other utilities"
                        else "Industrial symbiosis routing"
                    ),
                    "metric": metric,
                    "category": category,
                    "annual_value": value,
                    "per_batch_value": value / total_batches,
                    "saving_vs_bau_annual": bau_values[category] - value,
                })
        for category, value in costs[cid].items():
            bau_value = costs[bau["case_id"]][category]
            category_rows.append({
                "case_id": cid,
                "case_name": case["case_name"],
                "reporting_block": "Manufacturing electricity" if category == "Electricity" else "Industrial symbiosis routing",
                "metric": "private_cost",
                "category": category,
                "annual_value": value,
                "per_batch_value": value / total_batches,
                "saving_vs_bau_annual": bau_value - value,
            })
        for category, value in waters[cid].items():
            bau_value = waters[bau["case_id"]][category]
            category_rows.append({
                "case_id": cid,
                "case_name": case["case_name"],
                "reporting_block": "Freshwater withdrawal",
                "metric": "net_water",
                "category": category,
                "annual_value": value,
                "per_batch_value": value / total_batches,
                "saving_vs_bau_annual": bau_value - value,
            })

    return pd.DataFrame(summary_rows), pd.DataFrame(decision_rows), pd.DataFrame(category_rows)


def build_selected_factor_table(cases, impacts, costs, data_by_case, rf, total_batches: float) -> pd.DataFrame:
    """Compact high-level coefficient/factor audit for the displayed cases."""
    rows = []
    for case in cases:
        cid = case["case_id"]
        data = data_by_case[cid]
        sites = case["sites"]
        routes = case["routes"]
        mix = case["mix"]

        electricity_kwh = float(sites["grid_kwh"].sum() + sites["wind_kwh"].sum())
        rows.append({
            "case_id": cid,
            "case_name": case["case_name"],
            "category": "Electricity",
            "selected_route_or_source": electricity_text(sites),
            "annual_quantity": electricity_kwh,
            "quantity_unit": "kWh/year",
            "weighted_private_cost_coefficient": costs[cid]["Electricity"] / electricity_kwh,
            "private_cost_coefficient_unit": "GBP/kWh",
            "weighted_gwp_factor": impacts[cid]["gwp"]["Electricity"] / electricity_kwh,
            "weighted_fossil_resource_factor": impacts[cid]["fossil_resource"]["Electricity"] / electricity_kwh,
            "weighted_water_scarcity_factor": impacts[cid]["water_scarcity"]["Electricity"] / electricity_kwh,
            "weighted_human_health_daly_factor": impacts[cid]["human_health_daly"]["Electricity"] / electricity_kwh,
            "weighted_ecosystem_quality_factor": impacts[cid]["ecosystem_quality"]["Electricity"] / electricity_kwh,
            "environmental_factor_unit_basis": "impact unit/kWh",
            "factor_note": "Weighted average of selected grid and renewable-cPPA electricity factors.",
        })

        plastic_t = float(routes["tonnes_per_year"].sum())
        plastic_kg = plastic_t * 1000.0
        plastic_cost_net = sum(
            costs[cid][key]
            for key in [
                "Landfill fee", "EfW fee", "Recycling fee",
                "Autoclave", "Recyclate return",
            ]
        )
        plastic_categories = [
            "Virgin polymer production",
            "SUT component manufacturing",
            "Autoclave",
            "End-of-life treatment",
            "Recycling credit",
            "EfW energy credit",
        ]
        rows.append({
            "case_id": cid,
            "case_name": case["case_name"],
            "category": "Plastics",
            "selected_route_or_source": plastics_text(mix),
            "annual_quantity": plastic_t,
            "quantity_unit": "t plastic/year",
            "weighted_private_cost_coefficient": plastic_cost_net / plastic_t,
            "private_cost_coefficient_unit": "net GBP/t input; excludes transport",
            "weighted_gwp_factor": sum(impacts[cid]["gwp"][k] for k in plastic_categories) / plastic_kg,
            "weighted_fossil_resource_factor": sum(impacts[cid]["fossil_resource"][k] for k in plastic_categories) / plastic_kg,
            "weighted_water_scarcity_factor": sum(impacts[cid]["water_scarcity"][k] for k in plastic_categories) / plastic_kg,
            "weighted_human_health_daly_factor": sum(impacts[cid]["human_health_daly"][k] for k in plastic_categories) / plastic_kg,
            "weighted_ecosystem_quality_factor": sum(impacts[cid]["ecosystem_quality"][k] for k in plastic_categories) / plastic_kg,
            "environmental_factor_unit_basis": "impact unit/kg plastic input",
            "factor_note": "Mass-weighted net route factor with virgin-polymer production and pharmaceutical-grade SUT component manufacturing reported separately. Avoided material credits displace virgin polymer only; aggregate end-of-life treatment does not label EfW treatment as autoclaving.",
        })

        wastewater_m3 = float(
            sites["wastewater_to_reuse_m3"].sum()
            + sites["effluent_discharged_m3"].sum()
        )
        rows.append({
            "case_id": cid,
            "case_name": case["case_name"],
            "category": "Water",
            "selected_route_or_source": water_text(sites),
            "annual_quantity": wastewater_m3,
            "quantity_unit": "m3 wastewater/year",
            "weighted_private_cost_coefficient": costs[cid]["Water OPEX"] / wastewater_m3,
            "private_cost_coefficient_unit": "net operating GBP/m3 wastewater; excludes annualised CAPEX",
            "weighted_gwp_factor": impacts[cid]["gwp"]["Water"] / wastewater_m3,
            "weighted_fossil_resource_factor": impacts[cid]["fossil_resource"]["Water"] / wastewater_m3,
            "weighted_water_scarcity_factor": impacts[cid]["water_scarcity"]["Water"] / wastewater_m3,
            "weighted_human_health_daly_factor": impacts[cid]["human_health_daly"]["Water"] / wastewater_m3,
            "weighted_ecosystem_quality_factor": impacts[cid]["ecosystem_quality"]["Water"] / wastewater_m3,
            "environmental_factor_unit_basis": "impact unit/m3 wastewater",
            "factor_note": "Includes eligible utility-water supply plus the selected treatment/discharge or reuse route.",
        })

        biomass_kg = float(sites["biomass_kg_per_year"].sum())
        rows.append({
            "case_id": cid,
            "case_name": case["case_name"],
            "category": "Biomass",
            "selected_route_or_source": "; ".join(sorted(sites["biomass_route"].unique())),
            "annual_quantity": biomass_kg / 1000.0,
            "quantity_unit": "t biomass/year",
            "weighted_private_cost_coefficient": costs[cid]["Biomass"] / (biomass_kg / 1000.0),
            "private_cost_coefficient_unit": "effective GBP/t input",
            "weighted_gwp_factor": impacts[cid]["gwp"]["Biomass"] / biomass_kg,
            "weighted_fossil_resource_factor": impacts[cid]["fossil_resource"]["Biomass"] / biomass_kg,
            "weighted_water_scarcity_factor": impacts[cid]["water_scarcity"]["Biomass"] / biomass_kg,
            "weighted_human_health_daly_factor": impacts[cid]["human_health_daly"]["Biomass"] / biomass_kg,
            "weighted_ecosystem_quality_factor": impacts[cid]["ecosystem_quality"]["Biomass"] / biomass_kg,
            "environmental_factor_unit_basis": "impact unit/kg biomass input",
            "factor_note": "Site-level AD route choice; uncapped external steam displacement is a prospective assumption.",
        })

        vehicle_km = float(mix["vehicle_km_per_year"].sum())
        transport_cost = costs[cid]["Diesel HGV"] + costs[cid]["Electric HGV"]
        transport_gwp = (
            impacts[cid]["gwp"]["Diesel HGV"]
            + impacts[cid]["gwp"]["Electric HGV"]
        )
        rows.append({
            "case_id": cid,
            "case_name": case["case_name"],
            "category": "Transport",
            "selected_route_or_source": transport_text(mix),
            "annual_quantity": vehicle_km,
            "quantity_unit": "vehicle-km/year",
            "weighted_private_cost_coefficient": transport_cost / vehicle_km,
            "private_cost_coefficient_unit": "GBP/vehicle-km",
            "weighted_gwp_factor": transport_gwp / vehicle_km,
            "weighted_fossil_resource_factor": np.nan,
            "weighted_water_scarcity_factor": np.nan,
            "weighted_human_health_daly_factor": np.nan,
            "environmental_factor_unit_basis": "kg CO2e/vehicle-km for GWP",
            "factor_note": "Fossil-resource, water-scarcity and endpoint human-health transport factors are unavailable and are not invented.",
        })
    output = pd.DataFrame(rows)
    output["gwp_method"] = rf["gwp_method"].dropna().astype(str).iloc[0]
    output["water_scarcity_method"] = rf["water_scarcity_method"].dropna().astype(str).iloc[0]
    output["water_scarcity_regionalisation"] = (
        "site-specific AWARE 2.0 foreground; native-geography background"
    )
    return output


# -----------------------------------------------------------------------------
def build_improvement_split_frame(
    summary: pd.DataFrame,
    category_breakdown: pd.DataFrame,
) -> pd.DataFrame:
    """Build a six-outcome improvement matrix versus the base case.

    Positive values are reductions relative to the base case. Each total change
    is decomposed into manufacturing-electricity and industrial-symbiosis (IS)
    percentage-point contributions. Net freshwater has no electricity term, so
    its full change is assigned to IS water routing.
    """
    base_rows = summary.loc[summary["case_id"].eq("BAU")]
    require(len(base_rows) == 1, "The summary table must contain one base case.")
    base = base_rows.iloc[0]
    definitions = [
        ("gwp", "GWP", "gwp_kgco2e_per_year"),
        ("fossil_resource", "Fossil resources", "fossil_resource_kg_oil_eq_per_year"),
        ("water_scarcity", "Water scarcity", "water_scarcity_m3_world_eq_per_year"),
        ("human_health_daly", "Human health", "human_health_daly_per_year"),
        ("ecosystem_quality", "Ecosystem quality", "ecosystem_quality_species_year_per_year"),
        ("net_water", "Freshwater", "net_freshwater_withdrawal_m3_per_year"),
        ("private_cost", "Annual cost", "private_annual_cost_gbp"),
    ]
    rows = []
    for case in summary.itertuples(index=False):
        is_base = str(case.case_id) == "BAU"
        subset = category_breakdown.loc[category_breakdown["case_id"].eq(case.case_id)]
        for metric, label, total_column in definitions:
            base_total = float(base[total_column])
            total_value = float(getattr(case, total_column))
            if is_base:
                total_improvement = 0.0
                electricity_pp = 0.0
                is_pp = 0.0
            else:
                total_improvement = 100.0 * (base_total - total_value) / abs(base_total)
                metric_rows = subset.loc[subset["metric"].eq(metric)]
                if metric == "net_water":
                    electricity_saving = 0.0
                    is_saving = float(metric_rows["saving_vs_bau_annual"].sum())
                else:
                    electricity_saving = float(
                        metric_rows.loc[
                            metric_rows["reporting_block"].eq("Manufacturing electricity"),
                            "saving_vs_bau_annual",
                        ].sum()
                    )
                    is_saving = float(
                        metric_rows.loc[
                            metric_rows["reporting_block"].isin([
                                "Industrial symbiosis routing",
                                "Fixed manufacturing background",
                            ]),
                            "saving_vs_bau_annual",
                        ].sum()
                    )
                electricity_pp = 100.0 * electricity_saving / abs(base_total)
                is_pp = 100.0 * is_saving / abs(base_total)
                require(
                    abs((electricity_pp + is_pp) - total_improvement) <= 1e-6,
                    f"Improvement decomposition does not reconcile for {case.case_id}/{metric}: "
                    f"electricity={electricity_pp}, IS={is_pp}, total={total_improvement}",
                )
            rows.append({
                "case_id": case.case_id,
                "case_name": case.case_name,
                "metric_key": metric,
                "metric": label,
                "is_base": is_base,
                "total_improvement_pct": total_improvement,
                "electricity_percentage_points": electricity_pp,
                "is_percentage_points": is_pp,
            })
    return pd.DataFrame(rows)










def build_route_results_table(cases, data_by_case) -> pd.DataFrame:
    rows = []
    pretty_route = {
        "landfill_all": "Landfill",
        "incineration_efw_heat_electricity": "EfW",
        "mechanical_recycling_polyolefins": "Mechanical recycling – polyolefins",
        "mechanical_recycling_tpe_tubing": "Mechanical recycling – TPE tubing",
    }
    pretty_mode = {
        "diesel_articulated_hgv": "Diesel HGV",
        "battery_electric_articulated_hgv": "Battery-electric HGV",
    }
    for case in cases:
        cid = str(case["case_id"])
        data = data_by_case[cid]
        routes = case["routes"]
        mix = case["mix"]
        route_agg = routes.groupby(["route"], as_index=False).agg(
            feed_t=("tonnes_per_year", "sum"),
            recovered_t=("recovered_tonnes", "sum"),
            residual_t=("residual_tonnes", "sum"),
        )
        mix_agg = mix.groupby(["route"], as_index=False).agg(
            transport_modes=("transport_mode", lambda s: "; ".join(
                sorted({pretty_mode.get(str(v), str(v)) for v in s})
            )),
            journeys=("journeys_per_year", "sum"),
            vehicle_km=("vehicle_km_per_year", "sum"),
        )
        merged = route_agg.merge(mix_agg, on="route", how="left")
        total = float(route_agg["feed_t"].sum())
        for row in merged.itertuples(index=False):
            route_rows = routes.loc[routes["route"].astype(str).eq(str(row.route))]
            autoclaved_t = 0.0
            for rr in route_rows.itertuples(index=False):
                autoclaved_t += float(rr.tonnes_per_year) * data.autoclave_required[
                    str(rr.polymer), str(rr.route)
                ]
            rows.append({
                "case_id": cid,
                "case": str(case["case_name"]).replace("<br>", " – "),
                "route": pretty_route.get(str(row.route), str(row.route).replace("_", " ").title()),
                "feed_t_per_year": float(row.feed_t),
                "share_of_plastic_pct": 100.0 * float(row.feed_t) / total,
                "autoclaved_t_per_year": autoclaved_t,
                "recovered_t_per_year": float(row.recovered_t),
                "residual_t_per_year": float(row.residual_t),
                "transport_mode": row.transport_modes if isinstance(row.transport_modes, str) else "—",
                "journeys_per_year": float(row.journeys or 0.0),
                "vehicle_km_per_year": float(row.vehicle_km or 0.0),
            })
    return pd.DataFrame(rows)




def write_comparison_workbook(
    summary: pd.DataFrame,
    improvement_split: pd.DataFrame,
    decisions: pd.DataFrame,
    route_results: pd.DataFrame,
    category_breakdown: pd.DataFrame,
    selected_factors: pd.DataFrame,
    forced_location_audit: pd.DataFrame,
) -> Path:
    """Write one consolidated drill-down workbook without changing model logic."""
    output = OUTPUT_DIR / "routing_comparison_results.xlsx"
    daly = summary[[
        "case_name", "human_health_daly_per_year",
        "avoided_human_health_daly_vs_bau", "human_health_daly_reduction_vs_bau_pct",
    ]].rename(columns={
        "case_name": "Solution",
        "human_health_daly_per_year": "DALY/yr",
        "avoided_human_health_daly_vs_bau": "Avoided DALY vs BAU",
        "human_health_daly_reduction_vs_bau_pct": "Reduction vs BAU [%]",
    })
    ecosystem = summary[[
        "case_name", "ecosystem_quality_species_year_per_year",
        "avoided_ecosystem_quality_vs_bau", "ecosystem_quality_reduction_vs_bau_pct",
    ]].rename(columns={
        "case_name": "Solution",
        "ecosystem_quality_species_year_per_year": "species.yr/yr",
        "avoided_ecosystem_quality_vs_bau": "Avoided species.yr vs BAU",
        "ecosystem_quality_reduction_vs_bau_pct": "Reduction vs BAU [%]",
    })
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sheets = [
        ('Case summary', summary),
        ('Improvement split', improvement_split),
        ('Selected decisions', decisions),
        ('Plastic route results', route_results),
        ('Contribution data', category_breakdown),
        ('Selected factors', selected_factors),
        ('EfW location audit', forced_location_audit),
        ('Human health DALY', daly),
        ('Ecosystem quality', ecosystem),
    ]
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for name, frame in sheets:
            frame.to_excel(writer, sheet_name=name, index=False)
    return output


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------
def main() -> None:
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Main results read from: {MAIN_RESULT_DIR}")
    print(f"Comparison outputs written to: {OUTPUT_DIR}")
    for name, path in INPUTS.items():
        require(path.exists(), f"Missing input {name}: {path}")

    outputs = load_existing_outputs()
    bau = extract_existing_case(outputs, "BAU", EXISTING_CASES["BAU"])
    optimal = extract_existing_case(
        outputs,
        "OPT_COST_PLUS_CARBON_VALUE",
        EXISTING_CASES["OPT_COST_PLUS_CARBON_VALUE"],
    )
    forced, forced_location_audit, core = solve_forced_efw()

    # Load exact coefficients for post-processing the already-solved BAU and
    # optimal cases. No full objective/Pareto rerun is performed here.
    with tempfile.TemporaryDirectory(prefix="comparison_coefficients_") as tmp:
        tmp = Path(tmp)
        compat_params = tmp / "MILP_params_compat.json"
        make_compatibility_params(INPUTS["params"], compat_params)
        core.INPUT_FILES.update({
            "supply_chain": INPUTS["supply_chain"],
            "eligibility": INPUTS["eligibility"],
            "lca": INPUTS["lca"],
            "superpro_master": INPUTS["superpro_master"],
            "params": compat_params,
        })
        base_data = core.load_model_data()

    total_batches = float(sum(base_data.batches.values()))
    require(
        abs(total_batches - forced["sites"]["annual_batches"].sum()) <= 1e-6,
        "Case batch totals differ.",
    )

    rf = pd.read_excel(INPUTS["lca"], sheet_name="route_factor_matrix")
    cases = [bau, forced, optimal]
    data_by_case = {
        bau["case_id"]: base_data,
        optimal["case_id"]: base_data,
        forced["case_id"]: forced["data"],
    }

    impacts = {
        case["case_id"]: calculate_impact_components(
            case, data_by_case[case["case_id"]], rf, total_batches
        )
        for case in cases
    }
    costs = {
        case["case_id"]: calculate_cost_components(
            case, data_by_case[case["case_id"]]
        )
        for case in cases
    }
    waters = {
        case["case_id"]: calculate_net_water_components(case)
        for case in cases
    }

    summary, decisions, category_breakdown = build_tables(
        cases, impacts, costs, waters, total_batches, rf
    )
    selected_factors = build_selected_factor_table(
        cases, impacts, costs, data_by_case, rf, total_batches
    )
    route_results = build_route_results_table(cases, data_by_case)
    improvement_split = build_improvement_split_frame(summary, category_breakdown)

    # Persist underlying data exactly once at a compact reporting level.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    forced_location_audit.to_csv(OUTPUT_DIR / "forced_efw_location_audit.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "routing_case_summary.csv", index=False)
    improvement_split.to_csv(OUTPUT_DIR / "routing_improvement_split.csv", index=False)
    decisions.to_csv(OUTPUT_DIR / "routing_case_decisions.csv", index=False)
    category_breakdown.to_csv(OUTPUT_DIR / "routing_case_category_breakdown.csv", index=False)
    selected_factors.to_csv(OUTPUT_DIR / "routing_case_selected_lcia_factors.csv", index=False)
    route_results.to_csv(OUTPUT_DIR / "routing_case_route_results.csv", index=False)
    summary[[
        "case_name", "human_health_daly_per_year",
        "avoided_human_health_daly_vs_bau", "human_health_daly_reduction_vs_bau_pct",
    ]].rename(columns={
        "case_name": "Solution",
        "human_health_daly_per_year": "DALY/yr",
        "avoided_human_health_daly_vs_bau": "Avoided DALY vs BAU",
        "human_health_daly_reduction_vs_bau_pct": "Reduction vs BAU [%]",
    }).to_csv(OUTPUT_DIR / "human_health_daly_comparison.csv", index=False)
    summary[[
        "case_name", "ecosystem_quality_species_year_per_year",
        "avoided_ecosystem_quality_vs_bau", "ecosystem_quality_reduction_vs_bau_pct",
    ]].rename(columns={
        "case_name": "Solution",
        "ecosystem_quality_species_year_per_year": "species.yr/yr",
        "avoided_ecosystem_quality_vs_bau": "Avoided species.yr vs BAU",
        "ecosystem_quality_reduction_vs_bau_pct": "Reduction vs BAU [%]",
    }).to_csv(OUTPUT_DIR / "ecosystem_quality_comparison.csv", index=False)
    forced["routes"].to_csv(OUTPUT_DIR / "forced_efw_polymer_routing.csv", index=False)
    forced["sites"].to_csv(OUTPUT_DIR / "forced_efw_site_decisions.csv", index=False)
    forced["mix"].to_csv(OUTPUT_DIR / "forced_efw_route_mix.csv", index=False)

    workbook_path = write_comparison_workbook(
        summary,
        improvement_split,
        decisions,
        route_results,
        category_breakdown,
        selected_factors,
        forced_location_audit,
    )

    display_cols = [
        "case_name",
        "low_batch_location",
        "private_annual_cost_gbp",
        "pct_change_vs_bau__private_cost",
        "cost_plus_carbon_value_gbp",
        "pct_change_vs_bau__cost_plus_carbon",
        "gwp_kgco2e_per_year",
        "pct_change_vs_bau__gwp",
        "fossil_resource_kg_oil_eq_per_year",
        "pct_change_vs_bau__fossil_resource",
        "water_scarcity_m3_world_eq_per_year",
        "pct_change_vs_bau__water_scarcity",
        "human_health_daly_per_year",
        "avoided_human_health_daly_vs_bau",
        "human_health_daly_reduction_vs_bau_pct",
        "ecosystem_quality_species_year_per_year",
        "avoided_ecosystem_quality_vs_bau",
        "ecosystem_quality_reduction_vs_bau_pct",
        "net_freshwater_withdrawal_m3_per_year",
        "pct_change_vs_bau__net_freshwater",
        "residual_disposal_t_per_year",
    ]
    print("\nROUTING CASE COMPARISON")
    print(summary[display_cols].to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    print("\nDECISIONS")
    print(decisions.to_string(index=False))
    print("\nPLASTIC ROUTE RESULTS")
    print(route_results.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    print("\nFORCED EFW LOCATION AUDIT")
    print(forced_location_audit.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    print(f"\nConsolidated workbook: {workbook_path}")



if __name__ == "__main__":
    main()
