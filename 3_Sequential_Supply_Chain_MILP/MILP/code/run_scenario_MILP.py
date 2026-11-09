from __future__ import annotations

"""Scenario analysis runner for the validated industrial-symbiosis MILP.

The script imports the existing ``run_master_MILP.py`` model rather than duplicating
its formulation.  It then applies one scenario family at a time, solves every
case with the same GAMS/Pyomo interface, and writes detailed scenario tables.

Comparison and constraint references
------------------------------------
All reported changes use the current central-carbon-value optimum as the
comparison baseline. A separate private-cost/grid reference is solved only where
scenario definitions require a BAU quantity for a cap or site constraint.
Carbon-cap and residual-waste-cap cases are solved as cost-minimum constrained
policy cases so that the cap is not stacked on top of the central carbon value.
"""

import copy
import importlib.util
import json
import math
import os
import shutil
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, vstack


# -----------------------------------------------------------------------------
# Paths and controls
# -----------------------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
INPUT_DIR = PROJECT_ROOT.parent / "parameters"
RESULT_ROOT = PROJECT_ROOT / "results"
RUN_TAG = os.getenv("IS_SCENARIO_RUN_TAG", "20260729").strip()
RESULT_DIR = RESULT_ROOT / f"industrial_symbiosis_scenario_results_{RUN_TAG}"
GAMS_LOG_DIR = RESULT_DIR / "gams_logs"
GAMS_WORK_DIR = RESULT_DIR / "gams_work"
GAMS_SOLVE_REPORT = RESULT_DIR / "gams_solve_report.csv"
RESULTS_WORKBOOK = RESULT_DIR / f"industrial_symbiosis_scenario_MILP_results_{RUN_TAG}.xlsx"
REFERENCE_FILE = RESULT_DIR / "scenario_reference_metrics.json"
MANIFEST_FILE = RESULT_DIR / "scenario_manifest.json"

BASELINE_CASE_ID = "BASE_CURRENT_POLICY"
BASELINE_CASE_NAME = "Current optimum (central carbon value)"
PRIVATE_REFERENCE_CASE_ID = "REFERENCE_JSON_BASE_2026"
BASELINE_CARBON_SCENARIO_ID = os.getenv(
    "IS_SCENARIO_BASELINE_CARBON", "CARBON_VALUE_CENTRAL_2026"
).strip()

STRAP_ROUTE = "chemical_recycling_strap_hypothetical"

# Materiality is assessed on physical/private-cost outputs, not merely a changed
# carbon-accounting objective value.
MATERIALITY_ABSOLUTE = {
    "cost": 1_000.0,          # GBP/year
    "gwp": 1_000.0,          # kg CO2e/year
    "freshwater": 1.0,       # m3/year
    "circularity_loss": 0.1, # t/year
    "residual": 0.1,         # t/year
}
MATERIALITY_RELATIVE = float(os.getenv("IS_SCENARIO_MATERIAL_RELATIVE", "0.001"))
DECISION_TOL = 1e-6


@dataclass(frozen=True)
class ScenarioSpec:
    family: str
    scenario_id: str
    scenario_name: str
    description: str
    payload: Mapping[str, Any]
    baseline_setting_id: Optional[str] = None
    is_baseline: bool = False


@dataclass
class ScenarioBuild:
    spec: ScenarioSpec
    model: Any
    applied: Dict[str, Any]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _first_existing(candidates: Iterable[Path], description: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    rendered = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not locate {description}. Checked:\n  - {rendered}")


def resolve_core_module_path() -> Path:
    explicit = os.getenv("IS_MASTER_MILP_MODULE")
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.extend([
        CODE_DIR / "run_master_MILP.py",
        CODE_DIR / "master_MILP(11).py",
        INPUT_DIR / "run_master_MILP.py",
    ])
    return _first_existing(candidates, "the validated master MILP module")


def resolve_scenario_file() -> Path:
    explicit = os.getenv("IS_SCENARIO_FILE")
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.extend([
        INPUT_DIR / "MILP_scenarios.json",
        INPUT_DIR / "scenarios(10).json",
        INPUT_DIR / "scenarios(8).json",
        CODE_DIR / "MILP_scenarios.json",
        CODE_DIR / "scenarios(10).json",
        CODE_DIR / "scenarios(8).json",
    ])
    return _first_existing(candidates, "the scenario JSON file")


def import_core_module():
    path = resolve_core_module_path()
    spec = importlib.util.spec_from_file_location("is_master_milp_for_scenarios", path)
    require(spec is not None and spec.loader is not None, f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def configure_core_outputs(core) -> None:
    """Keep all scenario GAMS artefacts separate from the base-case run."""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    GAMS_LOG_DIR.mkdir(parents=True, exist_ok=True)
    GAMS_WORK_DIR.mkdir(parents=True, exist_ok=True)
    core.RESULT_DIR = RESULT_DIR
    core.GAMS_LOG_DIR = GAMS_LOG_DIR
    core.GAMS_WORK_DIR = GAMS_WORK_DIR
    core.GAMS_SOLVE_REPORT = GAMS_SOLVE_REPORT
    core.GAMS_PREFLIGHT_LOG = GAMS_LOG_DIR / "0000_gams_preflight.log"
    core._GAMS_SOLVE_COUNTER = 0
    core._GAMS_PREFLIGHT_DONE = False


def reset_outputs() -> None:
    if RESULT_DIR.exists():
        for path in RESULT_DIR.iterdir():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    GAMS_LOG_DIR.mkdir(parents=True, exist_ok=True)
    GAMS_WORK_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_inputs(core) -> Tuple[Any, Dict[str, Any], Dict[str, Any], Path]:
    scenario_file = resolve_scenario_file()
    scenario_doc = load_json(scenario_file)
    validate_scenario_document(scenario_doc)
    params_file = Path(core.INPUT_FILES["params"])
    require(params_file.exists(), f"Missing MILP parameter file: {params_file}")
    params = load_json(params_file)
    base_data = core.load_model_data()
    return base_data, params, scenario_doc, scenario_file


def _nominal_multiplier(params: Mapping[str, Any], key: str) -> float:
    return float(
        params["currency_and_price_year_conversions"]
        ["multipliers_to_nominal_GBP_2026"][key]
    )


def baseline_setting_ids(scenario_doc: Mapping[str, Any]) -> Dict[str, str]:
    """Return the explicit reporting-baseline settings from the scenario file."""
    if "reporting_baseline" in scenario_doc:
        settings = dict(scenario_doc["reporting_baseline"]["settings"])
    else:
        settings = dict(scenario_doc["base_case"]["settings"])
        settings["lifecycle_carbon_value"] = BASELINE_CARBON_SCENARIO_ID
    return settings


def reporting_carbon_scenario_id(scenario_doc: Mapping[str, Any]) -> str:
    return str(
        baseline_setting_ids(scenario_doc).get(
            "lifecycle_carbon_value", BASELINE_CARBON_SCENARIO_ID
        )
    )


def optimisation_objective_for_spec(spec: ScenarioSpec) -> str:
    """Do not stack the central carbon value onto explicit cap instruments."""
    if spec.family in {"carbon_cap", "residual_waste_policy"}:
        return "cost"
    return "policy_cost"


def validate_scenario_document(scenario_doc: Mapping[str, Any]) -> None:
    """Fail early when scenario semantics would create misleading comparisons."""
    require("base_case" in scenario_doc, "Scenario JSON requires base_case.")
    settings = baseline_setting_ids(scenario_doc)
    families = scenario_doc.get("scenario_families", {})
    for family in settings:
        require(family in families, f"Reporting baseline refers to unknown family {family!r}.")

    electricity = families.get("industrial_electricity_conditions", {}).get("scenarios", {})
    if "ELECTRICITY_RENEWABLE_CPPA" in electricity:
        cppa = electricity["ELECTRICITY_RENEWABLE_CPPA"]
        require(not bool(cppa.get("force_source", False)),
                "The cPPA cost scenario must not force renewable selection.")
        rule = cppa.get("cost_rule", {})
        require(rule.get("type") in {"grid_price_multiplier", "multiplier_on_loaded"},
                "The cPPA scenario must be expressed as a transparent premium/discount rule.")
    if "ELECTRICITY_BICS_2027_CENTRAL" in electricity:
        bics = electricity["ELECTRICITY_BICS_2027_CENTRAL"]
        require(set(bics.get("target_sources", [])) == {"grid", "wind"},
                "BICS must explicitly state whether the bill reduction applies to both delivered sources.")


def _display_name(scenario_id: str) -> str:
    return scenario_id.replace("_", " ").title()


def list_scenario_specs(scenario_doc: Mapping[str, Any]) -> List[ScenarioSpec]:
    settings = baseline_setting_ids(scenario_doc)
    specs: List[ScenarioSpec] = [
        ScenarioSpec(
            family="baseline",
            scenario_id=BASELINE_CASE_ID,
            scenario_name=BASELINE_CASE_NAME,
            description=(
                "Loaded 2026 model parameters with the central lifecycle carbon value; "
                "all reported scenario deltas use this optimum as the baseline."
            ),
            payload={},
            is_baseline=True,
        )
    ]

    families = scenario_doc.get("scenario_families", {})
    for family, family_doc in families.items():
        if family == "CHEMICAL_RECYCLING_STRAP_HYPOTHETICAL":
            specs.append(
                ScenarioSpec(
                    family=family,
                    scenario_id=family,
                    scenario_name="Hypothetical STRAP chemical recycling",
                    description=(
                        "Hypothetical external shared STRAP facility for coupled LLDPE and EVOH recovery."
                    ),
                    payload=family_doc,
                    baseline_setting_id=None,
                )
            )
            continue

        scenarios = family_doc.get("scenarios", {})
        base_id = settings.get(family)
        require(base_id is not None, f"No baseline setting is defined for scenario family {family!r}.")
        require(base_id in scenarios, f"Baseline setting {base_id!r} is missing from family {family!r}.")
        for scenario_id, payload in scenarios.items():
            if scenario_id == base_id:
                continue
            specs.append(
                ScenarioSpec(
                    family=family,
                    scenario_id=scenario_id,
                    scenario_name=_display_name(scenario_id),
                    description=str(payload.get("description", "")),
                    payload=payload,
                    baseline_setting_id=base_id,
                )
            )
    return specs


def scenario_spec_by_id(scenario_doc: Mapping[str, Any], case_id: str) -> ScenarioSpec:
    matches = [spec for spec in list_scenario_specs(scenario_doc) if spec.scenario_id == case_id]
    require(len(matches) == 1, f"Expected one scenario definition for {case_id}; found {len(matches)}.")
    return matches[0]


def _carbon_value_nominal(
    scenario_doc: Mapping[str, Any], params: Mapping[str, Any], scenario_id: str
) -> Tuple[float, float]:
    payload = scenario_doc["scenario_families"]["lifecycle_carbon_value"]["scenarios"][scenario_id]
    source_value = float(payload["carbon_value_gbp_per_tco2e"])
    nominal_value = source_value * _nominal_multiplier(params, "GBP_2020_real")
    return source_value, nominal_value


def make_baseline_data(
    base_data: Any, scenario_doc: Mapping[str, Any], params: Mapping[str, Any]
) -> Tuple[Any, Dict[str, Any]]:
    data = copy.deepcopy(base_data)
    carbon_scenario_id = reporting_carbon_scenario_id(scenario_doc)
    raw, nominal = _carbon_value_nominal(
        scenario_doc, params, carbon_scenario_id
    )
    data.lifecycle_carbon_value_gbp_per_t = nominal
    return data, {
        "carbon_value_scenario_id": carbon_scenario_id,
        "carbon_value_source_gbp_per_tco2e": raw,
        "carbon_value_nominal_2026_gbp_per_tco2e": nominal,
    }


def make_private_reference_data(base_data: Any) -> Any:
    """JSON BASE_2026: private cost, current grid electricity and no carbon value."""
    data = copy.deepcopy(base_data)
    data.lifecycle_carbon_value_gbp_per_t = 0.0
    for site in data.sites:
        data.electricity_available[site, "grid"] = 1
        data.electricity_available[site, "wind"] = 0
    return data



def _strap_distance_by_location(payload: Mapping[str, Any]) -> Dict[str, float]:
    transport = payload.get("transport", {})
    distances = transport.get("distance_km_by_location")
    require(isinstance(distances, Mapping) and distances,
            "Hypothetical STRAP scenario requires explicit distance_km_by_location in MILP_scenarios.json")
    out = {str(k): float(v) for k, v in distances.items()}
    require(all(value > 0 for value in out.values()), "STRAP distances must be positive")
    return out


def add_strap_route(
    data: Any,
    params: Mapping[str, Any],
    payload: Mapping[str, Any],
    applied: Dict[str, Any],
) -> None:
    require(STRAP_ROUTE not in data.routes, f"Route {STRAP_ROUTE} already exists in the base model.")
    polymer_params = payload["polymer_parameters"]
    eligible_polymers = ["LLDPE", "EVOH"]
    require(set(eligible_polymers).issubset(data.polymers), "STRAP requires LLDPE and EVOH in the model inventory.")

    data.routes.append(STRAP_ROUTE)
    data.route_lca_pathway[STRAP_ROUTE] = STRAP_ROUTE
    distance_by_location = _strap_distance_by_location(payload)
    processing_gbp_per_kg = float(
        payload["shared_processing_cost"]["recommended_system_processing_cost_gbp_per_kg_input"]
    )

    for polymer in data.polymers:
        eligible = int(polymer in eligible_polymers)
        data.eligibility[polymer, STRAP_ROUTE] = eligible
        data.autoclave_required[polymer, STRAP_ROUTE] = eligible
        if eligible:
            pdoc = polymer_params[polymer]
            recovery = float(pdoc["recovery_fraction"])
            residual = float(pdoc["residual_to_efw_fraction"])
            require(abs(recovery + residual - 1.0) <= 1e-9, f"STRAP mass balance fails for {polymer}.")
            data.recovery_yield[polymer, STRAP_ROUTE] = recovery
            data.residual_fraction[polymer, STRAP_ROUTE] = residual
            data.route_cost_gbp_per_t[polymer, STRAP_ROUTE] = 1000.0 * processing_gbp_per_kg
            data.recovered_value_gbp_per_t[polymer, STRAP_ROUTE] = (
                1000.0 * float(pdoc["recyclate_value_gbp_per_kg_output"])
            )
            data.value_capture[polymer, STRAP_ROUTE] = 1.0
            data.plastic_gwp_kg_per_kg[polymer, STRAP_ROUTE] = float(
                pdoc["optimiser_gwp_proxy"]["strap_route_kgco2e_per_kg_input"]
            )
            reference_route = "landfill_all"
            data.virgin_polymer_gwp_kg_per_kg[polymer, STRAP_ROUTE] = float(
                data.virgin_polymer_gwp_kg_per_kg[polymer, reference_route]
            )
            data.sut_component_manufacturing_gwp_kg_per_kg[polymer, STRAP_ROUTE] = float(
                data.sut_component_manufacturing_gwp_kg_per_kg[polymer, reference_route]
            )
            # The supplied STRAP optimiser proxy is aggregate and does not expose
            # a distinct avoided-material credit. Keep it in the residual route
            # balance rather than inventing a pharma- or polymer-production credit.
            data.avoided_virgin_material_credit_gwp_kg_per_kg[polymer, STRAP_ROUTE] = 0.0
        else:
            data.recovery_yield[polymer, STRAP_ROUTE] = 0.0
            data.residual_fraction[polymer, STRAP_ROUTE] = 1.0
            data.route_cost_gbp_per_t[polymer, STRAP_ROUTE] = 0.0
            data.recovered_value_gbp_per_t[polymer, STRAP_ROUTE] = 0.0
            data.value_capture[polymer, STRAP_ROUTE] = 0.0
            data.plastic_gwp_kg_per_kg[polymer, STRAP_ROUTE] = 0.0
            data.virgin_polymer_gwp_kg_per_kg[polymer, STRAP_ROUTE] = 0.0
            data.sut_component_manufacturing_gwp_kg_per_kg[polymer, STRAP_ROUTE] = 0.0
            data.avoided_virgin_material_credit_gwp_kg_per_kg[polymer, STRAP_ROUTE] = 0.0

    for site in data.sites:
        location = str(data.site_location[site])
        require(location in distance_by_location, f"No STRAP distance is available for {location}.")
        data.distance_km[site, STRAP_ROUTE] = float(distance_by_location[location])
        for mode in data.transport_modes:
            data.roundtrip_multiplier[site, STRAP_ROUTE, mode] = 2.0

    capacity = float(payload["facility_context"]["paper_reference_capacity_tonnes_per_year"])
    data.route_capacities_t[STRAP_ROUTE] = capacity
    applied.update({
        "strap_route": STRAP_ROUTE,
        "strap_processing_cost_gbp_per_t_input": 1000.0 * processing_gbp_per_kg,
        "strap_capacity_t_per_year": capacity,
        "strap_distance_km_by_location": distance_by_location,
        "strap_autoclave_required": True,
        "strap_coupling": "equal site-level route shares for LLDPE and EVOH",
    })


def _landfill_gate_component_from_loaded_model(
    data: Any,
    scenario_doc: Mapping[str, Any],
) -> Tuple[float, float]:
    current_id = baseline_setting_ids(scenario_doc)["landfill_tax"]
    current_payload = scenario_doc["scenario_families"]["landfill_tax"]["scenarios"][current_id]
    current_tax = float(current_payload["landfill_tax_gbp_per_tonne"])
    loaded_totals = {
        round(float(data.route_cost_gbp_per_t[polymer, "landfill_all"]), 9)
        for polymer in data.polymers
    }
    require(len(loaded_totals) == 1, "Loaded landfill totals differ by polymer.")
    loaded_total = loaded_totals.pop()
    gate_component = loaded_total - current_tax
    require(gate_component >= -1e-9, "Derived landfill gate-fee component is negative.")
    return gate_component, current_tax


def _apply_electricity_cost_rule(data: Any, payload: Mapping[str, Any]) -> Dict[str, Any]:
    targets = [str(source) for source in payload.get("target_sources", [])]
    rule = dict(payload.get("cost_rule", {}))
    rule_type = str(rule.get("type", ""))
    before = {source: float(data.electricity_cost_gbp_per_kwh[source]) for source in data.electricity_sources}
    require(not bool(payload.get("force_source", False)),
            "Electricity price scenarios must not force a source; use a separate availability scenario if required.")

    if rule_type == "loaded_baseline":
        targets = []
    elif rule_type == "absolute_gbp_per_kwh":
        value = float(rule["value"])
        require(value >= 0.0, "Electricity price cannot be negative.")
        for source in targets:
            require(source in data.electricity_sources, f"Unknown electricity source {source!r}.")
            data.electricity_cost_gbp_per_kwh[source] = value
    elif rule_type == "subtract_gbp_per_kwh_from_loaded":
        reduction = float(rule["value"])
        floor = float(rule.get("floor", 0.0))
        require(reduction >= 0.0 and floor >= 0.0, "Invalid electricity reduction/floor.")
        for source in targets:
            require(source in data.electricity_sources, f"Unknown electricity source {source!r}.")
            data.electricity_cost_gbp_per_kwh[source] = max(
                floor, before[source] - reduction
            )
    elif rule_type == "grid_price_multiplier":
        multiplier = float(rule["value"])
        require(multiplier >= 0.0, "Electricity price multiplier cannot be negative.")
        grid_price = before["grid"]
        for source in targets:
            require(source in data.electricity_sources, f"Unknown electricity source {source!r}.")
            data.electricity_cost_gbp_per_kwh[source] = grid_price * multiplier
    elif rule_type == "multiplier_on_loaded":
        multiplier = float(rule["value"])
        require(multiplier >= 0.0, "Electricity price multiplier cannot be negative.")
        for source in targets:
            require(source in data.electricity_sources, f"Unknown electricity source {source!r}.")
            data.electricity_cost_gbp_per_kwh[source] = before[source] * multiplier
    else:
        raise KeyError(f"Unknown electricity cost rule {rule_type!r}")

    after = {source: float(data.electricity_cost_gbp_per_kwh[source]) for source in data.electricity_sources}
    return {
        "electricity_cost_rule_type": rule_type,
        "electricity_target_sources": targets,
        "electricity_cost_before_by_source": before,
        "electricity_cost_after_by_source": after,
        "electricity_gwp_factors_changed": False,
        "electricity_availability_changed": False,
        "autoclave_route_coefficients_changed": False,
    }


def apply_data_scenario(
    data: Any,
    spec: ScenarioSpec,
    scenario_doc: Mapping[str, Any],
    params: Mapping[str, Any],
) -> Dict[str, Any]:
    applied: Dict[str, Any] = {
        "scenario_family": spec.family,
        "scenario_id": spec.scenario_id,
        "baseline_setting_id": spec.baseline_setting_id,
        "reporting_baseline_case_id": BASELINE_CASE_ID,
    }
    if spec.is_baseline:
        return applied

    family = spec.family
    payload = spec.payload

    if family == "lifecycle_carbon_value":
        raw = float(payload["carbon_value_gbp_per_tco2e"])
        nominal = raw * _nominal_multiplier(params, "GBP_2020_real")
        data.lifecycle_carbon_value_gbp_per_t = nominal
        applied.update({
            "carbon_value_source_gbp_per_tco2e": raw,
            "carbon_value_nominal_2026_gbp_per_tco2e": nominal,
        })

    elif family in {"carbon_cap", "residual_waste_policy"}:
        # These are alternative policy instruments. Their constraints are added
        # after the model is built and the central carbon value is removed so the
        # analysis does not apply two climate/circularity instruments at once.
        data.lifecycle_carbon_value_gbp_per_t = 0.0
        applied.update({
            "objective_override": "cost",
            "carbon_value_nominal_2026_gbp_per_tco2e": 0.0,
            "policy_instrument_stacked_with_carbon_value": False,
        })

    elif family == "landfill_tax":
        new_tax = float(payload["landfill_tax_gbp_per_tonne"])
        gate_component, current_tax = _landfill_gate_component_from_loaded_model(data, scenario_doc)
        total = gate_component + new_tax
        for polymer in data.polymers:
            data.route_cost_gbp_per_t[polymer, "landfill_all"] = total
        applied.update({
            "landfill_gate_component_loaded_gbp_per_t": gate_component,
            "landfill_current_tax_gbp_per_t": current_tax,
            "landfill_scenario_tax_gbp_per_t": new_tax,
            "landfill_total_cost_gbp_per_t": total,
        })

    elif family == "efw_uk_ets":
        allowance_source = float(payload["allowance_value_gbp_per_tco2e"])
        pass_through = float(payload["pass_through_fraction"])
        allowance_nominal = allowance_source * _nominal_multiplier(params, "GBP_2025_real")
        stack = params["efw_gross_fossil_stack_tco2_per_tonne_input"]
        additions = {}
        for polymer in data.polymers:
            factor = float(stack[polymer])
            addition = factor * allowance_nominal * pass_through
            data.route_cost_gbp_per_t[polymer, "incineration_efw_heat_electricity"] += addition
            additions[polymer] = addition
        applied.update({
            "ets_allowance_source_gbp_per_tco2e": allowance_source,
            "ets_allowance_nominal_2026_gbp_per_tco2e": allowance_nominal,
            "ets_pass_through_fraction": pass_through,
            "efw_ets_cost_addition_gbp_per_t_by_polymer": additions,
        })

    elif family == "recyclate_market":
        multiplier = float(payload["product_value_multiplier"])
        for polymer in ("LDPE", "LLDPE", "PP"):
            for route in data.routes:
                key = (polymer, route)
                if data.value_capture.get(key, 0.0) > 0.0:
                    data.recovered_value_gbp_per_t[key] *= multiplier
        applied["polyolefin_recyclate_value_multiplier"] = multiplier

    elif family == "industrial_electricity_conditions":
        applied.update(_apply_electricity_cost_rule(data, payload))

    elif family == "CHEMICAL_RECYCLING_STRAP_HYPOTHETICAL":
        add_strap_route(data, params, payload, applied)

    elif family == "local_mains_water_availability":
        # Constraint added after the model is built.
        pass
    else:
        raise KeyError(f"Scenario family {family!r} is not implemented.")

    return applied


def append_constraint(model: Any, coeff: np.ndarray, lb: float, ub: float) -> None:
    require(coeff.shape == (len(model.var_names),), "Scenario constraint coefficient has wrong shape.")
    row = coo_matrix(coeff.reshape(1, -1))
    model.A = vstack([model.A.tocsr(), row.tocsr()], format="coo")
    model.b_l = np.append(model.b_l, float(lb))
    model.b_u = np.append(model.b_u, float(ub))


def add_expression_upper_bound(model: Any, expression_name: str, limit: float) -> None:
    expression = model.expressions[expression_name]
    rhs = float(limit) - float(expression.const)
    append_constraint(model, expression.coeff.copy(), -np.inf, rhs)


def add_postbuild_scenario_constraints(
    model: Any,
    spec: ScenarioSpec,
    references: Mapping[str, Any],
    applied: Dict[str, Any],
) -> None:
    if spec.is_baseline:
        return
    family = spec.family
    payload = spec.payload

    if family == "carbon_cap":
        multiplier = float(payload["cap_multiplier"])
        reference = float(references["private_cost_reference"]["metrics"]["gwp"])
        limit = reference * multiplier
        add_expression_upper_bound(model, "gwp", limit)
        applied.update({
            "carbon_cap_reference_gwp_kgco2e_per_year": reference,
            "carbon_cap_multiplier": multiplier,
            "carbon_cap_limit_kgco2e_per_year": limit,
        })

    elif family == "residual_waste_policy":
        multiplier = float(payload["cap_multiplier"])
        reference = float(references["private_cost_reference"]["metrics"]["residual"])
        limit = reference * multiplier
        add_expression_upper_bound(model, "residual", limit)
        applied.update({
            "residual_cap_reference_t_per_year": reference,
            "residual_cap_multiplier": multiplier,
            "residual_cap_limit_t_per_year": limit,
        })

    elif family == "local_mains_water_availability":
        fractions = {str(k): float(v) for k, v in payload["supply_fraction_by_site"].items()}
        reference_by_location = references["private_cost_reference"]["mains_water_m3_by_location"]
        limits = {}
        for site in model.data.sites:
            location = str(model.data.site_location[site])
            require(location in fractions, f"No water-supply fraction for {location}.")
            require(location in reference_by_location, f"No reference mains-water demand for {location}.")
            limit = float(reference_by_location[location]) * fractions[location]
            coeff = np.zeros(len(model.var_names), dtype=float)
            coeff[model.index["B", site]] = model.data.fixed_mains_water_m3_per_batch
            coeff[model.index["f", site]] = 1.0
            append_constraint(model, coeff, -np.inf, limit)
            limits[location] = limit
        applied.update({
            "mains_water_supply_fraction_by_location": fractions,
            "mains_water_limit_m3_per_year_by_location": limits,
        })

    elif family == "CHEMICAL_RECYCLING_STRAP_HYPOTHETICAL":
        # Equal route shares preserve the actual site-specific LLDPE:EVOH mass ratio.
        for site in model.data.sites:
            coeff = np.zeros(len(model.var_names), dtype=float)
            for mode in model.data.transport_modes:
                lldpe_key = ("x", site, "LLDPE", STRAP_ROUTE, mode)
                evoh_key = ("x", site, "EVOH", STRAP_ROUTE, mode)
                if lldpe_key in model.index:
                    coeff[model.index[lldpe_key]] += 1.0 / model.data.polymer_kg_per_batch["LLDPE"]
                if evoh_key in model.index:
                    coeff[model.index[evoh_key]] -= 1.0 / model.data.polymer_kg_per_batch["EVOH"]
            append_constraint(model, coeff, 0.0, 0.0)


def build_case_model(
    core: Any,
    base_data: Any,
    params: Mapping[str, Any],
    scenario_doc: Mapping[str, Any],
    spec: ScenarioSpec,
    references: Mapping[str, Any],
) -> ScenarioBuild:
    data, baseline_applied = make_baseline_data(base_data, scenario_doc, params)
    applied = dict(baseline_applied)
    applied.update(apply_data_scenario(data, spec, scenario_doc, params))
    model = core.build_linear_model(data)
    add_postbuild_scenario_constraints(model, spec, references, applied)
    return ScenarioBuild(spec=spec, model=model, applied=applied)


def solve_private_reference(core: Any, base_data: Any) -> Tuple[Any, Any, Dict[str, Any]]:
    data = make_private_reference_data(base_data)
    model = core.build_linear_model(data)
    result = core.solve_model(
        model,
        PRIVATE_REFERENCE_CASE_ID,
        "cost",
        selection_method=(
            "JSON BASE_2026 reference: private annual network cost; grid electricity; "
            "no lifecycle carbon value; no scenario constraints"
        ),
        metadata={"scenario_family": "reference", "scenario_id": PRIVATE_REFERENCE_CASE_ID},
    )
    require(result.success, f"Private-cost scenario reference failed: {result.message}")
    routes, sites, mix = core.solution_decisions(model, result)
    reference = {
        "case_id": PRIVATE_REFERENCE_CASE_ID,
        "metrics": {key: float(value) for key, value in result.metrics.items()},
        "mains_water_m3_by_location": {
            str(row.location): float(row.net_mains_freshwater_withdrawal_m3)
            for row in sites.itertuples(index=False)
        },
        "low_batch_location": str(sites.loc[sites["is_low_batch_location"], "location"].iloc[0]),
        "selection_method": result.selection_method,
    }
    return model, result, reference


def _case_summary_success(
    core: Any,
    build: ScenarioBuild,
    result: Any,
) -> Dict[str, Any]:
    routes, sites, mix = core.solution_decisions(build.model, result)
    low = sites.loc[sites["is_low_batch_location"]]
    require(len(low) == 1, f"Expected one low-batch site in {result.case_id}.")
    total_electricity = float(sites["grid_kwh"].sum() + sites["wind_kwh"].sum())
    eligible_water = float((sites["fresh_utility_water_m3"] + sites["water_reused_m3"]).sum())
    # The supplied STRAP scenario has a GWP proxy but no DALY or ecosystem
    # factors. Keep these reporting metrics unavailable instead of assigning
    # unsupported values or treating a successful optimisation as a failure.
    missing_strap_lcia = STRAP_ROUTE in build.model.data.routes
    return {
        "case_id": result.case_id,
        "case_name": build.spec.scenario_name,
        "scenario_family": build.spec.family,
        "scenario_description": build.spec.description,
        "baseline_setting_id": build.spec.baseline_setting_id,
        "success": True,
        "solver_message": result.message,
        "solver_backend": result.metadata.get("solver_backend", ""),
        "selection_method": result.selection_method,
        "optimisation_objective": result.primary_objective,
        "objective_carbon_value_gbp_per_tco2e": float(build.model.data.lifecycle_carbon_value_gbp_per_t),
        "comparison_baseline_case_id": BASELINE_CASE_ID,
        "cost": float(result.metrics["cost"]),
        "policy_cost": float(result.metrics["policy_cost"]),
        "gwp": float(result.metrics["gwp"]),
        "human_health_daly": (np.nan if missing_strap_lcia else
                              float(core.evaluate_human_health_daly(build.model, result))),
        "ecosystem_quality": (np.nan if missing_strap_lcia else
                              float(core.evaluate_ecosystem_quality(build.model, result))),
        "lcia_reporting_note": (
            "STRAP DALY and ecosystem factors are not supplied; metrics unavailable"
            if missing_strap_lcia else ""
        ),
        "freshwater": float(result.metrics["freshwater"]),
        "circularity_loss": float(result.metrics["circularity_loss"]),
        "residual": float(result.metrics["residual"]),
        "recovered": float(result.metrics["recovered"]),
        "virgin_polymer_production_gwp_kgco2e_per_year": float(
            routes["virgin_polymer_production_gwp_kgco2e_per_year"].sum()
        ),
        "sut_component_manufacturing_gwp_kgco2e_per_year": float(
            routes["sut_component_manufacturing_gwp_kgco2e_per_year"].sum()
        ),
        "avoided_virgin_material_credit_gwp_kgco2e_per_year": float(
            routes["avoided_virgin_material_credit_gwp_kgco2e_per_year"].sum()
        ),
        "low_batch_location": str(low.iloc[0]["location"]),
        "wind_share": float(sites["wind_kwh"].sum() / total_electricity) if total_electricity > 0 else np.nan,
        "water_reuse_share": float(sites["water_reused_m3"].sum() / eligible_water) if eligible_water > 0 else np.nan,
        "decision_snapshot": core.compact_decision_text(build.model, result),
        **core.solution_validation(build.model, result),
    }


def _case_summary_failure(spec: ScenarioSpec, message: str) -> Dict[str, Any]:
    return {
        "case_id": spec.scenario_id,
        "case_name": spec.scenario_name,
        "scenario_family": spec.family,
        "scenario_description": spec.description,
        "baseline_setting_id": spec.baseline_setting_id,
        "success": False,
        "solver_message": message,
        "solver_backend": "",
        "selection_method": "",
        "optimisation_objective": "",
        "objective_carbon_value_gbp_per_tco2e": np.nan,
        "comparison_baseline_case_id": BASELINE_CASE_ID,
        "cost": np.nan,
        "policy_cost": np.nan,
        "gwp": np.nan,
        "human_health_daly": np.nan,
        "ecosystem_quality": np.nan,
        "freshwater": np.nan,
        "circularity_loss": np.nan,
        "residual": np.nan,
        "recovered": np.nan,
        "virgin_polymer_production_gwp_kgco2e_per_year": np.nan,
        "sut_component_manufacturing_gwp_kgco2e_per_year": np.nan,
        "avoided_virgin_material_credit_gwp_kgco2e_per_year": np.nan,
        "low_batch_location": "",
        "wind_share": np.nan,
        "water_reuse_share": np.nan,
        "decision_snapshot": "",
        "max_core_constraint_violation": np.nan,
        "max_integrality_violation": np.nan,
        "max_bound_violation": np.nan,
    }


def _normalised_route_map(routes: pd.DataFrame, case_id: str) -> Dict[Tuple[str, str, str, str], float]:
    subset = routes.loc[routes["case_id"].eq(case_id)]
    if subset.empty:
        return {}
    grouped = subset.groupby(["location", "polymer", "route", "transport_mode"])["tonnes_per_year"].sum()
    return {tuple(map(str, key)): float(value) for key, value in grouped.items() if abs(value) > DECISION_TOL}


def _map_changed(left: Mapping[Any, float], right: Mapping[Any, float], tol: float = DECISION_TOL) -> bool:
    keys = set(left) | set(right)
    return any(abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))) > tol for key in keys)


def assess_materiality(
    summary: pd.DataFrame,
    routes: pd.DataFrame,
) -> pd.DataFrame:
    base = summary.loc[summary["case_id"].eq(BASELINE_CASE_ID)].iloc[0]
    base_routes = _normalised_route_map(routes, BASELINE_CASE_ID)
    rows = []
    for row in summary.itertuples(index=False):
        record = row._asdict()
        if row.case_id == BASELINE_CASE_ID:
            record.update({
                "decision_changed": False,
                "material_net_total_change": False,
                "material": True,
                "materiality_reason": "comparison baseline",
            })
            for metric in MATERIALITY_ABSOLUTE:
                record[f"delta_{metric}"] = 0.0
                record[f"pct_{metric}_vs_base"] = 0.0
                record[f"material_{metric}"] = False
            rows.append(record)
            continue

        if not bool(row.success):
            record.update({
                "decision_changed": True,
                "material_net_total_change": False,
                "material": True,
                "materiality_reason": "infeasible or solve failure",
            })
            for metric in MATERIALITY_ABSOLUTE:
                record[f"delta_{metric}"] = np.nan
                record[f"pct_{metric}_vs_base"] = np.nan
                record[f"material_{metric}"] = False
            rows.append(record)
            continue

        reasons: List[str] = []
        route_changed = _map_changed(base_routes, _normalised_route_map(routes, row.case_id))
        if route_changed:
            reasons.append("routing")
        if str(row.low_batch_location) != str(base.low_batch_location):
            reasons.append("low-batch location")
        if abs(float(row.wind_share) - float(base.wind_share)) > DECISION_TOL:
            reasons.append("electricity mix")
        if abs(float(row.water_reuse_share) - float(base.water_reuse_share)) > DECISION_TOL:
            reasons.append("water reuse")
        decision_changed = bool(reasons)

        total_flags = []
        for metric, abs_threshold in MATERIALITY_ABSOLUTE.items():
            value = float(getattr(row, metric))
            base_value = float(base[metric])
            delta = value - base_value
            pct = 100.0 * delta / abs(base_value) if abs(base_value) > 1e-12 else np.nan
            threshold = max(float(abs_threshold), MATERIALITY_RELATIVE * abs(base_value))
            flag = abs(delta) >= threshold
            record[f"delta_{metric}"] = delta
            record[f"pct_{metric}_vs_base"] = pct
            record[f"material_{metric}"] = flag
            if flag:
                total_flags.append(metric)
        material_total = bool(total_flags)
        if total_flags:
            reasons.append("net totals: " + ", ".join(total_flags))
        record.update({
            "decision_changed": decision_changed,
            "material_net_total_change": material_total,
            "material": decision_changed or material_total,
            "materiality_reason": "; ".join(reasons) if reasons else "no material change",
        })
        rows.append(record)
    return pd.DataFrame(rows)



def _decision_change_text(
    case_id: str,
    summary: pd.DataFrame,
    routes: pd.DataFrame,
) -> str:
    """Describe material operational changes relative to the reporting baseline."""
    if case_id == BASELINE_CASE_ID:
        return "Comparison baseline"
    row = summary.loc[summary["case_id"].eq(case_id)].iloc[0]
    base = summary.loc[summary["case_id"].eq(BASELINE_CASE_ID)].iloc[0]
    if not bool(row["success"]):
        return "No feasible decision set"

    changes: List[str] = []
    base_map = _normalised_route_map(routes, BASELINE_CASE_ID)
    case_map = _normalised_route_map(routes, case_id)
    route_keys = sorted(set(base_map) | set(case_map))
    route_deltas = []
    for key in route_keys:
        delta = float(case_map.get(key, 0.0)) - float(base_map.get(key, 0.0))
        if abs(delta) > max(DECISION_TOL, 0.01):
            location, polymer, route, mode = key
            route_deltas.append((abs(delta), delta, location, polymer, route, mode))
    for _, delta, location, polymer, route, mode in sorted(route_deltas, reverse=True)[:6]:
        direction = "to" if delta > 0 else "from"
        changes.append(f"{abs(delta):,.2f} t/y {polymer} {direction} {route} at {location} ({mode})")

    if str(row.get("low_batch_location", "")) != str(base.get("low_batch_location", "")):
        changes.append(f"low-batch node {base.get('low_batch_location')} → {row.get('low_batch_location')}")
    for column, label in [("wind_share", "renewable electricity"), ("water_reuse_share", "water reuse")]:
        if column in summary.columns:
            before = float(base[column])
            after = float(row[column])
            if abs(after - before) > DECISION_TOL:
                changes.append(f"{label} {100*before:.1f}% → {100*after:.1f}%")
    return "; ".join(changes) if changes else "No operational decision changed"


def _reporting_table(screening: pd.DataFrame, routes: pd.DataFrame) -> pd.DataFrame:
    """Create a compact absolute-and-relative scenario table for interpretation."""
    frame = screening.copy()
    frame["decision_changes_vs_central"] = [
        _decision_change_text(str(case_id), screening, routes) for case_id in frame["case_id"]
    ]
    columns = [
        "case_id", "case_name", "scenario_family", "success",
        "cost", "pct_cost_vs_base", "gwp", "pct_gwp_vs_base",
        "human_health_daly", "avoided_human_health_daly_vs_bau", "human_health_daly_reduction_vs_bau_pct",
        "ecosystem_quality", "avoided_ecosystem_quality_vs_bau", "ecosystem_quality_reduction_vs_bau_pct",
        "freshwater", "pct_freshwater_vs_base",
        "circularity_loss", "pct_circularity_loss_vs_base",
        "residual", "pct_residual_vs_base",
        "low_batch_location", "decision_changes_vs_central",
        "materiality_reason", "solver_message",
    ]
    return frame[[column for column in columns if column in frame.columns]]

def _write_frame(writer: pd.ExcelWriter, sheet_name: str, frame: pd.DataFrame) -> None:
    frame.to_excel(writer, sheet_name=sheet_name[:31], index=False)


def write_results_workbook(
    screening: pd.DataFrame,
    routes: pd.DataFrame,
    sites: pd.DataFrame,
    mix: pd.DataFrame,
    breakdown: pd.DataFrame,
    applied: pd.DataFrame,
    reference_summary: pd.DataFrame,
) -> Path:
    reporting = _reporting_table(screening, routes)
    material = reporting.loc[screening["material"].to_numpy()].copy()
    unchanged = reporting.loc[
        screening["success"].astype(bool).to_numpy()
        & ~screening["material"].astype(bool).to_numpy()
    ].copy()
    infeasible = reporting.loc[~screening["success"].astype(bool).to_numpy()].copy()
    with pd.ExcelWriter(RESULTS_WORKBOOK, engine="openpyxl") as writer:
        _write_frame(writer, "Material scenarios", material)
        _write_frame(writer, "Tested - no material change", unchanged)
        _write_frame(writer, "Infeasible scenarios", infeasible)
        _write_frame(writer, "All scenario screening", reporting)
        _write_frame(writer, "Polymer routing", routes)
        _write_frame(writer, "Site decisions", sites)
        _write_frame(writer, "Route mix", mix)
        _write_frame(writer, "Objective breakdown", breakdown)
        _write_frame(writer, "Applied parameters", applied)
        _write_frame(writer, "Reference cases", reference_summary)
        _write_frame(
            writer,
            "Human health DALY",
            screening[[column for column in [
                "case_id", "case_name", "human_health_daly",
                "avoided_human_health_daly_vs_bau",
                "human_health_daly_reduction_vs_bau_pct",
            ] if column in screening.columns]],
        )
        _write_frame(
            writer,
            "Ecosystem quality",
            screening[[column for column in [
                "case_id", "case_name", "ecosystem_quality",
                "avoided_ecosystem_quality_vs_bau",
                "ecosystem_quality_reduction_vs_bau_pct",
            ] if column in screening.columns]],
        )
        if GAMS_SOLVE_REPORT.exists():
            _write_frame(writer, "Solver status", pd.read_csv(GAMS_SOLVE_REPORT))
    return RESULTS_WORKBOOK


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def run_analysis() -> Dict[str, Any]:
    reset_outputs()
    core = import_core_module()
    configure_core_outputs(core)
    base_data, params, scenario_doc, scenario_file = load_inputs(core)
    specs = list_scenario_specs(scenario_doc)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Core model: {resolve_core_module_path()}")
    print(f"Scenario file: {scenario_file}")
    print(f"Scenario output directory: {RESULT_DIR}")
    print(f"Configured solver: {core.GAMS_SOLVER_NAME}")
    print(f"Cases requested: {len(specs)} including the comparison baseline")

    reference_model, reference_result, private_reference = solve_private_reference(core, base_data)
    references: Dict[str, Any] = {"private_cost_reference": private_reference}
    bau_model = core.build_linear_model(copy.deepcopy(base_data))
    bau_result = core.evaluate_bau(bau_model)
    bau_human_health_daly = float(core.evaluate_human_health_daly(bau_model, bau_result))
    bau_ecosystem_quality = float(core.evaluate_ecosystem_quality(bau_model, bau_result))

    summary_rows: List[Dict[str, Any]] = []
    route_frames: List[pd.DataFrame] = []
    site_frames: List[pd.DataFrame] = []
    mix_frames: List[pd.DataFrame] = []
    breakdown_frames: List[pd.DataFrame] = []
    applied_rows: List[Dict[str, Any]] = []
    technical_failures: List[str] = []
    solved_models: Dict[str, Any] = {}
    solved_results: Dict[str, Any] = {}

    for spec in specs:
        print(f"\n--- {spec.scenario_id}: {spec.description}")
        try:
            build = build_case_model(core, base_data, params, scenario_doc, spec, references)
            objective_name = optimisation_objective_for_spec(spec)
            objective_description = (
                "minimum annual private network cost subject to the scenario constraint"
                if objective_name == "cost"
                else "minimum annual network cost plus the scenario lifecycle carbon value"
            )
            result = core.solve_model(
                build.model,
                spec.scenario_id,
                objective_name,
                selection_method=(
                    f"{objective_description}; one-at-a-time scenario family={spec.family}; "
                    f"reported versus {BASELINE_CASE_ID}"
                ),
                metadata={
                    "scenario_family": spec.family,
                    "scenario_id": spec.scenario_id,
                    "scenario_description": spec.description,
                    **build.applied,
                },
            )
            if not result.success:
                summary_rows.append(_case_summary_failure(spec, result.message))
                applied_rows.append({**build.applied, "success": False, "solver_message": result.message})
                continue

            summary_rows.append(_case_summary_success(core, build, result))
            routes, sites, mix = core.solution_decisions(build.model, result)
            routes["case_name"] = spec.scenario_name
            routes["scenario_family"] = spec.family
            sites["case_name"] = spec.scenario_name
            sites["scenario_family"] = spec.family
            mix["case_name"] = spec.scenario_name
            mix["scenario_family"] = spec.family
            route_frames.append(routes)
            site_frames.append(sites)
            mix_frames.append(mix)
            breakdown = core.component_breakdown(build.model, result)
            breakdown["scenario_family"] = spec.family
            breakdown_frames.append(breakdown)
            applied_rows.append({**build.applied, "success": True, "solver_message": result.message})
            solved_models[spec.scenario_id] = build.model
            solved_results[spec.scenario_id] = result
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            technical_failures.append(f"{spec.scenario_id}: {message}")
            print(message)
            traceback.print_exc()
            summary_rows.append(_case_summary_failure(spec, message))
            applied_rows.append({
                "scenario_family": spec.family,
                "scenario_id": spec.scenario_id,
                "success": False,
                "solver_message": message,
            })

    summary = pd.DataFrame(summary_rows)
    require(BASELINE_CASE_ID in set(summary["case_id"]), "Comparison baseline was not produced.")
    require(bool(summary.loc[summary["case_id"].eq(BASELINE_CASE_ID), "success"].iloc[0]), "Comparison baseline failed.")
    summary["avoided_human_health_daly_vs_bau"] = bau_human_health_daly - summary["human_health_daly"]
    summary["human_health_daly_reduction_vs_bau_pct"] = np.where(
        bau_human_health_daly != 0.0,
        100.0 * summary["avoided_human_health_daly_vs_bau"] / bau_human_health_daly,
        np.nan,
    )
    summary["avoided_ecosystem_quality_vs_bau"] = bau_ecosystem_quality - summary["ecosystem_quality"]
    summary["ecosystem_quality_reduction_vs_bau_pct"] = np.where(
        bau_ecosystem_quality != 0.0,
        100.0 * summary["avoided_ecosystem_quality_vs_bau"] / bau_ecosystem_quality,
        np.nan,
    )

    routes = pd.concat(route_frames, ignore_index=True) if route_frames else pd.DataFrame()
    sites = pd.concat(site_frames, ignore_index=True) if site_frames else pd.DataFrame()
    mix = pd.concat(mix_frames, ignore_index=True) if mix_frames else pd.DataFrame()
    breakdown = pd.concat(breakdown_frames, ignore_index=True) if breakdown_frames else pd.DataFrame()
    applied = pd.json_normalize(applied_rows, sep="__")
    screening = assess_materiality(summary, routes)

    # Friendly aliases used by the workbook and comparison script.
    for metric in MATERIALITY_ABSOLUTE:
        if f"pct_{metric}_vs_base" in screening.columns:
            screening.rename(columns={f"pct_{metric}_vs_base": f"pct_{metric}_vs_base"}, inplace=True)
    screening["pct_cost_vs_base"] = screening["pct_cost_vs_base"] if "pct_cost_vs_base" in screening else screening.get("pct_cost_vs_base", np.nan)
    screening["pct_gwp_vs_base"] = screening["pct_gwp_vs_base"] if "pct_gwp_vs_base" in screening else screening.get("pct_gwp_vs_base", np.nan)

    reference_summary = pd.DataFrame([{
        "case_id": PRIVATE_REFERENCE_CASE_ID,
        "case_name": "JSON BASE_2026 private-cost reference",
        **private_reference["metrics"],
        "low_batch_location": private_reference["low_batch_location"],
        "selection_method": private_reference["selection_method"],
    }])
    bau_reference = pd.DataFrame([{
        "case_id": "BAU",
        "case_name": "Business as usual",
        "human_health_daly": bau_human_health_daly,
        "ecosystem_quality": bau_ecosystem_quality,
        "selection_method": "constructed BAU benchmark",
    }])
    reference_summary = pd.concat([reference_summary, bau_reference], ignore_index=True, sort=False)

    screening.to_csv(RESULT_DIR / "scenario_case_summary.csv", index=False)
    routes.to_csv(RESULT_DIR / "scenario_polymer_routing.csv", index=False)
    sites.to_csv(RESULT_DIR / "scenario_site_decisions.csv", index=False)
    mix.to_csv(RESULT_DIR / "scenario_route_mix.csv", index=False)
    breakdown.to_csv(RESULT_DIR / "scenario_objective_breakdown.csv", index=False)
    applied.to_csv(RESULT_DIR / "scenario_parameter_audit.csv", index=False)
    reference_summary.to_csv(RESULT_DIR / "scenario_reference_summary.csv", index=False)
    screening[[
        "case_id", "case_name", "human_health_daly",
        "avoided_human_health_daly_vs_bau", "human_health_daly_reduction_vs_bau_pct",
    ]].to_csv(RESULT_DIR / "scenario_human_health_daly.csv", index=False)
    screening[[
        "case_id", "case_name", "ecosystem_quality",
        "avoided_ecosystem_quality_vs_bau", "ecosystem_quality_reduction_vs_bau_pct",
    ]].to_csv(RESULT_DIR / "scenario_ecosystem_quality.csv", index=False)

    references["reporting_baseline"] = {
        "case_id": BASELINE_CASE_ID,
        "metrics": {
            metric: float(screening.loc[screening["case_id"].eq(BASELINE_CASE_ID), metric].iloc[0])
            for metric in ["cost", "policy_cost", "gwp", "human_health_daly", "ecosystem_quality", "freshwater", "circularity_loss", "residual", "recovered"]
        },
    }
    REFERENCE_FILE.write_text(json.dumps(_json_safe(references), indent=2), encoding="utf-8")
    manifest = {
        "run_tag": RUN_TAG,
        "core_module": str(resolve_core_module_path()),
        "scenario_file": str(scenario_file),
        "result_dir": str(RESULT_DIR),
        "reporting_baseline_case_id": BASELINE_CASE_ID,
        "reporting_baseline_carbon_scenario_id": reporting_carbon_scenario_id(scenario_doc),
        "private_reference_case_id": PRIVATE_REFERENCE_CASE_ID,
        "comparison_rule": "All deltas and materiality tests use BASE_CURRENT_POLICY.",
        "cap_objective_rule": "Carbon and residual caps use cost minimisation with zero lifecycle carbon value; cap limits may use the private reference.",
        "materiality_absolute": MATERIALITY_ABSOLUTE,
        "materiality_relative": MATERIALITY_RELATIVE,
        "scenario_specs": [_json_safe(spec.__dict__) for spec in specs],
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    workbook = write_results_workbook(
        screening, routes, sites, mix, breakdown, applied, reference_summary
    )

    display = screening.loc[screening["material"], [
        "case_id", "scenario_family", "success", "cost", "gwp", "human_health_daly", "ecosystem_quality", "freshwater",
        "residual", "materiality_reason",
    ]]
    print("\nMATERIAL SCENARIOS")
    print(display.to_string(index=False, float_format=lambda value: f"{value:,.4f}"))
    print(f"\nWorkbook: {workbook}")
    print(f"Scenario comparison inputs: {RESULT_DIR}")
    print(f"GAMS solve report: {GAMS_SOLVE_REPORT}")
    if technical_failures:
        raise RuntimeError("Scenario execution errors: " + "; ".join(technical_failures))

    return {
        "core": core,
        "base_data": base_data,
        "params": params,
        "scenario_doc": scenario_doc,
        "references": references,
        "summary": screening,
        "routes": routes,
        "sites": sites,
        "mix": mix,
        "breakdown": breakdown,
        "applied": applied,
        "reference_summary": reference_summary,
        "models": solved_models,
        "results": solved_results,
        "workbook": workbook,
    }


def reconstruct_case(case_id: str) -> Tuple[Any, ScenarioBuild, Mapping[str, Any], Mapping[str, Any]]:
    """Rebuild a solved case without rerunning GAMS; used to inspect solved cases."""
    core = import_core_module()
    base_data, params, scenario_doc, _ = load_inputs(core)
    require(REFERENCE_FILE.exists(), f"Run scenario_MILP.py first; missing {REFERENCE_FILE}")
    references = load_json(REFERENCE_FILE)
    spec = scenario_spec_by_id(scenario_doc, case_id)
    build = build_case_model(core, base_data, params, scenario_doc, spec, references)
    return core, build, params, scenario_doc


if __name__ == "__main__":
    run_analysis()
