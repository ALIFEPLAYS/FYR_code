from __future__ import annotations

"""
Local one-variable-at-a-time decision-boundary analysis for the industrial-
symbiosis MILP.

This script varies one parameter at a time over a deliberately wide search
range, resolves the MILP, detects every change in the structural optimiser
decision, refines each transition by bisection, and persists dense decision
curves as data tables. It is dedicated to local structural threshold discovery and does not perform joint uncertainty propagation.

Run beside run_master_MILP.py:
    python run_oat_decision_boundaries.py
"""

import argparse
import copy
import hashlib
import fnmatch
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import time
import tempfile
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# USER CONTROLS — LOCAL DECISION-BOUNDARY ANALYSIS
# =============================================================================
# Sampling controls belong at the top of the analytical script. Parameter
# selection and all numerical ranges belong in MILP_OAT_sensitivity_analysis_parameters.json.
SOLVER_BACKEND = "gams"                 # "gams" for final results; "scipy" for screening
GAMS_TEE = False
BOUNDARY_DISCOVERY_POINTS = 21          # initial OAT points across each boundary-search range
DECISION_CURVE_POINTS = 61              # persisted points for each parameter with a boundary
BOUNDARY_BISECTION_ITERATIONS = 24      # maximum refinement solves per transition
BOUNDARY_RELATIVE_TOLERANCE = 1e-5
BOUNDARY_ABSOLUTE_TOLERANCE = 1e-7
DECISION_ACTIVE_SHARE_TOLERANCE = 0.005
DECISION_FULL_SHARE_TOLERANCE = 0.995
CLEAR_PREVIOUS_RESULTS = True
RESULT_DIRECTORY = "results/oat_decision_boundaries"
PARAMETER_CONFIG_FILENAME = "MILP_OAT_sensitivity_analysis_parameters.json"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_RESULT_DIR = PROJECT_ROOT / RESULT_DIRECTORY

IMPACT_CATEGORIES = ("gwp", "fossil_resource_use", "water_scarcity", "human_health_daly", "ecosystem_quality")
# DALY is an output only. It is deliberately excluded from uncertain LCIA inputs.
UNCERTAIN_IMPACT_CATEGORIES = ("gwp", "fossil_resource_use", "water_scarcity")
OUTPUT_METRICS = ("cost", "gwp", "fossil_resource_use", "water_scarcity", "human_health_daly", "ecosystem_quality")
OUTPUT_LABELS = {
    "cost": "Annual cost",
    "gwp": "Carbon footprint (IMPACT World+ v2.1)",
    "fossil_resource_use": "Fossil-resource use",
    "water_scarcity": "Water scarcity (AWARE 2.0)",
    "human_health_daly": "Human-health damage",
    "ecosystem_quality": "Ecosystem quality",
}
OUTPUT_UNITS = {
    "cost": "GBP/year",
    "gwp": "kg CO2e/year",
    "fossil_resource_use": "kg oil-eq/year (method unit)",
    "water_scarcity": "m3 world-eq/year (method unit)",
    "human_health_daly": "DALY/year",
    "ecosystem_quality": "species.yr/year",
}
ROUTE_LABELS = {
    "landfill_all": "Landfill",
    "incineration_efw_heat_electricity": "Energy from Waste",
    "mechanical_recycling_polyolefins": "Polyolefin recycling",
    "mechanical_recycling_tpe_tubing": "TPE recycling",
}
MODE_LABELS = {
    "diesel_articulated_hgv": "Diesel HGV",
    "battery_electric_articulated_hgv": "Electric HGV",
}

DECISION_ACTIVE_SHARE_TOL = DECISION_ACTIVE_SHARE_TOLERANCE
DECISION_FULL_SHARE_TOL = DECISION_FULL_SHARE_TOLERANCE


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def safe_token(value: Any, max_length: int = 90) -> str:
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")
    return (text or "item")[:max_length]


def finite(value: Any, default: float = np.nan) -> float:
    try:
        result = float(value)
    except Exception:
        return default
    return result if math.isfinite(result) else default


def unique_sorted(values: Iterable[float], digits: int = 12) -> List[float]:
    keyed: Dict[float, float] = {}
    for value in values:
        f = float(value)
        keyed[round(f, digits)] = f
    return sorted(keyed.values())


def relative_distance(value: float, central: float, low: float, high: float) -> float:
    if abs(central) > 1e-12:
        return abs(value - central) / abs(central)
    span = max(abs(high - low), 1e-12)
    return abs(value - central) / span


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_replace_text(path: Path, text: str) -> None:
    """Write text atomically so interruption cannot leave a zero-byte result file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def write_json(path: Path, payload: Any) -> None:
    _atomic_replace_text(path, json.dumps(payload, indent=2, default=str) + "\n")


def atomic_to_csv(frame: pd.DataFrame, path: Path) -> None:
    """Persist a DataFrame atomically, including headers for an empty table."""
    _atomic_replace_text(path, frame.to_csv(index=False))


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def estimated_finish_text(seconds: float) -> str:
    """Return a local clock-time estimate without pretending the ETA is exact."""
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    finish = datetime.now().astimezone() + timedelta(seconds=float(seconds))
    return finish.strftime("%H:%M %Z")


def normalise_sheet_name(name: str) -> str:
    cleaned = re.sub(r"[\\/*?:\[\]]", "_", name)
    return cleaned[:31] or "Sheet"




# -----------------------------------------------------------------------------
# Dynamic import of the validated master model
# -----------------------------------------------------------------------------

def locate_master_file() -> Path:
    override = os.environ.get("IS_MASTER_MILP")
    candidates: List[Path] = []
    if override:
        candidates.append(Path(override).expanduser())
    candidates.extend([
        SCRIPT_DIR / "run_master_MILP.py",
        SCRIPT_DIR / "master_milp.py",
    ])
    candidates.extend(sorted(SCRIPT_DIR.glob("master_MILP*.py"), reverse=True))
    candidates.extend(sorted(SCRIPT_DIR.glob("master_milp*.py"), reverse=True))
    this_file = Path(__file__).resolve()
    for path in candidates:
        try:
            resolved = path.resolve()
        except Exception:
            continue
        if resolved != this_file and resolved.exists():
            return resolved
    raise FileNotFoundError(
        "Could not locate run_master_MILP.py beside the sensitivity runner. "
        "Set IS_MASTER_MILP to the exact master script path."
    )


def import_master_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("is_master_milp_for_sensitivity", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import master model from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    required = [
        "load_model_data", "build_linear_model", "solve_model",
        "solution_decisions", "solution_validation", "SolveResult",
    ]
    missing = [name for name in required if not hasattr(module, name)]
    require(not missing, f"Master model is missing required objects: {missing}")
    return module


# -----------------------------------------------------------------------------
# LCIA reconstruction: preserve baked assumptions without rerunning Brightway
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class RouteImpactBase:
    polymer: str
    route: str
    pathway: str
    category: str
    production: float
    supply: float
    treatment: float
    credit: float
    net: float
    autoclave_required: bool
    central_recovery: float
    central_substitution: float
    non_autoclave_treatment: float
    recycling_treatment_per_recovered_kg: float
    recycling_credit_per_recovered_kg_at_full_substitution: float
    is_recycling_route: bool


@dataclass
class ImpactModel:
    route_base: Dict[Tuple[str, str, str], RouteImpactBase]
    electricity_base: Dict[Tuple[str, str], float]
    treatment_base: Dict[str, float]
    fresh_water_base: Dict[Tuple[str, str], float]
    fixed_non_electric_per_batch: Dict[str, float]
    biomass_route_base: Dict[Tuple[str, str], float]
    central_autoclave_kwh_per_kg: float
    source_workbook: str
    transport_coverage_note: str
    audit: pd.DataFrame


@dataclass
class ImpactControls:
    autoclave_kwh_per_kg: float
    electricity_multiplier: Dict[Tuple[str, str], float] = field(default_factory=dict)
    avoided_credit_multiplier: Dict[Tuple[str, str], float] = field(default_factory=dict)
    substitution_efficiency: Dict[str, float] = field(default_factory=dict)
    # Zero by default preserves OAT factor-only tests. Prospective-grid sampling
    # sets this to the EfW export intensity reconstructed from the LCA workbook.
    efw_grid_credit_kwh_per_kg: float = 0.0

    def clone(self) -> "ImpactControls":
        return copy.deepcopy(self)


@dataclass
class ScenarioState:
    data: Any
    controls: ImpactControls
    notes: List[str] = field(default_factory=list)
    effective_value: Optional[float] = None


def _route_factor_columns(category: str) -> Tuple[str, str, str, str, str]:
    return (
        f"net_{category}_factor_per_optimisation_unit",
        f"{category}__production",
        f"{category}__supply",
        f"{category}__treatment",
        f"{category}__credit",
    )


def load_impact_model(master: Any, data: Any, lca_settings_path: Optional[Path] = None) -> ImpactModel:
    route_path = Path(master.INPUT_FILES["lca"])
    route_factors = pd.read_excel(route_path, sheet_name="route_factor_matrix")

    def one(flow: str, pathway: Optional[str] = None, material: Optional[str] = None) -> pd.Series:
        mask = route_factors["optimisation_flow"].astype(str).eq(flow)
        if pathway is not None:
            mask &= route_factors["pathway"].astype(str).eq(pathway)
        if material is not None:
            mask &= route_factors["material_or_service"].astype(str).eq(material)
        rows = route_factors.loc[mask]
        require(len(rows) == 1, f"Expected one route-factor row for {flow}/{pathway}/{material}; found {len(rows)}")
        return rows.iloc[0]

    grid_row = one("manufacturing__electricity", "electricity_grid_uk_bau")
    wind_row = one("manufacturing__electricity", "electricity_renewable_onshore_wind")
    ww_row = one("plant_wastewater", "treatment_and_discharge")
    fresh_row = one("fresh_utility_water", "fresh_utility_water_supply")
    fixed_rows = [
        one("manufacturing__wfi", "baseline_supply"),
        one("manufacturing__chilled_water", "baseline_supply"),
        one("manufacturing__cooling_water", "baseline_supply"),
        one("manufacturing__steam", "baseline_supply"),
    ]

    electricity_base: Dict[Tuple[str, str], float] = {}
    treatment_base: Dict[str, float] = {}
    fresh_water_base: Dict[Tuple[str, str], float] = {}
    fixed_non_electric_per_batch: Dict[str, float] = {}
    biomass_route_base: Dict[Tuple[str, str], float] = {}

    for category in IMPACT_CATEGORIES:
        net_col = f"net_{category}_factor_per_optimisation_unit"
        electricity_base["grid", category] = float(grid_row[net_col])
        electricity_base["wind", category] = float(wind_row[net_col])
        treatment_base[category] = float(ww_row[net_col])
        for site in data.sites:
            location = data.site_location[site]
            site_column = f"net_water_scarcity_factor__{location}"
            fresh_water_base[site, category] = float(
                fresh_row[site_column]
                if category == "water_scarcity"
                else fresh_row[net_col]
            )
        fixed_total = sum(float(row["represented_quantity_per_batch"]) * float(row[net_col]) for row in fixed_rows)
        # Match master_MILP: fixed WFI proxy is subtracted here and restored through
        # fixed_mains_water in the water block, avoiding double counting.
        fixed_total -= data.fixed_mains_water_m3_per_batch * float(fresh_row[net_col])
        fixed_non_electric_per_batch[category] = fixed_total
        for route in data.biomass_routes:
            biomass_route_base[route, category] = float(data.biomass_route_factors[route][category])

    central_autoclave = float(next(iter(data.autoclave_cost_gbp_per_t.values()))) / (
        1000.0 * float(data.electricity_cost_gbp_per_kwh["grid"])
    )

    plastic_rows = route_factors.loc[
        route_factors["optimisation_flow"].astype(str).str.startswith("sut_plastic__")
    ].copy()
    route_base: Dict[Tuple[str, str, str], RouteImpactBase] = {}
    audit_rows: List[Dict[str, Any]] = []

    for polymer in data.polymers:
        for route in data.routes:
            pathway = data.route_lca_pathway[route]
            row_match = plastic_rows.loc[
                plastic_rows["material_or_service"].astype(str).eq(polymer)
                & plastic_rows["pathway"].astype(str).eq(pathway)
            ]
            require(len(row_match) == 1, f"Missing or duplicate plastic route factor for {polymer}/{pathway}")
            row = row_match.iloc[0]
            eta0 = float(data.recovery_yield[polymer, route])
            auto = bool(data.autoclave_required[polymer, route])
            is_recycling = route.startswith("mechanical_recycling") and eta0 > 0
            substitution0 = float(data.substitution_ratio[polymer, route]) if is_recycling else 0.0
            if is_recycling:
                require(
                    0.0 < substitution0 <= 1.0,
                    f"Invalid central substitution ratio for {polymer}/{route}: {substitution0}",
                )
            for category in IMPACT_CATEGORIES:
                net_col, prod_col, supply_col, treatment_col, credit_col = _route_factor_columns(category)
                production = float(row[prod_col])
                supply = float(row[supply_col])
                treatment = float(row[treatment_col])
                credit = float(row[credit_col])
                net = float(row[net_col])
                if is_recycling:
                    virgin_production_col = f"{category}__virgin_polymer_production"
                    avoided_virgin_col = f"{category}__avoided_virgin_material_credit"
                    require(
                        virgin_production_col in row.index and avoided_virgin_col in row.index,
                        f"Named virgin-production/credit stages are missing for {polymer}/{route}/{category}.",
                    )
                    virgin_production = float(row[virgin_production_col])
                    avoided_virgin_credit = float(row[avoided_virgin_col])
                    require(
                        math.isclose(
                            avoided_virgin_credit,
                            -virgin_production * eta0 * substitution0,
                            rel_tol=1e-9,
                            abs_tol=1e-12,
                        ),
                        f"Avoided credit for {polymer}/{route}/{category} must offset "
                        "virgin-polymer production only, excluding SUT component manufacture.",
                    )
                require(
                    abs((production + supply + treatment + credit) - net) <= 1e-7,
                    f"Route-factor stage reconciliation failed for {polymer}/{route}/{category}",
                )
                auto_component = central_autoclave * electricity_base["grid", category] if auto else 0.0
                non_auto = treatment - auto_component
                if abs(non_auto) < 1e-12:
                    non_auto = 0.0
                recycling_treatment_per_recovered = non_auto / eta0 if is_recycling else 0.0
                # The workbook credit already contains both recovery and the central
                # 0.81 quality/downcycling substitution factor. Remove both here so
                # OAT scenarios can vary them independently and reconstruct the
                # central route factor without applying either factor twice.
                recycling_credit_per_recovered_at_full_substitution = (
                    credit / (eta0 * substitution0) if is_recycling else 0.0
                )
                base = RouteImpactBase(
                    polymer=polymer,
                    route=route,
                    pathway=pathway,
                    category=category,
                    production=production,
                    supply=supply,
                    treatment=treatment,
                    credit=credit,
                    net=net,
                    autoclave_required=auto,
                    central_recovery=eta0,
                    central_substitution=substitution0,
                    non_autoclave_treatment=non_auto,
                    recycling_treatment_per_recovered_kg=recycling_treatment_per_recovered,
                    recycling_credit_per_recovered_kg_at_full_substitution=(
                        recycling_credit_per_recovered_at_full_substitution
                    ),
                    is_recycling_route=is_recycling,
                )
                route_base[polymer, route, category] = base
                audit_rows.append({
                    "audit_item": "plastic_route_stage_decomposition",
                    "polymer": polymer,
                    "route": route,
                    "impact_category": category,
                    "central_net_factor": net,
                    "production": production,
                    "supply": supply,
                    "treatment_including_autoclave": treatment,
                    "credit": credit,
                    "central_recovery_fraction": eta0,
                    "central_substitution_ratio": substitution0,
                    "autoclave_component_reconstructed": auto_component,
                    "non_autoclave_treatment": non_auto,
                    "recycling_treatment_per_recovered_kg": recycling_treatment_per_recovered,
                    "recycling_credit_per_recovered_kg_at_full_substitution": (
                        recycling_credit_per_recovered_at_full_substitution
                    ),
                    "note": (
                        "Recovery and substitution sensitivities independently scale the recycling-treatment and "
                        "avoided-credit components. "
                        "No additional residual-incineration burden is invented because it is not present as a separate "
                        "stage in the supplied net route factor."
                    ),
                })

    # Central GWP must reproduce the data supplied to the MILP exactly.
    for polymer in data.polymers:
        for route in data.routes:
            expected = route_base[polymer, route, "gwp"].net
            actual = float(data.plastic_gwp_kg_per_kg[polymer, route])
            require(abs(expected - actual) <= 1e-8, f"Central GWP mismatch for {polymer}/{route}: {expected} vs {actual}")

    # Optional audit against the LCA settings script/config. The current MILP remains
    # authoritative: the route-factor workbook and master model are the central case.
    if lca_settings_path and lca_settings_path.exists():
        try:
            lca_settings = read_json(lca_settings_path)
            configured_recovery = lca_settings.get("plastics", {}).get("recycling_recovery_fraction", {})
            for polymer, configured in configured_recovery.items():
                active = [
                    data.recovery_yield[polymer, route]
                    for route in data.routes
                    if (polymer, route) in data.recovery_yield and data.recovery_yield[polymer, route] > 0
                ]
                if active:
                    audit_rows.append({
                        "audit_item": "settings_vs_loaded_recovery",
                        "polymer": polymer,
                        "configured_value": float(configured),
                        "loaded_value": float(active[0]),
                        "difference": float(active[0]) - float(configured),
                        "note": "Loaded route-factor/MILP value is used as central.",
                    })
            configured_auto = lca_settings.get("plastics", {}).get("autoclave_kwh_per_kg")
            if configured_auto is not None:
                audit_rows.append({
                    "audit_item": "settings_vs_loaded_autoclave",
                    "configured_value": float(configured_auto),
                    "loaded_value": central_autoclave,
                    "difference": central_autoclave - float(configured_auto),
                    "note": "Loaded MILP cost coefficient is used as central.",
                })
            configured_reuse = (
                lca_settings.get("wastewater", {})
                .get("routes", {})
                .get("treatment_for_utility_reuse", {})
                .get("reuse_fraction_per_m3_routed")
            )
            loaded_reuse = float(next(iter(data.water_recovery.values())))
            if configured_reuse is not None:
                audit_rows.append({
                    "audit_item": "settings_vs_loaded_water_recovery",
                    "configured_value": float(configured_reuse),
                    "loaded_value": loaded_reuse,
                    "difference": loaded_reuse - float(configured_reuse),
                    "note": (
                        "The master MILP derives recovery from the current input-output workbook. "
                        "The loaded value is used as central; any mismatch is retained as an explicit audit warning."
                    ),
                })
        except Exception as exc:
            audit_rows.append({
                "audit_item": "lca_settings_audit_error",
                "note": f"Could not audit {lca_settings_path}: {exc}",
            })

    audit_rows.extend([
        {
            "audit_item": "non_gwp_transport_coverage",
            "impact_category": "fossil_resource_use; water_scarcity",
            "note": (
                "The supplied master model contains transport cost and transport GWP coefficients only. "
                "Transport contributions to fossil-resource use and water-scarcity impact are therefore reported as "
                "not available rather than estimated."
            ),
        },
        {
            "audit_item": "objective_scope",
            "impact_category": "gwp",
            "note": (
                "Decisions are selected by annual cost + lifecycle carbon value × GWP. Fossil-resource use and "
                "water-scarcity factors are evaluated after the solve and cannot change decisions in this formulation."
            ),
        },
    ])

    return ImpactModel(
        route_base=route_base,
        electricity_base=electricity_base,
        treatment_base=treatment_base,
        fresh_water_base=fresh_water_base,
        fixed_non_electric_per_batch=fixed_non_electric_per_batch,
        biomass_route_base=biomass_route_base,
        central_autoclave_kwh_per_kg=central_autoclave,
        source_workbook=str(route_path),
        transport_coverage_note="GWP only; fossil-resource and water-scarcity transport factors unavailable",
        audit=pd.DataFrame(audit_rows),
    )


def electricity_factor(impact: ImpactModel, controls: ImpactControls, source: str, category: str) -> float:
    return impact.electricity_base[source, category] * controls.electricity_multiplier.get((source, category), 1.0)


def route_impact_factor(
    impact: ImpactModel,
    controls: ImpactControls,
    data: Any,
    polymer: str,
    route: str,
    category: str,
) -> float:
    base = impact.route_base[polymer, route, category]
    eta = float(data.recovery_yield[polymer, route])
    auto = (
        controls.autoclave_kwh_per_kg * electricity_factor(impact, controls, "grid", category)
        if base.autoclave_required
        else 0.0
    )
    if base.is_recycling_route:
        treatment = auto + eta * base.recycling_treatment_per_recovered_kg
        credit = (
            eta
            * base.recycling_credit_per_recovered_kg_at_full_substitution
            * controls.avoided_credit_multiplier.get((polymer, category), 1.0)
            * controls.substitution_efficiency.get(polymer, 1.0)
        )
    else:
        treatment = base.non_autoclave_treatment + auto if base.autoclave_required else base.treatment
        credit = base.credit
        if category == "gwp" and route == "incineration_efw_heat_electricity":
            grid_delta = (electricity_factor(impact, controls, "grid", "gwp")
                          - impact.electricity_base["grid", "gwp"])
            credit -= controls.efw_grid_credit_kwh_per_kg * grid_delta
    return base.production + base.supply + treatment + credit


def sync_gwp_coefficients(state: ScenarioState, impact: ImpactModel) -> None:
    for polymer in state.data.polymers:
        for route in state.data.routes:
            state.data.plastic_gwp_kg_per_kg[polymer, route] = route_impact_factor(
                impact, state.controls, state.data, polymer, route, "gwp"
            )
            base = impact.route_base[polymer, route, "gwp"]
            if base.is_recycling_route:
                state.data.avoided_virgin_material_credit_gwp_kg_per_kg[polymer, route] = (
                    float(state.data.recovery_yield[polymer, route])
                    * base.recycling_credit_per_recovered_kg_at_full_substitution
                    * state.controls.avoided_credit_multiplier.get((polymer, "gwp"), 1.0)
                    * state.controls.substitution_efficiency.get(polymer, 1.0)
                )
            else:
                state.data.avoided_virgin_material_credit_gwp_kg_per_kg[polymer, route] = 0.0
    state.data.electricity_gwp_kg_per_kwh["grid"] = electricity_factor(impact, state.controls, "grid", "gwp")
    state.data.electricity_gwp_kg_per_kwh["wind"] = electricity_factor(impact, state.controls, "wind", "gwp")
    # The master model derives BEV transport GWP from the grid electricity factor.
    if "battery_electric_articulated_hgv" in state.data.transport_gwp_kg_per_km:
        central_grid = impact.electricity_base["grid", "gwp"]
        central_bev = state.data.transport_gwp_kg_per_km["battery_electric_articulated_hgv"]
        if central_grid > 0:
            bev_kwh_per_km = central_bev / central_grid
            state.data.transport_gwp_kg_per_km["battery_electric_articulated_hgv"] = (
                bev_kwh_per_km * state.data.electricity_gwp_kg_per_kwh["grid"]
            )


def evaluate_extended_impacts(
    master: Any,
    model: Any,
    result: Any,
    state: ScenarioState,
    impact: ImpactModel,
) -> Dict[str, float]:
    if not result.success:
        return {metric: np.nan for metric in OUTPUT_METRICS}
    data = state.data
    x = result.x
    totals: Dict[str, float] = {
        "cost": float(result.metrics["cost"]),
        "gwp": 0.0,
        "fossil_resource_use": 0.0,
        "water_scarcity": 0.0,
        "human_health_daly": 0.0,
        "ecosystem_quality": 0.0,
    }

    for category in IMPACT_CATEGORIES:
        total = data.total_batches * impact.fixed_non_electric_per_batch[category]
        for site in data.sites:
            for route in data.biomass_routes:
                key = ("bio_q", site, route)
                if key in model.index:
                    total += x[model.index[key]] * data.biomass_route_factors[route][category]
        # Electricity.
        for site in data.sites:
            for source in data.electricity_sources:
                key = ("E", site, source)
                if key in model.index:
                    total += x[model.index[key]] * electricity_factor(impact, state.controls, source, category)
        # Plastics.
        for key, idx in model.index.items():
            if key[0] == "x":
                _, site, polymer, route, mode = key
                total += x[idx] * 1000.0 * route_impact_factor(
                    impact, state.controls, data, polymer, route, category
                )
        # Water treatment and water supply.
        for site in data.sites:
            batches = x[model.index["B", site]]
            fresh = x[model.index["f", site]]
            fixed_mains = batches * data.fixed_mains_water_m3_per_batch
            total += batches * data.wastewater_m3_per_batch * impact.treatment_base[category]
            total += (fixed_mains + fresh) * impact.fresh_water_base[site, category]
        # Transport is available for GWP only in the supplied model.
        if category == "gwp":
            for key, idx in model.index.items():
                if key[0] == "n":
                    _, site, route, mode = key
                    vehicle_km = (
                        x[idx]
                        * data.roundtrip_multiplier[site, route, mode]
                        * data.distance_km[site, route]
                    )
                    total += vehicle_km * data.transport_gwp_kg_per_km[mode]
        totals[category] = float(total)

    # Central/reconstructed GWP should remain identical to the master objective expression.
    if math.isfinite(totals["gwp"]) and "gwp" in result.metrics:
        difference = abs(totals["gwp"] - float(result.metrics["gwp"]))
        tolerance = max(1e-4, 1e-9 * max(abs(totals["gwp"]), 1.0))
        require(
            difference <= tolerance,
            f"Extended GWP does not reconcile with master expression: {totals['gwp']} vs {result.metrics['gwp']}",
        )
    return totals


# -----------------------------------------------------------------------------
# Parameter registry
# -----------------------------------------------------------------------------

ApplyFunction = Callable[[ScenarioState, float], None]


@dataclass
class SensitivityParameter:
    parameter_id: str
    label: str
    group: str
    unit: str
    central: float
    low: float
    high: float
    boundary_low: float
    boundary_high: float
    apply: ApplyFunction
    description: str
    source_or_basis: str
    affects_optimisation: bool = True
    grid_points: Optional[int] = None
    parameter_type: str = "continuous"  # only continuous parameters are permitted in sensitivity analysis
    display_decimals: int = 4
    classification: str = ""
    lcia_factor_status: str = "not_an_lcia_factor"
    hard_low: Optional[float] = None
    hard_high: Optional[float] = None
    scope_level: str = "system"
    target: str = ""
    configuration_source: str = "built_in_default"
    linked_quantity_note: str = ""

    def clip(self, value: float) -> float:
        return min(max(float(value), self.boundary_low), self.boundary_high)

    def grid(self, default_points: int) -> List[float]:
        if self.parameter_type == "binary":
            return unique_sorted([self.low, self.central, self.high])
        n = int(self.grid_points or default_points)
        values = list(np.linspace(self.low, self.high, max(n, 3))) + [self.central]
        return unique_sorted(values)

    def discovery_grid(self, points: int) -> List[float]:
        if self.parameter_type == "binary":
            return self.grid(points)
        values = list(np.linspace(self.boundary_low, self.boundary_high, max(points, 5))) + [
            self.low, self.central, self.high
        ]
        return unique_sorted(values)


def clone_state(central_data: Any, impact: ImpactModel) -> ScenarioState:
    central_substitution: Dict[str, float] = {}
    for polymer in central_data.polymers:
        active_values = [
            float(central_data.substitution_ratio[polymer, route])
            for route in central_data.routes
            if route.startswith("mechanical_recycling")
            and central_data.eligibility[polymer, route]
            and central_data.recovery_yield[polymer, route] > 0
        ]
        if active_values:
            require(
                max(active_values) - min(active_values) <= 1e-12,
                f"OAT uses one substitution control per polymer; central routes disagree for {polymer}.",
            )
            central_substitution[polymer] = active_values[0]
        else:
            central_substitution[polymer] = 1.0
    controls = ImpactControls(
        autoclave_kwh_per_kg=impact.central_autoclave_kwh_per_kg,
        electricity_multiplier={(source, category): 1.0 for source in ("grid", "wind") for category in IMPACT_CATEGORIES},
        avoided_credit_multiplier={(polymer, category): 1.0 for polymer in central_data.polymers for category in IMPACT_CATEGORIES},
        substitution_efficiency=central_substitution,
    )
    return ScenarioState(data=copy.deepcopy(central_data), controls=controls)


def rescale_throughput(data: Any, new_total_batches: float) -> None:
    active_sites = [site for site in data.sites if data.site_open[site] == 1]
    n_active = len(active_sites)
    require(n_active >= 2, "Throughput rescaling expects at least two active sites.")
    ratio = data.batch_low / data.batch_high
    high = max(1, int(round(float(new_total_batches) / ((n_active - 1) + ratio))))
    low = max(1, int(round(ratio * high)))
    total = (n_active - 1) * high + low
    low_sites = [site for site in active_sites if abs(data.batches[site] - data.batch_low) <= 1e-9]
    low_site = low_sites[0] if low_sites else active_sites[-1]
    old_total = float(data.total_batches)
    data.batch_high = float(high)
    data.batch_low = float(low)
    data.batch_delta = float(high - low)
    require(data.batch_delta >= 0, "Rescaled low batch count cannot exceed high batch count.")
    data.total_batches = float(total)
    for site in data.sites:
        if data.site_open[site] == 0:
            data.batches[site] = 0.0
        else:
            data.batches[site] = float(low if site == low_site else high)
    scale = data.total_batches / old_total
    data.fixed_manufacturing_gwp_kg *= scale
    data.biomass_gwp_kg *= scale
    data.total_product_kg *= scale
    for site in data.sites:
        data.electricity_kwh[site] = data.batches[site] * data.electricity_kwh_per_batch
        data.wastewater_m3[site] = data.batches[site] * data.wastewater_m3_per_batch
        data.utility_water_m3[site] = data.batches[site] * data.utility_water_m3_per_batch
        data.fixed_mains_water_m3[site] = data.batches[site] * data.fixed_mains_water_m3_per_batch


def set_low_high_ratio(data: Any, ratio: float) -> None:
    ratio = min(max(float(ratio), 0.01), 1.0)
    active_sites = [site for site in data.sites if data.site_open[site] == 1]
    n_active = len(active_sites)
    require(n_active >= 2, "Batch-allocation sensitivity expects at least two active sites.")
    total = int(round(data.total_batches))
    high = max(1, int(round(total / ((n_active - 1) + ratio))))
    low = total - (n_active - 1) * high
    while low < 1 and high > 1:
        high -= 1
        low = total - (n_active - 1) * high
    while low > high:
        high += 1
        low = total - (n_active - 1) * high
    # The master formulation assigns exactly one low-volume node.  At a requested
    # ratio of 1.0, retain the closest meaningful integral pattern (one batch
    # difference) rather than creating an arbitrary binary low-node label with
    # zero batch_delta.
    if low == high and total >= n_active + 1:
        high += 1
        low = total - (n_active - 1) * high
        if low < 1 or low >= high:
            high -= 1
            low = total - (n_active - 1) * high
    require(1 <= low < high, f"Could not create a distinct feasible high/low allocation for ratio {ratio}")
    low_sites = [site for site in active_sites if abs(data.batches[site] - data.batch_low) <= 1e-9]
    low_site = low_sites[0] if low_sites else active_sites[-1]
    data.batch_high = float(high)
    data.batch_low = float(low)
    data.batch_delta = float(high - low)
    for site in data.sites:
        data.batches[site] = 0.0 if data.site_open[site] == 0 else float(low if site == low_site else high)
    for site in data.sites:
        data.electricity_kwh[site] = data.batches[site] * data.electricity_kwh_per_batch
        data.wastewater_m3[site] = data.batches[site] * data.wastewater_m3_per_batch
        data.utility_water_m3[site] = data.batches[site] * data.utility_water_m3_per_batch
        data.fixed_mains_water_m3[site] = data.batches[site] * data.fixed_mains_water_m3_per_batch


def set_composition_share(data: Any, polymer: str, new_share: float) -> None:
    new_share = min(max(float(new_share), 0.0), 0.98)
    total = sum(float(v) for v in data.polymer_kg_per_batch.values())
    others = [p for p in data.polymers if p != polymer]
    other_total = sum(float(data.polymer_kg_per_batch[p]) for p in others)
    target = total * new_share
    remaining = total - target
    require(remaining >= -1e-9, f"Composition share too large for {polymer}: {new_share}")
    data.polymer_kg_per_batch[polymer] = max(0.0, target)
    if other_total > 0:
        for other in others:
            data.polymer_kg_per_batch[other] = remaining * float(data.polymer_kg_per_batch[other]) / other_total
    elif remaining > 1e-9:
        raise ValueError("Cannot redistribute composition because all other polymer masses are zero.")


def eligible_feed_tonnes(data: Any, route: str) -> float:
    eligible_mass_per_batch = sum(
        data.polymer_kg_per_batch[p] for p in data.polymers if data.eligibility[p, route]
    )
    return data.total_batches * eligible_mass_per_batch / 1000.0


def locate_parameter_config(explicit: Optional[Path] = None) -> Optional[Path]:
    candidates: List[Path] = []
    if explicit is not None:
        explicit_path = Path(explicit).expanduser()
        if not explicit_path.exists():
            raise FileNotFoundError(f"Explicit sensitivity configuration not found: {explicit_path}")
        candidates.append(explicit_path)
    env_value = os.environ.get("IS_SENSITIVITY_CONFIG")
    if env_value:
        candidates.append(Path(env_value).expanduser())
    candidates.extend([
        SCRIPT_DIR / PARAMETER_CONFIG_FILENAME,
        PROJECT_ROOT.parent / "parameters" / PARAMETER_CONFIG_FILENAME,
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def load_parameter_config(
    explicit: Optional[Path] = None,
) -> Tuple[Optional[Path], Dict[str, Any]]:
    """
    Load optional JSON overrides.

    The runner is intentionally self-contained: when no JSON is present it uses
    the auditable numerical defaults embedded in the parameter registry. An
    existing JSON can still override ranges and can define a preferred ordering
    or a configured-only subset.
    """
    path = locate_parameter_config(explicit)
    if path is None:
        return None, {
            "parameters": {},
            "analyses": {
                "local_decision_boundaries": {
                    "parameter_ids": [],
                }
            },
        }
    config = read_json(path)
    config.setdefault("parameters", {})
    config.setdefault("analyses", {})
    config["analyses"].setdefault("local_decision_boundaries", {"parameter_ids": []})
    require(isinstance(config.get("parameters"), Mapping),
            "Parameter JSON 'parameters' entry must be an object.")
    require(isinstance(config.get("analyses"), Mapping),
            "Parameter JSON 'analyses' entry must be an object.")
    local = config["analyses"].get("local_decision_boundaries", {})
    require(isinstance(local, Mapping),
            "analyses.local_decision_boundaries must be an object.")
    local.setdefault("parameter_ids", [])
    require(isinstance(local.get("parameter_ids"), list),
            "analyses.local_decision_boundaries.parameter_ids must be a list.")
    return path, config


def _resolve_range_value(definition: Mapping[str, Any], central: float, side: str) -> float:
    mode = str(definition.get("mode", "absolute"))
    if mode == "absolute":
        return float(definition[side])
    if mode == "nominal_multiplier":
        return central * float(definition[f"{side}_factor"])
    if mode == "nominal_additive":
        return central + float(definition[f"{side}_delta"])
    raise ValueError(f"Unsupported range mode {mode!r} for {side} bound.")


def _apply_optional_parameter_metadata(
    spec: SensitivityParameter,
    definition: Mapping[str, Any],
) -> None:
    spec.classification = str(definition.get("classification", spec.classification))
    spec.lcia_factor_status = str(
        definition.get("lcia_factor_status", spec.lcia_factor_status)
    )
    spec.source_or_basis = str(definition.get("range_basis", spec.source_or_basis))
    spec.scope_level = str(definition.get("scope_level", spec.scope_level))
    spec.target = str(definition.get("target", spec.target))
    spec.linked_quantity_note = str(
        definition.get("linked_quantity_note", spec.linked_quantity_note)
    )
    if "label" in definition:
        spec.label = str(definition["label"])
    if "description" in definition:
        spec.description = str(definition["description"])
    if "affects_optimisation" in definition:
        spec.affects_optimisation = bool(definition["affects_optimisation"])


def apply_config_definition(
    spec: SensitivityParameter,
    config: Mapping[str, Any],
) -> SensitivityParameter:
    """
    Apply optional JSON overrides without silently dropping built-in parameters.

    The previous implementation treated the external JSON as an allow-list:
    any parameter without a JSON entry was relabelled ``__not_configured__`` and
    removed from the analysis. That behaviour was the main reason plastic
    parameters could exist in the registry but never be solved. The registry is
    now authoritative; the JSON only overrides ranges, labels and metadata.
    """
    parameter_definitions = config.get("parameters", {})
    definition = parameter_definitions.get(spec.parameter_id)
    matched_key = spec.parameter_id if definition is not None else None
    if definition is None:
        wildcard_matches = [
            (key, value)
            for key, value in parameter_definitions.items()
            if any(token in str(key) for token in ("*", "?", "["))
            and fnmatch.fnmatchcase(spec.parameter_id, str(key))
        ]
        if wildcard_matches:
            # Prefer the most specific pattern: most literal characters, then
            # the earliest JSON entry for stable and auditable precedence.
            wildcard_matches.sort(
                key=lambda item: sum(ch not in "*?[]" for ch in str(item[0])),
                reverse=True,
            )
            matched_key, definition = wildcard_matches[0]
    if definition is None:
        spec.configuration_source = "built_in_default"
        if not spec.classification:
            spec.classification = "built_in_decision_boundary_parameter"
        if spec.parameter_type == "binary":
            spec.hard_low, spec.hard_high = 0.0, 1.0
        return spec

    require(isinstance(definition, Mapping),
            f"Configuration for {spec.parameter_id!r} must be an object.")
    spec.configuration_source = f"external_json_override:{matched_key}"
    _apply_optional_parameter_metadata(spec, definition)

    # Binary facility/acceptance assumptions are discrete stress tests rather
    # than numerical boundaries. They may be named in the JSON, but they do not
    # require continuous uncertainty/boundary ranges.
    if spec.parameter_type == "binary":
        spec.central = float(definition.get("nominal", spec.central))
        spec.low = 0.0
        spec.high = 1.0
        spec.boundary_low = 0.0
        spec.boundary_high = 1.0
        spec.hard_low = 0.0
        spec.hard_high = 1.0
        require(spec.central in (0.0, 1.0),
                f"Binary parameter {spec.parameter_id!r} must have nominal 0 or 1.")
        return spec

    uncertainty = definition.get("uncertainty_range")
    boundary = definition.get("boundary_range")
    if uncertainty is not None:
        require(isinstance(uncertainty, Mapping),
                f"uncertainty_range for {spec.parameter_id} must be an object")
        spec.low = _resolve_range_value(uncertainty, spec.central, "low")
        spec.high = _resolve_range_value(uncertainty, spec.central, "high")
    if boundary is not None:
        require(isinstance(boundary, Mapping),
                f"boundary_range for {spec.parameter_id} must be an object")
        spec.boundary_low = _resolve_range_value(boundary, spec.central, "low")
        spec.boundary_high = _resolve_range_value(boundary, spec.central, "high")

    hard = definition.get("hard_bounds", {}) or {}
    spec.hard_low = float(hard["low"]) if "low" in hard else spec.hard_low
    spec.hard_high = float(hard["high"]) if "high" in hard else spec.hard_high
    if spec.hard_low is not None:
        spec.low = max(spec.low, spec.hard_low)
        spec.boundary_low = max(spec.boundary_low, spec.hard_low)
    if spec.hard_high is not None:
        spec.high = min(spec.high, spec.hard_high)
        spec.boundary_high = min(spec.boundary_high, spec.hard_high)

    require(spec.boundary_low <= spec.low <= spec.central <= spec.high <= spec.boundary_high,
            f"Invalid configured range ordering for {spec.parameter_id}: "
            f"{spec.boundary_low}, {spec.low}, {spec.central}, {spec.high}, {spec.boundary_high}")
    return spec


def build_parameter_registry(
    central_data: Any,
    impact: ImpactModel,
    config: Mapping[str, Any],
) -> List[SensitivityParameter]:
    specs: List[SensitivityParameter] = []
    add = specs.append

    grid_cost = float(central_data.electricity_cost_gbp_per_kwh["grid"])
    wind_cost = float(central_data.electricity_cost_gbp_per_kwh["wind"])
    premium = wind_cost / grid_cost - 1.0

    def apply_premium(state: ScenarioState, value: float) -> None:
        require(value > -1.0, "Renewable premium cannot make electricity price negative.")
        state.data.electricity_cost_gbp_per_kwh["wind"] = (
            state.data.electricity_cost_gbp_per_kwh["grid"] * (1.0 + value)
        )
        state.effective_value = value

    def apply_grid_price_multiplier(state: ScenarioState, value: float) -> None:
        require(value > 0.0, "Grid electricity price multiplier must be positive.")
        new_grid_price = grid_cost * float(value)
        state.data.electricity_cost_gbp_per_kwh["grid"] = new_grid_price
        state.data.autoclave_cost_gbp_per_t = {
            site: state.controls.autoclave_kwh_per_kg * 1000.0 * new_grid_price
            for site in state.data.sites
        }
        state.effective_value = value

    add(SensitivityParameter(
        "grid_electricity_price_multiplier", "UK grid electricity price", "electricity", "× current grid price",
        1.0, 0.75, 1.25, 0.25, 3.0, apply_grid_price_multiplier,
        "Scales the UK grid electricity purchase price and the electricity component of autoclave operating cost. The renewable price is held at its nominal absolute value in this OAT test.",
        "Default local market uncertainty of ±25%; wider decision-boundary range permits major electricity-price shocks.",
    ))

    add(SensitivityParameter(
        "renewable_cppa_premium", "Renewable cPPA premium / discount", "electricity", "fraction vs grid",
        premium, -0.20, 0.50, -0.75, 2.00, apply_premium,
        "Wind contract price expressed as a premium (+) or discount (-) to the current UK grid price.",
        "Current MILP grid and renewable electricity prices; broad market-shock range.",
        display_decimals=4,
    ))

    for source in ("grid", "wind"):
        for category in UNCERTAIN_IMPACT_CATEGORIES:
            affects = category == "gwp"
            pid = f"electricity_lcia_multiplier__{source}__{category}"
            label = f"{source.title()} electricity {OUTPUT_LABELS.get(category, category)} factor"

            def apply_electricity_lcia(state: ScenarioState, value: float, s=source, c=category) -> None:
                state.controls.electricity_multiplier[s, c] = float(value)
                state.effective_value = value

            add(SensitivityParameter(
                pid, label, "electricity", "× current LCIA factor",
                1.0, 0.50, 1.50, 0.10, 3.00, apply_electricity_lcia,
                (
                    f"Scales the current {source} electricity {category} factor. "
                    + ("Grid GWP also changes BEV transport GWP and embedded autoclave GWP."
                       if source == "grid" and category == "gwp"
                       else "This category is reported after optimisation unless it is GWP, which enters the current objective.")
                ),
                "Current route-factor workbook; multiplicative uncertainty because no alternative factor distribution was supplied.",
                affects_optimisation=affects,
            ))

    central_electricity = float(central_data.electricity_kwh_per_batch)

    def apply_electricity_demand(state: ScenarioState, value: float) -> None:
        state.data.electricity_kwh_per_batch = float(value)
        for site in state.data.sites:
            state.data.electricity_kwh[site] = state.data.batches[site] * float(value)
        state.effective_value = value

    add(SensitivityParameter(
        "electricity_demand_per_batch", "Process electricity demand", "process_variability", "kWh/batch",
        central_electricity, 0.75 * central_electricity, 1.25 * central_electricity,
        0.25 * central_electricity, 2.00 * central_electricity, apply_electricity_demand,
        "Changes manufacturing electricity demand per batch while holding all other per-batch flows fixed.",
        "Local process variability default of ±25%; adaptive boundary search extends to 25–200% of central.",
        display_decimals=1,
    ))

    central_auto = impact.central_autoclave_kwh_per_kg

    def apply_autoclave(state: ScenarioState, value: float) -> None:
        state.controls.autoclave_kwh_per_kg = float(value)
        state.data.autoclave_cost_gbp_per_t = {
            site: float(value) * 1000.0 * state.data.electricity_cost_gbp_per_kwh["grid"]
            for site in state.data.sites
        }
        state.effective_value = value

    add(SensitivityParameter(
        "autoclave_energy_demand", "Autoclave energy demand", "plastics", "kWh/kg plastic",
        central_auto, 0.20, 1.40, 0.0, 3.0, apply_autoclave,
        "Replaces the embedded autoclave component in landfill and recycling route factors and updates autoclave operating cost.",
        "Existing LCA low/central/high assumptions 0.2/0.8/1.4 kWh/kg; boundary search extends to 3 kWh/kg.",
    ))

    recyclable_polymers: List[str] = []
    for polymer in central_data.polymers:
        active_routes = [
            route for route in central_data.routes
            if central_data.eligibility[polymer, route] and central_data.recovery_yield[polymer, route] > 0
        ]
        if not active_routes:
            continue
        recyclable_polymers.append(polymer)
        central_eta = float(central_data.recovery_yield[polymer, active_routes[0]])

        def apply_recovery(state: ScenarioState, value: float, p=polymer) -> None:
            require(0.0 <= value <= 1.0, "Recovery fraction must be in [0,1].")
            for route in state.data.routes:
                if state.data.eligibility[p, route] and route.startswith("mechanical_recycling"):
                    state.data.recovery_yield[p, route] = float(value)
                    state.data.residual_fraction[p, route] = 1.0 - float(value)
            state.effective_value = value

        add(SensitivityParameter(
            f"recovery_fraction__{polymer}", f"{polymer} mechanical-recycling recovery", "plastics", "fraction of input",
            central_eta, max(0.0, central_eta - 0.30), min(1.0, central_eta + 0.20), 0.0, 1.0,
            apply_recovery,
            "Scales recovered mass, recyclate revenue, recycling-process burden and avoided virgin-material LCIA credit. Rejection is 1 − recovery.",
            "Central recovery is loaded explicitly from the route-factor workbook; the avoided-material credit is separately constrained to virgin-polymer production × recovery × the 0.81 substitution ratio.",
        ))

        central_value = max(
            float(central_data.recovered_value_gbp_per_t[polymer, route])
            for route in active_routes
        )

        def apply_market_value(state: ScenarioState, value: float, p=polymer) -> None:
            for route in state.data.routes:
                if (p, route) in state.data.recovered_value_gbp_per_t and route.startswith("mechanical_recycling"):
                    state.data.recovered_value_gbp_per_t[p, route] = float(value)
            state.effective_value = value

        add(SensitivityParameter(
            f"recyclate_value__{polymer}", f"{polymer} recyclate market value", "plastics", "GBP/t recovered output",
            central_value, 0.25 * central_value, 1.75 * central_value, 0.0, 3.0 * central_value,
            apply_market_value,
            "Changes the sale value of recovered recyclate; the credit is still multiplied by recovery yield and value-capture fraction.",
            "Current polymer-specific market-value assumption; 25–175% reporting range and 0–300% boundary range.",
            display_decimals=1,
        ))

        def apply_value_capture(state: ScenarioState, value: float, p=polymer) -> None:
            require(0.0 <= value <= 1.0, "Value-capture fraction must be in [0,1].")
            for route in state.data.routes:
                if route.startswith("mechanical_recycling") and (p, route) in state.data.value_capture:
                    state.data.value_capture[p, route] = float(value)
            state.effective_value = value

        add(SensitivityParameter(
            f"recyclate_quality_value_capture__{polymer}", f"{polymer} recyclate quality / value capture", "plastics", "fraction",
            1.0, 0.50, 1.0, 0.0, 1.0, apply_value_capture,
            "Economic quality factor: fraction of the stated recyclate market value that can actually be captured.",
            "Central model assumes saleable recovered output (1.0); bounded physical range 0–1.",
        ))

        def apply_substitution(state: ScenarioState, value: float, p=polymer) -> None:
            require(0.0 <= value <= 1.0, "Substitution efficiency must be in [0,1].")
            state.controls.substitution_efficiency[p] = float(value)
            state.effective_value = value

        substitution_values = [
            float(central_data.substitution_ratio[polymer, route])
            for route in active_routes
        ]
        require(
            max(substitution_values) - min(substitution_values) <= 1e-12,
            f"Substitution-ratio sensitivity requires equal central values for {polymer}.",
        )
        central_substitution = substitution_values[0]
        add(SensitivityParameter(
            f"substitution_efficiency__{polymer}", f"{polymer} virgin-material substitution efficiency", "plastics", "fraction",
            central_substitution, 0.50, 1.0, 0.0, 1.0, apply_substitution,
            "Environmental quality factor: recovered output displaces this fraction of virgin production on a functional-equivalence basis.",
            "Central route factor uses the explicit 0.81 Volk quality/downcycling substitution ratio; bounded range 0–1.",
        ))

        def apply_credit_multiplier(state: ScenarioState, value: float, p=polymer) -> None:
            require(value >= 0.0, "Avoided-credit multiplier must be non-negative.")
            for category in UNCERTAIN_IMPACT_CATEGORIES:
                state.controls.avoided_credit_multiplier[p, category] = float(value)
            state.effective_value = value

        add(SensitivityParameter(
            f"avoided_virgin_credit_multiplier__{polymer}", f"{polymer} avoided virgin-production LCIA credit", "plastics", "× current avoided credit",
            1.0, 0.50, 1.50, 0.0, 3.0, apply_credit_multiplier,
            "Scales the avoided virgin-production credit in the three established midpoint/output categories without changing recovery, market value, or the reporting-only DALY factor.",
            "Multiplicative uncertainty around the current mapped virgin-production proxy.",
        ))

    # Plastic-wide parameters complement the polymer-specific tests above.
    # They answer the system-level question "how robust is the overall recycling
    # strategy?" while the individual parameters identify which polymer drives
    # the transition.
    recycling_pairs: List[Tuple[str, str]] = [
        (polymer, route)
        for polymer in recyclable_polymers
        for route in central_data.routes
        if route.startswith("mechanical_recycling")
        and central_data.eligibility[polymer, route]
        and central_data.recovery_yield[polymer, route] > 0
    ]

    if recycling_pairs:
        central_substitution_values = [
            float(central_data.substitution_ratio[pair])
            for pair in recycling_pairs
        ]
        require(
            max(central_substitution_values) - min(central_substitution_values) <= 1e-12,
            "The all-polymer substitution sensitivity requires one common central substitution ratio.",
        )
        central_all_substitution = central_substitution_values[0]
        central_recovery_by_pair = {
            pair: float(central_data.recovery_yield[pair])
            for pair in recycling_pairs
        }

        def apply_all_recovery_multiplier(state: ScenarioState, value: float) -> None:
            require(value >= 0.0, "Recovery multiplier must be non-negative.")
            for polymer, route in recycling_pairs:
                recovery = min(1.0, max(
                    0.0,
                    central_recovery_by_pair[polymer, route] * float(value),
                ))
                state.data.recovery_yield[polymer, route] = recovery
                state.data.residual_fraction[polymer, route] = 1.0 - recovery
            state.effective_value = value

        add(SensitivityParameter(
            "mechanical_recycling_recovery_multiplier__all",
            "All mechanical-recycling recovery rates",
            "plastics",
            "× current polymer recovery",
            1.0, 0.50, 1.25, 0.0, 1.50,
            apply_all_recovery_multiplier,
            "Scales every polymer-specific mechanical-recycling recovery rate together. "
            "Each linked rejection rate is recalculated exactly as 1 − recovery, so "
            "recovery and rejection cannot become physically inconsistent.",
            "System-wide plastic recovery stress test around the loaded polymer-specific values.",
            hard_low=0.0,
            hard_high=2.0,
            scope_level="all_plastics",
            target="LDPE, LLDPE, PP and TPE mechanical-recycling routes",
            linked_quantity_note="rejection fraction = 1 - recovery fraction",
        ))

        central_value_by_pair = {
            pair: float(central_data.recovered_value_gbp_per_t[pair])
            for pair in recycling_pairs
            if pair in central_data.recovered_value_gbp_per_t
        }

        def apply_all_market_value_multiplier(state: ScenarioState, value: float) -> None:
            require(value >= 0.0, "Recyclate-value multiplier must be non-negative.")
            for pair, central_value in central_value_by_pair.items():
                state.data.recovered_value_gbp_per_t[pair] = central_value * float(value)
            state.effective_value = value

        add(SensitivityParameter(
            "recyclate_market_value_multiplier__all",
            "All recyclate market values",
            "plastics",
            "× current polymer value",
            1.0, 0.50, 1.50, 0.0, 3.0,
            apply_all_market_value_multiplier,
            "Scales the market value of all recovered polymers together while retaining "
            "the central relative value differences between LDPE, LLDPE, PP and TPE.",
            "System-wide recyclate-market stress test; polymer-specific absolute values are also tested separately.",
            hard_low=0.0,
            scope_level="all_plastics",
            target="all saleable mechanically recycled polymers",
        ))

        def apply_all_value_capture(state: ScenarioState, value: float) -> None:
            require(0.0 <= value <= 1.0, "Value-capture fraction must be in [0,1].")
            for pair in recycling_pairs:
                if pair in state.data.value_capture:
                    state.data.value_capture[pair] = float(value)
            state.effective_value = value

        add(SensitivityParameter(
            "recyclate_quality_value_capture__all",
            "All recyclate economic quality / value capture",
            "plastics",
            "fraction",
            1.0, 0.50, 1.0, 0.0, 1.0,
            apply_all_value_capture,
            "Applies one economic-quality factor to all recyclates: the fraction of each "
            "polymer's stated market price that can actually be realised.",
            "Central model assumes fully saleable output; physical range 0–1.",
            hard_low=0.0,
            hard_high=1.0,
            scope_level="all_plastics",
            target="all saleable mechanically recycled polymers",
        ))

        def apply_all_substitution(state: ScenarioState, value: float) -> None:
            require(0.0 <= value <= 1.0, "Substitution efficiency must be in [0,1].")
            for polymer in recyclable_polymers:
                state.controls.substitution_efficiency[polymer] = float(value)
            state.effective_value = value

        add(SensitivityParameter(
            "substitution_efficiency__all",
            "All recyclate virgin-material substitution efficiencies",
            "plastics",
            "fraction",
            central_all_substitution, 0.50, 1.0, 0.0, 1.0,
            apply_all_substitution,
            "Applies one functional-substitution efficiency to every mechanically "
            "recycled polymer. This changes avoided virgin-production impacts but not "
            "the physical recovery yield or recyclate sale revenue.",
            "Central route factors use the explicit 0.81 Volk quality/downcycling substitution ratio; physical range 0–1.",
            hard_low=0.0,
            hard_high=1.0,
            scope_level="all_plastics",
            target="all mechanically recycled polymers",
        ))

        def apply_all_credit_multiplier(state: ScenarioState, value: float) -> None:
            require(value >= 0.0, "Avoided-credit multiplier must be non-negative.")
            for polymer in recyclable_polymers:
                for category in UNCERTAIN_IMPACT_CATEGORIES:
                    state.controls.avoided_credit_multiplier[polymer, category] = float(value)
            state.effective_value = value

        add(SensitivityParameter(
            "avoided_virgin_credit_multiplier__all",
            "All avoided virgin-production LCIA credits",
            "plastics",
            "× current avoided credit",
            1.0, 0.50, 1.50, 0.0, 3.0,
            apply_all_credit_multiplier,
            "Scales the avoided virgin-production credit for every recyclable polymer "
            "in the three established LCIA categories. GWP changes the optimiser objective; "
            "fossil-resource and water-scarcity changes are reported after optimisation. "
            "The reporting-only DALY output is not an uncertain input.",
            "System-wide proxy uncertainty around the current virgin-material mappings.",
            hard_low=0.0,
            scope_level="all_plastics",
            target="all recyclable polymers and LCIA categories",
        ))

    # Direct polymer-mass parameters support a focused polymer-level inventory
    # analysis without independently sampling composition shares that must sum
    # to one. Only polymers selected in the external JSON are retained.
    for polymer in central_data.polymers:
        central_mass = float(central_data.polymer_kg_per_batch[polymer])

        def apply_polymer_mass(state: ScenarioState, value: float, p=polymer) -> None:
            require(value >= 0.0, "Polymer mass per batch cannot be negative.")
            state.data.polymer_kg_per_batch[p] = float(value)
            state.effective_value = value

        add(SensitivityParameter(
            f"polymer_mass_per_batch__{polymer}", f"{polymer} mass per batch",
            "material_composition", "kg/batch", central_mass,
            0.75 * central_mass, 1.25 * central_mass,
            0.25 * central_mass, 2.0 * central_mass, apply_polymer_mass,
            f"Changes the {polymer} mass generated per batch while other polymer masses remain at their central values.",
            "Configured externally; only selected dominant polymers are retained.",
            display_decimals=3,
        ))

    total_plastic = sum(float(v) for v in central_data.polymer_kg_per_batch.values())

    def apply_total_plastic(state: ScenarioState, value: float) -> None:
        factor = float(value) / total_plastic
        for polymer in state.data.polymers:
            state.data.polymer_kg_per_batch[polymer] *= factor
        state.effective_value = value

    add(SensitivityParameter(
        "total_plastic_waste_per_batch", "Total SUT plastic waste", "material_composition", "kg/batch",
        total_plastic, 0.75 * total_plastic, 1.25 * total_plastic, 0.25 * total_plastic, 2.0 * total_plastic,
        apply_total_plastic,
        "Scales all polymer masses proportionally, preserving central composition.",
        "Process/inventory variability default ±25%; extended boundary search 25–200% of central.",
        display_decimals=2,
    ))

    for polymer in central_data.polymers:
        central_share = float(central_data.polymer_kg_per_batch[polymer]) / total_plastic
        low = max(0.0, 0.50 * central_share)
        high = min(0.95, 1.50 * central_share)

        def apply_share(state: ScenarioState, value: float, p=polymer) -> None:
            set_composition_share(state.data, p, value)
            state.effective_value = value

        add(SensitivityParameter(
            f"polymer_composition_share__{polymer}", f"SUT composition: {polymer}", "material_composition", "mass fraction",
            central_share, low, high, 0.0, 0.98, apply_share,
            f"Changes {polymer} share while holding total plastic mass constant; all other polymer masses are rescaled proportionally.",
            "Composition OAT range is ±50% relative around the current share, bounded to a feasible 0–98%.",
        ))

    central_batches = float(central_data.total_batches)

    def apply_throughput(state: ScenarioState, value: float) -> None:
        rescale_throughput(state.data, value)
        state.effective_value = state.data.total_batches

    add(SensitivityParameter(
        "manufacturing_throughput", "Manufacturing throughput", "throughput", "batches/year",
        central_batches, 0.80 * central_batches, 1.20 * central_batches, 0.50 * central_batches, 1.50 * central_batches,
        apply_throughput,
        "Changes annual throughput while preserving the central low/high site-allocation ratio and active-site pattern.",
        "±20% reporting range and ±50% boundary range; effective values are integer batch allocations.",
        display_decimals=0,
    ))

    central_ratio = float(central_data.batch_low / central_data.batch_high)

    def apply_batch_ratio(state: ScenarioState, value: float) -> None:
        set_low_high_ratio(state.data, value)
        state.effective_value = state.data.batch_low / state.data.batch_high

    add(SensitivityParameter(
        "low_to_high_batch_ratio", "Low-site / high-site batch allocation", "throughput", "ratio",
        central_ratio, max(0.20, central_ratio - 0.25), 1.0, 0.05, 1.0, apply_batch_ratio,
        "Redistributes the fixed total batches between the unique low-volume site and the two high-volume sites.",
        "Bounded by one equal-allocation case and a near-zero low-site allocation.",
    ))

    central_utility = float(central_data.utility_water_m3_per_batch)

    def apply_utility_water(state: ScenarioState, value: float) -> None:
        state.data.utility_water_m3_per_batch = float(value)
        for site in state.data.sites:
            state.data.utility_water_m3[site] = state.data.batches[site] * float(value)
        state.effective_value = value

    add(SensitivityParameter(
        "utility_water_demand_per_batch", "Eligible utility-water demand", "water", "m3/batch",
        central_utility, 0.50 * central_utility, 1.50 * central_utility, 0.0, 2.0 * central_utility,
        apply_utility_water,
        "Changes non-contact utility-water demand that may be supplied by fresh water or treated-water reuse; WFI is unchanged.",
        "Broad range reflects that the current utility-water demand is an explicit modelling assumption.",
    ))

    central_ww = float(central_data.wastewater_m3_per_batch)

    def apply_wastewater(state: ScenarioState, value: float) -> None:
        state.data.wastewater_m3_per_batch = float(value)
        for site in state.data.sites:
            state.data.wastewater_m3[site] = state.data.batches[site] * float(value)
        state.effective_value = value

    add(SensitivityParameter(
        "wastewater_generation_per_batch", "Wastewater generation", "water", "m3/batch",
        central_ww, 0.75 * central_ww, 1.25 * central_ww, 0.25 * central_ww, 2.0 * central_ww,
        apply_wastewater,
        "Changes wastewater routed through treatment and the treatment-capacity constraint.",
        "Process variability default ±25%; boundary search extends to 25–200%.",
    ))

    central_reuse = float(next(iter(central_data.water_recovery.values())))

    def apply_reuse(state: ScenarioState, value: float) -> None:
        require(0.0 <= value <= 1.0, "Water recovery fraction must be in [0,1].")
        state.data.water_recovery = {site: float(value) for site in state.data.sites}
        state.effective_value = value

    add(SensitivityParameter(
        "achievable_water_reuse_fraction", "Achievable treated-water recovery", "water", "fraction",
        central_reuse, max(0.0, central_reuse - 0.50), min(1.0, central_reuse), 0.0, 1.0,
        apply_reuse,
        "Fraction of wastewater routed to reuse that becomes usable non-contact utility water.",
        "Central value is taken from the current MILP input-output workbook, not overwritten by an older settings file.",
    ))

    # Facility capacities are expressed as a fraction of all currently eligible annual feed.
    for route in [r for r in central_data.routes if r.startswith("mechanical_recycling")]:
        central_feed = eligible_feed_tonnes(central_data, route)
        if central_feed <= 0:
            continue
        route_label = ROUTE_LABELS.get(route, route)

        def apply_capacity(state: ScenarioState, value: float, r=route, feed=central_feed) -> None:
            require(value >= 0.0, "Capacity fraction must be non-negative.")
            state.data.route_capacities_t[r] = float(value) * feed
            state.effective_value = value

        loaded_capacity = finite(
            central_data.route_capacities_t.get(route, np.inf)
            if hasattr(central_data.route_capacities_t, "get")
            else central_data.route_capacities_t[route],
            np.inf,
        )
        central_capacity_fraction = (
            loaded_capacity / central_feed if math.isfinite(loaded_capacity) else 1.0
        )
        # Treat a numerically non-binding capacity as one full central feed in
        # the reported parameterisation; the applied value still writes an
        # explicit physical capacity to the scenario data.
        if central_capacity_fraction >= 1e6:
            central_capacity_fraction = 1.0

        add(SensitivityParameter(
            f"recycling_capacity_fraction__{route}",
            f"{route_label} capacity",
            "network",
            "fraction of eligible central feed",
            central_capacity_fraction,
            0.0,
            max(1.20, 1.25 * central_capacity_fraction),
            0.0,
            max(2.0, 2.0 * central_capacity_fraction),
            apply_capacity,
            "Caps total annual route throughput relative to the central eligible feed. "
            "The nominal value is reconstructed from the loaded route capacity rather "
            "than hard-coded to 1.0.",
            "Loaded MILP route capacity divided by central eligible annual feed; "
            "non-binding/infinite capacities are represented as 1.0.",
            hard_low=0.0,
            scope_level="route",
            target=route,
        ))

        def apply_availability(state: ScenarioState, value: float, r=route) -> None:
            available = int(round(value)) == 1
            for polymer in state.data.polymers:
                if r in state.data.routes and state.data.eligibility[polymer, r]:
                    state.data.eligibility[polymer, r] = 1 if available else 0
            state.effective_value = float(int(available))

        add(SensitivityParameter(
            f"recycling_facility_available__{route}", f"{route_label} facility available", "discrete_availability", "0/1",
            1.0, 0.0, 1.0, 0.0, 1.0, apply_availability,
            "Binary availability stress test: removes this route from the superstructure when set to zero.",
            "Discrete future facility-availability uncertainty.",
            parameter_type="binary",
        ))

    for polymer in recyclable_polymers:
        def apply_acceptance(state: ScenarioState, value: float, p=polymer) -> None:
            accepted = int(round(value)) == 1
            for route in state.data.routes:
                if route.startswith("mechanical_recycling") and (p, route) in state.data.eligibility:
                    state.data.eligibility[p, route] = 1 if accepted and central_data.eligibility[p, route] else 0
            state.effective_value = float(int(accepted))

        add(SensitivityParameter(
            f"pharma_plastic_accepted__{polymer}", f"Recycler accepts pharmaceutical {polymer}", "discrete_availability", "0/1",
            1.0, 0.0, 1.0, 0.0, 1.0, apply_acceptance,
            f"Binary acceptance test for contaminated/pharmaceutical {polymer} waste at the current mechanical-recycling facility.",
            "Discrete acceptance uncertainty; all non-recycling routes remain available.",
            parameter_type="binary",
        ))

    def apply_recycling_distance(state: ScenarioState, value: float) -> None:
        for site in state.data.sites:
            for route in state.data.routes:
                if route.startswith("mechanical_recycling"):
                    state.data.distance_km[site, route] = central_data.distance_km[site, route] * float(value)
        state.effective_value = value

    add(SensitivityParameter(
        "recycling_transport_distance_multiplier", "Distance to recycling facilities", "transport", "× current one-way distance",
        1.0, 0.50, 2.00, 0.10, 4.00, apply_recycling_distance,
        "Scales all site-to-recycler distances while leaving landfill and EfW distances unchanged.",
        "Location/network uncertainty; broad 50–200% reporting range.",
    ))

    def apply_all_distance(state: ScenarioState, value: float) -> None:
        for key, distance in central_data.distance_km.items():
            state.data.distance_km[key] = float(distance) * float(value)
        state.effective_value = value

    add(SensitivityParameter(
        "all_waste_transport_distance_multiplier", "All waste-route distances", "transport", "× current one-way distance",
        1.0, 0.75, 1.50, 0.25, 3.00, apply_all_distance,
        "Scales all waste transport distances, preserving their relative network geography.",
        "General network-distance uncertainty.",
    ))

    for mode in central_data.transport_modes:
        central_payload = float(central_data.payload_t[mode])

        def apply_payload(state: ScenarioState, value: float, m=mode) -> None:
            require(value > 0.0, "Payload must be positive.")
            state.data.payload_t[m] = float(value)
            state.effective_value = value

        add(SensitivityParameter(
            f"vehicle_payload__{mode}", f"{MODE_LABELS.get(mode, mode)} payload", "transport", "t/vehicle",
            central_payload, 0.75 * central_payload, 1.25 * central_payload,
            0.50 * central_payload, 1.50 * central_payload, apply_payload,
            "Changes the payload constraint and therefore the integer number of annual journeys.",
            "Operational loading uncertainty around the current maximum-payload assumption.",
        ))

        central_mode_cost = float(central_data.transport_cost_gbp_per_km[mode])

        def apply_mode_cost(state: ScenarioState, value: float, m=mode) -> None:
            require(value >= 0.0, "Transport cost cannot be negative.")
            state.data.transport_cost_gbp_per_km[m] = float(value)
            state.effective_value = value

        add(SensitivityParameter(
            f"transport_cost__{mode}", f"{MODE_LABELS.get(mode, mode)} transport cost", "transport", "GBP/vehicle-route-km",
            central_mode_cost, 0.75 * central_mode_cost, 1.50 * central_mode_cost,
            0.25 * central_mode_cost, 3.0 * central_mode_cost, apply_mode_cost,
            "Changes the all-in per-vehicle-route-km service cost. BEV charging remains embedded in the BEV rate.",
            "Current transport-service coefficient with broad market/technology cost range.",
        ))

    # Gate/reprocessing fees. Landfill, EfW and the current mechanical-
    # recycling regranulation fee are common coefficients. The source parameter
    # file repeats the documented Volk-derived GBP2026/t value for LDPE, LLDPE,
    # PP and TPE, so one uncertainty parameter
    # updates every eligible mechanical-recycling entry together.
    landfill_routes = [(p, r) for p in central_data.polymers for r in central_data.routes if r == "landfill_all"]
    efw_routes = [(p, r) for p in central_data.polymers for r in central_data.routes if r == "incineration_efw_heat_electricity"]
    recycling_routes = [
        (p, r) for p in central_data.polymers for r in central_data.routes
        if r.startswith("mechanical_recycling") and central_data.eligibility[p, r]
    ]
    central_landfill = float(central_data.route_cost_gbp_per_t[landfill_routes[0]])
    central_efw = float(central_data.route_cost_gbp_per_t[efw_routes[0]])

    def apply_landfill_fee(state: ScenarioState, value: float) -> None:
        for key in landfill_routes:
            state.data.route_cost_gbp_per_t[key] = float(value)
        state.effective_value = value

    add(SensitivityParameter(
        "landfill_gate_fee", "Landfill gate fee + tax", "market_fees", "GBP/t input",
        central_landfill, 0.50 * central_landfill, 1.50 * central_landfill, 0.0, 3.0 * central_landfill,
        apply_landfill_fee,
        "Changes the all-in landfill disposal coefficient used by every polymer.",
        "Configured externally.",
        display_decimals=1,
    ))

    def apply_efw_fee(state: ScenarioState, value: float) -> None:
        for key in efw_routes:
            state.data.route_cost_gbp_per_t[key] = float(value)
        state.effective_value = value

    add(SensitivityParameter(
        "efw_gate_fee", "Energy-from-Waste gate fee", "market_fees", "GBP/t input",
        central_efw, 0.50 * central_efw, 1.50 * central_efw, 0.0, 3.0 * central_efw,
        apply_efw_fee,
        "Changes the EfW gate-fee coefficient used by every polymer.",
        "Configured externally.",
        display_decimals=1,
    ))

    if recycling_routes:
        fee_values = [float(central_data.route_cost_gbp_per_t[key]) for key in recycling_routes]
        require(max(fee_values) - min(fee_values) < 1e-9,
                "The common recycling-fee sensitivity requires equal central fees across eligible polymers.")
        central_recycling_fee = fee_values[0]

        def apply_recycling_fee(state: ScenarioState, value: float) -> None:
            require(value >= 0.0, "Recycling fee cannot be negative.")
            for key in recycling_routes:
                state.data.route_cost_gbp_per_t[key] = float(value)
            state.effective_value = value

        add(SensitivityParameter(
            "recycling_gate_fee", "Mechanical-recycling handling fee", "market_fees", "GBP/t input",
            central_recycling_fee, 0.50 * central_recycling_fee, 1.50 * central_recycling_fee,
            0.0, 3.0 * max(central_recycling_fee, 1.0), apply_recycling_fee,
            "Changes the common handling/reprocessing fee per tonne of accepted plastic input for all eligible mechanical-recycling routes.",
            "Configured externally; the current model repeats the same Volk-derived GBP2026/t coefficient for LDPE, LLDPE, PP and TPE.",
            display_decimals=1,
        ))

    central_carbon = float(central_data.lifecycle_carbon_value_gbp_per_t)

    def apply_carbon_value(state: ScenarioState, value: float) -> None:
        require(value >= 0.0, "Carbon value cannot be negative.")
        state.data.lifecycle_carbon_value_gbp_per_t = float(value)
        state.effective_value = value

    add(SensitivityParameter(
        "lifecycle_carbon_value", "Lifecycle carbon value", "policy", "GBP/t CO2e",
        central_carbon, 0.0, 2.0 * central_carbon, 0.0, 4.0 * max(central_carbon, 1.0),
        apply_carbon_value,
        "Changes the carbon-value coefficient in the single cost + carbon-value objective.",
        "Central government-policy value; range includes zero and strong future carbon values.",
        display_decimals=1,
    ))

    central_capex = float(next(iter(central_data.treatment_fixed_cost_gbp.values())))

    def apply_wwt_capex(state: ScenarioState, value: float) -> None:
        state.data.treatment_fixed_cost_gbp = {site: float(value) for site in state.data.sites}
        state.effective_value = value

    add(SensitivityParameter(
        "wastewater_treatment_annualised_capex", "Wastewater-treatment annualised CAPEX", "water", "GBP/site-year",
        central_capex, 0.50 * central_capex, 2.0 * central_capex, 0.0, 4.0 * max(central_capex, 1.0),
        apply_wwt_capex,
        "Changes annualised fixed treatment-facility CAPEX at every active manufacturing site.",
        "Current small-package treatment-facility estimate; wide capital-cost uncertainty.",
        display_decimals=1,
    ))

    central_opex = float(next(iter(central_data.treatment_opex_gbp_per_m3.values())))

    def apply_wwt_opex(state: ScenarioState, value: float) -> None:
        state.data.treatment_opex_gbp_per_m3 = {site: float(value) for site in state.data.sites}
        state.effective_value = value

    add(SensitivityParameter(
        "wastewater_treatment_opex", "Wastewater-treatment OPEX", "water", "GBP/m3 treated",
        central_opex, 0.50 * central_opex, 2.0 * central_opex, 0.0, 5.0 * max(central_opex, 1.0),
        apply_wwt_opex,
        "Changes variable treatment cost for all wastewater, regardless of discharge or reuse destination.",
        "Blended central OPEX assumption; broad uncertainty range.",
    ))

    def infer_scope(spec: SensitivityParameter) -> None:
        if spec.scope_level != "system" or spec.target:
            return
        pid = spec.parameter_id
        for polymer in central_data.polymers:
            if pid.endswith(f"__{polymer}"):
                spec.scope_level = "polymer"
                spec.target = polymer
                if pid.startswith("recovery_fraction__"):
                    spec.linked_quantity_note = "rejection fraction = 1 - recovery fraction"
                return
        for route in central_data.routes:
            if pid.endswith(f"__{route}"):
                spec.scope_level = "route"
                spec.target = route
                return
        for mode in central_data.transport_modes:
            if pid.endswith(f"__{mode}"):
                spec.scope_level = "vehicle_mode"
                spec.target = mode
                return
        if pid.endswith("__all") or pid in {
            "total_plastic_waste_per_batch",
            "autoclave_energy_demand",
            "recycling_gate_fee",
            "recycling_transport_distance_multiplier",
        }:
            spec.scope_level = "all_plastics"
            spec.target = "plastic system"

    # Continuous parameters are analysed as numerical decision boundaries.
    # Binary route-availability and pharmaceutical-acceptance assumptions are
    # retained as two-point discrete stress tests in the same auditable output.
    final_specs: List[SensitivityParameter] = []
    seen_ids: set[str] = set()
    for spec in specs:
        require(spec.parameter_id not in seen_ids,
                f"Duplicate sensitivity parameter ID: {spec.parameter_id}")
        seen_ids.add(spec.parameter_id)
        infer_scope(spec)
        spec = apply_config_definition(spec, config)
        final_specs.append(spec)
    return final_specs


# -----------------------------------------------------------------------------
# Solving and decision signatures
# -----------------------------------------------------------------------------

@dataclass
class ScenarioRecord:
    parameter_id: str
    parameter_label: str
    group: str
    requested_value: float
    effective_value: float
    unit: str
    success: bool
    status: str
    solve_seconds: float
    model: Any
    result: Any
    state: ScenarioState
    metrics: Dict[str, float]
    signature_hash: str
    signature_json: str
    decision_snapshot: str
    validation: Dict[str, float]
    error: str = ""

    def row(self, central_metrics: Mapping[str, float]) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "parameter_id": self.parameter_id,
            "parameter_label": self.parameter_label,
            "group": self.group,
            "requested_value": self.requested_value,
            "effective_value": self.effective_value,
            "unit": self.unit,
            "success": self.success,
            "status": self.status,
            "solve_seconds": self.solve_seconds,
            "decision_signature": self.signature_hash,
            "decision_snapshot": self.decision_snapshot,
            "error": self.error,
        }
        for metric in OUTPUT_METRICS:
            value = self.metrics.get(metric, np.nan)
            central = central_metrics.get(metric, np.nan)
            row[metric] = value
            row[f"{metric}_change"] = value - central if math.isfinite(value) and math.isfinite(central) else np.nan
            row[f"{metric}_change_pct"] = (
                100.0 * (value - central) / abs(central)
                if math.isfinite(value) and math.isfinite(central) and abs(central) > 1e-12
                else np.nan
            )
        row.update(self.validation)
        return row


class ScenarioSolver:
    def __init__(
        self,
        master: Any,
        central_data: Any,
        impact: ImpactModel,
        result_dir: Path,
        backend: str,
        objective: str,
        tee: bool,
    ) -> None:
        self.master = master
        self.central_data = central_data
        self.impact = impact
        self.result_dir = result_dir
        self.backend = backend
        self.objective = objective
        self.cache: Dict[Tuple[str, float], ScenarioRecord] = {}
        self.scipy_report_rows: List[Dict[str, Any]] = []
        self.progress_started = time.perf_counter()
        self.planned_solve_estimate = 0
        self.initial_planned_solve_estimate = 0
        self.parameter_planning_estimates: Dict[str, int] = {}
        self.completed_model_solves = 0
        self.solve_durations: List[float] = []
        self.current_parameter = "initialisation"

        # Redirect all master GAMS logs/reports away from the core results folder.
        master.RESULT_DIR = result_dir
        master.GAMS_LOG_DIR = result_dir / "gams_logs"
        master.GAMS_WORK_DIR = result_dir / "gams_work"
        master.GAMS_SOLVE_REPORT = result_dir / "gams_solve_report.csv"
        master.GAMS_PREFLIGHT_LOG = master.GAMS_LOG_DIR / "0000_gams_preflight.log"
        master.GAMS_LOG_DIR.mkdir(parents=True, exist_ok=True)
        master.GAMS_WORK_DIR.mkdir(parents=True, exist_ok=True)
        master.GAMS_TEE = bool(tee)
        master._GAMS_SOLVE_COUNTER = 0
        master._GAMS_PREFLIGHT_DONE = False

    def configure_progress(self, solve_plan: Mapping[str, Any]) -> None:
        estimate = int(solve_plan.get("estimated_model_solves", 1))
        self.planned_solve_estimate = max(estimate, 1)
        self.initial_planned_solve_estimate = self.planned_solve_estimate
        self.parameter_planning_estimates = {
            str(row["parameter_id"]): int(row.get("estimated_total", 0))
            for row in solve_plan.get("parameter_breakdown", [])
        }
        self.progress_started = time.perf_counter()

    def reforecast_for_remaining_parameters(self, parameter_ids: Sequence[str]) -> None:
        """Reforecast after each completed parameter using actual solves completed so far."""
        remaining = sum(self.parameter_planning_estimates.get(str(pid), 0) for pid in parameter_ids)
        self.planned_solve_estimate = max(self.completed_model_solves + remaining, self.completed_model_solves)

    def set_current_parameter(self, parameter_id: str) -> None:
        self.current_parameter = str(parameter_id)

    def report_completed_solve(self, solve_seconds: float, success: bool) -> None:
        self.completed_model_solves += 1
        if math.isfinite(solve_seconds) and solve_seconds >= 0:
            self.solve_durations.append(float(solve_seconds))
        elapsed = time.perf_counter() - self.progress_started
        average = float(np.mean(self.solve_durations)) if self.solve_durations else np.nan
        remaining = max(self.planned_solve_estimate - self.completed_model_solves, 0)
        eta = average * remaining if math.isfinite(average) else np.nan
        denominator = max(self.planned_solve_estimate, self.completed_model_solves, 1)
        percent = min(100.0, 100.0 * self.completed_model_solves / denominator)
        print(
            f"    solve {self.completed_model_solves:,} / ~{self.planned_solve_estimate:,} "
            f"({percent:5.1f}%) | latest {solve_seconds:6.2f}s | "
            f"mean {average:6.2f}s | elapsed {format_duration(elapsed)} | "
            f"ETA {format_duration(eta)} | finish ~{estimated_finish_text(eta)} | "
            f"{'OK' if success else 'FAILED'} | {self.current_parameter}",
            flush=True,
        )

    def solve_backend(self, model: Any, case_id: str, metadata: Mapping[str, Any]) -> Any:
        if self.backend == "gams":
            return self.master.solve_model(
                model,
                case_id=case_id,
                primary=self.objective,
                selection_method="sensitivity analysis",
                metadata=dict(metadata),
            )
        if self.backend != "scipy":
            raise ValueError(f"Unknown backend: {self.backend}")

        from scipy.optimize import Bounds, LinearConstraint, milp

        c, A, b_l, b_u = self.master._assemble_problem(model, self.objective)
        started = time.perf_counter()
        try:
            optimisation = milp(
                c=c,
                integrality=model.integrality,
                bounds=Bounds(model.lower, model.upper),
                constraints=LinearConstraint(A.tocsr(), b_l, b_u),
                options={"mip_rel_gap": float(getattr(self.master, "MIP_REL_GAP", 1e-9))},
            )
            elapsed = time.perf_counter() - started
            success = bool(optimisation.success and optimisation.x is not None)
            x = np.array(optimisation.x, dtype=float) if success else np.full(len(model.var_names), np.nan)
            metrics = {
                name: expression.evaluate(x) for name, expression in model.expressions.items()
            } if success else {}
            self.scipy_report_rows.append({
                "case_id": case_id,
                "success": success,
                "status": int(optimisation.status),
                "message": str(optimisation.message),
                "elapsed_seconds": elapsed,
                "objective_without_constant": finite(optimisation.fun),
                "mip_gap": finite(getattr(optimisation, "mip_gap", np.nan)),
                "mip_node_count": finite(getattr(optimisation, "mip_node_count", np.nan)),
            })
            return self.master.SolveResult(
                case_id=case_id,
                success=success,
                status=int(optimisation.status),
                message=str(optimisation.message),
                x=x,
                metrics=metrics,
                raw_objective=finite(optimisation.fun),
                primary_objective=self.objective,
                selection_method="scipy.optimize.milp sensitivity analysis",
                metadata={**dict(metadata), "solver_backend": "scipy_highs_milp"},
            )
        except Exception:
            elapsed = time.perf_counter() - started
            self.scipy_report_rows.append({
                "case_id": case_id,
                "success": False,
                "status": -1,
                "message": traceback.format_exc(),
                "elapsed_seconds": elapsed,
            })
            raise

    def solve_parameter(
        self,
        spec: SensitivityParameter,
        value: float,
        central_model: Optional[Any] = None,
        central_result: Optional[Any] = None,
    ) -> ScenarioRecord:
        key = (spec.parameter_id, round(float(value), 12))
        if key in self.cache:
            return self.cache[key]

        state = clone_state(self.central_data, self.impact)
        case_id = f"SA_{safe_token(spec.parameter_id)}_{safe_token(f'{value:.10g}')}"
        started = time.perf_counter()
        performed_model_solve = False
        try:
            spec.apply(state, float(value))
            sync_gwp_coefficients(state, self.impact)
            # Parameters that affect only fossil-resource or water-scarcity reporting
            # can reuse the central optimisation decision exactly.
            if not spec.affects_optimisation and central_model is not None and central_result is not None:
                model = central_model
                result = central_result
                solve_seconds = 0.0
                status = "central decision reused; parameter outside objective"
            else:
                model = self.master.build_linear_model(state.data)
                performed_model_solve = True
                result = self.solve_backend(model, case_id, {
                    "parameter_id": spec.parameter_id,
                    "requested_value": value,
                    "effective_value": state.effective_value,
                    "unit": spec.unit,
                })
                solve_seconds = time.perf_counter() - started
                status = result.message

            if not result.success:
                record = ScenarioRecord(
                    parameter_id=spec.parameter_id,
                    parameter_label=spec.label,
                    group=spec.group,
                    requested_value=float(value),
                    effective_value=float(state.effective_value if state.effective_value is not None else value),
                    unit=spec.unit,
                    success=False,
                    status=status,
                    solve_seconds=solve_seconds,
                    model=model,
                    result=result,
                    state=state,
                    metrics={metric: np.nan for metric in OUTPUT_METRICS},
                    signature_hash="INFEASIBLE",
                    signature_json="{}",
                    decision_snapshot="",
                    validation={},
                    error=status,
                )
            else:
                metrics = evaluate_extended_impacts(self.master, model, result, state, self.impact)
                signature, snapshot = decision_signature(self.master, model, result)
                validation = self.master.solution_validation(model, result)
                record = ScenarioRecord(
                    parameter_id=spec.parameter_id,
                    parameter_label=spec.label,
                    group=spec.group,
                    requested_value=float(value),
                    effective_value=float(state.effective_value if state.effective_value is not None else value),
                    unit=spec.unit,
                    success=True,
                    status=status,
                    solve_seconds=solve_seconds,
                    model=model,
                    result=result,
                    state=state,
                    metrics=metrics,
                    signature_hash=signature["hash"],
                    signature_json=signature["json"],
                    decision_snapshot=snapshot,
                    validation=validation,
                )
        except Exception as exc:
            record = ScenarioRecord(
                parameter_id=spec.parameter_id,
                parameter_label=spec.label,
                group=spec.group,
                requested_value=float(value),
                effective_value=float(state.effective_value if state.effective_value is not None else value),
                unit=spec.unit,
                success=False,
                status="exception",
                solve_seconds=time.perf_counter() - started,
                model=None,
                result=None,
                state=state,
                metrics={metric: np.nan for metric in OUTPUT_METRICS},
                signature_hash="ERROR",
                signature_json="{}",
                decision_snapshot="",
                validation={},
                error="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            )
        self.cache[key] = record
        if performed_model_solve:
            self.report_completed_solve(record.solve_seconds, record.success)
        return record

    def write_solver_reports(self) -> None:
        if self.scipy_report_rows:
            atomic_to_csv(
                pd.DataFrame(self.scipy_report_rows),
                self.result_dir / "scipy_milp_solve_report.csv",
            )


def rounded_share(value: float) -> float:
    if abs(value) < 5e-5:
        return 0.0
    if abs(value - 1.0) < 5e-5:
        return 1.0
    return round(float(value), 4)


def _continuous_choice_state(share: float, zero_label: str, full_label: str, partial_label: str) -> str:
    share = float(share)
    if share <= DECISION_ACTIVE_SHARE_TOL:
        return zero_label
    if share >= DECISION_FULL_SHARE_TOL:
        return full_label
    return partial_label


def _route_allocation_state(group: pd.DataFrame) -> str:
    """Summarise network-wide routing while deliberately ignoring site and HGV mode."""
    total = float(group["tonnes_per_year"].sum())
    if total <= 1e-12:
        return "no generated flow"
    entries = []
    for row in group.groupby("route", as_index=False)["tonnes_per_year"].sum().itertuples(index=False):
        share = float(row.tonnes_per_year / total)
        entries.append((share, str(row.route)))
    entries.sort(reverse=True)
    material = [(share, route) for share, route in entries if share >= DECISION_ACTIVE_SHARE_TOL]
    if not material:
        material = [entries[0]]
    labels = [route for _, route in material]
    if len(material) == 1 and material[0][0] >= DECISION_FULL_SHARE_TOL:
        return "exclusive:" + labels[0]
    return "split:" + ";".join(sorted(labels))


def _transport_mode_state(group: pd.DataFrame) -> str:
    """Summarise the HGV choice for a site/route combination."""
    total = float(group["tonnes_per_year"].sum())
    if total <= 1e-12:
        return "no transported flow"
    entries = []
    for row in group.groupby("transport_mode", as_index=False)["tonnes_per_year"].sum().itertuples(index=False):
        share = float(row.tonnes_per_year / total)
        entries.append((share, str(row.transport_mode)))
    entries.sort(reverse=True)
    material = [(share, mode) for share, mode in entries if share >= DECISION_ACTIVE_SHARE_TOL]
    if not material:
        material = [entries[0]]
    labels = [mode for _, mode in material]
    if len(material) == 1 and material[0][0] >= DECISION_FULL_SHARE_TOL:
        return "exclusive_mode:" + labels[0]
    return "mixed_modes:" + ";".join(sorted(labels))


def _electric_hgv_share(group: pd.DataFrame) -> float:
    total = float(group["tonnes_per_year"].sum())
    if total <= 1e-12:
        return 0.0
    electric = float(group.loc[
        group["transport_mode"].astype(str).eq("battery_electric_articulated_hgv"),
        "tonnes_per_year",
    ].sum())
    return electric / total


def _route_share_components(
    components: Dict[str, Any],
    group: pd.DataFrame,
    prefix: str,
) -> None:
    total = float(group["tonnes_per_year"].sum())
    components[f"{prefix} | allocation state"] = _route_allocation_state(group)
    if total <= 1e-12:
        return
    route_totals = group.groupby("route", as_index=False)["tonnes_per_year"].sum()
    for row in route_totals.itertuples(index=False):
        label = ROUTE_LABELS.get(str(row.route), str(row.route))
        components[f"{prefix} | {label} share"] = float(row.tonnes_per_year / total)


def decision_signature(master: Any, model: Any, result: Any) -> Tuple[Dict[str, str], str]:
    """
    Build the structural decision signature used to discover boundaries.

    Electricity and plastic routing are intentionally represented at network
    level. A mere redistribution of the same plastic route or electricity
    source between sites therefore does not create a false decision boundary.
    Site detail is retained for decisions where geography is materially part of
    the choice: the low-batch node, water reuse and HGV mode by destination.
    """
    routes, sites, mix = master.solution_decisions(model, result)
    payload: Dict[str, Any] = {}

    if not sites.empty:
        low_rows = sites.loc[sites["is_low_batch_location"]]
        payload["low_batch_location"] = str(low_rows.iloc[0]["location"]) if len(low_rows) else ""

        wind = float(sites["wind_kwh"].sum())
        grid = float(sites["grid_kwh"].sum())
        wind_share = wind / (wind + grid) if wind + grid > 1e-9 else 0.0
        payload["network_electricity_state"] = _continuous_choice_state(
            wind_share, "grid only", "wind only", "grid/wind mix"
        )

        reused = float(sites["water_reused_m3"].sum())
        fresh = float(sites["fresh_utility_water_m3"].sum())
        reuse_share = reused / (reused + fresh) if reused + fresh > 1e-9 else 0.0
        payload["network_water_reuse_state"] = _continuous_choice_state(
            reuse_share, "no reuse", "full reuse", "partial reuse"
        )
        payload["water_reuse_state_by_location"] = [
            (
                str(row.location),
                _continuous_choice_state(
                    float(row.water_reused_m3 / (row.water_reused_m3 + row.fresh_utility_water_m3))
                    if float(row.water_reused_m3 + row.fresh_utility_water_m3) > 1e-9 else 0.0,
                    "no reuse", "full reuse", "partial reuse",
                ),
            )
            for row in sites.sort_values("location").itertuples(index=False)
        ]

    if not routes.empty:
        payload["polymer_route_state_network"] = [
            (str(polymer), _route_allocation_state(group))
            for polymer, group in routes.groupby("polymer", sort=True)
        ]
        # Treat diesel-only transport as the baseline rather than a structural
        # site/route decision. This prevents a plastic flow moving between sites
        # (while remaining diesel) from creating a false HGV boundary. Site and
        # route detail enters the signature only when electric HGV use is active.
        hgv_states = []
        for (location, route), group in routes.groupby(["location", "route"], sort=True):
            if float(group["tonnes_per_year"].sum()) <= 1e-12:
                continue
            electric_share = _electric_hgv_share(group)
            if electric_share <= DECISION_ACTIVE_SHARE_TOL:
                continue
            hgv_states.append((str(location), str(route), _transport_mode_state(group)))
        payload["electric_hgv_state_by_location_route"] = hgv_states

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]

    if sites.empty:
        snapshot = "No site decisions"
    else:
        wind = float(sites["wind_kwh"].sum())
        grid = float(sites["grid_kwh"].sum())
        wind_share = wind / (wind + grid) if wind + grid > 0 else 0.0
        reused = float(sites["water_reused_m3"].sum())
        fresh = float(sites["fresh_utility_water_m3"].sum())
        reuse_share = reused / (reused + fresh) if reused + fresh > 0 else 0.0
        low = payload.get("low_batch_location", "")
        polymer_decisions = []
        if not routes.empty:
            for polymer, group in routes.groupby("polymer", sort=True):
                state = _route_allocation_state(group)
                polymer_decisions.append(f"{polymer}: {state}")
        snapshot = (
            f"Low-batch node {low}; network electricity {100*wind_share:.1f}% renewable; "
            f"network eligible water {100*reuse_share:.1f}% reused; "
            + "; ".join(polymer_decisions)
        )
    return {"hash": digest, "json": canonical}, snapshot


def decision_components(master: Any, model: Any, result: Any) -> Dict[str, Any]:
    """Return decisions with network aggregation applied consistently."""
    routes, sites, mix = master.solution_decisions(model, result)
    components: Dict[str, Any] = {}

    if not sites.empty:
        low = sites.loc[sites["is_low_batch_location"]]
        components["System | low-batch location"] = str(low.iloc[0]["location"]) if len(low) else ""

        wind = float(sites["wind_kwh"].sum())
        grid = float(sites["grid_kwh"].sum())
        wind_share = wind / (wind + grid) if wind + grid > 0 else 0.0
        components["Electricity | network | renewable share"] = wind_share
        components["Electricity | network | supply state"] = _continuous_choice_state(
            wind_share, "grid only", "wind only", "grid/wind mix"
        )

        reused = float(sites["water_reused_m3"].sum())
        fresh = float(sites["fresh_utility_water_m3"].sum())
        reuse_share = reused / (reused + fresh) if reused + fresh > 0 else 0.0
        components["Water | network | reuse share"] = reuse_share
        components["Water | network | strategy"] = _continuous_choice_state(
            reuse_share, "no reuse", "full reuse", "partial reuse"
        )
        for row in sites.sort_values("location").itertuples(index=False):
            eligible = float(row.water_reused_m3 + row.fresh_utility_water_m3)
            site_share = float(row.water_reused_m3 / eligible) if eligible > 1e-9 else 0.0
            components[f"Water | {row.location} | reuse share"] = site_share
            components[f"Water | {row.location} | strategy"] = _continuous_choice_state(
                site_share, "no reuse", "full reuse", "partial reuse"
            )

    if not routes.empty:
        _route_share_components(components, routes, "Plastics | all polymers")
        for polymer, group in routes.groupby("polymer", sort=True):
            _route_share_components(components, group, f"Plastics | {polymer}")

        components["HGV | network | electric share"] = _electric_hgv_share(routes)
        for (location, route), group in routes.groupby(["location", "route"], sort=True):
            if float(group["tonnes_per_year"].sum()) <= 1e-12:
                continue
            route_label = ROUTE_LABELS.get(str(route), str(route))
            prefix = f"HGV | {location} | {route_label}"
            components[f"{prefix} | electric share"] = _electric_hgv_share(group)
            components[f"{prefix} | mode state"] = _transport_mode_state(group)

    return components


def component_metadata(component: str) -> Dict[str, str]:
    """Classify decision components for downstream tabular analysis."""
    parts = [part.strip() for part in str(component).split("|")]
    family = parts[0].lower() if parts else "other"
    scope = "network"
    target = ""
    location = ""
    if family == "plastics":
        target = parts[1] if len(parts) > 1 else "all polymers"
        scope = "all_plastics" if target == "all polymers" else "polymer"
    elif family == "hgv":
        location = parts[1] if len(parts) > 1 else "network"
        target = parts[2] if len(parts) > 2 else "all routes"
        scope = "site_route" if location != "network" else "network"
    elif family == "water":
        location = parts[1] if len(parts) > 2 else "network"
        scope = "site" if location != "network" else "network"
        target = location
    elif family == "electricity":
        target = "network electricity"
    elif family == "system":
        target = "site allocation"
    elif component == "Model feasible":
        family, scope, target = "feasibility", "system", "model feasibility"
    return {
        "component_family": family,
        "component_scope": scope,
        "component_target": target,
        "component_location": location,
    }

def compare_decisions(master: Any, before: ScenarioRecord, after: ScenarioRecord) -> pd.DataFrame:
    if not before.success or not after.success:
        return pd.DataFrame()
    a = decision_components(master, before.model, before.result)
    b = decision_components(master, after.model, after.result)
    rows = []
    for key in sorted(set(a) | set(b)):
        a_value = a.get(key)
        b_value = b.get(key)
        is_text = isinstance(a_value, str) or isinstance(b_value, str)
        old = a_value if a_value is not None else ("" if is_text else 0.0)
        new = b_value if b_value is not None else ("" if is_text else 0.0)
        changed = old != new if is_text else abs(float(old) - float(new)) > 1e-4
        if not changed:
            continue
        rows.append({
            "decision_component": key,
            "before": old,
            "after": new,
            "absolute_change": (float(new) - float(old)) if not is_text else np.nan,
        })
    return pd.DataFrame(rows)



# -----------------------------------------------------------------------------
# Local decision-boundary analysis
# -----------------------------------------------------------------------------

@dataclass
class BoundaryResult:
    parameter_id: str
    parameter_label: str
    group: str
    unit: str
    direction: str
    boundary_estimate: float
    lower_bracket: float
    upper_bracket: float
    central_value: float
    deviation_from_central_pct: float
    before_signature: str
    after_signature: str
    before_snapshot: str
    after_snapshot: str
    boundary_status: str
    before_record: ScenarioRecord
    after_record: ScenarioRecord

    def row(self) -> Dict[str, Any]:
        return {
            "parameter_id": self.parameter_id,
            "parameter_label": self.parameter_label,
            "group": self.group,
            "unit": self.unit,
            "direction": self.direction,
            "boundary_estimate": self.boundary_estimate,
            "lower_bracket": self.lower_bracket,
            "upper_bracket": self.upper_bracket,
            "boundary_bracket_width": self.upper_bracket - self.lower_bracket,
            "central_value": self.central_value,
            "deviation_from_central_pct": self.deviation_from_central_pct,
            "before_signature": self.before_signature,
            "after_signature": self.after_signature,
            "before_decisions": self.before_snapshot,
            "after_decisions": self.after_snapshot,
            "boundary_status": self.boundary_status,
        }


class LocalBoundaryAnalysis:
    def __init__(
        self,
        master: Any,
        central_data: Any,
        impact: ImpactModel,
        specs: Sequence[SensitivityParameter],
        solver: ScenarioSolver,
        boundary_points: int,
        decision_curve_points: int,
        bisection_iterations: int,
    ) -> None:
        self.master = master
        self.central_data = central_data
        self.impact = impact
        self.specs = list(specs)
        self.solver = solver
        self.boundary_points = int(boundary_points)
        self.decision_curve_points = int(decision_curve_points)
        self.bisection_iterations = int(bisection_iterations)
        require(self.boundary_points >= 5, "BOUNDARY_DISCOVERY_POINTS must be at least 5.")
        require(self.decision_curve_points >= 9, "DECISION_CURVE_POINTS must be at least 9.")
        require(self.bisection_iterations >= 1, "BOUNDARY_BISECTION_ITERATIONS must be positive.")
        self.central_model: Any = None
        self.central_result: Any = None
        self.central_record: Optional[ScenarioRecord] = None
        self.boundary_scan_records: Dict[str, List[ScenarioRecord]] = {}
        self.decision_curve_records: Dict[str, List[ScenarioRecord]] = {}
        self.boundaries: List[BoundaryResult] = []
        self.decision_change_rows: List[pd.DataFrame] = []
        self.completed_parameter_ids: List[str] = []
        self.run_complete = False
        self.checkpoint_callback: Optional[Callable[[str], None]] = None

    def solve_central(self) -> ScenarioRecord:
        dummy = SensitivityParameter(
            parameter_id="CENTRAL", label="Central case", group="central", unit="not applicable",
            central=1.0, low=1.0, high=1.0, boundary_low=1.0, boundary_high=1.0,
            apply=lambda state, value: setattr(state, "effective_value", 1.0),
            description="Unchanged central case", source_or_basis="Current master MILP inputs",
        )
        record = self.solver.solve_parameter(dummy, 1.0)
        require(record.success, f"Central solve failed:\n{record.error}")
        self.central_model, self.central_result, self.central_record = record.model, record.result, record
        return record

    def get_record(self, spec: SensitivityParameter, value: float) -> ScenarioRecord:
        return self.solver.solve_parameter(
            spec, value, central_model=self.central_model, central_result=self.central_result
        )

    def run_boundary_scan(self, spec: SensitivityParameter) -> List[ScenarioRecord]:
        values = spec.discovery_grid(self.boundary_points)
        records = [self.get_record(spec, value) for value in values]
        records.sort(key=lambda record: record.requested_value)
        self.boundary_scan_records[spec.parameter_id] = records
        return records

    def refine_decision_boundary(
        self, spec: SensitivityParameter, left: ScenarioRecord, right: ScenarioRecord
    ) -> BoundaryResult:
        require(left.success and right.success, "Decision-boundary endpoints must be feasible.")
        require(left.signature_hash != right.signature_hash, "Boundary endpoints must have different decisions.")
        low_record, high_record = (left, right) if left.requested_value <= right.requested_value else (right, left)
        low_value, high_value = low_record.requested_value, high_record.requested_value
        reference_signature = low_record.signature_hash
        for _ in range(self.bisection_iterations):
            tolerance = BOUNDARY_ABSOLUTE_TOLERANCE + BOUNDARY_RELATIVE_TOLERANCE * max(
                abs(spec.central), abs(low_value), abs(high_value), 1.0
            )
            if high_value - low_value <= tolerance:
                break
            midpoint = 0.5 * (low_value + high_value)
            middle = self.get_record(spec, midpoint)
            if not middle.success or middle.signature_hash != reference_signature:
                high_value, high_record = midpoint, middle
            else:
                low_value, low_record = midpoint, middle
        return self._make_boundary(spec, low_value, high_value, low_record, high_record,
                                   "decision transition bracketed")

    def refine_feasibility_boundary(
        self, spec: SensitivityParameter, left: ScenarioRecord, right: ScenarioRecord
    ) -> BoundaryResult:
        low_value, high_value = sorted([left.requested_value, right.requested_value])
        low_record = left if left.requested_value == low_value else right
        high_record = right if right.requested_value == high_value else left
        feasibility_at_low = low_record.success
        for _ in range(self.bisection_iterations):
            tolerance = BOUNDARY_ABSOLUTE_TOLERANCE + BOUNDARY_RELATIVE_TOLERANCE * max(
                abs(spec.central), abs(low_value), abs(high_value), 1.0
            )
            if high_value - low_value <= tolerance:
                break
            midpoint = 0.5 * (low_value + high_value)
            middle = self.get_record(spec, midpoint)
            if middle.success == feasibility_at_low:
                low_value, low_record = midpoint, middle
            else:
                high_value, high_record = midpoint, middle
        return self._make_boundary(spec, low_value, high_value, low_record, high_record,
                                   "feasibility boundary bracketed")

    @staticmethod
    def _make_boundary(
        spec: SensitivityParameter,
        low_value: float,
        high_value: float,
        low_record: ScenarioRecord,
        high_record: ScenarioRecord,
        status: str,
    ) -> BoundaryResult:
        estimate = 0.5 * (low_value + high_value)
        deviation = (
            100.0 * (estimate - spec.central) / abs(spec.central)
            if abs(spec.central) > 1e-12 else np.nan
        )
        return BoundaryResult(
            parameter_id=spec.parameter_id,
            parameter_label=spec.label,
            group=spec.group,
            unit=spec.unit,
            direction="below central" if estimate < spec.central else "above central",
            boundary_estimate=estimate,
            lower_bracket=min(low_value, high_value),
            upper_bracket=max(low_value, high_value),
            central_value=spec.central,
            deviation_from_central_pct=deviation,
            before_signature=low_record.signature_hash,
            after_signature=high_record.signature_hash,
            before_snapshot=low_record.decision_snapshot if low_record.success else low_record.status,
            after_snapshot=high_record.decision_snapshot if high_record.success else high_record.status,
            boundary_status=status,
            before_record=low_record,
            after_record=high_record,
        )

    def find_boundaries(self, spec: SensitivityParameter) -> List[BoundaryResult]:
        records = self.boundary_scan_records.get(spec.parameter_id) or self.run_boundary_scan(spec)
        if spec.parameter_type == "binary":
            feasible = [record for record in records if record.success]
            if len(feasible) >= 2 and feasible[0].signature_hash != feasible[-1].signature_hash:
                boundary = self._make_boundary(spec, feasible[0].requested_value, feasible[-1].requested_value,
                                               feasible[0], feasible[-1], "binary stress-test decision change")
                boundary.boundary_estimate = np.nan
                boundary.direction = "discrete switch"
                boundary.deviation_from_central_pct = np.nan
                return [boundary]
            return []

        found: List[BoundaryResult] = []
        for left, right in zip(records[:-1], records[1:]):
            if left.success != right.success:
                boundary = self.refine_feasibility_boundary(spec, left, right)
            elif left.success and right.success and left.signature_hash != right.signature_hash:
                boundary = self.refine_decision_boundary(spec, left, right)
            else:
                continue
            duplicate = any(
                math.isfinite(existing.boundary_estimate)
                and math.isfinite(boundary.boundary_estimate)
                and abs(existing.boundary_estimate - boundary.boundary_estimate)
                    <= max(BOUNDARY_ABSOLUTE_TOLERANCE, 1e-6 * max(abs(spec.central), 1.0))
                for existing in found
            )
            if not duplicate:
                found.append(boundary)
        return found

    def run_decision_curve(
        self, spec: SensitivityParameter, boundaries: Sequence[BoundaryResult]
    ) -> List[ScenarioRecord]:
        if spec.parameter_type == "binary":
            records = self.boundary_scan_records.get(spec.parameter_id, [])
            self.decision_curve_records[spec.parameter_id] = records
            return records
        finite_boundaries = [item for item in boundaries if math.isfinite(item.boundary_estimate)]
        anchors = [spec.low, spec.central, spec.high]
        for item in finite_boundaries:
            anchors.extend([item.lower_bracket, item.boundary_estimate, item.upper_bracket])
        lo, hi = max(spec.boundary_low, min(anchors)), min(spec.boundary_high, max(anchors))
        if hi > lo:
            margin = 0.08 * (hi - lo)
            lo, hi = max(spec.boundary_low, lo - margin), min(spec.boundary_high, hi + margin)
        values = unique_sorted(list(np.linspace(lo, hi, self.decision_curve_points)) + anchors)
        records = [self.get_record(spec, value) for value in values]
        records.sort(key=lambda record: record.requested_value)
        self.decision_curve_records[spec.parameter_id] = records
        return records

    def run(self) -> None:
        self.solver.set_current_parameter("CENTRAL")
        central = self.solve_central()
        print("\nCENTRAL DECISIONS", flush=True)
        print(central.decision_snapshot, flush=True)
        if self.checkpoint_callback is not None:
            self.checkpoint_callback("central complete")

        parameter_started = time.perf_counter()
        for index, spec in enumerate(self.specs, start=1):
            self.solver.set_current_parameter(spec.parameter_id)
            print(
                f"\n[{index}/{len(self.specs)}] {spec.parameter_id}: {spec.label}",
                flush=True,
            )
            scan = self.run_boundary_scan(spec)
            regimes = {record.signature_hash for record in scan if record.success}
            print(
                f"  discovery points={len(scan)}, feasible regimes={len(regimes)}",
                flush=True,
            )
            boundaries = self.find_boundaries(spec)
            self.boundaries.extend(boundaries)
            if boundaries:
                print(f"  refined boundaries={len(boundaries)}; generating dense curve", flush=True)
                self.run_decision_curve(spec, boundaries)
                for number, boundary in enumerate(boundaries, start=1):
                    changes = compare_decisions(self.master, boundary.before_record, boundary.after_record)
                    if not changes.empty:
                        changes.insert(0, "boundary_number", number)
                        changes.insert(0, "parameter_id", spec.parameter_id)
                        changes.insert(2, "boundary_estimate", boundary.boundary_estimate)
                        changes.insert(3, "unit", spec.unit)
                        self.decision_change_rows.append(changes)
            else:
                print("  no decision boundary in configured search range", flush=True)

            self.completed_parameter_ids.append(spec.parameter_id)
            remaining_ids = [item.parameter_id for item in self.specs[index:]]
            self.solver.reforecast_for_remaining_parameters(remaining_ids)
            if self.checkpoint_callback is not None:
                self.checkpoint_callback(f"parameter {index}/{len(self.specs)} complete")

            elapsed = time.perf_counter() - parameter_started
            average_parameter = elapsed / index
            remaining_parameters = len(self.specs) - index
            parameter_eta = average_parameter * remaining_parameters
            print(
                f"  parameter progress {index}/{len(self.specs)} "
                f"({100.0*index/len(self.specs):.1f}%) | "
                f"elapsed {format_duration(elapsed)} | "
                f"parameter-level ETA {format_duration(parameter_eta)} | "
                f"finish ~{estimated_finish_text(parameter_eta)}",
                flush=True,
            )

        self.run_complete = True
        self.solver.write_solver_reports()
        if self.checkpoint_callback is not None:
            self.checkpoint_callback("analysis complete")

    def boundary_table(self) -> pd.DataFrame:
        columns = [
            "parameter_id", "parameter_label", "group", "unit", "direction",
            "boundary_estimate", "lower_bracket", "upper_bracket",
            "boundary_bracket_width", "central_value", "deviation_from_central_pct",
            "before_signature", "after_signature", "before_decisions",
            "after_decisions", "boundary_status",
        ]
        return pd.DataFrame([item.row() for item in self.boundaries], columns=columns)

    def decision_change_table(self) -> pd.DataFrame:
        if not self.decision_change_rows:
            return pd.DataFrame(columns=[
                "parameter_id", "boundary_number", "boundary_estimate", "unit",
                "decision_component", "before", "after", "absolute_change",
            ])
        return pd.concat(self.decision_change_rows, ignore_index=True, sort=False)


def build_runtime_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return an isolated runtime copy of the OAT parameter configuration."""
    return copy.deepcopy(dict(config))


def parameter_coverage_table(
    specs: Sequence[SensitivityParameter],
    selected_ids: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Audit coverage of the requested decision-boundary parameter families."""
    ids = [spec.parameter_id for spec in specs]
    selected = set(selected_ids or ids)

    def matching(*patterns: str) -> List[str]:
        found: List[str] = []
        for pid in ids:
            if any(
                pid == pattern
                or (pattern.endswith("*") and pid.startswith(pattern[:-1]))
                for pattern in patterns
            ):
                found.append(pid)
        return found

    requirements = [
        ("Renewable cPPA price premium or discount", ("renewable_cppa_premium",),
         "Continuous electricity-price boundary."),
        ("Grid and renewable electricity LCIA factors", ("electricity_lcia_multiplier__*",),
         "GWP factors affect the current objective; other LCIA factors are outcome-only."),
        ("Electricity demand per manufacturing batch", ("electricity_demand_per_batch",),
         "Per-batch process electricity intensity."),
        ("Mechanical-recycling recovery and rejection rates",
         ("mechanical_recycling_recovery_multiplier__all", "recovery_fraction__*"),
         "Rejection is constrained as exactly 1 - recovery."),
        ("Recyclate market values by polymer",
         ("recyclate_market_value_multiplier__all", "recyclate_value__*"),
         "System-wide multiplier plus polymer-specific absolute values."),
        ("Virgin-material displacement and avoided-production credits",
         ("avoided_virgin_credit_multiplier__all", "avoided_virgin_credit_multiplier__*"),
         "System-wide and polymer-specific avoided-credit tests."),
        ("Recyclate quality and substitution efficiency",
         ("recyclate_quality_value_capture__all", "recyclate_quality_value_capture__*",
          "substitution_efficiency__all", "substitution_efficiency__*"),
         "Economic value capture and environmental functional substitution are separate."),
        ("Autoclave energy demand", ("autoclave_energy_demand",),
         "Updates both route LCIA factors and operating cost."),
        ("SUT polymer composition", ("polymer_composition_share__*", "polymer_mass_per_batch__*"),
         "LDPE, LLDPE, PP, TPE, EVOH and other loaded polymer classes are generated dynamically."),
        ("Total plastic waste generated per batch", ("total_plastic_waste_per_batch",),
         "Scales all polymer masses proportionally."),
        ("Manufacturing throughput and batch allocation",
         ("manufacturing_throughput", "low_to_high_batch_ratio"),
         "Annual throughput and low/high node allocation are separate."),
        ("Water demand, wastewater generation and reuse",
         ("utility_water_demand_per_batch", "wastewater_generation_per_batch",
          "achievable_water_reuse_fraction"),
         "Covers eligible demand, generated wastewater and recovery."),
        ("Process electricity, water and waste variability",
         ("electricity_demand_per_batch", "utility_water_demand_per_batch",
          "wastewater_generation_per_batch", "total_plastic_waste_per_batch"),
         "Implemented as interpretable per-flow OAT parameters."),
        ("Recycling-facility availability and capacity",
         ("recycling_capacity_fraction__*", "recycling_facility_available__*"),
         "Capacity is continuous; availability is a binary stress test."),
        ("Acceptance of pharmaceutical plastic waste", ("pharma_plastic_accepted__*",),
         "Binary polymer-specific recycler-acceptance stress tests."),
        ("Transport distance, payload and vehicle type",
         ("recycling_transport_distance_multiplier", "all_waste_transport_distance_multiplier",
          "vehicle_payload__*", "transport_cost__*"),
         "Vehicle type remains an endogenous MILP decision; its cost and payload boundaries are varied."),
        ("Landfill, EfW and recycling gate fees",
         ("landfill_gate_fee", "efw_gate_fee", "recycling_gate_fee"),
         "All major waste-route fees are included."),
        ("Carbon value applied in the objective", ("lifecycle_carbon_value",),
         "Directly updates the policy-cost objective."),
        ("Wastewater-treatment CAPEX and OPEX",
         ("wastewater_treatment_annualised_capex", "wastewater_treatment_opex"),
         "Fixed annualised and variable treatment costs are separate."),
    ]

    rows: List[Dict[str, Any]] = []
    for family, patterns, note in requirements:
        found = matching(*patterns)
        rows.append({
            "target_parameter_family": family,
            "implemented": bool(found),
            "registered_parameter_count": len(found),
            "selected_parameter_count": sum(pid in selected for pid in found),
            "parameter_ids": "; ".join(found),
            "implementation_note": note,
        })
    return pd.DataFrame(rows)


def validate_target_parameter_coverage(specs: Sequence[SensitivityParameter]) -> pd.DataFrame:
    coverage = parameter_coverage_table(specs)
    missing = coverage.loc[~coverage["implemented"], "target_parameter_family"].tolist()
    require(not missing, f"Missing requested parameter families: {missing}")
    return coverage


def parameter_ranges_table(specs: Sequence[SensitivityParameter], boundary_points: int,
                           curve_points: int, bisection_iterations: int) -> pd.DataFrame:
    rows = []
    for spec in specs:
        rows.append({
            "parameter_id": spec.parameter_id,
            "parameter_label": spec.label,
            "group": spec.group,
            "parameter_type": spec.parameter_type,
            "scope_level": spec.scope_level,
            "target": spec.target,
            "configuration_source": spec.configuration_source,
            "linked_quantity_note": spec.linked_quantity_note,
            "unit": spec.unit,
            "nominal_value": spec.central,
            "local_reporting_low": spec.low,
            "local_reporting_high": spec.high,
            "boundary_search_low": spec.boundary_low,
            "boundary_search_high": spec.boundary_high,
            "boundary_discovery_points": len(spec.discovery_grid(boundary_points)),
            "decision_curve_points_if_boundary": curve_points if spec.parameter_type != "binary" else len(spec.discovery_grid(boundary_points)),
            "maximum_bisection_iterations_per_transition": bisection_iterations,
            "affects_optimisation": spec.affects_optimisation,
            "classification": spec.classification,
            "lcia_factor_status": spec.lcia_factor_status,
            "hard_lower_bound": spec.hard_low,
            "hard_upper_bound": spec.hard_high,
            "description": spec.description,
            "range_basis": spec.source_or_basis,
        })
    return pd.DataFrame(rows)


def scenario_rows(records: Mapping[str, Sequence[ScenarioRecord]], central: ScenarioRecord) -> pd.DataFrame:
    rows = []
    for parameter_id, parameter_records in records.items():
        for sequence, record in enumerate(parameter_records, start=1):
            row = record.row(central.metrics)
            row.update({"sequence": sequence, "parameter_id": parameter_id})
            rows.append(row)
    return pd.DataFrame(rows)


def decision_curve_components(analysis: LocalBoundaryAnalysis) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for parameter_id, records in analysis.decision_curve_records.items():
        spec = next(item for item in analysis.specs if item.parameter_id == parameter_id)
        for record in records:
            feasibility_metadata = component_metadata("Model feasible")
            rows.append({
                "parameter_id": parameter_id,
                "parameter_label": spec.label,
                "group": spec.group,
                "parameter_scope_level": spec.scope_level,
                "parameter_target": spec.target,
                "unit": spec.unit,
                "parameter_value": record.effective_value,
                "requested_value": record.requested_value,
                "component": "Model feasible",
                **feasibility_metadata,
                "component_kind": "numeric",
                "component_value_numeric": 1.0 if record.success else 0.0,
                "component_value_text": "feasible" if record.success else "infeasible",
            })
            if not record.success:
                continue
            for component, value in decision_components(analysis.master, record.model, record.result).items():
                is_numeric = isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)
                metadata = component_metadata(component)
                rows.append({
                    "parameter_id": parameter_id,
                    "parameter_label": spec.label,
                    "group": spec.group,
                    "parameter_scope_level": spec.scope_level,
                    "parameter_target": spec.target,
                    "unit": spec.unit,
                    "parameter_value": record.effective_value,
                    "requested_value": record.requested_value,
                    "component": component,
                    **metadata,
                    "component_kind": "numeric" if is_numeric else "categorical",
                    "component_value_numeric": float(value) if is_numeric else np.nan,
                    "component_value_text": "" if is_numeric else str(value),
                })
    return pd.DataFrame(rows)


def no_change_table(analysis: LocalBoundaryAnalysis) -> pd.DataFrame:
    boundary_ids = {item.parameter_id for item in analysis.boundaries}
    rows = []
    for spec in analysis.specs:
        records = analysis.boundary_scan_records.get(spec.parameter_id, [])
        completed = spec.parameter_id in analysis.completed_parameter_ids
        feasible_count = sum(record.success for record in records)
        all_feasible = bool(records) and feasible_count == len(records)
        if not completed:
            conclusion = "Pending"
        elif spec.parameter_id in boundary_ids:
            conclusion = "Decision boundary found"
        elif not records:
            conclusion = "No discovery records were produced"
        elif feasible_count == 0:
            conclusion = "No feasible optimisation result in the configured search range"
        elif not all_feasible:
            conclusion = (
                "No structural decision change identified, but one or more discovery "
                "evaluations were infeasible or failed"
            )
        else:
            conclusion = "No decision change found within configured search range"
        rows.append({
            "parameter_id": spec.parameter_id,
            "parameter_label": spec.label,
            "group": spec.group,
            "analysis_completed": completed,
            "decision_boundary_found": spec.parameter_id in boundary_ids,
            "feasible_decision_regimes": len({r.signature_hash for r in records if r.success}),
            "feasible_discovery_points": feasible_count,
            "total_discovery_points": len(records),
            "all_discovery_points_feasible": all_feasible,
            "search_low": spec.boundary_low,
            "search_high": spec.boundary_high,
            "conclusion": conclusion,
        })
    return pd.DataFrame(rows)


def write_outputs(analysis: LocalBoundaryAnalysis, result_dir: Path, master_path: Path,
                  boundary_points: int, curve_points: int, bisection_iterations: int) -> Dict[str, Path]:
    require(analysis.central_record is not None, "Central case has not been solved.")
    result_dir.mkdir(parents=True, exist_ok=True)
    ranges = parameter_ranges_table(analysis.specs, boundary_points, curve_points, bisection_iterations)
    scans = scenario_rows(analysis.boundary_scan_records, analysis.central_record)
    curves = scenario_rows(analysis.decision_curve_records, analysis.central_record)
    components = decision_curve_components(analysis)
    boundaries = analysis.boundary_table()
    changes = analysis.decision_change_table()
    no_change = no_change_table(analysis)

    model_solve_attempts = sum(
        1 for record in analysis.solver.cache.values()
        if record.parameter_id == "CENTRAL" or record.solve_seconds > 0.0
    )
    feasible_records = sum(record.success for record in analysis.solver.cache.values())
    total_records = len(analysis.solver.cache)
    planned_discovery = 1 + int(ranges["boundary_discovery_points"].sum())
    summary = pd.DataFrame([
        {"item": "selected parameters", "value": len(analysis.specs), "unit": "parameters"},
        {"item": "continuous parameters", "value": sum(s.parameter_type == "continuous" for s in analysis.specs), "unit": "parameters"},
        {"item": "discrete stress tests", "value": sum(s.parameter_type != "continuous" for s in analysis.specs), "unit": "parameters"},
        {"item": "initial discovery evaluations planned", "value": planned_discovery, "unit": "scenario records including central"},
        {"item": "boundary discovery points per continuous parameter", "value": boundary_points, "unit": "nominal points before anchors"},
        {"item": "decision curve points per parameter with a boundary", "value": curve_points, "unit": "nominal points before anchors"},
        {"item": "maximum bisection evaluations per transition", "value": bisection_iterations, "unit": "model evaluations"},
        {"item": "decision boundaries detected", "value": len(boundaries), "unit": "boundaries"},
        {"item": "unique scenario records generated", "value": total_records, "unit": "records"},
        {"item": "model solves attempted", "value": model_solve_attempts, "unit": "MILP solves"},
        {"item": "feasible scenario records", "value": feasible_records, "unit": "records"},
        {"item": "infeasible or failed scenario records", "value": total_records - feasible_records, "unit": "records"},
    ])
    central = pd.DataFrame([{
        **{metric: analysis.central_record.metrics[metric] for metric in OUTPUT_METRICS},
        "decision_signature": analysis.central_record.signature_hash,
        "decision_snapshot": analysis.central_record.decision_snapshot,
    }])

    tables = {
        "parameter_ranges.csv": ranges,
        "sampling_summary.csv": summary,
        "central_solution.csv": central,
        "boundary_scan_results.csv": scans,
        "decision_boundaries.csv": boundaries,
        "decision_changes.csv": changes,
        "decision_curve_results.csv": curves,
        "decision_curve_components.csv": components,
        "no_change_trail.csv": no_change,
        "parameter_coverage.csv": getattr(analysis, "parameter_coverage", pd.DataFrame()),
        "lcia_assumptions_audit.csv": analysis.impact.audit,
    }
    for filename, frame in tables.items():
        atomic_to_csv(frame, result_dir / filename)

    manifest = {
        "analysis_type": "local one-variable-at-a-time decision-boundary analysis",
        "run_status": "complete" if analysis.run_complete else "partial checkpoint",
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "completed_parameters": len(analysis.completed_parameter_ids),
        "completed_parameter_ids": list(analysis.completed_parameter_ids),
        "selected_parameters": len(analysis.specs),
        "completed_model_solves": analysis.solver.completed_model_solves,
        "planned_model_solve_estimate": analysis.solver.planned_solve_estimate,
        "master_model": str(master_path),
        "route_factor_workbook": analysis.impact.source_workbook,
        "solver_backend": analysis.solver.backend,
        "objective": analysis.solver.objective,
        "controls": {
            "boundary_discovery_points": boundary_points,
            "decision_curve_points": curve_points,
            "boundary_bisection_iterations": bisection_iterations,
            "boundary_relative_tolerance": BOUNDARY_RELATIVE_TOLERANCE,
            "boundary_absolute_tolerance": BOUNDARY_ABSOLUTE_TOLERANCE,
        },
        "outputs": {name: str(result_dir / name) for name in tables},
    }
    manifest_path = result_dir / "run_manifest.json"
    write_json(manifest_path, manifest)
    return {"result_dir": result_dir, "manifest": manifest_path}


def find_lca_settings() -> Optional[Path]:
    lca_parameters = PROJECT_ROOT.parent.parent / '2_LCA' / 'parameters'
    candidates = [
        lca_parameters / "lca_parameters.json",
        SCRIPT_DIR / "sensitivity_parameters.json",
    ]
    candidates.extend(sorted(lca_parameters.glob("lca_parameters*.json"), reverse=True)
                      if lca_parameters.exists() else [])
    return next((path for path in candidates if path.exists()), None)



def expand_parameter_id_patterns(
    patterns: Sequence[str],
    available_ids: Sequence[str],
) -> Tuple[List[str], List[str]]:
    """Expand exact IDs or shell-style wildcards while preserving configured order."""
    available = list(available_ids)
    expanded: List[str] = []
    unmatched: List[str] = []
    for raw_pattern in patterns:
        pattern = str(raw_pattern).strip()
        if not pattern:
            continue
        matches = [pid for pid in available if fnmatch.fnmatchcase(pid, pattern)]
        if not matches:
            unmatched.append(pattern)
            continue
        for pid in matches:
            if pid not in expanded:
                expanded.append(pid)
    return expanded, unmatched


def estimate_model_solves(
    specs: Sequence[SensitivityParameter],
    boundary_points: int,
    curve_points: int,
    bisection_iterations: int,
) -> Dict[str, Any]:
    """Planning estimate assuming one transition for each active continuous parameter."""
    rows = []
    total = 1  # central solve
    for spec in specs:
        if not spec.affects_optimisation:
            discovery = curve = bisection = 0
        else:
            discovery = len(spec.discovery_grid(boundary_points))
            if spec.parameter_type == "binary":
                curve = 0  # binary curve reuses discovery records
                bisection = 0
            else:
                curve = int(curve_points) + 3
                bisection = int(bisection_iterations)
        parameter_total = discovery + curve + bisection
        total += parameter_total
        rows.append({
            "parameter_id": spec.parameter_id,
            "discovery_solves": discovery,
            "assumed_dense_curve_solves": curve,
            "assumed_bisection_solves": bisection,
            "estimated_total": parameter_total,
        })
    return {
        "central_solve": 1,
        "selected_parameters": len(specs),
        "optimisation_active_parameters": sum(spec.affects_optimisation for spec in specs),
        "assumption": "one decision transition per active continuous parameter; exact total updates during execution",
        "estimated_model_solves": int(total),
        "parameter_breakdown": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local OAT decision-boundary analysis")
    parser.add_argument("--backend", choices=["gams", "scipy"], default=None)
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None, help="Path to MILP_OAT_sensitivity_analysis_parameters.json")
    parser.add_argument("--only", type=str, default="", help="Comma-separated parameter IDs or wildcard patterns")
    parser.add_argument("--groups", type=str, default="", help="Comma-separated parameter groups")
    parser.add_argument(
        "--configured-only",
        action="store_true",
        help="Use only parameter_ids explicitly listed in the JSON. By default the full built-in registry is analysed.",
    )
    parser.add_argument(
        "--exclude-discrete",
        action="store_true",
        help="Exclude binary facility-availability and pharmaceutical-acceptance stress tests.",
    )
    parser.add_argument("--quick", action="store_true", help="Small structural test")
    parser.add_argument("--tee", action="store_true", help="Print full GAMS output")
    return parser.parse_args()


def main() -> Dict[str, Path]:
    args = parse_args()
    boundary_points = 7 if args.quick else BOUNDARY_DISCOVERY_POINTS
    curve_points = 17 if args.quick else DECISION_CURVE_POINTS
    bisection_iterations = 12 if args.quick else BOUNDARY_BISECTION_ITERATIONS

    master_path = locate_master_file()
    master = import_master_module(master_path)
    result_dir = (args.result_dir or DEFAULT_RESULT_DIR).resolve()
    if result_dir.exists() and CLEAR_PREVIOUS_RESULTS:
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    config_path, parameter_config = load_parameter_config(args.config)
    central_data = master.load_model_data()
    impact = load_impact_model(master, central_data, find_lca_settings())
    full_specs = build_parameter_registry(
        central_data,
        impact,
        build_runtime_config(parameter_config),
    )
    validate_target_parameter_coverage(full_specs)

    configured_patterns = list(
        parameter_config
        .get("analyses", {})
        .get("local_decision_boundaries", {})
        .get("parameter_ids", [])
    )
    require(len(configured_patterns) == len(set(configured_patterns)),
            "Duplicate local parameter patterns in the JSON configuration.")
    available_order = [spec.parameter_id for spec in full_specs]
    available = set(available_order)
    configured_ids, unmatched_configured = expand_parameter_id_patterns(
        configured_patterns, available_order
    )
    require(not unmatched_configured,
            f"JSON parameter patterns matched no registered parameters: {unmatched_configured}")

    # Preserve explicit JSON pattern ordering first, then append every built-in
    # parameter not selected by those patterns. Thus a partial JSON cannot
    # accidentally suppress the plastic analysis unless --configured-only is used.
    configured_rank = {pid: index for index, pid in enumerate(configured_ids)}
    full_specs.sort(key=lambda spec: (
        0 if spec.parameter_id in configured_rank else 1,
        configured_rank.get(spec.parameter_id, 10**9),
    ))
    specs = list(full_specs)

    if args.configured_only:
        require(configured_ids,
                "--configured-only was requested but no JSON parameter patterns resolved.")
        configured_set = set(configured_ids)
        specs = [spec for spec in specs if spec.parameter_id in configured_set]

    requested_patterns = [value.strip() for value in args.only.split(",") if value.strip()]
    requested_ids: set[str] = set()
    if requested_patterns:
        expanded_requested, unmatched_requested = expand_parameter_id_patterns(
            requested_patterns, available_order
        )
        require(not unmatched_requested,
                f"--only patterns matched no registered parameters: {unmatched_requested}")
        requested_ids = set(expanded_requested)
        specs = [spec for spec in specs if spec.parameter_id in requested_ids]

    groups = {value.strip() for value in args.groups.split(",") if value.strip()}
    if groups:
        specs = [spec for spec in specs if spec.group in groups]
    if args.exclude_discrete:
        specs = [spec for spec in specs if spec.parameter_type == "continuous"]
    require(specs, "No local sensitivity parameters selected.")

    selected_ids = [spec.parameter_id for spec in specs]
    coverage = parameter_coverage_table(full_specs, selected_ids=selected_ids)

    # Preserve the external configuration when present; otherwise write the
    # built-in empty override structure so the run is still completely auditable.
    copied_config_path = result_dir / PARAMETER_CONFIG_FILENAME
    if config_path is not None:
        _atomic_replace_text(copied_config_path, config_path.read_text(encoding="utf-8"))
        configuration_source = str(config_path)
    else:
        write_json(copied_config_path, parameter_config)
        configuration_source = "built-in parameter registry defaults"

    write_json(result_dir / "resolved_parameter_configuration.json", {
        "analysis": "local_decision_boundaries",
        "source_configuration": configuration_source,
        "selection_mode": (
            "explicit --only"
            if requested_ids else
            "JSON configured-only"
            if args.configured_only else
            "full built-in registry with optional JSON overrides"
        ),
        "configured_parameter_patterns": configured_patterns,
        "selected_parameter_ids": selected_ids,
        "parameters": {
            spec.parameter_id: {
                "label": spec.label,
                "classification": spec.classification,
                "scope_level": spec.scope_level,
                "target": spec.target,
                "parameter_type": spec.parameter_type,
                "configuration_source": spec.configuration_source,
                "unit": spec.unit,
                "nominal": spec.central,
                "uncertainty_low": spec.low,
                "uncertainty_high": spec.high,
                "boundary_low": spec.boundary_low,
                "boundary_high": spec.boundary_high,
                "hard_low": spec.hard_low,
                "hard_high": spec.hard_high,
                "lcia_factor_status": spec.lcia_factor_status,
                "linked_quantity_note": spec.linked_quantity_note,
                "affects_optimisation": spec.affects_optimisation,
                "range_basis": spec.source_or_basis,
                "description": spec.description,
            } for spec in specs
        },
    })

    solver = ScenarioSolver(
        master=master,
        central_data=central_data,
        impact=impact,
        result_dir=result_dir,
        backend=args.backend or SOLVER_BACKEND,
        objective="policy_cost",
        tee=args.tee or GAMS_TEE,
    )
    solve_plan = estimate_model_solves(
        specs, boundary_points, curve_points, bisection_iterations
    )
    write_json(result_dir / "solve_plan_estimate.json", solve_plan)
    solver.configure_progress(solve_plan)
    print("\nOAT SOLVE PLAN", flush=True)
    print(f"  selected parameters:          {solve_plan['selected_parameters']}", flush=True)
    print(f"  optimisation-active:          {solve_plan['optimisation_active_parameters']}", flush=True)
    print(f"  estimated MILP solves:        ~{solve_plan['estimated_model_solves']:,}", flush=True)
    print(f"  assumption: {solve_plan['assumption']}", flush=True)

    analysis = LocalBoundaryAnalysis(
        master, central_data, impact, specs, solver,
        boundary_points, curve_points, bisection_iterations,
    )
    analysis.parameter_coverage = coverage

    write_json(result_dir / "checkpoint_status.json", {
        "stage": "initialised",
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_complete": False,
        "completed_parameters": 0,
        "selected_parameters": len(analysis.specs),
        "completed_model_solves": 0,
        "planned_model_solve_estimate": solver.planned_solve_estimate,
    })

    def checkpoint(stage: str) -> None:
        write_outputs(
            analysis, result_dir, master_path,
            boundary_points, curve_points, bisection_iterations,
        )
        write_json(result_dir / "checkpoint_status.json", {
            "stage": stage,
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "run_complete": analysis.run_complete,
            "completed_parameters": len(analysis.completed_parameter_ids),
            "selected_parameters": len(analysis.specs),
            "completed_model_solves": solver.completed_model_solves,
            "planned_model_solve_estimate": solver.planned_solve_estimate,
        })
        print(
            f"  checkpoint saved: {stage} | "
            f"{len(analysis.completed_parameter_ids)}/{len(analysis.specs)} parameters",
            flush=True,
        )

    analysis.checkpoint_callback = checkpoint
    try:
        analysis.run()
    except BaseException as exc:
        failure_payload = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "completed_parameters": list(analysis.completed_parameter_ids),
            "completed_model_solves": solver.completed_model_solves,
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        }
        write_json(result_dir / "run_failure.json", failure_payload)
        if analysis.central_record is not None:
            try:
                checkpoint(f"interrupted or failed: {type(exc).__name__}")
            except Exception as checkpoint_exc:
                failure_payload["checkpoint_error"] = str(checkpoint_exc)
                write_json(result_dir / "run_failure.json", failure_payload)
        print(
            f"\nOAT RUN STOPPED: {type(exc).__name__}: {exc}\n"
            f"Last safe checkpoint: {result_dir}",
            file=sys.stderr,
            flush=True,
        )
        raise
    outputs = write_outputs(
        analysis, result_dir, master_path,
        boundary_points, curve_points, bisection_iterations,
    )
    print("\nLOCAL DECISION-BOUNDARY ANALYSIS COMPLETE")
    print(f"Results: {outputs['result_dir']}")
    print(f"Manifest: {outputs['manifest']}")
    return outputs


if __name__ == "__main__":
    main()
