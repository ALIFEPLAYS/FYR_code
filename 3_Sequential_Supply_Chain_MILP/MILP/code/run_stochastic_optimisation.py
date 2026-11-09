from __future__ import annotations

"""
Sobol quasi-Monte-Carlo solution-space assessment for the industrial-symbiosis MILP.

This is intentionally NOT a Saltelli/Jansen Sobol-index calculation and it is NOT
implemented as a coupled two-stage stochastic program.  A single scrambled Sobol
low-discrepancy design is used to cover a bounded uncertainty space.  For every
sampled realization the deterministic MILP is rebuilt and solved, and the optimal
KPI vector and structural decision regime are retained.

Carbon-value invariant
----------------------
This uncertainty experiment always uses the central carbon-value + cost objective.
The lifecycle carbon value is NOT a Sobol uncertainty dimension: it is inherited
from the validated master MILP after the master has converted the source carbon
value to its common nominal-GBP-2026 basis, and that exact value is held fixed for
every sample. Explicit low/high/no-carbon-value cases belong in scenario/policy
analysis rather than this base uncertainty ensemble.

The resulting empirical set
    {theta_s, x*(theta_s), y(theta_s, x*(theta_s))}
answers the thesis question: "what can the optimiser choose, and what outcomes can
result, over the stated uncertainty space?"

Run this file from the submission root after validating the central MILP inputs.

Typical final run:
    python run_stochastic_optimisation.py --backend gams

Quick smoke test:
    python run_stochastic_optimisation.py --quick --backend scipy
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import qmc


# =============================================================================
# USER CONTROLS
# =============================================================================
SOLVER_BACKEND = "gams"              # final thesis results: gams; screening: scipy
GAMS_TEE = False
SOBOL_SAMPLE_POWER = 11               # N = 2048 continuous uncertainty realizations (final thesis default)
SOBOL_SEED = 104729
CHECKPOINT_EVERY = 25
# Pyomo/GAMS shell solves are deliberately recycled through a fresh Python process
# during long runs.  This avoids accumulation of shell/temp/file-handle resources
# in a single interpreter on macOS.  Scientific results are unchanged.
GAMS_PROCESS_RESTART_EVERY = 64
MAX_GAMS_TECHNICAL_RETRIES_PER_SAMPLE = 3
CLEAR_PREVIOUS_RESULTS = True
RESULT_DIRECTORY = "results/stochastic_optimisation_solution_space"
PARAMETER_CONFIG_FILENAME = "MILP_stochastic_optimisation_parameters.json"
OBJECTIVE = "policy_cost"            # annual cost + FIXED central lifecycle-carbon-value objective
FIXED_CARBON_VALUE_PARAMETER_ID = "lifecycle_carbon_value"
CARBON_VALUE_POLICY_ID = "fixed_master_central_nominal_GBP_2026"

OUTPUT_METRICS = ("cost", "gwp", "fossil_resource_use", "water_scarcity", "human_health_daly", "ecosystem_quality")
OUTPUT_UNITS = {
    "cost": "GBP/year",
    "gwp": "kg CO2e/year",
    "fossil_resource_use": "kg oil-eq/year (method unit)",
    "water_scarcity": "m3 world-eq/year (method unit)",
    "human_health_daly": "DALY/year",
    "ecosystem_quality": "species.yr/year",
}

SCRIPT_DIR = Path(__file__).resolve().parent
# The thesis repository is organised as <project>/code and <project>/inputs.
# Keep generated outputs outside the input directory and outside the code folder.
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name.lower() == "code" else SCRIPT_DIR
DEFAULT_RESULT_DIR = PROJECT_ROOT / RESULT_DIRECTORY


# =============================================================================
# Utilities
# =============================================================================
def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def finish_time_text(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    finish = datetime.now().astimezone() + timedelta(seconds=float(seconds))
    return finish.strftime("%H:%M %Z")


def coerce_bool(value: Any) -> bool:
    """Robust bool parsing for checkpoint CSV rows."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def is_technical_exception_row(row: Mapping[str, Any]) -> bool:
    failure_class = str(row.get("failure_class", "")).strip().lower()
    status = str(row.get("status", "")).strip().lower()
    return failure_class == "technical_exception" or status == "exception"


def is_gams_solver_exception(row: Mapping[str, Any]) -> bool:
    if not is_technical_exception_row(row):
        return False
    text = str(row.get("error", "")).lower()
    markers = (
        "gams",
        "solverfactory('gams')",
        "gams encountered an error during solve",
        "solve failed for so_base",
    )
    return any(marker in text for marker in markers)


def restart_current_process_for_resume(reason: str) -> None:
    """Replace this interpreter with a clean one and resume the same run.

    This is intentionally an exec, not a child process: from the user's point of
    view it remains one command/run, while Python/Pyomo/GAMS process resources are
    fully reset between solve batches.
    """
    script = str(Path(__file__).resolve())
    original = [arg for arg in sys.argv[1:] if arg != "--resume"]
    argv = [sys.executable, script, *original, "--resume"]
    print(f"\n[GAMS PROCESS RECYCLE] {reason}", flush=True)
    print("Checkpoint is saved. Restarting the Python interpreter and resuming automatically...", flush=True)
    os.execv(sys.executable, argv)


def classify_nonoptimal_message(message: Any) -> str:
    text = str(message).lower()
    if "infeasible" in text:
        return "infeasible"
    if "unbounded" in text:
        return "unbounded"
    return "solver_nonoptimal"


def canonical_hash(payload: Mapping[str, Any], n: int = 12) -> Tuple[str, str]:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n], text


def classify_share(value: float, zero: str, full: str, mixed: str,
                   low_tol: float = 0.005, high_tol: float = 0.995) -> str:
    if value <= low_tol:
        return zero
    if value >= high_tol:
        return full
    return mixed


def resolve_range(definition: Mapping[str, Any], central: float,
                  hard_bounds: Optional[Mapping[str, Any]] = None) -> Tuple[float, float]:
    """Resolve the *JSON-declared* uncertainty interval around a live central value.

    The OAT/local registry also contains boundary-search ranges, but those are not the
    stochastic uncertainty bounds.  For this solution-space assessment the external
    JSON is authoritative.  This distinction is critical: otherwise OAT boundary
    ranges (including zero-capacity stress values) can leak into the Sobol design.
    """
    mode = str(definition.get("mode", "absolute"))
    if mode == "absolute":
        low, high = float(definition["low"]), float(definition["high"])
    elif mode == "nominal_multiplier":
        low = central * float(definition["low_factor"])
        high = central * float(definition["high_factor"])
    elif mode == "nominal_additive":
        low = central + float(definition["low_delta"])
        high = central + float(definition["high_delta"])
    else:
        raise ValueError(f"Unsupported uncertainty_range mode: {mode!r}")
    hard = dict(hard_bounds or definition.get("hard_bounds", {}) or {})
    if "low" in hard:
        low = max(low, float(hard["low"]))
    if "high" in hard:
        high = min(high, float(hard["high"]))
    require(math.isfinite(low) and math.isfinite(high),
            f"Non-finite uncertainty interval resolved around central value {central}.")
    require(low < high, f"Degenerate uncertainty interval [{low}, {high}] around central value {central}.")
    require(low <= central <= high,
            f"Resolved uncertainty interval [{low}, {high}] does not contain central value {central}.")
    return low, high


# =============================================================================
# Shared model / mutation layer discovery
# =============================================================================
def import_module_from_path(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def locate_shared_module(explicit: Optional[Path] = None) -> Path:
    candidates: List[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit).expanduser())
    env = os.environ.get("IS_SENSITIVITY_SHARED")
    if env:
        candidates.append(Path(env).expanduser())
    names = [
        "run_oat_decision_boundaries.py",
        "run_local_decision_boundaries.py",
    ]
    for name in names:
        candidates.append(SCRIPT_DIR / name)
    for pattern in ("run_oat_decision_boundaries*.py", "run_local_decision_boundaries*.py"):
        candidates.extend(sorted(SCRIPT_DIR.glob(pattern), reverse=True))

    required = {
        "build_parameter_registry", "build_runtime_config", "clone_state",
        "sync_gwp_coefficients", "evaluate_extended_impacts", "decision_signature",
        "decision_components", "ScenarioSolver", "load_impact_model", "find_lca_settings",
        "locate_master_file", "import_master_module",
    }
    seen: set[Path] = set()
    failures: List[str] = []
    for raw in candidates:
        try:
            path = raw.resolve()
        except Exception:
            continue
        if path in seen or not path.exists() or path == Path(__file__).resolve():
            continue
        seen.add(path)
        try:
            module = import_module_from_path(path, f"stochastic_shared_{len(seen)}")
            missing = sorted(required - set(dir(module)))
            if not missing:
                return path
            failures.append(f"{path.name}: missing {missing}")
        except Exception as exc:
            failures.append(f"{path.name}: import failed ({exc})")
    detail = "\n  ".join(failures[:8])
    raise FileNotFoundError(
        "Could not find a compatible shared OAT/local decision-boundary module beside this script. "
        "Place run_oat_decision_boundaries.py or run_local_decision_boundaries.py beside it, "
        "or pass --shared-module.\n  " + detail
    )


def load_config(explicit: Optional[Path]) -> Tuple[Path, Dict[str, Any]]:
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env = os.environ.get("IS_STOCHASTIC_CONFIG")
    if env:
        candidates.append(Path(env).expanduser())
    # Support both a self-contained package (JSON beside the runner) and the
    # thesis repository layout (<project>/inputs/<config>).
    candidates.append(SCRIPT_DIR / PARAMETER_CONFIG_FILENAME)
    candidates.append(PROJECT_ROOT.parent / "parameters" / PARAMETER_CONFIG_FILENAME)
    for p in candidates:
        if p.exists():
            cfg = read_json(p.resolve())
            require(isinstance(cfg.get("parameters"), Mapping), "Config must contain 'parameters'.")
            require(isinstance(cfg.get("analysis"), Mapping), "Config must contain 'analysis'.")
            return p.resolve(), cfg
    raise FileNotFoundError(f"Could not locate {PARAMETER_CONFIG_FILENAME}; pass --config explicitly.")


# =============================================================================
# Central carbon-value policy invariant
# =============================================================================
def central_carbon_value_2026_gbp(central_data: Any) -> float:
    """Return the master model's already-converted central carbon value.

    The master MILP is the single source of truth for price-year conversion.
    No carbon-value conversion or duplicated central constant is performed here.
    """
    value = float(central_data.lifecycle_carbon_value_gbp_per_t)
    require(math.isfinite(value) and value > 0.0,
            f"Master central lifecycle carbon value must be positive and finite; got {value!r}.")
    return value


def enforce_central_carbon_value(state: Any, fixed_value: float) -> None:
    """Pin a cloned/sample state to the master central carbon value."""
    state.data.lifecycle_carbon_value_gbp_per_t = float(fixed_value)
    realised = float(state.data.lifecycle_carbon_value_gbp_per_t)
    require(
        math.isclose(realised, float(fixed_value), rel_tol=0.0, abs_tol=1e-12),
        "Failed to enforce the fixed master central lifecycle carbon value.",
    )


# =============================================================================
# Parameter resolution
# =============================================================================
@dataclass(frozen=True)
class ParameterMeta:
    parameter_id: str
    label: str
    unit: str
    central: float
    low: float
    high: float
    group: str
    source_type: str                 # 'registry' or custom kind
    range_basis: str


def resolve_registry_parameters(core: Any, central_data: Any, impact: Any,
                                config: Mapping[str, Any], requested: Optional[set[str]]) -> Tuple[List[Any], List[ParameterMeta]]:
    registry = core.build_parameter_registry(
        central_data, impact, core.build_runtime_config(config)
    )
    by_id = {s.parameter_id: s for s in registry}
    ids = list(config["analysis"]["registry_parameter_ids"])
    require(len(ids) == len(set(ids)), "Duplicate registry_parameter_ids in config.")

    # Thesis policy invariant: carbon value is part of the objective definition,
    # not a Sobol uncertainty dimension. Keep the JSON entry available for other
    # analyses, but exclude it here even if an older configuration still lists it.
    ids = [pid for pid in ids if pid != FIXED_CARBON_VALUE_PARAMETER_ID]
    if requested is not None:
        ids = [pid for pid in ids if pid in requested]
    missing = [pid for pid in ids if pid not in by_id]
    require(not missing,
            "Configured registry parameter(s) are not available from the shared model layer: " + ", ".join(missing))
    specs = [by_id[pid] for pid in ids]
    metas: List[ParameterMeta] = []
    for spec in specs:
        definition = config.get("parameters", {}).get(spec.parameter_id, {})
        require(
            isinstance(definition.get("uncertainty_range"), Mapping),
            f"Missing JSON uncertainty_range for registry parameter {spec.parameter_id}. "
            "The stochastic runner deliberately does not fall back to OAT boundary ranges.",
        )
        central = float(spec.central)
        low, high = resolve_range(
            definition["uncertainty_range"], central, definition.get("hard_bounds")
        )
        metas.append(ParameterMeta(
            spec.parameter_id,
            str(definition.get("label", spec.label)),
            str(definition.get("unit", spec.unit)),
            central,
            low,
            high,
            spec.group,
            "registry",
            str(definition.get("range_basis", spec.source_or_basis)),
        ))
    return specs, metas


def resolve_custom_parameters(config: Mapping[str, Any], requested: Optional[set[str]]) -> List[ParameterMeta]:
    out: List[ParameterMeta] = []
    for pid in config["analysis"].get("custom_parameter_ids", []):
        if requested is not None and pid not in requested:
            continue
        d = config["custom_parameters"][pid]
        central = float(d.get("central", 1.0))
        low, high = resolve_range(d["uncertainty_range"], central, d.get("hard_bounds"))
        out.append(ParameterMeta(
            parameter_id=pid,
            label=str(d.get("label", pid)),
            unit=str(d.get("unit", "-")),
            central=central,
            low=low,
            high=high,
            group=str(d.get("group", "custom")),
            source_type=str(d["kind"]),
            range_basis=str(d.get("range_basis", "")),
        ))
    return out


def validate_sampling_ranges(metas: Sequence[ParameterMeta], config: Mapping[str, Any]) -> None:
    """Fail fast on implausible/accidental zero-width or zero-valued Sobol ranges."""
    for meta in metas:
        definition = config.get("parameters", {}).get(
            meta.parameter_id, config.get("custom_parameters", {}).get(meta.parameter_id, {})
        )
        require(meta.high > meta.low, f"Invalid sampling range for {meta.parameter_id}: [{meta.low}, {meta.high}]")
        # Zero-capacity outage cases belong in the separate stress analysis.
        require(
            meta.low > 0.0,
            f"Sampling lower bound for {meta.parameter_id} is {meta.low}. "
            "Zero is reserved for explicit outage/stress cases and is not allowed in the continuous Sobol ensemble.",
        )
        hard = definition.get("hard_bounds", {}) or {}
        if "low" in hard:
            require(meta.low >= float(hard["low"]) - 1e-12, f"Hard lower bound not respected for {meta.parameter_id}")
        if "high" in hard:
            require(meta.high <= float(hard["high"]) + 1e-12, f"Hard upper bound not respected for {meta.parameter_id}")


def parameter_table(metas: Sequence[ParameterMeta], config: Mapping[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for m in metas:
        d = config.get("parameters", {}).get(m.parameter_id, config.get("custom_parameters", {}).get(m.parameter_id, {}))
        rows.append({
            "parameter_id": m.parameter_id,
            "label": m.label,
            "group": m.group,
            "unit": m.unit,
            "nominal": m.central,
            "sampling_low": m.low,
            "sampling_high": m.high,
            "sampling_range_origin": "MILP_stochastic_optimisation_parameters.json",
            "sampling_measure": "triangular distribution (nominal mode) via scrambled Sobol sequence",
            "parameter_source_type": m.source_type,
            "classification": d.get("classification", ""),
            "range_basis": m.range_basis,
            "interpretation_note": d.get("interpretation_note", ""),
        })
    return pd.DataFrame(rows)


# =============================================================================
# Sampling design and custom mutations
# =============================================================================
def generate_sobol_design(metas: Sequence[ParameterMeta], power: int, seed: int) -> pd.DataFrame:
    require(power >= 1, "Sobol sample power must be >= 1.")
    require(len(metas) >= 1, "At least one continuous uncertainty dimension is required.")
    sampler = qmc.Sobol(d=len(metas), scramble=True, seed=int(seed))
    unit = sampler.random_base2(m=int(power))
    lows = np.array([m.low for m in metas], dtype=float)
    modes = np.array([m.central for m in metas], dtype=float)
    highs = np.array([m.high for m in metas], dtype=float)
    require(np.all((lows <= modes) & (modes <= highs)), "Triangular modes must lie within sampling bounds.")
    # Inverse CDF maps the nested Sobol design to bounded triangular marginals.
    # The nominal model value is the mode, including when it equals a bound.
    turning_points = (modes - lows) / (highs - lows)
    scaled = np.where(
        unit < turning_points,
        lows + np.sqrt(unit * (highs - lows) * (modes - lows)),
        highs - np.sqrt((1.0 - unit) * (highs - lows) * (highs - modes)),
    )
    rows: List[Dict[str, Any]] = []
    for i in range(len(scaled)):
        row: Dict[str, Any] = {"sobol_index": i}
        for j, meta in enumerate(metas):
            row[meta.parameter_id] = float(scaled[i, j])
            row[f"u__{meta.parameter_id}"] = float(unit[i, j])
        rows.append(row)
    return pd.DataFrame(rows)


def efw_export_kwh_per_kg_from_lca(master: Any, impact: Any) -> float:
    """Recover the EfW electricity export intensity already scored in the LCA."""
    route_factors = pd.read_excel(master.INPUT_FILES["lca"], sheet_name="route_factor_matrix")
    efw_rows = route_factors.loc[
        route_factors["optimisation_flow"].astype(str).str.startswith("sut_plastic__")
        & route_factors["pathway"].astype(str).eq("incineration_efw_heat_electricity")
    ]
    require(not efw_rows.empty, "No EfW route rows in the LCA workbook.")
    grid = float(impact.electricity_base["grid", "gwp"])
    require(grid > 0.0, "The central grid GWP factor must be positive.")
    export = -pd.to_numeric(efw_rows["gwp__efw_electricity_credit"], errors="raise").to_numpy() / grid
    require(np.all(np.isfinite(export)) and np.all(export > 0.0), "Invalid EfW electricity credit in LCA workbook.")
    require(np.allclose(export, export[0], rtol=0.0, atol=1e-9),
            "EfW export intensity differs across polymer routes; a single grid linkage is not valid.")
    return float(export[0])


def apply_custom_parameters(state: Any, central_data: Any, values: Mapping[str, float],
                            custom_metas: Sequence[ParameterMeta], config: Mapping[str, Any]) -> None:
    # Route distances are independent route-level multipliers about the central geography.
    for meta in custom_metas:
        if meta.source_type != "route_distance_multiplier":
            continue
        if meta.parameter_id not in values:
            continue
        d = config["custom_parameters"][meta.parameter_id]
        route = str(d["target_route"])
        factor = float(values[meta.parameter_id])
        require(route in state.data.routes, f"Unknown target route for {meta.parameter_id}: {route}")
        for site in state.data.sites:
            state.data.distance_km[site, route] = float(central_data.distance_km[site, route]) * factor

    # Composition is a compositional variable.  Do not apply six shares independently.
    # Five relative weights are sampled against a closure/reference component; all masses
    # are renormalised so the sampled total-plastic amount is exactly preserved.
    comp_metas = [m for m in custom_metas if m.source_type == "composition_weight"]
    if comp_metas:
        total = sum(float(state.data.polymer_kg_per_batch[p]) for p in state.data.polymers)
        weighted: Dict[str, float] = {p: float(state.data.polymer_kg_per_batch[p]) for p in state.data.polymers}
        for meta in comp_metas:
            if meta.parameter_id not in values:
                continue
            d = config["custom_parameters"][meta.parameter_id]
            polymer = str(d["polymer"])
            require(polymer in state.data.polymers, f"Unknown polymer in {meta.parameter_id}: {polymer}")
            weighted[polymer] *= float(values[meta.parameter_id])
        denom = sum(weighted.values())
        require(denom > 0, "Composition closure produced zero total mass.")
        scale = total / denom
        for polymer in state.data.polymers:
            state.data.polymer_kg_per_batch[polymer] = weighted[polymer] * scale


def couple_electricity_prices(state: Any, central_data: Any,
                              values: Mapping[str, float], config: Mapping[str, Any]) -> None:
    """Apply the same sampled real-price path to both 2026 nominal tariffs."""
    rule = config["analysis"].get("cppa_price_rule")
    if rule is None:
        return  # Preserve reproducibility of older, independently priced runs.
    require(rule == "shared_grid_price_multiplier", f"Unknown cPPA price rule: {rule}")
    require("renewable_cppa_premium" not in values,
            "A separate cPPA premium cannot be sampled with shared-price coupling")
    multiplier = float(values.get("grid_electricity_price_multiplier", 1.0))
    grid_nominal = float(central_data.electricity_cost_gbp_per_kwh["grid"])
    wind_nominal = float(central_data.electricity_cost_gbp_per_kwh["wind"])
    grid_realised = float(state.data.electricity_cost_gbp_per_kwh["grid"])
    require(math.isclose(grid_realised, grid_nominal * multiplier, rel_tol=0, abs_tol=1e-10),
            "Grid tariff does not match the sampled common electricity-price multiplier")
    state.data.electricity_cost_gbp_per_kwh["wind"] = wind_nominal * multiplier


def biomass_steam_credit_reference(master: Any, central_data: Any,
                                   config: Mapping[str, Any]) -> Dict[str, Any]:
    """Recover the already-scored gas and steam credits without inventing flows."""
    definition = config.get("custom_parameters", {}).get("biomass_steam_efficiency")
    if definition is None:
        return {}
    nominal_efficiency = float(definition["central"])
    assumptions = read_json(PROJECT_ROOT.parent / "parameters"
                            / "electrcity_biomass_route_assumptions.json")
    require(math.isclose(nominal_efficiency,
                         float(assumptions["biomethane_to_useful_steam_efficiency"]),
                         rel_tol=0.0, abs_tol=1e-12),
            "Stochastic biomass efficiency mode differs from the master route assumption")
    stages = pd.read_excel(master.INPUT_FILES["lca"], sheet_name="lca_stage_breakdown")
    gas_credit: Dict[str, float] = {}
    steam_credit: Dict[str, float] = {}
    for category in central_data.biomass_route_factors["ad_gas_supply"]:
        rows = stages.loc[
            stages["source_module"].eq("biomass")
            & stages["life_cycle_stage"].eq("avoided_natural_gas")
            & stages["impact_category"].eq(category)
        ]
        require(len(rows) == 1, f"Expected one scored biomass gas credit for {category}")
        gas_credit[category] = float(rows.iloc[0]["represented_impact_per_batch"]) / float(
            central_data.biomass_kg_per_batch)
        steam_rows = stages.loc[
            stages["optimisation_flow"].eq("manufacturing__steam")
            & stages["life_cycle_stage"].eq("manufacturing_utility_supply")
            & stages["impact_category"].eq(category)
        ]
        require(len(steam_rows) == 1,
                f"Expected one scored external-steam activity for {category}")
        gas = float(central_data.biomass_route_factors["ad_gas_supply"][category])
        steam = float(central_data.biomass_route_factors["ad_external_steam"][category])
        steam_credit[category] = gas - gas_credit[category] - steam
        scored_credit = (float(central_data.biomass_useful_heat_mj_per_kg["ad_external_steam"])
                         * float(steam_rows.iloc[0]["impact_factor"]))
        require(math.isclose(steam_credit[category], scored_credit,
                             rel_tol=1e-8, abs_tol=1e-10),
                f"Recovered biomass steam credit disagrees with the scored activity for {category}")
        require(gas_credit[category] < 0 and steam_credit[category] > 0,
                f"Unexpected gas/steam biomass credit signs for {category}")
    return {"nominal_efficiency": nominal_efficiency,
            "gas_credit_per_kg": gas_credit, "steam_credit_per_kg": steam_credit}


def apply_biomass_steam_efficiency(state: Any, central_data: Any,
                                  values: Mapping[str, float],
                                  reference: Mapping[str, Any]) -> None:
    """Rescale only useful-steam displacement; retain AD and gas-route inventory."""
    if "biomass_steam_efficiency" not in values:
        return
    require(bool(reference), "Missing scored biomass steam-credit reference")
    efficiency = float(values["biomass_steam_efficiency"])
    require(0.0 < efficiency <= 1.0, "Biomass useful-steam efficiency is outside (0, 1]")
    ratio = efficiency / float(reference["nominal_efficiency"])
    for category, heat_credit in reference["steam_credit_per_kg"].items():
        gas = float(central_data.biomass_route_factors["ad_gas_supply"][category])
        gas_credit = float(reference["gas_credit_per_kg"][category])
        state.data.biomass_route_factors["ad_external_steam"][category] = (
            gas - gas_credit - ratio * float(heat_credit))
    state.data.biomass_useful_heat_mj_per_kg["ad_external_steam"] = (
        float(central_data.biomass_useful_heat_mj_per_kg["ad_external_steam"]) * ratio)


def realised_composition(state: Any) -> Dict[str, float]:
    masses = {p: float(state.data.polymer_kg_per_batch[p]) for p in state.data.polymers}
    total = sum(masses.values())
    return {p: (masses[p] / total if total > 0 else np.nan) for p in state.data.polymers}


def structural_precheck(state: Any) -> Dict[str, Any]:
    """Record transparent feasibility diagnostics before the MILP is solved.

    These do not replace the solver.  They simply make it possible to explain why
    a *genuine* infeasible realization occurred.  In particular, the current master
    model has fixed installed wastewater-treatment capacity while throughput, batch
    allocation and wastewater generation are uncertain.
    """
    data = state.data
    out: Dict[str, Any] = {}
    active = [site for site in data.sites if int(data.site_open[site]) == 1]
    ww = float(data.wastewater_m3_per_batch)
    high_req = float(data.batch_high) * ww
    low_req = float(data.batch_low) * ww
    candidate_ratios: List[float] = []
    feasible_low_sites: List[str] = []
    for low_site in active:
        ratios = []
        ok = True
        for site in active:
            cap = float(data.wwt_capacity_m3[site])
            req = low_req if site == low_site else high_req
            ratio = req / cap if cap > 0 else np.inf
            ratios.append(ratio)
            if ratio > 1.0 + 1e-10:
                ok = False
        candidate_ratios.append(max(ratios) if ratios else np.nan)
        if ok:
            feasible_low_sites.append(str(low_site))
    out["precheck__wwt_allocation_feasible"] = bool(feasible_low_sites)
    out["precheck__best_wwt_capacity_utilisation"] = (
        float(min(candidate_ratios)) if candidate_ratios else np.nan
    )
    out["precheck__wwt_feasible_low_site_count"] = int(len(feasible_low_sites))
    out["precheck__batch_high"] = float(data.batch_high)
    out["precheck__batch_low"] = float(data.batch_low)
    out["precheck__total_batches"] = float(data.total_batches)
    out["precheck__wastewater_m3_per_batch"] = ww
    out["precheck__high_site_wastewater_m3_per_year"] = high_req

    # Route-capacity loading ratios are descriptive because alternative disposal
    # routes remain available; values >1 therefore need not imply infeasibility.
    for route, cap in data.route_capacities_t.items():
        eligible_mass_per_batch = sum(
            float(data.polymer_kg_per_batch[p]) for p in data.polymers if data.eligibility[p, route]
        )
        eligible_feed = float(data.total_batches) * eligible_mass_per_batch / 1000.0
        ratio = eligible_feed / float(cap) if float(cap) > 0 else np.inf
        out[f"precheck__eligible_feed_to_capacity__{route}"] = float(ratio)
    return out


# =============================================================================
# Decision extraction
# =============================================================================
def macro_decision_regime(master: Any, model: Any, result: Any) -> Tuple[str, str, str, Dict[str, Any]]:
    routes, sites, mix = master.solution_decisions(model, result)
    numeric: Dict[str, Any] = {}
    payload: Dict[str, Any] = {}

    if sites.empty:
        payload.update({"low_batch_location": "", "electricity": "no site decision", "water": "no site decision"})
        wind_share = reuse_share = 0.0
    else:
        low = sites.loc[sites["is_low_batch_location"]]
        low_location = str(low.iloc[0]["location"]) if len(low) else ""
        wind = float(sites["wind_kwh"].sum())
        grid = float(sites["grid_kwh"].sum())
        wind_share = wind / (wind + grid) if wind + grid > 1e-12 else 0.0
        reused = float(sites["water_reused_m3"].sum())
        fresh = float(sites["fresh_utility_water_m3"].sum())
        reuse_share = reused / (reused + fresh) if reused + fresh > 1e-12 else 0.0
        payload["low_batch_location"] = low_location
        payload["electricity"] = classify_share(wind_share, "grid", "wind", "grid+wind")
        payload["water"] = classify_share(reuse_share, "no reuse", "full reuse", "partial reuse")
        numeric["decision__low_batch_location"] = low_location
        if "biomass_route" in sites:
            biomass_by_site = tuple(sorted((str(row.site), str(row.biomass_route))
                                           for row in sites.itertuples(index=False)
                                           if str(row.biomass_route)))
            payload["biomass_routes"] = biomass_by_site
            numeric["decision__biomass_route"] = ", ".join(
                f"{site}:{route}" for site, route in biomass_by_site)

    numeric["decision__wind_share"] = wind_share
    numeric["decision__water_reuse_share"] = reuse_share

    # Aggregate route shares over the whole network and by polymer.
    route_share: Dict[str, float] = {}
    polymer_route_payload: List[Tuple[str, str]] = []
    if not routes.empty:
        numeric["gwp_stage__virgin_polymer_production_kgco2e_per_year"] = float(
            routes["virgin_polymer_production_gwp_kgco2e_per_year"].sum()
        )
        numeric["gwp_stage__sut_component_manufacturing_kgco2e_per_year"] = float(
            routes["sut_component_manufacturing_gwp_kgco2e_per_year"].sum()
        )
        numeric["gwp_stage__avoided_virgin_material_credit_kgco2e_per_year"] = float(
            routes["avoided_virgin_material_credit_gwp_kgco2e_per_year"].sum()
        )
        total_t = float(routes["tonnes_per_year"].sum())
        for route, group in routes.groupby("route"):
            share = float(group["tonnes_per_year"].sum()) / total_t if total_t > 0 else 0.0
            route_share[str(route)] = share
            numeric[f"decision__route_share__{route}"] = share
        for polymer, pgroup in routes.groupby("polymer"):
            ptotal = float(pgroup["tonnes_per_year"].sum())
            options: List[Tuple[float, str]] = []
            for route, group in pgroup.groupby("route"):
                share = float(group["tonnes_per_year"].sum()) / ptotal if ptotal > 0 else 0.0
                numeric[f"decision__polymer_route_share__{polymer}__{route}"] = share
                if share >= 0.05:
                    options.append((share, str(route)))
            options.sort(reverse=True)
            if not options:
                state = "none"
            elif len(options) == 1 and options[0][0] >= 0.95:
                state = options[0][1]
            else:
                state = "+".join(sorted(route for _, route in options))
            polymer_route_payload.append((str(polymer), state))
    payload["polymer_routes"] = polymer_route_payload

    # Aggregate transport choice by actual vehicle-km, not by tonnes.
    bev_share = 0.0
    if mix is not None and not mix.empty and "vehicle_km_per_year" in mix.columns:
        by_mode = mix.groupby("transport_mode")["vehicle_km_per_year"].sum()
        total_km = float(by_mode.sum())
        bev_km = float(by_mode.get("battery_electric_articulated_hgv", 0.0))
        bev_share = bev_km / total_km if total_km > 1e-12 else 0.0
    payload["transport"] = classify_share(bev_share, "diesel", "battery-electric", "mixed HGV")
    numeric["decision__bev_vehicle_km_share"] = bev_share
    numeric["decision__diesel_vehicle_km_share"] = 1.0 - bev_share

    digest, canonical = canonical_hash(payload)
    route_text = ", ".join(f"{p}:{r}" for p, r in polymer_route_payload)
    snapshot = (
        f"low node={payload.get('low_batch_location','')}; wind={100*wind_share:.1f}%; "
        f"water reuse={100*reuse_share:.1f}%; BEV-km={100*bev_share:.1f}%; "
        f"biomass={numeric.get('decision__biomass_route', '')}; {route_text}"
    )
    return digest, canonical, snapshot, numeric


# =============================================================================
# Solve one uncertainty realization
# =============================================================================
def solve_realisation(
    evaluation_id: int,
    sobol_index: int,
    values: Mapping[str, float],
    registry_specs: Sequence[Any],
    custom_metas: Sequence[ParameterMeta],
    core: Any,
    master: Any,
    central_data: Any,
    impact: Any,
    solver: Any,
    config: Mapping[str, Any],
    fixed_carbon_value_gbp_per_t: float,
    efw_export_kwh_per_kg: float,
    biomass_reference: Mapping[str, Any],
) -> Dict[str, Any]:
    state = core.clone_state(central_data, impact)
    started = time.perf_counter()
    row: Dict[str, Any] = {
        "evaluation_id": evaluation_id,
        "sobol_index": sobol_index,
        "analysis_case": "BASE",
        "success": False,
        "status": "",
        "solve_seconds": np.nan,
        "failure_class": "",
        "error": "",
    }
    row.update({pid: float(v) for pid, v in values.items()})

    try:
        # Ordering is deliberate: throughput before site-allocation ratio, and
        # total plastic before composition closure.
        for spec in registry_specs:
            if spec.parameter_id in values:
                spec.apply(state, float(values[spec.parameter_id]))
        apply_custom_parameters(state, central_data, values, custom_metas, config)
        couple_electricity_prices(state, central_data, values, config)
        apply_biomass_steam_efficiency(state, central_data, values, biomass_reference)
        row["grid_price_gbp2026_per_mwh"] = 1000.0 * state.data.electricity_cost_gbp_per_kwh["grid"]
        row["cppa_price_gbp2026_per_mwh"] = 1000.0 * state.data.electricity_cost_gbp_per_kwh["wind"]
        row["biomass_useful_steam_mj_per_kg"] = float(
            state.data.biomass_useful_heat_mj_per_kg["ad_external_steam"])

        # Defensive reset after all sample mutations: even if a future shared
        # mutation layer touches carbon value indirectly, the BASE Sobol ensemble
        # remains on the master central cost + carbon objective.
        enforce_central_carbon_value(state, fixed_carbon_value_gbp_per_t)
        row.update(structural_precheck(state))
        state.controls.efw_grid_credit_kwh_per_kg = efw_export_kwh_per_kg
        core.sync_gwp_coefficients(state, impact)

        model = master.build_linear_model(state.data)
        result = solver.solve_backend(
            model,
            case_id=f"SO_BASE_{sobol_index:05d}",
            metadata={
                "analysis": "sobol_qmc_solution_space",
                "sobol_index": sobol_index,
                "analysis_case": "BASE",
            },
        )
        elapsed = time.perf_counter() - started
        row["solve_seconds"] = elapsed
        row["status"] = str(result.message)
        if not result.success:
            row["failure_class"] = classify_nonoptimal_message(result.message)
            row["error"] = str(result.message)
            return row

        metrics = core.evaluate_extended_impacts(master, model, result, state, impact)
        detailed_sig, detailed_snapshot = core.decision_signature(master, model, result)
        macro_hash, macro_json, macro_snapshot, decision_values = macro_decision_regime(master, model, result)
        validation = master.solution_validation(model, result)
        composition = realised_composition(state)

        row.update({metric: metrics.get(metric, np.nan) for metric in OUTPUT_METRICS})
        row["policy_objective_gbp_reconstructed"] = (
            float(metrics["cost"]) + float(state.data.lifecycle_carbon_value_gbp_per_t) * float(metrics["gwp"]) / 1000.0
        )
        row["lifecycle_carbon_value_realised"] = float(state.data.lifecycle_carbon_value_gbp_per_t)
        row["carbon_value_policy"] = CARBON_VALUE_POLICY_ID
        row["success"] = True
        row["failure_class"] = ""
        row["decision_signature_detailed"] = detailed_sig["hash"]
        row["decision_signature_detailed_json"] = detailed_sig["json"]
        row["decision_snapshot_detailed"] = detailed_snapshot
        row["macro_regime"] = macro_hash
        row["macro_regime_json"] = macro_json
        row["macro_regime_snapshot"] = macro_snapshot
        row.update(decision_values)
        for polymer, share in composition.items():
            row[f"realised_composition_share__{polymer}"] = share
        for key, value in validation.items():
            if isinstance(value, (int, float, np.number)):
                row[f"validation__{key}"] = value
        return row
    except Exception as exc:
        row["solve_seconds"] = time.perf_counter() - started
        row["status"] = "exception"
        row["failure_class"] = "technical_exception"
        row["error"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        return row


def solve_central(
    core: Any, master: Any, central_data: Any, impact: Any, solver: Any,
    fixed_carbon_value_gbp_per_t: float,
) -> Dict[str, Any]:
    state = core.clone_state(central_data, impact)
    enforce_central_carbon_value(state, fixed_carbon_value_gbp_per_t)
    core.sync_gwp_coefficients(state, impact)
    model = master.build_linear_model(state.data)
    started = time.perf_counter()
    result = solver.solve_backend(model, "SO_CENTRAL", {"analysis": "central_reference"})
    elapsed = time.perf_counter() - started
    require(result.success, f"Central reference solve failed: {result.message}")
    metrics = core.evaluate_extended_impacts(master, model, result, state, impact)
    detailed_sig, detailed_snapshot = core.decision_signature(master, model, result)
    macro_hash, macro_json, macro_snapshot, decisions = macro_decision_regime(master, model, result)
    row: Dict[str, Any] = {
        "evaluation_id": 0,
        "sobol_index": -1,
        "analysis_case": "CENTRAL",
        "success": True,
        "status": str(result.message),
        "solve_seconds": elapsed,
        "decision_signature_detailed": detailed_sig["hash"],
        "decision_signature_detailed_json": detailed_sig["json"],
        "decision_snapshot_detailed": detailed_snapshot,
        "macro_regime": macro_hash,
        "macro_regime_json": macro_json,
        "macro_regime_snapshot": macro_snapshot,
        **metrics,
        **decisions,
    }
    row["policy_objective_gbp_reconstructed"] = (
        float(metrics["cost"]) + float(state.data.lifecycle_carbon_value_gbp_per_t) * float(metrics["gwp"]) / 1000.0
    )
    row["lifecycle_carbon_value_realised"] = float(state.data.lifecycle_carbon_value_gbp_per_t)
    row["carbon_value_policy"] = CARBON_VALUE_POLICY_ID
    for polymer, share in realised_composition(state).items():
        row[f"realised_composition_share__{polymer}"] = share
    return row


# =============================================================================
# Post-processing summaries (descriptive, not sensitivity indices)
# =============================================================================
def regime_summary(evaluations: pd.DataFrame) -> pd.DataFrame:
    base = evaluations.loc[evaluations["analysis_case"].eq("BASE") & evaluations["success"]].copy()
    if base.empty:
        return pd.DataFrame()
    counts = base["macro_regime"].value_counts()
    rows: List[Dict[str, Any]] = []
    for signature, count in counts.items():
        group = base.loc[base["macro_regime"].eq(signature)]
        row: Dict[str, Any] = {
            "macro_regime": signature,
            "sample_count": int(count),
            "sample_share": float(count / len(base)),
            "representative_snapshot": str(group["macro_regime_snapshot"].mode().iloc[0]),
        }
        for metric in OUTPUT_METRICS:
            x = pd.to_numeric(group[metric], errors="coerce").dropna()
            if len(x):
                row.update({
                    f"{metric}__min": x.min(), f"{metric}__p05": x.quantile(0.05),
                    f"{metric}__median": x.median(), f"{metric}__p95": x.quantile(0.95),
                    f"{metric}__max": x.max(),
                })
        for col in [c for c in group.columns if c.startswith("decision__") and pd.api.types.is_numeric_dtype(group[c])]:
            row[f"median__{col}"] = pd.to_numeric(group[col], errors="coerce").median()
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["sample_count", "macro_regime"], ascending=[False, True]).reset_index(drop=True)
    out.insert(0, "regime_label", [f"R{i}" for i in range(1, len(out) + 1)])
    out["cumulative_sample_share"] = out["sample_share"].cumsum()
    return out


def outcome_summary(evaluations: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for analysis_case, group in evaluations.loc[evaluations["analysis_case"].ne("CENTRAL")].groupby("analysis_case", sort=False):
        feasible = group.loc[group["success"]]
        row: Dict[str, Any] = {
            "analysis_case": analysis_case,
            "n_attempted": len(group),
            "n_feasible": int(group["success"].sum()),
            "feasible_sample_share": float(group["success"].mean()) if len(group) else np.nan,
            "n_macro_regimes": int(feasible["macro_regime"].nunique()) if len(feasible) else 0,
        }
        for metric in OUTPUT_METRICS:
            x = pd.to_numeric(feasible[metric], errors="coerce").dropna()
            if len(x):
                for name, q in (("min", 0.0), ("p05", 0.05), ("p25", 0.25), ("median", 0.5),
                                ("p75", 0.75), ("p95", 0.95), ("max", 1.0)):
                    row[f"{metric}__{name}"] = float(x.quantile(q))
        rows.append(row)
    return pd.DataFrame(rows)


def convergence_summary(evaluations: pd.DataFrame, base_power: int, central_regime: str) -> pd.DataFrame:
    """Nested power-of-two stability diagnostics for the single final Sobol run.

    The central-regime share is the fraction of feasible Sobol realizations whose
    optimal macro decision regime exactly matches the central deterministic optimum.
    It is a share of the bounded Sobol uncertainty space, not an empirical probability.
    """
    base = evaluations.loc[evaluations["analysis_case"].eq("BASE")].sort_values("sobol_index")
    rows: List[Dict[str, Any]] = []
    start_power = min(base_power, 5)
    for power in range(start_power, base_power + 1):
        n = 2 ** power
        sub = base.loc[base["sobol_index"] < n]
        feasible = sub.loc[sub["success"]]
        technical_series = sub.apply(lambda r: is_technical_exception_row(r), axis=1) if len(sub) else pd.Series(dtype=bool)
        failure_series = sub.get("failure_class", pd.Series("", index=sub.index)).fillna("").astype(str).str.lower()
        n_infeasible = int(failure_series.eq("infeasible").sum()) if len(sub) else 0
        row: Dict[str, Any] = {
            "sample_power": power,
            "n_attempted": len(sub),
            "n_feasible": int(sub["success"].sum()),
            "n_infeasible": n_infeasible,
            "n_technical_failures": int(technical_series.sum()) if len(sub) else 0,
            "n_valid_nonoptimal": int((~sub["success"].astype(bool) & ~technical_series).sum()) if len(sub) else 0,
            "feasible_sample_share": float(sub["success"].mean()) if len(sub) else np.nan,
            "infeasible_sample_share": float(n_infeasible / len(sub)) if len(sub) else np.nan,
            "n_macro_regimes_discovered": int(feasible["macro_regime"].nunique()) if len(feasible) else 0,
            "central_regime": central_regime,
        }
        if len(feasible):
            vc = feasible["macro_regime"].value_counts(normalize=True)
            row["largest_regime_share"] = float(vc.iloc[0])
            row["top5_regime_coverage"] = float(vc.head(5).sum())
            central_count = int(feasible["macro_regime"].astype(str).eq(str(central_regime)).sum())
            row["central_regime_count"] = central_count
            row["central_regime_share"] = float(central_count / len(feasible))
            row["central_regime_is_largest"] = bool(len(vc) and str(vc.index[0]) == str(central_regime))
        else:
            row["central_regime_count"] = 0
            row["central_regime_share"] = np.nan
            row["central_regime_is_largest"] = False
        for metric in OUTPUT_METRICS:
            x = pd.to_numeric(feasible[metric], errors="coerce").dropna()
            if len(x):
                row[f"{metric}__p05"] = float(x.quantile(0.05))
                row[f"{metric}__median"] = float(x.median())
                row[f"{metric}__p95"] = float(x.quantile(0.95))
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["central_regime_share_pct"] = 100.0 * out["central_regime_share"]
    out["central_regime_change_vs_previous_pp"] = out["central_regime_share_pct"].diff()
    final_share_pct = float(out["central_regime_share_pct"].iloc[-1])
    out["central_regime_abs_difference_vs_final_pp"] = (out["central_regime_share_pct"] - final_share_pct).abs()
    out["final_central_regime_share_pct_reference"] = final_share_pct
    return out


def regime_parameter_fingerprints(evaluations: pd.DataFrame, metas: Sequence[ParameterMeta]) -> pd.DataFrame:
    base = evaluations.loc[evaluations["analysis_case"].eq("BASE") & evaluations["success"]].copy()
    if base.empty:
        return pd.DataFrame()
    counts = base["macro_regime"].value_counts()
    labels = {sig: f"R{i}" for i, sig in enumerate(counts.index, start=1)}
    rows: List[Dict[str, Any]] = []
    meta_by = {m.parameter_id: m for m in metas}
    for signature, group in base.groupby("macro_regime"):
        for pid, meta in meta_by.items():
            if pid not in group.columns:
                continue
            median = pd.to_numeric(group[pid], errors="coerce").median()
            if not math.isfinite(float(median)) or meta.high <= meta.low:
                continue
            position = (float(median) - meta.low) / (meta.high - meta.low)
            rows.append({
                "regime_label": labels.get(signature, signature),
                "macro_regime": signature,
                "regime_sample_count": len(group),
                "parameter_id": pid,
                "parameter_label": meta.label,
                "parameter_group": meta.group,
                "median_value": median,
                "normalized_median_position": position,
                "shift_from_uncertainty_midpoint": position - 0.5,
                "note": "Descriptive conditional location only; not a Sobol index or causal sensitivity measure.",
            })
    return pd.DataFrame(rows)


def infeasibility_parameter_profile(evaluations: pd.DataFrame, metas: Sequence[ParameterMeta]) -> pd.DataFrame:
    """Descriptive contrast of where feasible vs infeasible samples sit in each bound.

    This is intentionally *not* a causal sensitivity ranking.  It is a diagnostic
    profile to help interpret structural infeasibility in the sampled solution space.
    """
    base = evaluations.loc[evaluations["analysis_case"].eq("BASE")].copy()
    feasible = base.loc[base["success"].astype(bool)]
    failure = base.get("failure_class", pd.Series("", index=base.index)).fillna("").astype(str).str.lower()
    infeasible = base.loc[failure.eq("infeasible")]
    rows: List[Dict[str, Any]] = []
    for meta in metas:
        if meta.parameter_id not in base.columns or meta.high <= meta.low:
            continue
        row: Dict[str, Any] = {
            "parameter_id": meta.parameter_id,
            "parameter_label": meta.label,
            "parameter_group": meta.group,
            "sampling_low": meta.low,
            "sampling_high": meta.high,
            "n_feasible": len(feasible),
            "n_infeasible": len(infeasible),
        }
        for label, group in (("feasible", feasible), ("infeasible", infeasible)):
            x = pd.to_numeric(group[meta.parameter_id], errors="coerce").dropna()
            if len(x):
                row[f"{label}__p25"] = float(x.quantile(0.25))
                row[f"{label}__median"] = float(x.median())
                row[f"{label}__p75"] = float(x.quantile(0.75))
                row[f"{label}__normalized_median"] = float((x.median() - meta.low) / (meta.high - meta.low))
        if "feasible__normalized_median" in row and "infeasible__normalized_median" in row:
            row["normalized_median_shift_infeasible_minus_feasible"] = (
                row["infeasible__normalized_median"] - row["feasible__normalized_median"]
            )
            row["absolute_normalized_median_shift"] = abs(row["normalized_median_shift_infeasible_minus_feasible"])
        row["interpretation_note"] = "Descriptive feasible/infeasible location contrast; not a Sobol index or causal attribution."
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty and "absolute_normalized_median_shift" in out.columns:
        out = out.sort_values("absolute_normalized_median_shift", ascending=False, na_position="last")
    return out.reset_index(drop=True)


def feasibility_summary_table(evaluations: pd.DataFrame) -> pd.DataFrame:
    base = evaluations.loc[evaluations["analysis_case"].eq("BASE")].copy()
    failure = base.get("failure_class", pd.Series("", index=base.index)).fillna("").astype(str).str.lower()
    n = len(base)
    feasible = int(base["success"].astype(bool).sum())
    infeasible = int(failure.eq("infeasible").sum())
    nonoptimal = int((~base["success"].astype(bool) & ~failure.isin(["infeasible", "technical_exception"])).sum())
    technical = int(failure.eq("technical_exception").sum())
    return pd.DataFrame([{
        "n_sobol_samples": n,
        "n_feasible_optimal": feasible,
        "feasible_share": feasible / n if n else np.nan,
        "n_genuine_infeasible": infeasible,
        "infeasible_share": infeasible / n if n else np.nan,
        "n_other_solver_nonoptimal": nonoptimal,
        "n_technical_exceptions": technical,
        "scientifically_accounted_share": (feasible + infeasible + nonoptimal) / n if n else np.nan,
    }])


# =============================================================================
# CLI / main
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sobol QMC MILP solution-space assessment under bounded uncertainty")
    p.add_argument("--backend", choices=["gams", "scipy"], default=None)
    p.add_argument("--sample-power", type=int, default=None, help="Base Sobol power m, N=2^m")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--result-dir", type=Path, default=None)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--shared-module", type=Path, default=None)
    p.add_argument("--only", type=str, default="", help="Comma-separated continuous parameter IDs for a reduced run")
    p.add_argument("--quick", action="store_true", help="Smoke test: N=32 Sobol realizations")
    p.add_argument("--resume", action="store_true", help="Resume an interrupted run from evaluations_checkpoint.csv")
    p.add_argument(
        "--gams-restart-every", type=int, default=GAMS_PROCESS_RESTART_EVERY,
        help="For GAMS only: recycle the Python process after this many newly evaluated Sobol samples (0 disables).",
    )
    p.add_argument("--tee", action="store_true")
    return p.parse_args()


def main() -> Dict[str, Path]:
    args = parse_args()
    sample_power = 5 if args.quick else (args.sample_power or SOBOL_SAMPLE_POWER)
    seed = args.seed if args.seed is not None else SOBOL_SEED

    shared_path = locate_shared_module(args.shared_module)
    core = import_module_from_path(shared_path, "stochastic_solution_space_shared")
    master_path = core.locate_master_file()
    master = core.import_master_module(master_path)
    config_path, config = load_config(args.config)

    result_dir = (args.result_dir or DEFAULT_RESULT_DIR).expanduser().resolve()

    # Guard against accidentally writing generated outputs into the source
    # input/config directory. This was the cause of SameFileError when the
    # config itself was treated as the output snapshot destination.
    if args.result_dir is not None and result_dir == config_path.parent.resolve():
        raise ValueError(
            f"--result-dir resolves to the parameter input directory: {result_dir}. "
            "Choose a separate results directory (for example ../results/stochastic_optimisation_solution_space)."
        )

    checkpoint_path = result_dir / "evaluations_checkpoint.csv"
    if result_dir.exists() and not args.resume and CLEAR_PREVIOUS_RESULTS:
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    central_data = master.load_model_data()
    fixed_carbon_value_gbp_per_t = central_carbon_value_2026_gbp(central_data)
    require(
        OBJECTIVE == "policy_cost",
        f"This stochastic thesis runner must use policy_cost; configured objective is {OBJECTIVE!r}.",
    )
    impact = core.load_impact_model(master, central_data, core.find_lca_settings())
    efw_export_kwh_per_kg = efw_export_kwh_per_kg_from_lca(master, impact)
    biomass_reference = biomass_steam_credit_reference(master, central_data, config)

    requested = {x.strip() for x in args.only.split(",") if x.strip()} or None
    if requested is not None and FIXED_CARBON_VALUE_PARAMETER_ID in requested:
        raise ValueError(
            "lifecycle_carbon_value is intentionally fixed at the master central 2026-GBP value in this "
            "stochastic uncertainty analysis. Test low/high/no-carbon values in the explicit scenario or OAT analysis instead."
        )
    if FIXED_CARBON_VALUE_PARAMETER_ID in config["analysis"].get("registry_parameter_ids", []):
        print(
            "NOTE: lifecycle_carbon_value is listed in the stochastic JSON but is intentionally excluded "
            "from Sobol sampling; the master central 2026-GBP value is fixed for every solve.",
            flush=True,
        )
    registry_specs, registry_metas = resolve_registry_parameters(core, central_data, impact, config, requested)
    custom_metas = resolve_custom_parameters(config, requested)
    metas = registry_metas + custom_metas
    require(metas, "No continuous uncertainty parameters selected.")
    validate_sampling_ranges(metas, config)
    known = {m.parameter_id for m in metas}
    if requested is not None:
        missing_requested = requested - known
        require(not missing_requested, f"--only references unknown or unavailable parameter IDs: {sorted(missing_requested)}")

    # Preserve source config and exact runtime-resolved ranges. Never attempt
    # to copy a file onto itself (e.g. after a custom directory layout).
    config_snapshot_path = (result_dir / PARAMETER_CONFIG_FILENAME).resolve()
    if config_path.resolve() != config_snapshot_path:
        shutil.copy2(config_path, config_snapshot_path)
    parameter_table(metas, config).to_csv(result_dir / "parameter_ranges_resolved.csv", index=False)
    write_json(result_dir / "resolved_parameter_configuration.json", {
        "analysis_type": "Sobol quasi-Monte-Carlo solution-space assessment under bounded uncertainty",
        "important_method_note": (
            "Sobol refers to the low-discrepancy sampling sequence. No Sobol sensitivity indices, "
            "Saltelli cross-samples, or Jansen estimators are calculated. Each sample is an independent deterministic MILP solve."
        ),
        "sampling_measure_note": (
            "Scrambled Sobol points are mapped through triangular inverse CDFs, using nominal values as modes. "
            "These judgement-based distributions are not calibrated empirical probabilities."
        ),
        "sample_power": sample_power,
        "base_sample_size": 2 ** sample_power,
        "seed": seed,
        "objective": OBJECTIVE,
        "carbon_value_policy": CARBON_VALUE_POLICY_ID,
        "fixed_central_carbon_value_nominal_2026_gbp_per_tco2e": fixed_carbon_value_gbp_per_t,
        "carbon_value_sampled": False,
        "efw_avoided_grid_electricity_kwh_per_kg_plastic": efw_export_kwh_per_kg,
        "efw_credit_grid_linkage": "Only the workbook EfW avoided-electricity GWP credit follows sampled grid GWP; gross process and heat credit remain fixed.",
        "excluded_from_sobol_dimensions": [FIXED_CARBON_VALUE_PARAMETER_ID],
        "selected_parameters": [m.__dict__ for m in metas],
    })

    design = generate_sobol_design(metas, sample_power, seed)
    require(
        FIXED_CARBON_VALUE_PARAMETER_ID not in design.columns
        and f"u__{FIXED_CARBON_VALUE_PARAMETER_ID}" not in design.columns,
        "Internal error: lifecycle carbon value leaked into the Sobol design.",
    )
    design.to_csv(result_dir / "sobol_qmc_design.csv", index=False)


    # Immutable run identity: prevents --resume from silently mixing samples from
    # a different parameter set, power, seed, objective or backend.
    config_hash, _ = canonical_hash(config, n=20)
    run_identity_payload = {
        "config_hash": config_hash,
        "sample_power": sample_power,
        "seed": seed,
        "parameter_ids": [m.parameter_id for m in metas],
        "objective": OBJECTIVE,
        "carbon_value_policy": CARBON_VALUE_POLICY_ID,
        "fixed_central_carbon_value_nominal_2026_gbp_per_tco2e": fixed_carbon_value_gbp_per_t,
        "carbon_value_sampled": False,
        "backend": args.backend or SOLVER_BACKEND,
    }
    run_identity_hash, _ = canonical_hash(run_identity_payload, n=20)
    run_plan = {**run_identity_payload, "run_identity_hash": run_identity_hash}
    run_plan_path = result_dir / "run_plan.json"
    if args.resume and run_plan_path.exists():
        previous = read_json(run_plan_path)
        require(
            previous.get("run_identity_hash") == run_identity_hash,
            "Refusing to resume: current configuration/sampling plan differs from the existing run_plan.json. "
            "Use a new result directory or rerun without --resume.",
        )
    else:
        write_json(run_plan_path, run_plan)

    # Planned evaluation keys: the full nested Sobol sequence only.  Structural
    # route-acceptance/outage stress tests are intentionally excluded from this
    # final thesis uncertainty run; effort is concentrated on continuous process,
    # composition, recovery, market and environmental uncertainty.
    plans: List[int] = [int(i) for i in design["sobol_index"]]

    # Resume is deliberately selective. A genuine solver result such as
    # "infeasible" is a valid sampled outcome and is retained. A Python/GAMS
    # technical exception is NOT a scientific outcome and is discarded/retried.
    existing_rows: List[Dict[str, Any]] = []
    completed: set[int] = set()
    discarded_technical = 0
    if args.resume and checkpoint_path.exists():
        old = pd.read_csv(checkpoint_path)
        reusable_by_index: Dict[int, Dict[str, Any]] = {}
        for raw in old.to_dict(orient="records"):
            if str(raw.get("analysis_case", "BASE")) != "BASE":
                continue
            idx = int(raw["sobol_index"])
            if is_technical_exception_row(raw):
                discarded_technical += 1
                continue
            reusable_by_index[idx] = raw
        existing_rows = [reusable_by_index[i] for i in sorted(reusable_by_index)]
        completed = set(reusable_by_index)
        print(f"Resuming from {len(completed):,} scientifically completed evaluation records.")
        if discarded_technical:
            print(f"  Retrying {discarded_technical:,} prior technical exception record(s); they are not treated as completed samples.")

    total_planned = len(plans)
    pending = [i for i in plans if i not in completed]
    rows = list(existing_rows)

    solver = core.ScenarioSolver(
        master=master,
        central_data=central_data,
        impact=impact,
        result_dir=result_dir,
        backend=args.backend or SOLVER_BACKEND,
        objective=OBJECTIVE,
        tee=args.tee or GAMS_TEE,
    )

    # Keep GAMS log numbering unique across automatic process recycles.
    if (args.backend or SOLVER_BACKEND) == "gams" and completed:
        try:
            master._GAMS_SOLVE_COUNTER = len(completed) + 1  # +1 for the central reference solve
        except Exception:
            pass

    central_path = result_dir / "central_solution.csv"
    if args.resume and central_path.exists():
        central_row = pd.read_csv(central_path).iloc[0].to_dict()
        print("Reusing existing central deterministic reference solution.")
    else:
        central_row = solve_central(
            core, master, central_data, impact, solver, fixed_carbon_value_gbp_per_t
        )
        pd.DataFrame([central_row]).to_csv(central_path, index=False)
    require(
        math.isclose(
            safe_float(central_row.get("lifecycle_carbon_value_realised")),
            fixed_carbon_value_gbp_per_t, rel_tol=0.0, abs_tol=1e-9,
        ),
        "Central reference solution does not use the master fixed central carbon value. "
        "Delete the old result directory or rerun without --resume.",
    )
    if hasattr(impact, "audit"):
        impact.audit.to_csv(result_dir / "lcia_assumptions_audit.csv", index=False)
    solve_times = [safe_float(r.get("solve_seconds")) for r in rows if safe_float(r.get("solve_seconds")) >= 0]
    print("\nSOBOL QMC SOLUTION-SPACE ASSESSMENT")
    print(f"  Continuous dimensions D : {len(metas)}")
    print(f"  Base Sobol samples N     : {2 ** sample_power}")
    print(f"  Planned MILP evaluations : {total_planned} (+ 1 central reference)")
    print(f"  Solver backend           : {solver.backend}")
    print(f"  Objective                : cost + fixed central carbon value × GWP")
    print(f"  Central carbon value     : £{fixed_carbon_value_gbp_per_t:,.6f}/tCO2e (nominal GBP 2026; inherited from master)")
    print(f"  Carbon value sampled?    : NO")
    if solver.backend == "gams" and args.gams_restart_every > 0:
        print(f"  GAMS process recycle     : every {args.gams_restart_every} newly evaluated samples")
    print(f"  Results                  : {result_dir}\n")

    retry_state_path = result_dir / "technical_retry_counts.json"
    retry_counts: Dict[str, int] = read_json(retry_state_path) if retry_state_path.exists() else {}
    newly_evaluated_this_process = 0

    for seq, sobol_index in enumerate(pending, start=1):
        design_row = design.loc[design["sobol_index"].eq(sobol_index)].iloc[0]
        values = {m.parameter_id: float(design_row[m.parameter_id]) for m in metas}
        record = solve_realisation(
            evaluation_id=sobol_index + 1,
            sobol_index=sobol_index,
            values=values,
            registry_specs=registry_specs,
            custom_metas=custom_metas,
            core=core,
            master=master,
            central_data=central_data,
            impact=impact,
            solver=solver,
            config=config,
            fixed_carbon_value_gbp_per_t=fixed_carbon_value_gbp_per_t,
            efw_export_kwh_per_kg=efw_export_kwh_per_kg,
            biomass_reference=biomass_reference,
        )
        rows.append(record)
        newly_evaluated_this_process += 1
        if math.isfinite(safe_float(record.get("solve_seconds"))):
            solve_times.append(float(record["solve_seconds"]))

        # A technical exception is never accepted as an uncertainty outcome.
        # For GAMS shell failures, save the audit record, recycle the interpreter,
        # and retry exactly the same Sobol index from the checkpoint.
        if is_technical_exception_row(record):
            pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
            key = str(sobol_index)
            retry_counts[key] = int(retry_counts.get(key, 0)) + 1
            write_json(retry_state_path, retry_counts)
            error_tail = str(record.get("error", ""))[-2500:]
            if solver.backend == "gams" and is_gams_solver_exception(record) and retry_counts[key] <= MAX_GAMS_TECHNICAL_RETRIES_PER_SAMPLE:
                restart_current_process_for_resume(
                    f"technical GAMS/Pyomo exception at Sobol sample {sobol_index}; "
                    f"retry {retry_counts[key]}/{MAX_GAMS_TECHNICAL_RETRIES_PER_SAMPLE}"
                )
            raise RuntimeError(
                f"Technical exception at Sobol sample {sobol_index}. The run was stopped rather than "
                f"silently treating it as an uncertainty result. Check evaluations_checkpoint.csv and GAMS logs.\n"
                f"Last error details:\n{error_tail}"
            )

        done = len(completed) + seq
        mean_s = float(np.mean(solve_times[-200:])) if solve_times else np.nan
        eta_s = mean_s * max(total_planned - done, 0) if math.isfinite(mean_s) else np.nan
        ok = "ok" if record.get("success") else str(record.get("failure_class", "nonoptimal")).upper()
        regimes = len({r.get("macro_regime") for r in rows if coerce_bool(r.get("success")) and r.get("macro_regime")})
        print(
            f"[{done:4d}/{total_planned} | {100*done/total_planned:5.1f}%] "
            f"BASE #{sobol_index:04d} {ok:<12s} "
            f"{safe_float(record.get('solve_seconds')):6.2f}s | mean {mean_s:5.2f}s | "
            f"ETA {finish_time_text(eta_s)} | regimes {regimes}",
            flush=True,
        )
        if done % CHECKPOINT_EVERY == 0 or seq == len(pending):
            pd.DataFrame(rows).to_csv(checkpoint_path, index=False)

        # Proactive GAMS/Pyomo process recycling.  The checkpoint makes this an
        # uninterrupted logical run; only the interpreter is replaced.
        if (
            solver.backend == "gams"
            and args.gams_restart_every > 0
            and newly_evaluated_this_process >= args.gams_restart_every
            and done < total_planned
        ):
            pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
            restart_current_process_for_resume(
                f"planned resource reset after {newly_evaluated_this_process} GAMS solves in this interpreter"
            )

    evaluations = pd.DataFrame(rows)
    if "success" in evaluations.columns:
        evaluations["success"] = evaluations["success"].map(coerce_bool)
    # Stable Sobol ordering supports nested power-of-two convergence diagnostics.
    evaluations = evaluations.sort_values("sobol_index").reset_index(drop=True)
    evaluations.to_csv(result_dir / "evaluations.csv", index=False)
    evaluations.loc[~evaluations["success"].astype(bool)].to_csv(result_dir / "failed_evaluations.csv", index=False)
    failure_class = evaluations.get("failure_class", pd.Series("", index=evaluations.index)).fillna("").astype(str).str.lower()
    evaluations.loc[failure_class.eq("infeasible")].to_csv(result_dir / "infeasible_evaluations.csv", index=False)
    evaluations.loc[failure_class.eq("technical_exception")].to_csv(result_dir / "technical_exception_evaluations.csv", index=False)

    technical_mask = evaluations.apply(lambda r: is_technical_exception_row(r), axis=1)
    technical_failure_count = int(technical_mask.sum())
    infeasible_count = int(failure_class.eq("infeasible").sum())
    other_nonoptimal_count = int((~evaluations["success"].astype(bool) & ~technical_mask & ~failure_class.eq("infeasible")).sum())
    valid_nonoptimal_count = infeasible_count + other_nonoptimal_count
    quality = pd.DataFrame([{
        "planned_sobol_samples": total_planned,
        "records_present": len(evaluations),
        "feasible_optimal_samples": int(evaluations["success"].astype(bool).sum()),
        "genuine_infeasible_samples": infeasible_count,
        "other_valid_solver_nonoptimal_samples": other_nonoptimal_count,
        "valid_nonoptimal_or_infeasible_samples": valid_nonoptimal_count,
        "technical_exception_samples": technical_failure_count,
        "all_planned_samples_scientifically_accounted_for": bool(len(evaluations) == total_planned and technical_failure_count == 0),
        "thesis_run_valid": bool(len(evaluations) == total_planned and technical_failure_count == 0),
    }])
    quality.to_csv(result_dir / "run_quality_summary.csv", index=False)
    if technical_failure_count:
        invalid_note = result_dir / "RUN_INVALID_TECHNICAL_FAILURES.txt"
        invalid_note.write_text(
            f"This run contains {technical_failure_count} technical solver/Python exception record(s).\n"
            "These are not valid uncertainty outcomes. Resume with the robust runner; technical exception rows will be retried.\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            f"Run is not scientifically valid: {technical_failure_count} technical exception record(s) remain. "
            f"See {invalid_note}."
        )

    regimes = regime_summary(evaluations)
    regimes.to_csv(result_dir / "decision_regime_summary.csv", index=False)
    outcome_summary(evaluations).to_csv(result_dir / "outcome_distribution_summary.csv", index=False)
    convergence = convergence_summary(evaluations, sample_power, str(central_row["macro_regime"]))
    convergence.to_csv(result_dir / "sampling_convergence.csv", index=False)
    central_cols = [
        "sample_power", "n_attempted", "n_feasible", "n_infeasible", "n_technical_failures", "n_valid_nonoptimal", "n_macro_regimes_discovered",
        "central_regime", "central_regime_count", "central_regime_share",
        "central_regime_share_pct", "central_regime_change_vs_previous_pp",
        "central_regime_abs_difference_vs_final_pp", "central_regime_is_largest",
    ]
    convergence[[c for c in central_cols if c in convergence.columns]].to_csv(
        result_dir / "central_regime_share_convergence.csv", index=False
    )
    regime_parameter_fingerprints(evaluations, metas).to_csv(result_dir / "regime_parameter_fingerprints.csv", index=False)
    infeasibility_parameter_profile(evaluations, metas).to_csv(result_dir / "infeasibility_parameter_profile.csv", index=False)
    feasibility_summary_table(evaluations).to_csv(result_dir / "feasibility_summary.csv", index=False)

    base_ok = evaluations.loc[evaluations["analysis_case"].eq("BASE") & evaluations["success"]].copy()
    if len(base_ok):
        realised_carbon = pd.to_numeric(base_ok["lifecycle_carbon_value_realised"], errors="coerce")
        require(realised_carbon.notna().all(), "Successful Sobol rows contain missing realised carbon values.")
        max_carbon_drift = float((realised_carbon - fixed_carbon_value_gbp_per_t).abs().max())
        require(
            max_carbon_drift <= 1e-9,
            f"Carbon-value invariant failed: maximum drift across successful Sobol solves is {max_carbon_drift} GBP/tCO2e.",
        )
    else:
        max_carbon_drift = 0.0

    manifest = {
        "analysis_type": "Sobol QMC solution-space assessment / sampling-based optimisation under bounded uncertainty",
        "terminology_warning": (
            "This package uses a Sobol sequence for low-discrepancy scenario generation. It does not report Sobol sensitivity indices. "
            "It is also distinct from a coupled two-stage stochastic program because no scenario-independent first-stage decision is shared across samples."
        ),
        "master_model": str(master_path),
        "shared_mutation_layer": str(shared_path),
        "parameter_config": str(config_path),
        "solver_backend": solver.backend,
        "objective": OBJECTIVE,
        "objective_definition": "annual network cost + fixed central lifecycle carbon value * IMPACT World+ v2.1 carbon footprint / 1000",
        "carbon_value_policy": CARBON_VALUE_POLICY_ID,
        "fixed_central_carbon_value_nominal_2026_gbp_per_tco2e": fixed_carbon_value_gbp_per_t,
        "carbon_value_sampled": False,
        "carbon_value_source_of_truth": "master.load_model_data().lifecycle_carbon_value_gbp_per_t",
        "maximum_realised_carbon_value_drift_gbp_per_tco2e": max_carbon_drift,
        "sobol_seed": seed,
        "base_sample_power": sample_power,
        "base_sample_size": 2 ** sample_power,
        "continuous_dimensions": len(metas),
        "total_evaluation_records": len(evaluations),
        "base_feasible_records": int(len(base_ok)),
        "base_valid_nonoptimal_or_infeasible_records": valid_nonoptimal_count,
        "base_technical_exception_records": technical_failure_count,
        "thesis_run_valid": bool(technical_failure_count == 0 and len(evaluations) == total_planned),
        "base_macro_regimes_discovered": int(base_ok["macro_regime"].nunique()) if len(base_ok) else 0,
        "central_macro_regime": str(central_row["macro_regime"]),
        "final_central_regime_share": (
            float(base_ok["macro_regime"].astype(str).eq(str(central_row["macro_regime"])).mean()) if len(base_ok) else np.nan
        ),
        "sampling_measure": "scrambled Sobol low-discrepancy sequence mapped through triangular inverse CDFs with nominal modes",
        "efw_avoided_grid_electricity_kwh_per_kg_plastic": efw_export_kwh_per_kg,
        "probability_interpretation": "Scenario shares describe assumed triangular measures; they are not empirically calibrated event probabilities.",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    write_json(result_dir / "run_manifest.json", manifest)

    # Keep checkpoint for audit/resume; it is intentionally not deleted.
    print("\nCOMPLETE — RUN QUALITY CHECK PASSED")
    print(f"  Scientifically accounted samples : {len(evaluations):,}/{2 ** sample_power:,}")
    print(f"  Technical solver exceptions       : {technical_failure_count}")
    print(f"  Valid infeasible/nonoptimal cases : {valid_nonoptimal_count}")
    print(f"  Feasible base samples             : {len(base_ok):,}/{2 ** sample_power:,}")
    print(f"  Macro regimes found   : {manifest['base_macro_regimes_discovered']}")
    print(f"  Central regime share  : {100*manifest['final_central_regime_share']:.2f}% of feasible Sobol samples")
    print(f"  Fixed carbon value    : £{fixed_carbon_value_gbp_per_t:,.6f}/tCO2e nominal GBP 2026")
    print(f"  Results directory     : {result_dir}")
    return {"result_dir": result_dir, "manifest": result_dir / "run_manifest.json"}


if __name__ == "__main__":
    main()
