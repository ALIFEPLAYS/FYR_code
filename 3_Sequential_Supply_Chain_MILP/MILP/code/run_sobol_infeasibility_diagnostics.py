#!/usr/bin/env python3
"""Repeat a Sobol run to diagnose infeasible evaluations.

Requires the archived stochastic run's design and plan files in the reference
result directory. These generated files are not supplied with this package.
Use --reference-result-dir to select an existing run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name.lower() == "code" else SCRIPT_DIR
DEFAULT_DIAGNOSTIC_DIR = (
    PROJECT_ROOT / "results" / "stochastic_optimisation_infeasibility_diagnostic"
)
DEFAULT_REFERENCE_DIR = (
    PROJECT_ROOT / "results" / "stochastic_optimisation_solution_space"
)
SOURCE_RUNNER_NAME = "run_stochastic_optimisation.py"
CHECKPOINT_EVERY = 25
EXPECTED_INFEASIBLE = 91
# The archived design has passed through CSV text serialisation.  Check the
# unit-cube coordinates very strictly because they define the Sobol design,
# then allow only harmless round-trip error in the transformed physical values.
UNIT_CUBE_ABSOLUTE_TOLERANCE = 5e-15
PHYSICAL_VALUE_ABSOLUTE_TOLERANCE = 1e-10
PHYSICAL_VALUE_RELATIVE_TOLERANCE = 1e-13


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        converted = float(value)
        return converted if math.isfinite(converted) else default
    except Exception:
        return default


def coerce_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def canonical_hash(payload: Mapping[str, Any], length: int = 20) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def finish_time_text(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    finish = datetime.now().astimezone() + timedelta(seconds=float(seconds))
    return finish.strftime("%H:%M %Z")


def import_module(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def locate_source_runner(explicit: Optional[Path]) -> Path:
    candidates: List[Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    candidates.append(SCRIPT_DIR / SOURCE_RUNNER_NAME)
    candidates.extend(sorted(SCRIPT_DIR.glob("run_stochastic_optimisation*.py")))
    current = Path(__file__).resolve()
    required = {
        "generate_sobol_design",
        "resolve_registry_parameters",
        "resolve_custom_parameters",
        "validate_sampling_ranges",
        "apply_custom_parameters",
        "structural_precheck",
        "realised_composition",
        "enforce_central_carbon_value",
        "central_carbon_value_2026_gbp",
        "locate_shared_module",
        "load_config",
    }
    failures: List[str] = []
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            path = candidate.resolve()
        except Exception:
            continue
        if path in seen or path == current or not path.is_file():
            continue
        seen.add(path)
        try:
            module = import_module(path, f"sobol_source_probe_{len(seen)}")
            missing = sorted(required - set(dir(module)))
            if not missing:
                return path
            failures.append(f"{path.name}: missing {missing}")
        except Exception as exc:
            failures.append(f"{path.name}: import failed ({exc})")
    detail = "\n  ".join(failures[:8])
    raise FileNotFoundError(
        f"Could not find a compatible {SOURCE_RUNNER_NAME} beside this script. "
        "Pass --source-runner explicitly if it has another name.\n  " + detail
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeat the original Sobol MILP run and diagnose only infeasible cases."
    )
    parser.add_argument("--source-runner", type=Path)
    parser.add_argument("--shared-module", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--reference-result-dir", type=Path)
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--backend", choices=["gams", "scipy"], default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing diagnostic result directory. Never affects the original result directory.",
    )
    parser.add_argument("--tee", action="store_true")
    parser.add_argument(
        "--gams-restart-every",
        type=int,
        default=None,
        help="Process recycle interval. Defaults to the original runner's setting.",
    )
    parser.add_argument(
        "--expected-infeasible",
        type=int,
        default=EXPECTED_INFEASIBLE,
    )
    parser.add_argument(
        "--keep-feasible-logs",
        action="store_true",
        help="Retain logs from feasible solves. By default only infeasible and failed-case logs are retained.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Development only: solve the first N samples. A limited run is not a final identity check.",
    )
    return parser.parse_args()


def load_reference_identity(reference_dir: Path) -> Tuple[Dict[str, Any], pd.DataFrame]:
    plan_path = reference_dir / "run_plan.json"
    design_path = reference_dir / "sobol_qmc_design.csv"
    require(plan_path.is_file(), f"Original run plan not found: {plan_path}")
    require(design_path.is_file(), f"Original Sobol design not found: {design_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    design = pd.read_csv(design_path)
    require("sobol_index" in design.columns, f"Reference design lacks sobol_index: {design_path}")
    return plan, design


def verify_design_identity(
    regenerated: pd.DataFrame,
    reference: pd.DataFrame,
    parameter_ids: Sequence[str],
) -> Dict[str, Any]:
    expected_columns = [
        "sobol_index",
        *[item for parameter_id in parameter_ids for item in (parameter_id, f"u__{parameter_id}")],
    ]
    missing_regenerated = [column for column in expected_columns if column not in regenerated.columns]
    missing_reference = [column for column in expected_columns if column not in reference.columns]
    require(not missing_regenerated, f"Regenerated design lacks columns: {missing_regenerated}")
    require(not missing_reference, f"Reference design lacks columns: {missing_reference}")
    require(len(regenerated) == len(reference), (
        f"Sobol row-count mismatch: regenerated {len(regenerated)}, reference {len(reference)}"
    ))

    left = regenerated[expected_columns].sort_values("sobol_index").reset_index(drop=True)
    right = reference[expected_columns].sort_values("sobol_index").reset_index(drop=True)
    require(
        np.array_equal(left["sobol_index"].to_numpy(), right["sobol_index"].to_numpy()),
        "Sobol indices differ from the original run.",
    )
    unit_cube_columns = [f"u__{parameter_id}" for parameter_id in parameter_ids]
    physical_columns = list(parameter_ids)
    left_unit_cube = left[unit_cube_columns].to_numpy(dtype=float)
    right_unit_cube = right[unit_cube_columns].to_numpy(dtype=float)
    left_physical = left[physical_columns].to_numpy(dtype=float)
    right_physical = right[physical_columns].to_numpy(dtype=float)
    maximum_unit_cube_difference = (
        float(np.max(np.abs(left_unit_cube - right_unit_cube))) if left_unit_cube.size else 0.0
    )
    maximum_physical_difference = (
        float(np.max(np.abs(left_physical - right_physical))) if left_physical.size else 0.0
    )
    require(
        np.allclose(
            left_unit_cube,
            right_unit_cube,
            rtol=0.0,
            atol=UNIT_CUBE_ABSOLUTE_TOLERANCE,
        ),
        "Regenerated Sobol unit-cube coordinates differ from the original design; "
        f"maximum absolute difference {maximum_unit_cube_difference:.3e}.",
    )
    require(
        np.allclose(
            left_physical,
            right_physical,
            rtol=PHYSICAL_VALUE_RELATIVE_TOLERANCE,
            atol=PHYSICAL_VALUE_ABSOLUTE_TOLERANCE,
        ),
        "Regenerated transformed parameter values differ from the original design beyond "
        "CSV round-trip precision; maximum absolute difference "
        f"{maximum_physical_difference:.3e}.",
    )
    numeric_columns = expected_columns[1:]
    left_values = left[numeric_columns].to_numpy(dtype=float)
    rounded_payload = np.round(left_values, decimals=14).tobytes()
    return {
        "row_count": len(left),
        "parameter_count": len(parameter_ids),
        "columns_compared": expected_columns,
        "maximum_unit_cube_absolute_difference": maximum_unit_cube_difference,
        "maximum_physical_value_absolute_difference": maximum_physical_difference,
        "unit_cube_absolute_tolerance": UNIT_CUBE_ABSOLUTE_TOLERANCE,
        "physical_value_absolute_tolerance": PHYSICAL_VALUE_ABSOLUTE_TOLERANCE,
        "physical_value_relative_tolerance": PHYSICAL_VALUE_RELATIVE_TOLERANCE,
        "design_match": True,
        "regenerated_design_sha1_rounded_14dp": hashlib.sha1(rounded_payload).hexdigest(),
    }


def snapshot_files(directory: Path) -> set[Path]:
    if not directory.exists():
        return set()
    return {path.resolve() for path in directory.rglob("*") if path.is_file()}


def remove_new_feasible_logs(before: set[Path], log_dir: Path) -> None:
    after = snapshot_files(log_dir)
    root = log_dir.resolve()
    for path in sorted(after - before):
        try:
            path.relative_to(root)
        except ValueError:
            continue
        path.unlink(missing_ok=True)


def detailed_structural_precheck(source: Any, state: Any) -> Dict[str, Any]:
    """Extend the original precheck without altering the model or its inputs."""
    output = dict(source.structural_precheck(state))
    data = state.data
    active = [site for site in data.sites if int(data.site_open[site]) == 1]
    output["precheck__active_site_count"] = len(active)
    output["precheck__active_sites"] = "|".join(str(site) for site in active)

    wastewater_per_batch = float(data.wastewater_m3_per_batch)
    high_batches = float(data.batch_high)
    low_batches = float(data.batch_low)
    total_batches = float(data.total_batches)
    expected_batches = high_batches * max(len(active) - 1, 0) + low_batches
    output["precheck__batch_allocation_expected_total"] = expected_batches
    output["precheck__batch_allocation_closure_error"] = expected_batches - total_batches
    output["precheck__batch_allocation_closes"] = bool(
        math.isclose(expected_batches, total_batches, rel_tol=0.0, abs_tol=1e-8)
    )

    allocation_diagnostics: List[Dict[str, Any]] = []
    for low_site in active:
        total_shortfall = 0.0
        maximum_shortfall = 0.0
        maximum_utilisation = 0.0
        violating_sites: List[str] = []
        site_details: List[Dict[str, Any]] = []
        for site in active:
            batches = low_batches if site == low_site else high_batches
            requirement = batches * wastewater_per_batch
            capacity = float(data.wwt_capacity_m3[site])
            shortfall = max(requirement - capacity, 0.0)
            utilisation = requirement / capacity if capacity > 0 else np.inf
            if shortfall > 1e-8:
                violating_sites.append(str(site))
            total_shortfall += shortfall
            maximum_shortfall = max(maximum_shortfall, shortfall)
            maximum_utilisation = max(maximum_utilisation, utilisation)
            site_details.append(
                {
                    "site": str(site),
                    "batches": batches,
                    "wastewater_requirement_m3_per_year": requirement,
                    "installed_wwt_capacity_m3_per_year": capacity,
                    "capacity_utilisation": utilisation,
                    "capacity_shortfall_m3_per_year": shortfall,
                }
            )
            safe_site = re_key(str(site))
            output[f"precheck__wwt_capacity__{safe_site}"] = capacity
            output[f"precheck__wwt_high_allocation_requirement__{safe_site}"] = high_batches * wastewater_per_batch
            output[f"precheck__wwt_low_allocation_requirement__{safe_site}"] = low_batches * wastewater_per_batch
        allocation_diagnostics.append(
            {
                "low_batch_site": str(low_site),
                "feasible": not violating_sites,
                "violating_sites": violating_sites,
                "total_shortfall_m3_per_year": total_shortfall,
                "maximum_site_shortfall_m3_per_year": maximum_shortfall,
                "maximum_capacity_utilisation": maximum_utilisation,
                "site_details": site_details,
            }
        )

    if allocation_diagnostics:
        best = min(
            allocation_diagnostics,
            key=lambda item: (
                float(item["total_shortfall_m3_per_year"]),
                float(item["maximum_capacity_utilisation"]),
                str(item["low_batch_site"]),
            ),
        )
        output["precheck__best_low_batch_site"] = best["low_batch_site"]
        output["precheck__minimum_total_wwt_shortfall_m3_per_year"] = float(
            best["total_shortfall_m3_per_year"]
        )
        output["precheck__minimum_maximum_site_wwt_shortfall_m3_per_year"] = float(
            best["maximum_site_shortfall_m3_per_year"]
        )
        output["precheck__best_wwt_capacity_utilisation"] = float(
            best["maximum_capacity_utilisation"]
        )
        output["precheck__best_allocation_violating_sites"] = "|".join(
            best["violating_sites"]
        )
    else:
        output["precheck__best_low_batch_site"] = ""
        output["precheck__minimum_total_wwt_shortfall_m3_per_year"] = np.nan
        output["precheck__minimum_maximum_site_wwt_shortfall_m3_per_year"] = np.nan
        output["precheck__best_wwt_capacity_utilisation"] = np.nan
        output["precheck__best_allocation_violating_sites"] = ""
    output["precheck__wwt_allocation_details_json"] = json.dumps(
        allocation_diagnostics, sort_keys=True, default=str
    )

    polymers_without_route: List[str] = []
    for polymer in data.polymers:
        available_routes = [
            route
            for route in data.routes
            if bool(data.eligibility[polymer, route])
            and float(data.route_capacities_t.get(route, 0.0)) > 0.0
        ]
        if not available_routes:
            polymers_without_route.append(str(polymer))
    output["precheck__all_polymers_have_positive_capacity_route"] = not polymers_without_route
    output["precheck__polymers_without_positive_capacity_route"] = "|".join(polymers_without_route)

    total_plastic_t = (
        float(data.total_batches)
        * sum(float(data.polymer_kg_per_batch[polymer]) for polymer in data.polymers)
        / 1000.0
    )
    output["precheck__total_plastic_t_per_year"] = total_plastic_t
    for polymer in data.polymers:
        safe_polymer = re_key(str(polymer))
        output[f"precheck__polymer_generation_t_per_year__{safe_polymer}"] = (
            float(data.total_batches)
            * float(data.polymer_kg_per_batch[polymer])
            / 1000.0
        )
    return output


def re_key(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_")


def classify_physical_mechanism(precheck: Mapping[str, Any]) -> Tuple[str, str]:
    if int(precheck.get("precheck__active_site_count", 0)) == 0:
        return "no_active_manufacturing_site", "no active manufacturing site is available"
    if not bool(precheck.get("precheck__batch_allocation_closes", False)):
        return (
            "batch_allocation_inconsistency",
            "the permitted high/low batch allocation does not close to the annual production requirement",
        )
    if not bool(precheck.get("precheck__wwt_allocation_feasible", False)):
        return (
            "wastewater_treatment_capacity",
            "annual wastewater generation exceeds installed treatment capacity under every permitted low-batch allocation",
        )
    if not bool(precheck.get("precheck__all_polymers_have_positive_capacity_route", False)):
        return (
            "polymer_without_available_treatment_route",
            "at least one generated polymer has no eligible positive-capacity treatment route",
        )
    return (
        "unresolved_by_structural_precheck",
        "the MILP is infeasible but the audited physical prechecks do not identify the conflicting mechanism",
    )


def is_technical_exception(row: Mapping[str, Any]) -> bool:
    return str(row.get("failure_class", "")).lower() == "technical_exception"


def is_gams_exception(row: Mapping[str, Any]) -> bool:
    if not is_technical_exception(row):
        return False
    text = str(row.get("error", "")).lower()
    return any(
        marker in text
        for marker in (
            "gams",
            "solverfactory('gams')",
            "gams encountered an error during solve",
            "solve failed for so_base",
        )
    )


def restart_for_resume(reason: str) -> None:
    script = str(Path(__file__).resolve())
    arguments = [argument for argument in sys.argv[1:] if argument != "--resume"]
    print(f"\n[GAMS PROCESS RECYCLE] {reason}", flush=True)
    print("Checkpoints saved; restarting and resuming the same diagnostic run.", flush=True)
    os.execv(sys.executable, [sys.executable, script, *arguments, "--resume"])


def solve_diagnostic_realisation(
    source: Any,
    sobol_index: int,
    values: Mapping[str, float],
    unit_values: Mapping[str, float],
    registry_specs: Sequence[Any],
    custom_metas: Sequence[Any],
    core: Any,
    master: Any,
    central_data: Any,
    impact: Any,
    solver: Any,
    config: Mapping[str, Any],
    fixed_carbon_value: float,
    result_dir: Path,
    keep_feasible_logs: bool,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    state = core.clone_state(central_data, impact)
    started = time.perf_counter()
    status_row: Dict[str, Any] = {
        "evaluation_id": sobol_index + 1,
        "sobol_index": sobol_index,
        "analysis_case": "BASE",
        "success": False,
        "status": "",
        "failure_class": "",
        "solve_seconds": np.nan,
        "physical_mechanism_code": "",
        "physical_mechanism": "",
        "error": "",
    }
    log_dir = result_dir / "gams_logs"
    before_logs: set[Path] = set()
    try:
        for specification in registry_specs:
            if specification.parameter_id in values:
                specification.apply(state, float(values[specification.parameter_id]))
        source.apply_custom_parameters(state, central_data, values, custom_metas, config)
        source.enforce_central_carbon_value(state, fixed_carbon_value)
        precheck = detailed_structural_precheck(source, state)
        core.sync_gwp_coefficients(state, impact)
        model = master.build_linear_model(state.data)

        before_logs = snapshot_files(log_dir)
        result = solver.solve_backend(
            model,
            case_id=f"SO_BASE_{sobol_index:05d}",
            metadata={
                "analysis": "sobol_qmc_solution_space",
                "sobol_index": sobol_index,
                "analysis_case": "BASE",
                "diagnostic_only": True,
            },
        )
        status_row["solve_seconds"] = time.perf_counter() - started
        status_row["status"] = str(result.message)
        if result.success:
            status_row["success"] = True
            if not keep_feasible_logs:
                remove_new_feasible_logs(before_logs, log_dir)
            return status_row, None

        status_row["failure_class"] = source.classify_nonoptimal_message(result.message)
        status_row["error"] = str(result.message)
        detail: Dict[str, Any] = {
            **status_row,
            **{parameter_id: float(value) for parameter_id, value in values.items()},
            **{name: float(value) for name, value in unit_values.items()},
            **precheck,
            "lifecycle_carbon_value_realised": float(
                state.data.lifecycle_carbon_value_gbp_per_t
            ),
        }
        for polymer, share in source.realised_composition(state).items():
            detail[f"realised_composition_share__{polymer}"] = share
        if status_row["failure_class"] == "infeasible":
            code, explanation = classify_physical_mechanism(precheck)
            status_row["physical_mechanism_code"] = code
            status_row["physical_mechanism"] = explanation
            detail["physical_mechanism_code"] = code
            detail["physical_mechanism"] = explanation
        return status_row, detail
    except Exception as exc:
        status_row["solve_seconds"] = time.perf_counter() - started
        status_row["status"] = "exception"
        status_row["failure_class"] = "technical_exception"
        status_row["error"] = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        return status_row, dict(status_row)


def load_checkpoints(
    status_path: Path,
    detail_path: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], set[int], int]:
    if not status_path.exists():
        return [], [], set(), 0
    status_frame = pd.read_csv(status_path)
    reusable: Dict[int, Dict[str, Any]] = {}
    discarded_technical = 0
    for row in status_frame.to_dict(orient="records"):
        index = int(row["sobol_index"])
        if is_technical_exception(row):
            discarded_technical += 1
            continue
        reusable[index] = row
    detail_rows: Dict[int, Dict[str, Any]] = {}
    if detail_path.exists():
        for row in pd.read_csv(detail_path).to_dict(orient="records"):
            index = int(row["sobol_index"])
            if index in reusable and str(row.get("failure_class", "")) != "technical_exception":
                detail_rows[index] = row
    ordered_status = [reusable[index] for index in sorted(reusable)]
    ordered_details = [detail_rows[index] for index in sorted(detail_rows)]
    return ordered_status, ordered_details, set(reusable), discarded_technical


def save_checkpoints(
    status_path: Path,
    detail_path: Path,
    status_rows: Sequence[Mapping[str, Any]],
    detail_rows: Sequence[Mapping[str, Any]],
) -> None:
    pd.DataFrame(status_rows).sort_values("sobol_index").to_csv(status_path, index=False)
    if detail_rows:
        pd.DataFrame(detail_rows).sort_values("sobol_index").to_csv(detail_path, index=False)
    else:
        pd.DataFrame(columns=["sobol_index", "failure_class"]).to_csv(detail_path, index=False)


def parameter_profile(
    source: Any,
    design: pd.DataFrame,
    statuses: pd.DataFrame,
    metas: Sequence[Any],
) -> pd.DataFrame:
    evaluation = design.merge(
        statuses[["sobol_index", "success", "failure_class"]],
        on="sobol_index",
        how="inner",
        validate="one_to_one",
    )
    evaluation["analysis_case"] = "BASE"
    evaluation["success"] = evaluation["success"].map(coerce_bool)
    return source.infeasibility_parameter_profile(evaluation, metas)


def describe_numeric(series: pd.Series) -> Dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {}
    return {
        "minimum": float(values.min()),
        "p25": float(values.quantile(0.25)),
        "median": float(values.median()),
        "p75": float(values.quantile(0.75)),
        "maximum": float(values.max()),
    }


def compare_infeasible_indices(
    reference_dir: Path,
    new_indices: set[int],
) -> Dict[str, Any]:
    path = reference_dir / "infeasible_evaluations.csv"
    if not path.is_file():
        return {
            "reference_file_found": False,
            "index_set_match": None,
            "new_infeasible_count": len(new_indices),
        }
    frame = pd.read_csv(path)
    require("sobol_index" in frame.columns, f"Reference infeasible file lacks sobol_index: {path}")
    reference_indices = {int(value) for value in frame["sobol_index"]}
    missing_now = sorted(reference_indices - new_indices)
    newly_infeasible = sorted(new_indices - reference_indices)
    return {
        "reference_file_found": True,
        "reference_infeasible_count": len(reference_indices),
        "new_infeasible_count": len(new_indices),
        "index_set_match": not missing_now and not newly_infeasible,
        "reference_indices_not_reproduced": missing_now,
        "new_indices_not_in_reference": newly_infeasible,
    }




def create_report(
    result_dir: Path,
    statuses: pd.DataFrame,
    details: pd.DataFrame,
    profile: pd.DataFrame,
    identity: Mapping[str, Any],
    index_comparison: Mapping[str, Any],
    expected_infeasible: int,
) -> Dict[str, Any]:
    failure = statuses["failure_class"].fillna("").astype(str).str.lower()
    infeasible_mask = failure.eq("infeasible")
    infeasible_count = int(infeasible_mask.sum())
    total = len(statuses)
    mechanism_counter = Counter(
        statuses.loc[infeasible_mask, "physical_mechanism"].fillna("unresolved").replace("", "unresolved")
    )
    code_counter = Counter(
        statuses.loc[infeasible_mask, "physical_mechanism_code"].fillna("unresolved").replace("", "unresolved")
    )
    wwt = details.loc[
        details.get("physical_mechanism_code", pd.Series("", index=details.index))
        .fillna("")
        .astype(str)
        .eq("wastewater_treatment_capacity")
    ]
    utilisation_stats = (
        describe_numeric(wwt["precheck__best_wwt_capacity_utilisation"])
        if "precheck__best_wwt_capacity_utilisation" in wwt.columns
        else {}
    )
    shortfall_stats = (
        describe_numeric(wwt["precheck__minimum_total_wwt_shortfall_m3_per_year"])
        if "precheck__minimum_total_wwt_shortfall_m3_per_year" in wwt.columns
        else {}
    )

    top_parameters: List[Dict[str, Any]] = []
    if not profile.empty:
        available = [
            column
            for column in (
                "parameter_id",
                "parameter_label",
                "parameter_group",
                "normalized_median_shift_infeasible_minus_feasible",
                "absolute_normalized_median_shift",
            )
            if column in profile.columns
        ]
        top_parameters = profile[available].head(10).to_dict(orient="records")

    summary: Dict[str, Any] = {
        "sobol_sample_count": total,
        "infeasible_count": infeasible_count,
        "infeasible_share": infeasible_count / total if total else np.nan,
        "expected_infeasible_count": expected_infeasible,
        "expected_count_matched": infeasible_count == expected_infeasible,
        "mechanism_code_counts": dict(code_counter),
        "mechanism_counts": dict(mechanism_counter),
        "wwt_best_allocation_capacity_utilisation": utilisation_stats,
        "wwt_minimum_total_shortfall_m3_per_year": shortfall_stats,
        "design_identity": dict(identity),
        "infeasible_index_comparison": dict(index_comparison),
        "top_descriptive_parameter_contrasts": top_parameters,
    }
    write_json(result_dir / "infeasibility_diagnostic_summary.json", summary)

    return summary


def main() -> Dict[str, Path]:
    args = parse_args()
    source_path = locate_source_runner(args.source_runner)
    source = import_module(source_path, "original_stochastic_runner")

    reference_dir = (
        args.reference_result_dir.expanduser().resolve()
        if args.reference_result_dir
        else DEFAULT_REFERENCE_DIR.resolve()
    )
    result_dir = (
        args.result_dir.expanduser().resolve()
        if args.result_dir
        else DEFAULT_DIAGNOSTIC_DIR.resolve()
    )
    require(result_dir != reference_dir, "Diagnostic and original result directories must differ.")
    if result_dir.exists() and not args.resume:
        if args.overwrite:
            shutil.rmtree(result_dir)
        elif any(result_dir.iterdir()):
            raise FileExistsError(
                f"Diagnostic result directory already contains files: {result_dir}. "
                "Use --resume or --overwrite."
            )
    result_dir.mkdir(parents=True, exist_ok=True)

    reference_plan, reference_design = load_reference_identity(reference_dir)
    sample_power = int(reference_plan["sample_power"])
    seed = int(reference_plan["seed"])
    backend = args.backend or str(reference_plan.get("backend", source.SOLVER_BACKEND))
    require(
        backend == str(reference_plan.get("backend", backend)),
        f"Backend mismatch: diagnostic {backend}, original {reference_plan.get('backend')}",
    )
    restart_every = (
        int(args.gams_restart_every)
        if args.gams_restart_every is not None
        else int(source.GAMS_PROCESS_RESTART_EVERY)
    )

    shared_path = source.locate_shared_module(args.shared_module)
    core = source.import_module_from_path(shared_path, "infeasibility_shared_layer")
    master_path = core.locate_master_file()
    master = core.import_master_module(master_path)
    config_path, config = source.load_config(args.config)
    central_data = master.load_model_data()
    fixed_carbon_value = source.central_carbon_value_2026_gbp(central_data)
    impact = core.load_impact_model(master, central_data, core.find_lca_settings())

    registry_specs, registry_metas = source.resolve_registry_parameters(
        core, central_data, impact, config, requested=None
    )
    custom_metas = source.resolve_custom_parameters(config, requested=None)
    metas = registry_metas + custom_metas
    source.validate_sampling_ranges(metas, config)
    parameter_ids = [meta.parameter_id for meta in metas]
    require(
        parameter_ids == list(reference_plan["parameter_ids"]),
        "Resolved parameter order differs from the original run plan.",
    )
    require(str(reference_plan["objective"]) == str(source.OBJECTIVE), "Objective differs from original run.")
    require(
        math.isclose(
            float(reference_plan["fixed_central_carbon_value_nominal_2026_gbp_per_tco2e"]),
            fixed_carbon_value,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "Fixed central carbon value differs from the original run.",
    )
    current_config_hash = canonical_hash(config, length=20)
    require(
        current_config_hash == str(reference_plan["config_hash"]),
        "Current stochastic configuration differs from the original run; no diagnostic solves were started.",
    )

    design = source.generate_sobol_design(metas, sample_power, seed)
    identity = verify_design_identity(design, reference_design, parameter_ids)
    identity.update(
        {
            "source_runner": str(source_path),
            "shared_mutation_layer": str(shared_path),
            "master_model": str(master_path),
            "config": str(config_path),
            "config_hash": current_config_hash,
            "sample_power": sample_power,
            "seed": seed,
            "backend": backend,
            "objective": source.OBJECTIVE,
            "fixed_carbon_value_gbp_per_tco2e": fixed_carbon_value,
        }
    )
    write_json(result_dir / "design_identity_check.json", identity)

    plans = [int(value) for value in design["sobol_index"]]
    limited_run = args.limit is not None
    if limited_run:
        require(args.limit is not None and args.limit > 0, "--limit must be positive")
        plans = plans[: int(args.limit)]

    run_identity = {
        "reference_run_identity_hash": reference_plan.get("run_identity_hash"),
        "diagnostic_design_hash": identity["regenerated_design_sha1_rounded_14dp"],
        "planned_indices": plans,
        "limited_run": limited_run,
        "backend": backend,
    }
    run_identity_hash = canonical_hash(run_identity)
    diagnostic_plan_path = result_dir / "diagnostic_run_plan.json"
    if args.resume:
        require(diagnostic_plan_path.is_file(), f"Resume plan not found: {diagnostic_plan_path}")
        prior_plan = json.loads(diagnostic_plan_path.read_text(encoding="utf-8"))
        require(
            prior_plan.get("diagnostic_run_identity_hash") == run_identity_hash,
            "Diagnostic run identity changed; refusing to mix checkpoints.",
        )
    else:
        write_json(
            diagnostic_plan_path,
            {**run_identity, "diagnostic_run_identity_hash": run_identity_hash},
        )

    status_checkpoint = result_dir / "status_checkpoint.csv"
    detail_checkpoint = result_dir / "infeasible_cases_checkpoint.csv"
    if args.resume:
        status_rows, detail_rows, completed, discarded = load_checkpoints(
            status_checkpoint, detail_checkpoint
        )
        print(f"Resuming from {len(completed):,} completed Sobol cases.")
        if discarded:
            print(f"Retrying {discarded} technical exception case(s).")
    else:
        status_rows, detail_rows, completed = [], [], set()

    solver = core.ScenarioSolver(
        master=master,
        central_data=central_data,
        impact=impact,
        result_dir=result_dir,
        backend=backend,
        objective=source.OBJECTIVE,
        tee=args.tee or source.GAMS_TEE,
    )

    if backend == "gams" and completed:
        try:
            master._GAMS_SOLVE_COUNTER = len(completed) + 1
        except Exception:
            pass

    central_path = result_dir / "central_reference.csv"
    if args.resume and central_path.is_file():
        print("Reusing diagnostic central reference.")
    else:
        central_log_before = snapshot_files(result_dir / "gams_logs")
        central_row = source.solve_central(
            core, master, central_data, impact, solver, fixed_carbon_value
        )
        pd.DataFrame([central_row]).to_csv(central_path, index=False)
        if not args.keep_feasible_logs:
            remove_new_feasible_logs(central_log_before, result_dir / "gams_logs")

    pending = [index for index in plans if index not in completed]
    solve_times = [
        safe_float(row.get("solve_seconds"))
        for row in status_rows
        if math.isfinite(safe_float(row.get("solve_seconds")))
    ]
    retry_path = result_dir / "technical_retry_counts.json"
    retry_counts: Dict[str, int] = (
        json.loads(retry_path.read_text(encoding="utf-8")) if retry_path.is_file() else {}
    )
    newly_solved = 0
    maximum_retries = int(source.MAX_GAMS_TECHNICAL_RETRIES_PER_SAMPLE)

    print("\nSOBOL INFEASIBILITY DIAGNOSTIC")
    print(f"  Original design match : yes")
    print(f"  Dimensions            : {len(metas)}")
    print(f"  Sobol samples         : {len(plans)}")
    print(f"  Seed                   : {seed}")
    print(f"  Backend                : {backend}")
    print(f"  Output                 : {result_dir}\n")

    for sequence, sobol_index in enumerate(pending, start=1):
        design_row = design.loc[design["sobol_index"].eq(sobol_index)].iloc[0]
        values = {meta.parameter_id: float(design_row[meta.parameter_id]) for meta in metas}
        unit_values = {
            f"u__{meta.parameter_id}": float(design_row[f"u__{meta.parameter_id}"])
            for meta in metas
        }
        status, detail = solve_diagnostic_realisation(
            source=source,
            sobol_index=sobol_index,
            values=values,
            unit_values=unit_values,
            registry_specs=registry_specs,
            custom_metas=custom_metas,
            core=core,
            master=master,
            central_data=central_data,
            impact=impact,
            solver=solver,
            config=config,
            fixed_carbon_value=fixed_carbon_value,
            result_dir=result_dir,
            keep_feasible_logs=args.keep_feasible_logs,
        )
        status_rows.append(status)
        if detail is not None:
            detail_rows.append(detail)
        newly_solved += 1
        if math.isfinite(safe_float(status.get("solve_seconds"))):
            solve_times.append(float(status["solve_seconds"]))

        if is_technical_exception(status):
            save_checkpoints(status_checkpoint, detail_checkpoint, status_rows, detail_rows)
            key = str(sobol_index)
            retry_counts[key] = int(retry_counts.get(key, 0)) + 1
            write_json(retry_path, retry_counts)
            if backend == "gams" and is_gams_exception(status) and retry_counts[key] <= maximum_retries:
                restart_for_resume(
                    f"technical GAMS exception at sample {sobol_index}; "
                    f"retry {retry_counts[key]}/{maximum_retries}"
                )
            raise RuntimeError(
                f"Technical exception at Sobol sample {sobol_index}; see status_checkpoint.csv.\n"
                + str(status.get("error", ""))[-2500:]
            )

        done = len(completed) + sequence
        mean_seconds = float(np.mean(solve_times[-200:])) if solve_times else np.nan
        remaining = max(len(plans) - done, 0)
        eta_seconds = mean_seconds * remaining if math.isfinite(mean_seconds) else np.nan
        label = "OK" if status["success"] else str(status["failure_class"]).upper()
        infeasible_so_far = sum(
            str(row.get("failure_class", "")).lower() == "infeasible"
            for row in status_rows
        )
        print(
            f"[{done:4d}/{len(plans)} | {100.0 * done / len(plans):5.1f}%] "
            f"BASE #{sobol_index:04d} {label:<12s} "
            f"{safe_float(status.get('solve_seconds')):6.2f}s | "
            f"infeasible {infeasible_so_far:3d} | ETA {finish_time_text(eta_seconds)}",
            flush=True,
        )

        if done % CHECKPOINT_EVERY == 0 or sequence == len(pending):
            save_checkpoints(status_checkpoint, detail_checkpoint, status_rows, detail_rows)

        if (
            backend == "gams"
            and restart_every > 0
            and newly_solved >= restart_every
            and done < len(plans)
        ):
            save_checkpoints(status_checkpoint, detail_checkpoint, status_rows, detail_rows)
            restart_for_resume(
                f"planned resource reset after {newly_solved} solves in this interpreter"
            )

    statuses = pd.DataFrame(status_rows).sort_values("sobol_index").reset_index(drop=True)
    statuses["success"] = statuses["success"].map(coerce_bool)
    details = (
        pd.DataFrame(detail_rows).sort_values("sobol_index").reset_index(drop=True)
        if detail_rows
        else pd.DataFrame(columns=["sobol_index", "failure_class", "physical_mechanism_code"])
    )
    statuses.to_csv(result_dir / "solve_status_audit.csv", index=False)
    details.to_csv(result_dir / "infeasible_cases.csv", index=False)

    technical_count = int(
        statuses["failure_class"].fillna("").astype(str).str.lower().eq("technical_exception").sum()
    )
    require(technical_count == 0, f"{technical_count} technical exceptions remain")
    require(len(statuses) == len(plans), f"Only {len(statuses)}/{len(plans)} cases completed")

    profile = parameter_profile(source, design.loc[design["sobol_index"].isin(plans)], statuses, metas)
    profile.to_csv(result_dir / "infeasibility_parameter_profile.csv", index=False)
    mechanism_counts = (
        details.loc[
            details["failure_class"].fillna("").astype(str).str.lower().eq("infeasible"),
            ["physical_mechanism_code", "physical_mechanism"],
        ]
        .value_counts(dropna=False)
        .rename("case_count")
        .reset_index()
    )
    infeasible_total = int(mechanism_counts["case_count"].sum()) if not mechanism_counts.empty else 0
    mechanism_counts["percent_of_infeasible"] = (
        100.0 * mechanism_counts["case_count"] / infeasible_total if infeasible_total else 0.0
    )
    mechanism_counts.to_csv(result_dir / "infeasibility_mechanism_counts.csv", index=False)

    new_infeasible_indices = {
        int(value)
        for value in statuses.loc[
            statuses["failure_class"].fillna("").astype(str).str.lower().eq("infeasible"),
            "sobol_index",
        ]
    }
    index_comparison = compare_infeasible_indices(reference_dir, new_infeasible_indices)
    write_json(result_dir / "infeasible_index_comparison.json", index_comparison)

    summary = create_report(
        result_dir=result_dir,
        statuses=statuses,
        details=details,
        profile=profile,
        identity=identity,
        index_comparison=index_comparison,
        expected_infeasible=args.expected_infeasible,
    )
    if not limited_run:
        require(
            int(summary["infeasible_count"]) == int(args.expected_infeasible),
            f"Infeasible count {summary['infeasible_count']} did not reproduce expected {args.expected_infeasible}",
        )
        if index_comparison.get("reference_file_found"):
            require(
                bool(index_comparison.get("index_set_match")),
                "The reproduced infeasible Sobol indices differ from the original run.",
            )

    print("\nCOMPLETE — INFEASIBILITY DIAGNOSTIC PASSED")
    print(f"  Sobol cases solved      : {len(statuses):,}")
    print(f"  Infeasible cases        : {summary['infeasible_count']:,}")
    print(f"  Design identity match   : yes")
    print(f"  Index-set match         : {index_comparison.get('index_set_match')}")
    print(f"  Infeasible case details : {result_dir / 'infeasible_cases.csv'}")
    return {
        "result_dir": result_dir,
    }


if __name__ == "__main__":
    main()
