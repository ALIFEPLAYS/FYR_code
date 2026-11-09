from __future__ import annotations

"""
Independent epsilon-constraint cost-GWP Pareto analysis.

Purpose
-------
This script performs TWO DISTINCT optimisation methodologies on the same
industrial-symbiosis MILP and compares their resulting solutions only after
both have been solved independently.

Method A -- epsilon-constraint Pareto frontier
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For a uniformly spaced set of GWP caps epsilon spanning the independently
solved GWP-minimum and cost-minimum anchors:

    minimise  C(x)
    subject to G(x) <= epsilon

where C is annual network/private cost [GBP/year] and G is annual life-cycle
GWP [kg CO2e/year].  The lifecycle carbon value is explicitly set to ZERO in
the model used for every epsilon solve.  It therefore cannot influence the
frontier, even indirectly.

Each epsilon point is solved lexicographically:

    Stage 1: minimise cost subject to GWP <= epsilon;
    Stage 2: minimise GWP under the same epsilon cap while constraining cost
             to remain within a negligible numerical tolerance of the Stage-1
             minimum.

Stage 2 removes weakly efficient cost-tie solutions; it does not introduce a
carbon value or alter the primary epsilon-constraint methodology.

Method B -- central carbon-value comparator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Separately, using the unmodified central master model:

    minimise  C(x) + v_CO2 * G(x) / 1000

where v_CO2 is the central lifecycle carbon value already converted by the
master model to its common 2026 GBP price basis.

CRITICAL INDEPENDENCE RULE
--------------------------
The carbon-value solution is NOT used to construct the epsilon grid.  Its GWP
is not inserted as an epsilon cap, no local epsilon refinement is centred on
it, and it has no role in Pareto filtering.  Only after the independent
cost-GWP frontier is complete is the carbon-value solution overlaid as a
comparison marker and compared post hoc with the sampled frontier.

The script imports and reuses run_master_MILP.py so that both methodologies
share identical physical constraints, route factors, costs, LCIA factors,
solver settings and interpretation.  A model-alignment audit verifies that the
epsilon model differs from the central model only in the carbon-value field /
policy-cost expression; the cost and GWP objectives and all constraints must
remain identical.

Outputs
-------
Results are written to a NEW directory so they cannot be mixed with the older
carbon-centred epsilon design:

    results/epsilon_constraint_pareto_independent_cost_gwp/

Results are written to results/epsilon_constraint_pareto_independent_cost_gwp/.
"""

import argparse
import copy
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, vstack


# =============================================================================
# USER CONTROLS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
MASTER_SCRIPT = SCRIPT_DIR / "run_master_MILP.py"
PROJECT_ROOT = SCRIPT_DIR.parent

RESULT_DIRECTORY = PROJECT_ROOT / "results" / "epsilon_constraint_pareto_independent_cost_gwp"
TABLE_DIRECTORY = RESULT_DIRECTORY / "tables"
VECTOR_DIRECTORY = RESULT_DIRECTORY / "solution_vectors"
SOLVER_DIRECTORY = RESULT_DIRECTORY / "solver"

# Independent epsilon design: uniformly spaced caps between the two payoff-table anchors.
DEFAULT_GLOBAL_EPSILON_POINTS = 201

# Lexicographic tie tolerances. These are intentionally economically/environmentally
# negligible; they are only to prevent numerical infeasibility when the Stage-1
# optimum is reintroduced as a Stage-2 upper bound.
LEX_COST_TOL_ABS_GBP = 0.10
LEX_COST_TOL_REL = 1e-9
LEX_GWP_TOL_ABS_KG = 1e-3
LEX_GWP_TOL_REL = 1e-10
CAP_FEASIBILITY_TOL_ABS_KG = 1e-3
CAP_FEASIBILITY_TOL_REL = 1e-10

# Pareto filtering / equivalence tolerances.
PARETO_DOMINANCE_COST_TOL_GBP = 0.01
PARETO_DOMINANCE_GWP_TOL_KG = 1e-3
POLICY_EQUIV_COST_TOL_GBP = 1.00
POLICY_EQUIV_GWP_TOL_KG = 1.00

# =============================================================================
# GENERIC HELPERS
# =============================================================================


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def estimated_finish_text(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    finish = datetime.now().astimezone() + timedelta(seconds=float(seconds))
    return finish.strftime("%H:%M %Z")


def finite(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return np.nan
    return out if math.isfinite(out) else np.nan


def pct_change(value: float, baseline: float) -> float:
    if not math.isfinite(value) or not math.isfinite(baseline) or abs(baseline) <= 1e-12:
        return np.nan
    return 100.0 * (value - baseline) / abs(baseline)


def atomic_write_text(path: Path, text: str) -> None:
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


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_write_text(path, frame.to_csv(index=False))


def atomic_copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        shutil.copyfile(source, tmp_path)
        os.replace(tmp_path, destination)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cap_key(cap_kg: float) -> str:
    return hashlib.sha1(f"{float(cap_kg):.12f}".encode("utf-8")).hexdigest()[:14]




@dataclass
class SolveProgress:
    total_solves: int
    completed: int = 0
    started: float = 0.0
    durations: List[float] = None

    def __post_init__(self) -> None:
        self.started = time.perf_counter()
        self.durations = []

    def finish(self, label: str, solve_started: float) -> None:
        duration = time.perf_counter() - solve_started
        self.completed += 1
        self.durations.append(duration)
        average = float(np.mean(self.durations[-30:])) if self.durations else np.nan
        remaining = max(self.total_solves - self.completed, 0)
        eta = average * remaining if math.isfinite(average) else np.nan
        elapsed = time.perf_counter() - self.started
        print(
            f"    solve {self.completed}/{self.total_solves} "
            f"({100.0*self.completed/max(self.total_solves,1):.1f}%) | "
            f"latest {format_duration(duration)} | elapsed {format_duration(elapsed)} | "
            f"ETA {format_duration(eta)} | finish ~{estimated_finish_text(eta)} | {label}",
            flush=True,
        )


# =============================================================================
# MASTER-MODEL LOADING AND CONFIGURATION
# =============================================================================


def import_master_module() -> Any:
    require(MASTER_SCRIPT.exists(), f"Master MILP script not found: {MASTER_SCRIPT}")
    module_name = "_epsilon_master_milp"
    spec = importlib.util.spec_from_file_location(module_name, MASTER_SCRIPT)
    require(spec is not None and spec.loader is not None, f"Could not import {MASTER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def reset_output_directories(resume: bool) -> None:
    if RESULT_DIRECTORY.exists() and not resume:
        shutil.rmtree(RESULT_DIRECTORY)
    for directory in [RESULT_DIRECTORY, TABLE_DIRECTORY, VECTOR_DIRECTORY, SOLVER_DIRECTORY]:
        directory.mkdir(parents=True, exist_ok=True)


def configure_master_solver(master: Any, args: argparse.Namespace) -> None:
    """Redirect the imported master model's GAMS audit outputs to this analysis."""
    master.RESULT_DIR = RESULT_DIRECTORY
    master.GAMS_LOG_DIR = SOLVER_DIRECTORY / "gams_logs"
    master.GAMS_WORK_DIR = SOLVER_DIRECTORY / "gams_work"
    master.GAMS_SOLVE_REPORT = SOLVER_DIRECTORY / "gams_solve_report.csv"
    master.GAMS_PREFLIGHT_LOG = master.GAMS_LOG_DIR / "0000_gams_preflight.log"
    master.GAMS_LOG_DIR.mkdir(parents=True, exist_ok=True)
    if args.keep_gams_files:
        master.GAMS_WORK_DIR.mkdir(parents=True, exist_ok=True)
    master.GAMS_TEE = bool(args.solver_output)
    master.GAMS_KEEPFILES = bool(args.keep_gams_files)
    master._GAMS_SOLVE_COUNTER = 0
    master._GAMS_PREFLIGHT_DONE = False


def validate_independent_epsilon_model(policy_model: Any, epsilon_model: Any) -> None:
    """Prove that removing the carbon value does not alter the feasible MILP or C/G expressions."""
    require(
        policy_model.var_keys == epsilon_model.var_keys,
        "Epsilon and central-policy models do not have identical variable ordering.",
    )
    require(np.allclose(policy_model.lower, epsilon_model.lower, rtol=0.0, atol=1e-12),
            "Variable lower bounds changed when constructing the epsilon model.")
    require(np.allclose(policy_model.upper, epsilon_model.upper, rtol=0.0, atol=1e-12),
            "Variable upper bounds changed when constructing the epsilon model.")
    require(np.array_equal(policy_model.integrality, epsilon_model.integrality),
            "Variable integrality changed when constructing the epsilon model.")
    require(np.allclose(policy_model.b_l, epsilon_model.b_l, rtol=0.0, atol=1e-12, equal_nan=True),
            "Constraint lower bounds changed when constructing the epsilon model.")
    require(np.allclose(policy_model.b_u, epsilon_model.b_u, rtol=0.0, atol=1e-12, equal_nan=True),
            "Constraint upper bounds changed when constructing the epsilon model.")

    delta_a = policy_model.A.tocsr() - epsilon_model.A.tocsr()
    if delta_a.nnz:
        require(float(np.max(np.abs(delta_a.data))) <= 1e-12,
                "Constraint matrix changed when constructing the epsilon model.")

    for expression_name in ("cost", "gwp"):
        p = policy_model.expressions[expression_name]
        e = epsilon_model.expressions[expression_name]
        require(np.allclose(p.coeff, e.coeff, rtol=0.0, atol=1e-12),
                f"{expression_name} coefficients differ between policy and epsilon models.")
        require(abs(float(p.const) - float(e.const)) <= 1e-12,
                f"{expression_name} constant differs between policy and epsilon models.")

    require(abs(float(epsilon_model.data.lifecycle_carbon_value_gbp_per_t)) <= 1e-12,
            "Epsilon model carbon value must be exactly zero.")


def solve_quietly(
    master: Any,
    model: Any,
    case_id: str,
    primary: str,
    selection_method: str,
    metadata: Optional[Dict[str, Any]],
    show_solver_output: bool,
) -> Any:
    if show_solver_output:
        return master.solve_model(
            model, case_id, primary,
            selection_method=selection_method,
            metadata=metadata,
        )

    capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(capture):
            return master.solve_model(
                model, case_id, primary,
                selection_method=selection_method,
                metadata=metadata,
            )
    except Exception:
        text = capture.getvalue().strip()
        if text:
            print("\nCaptured solver output before failure:\n" + text[-8000:], flush=True)
        raise


# =============================================================================
# LINEAR EPSILON / LEXICOGRAPHIC CONSTRAINTS
# =============================================================================


def add_expression_upper_bound(model: Any, expression_name: str, upper_bound: float) -> Any:
    """Return a dataclass copy with expr(x) <= upper_bound appended to A/b bounds."""
    require(expression_name in model.expressions, f"Unknown model expression: {expression_name}")
    expression = model.expressions[expression_name]
    rhs = float(upper_bound) - float(expression.const)
    row = coo_matrix(np.asarray(expression.coeff, dtype=float).reshape(1, -1))
    A_new = vstack([model.A.tocsr(), row], format="coo")
    b_l_new = np.append(np.asarray(model.b_l, dtype=float), -np.inf)
    b_u_new = np.append(np.asarray(model.b_u, dtype=float), rhs)
    return replace(model, A=A_new, b_l=b_l_new, b_u=b_u_new)


def cost_tolerance(cost_gbp: float) -> float:
    return max(LEX_COST_TOL_ABS_GBP, LEX_COST_TOL_REL * abs(float(cost_gbp)))


def gwp_tolerance(gwp_kg: float) -> float:
    return max(LEX_GWP_TOL_ABS_KG, LEX_GWP_TOL_REL * abs(float(gwp_kg)))


def cap_tolerance(gwp_cap_kg: float) -> float:
    return max(CAP_FEASIBILITY_TOL_ABS_KG, CAP_FEASIBILITY_TOL_REL * abs(float(gwp_cap_kg)))


def solve_and_track(
    master: Any,
    model: Any,
    *,
    case_id: str,
    primary: str,
    selection_method: str,
    metadata: Optional[Dict[str, Any]],
    progress: SolveProgress,
    show_solver_output: bool,
) -> Any:
    started = time.perf_counter()
    result = solve_quietly(
        master, model, case_id, primary, selection_method, metadata, show_solver_output
    )
    progress.finish(case_id, started)
    return result


def solve_cost_anchor(master: Any, model: Any, progress: SolveProgress, args: argparse.Namespace) -> Tuple[Any, Any]:
    stage1 = solve_and_track(
        master, model,
        case_id="ANCHOR_COST_STAGE1",
        primary="cost",
        selection_method="single-objective minimum annual network cost",
        metadata={"anchor": "cost", "stage": 1},
        progress=progress,
        show_solver_output=args.solver_output,
    )
    require(stage1.success, f"Cost anchor Stage 1 failed: {stage1.message}")

    tol = cost_tolerance(stage1.metrics["cost"])
    tied = add_expression_upper_bound(model, "cost", stage1.metrics["cost"] + tol)
    stage2 = solve_and_track(
        master, tied,
        case_id="ANCHOR_COST_LEX_GWP",
        primary="gwp",
        selection_method="lexicographic cost then GWP",
        metadata={
            "anchor": "cost",
            "stage": 2,
            "stage1_cost_gbp": stage1.metrics["cost"],
            "cost_tolerance_gbp": tol,
        },
        progress=progress,
        show_solver_output=args.solver_output,
    )
    require(stage2.success, f"Cost anchor lexicographic Stage 2 failed: {stage2.message}")
    return stage1, stage2


def solve_gwp_anchor(master: Any, model: Any, progress: SolveProgress, args: argparse.Namespace) -> Tuple[Any, Any]:
    stage1 = solve_and_track(
        master, model,
        case_id="ANCHOR_GWP_STAGE1",
        primary="gwp",
        selection_method="single-objective minimum life-cycle GWP",
        metadata={"anchor": "gwp", "stage": 1},
        progress=progress,
        show_solver_output=args.solver_output,
    )
    require(stage1.success, f"GWP anchor Stage 1 failed: {stage1.message}")

    tol = gwp_tolerance(stage1.metrics["gwp"])
    tied = add_expression_upper_bound(model, "gwp", stage1.metrics["gwp"] + tol)
    stage2 = solve_and_track(
        master, tied,
        case_id="ANCHOR_GWP_LEX_COST",
        primary="cost",
        selection_method="lexicographic GWP then cost",
        metadata={
            "anchor": "gwp",
            "stage": 2,
            "stage1_gwp_kgco2e": stage1.metrics["gwp"],
            "gwp_tolerance_kgco2e": tol,
        },
        progress=progress,
        show_solver_output=args.solver_output,
    )
    require(stage2.success, f"GWP anchor lexicographic Stage 2 failed: {stage2.message}")
    return stage1, stage2


def solve_epsilon_point(
    master: Any,
    base_model: Any,
    epsilon_cap_kg: float,
    ordinal: int,
    total_caps: int,
    progress: SolveProgress,
    args: argparse.Namespace,
) -> Tuple[Optional[Any], Dict[str, Any]]:
    cap_tol = cap_tolerance(epsilon_cap_kg)
    effective_cap = float(epsilon_cap_kg) + cap_tol
    cap_model = add_expression_upper_bound(base_model, "gwp", effective_cap)
    key = cap_key(epsilon_cap_kg)

    stage1_id = f"EPS_{ordinal:04d}_COST_{key}"
    stage1 = solve_and_track(
        master, cap_model,
        case_id=stage1_id,
        primary="cost",
        selection_method="epsilon-constraint Stage 1: min cost subject to GWP cap",
        metadata={
            "epsilon_cap_target_kgco2e": float(epsilon_cap_kg),
            "epsilon_cap_effective_kgco2e": effective_cap,
            "epsilon_ordinal": ordinal,
            "epsilon_total_caps": total_caps,
            "stage": 1,
            "primary_objective": "cost_only",
            "lifecycle_carbon_value_gbp_per_tco2e": 0.0,
            "carbon_value_used_to_construct_epsilon_grid": False,
        },
        progress=progress,
        show_solver_output=args.solver_output,
    )

    base_row: Dict[str, Any] = {
        "cap_key": key,
        "epsilon_ordinal": ordinal,
        "epsilon_cap_target_kgco2e": float(epsilon_cap_kg),
        "epsilon_cap_target_tco2e": float(epsilon_cap_kg) / 1000.0,
        "epsilon_cap_effective_kgco2e": effective_cap,
        "cap_numerical_tolerance_kgco2e": cap_tol,
        "stage1_success": bool(stage1.success),
        "stage1_solver_message": stage1.message,
        "stage1_cost_gbp": finite(stage1.metrics.get("cost") if stage1.success else np.nan),
        "stage1_gwp_kgco2e": finite(stage1.metrics.get("gwp") if stage1.success else np.nan),
    }
    if not stage1.success:
        base_row.update({
            "stage2_success": False,
            "point_status": "stage1_failed",
            "solution_file": "",
        })
        return None, base_row

    lex_tol = cost_tolerance(stage1.metrics["cost"])
    stage2_model = add_expression_upper_bound(
        cap_model, "cost", stage1.metrics["cost"] + lex_tol
    )
    stage2_id = f"EPS_{ordinal:04d}_LEX_{key}"
    stage2 = solve_and_track(
        master, stage2_model,
        case_id=stage2_id,
        primary="gwp",
        selection_method=(
            "epsilon-constraint Stage 2: min GWP at Stage-1 minimum cost "
            "within numerical tolerance"
        ),
        metadata={
            "epsilon_cap_target_kgco2e": float(epsilon_cap_kg),
            "epsilon_cap_effective_kgco2e": effective_cap,
            "epsilon_ordinal": ordinal,
            "epsilon_total_caps": total_caps,
            "stage": 2,
            "primary_objective": "lexicographic_gwp_at_minimum_cost",
            "lifecycle_carbon_value_gbp_per_tco2e": 0.0,
            "carbon_value_used_to_construct_epsilon_grid": False,
            "stage1_cost_gbp": stage1.metrics["cost"],
            "cost_tolerance_gbp": lex_tol,
        },
        progress=progress,
        show_solver_output=args.solver_output,
    )

    base_row.update({
        "stage2_success": bool(stage2.success),
        "stage2_solver_message": stage2.message,
        "lexicographic_cost_tolerance_gbp": lex_tol,
    })
    if not stage2.success:
        base_row.update({
            "point_status": "stage2_failed",
            "solution_file": "",
        })
        return None, base_row

    base_row["point_status"] = "optimal"
    return stage2, base_row


# =============================================================================
# SOLUTION SUMMARIES / AUDIT METRICS
# =============================================================================


def save_solution_vector(path: Path, result: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            x=np.asarray(result.x, dtype=float),
            case_id=np.array([str(result.case_id)]),
            primary_objective=np.array([str(result.primary_objective)]),
            selection_method=np.array([str(result.selection_method)]),
        )
    os.replace(tmp_path, path)


def result_from_vector(master: Any, model: Any, path: Path, case_id: Optional[str] = None) -> Any:
    with np.load(path, allow_pickle=False) as payload:
        x = np.asarray(payload["x"], dtype=float)
        stored_case = str(payload["case_id"][0]) if "case_id" in payload else path.stem
        primary = str(payload["primary_objective"][0]) if "primary_objective" in payload else "reloaded"
        method = str(payload["selection_method"][0]) if "selection_method" in payload else "reloaded vector"
    metrics = {name: expression.evaluate(x) for name, expression in model.expressions.items()}
    return master.SolveResult(
        case_id=case_id or stored_case,
        success=True,
        status=0,
        message="Reloaded from persisted solution vector",
        x=x,
        metrics=metrics,
        raw_objective=metrics.get(primary, np.nan),
        primary_objective=primary,
        selection_method=method,
        metadata={"solution_vector": str(path)},
    )


def decision_metrics(master: Any, model: Any, result: Any) -> Dict[str, Any]:
    routes, sites, mix = master.solution_decisions(model, result)

    total_electricity = float(sites["grid_kwh"].sum() + sites["wind_kwh"].sum())
    wind_share = float(sites["wind_kwh"].sum() / total_electricity) if total_electricity > 0 else np.nan

    eligible_water = float(sites["water_reused_m3"].sum() + sites["fresh_utility_water_m3"].sum())
    reuse_share = float(sites["water_reused_m3"].sum() / eligible_water) if eligible_water > 0 else np.nan

    if mix.empty:
        total_vehicle_km = 0.0
        bev_vehicle_km = 0.0
    else:
        total_vehicle_km = float(mix["vehicle_km_per_year"].sum())
        bev_vehicle_km = float(
            mix.loc[
                mix["transport_mode"].astype(str).eq("battery_electric_articulated_hgv"),
                "vehicle_km_per_year",
            ].sum()
        )
    bev_share = bev_vehicle_km / total_vehicle_km if total_vehicle_km > 0 else np.nan

    low_rows = sites.loc[sites["is_low_batch_location"].astype(bool)]
    low_location = str(low_rows.iloc[0]["location"]) if not low_rows.empty else ""

    out: Dict[str, Any] = {
        "low_batch_location": low_location,
        "wind_share": wind_share,
        "wind_share_pct": 100.0 * wind_share if math.isfinite(wind_share) else np.nan,
        "water_reuse_share": reuse_share,
        "water_reuse_share_pct": 100.0 * reuse_share if math.isfinite(reuse_share) else np.nan,
        "bev_vehicle_km_share": bev_share,
        "bev_vehicle_km_share_pct": 100.0 * bev_share if math.isfinite(bev_share) else np.nan,
        "total_vehicle_km_per_year": total_vehicle_km,
        "decision_snapshot": master.compact_decision_text(model, result),
    }

    total_plastic = float(routes["tonnes_per_year"].sum()) if not routes.empty else 0.0
    route_shares: Dict[str, float] = {}
    for route in model.data.routes:
        mass = float(routes.loc[routes["route"].astype(str).eq(route), "tonnes_per_year"].sum()) if not routes.empty else 0.0
        share = mass / total_plastic if total_plastic > 0 else np.nan
        out[f"route_share__{route}"] = share
        out[f"route_share_pct__{route}"] = 100.0 * share if math.isfinite(share) else np.nan
        route_shares[route] = share

    signature_payload = {
        "low_batch_location": low_location,
        "wind_share": round(wind_share, 8) if math.isfinite(wind_share) else None,
        "reuse_share": round(reuse_share, 8) if math.isfinite(reuse_share) else None,
        "bev_vehicle_km_share": round(bev_share, 8) if math.isfinite(bev_share) else None,
        "route_shares": {
            route: round(value, 8) if math.isfinite(value) else None
            for route, value in route_shares.items()
        },
    }
    signature_json = json.dumps(signature_payload, sort_keys=True)
    out["network_decision_signature"] = hashlib.sha1(signature_json.encode("utf-8")).hexdigest()[:16]
    return out


def solution_row(master: Any, model: Any, result: Any, label: str, bau: Any, policy: Any) -> Dict[str, Any]:
    validation = master.solution_validation(model, result)
    row = {
        "case_id": result.case_id,
        "case_label": label,
        "solver_success": result.success,
        "solver_message": result.message,
        "primary_objective": result.primary_objective,
        "selection_method": result.selection_method,
        "cost_gbp_per_year": float(result.metrics["cost"]),
        "cost_million_gbp_per_year": float(result.metrics["cost"]) / 1e6,
        "gwp_kgco2e_per_year": float(result.metrics["gwp"]),
        "gwp_tco2e_per_year": float(result.metrics["gwp"]) / 1000.0,
        "policy_cost_gbp_per_year": float(result.metrics["policy_cost"]),
        "circularity_loss_t_per_year": float(result.metrics["circularity_loss"]),
        "freshwater_m3_per_year": float(result.metrics["freshwater"]),
        "residual_t_per_year": float(result.metrics["residual"]),
        "recovered_t_per_year": float(result.metrics["recovered"]),
        "delta_cost_vs_bau_gbp": float(result.metrics["cost"] - bau.metrics["cost"]),
        "pct_cost_vs_bau": pct_change(float(result.metrics["cost"]), float(bau.metrics["cost"])),
        "delta_gwp_vs_bau_kgco2e": float(result.metrics["gwp"] - bau.metrics["gwp"]),
        "pct_gwp_vs_bau": pct_change(float(result.metrics["gwp"]), float(bau.metrics["gwp"])),
        "delta_cost_vs_policy_gbp": float(result.metrics["cost"] - policy.metrics["cost"]),
        "delta_gwp_vs_policy_kgco2e": float(result.metrics["gwp"] - policy.metrics["gwp"]),
        **validation,
        **decision_metrics(master, model, result),
    }
    return row


def epsilon_row(
    master: Any,
    model: Any,
    result: Any,
    base_row: Dict[str, Any],
    cost_anchor: Any,
    gwp_anchor: Any,
    bau: Any,
    policy: Any,
    vector_path: Path,
) -> Dict[str, Any]:
    row = dict(base_row)
    row.update({
        "case_id": result.case_id,
        "solution_file": str(vector_path),
        "cost_gbp_per_year": float(result.metrics["cost"]),
        "cost_million_gbp_per_year": float(result.metrics["cost"]) / 1e6,
        "gwp_kgco2e_per_year": float(result.metrics["gwp"]),
        "gwp_tco2e_per_year": float(result.metrics["gwp"]) / 1000.0,
        "policy_cost_gbp_per_year": float(result.metrics["policy_cost"]),
        "gwp_cap_slack_kgco2e": float(base_row["epsilon_cap_target_kgco2e"] - result.metrics["gwp"]),
        "gwp_cap_slack_pct_of_span": (
            100.0 * (base_row["epsilon_cap_target_kgco2e"] - result.metrics["gwp"])
            / max(cost_anchor.metrics["gwp"] - gwp_anchor.metrics["gwp"], 1e-12)
        ),
        "pct_cost_vs_cost_anchor": pct_change(float(result.metrics["cost"]), float(cost_anchor.metrics["cost"])),
        "gwp_reduction_vs_cost_anchor_kgco2e": float(cost_anchor.metrics["gwp"] - result.metrics["gwp"]),
        "gwp_reduction_vs_cost_anchor_tco2e": float(cost_anchor.metrics["gwp"] - result.metrics["gwp"]) / 1000.0,
        "pct_gwp_vs_bau": pct_change(float(result.metrics["gwp"]), float(bau.metrics["gwp"])),
        "delta_cost_vs_bau_gbp": float(result.metrics["cost"] - bau.metrics["cost"]),
        "distance_to_policy_gwp_kgco2e": float(result.metrics["gwp"] - policy.metrics["gwp"]),
        "distance_to_policy_cost_gbp": float(result.metrics["cost"] - policy.metrics["cost"]),
        "epsilon_cap_violation_vs_effective_kgco2e": max(
            0.0, float(result.metrics["gwp"] - base_row["epsilon_cap_effective_kgco2e"])
        ),
        "stage2_cost_minus_stage1_gbp": float(result.metrics["cost"] - base_row["stage1_cost_gbp"]),
        "lexicographic_cost_bound_violation_gbp": max(
            0.0,
            float(result.metrics["cost"] - base_row["stage1_cost_gbp"] - base_row["lexicographic_cost_tolerance_gbp"]),
        ),
        **master.solution_validation(model, result),
        **decision_metrics(master, model, result),
    })
    return row


# =============================================================================
# EPSILON GRID AND PARETO PROCESSING
# =============================================================================


def build_epsilon_caps(
    gwp_min_kg: float,
    gwp_cost_anchor_kg: float,
    global_points: int,
) -> np.ndarray:
    """Uniform independent epsilon grid based ONLY on the two payoff-table anchors."""
    require(global_points >= 3, "--points must be at least 3.")
    require(
        gwp_cost_anchor_kg + PARETO_DOMINANCE_GWP_TOL_KG >= gwp_min_kg,
        "Cost anchor has lower GWP than the GWP anchor; anchor solve audit failed.",
    )
    caps = np.linspace(float(gwp_min_kg), float(gwp_cost_anchor_kg), int(global_points))
    require(len(caps) == int(global_points), "Unexpected epsilon-grid length.")
    return np.asarray(caps, dtype=float)


def load_checkpoint() -> pd.DataFrame:
    path = TABLE_DIRECTORY / "epsilon_sweep.csv"
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def pareto_filter(frame: pd.DataFrame) -> pd.DataFrame:
    """Return non-dominated unique cost-GWP points for two minimisation objectives."""
    if frame.empty:
        return frame.copy()
    work = frame.copy()
    work["cost_gbp_per_year"] = pd.to_numeric(work["cost_gbp_per_year"], errors="coerce")
    work["gwp_kgco2e_per_year"] = pd.to_numeric(work["gwp_kgco2e_per_year"], errors="coerce")
    work = work.dropna(subset=["cost_gbp_per_year", "gwp_kgco2e_per_year"]).copy()
    work = work.sort_values(["gwp_kgco2e_per_year", "cost_gbp_per_year", "epsilon_cap_target_kgco2e"])

    keep_indices: List[Any] = []
    best_cost = math.inf
    for idx, row in work.iterrows():
        cost = float(row["cost_gbp_per_year"])
        if cost < best_cost - PARETO_DOMINANCE_COST_TOL_GBP:
            keep_indices.append(idx)
            best_cost = cost
        elif abs(cost - best_cost) <= PARETO_DOMINANCE_COST_TOL_GBP:
            # Same cost within tolerance: the earlier row has lower/equal GWP and dominates this one.
            continue

    front = work.loc[keep_indices].copy()
    front = front.sort_values("gwp_kgco2e_per_year").reset_index(drop=True)
    front.insert(0, "pareto_point_id", [f"P{i:03d}" for i in range(1, len(front) + 1)])
    return front


def add_front_metrics(front: pd.DataFrame, cost_anchor: Any, policy: Any, carbon_value: float) -> pd.DataFrame:
    if front.empty:
        return front
    out = front.copy().sort_values("gwp_kgco2e_per_year").reset_index(drop=True)
    out["gwp_tco2e_per_year"] = out["gwp_kgco2e_per_year"] / 1000.0
    out["cost_million_gbp_per_year"] = out["cost_gbp_per_year"] / 1e6
    out["abatement_vs_cost_anchor_tco2e"] = (
        float(cost_anchor.metrics["gwp"]) - out["gwp_kgco2e_per_year"]
    ) / 1000.0
    out["incremental_cost_vs_cost_anchor_gbp"] = (
        out["cost_gbp_per_year"] - float(cost_anchor.metrics["cost"])
    )
    out["weighted_objective_at_central_carbon_value_gbp"] = (
        out["cost_gbp_per_year"] + carbon_value * out["gwp_tco2e_per_year"]
    )
    out["weighted_objective_gap_vs_policy_gbp"] = (
        out["weighted_objective_at_central_carbon_value_gbp"] - float(policy.metrics["policy_cost"])
    )

    # Marginal slopes are easier to interpret when moving from the cost anchor toward tighter GWP.
    reverse = out.sort_values("gwp_kgco2e_per_year", ascending=False).copy()
    reverse["delta_gwp_tco2e_from_previous"] = (
        reverse["gwp_kgco2e_per_year"].shift(1) - reverse["gwp_kgco2e_per_year"]
    ) / 1000.0
    reverse["delta_cost_gbp_from_previous"] = (
        reverse["cost_gbp_per_year"] - reverse["cost_gbp_per_year"].shift(1)
    )
    reverse["marginal_abatement_cost_gbp_per_tco2e"] = (
        reverse["delta_cost_gbp_from_previous"] / reverse["delta_gwp_tco2e_from_previous"]
    )
    slope_lookup = reverse.set_index("pareto_point_id")[
        ["delta_gwp_tco2e_from_previous", "delta_cost_gbp_from_previous", "marginal_abatement_cost_gbp_per_tco2e"]
    ]
    out = out.merge(slope_lookup, left_on="pareto_point_id", right_index=True, how="left")
    return out


def policy_front_comparison(front: pd.DataFrame, policy: Any, carbon_value: float) -> pd.DataFrame:
    require(not front.empty, "Cannot compare policy point with an empty Pareto front.")
    p_cost = float(policy.metrics["cost"])
    p_gwp = float(policy.metrics["gwp"])

    cost_span = max(float(front["cost_gbp_per_year"].max() - front["cost_gbp_per_year"].min()), 1.0)
    gwp_span = max(float(front["gwp_kgco2e_per_year"].max() - front["gwp_kgco2e_per_year"].min()), 1.0)
    distance = np.sqrt(
        ((front["cost_gbp_per_year"] - p_cost) / cost_span) ** 2
        + ((front["gwp_kgco2e_per_year"] - p_gwp) / gwp_span) ** 2
    )
    nearest_index = distance.idxmin()
    nearest = front.loc[nearest_index]

    dominated_mask = (
        (front["cost_gbp_per_year"] <= p_cost + PARETO_DOMINANCE_COST_TOL_GBP)
        & (front["gwp_kgco2e_per_year"] <= p_gwp + PARETO_DOMINANCE_GWP_TOL_KG)
        & (
            (front["cost_gbp_per_year"] < p_cost - PARETO_DOMINANCE_COST_TOL_GBP)
            | (front["gwp_kgco2e_per_year"] < p_gwp - PARETO_DOMINANCE_GWP_TOL_KG)
        )
    )
    is_dominated = bool(dominated_mask.any())
    sampled_coincidence = (
        abs(float(nearest["cost_gbp_per_year"]) - p_cost) <= POLICY_EQUIV_COST_TOL_GBP
        and abs(float(nearest["gwp_kgco2e_per_year"]) - p_gwp) <= POLICY_EQUIV_GWP_TOL_KG
    )

    row = {
        "central_carbon_value_gbp_per_tco2e": carbon_value,
        "policy_cost_gbp_per_year": p_cost,
        "policy_gwp_kgco2e_per_year": p_gwp,
        "policy_gwp_tco2e_per_year": p_gwp / 1000.0,
        "policy_weighted_objective_gbp_per_year": float(policy.metrics["policy_cost"]),
        "nearest_pareto_point_id": str(nearest["pareto_point_id"]),
        "nearest_pareto_cost_gbp_per_year": float(nearest["cost_gbp_per_year"]),
        "nearest_pareto_gwp_kgco2e_per_year": float(nearest["gwp_kgco2e_per_year"]),
        "cost_difference_policy_minus_nearest_gbp": p_cost - float(nearest["cost_gbp_per_year"]),
        "gwp_difference_policy_minus_nearest_kgco2e": p_gwp - float(nearest["gwp_kgco2e_per_year"]),
        "normalised_objective_space_distance": float(distance.loc[nearest_index]),
        "policy_coincides_with_sampled_epsilon_point": bool(sampled_coincidence),
        "policy_dominated_by_sampled_epsilon_point": bool(is_dominated),
        "interpretation": (
            "The independently solved carbon-value point happens to coincide with one uniformly sampled epsilon point within tolerance."
            if sampled_coincidence else
            "No exact sampled coincidence is expected: the carbon-value solution was not used to construct the independent epsilon grid."
        ),
    }
    return pd.DataFrame([row])


def marginal_abatement_table(front: pd.DataFrame, carbon_value: float) -> pd.DataFrame:
    if front.empty:
        return pd.DataFrame()
    work = front.sort_values("gwp_kgco2e_per_year", ascending=False).copy().reset_index(drop=True)
    rows: List[Dict[str, Any]] = []
    for i in range(1, len(work)):
        previous = work.iloc[i - 1]
        current = work.iloc[i]
        abatement = (float(previous["gwp_kgco2e_per_year"]) - float(current["gwp_kgco2e_per_year"])) / 1000.0
        delta_cost = float(current["cost_gbp_per_year"] - previous["cost_gbp_per_year"])
        if abatement <= 1e-12:
            continue
        mac = delta_cost / abatement
        rows.append({
            "from_pareto_point": previous["pareto_point_id"],
            "to_pareto_point": current["pareto_point_id"],
            "from_gwp_tco2e_per_year": float(previous["gwp_kgco2e_per_year"]) / 1000.0,
            "to_gwp_tco2e_per_year": float(current["gwp_kgco2e_per_year"]) / 1000.0,
            "incremental_abatement_tco2e_per_year": abatement,
            "incremental_cost_gbp_per_year": delta_cost,
            "marginal_abatement_cost_gbp_per_tco2e": mac,
            "central_carbon_value_gbp_per_tco2e": carbon_value,
            "mac_minus_central_carbon_value": mac - carbon_value,
            "segment_supported_at_central_value": abs(mac - carbon_value) <= max(1.0, 0.01 * carbon_value),
        })
    return pd.DataFrame(rows)


# =============================================================================
# DETAILED PARETO DECISION EXPORT
# =============================================================================


def export_pareto_decisions(master: Any, model: Any, front: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    route_frames: List[pd.DataFrame] = []
    site_frames: List[pd.DataFrame] = []
    mix_frames: List[pd.DataFrame] = []

    for row in front.itertuples(index=False):
        vector_path = Path(str(row.solution_file))
        require(vector_path.exists(), f"Missing persisted Pareto solution vector: {vector_path}")
        result = result_from_vector(master, model, vector_path, case_id=str(row.pareto_point_id))
        routes, sites, mix = master.solution_decisions(model, result)
        for frame in (routes, sites, mix):
            frame.insert(0, "pareto_point_id", str(row.pareto_point_id))
            frame["epsilon_cap_target_kgco2e"] = float(row.epsilon_cap_target_kgco2e)
            frame["cost_gbp_per_year"] = float(row.cost_gbp_per_year)
            frame["gwp_kgco2e_per_year"] = float(row.gwp_kgco2e_per_year)
        route_frames.append(routes)
        site_frames.append(sites)
        mix_frames.append(mix)

    routes_all = pd.concat(route_frames, ignore_index=True) if route_frames else pd.DataFrame()
    sites_all = pd.concat(site_frames, ignore_index=True) if site_frames else pd.DataFrame()
    mix_all = pd.concat(mix_frames, ignore_index=True) if mix_frames else pd.DataFrame()

    atomic_write_csv(TABLE_DIRECTORY / "pareto_polymer_routing.csv", routes_all)
    atomic_write_csv(TABLE_DIRECTORY / "pareto_site_decisions.csv", sites_all)
    atomic_write_csv(TABLE_DIRECTORY / "pareto_route_mix.csv", mix_all)
    return routes_all, sites_all, mix_all


# =============================================================================
# INPUT AND CHECKPOINT HELPERS
# =============================================================================






















# =============================================================================
# RUN MANIFEST / INPUT TRACEABILITY
# =============================================================================


def input_hash_table(master: Any) -> pd.DataFrame:
    rows = [{
        "input_name": "master_script",
        "path": str(MASTER_SCRIPT),
        "exists": MASTER_SCRIPT.exists(),
        "sha256": sha256_file(MASTER_SCRIPT) if MASTER_SCRIPT.exists() else "",
        "size_bytes": MASTER_SCRIPT.stat().st_size if MASTER_SCRIPT.exists() else 0,
    }]
    for name, raw_path in master.INPUT_FILES.items():
        path = Path(raw_path)
        rows.append({
            "input_name": str(name),
            "path": str(path),
            "exists": path.exists(),
            "sha256": sha256_file(path) if path.exists() else "",
            "size_bytes": path.stat().st_size if path.exists() else 0,
        })
    return pd.DataFrame(rows)


def write_checkpoint_manifest(master_hash: str, args: argparse.Namespace, status: str, extra: Optional[Mapping[str, Any]] = None) -> None:
    payload: Dict[str, Any] = {
        "analysis": "epsilon-constraint cost-GWP Pareto comparison",
        "release": "THESIS_READY_EPSILON_INDEPENDENT_V2",
        "status": status,
        "timestamp": datetime.now().astimezone().isoformat(),
        "master_script": str(MASTER_SCRIPT),
        "master_script_sha256": master_hash,
        "global_epsilon_points_requested": int(args.points),
        "epsilon_grid_basis": "uniform between independently solved GWP-minimum and cost-minimum anchors",
        "epsilon_objective": "cost only",
        "epsilon_lifecycle_carbon_value_gbp_per_tco2e": 0.0,
        "carbon_value_used_to_construct_epsilon_grid": False,
        "lexicographic_tolerances": {
            "cost_abs_gbp": LEX_COST_TOL_ABS_GBP,
            "cost_rel": LEX_COST_TOL_REL,
            "gwp_abs_kgco2e": LEX_GWP_TOL_ABS_KG,
            "gwp_rel": LEX_GWP_TOL_REL,
        },
    }
    if extra:
        payload.update(dict(extra))
    atomic_write_text(RESULT_DIRECTORY / "run_manifest.json", json.dumps(payload, indent=2) + "\n")


def validate_resume(master_hash: str) -> None:
    manifest_path = RESULT_DIRECTORY / "run_manifest.json"
    if not manifest_path.exists():
        return
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return
    old_hash = str(payload.get("master_script_sha256", ""))
    require(
        not old_hash or old_hash == master_hash,
        "--resume requested, but run_master_MILP.py has changed since the checkpoint. "
        "Start a clean run to avoid mixing models.",
    )


# =============================================================================
# MAIN ANALYSIS
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace a cost-GWP epsilon-constraint Pareto front and compare the central carbon-value solution."
    )
    parser.add_argument("--points", type=int, default=DEFAULT_GLOBAL_EPSILON_POINTS,
                        help="Number of uniformly spaced epsilon caps between the independently solved GWP and cost anchors (default: 201).")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse compatible epsilon points already persisted in epsilon_sweep.csv.")
    parser.add_argument("--solver-output", action="store_true",
                        help="Show full Pyomo/GAMS output for every solve; otherwise logs remain persisted but console output is compact.")
    parser.add_argument("--keep-gams-files", action="store_true",
                        help="Keep GAMS temporary work files in addition to solver logs.")
    return parser.parse_args()


def run_analysis() -> Dict[str, Any]:
    args = parse_args()
    require(args.points >= 3, "--points must be at least 3.")

    reset_output_directories(resume=bool(args.resume))
    master = import_master_module()
    master_hash = sha256_file(MASTER_SCRIPT)
    if args.resume:
        validate_resume(master_hash)
    configure_master_solver(master, args)

    input_hashes = input_hash_table(master)
    atomic_write_csv(TABLE_DIRECTORY / "input_hashes.csv", input_hashes)
    write_checkpoint_manifest(master_hash, args, "initialising")

    print("\nEPSILON-CONSTRAINT PARETO ANALYSIS", flush=True)
    print(f"Master model:         {MASTER_SCRIPT}", flush=True)
    print(f"Results:              {RESULT_DIRECTORY}", flush=True)
    print(f"Configured solver:    {master.GAMS_SOLVER_NAME} via {master.GAMS_SOLVER_IO}", flush=True)
    print(f"Global epsilon caps:  {args.points}", flush=True)

    data = master.load_model_data()
    carbon_value = float(data.lifecycle_carbon_value_gbp_per_t)
    require(carbon_value > 0.0, "Central master model must contain a positive lifecycle carbon value.")

    # Central-policy model: unchanged master model, used ONLY for the carbon-value comparator.
    policy_model = master.build_linear_model(data)

    # Independent epsilon model: same physical MILP, but carbon value explicitly zero.
    # Epsilon solves use only the 'cost' and 'gwp' expressions; policy_cost is irrelevant.
    epsilon_data = copy.deepcopy(data)
    epsilon_data.lifecycle_carbon_value_gbp_per_t = 0.0
    model = master.build_linear_model(epsilon_data)
    validate_independent_epsilon_model(policy_model, model)

    bau = master.evaluate_bau(model)

    print(f"Central carbon-value comparator: £{carbon_value:,.6f}/tCO2e (master model)", flush=True)
    print("Epsilon frontier carbon value:  £0/tCO2e (cost-only objective; GWP enters only as constraint)", flush=True)
    print("Carbon-value GWP used in epsilon grid: NO", flush=True)

    # Four anchor solves + one independent carbon-value comparator + two solves per epsilon cap.
    provisional_progress = SolveProgress(total_solves=5 + 2 * args.points)

    print("\n[1/5] Solving single-objective anchors...", flush=True)
    cost_stage1, cost_anchor = solve_cost_anchor(master, model, provisional_progress, args)
    gwp_stage1, gwp_anchor = solve_gwp_anchor(master, model, provisional_progress, args)

    print("\n[2/5] Solving central carbon-value objective...", flush=True)
    policy = solve_and_track(
        master, policy_model,
        case_id="OPT_COST_PLUS_CARBON_VALUE_INDEPENDENT_COMPARATOR",
        primary="policy_cost",
        selection_method="minimum annual network cost plus central lifecycle carbon value",
        metadata={
            "central_carbon_value_gbp_per_tco2e": carbon_value,
            "role": "independent comparator only",
            "used_to_construct_epsilon_grid": False,
        },
        progress=provisional_progress,
        show_solver_output=args.solver_output,
    )
    require(policy.success, f"Central carbon-value solve failed: {policy.message}")

    # Persist anchor/reference vectors immediately.
    anchor_vectors = {
        "cost_stage1": VECTOR_DIRECTORY / "anchor_cost_stage1.npz",
        "cost_anchor": VECTOR_DIRECTORY / "anchor_cost_lex_gwp.npz",
        "gwp_stage1": VECTOR_DIRECTORY / "anchor_gwp_stage1.npz",
        "gwp_anchor": VECTOR_DIRECTORY / "anchor_gwp_lex_cost.npz",
        "policy": VECTOR_DIRECTORY / "central_carbon_value.npz",
        "bau": VECTOR_DIRECTORY / "bau_constructed.npz",
    }
    for key, result in [
        ("cost_stage1", cost_stage1), ("cost_anchor", cost_anchor),
        ("gwp_stage1", gwp_stage1), ("gwp_anchor", gwp_anchor),
        ("policy", policy), ("bau", bau),
    ]:
        save_solution_vector(anchor_vectors[key], result)

    anchors = pd.DataFrame([
        solution_row(master, model, bau, "BAU benchmark", bau, policy),
        solution_row(master, model, cost_anchor, "Cost-minimum anchor", bau, policy),
        solution_row(master, policy_model, policy, f"Central carbon value (£{carbon_value:,.0f}/tCO2e; independent comparator)", bau, policy),
        solution_row(master, model, gwp_anchor, "GWP-minimum anchor", bau, policy),
    ])
    atomic_write_csv(TABLE_DIRECTORY / "anchor_solutions.csv", anchors)

    require(
        cost_anchor.metrics["gwp"] + PARETO_DOMINANCE_GWP_TOL_KG >= gwp_anchor.metrics["gwp"],
        "Anchor inconsistency: the cost-minimum solution has lower GWP than the GWP-minimum solution.",
    )

    epsilon_caps = build_epsilon_caps(
        gwp_min_kg=float(gwp_stage1.metrics["gwp"]),
        gwp_cost_anchor_kg=float(cost_anchor.metrics["gwp"]),
        global_points=int(args.points),
    )

    # Correct the progress denominator now that cap de-duplication is known.
    provisional_progress.total_solves = 5 + 2 * len(epsilon_caps)
    print(
        f"\nRealised epsilon design: {len(epsilon_caps)} unique caps, "
        f"{2*len(epsilon_caps)} epsilon MILP solves + 5 anchor/policy solves.",
        flush=True,
    )
    print(
        f"Independent GWP-cap range: {gwp_stage1.metrics['gwp']/1000.0:,.3f} to "
        f"{cost_anchor.metrics['gwp']/1000.0:,.3f} tCO2e/year. "
        f"Carbon-value comparator GWP ({policy.metrics['gwp']/1000.0:,.3f} tCO2e/year) was NOT inserted into this grid.",
        flush=True,
    )

    write_checkpoint_manifest(master_hash, args, "anchors_complete", {
        "central_carbon_value_gbp_per_tco2e": carbon_value,
        "realised_unique_epsilon_caps": int(len(epsilon_caps)),
        "estimated_total_solver_calls": int(5 + 2 * len(epsilon_caps)),
        "anchor_cost_gbp_per_year": float(cost_anchor.metrics["cost"]),
        "anchor_cost_gwp_kgco2e_per_year": float(cost_anchor.metrics["gwp"]),
        "anchor_gwp_cost_gbp_per_year": float(gwp_anchor.metrics["cost"]),
        "anchor_gwp_kgco2e_per_year": float(gwp_anchor.metrics["gwp"]),
        "policy_cost_gbp_per_year": float(policy.metrics["cost"]),
        "policy_gwp_kgco2e_per_year": float(policy.metrics["gwp"]),
    })

    print("\n[3/5] Running dense epsilon-constraint sweep...", flush=True)
    checkpoint = load_checkpoint() if args.resume else pd.DataFrame()
    completed_keys = set()
    rows: List[Dict[str, Any]] = []
    target_cap_map = {cap_key(float(cap)): (ordinal, float(cap)) for ordinal, cap in enumerate(epsilon_caps, start=1)}
    if not checkpoint.empty and "cap_key" in checkpoint.columns:
        for row in checkpoint.to_dict("records"):
            key = str(row.get("cap_key", ""))
            solution_file = Path(str(row.get("solution_file", ""))) if row.get("solution_file") else None
            if (
                key in target_cap_map
                and row.get("point_status") == "optimal"
                and solution_file is not None
                and solution_file.exists()
            ):
                ordinal_now, cap_now = target_cap_map[key]
                row["epsilon_ordinal"] = ordinal_now
                row["epsilon_cap_target_kgco2e"] = cap_now
                row["epsilon_cap_target_tco2e"] = cap_now / 1000.0
                completed_keys.add(key)
                rows.append(row)
        if completed_keys:
            print(f"Resuming {len(completed_keys)} previously completed epsilon caps.", flush=True)
            # Count resumed Stage-1/Stage-2 calls as completed for ETA display only.
            provisional_progress.completed += 2 * len(completed_keys)

    for ordinal, cap in enumerate(epsilon_caps, start=1):
        key = cap_key(float(cap))
        if key in completed_keys:
            print(f"    cap {ordinal}/{len(epsilon_caps)} already complete: {cap/1000.0:,.3f} tCO2e/year", flush=True)
            continue

        print(
            f"\n  ε cap {ordinal}/{len(epsilon_caps)}: {cap/1000.0:,.3f} tCO2e/year",
            flush=True,
        )
        result, base_row = solve_epsilon_point(
            master, model, float(cap), ordinal, len(epsilon_caps), provisional_progress, args
        )
        if result is not None:
            vector_path = VECTOR_DIRECTORY / f"epsilon_{key}.npz"
            save_solution_vector(vector_path, result)
            row = epsilon_row(
                master, model, result, base_row,
                cost_anchor=cost_anchor, gwp_anchor=gwp_anchor,
                bau=bau, policy=policy, vector_path=vector_path,
            )
        else:
            row = base_row
        rows.append(row)
        current = pd.DataFrame(rows)
        current = current.sort_values("epsilon_cap_target_kgco2e").drop_duplicates("cap_key", keep="last")
        atomic_write_csv(TABLE_DIRECTORY / "epsilon_sweep.csv", current)
        write_checkpoint_manifest(master_hash, args, "epsilon_sweep_in_progress", {
            "central_carbon_value_gbp_per_tco2e": carbon_value,
            "realised_unique_epsilon_caps": int(len(epsilon_caps)),
            "completed_epsilon_caps": int(current["cap_key"].nunique()),
            "successful_epsilon_caps": int(current.get("point_status", pd.Series(dtype=str)).eq("optimal").sum()),
        })

    sweep = pd.DataFrame(rows)
    sweep = sweep.sort_values("epsilon_cap_target_kgco2e").drop_duplicates("cap_key", keep="last").reset_index(drop=True)
    atomic_write_csv(TABLE_DIRECTORY / "epsilon_sweep.csv", sweep)
    optimal_sweep = sweep.loc[sweep["point_status"].eq("optimal")].copy()
    require(not optimal_sweep.empty, "No successful epsilon solutions were produced.")

    print("\n[4/5] Extracting non-dominated frontier and decision audit...", flush=True)
    front = pareto_filter(optimal_sweep)
    front = add_front_metrics(front, cost_anchor, policy, carbon_value)
    atomic_write_csv(TABLE_DIRECTORY / "pareto_front.csv", front)

    comparison = policy_front_comparison(front, policy, carbon_value)
    atomic_write_csv(TABLE_DIRECTORY / "policy_front_comparison.csv", comparison)

    mac = marginal_abatement_table(front, carbon_value)
    atomic_write_csv(TABLE_DIRECTORY / "marginal_abatement_costs.csv", mac)

    routes_all, sites_all, mix_all = export_pareto_decisions(master, model, front)

    # Method-level quality checks: warn rather than destroy useful outputs.
    comparison_row = comparison.iloc[0]
    if bool(comparison_row["policy_dominated_by_sampled_epsilon_point"]):
        print(
            "WARNING: the sampled epsilon frontier contains a point that dominates the central carbon-value solution. "
            "Inspect solver tolerances/logs because a globally optimal positive-weighted solution should be Pareto-efficient.",
            flush=True,
        )

    manifest = {
        "analysis": "epsilon-constraint cost-GWP Pareto comparison",
        "release": "THESIS_READY_EPSILON_INDEPENDENT_V2",
        "status": "complete",
        "completed_at": datetime.now().astimezone().isoformat(),
        "method": {
            "primary_tradeoff": "annual network cost versus net life-cycle GWP",
            "epsilon_formulation": "minimise cost subject to GWP <= epsilon",
            "lexicographic_second_stage": (
                "minimise GWP subject to the same epsilon cap and cost <= Stage-1 optimum + numerical tolerance"
            ),
            "reason_for_lexicographic_stage": "remove weakly efficient cost-tie solutions",
            "global_epsilon_points_requested": int(args.points),
            "epsilon_grid_basis": "uniform spacing between independent GWP-minimum and cost-minimum anchors",
            "epsilon_primary_objective": "annual network cost only",
            "epsilon_constraint": "GWP <= epsilon",
            "epsilon_lifecycle_carbon_value_gbp_per_tco2e": 0.0,
            "realised_unique_epsilon_caps": int(len(epsilon_caps)),
            "carbon_value_used_to_construct_epsilon_grid": False,
            "policy_gwp_inserted_as_exact_epsilon_cap": False,
            "local_refinement_around_policy_solution": False,
        },
        "weighted_policy_objective": {
            "form": "cost + carbon_value_gbp_per_tCO2e * GWP_kgCO2e / 1000",
            "central_carbon_value_gbp_per_tco2e": carbon_value,
            "policy_cost_gbp_per_year": float(policy.metrics["cost"]),
            "policy_gwp_kgco2e_per_year": float(policy.metrics["gwp"]),
            "policy_weighted_objective_gbp_per_year": float(policy.metrics["policy_cost"]),
        },
        "anchors": {
            "cost_minimum_cost_gbp_per_year": float(cost_anchor.metrics["cost"]),
            "cost_minimum_gwp_kgco2e_per_year": float(cost_anchor.metrics["gwp"]),
            "gwp_minimum_cost_gbp_per_year": float(gwp_anchor.metrics["cost"]),
            "gwp_minimum_gwp_kgco2e_per_year": float(gwp_anchor.metrics["gwp"]),
        },
        "tolerances": {
            "lex_cost_abs_gbp": LEX_COST_TOL_ABS_GBP,
            "lex_cost_rel": LEX_COST_TOL_REL,
            "lex_gwp_abs_kgco2e": LEX_GWP_TOL_ABS_KG,
            "lex_gwp_rel": LEX_GWP_TOL_REL,
            "pareto_dominance_cost_tolerance_gbp": PARETO_DOMINANCE_COST_TOL_GBP,
            "pareto_dominance_gwp_tolerance_kgco2e": PARETO_DOMINANCE_GWP_TOL_KG,
            "policy_equivalence_cost_tolerance_gbp": POLICY_EQUIV_COST_TOL_GBP,
            "policy_equivalence_gwp_tolerance_kgco2e": POLICY_EQUIV_GWP_TOL_KG,
        },
        "solver": {
            "backend": master.GAMS_SOLVER_NAME,
            "solver_io": master.GAMS_SOLVER_IO,
            "mip_relative_gap": master.MIP_REL_GAP,
            "solver_report": str(master.GAMS_SOLVE_REPORT),
            "gams_log_directory": str(master.GAMS_LOG_DIR),
        },
        "counts": {
            "successful_epsilon_points": int(len(optimal_sweep)),
            "failed_epsilon_points": int(len(sweep) - len(optimal_sweep)),
            "sampled_non_dominated_pareto_points": int(len(front)),
            "marginal_abatement_segments": int(len(mac)),
            "pareto_routing_rows": int(len(routes_all)),
            "pareto_site_rows": int(len(sites_all)),
            "pareto_route_mix_rows": int(len(mix_all)),
        },
        "policy_front_comparison": comparison.iloc[0].to_dict(),
        "traceability": {
            "master_script": str(MASTER_SCRIPT),
            "master_script_sha256": master_hash,
            "input_hash_table": str(TABLE_DIRECTORY / "input_hashes.csv"),
        },
        "outputs": {
            "anchor_solutions": str(TABLE_DIRECTORY / "anchor_solutions.csv"),
            "epsilon_sweep": str(TABLE_DIRECTORY / "epsilon_sweep.csv"),
            "pareto_front": str(TABLE_DIRECTORY / "pareto_front.csv"),
            "marginal_abatement_costs": str(TABLE_DIRECTORY / "marginal_abatement_costs.csv"),
            "policy_front_comparison": str(TABLE_DIRECTORY / "policy_front_comparison.csv"),
            "pareto_polymer_routing": str(TABLE_DIRECTORY / "pareto_polymer_routing.csv"),
            "pareto_site_decisions": str(TABLE_DIRECTORY / "pareto_site_decisions.csv"),
            "pareto_route_mix": str(TABLE_DIRECTORY / "pareto_route_mix.csv"),
        },
    }
    atomic_write_text(RESULT_DIRECTORY / "run_manifest.json", json.dumps(manifest, indent=2, default=str) + "\n")

    print("\nEPSILON-CONSTRAINT PARETO ANALYSIS COMPLETE", flush=True)
    print(f"Successful epsilon points: {len(optimal_sweep):,}", flush=True)
    print(f"Non-dominated sampled points: {len(front):,}", flush=True)
    print(
        "Independent carbon-value comparator coincides with a sampled epsilon point: "
        f"{bool(comparison_row['policy_coincides_with_sampled_epsilon_point'])}",
        flush=True,
    )
    print("Carbon-value solution used to construct epsilon grid: False", flush=True)
    print(f"Tables:     {TABLE_DIRECTORY}", flush=True)
    print(f"Manifest:   {RESULT_DIRECTORY / 'run_manifest.json'}", flush=True)

    return {
        "master": master,
        "data": data,
        "model": model,
        "policy_model": policy_model,
        "bau": bau,
        "cost_anchor": cost_anchor,
        "gwp_anchor": gwp_anchor,
        "policy": policy,
        "epsilon_sweep": sweep,
        "pareto_front": front,
        "marginal_abatement_costs": mac,
        "policy_front_comparison": comparison,
        "result_directory": RESULT_DIRECTORY,
    }


if __name__ == "__main__":
    run_analysis()
