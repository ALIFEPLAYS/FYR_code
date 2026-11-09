from __future__ import annotations

"""Build publication-ready data tables for Figure 3.

This script is a pure post-processing step. It does NOT rebuild or solve the MILP.
It reads the persisted deterministic-scenario, OAT and Sobol-QMC outputs and writes
small, stable CSV tables intended to be read directly by TikZ/PGFPlots in Overleaf.

Outputs
-------
Overleaf inputs (written to MILP/results/data/figure_03):
1. figure_03_scenarios.csv
   Selected scenario cases, compact decision states and changes vs the central scenario.
2. figure_03_oat.csv
   Selected OAT decision boundaries with display-ready labels/units.
3. figure_03_sobol_display.csv
   Compact prevalence + decision-fingerprint table for the displayed archetypes.
4. figure_03_sobol_kpi.csv
   Long-format 5th/median/95th percentile KPI envelopes for PGFPlots.
5. figure_03_summary.csv
   Scalar headline statistics used by the Figure 3 panel headers.

Audit/provenance outputs (not required by Overleaf):
6. figure_03_sobol_archetypes.csv
   Full grouped Sobol archetype table for reproducibility/audit.
7. figure_03_source_manifest.csv
   Paths of the persisted result files used in the build.

The Sobol archetype rules intentionally reproduce the article grouping logic in
thesis_visuals/loaders.py:
  - decision shares <= 0.05 -> zero state
  - decision shares >= 0.95 -> full state
  - polymer route is dominant when top share >= 0.70 OR second share < 0.20
  - otherwise the two leading route families form a mixed route state

Usage
-----
Run this script from its fixed location at MILP/code:

    python build_figure_03_data.py

No command-line arguments are required. The script reads the current scenario, OAT,
and Sobol-QMC outputs under MILP/results and writes Figure 3 tables to
``MILP/results/data/figure_03``.

IMPORTANT: this script does NOT read values from ``03_experimentation_three_prongs.py``,
``loaders.py`` or any other Python source file at runtime. Those source files were used
only to reproduce the validated aggregation rules below. Runtime scientific inputs are
only the persisted CSV/JSON result files discovered under ``results/``.
"""

from pathlib import Path
import hashlib
import json
import math
import re
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "2026-10-22-submission-v1"

# -----------------------------------------------------------------------------
# Constants copied from the validated Figure 3 loader logic
# -----------------------------------------------------------------------------
SCENARIO_BASELINE_CASE_ID = "BASE_CURRENT_POLICY"
POLYMERS = ["LDPE", "LLDPE", "PP", "TPE", "EVOH", "Other"]

ARTICLE_SHARE_LOW = 0.05
ARTICLE_SHARE_HIGH = 0.95
ARTICLE_POLYMER_DOMINANT = 0.70
ARTICLE_POLYMER_SECONDARY = 0.20

ROUTE_DISPLAY = {
    "landfill_all": "Landfill",
    "incineration_efw_heat_electricity": "EfW",
    "mechanical_recycling_polyolefins": "Mechanical recycling",
    "mechanical_recycling_tpe_tubing": "TPE recycling",
    "chemical_recycling_strap_hypothetical": "STRAP",
    "mechanical_recycling_main_residual_incineration": "Mechanical recycling",
}

OAT_GROUP_ORDER = {
    "plastics": 0,
    "water": 1,
    "electricity": 2,
    "transport": 3,
    "material_composition": 4,
    "network": 5,
    "costs": 6,
    "policy": 7,
    "process_variability": 8,
    "throughput": 9,
    "discrete_availability": 10,
}


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------
def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return s.fillna("").astype(str).str.strip().str.lower().isin(
        {"1", "true", "yes", "y", "success", "optimal"}
    )


def clean_text(value: Any) -> str:
    """Keep CSV labels LaTeX-friendly without inserting LaTeX markup into data."""
    s = str(value if value is not None else "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.10g")



# -----------------------------------------------------------------------------
# Submission result paths
# -----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent


def resolve_project_root() -> Path:
    root = HERE.parent
    if not (root / "results").is_dir():
        raise FileNotFoundError(f"MILP results directory is missing: {root / 'results'}")
    return root


def discover_inputs(root: Path) -> dict[str, Path]:
    results = root / "results"
    scenario_dir = results / "industrial_symbiosis_scenario_results_20260729"
    oat_dir = results / "oat_decision_boundaries"
    sobol_dir = results / "stochastic_optimisation_solution_space"
    inputs = {
        "scenario_summary": scenario_dir / "scenario_case_summary.csv",
        "scenario_routing": scenario_dir / "scenario_polymer_routing.csv",
        "scenario_route_mix": scenario_dir / "scenario_route_mix.csv",
        "oat_boundaries": oat_dir / "decision_boundaries.csv",
        "sobol_evaluations": sobol_dir / "evaluations.csv",
        "sobol_central": sobol_dir / "central_solution.csv",
        "sobol_quality": sobol_dir / "run_quality_summary.csv",
        "sobol_manifest": sobol_dir / "run_manifest.json",
        "existing_grouped_archetypes": sobol_dir / "figures" / "article_grouped_decision_archetypes.csv",
    }
    for key, path in inputs.items():
        if key == "existing_grouped_archetypes":
            continue
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Required Figure 3 input is absent or empty: {path}")
    return inputs


def validate_sobol_run(inputs: dict[str, Path]) -> None:
    q = pd.read_csv(inputs["sobol_quality"])
    if q.empty or "thesis_run_valid" not in q.columns or "technical_exception_samples" not in q.columns:
        raise ValueError("Sobol-QMC run_quality_summary.csv lacks required validity fields")
    row = q.iloc[0]
    tech = pd.to_numeric(row["technical_exception_samples"], errors="raise")
    if pd.isna(tech) or tech > 0 or str(row["thesis_run_valid"]).strip().lower() not in {"1", "true", "yes", "y"}:
        raise ValueError("Sobol-QMC run_quality_summary.csv marks this run invalid for thesis use")

    manifest = json.loads(inputs["sobol_manifest"].read_text(encoding="utf-8"))
    if str(manifest.get("thesis_run_valid")).strip().lower() not in {"1", "true", "yes", "y"}:
        raise ValueError("Sobol-QMC run_manifest.json marks this run invalid for thesis use")

# -----------------------------------------------------------------------------
# Scenario processing
# -----------------------------------------------------------------------------
def route_state_from_group(group: pd.DataFrame) -> str:
    if group.empty:
        return "-"
    totals = group.groupby("route")["tonnes_per_year"].sum().sort_values(ascending=False)
    total = float(totals.sum())
    if total <= 1e-12:
        return "-"
    active = [(str(route), float(mass / total)) for route, mass in totals.items() if mass / total > 0.005]
    if not active:
        return "-"
    if len(active) == 1 or active[0][1] >= 0.995:
        return ROUTE_DISPLAY.get(active[0][0], active[0][0])
    return " + ".join(ROUTE_DISPLAY.get(route, route) for route, _ in active[:2])


def plastics_state(routes: pd.DataFrame) -> str:
    if routes.empty:
        return "-"
    recovery: list[str] = []
    for route in routes.groupby("route")["tonnes_per_year"].sum().sort_values(ascending=False).index:
        r = str(route)
        if "mechanical_recycling" in r or "strap" in r:
            recovery.append(ROUTE_DISPLAY.get(r, r))
    return " + ".join(dict.fromkeys(recovery)) if recovery else "No recovery route"


def residual_route_state(routes: pd.DataFrame) -> str:
    if routes.empty:
        return "-"
    mask = routes["route"].astype(str).isin(["landfill_all", "incineration_efw_heat_electricity"])
    return route_state_from_group(routes.loc[mask])


def hgv_state(mix: pd.DataFrame) -> tuple[str, float]:
    if mix.empty or "vehicle_km_per_year" not in mix.columns:
        return "-", np.nan
    g = mix.groupby("transport_mode")["vehicle_km_per_year"].sum()
    total = float(g.sum())
    if total <= 1e-12:
        return "-", np.nan
    bev = float(g.get("battery_electric_articulated_hgv", 0.0)) / total
    if bev <= 0.005:
        return "Diesel", bev
    if bev >= 0.995:
        return "BEV", bev
    return "Mixed", bev


def share_state(value: float, zero: str, full: str, mixed: str) -> str:
    if not np.isfinite(value):
        return "-"
    if value <= 0.005:
        return zero
    if value >= 0.995:
        return full
    return mixed


def scenario_decision_table(summary: pd.DataFrame, routes: pd.DataFrame, mix: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, r in summary.iterrows():
        cid = str(r.get("case_id", ""))
        rr = routes.loc[routes["case_id"].astype(str).eq(cid)].copy() if "case_id" in routes else pd.DataFrame()
        mm = mix.loc[mix["case_id"].astype(str).eq(cid)].copy() if "case_id" in mix else pd.DataFrame()
        hgv, bev = hgv_state(mm)
        wind = safe_float(r.get("wind_share"))
        reuse = safe_float(r.get("water_reuse_share"))
        rows.append(
            {
                **r.to_dict(),
                "scenario": clean_text(r.get("case_name", cid)),
                "electricity": share_state(wind, "Grid", "Wind", "Mix"),
                "water": share_state(reuse, "No reuse", "Full", "Part."),
                "plastics": plastics_state(rr),
                "residual_route": residual_route_state(rr),
                "hgv": hgv,
                "bev_vehicle_km_share": bev,
                "low_batch": str(r.get("low_batch_location", "")),
                "_source_order": idx,
            }
        )
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    hit = d.index[d["case_id"].astype(str).eq(SCENARIO_BASELINE_CASE_ID)]
    if len(hit):
        base = d.loc[hit[0]]
        cols = ["electricity", "plastics", "residual_route", "hgv", "water", "low_batch"]
        d["physical_decision_change_count"] = [
            sum(str(row.get(c, "")) != str(base.get(c, "")) for c in cols) for _, row in d.iterrows()
        ]
    else:
        d["physical_decision_change_count"] = np.nan
    return d


def select_scenarios(decisions: pd.DataFrame, max_rows: int = 6) -> pd.DataFrame:
    d = decisions.loc[~decisions["case_id"].astype(str).eq(SCENARIO_BASELINE_CASE_ID)].copy()
    if "success" in d:
        d = d.loc[bool_series(d["success"])].copy()
    label = d.get("scenario", d.get("case_name", d["case_id"])).fillna("").astype(str).str.lower()
    priority = np.zeros(len(d), dtype=float)
    patterns = [
        ("no carbon", 100), ("carbon value none", 100), ("cost only", 100),
        ("carbon cap", 92), ("paris", 92), ("high carbon", 88),
        ("strap", 84), ("chemical recycling", 84),
        ("landfill unavailable", 80), ("landfill ban", 80),
        ("drought", 76), ("water scarcity", 76),
        ("efw ets", 68), ("landfill tax", 66), ("renewable", 60),
    ]
    for pat, weight in patterns:
        priority = np.maximum(priority, np.where(label.str.contains(pat, regex=False), weight, 0))
    change = pd.to_numeric(d.get("physical_decision_change_count", 0), errors="coerce").fillna(0).to_numpy(float)
    d = d.assign(_priority=priority + 8 * change, _change=change)
    d = d.sort_values(["_priority", "_change", "_source_order"], ascending=[False, False, True], kind="stable").head(max_rows)

    base = decisions.loc[decisions["case_id"].astype(str).eq(SCENARIO_BASELINE_CASE_ID)]
    if base.empty:
        raise ValueError("Scenario summary does not contain BASE_CURRENT_POLICY.")
    base = base.iloc[0]

    def delta_pct(value: Any, ref: Any) -> float:
        v, b = safe_float(value), safe_float(ref)
        return 100 * (v - b) / abs(b) if np.isfinite(v) and np.isfinite(b) and abs(b) > 1e-12 else np.nan

    out = pd.DataFrame({
        "case_id": d["case_id"].astype(str),
        "case_name": d["scenario"].map(clean_text),
        "scenario_family": d.get("scenario_family", ""),
        "electricity_state": d["electricity"],
        "recovery_state": d["plastics"].map(
            lambda s: "STRAP" if "STRAP" in str(s) else (
                "MR" if "recycling" in str(s).lower() else clean_text(s)
            )
        ),
        "residual_state": d["residual_route"].replace({"Landfill": "LF"}),
        "hgv_state": d["hgv"],
        "water_state": d["water"],
        "low_batch_location": d.get("low_batch", ""),
        "decision_change_count": d["physical_decision_change_count"],
        "gwp": pd.to_numeric(d.get("gwp"), errors="coerce"),
        "cost": pd.to_numeric(d.get("cost"), errors="coerce"),
    })
    out["gwp_change_pct"] = [delta_pct(v, base.get("gwp")) for v in out["gwp"]]
    out["gwp_change_tco2e"] = (out["gwp"] - safe_float(base.get("gwp"))) / 1000.0
    out["cost_change_pct"] = [delta_pct(v, base.get("cost")) for v in out["cost"]]
    out["cost_change_gbp"] = out["cost"] - safe_float(base.get("cost"))
    out["cost_change_kgbp"] = out["cost_change_gbp"] / 1000.0
    return out.reset_index(drop=True)


# -----------------------------------------------------------------------------
# OAT processing
# -----------------------------------------------------------------------------
def oat_text(row: pd.Series) -> str:
    return " ".join([str(row.get("parameter_id", "")), str(row.get("parameter_label", "")), str(row.get("group", ""))]).lower()


def first_match(df: pd.DataFrame, patterns: list[str], used: set[int]) -> list[int]:
    mask = df.apply(lambda r: any(p in oat_text(r) for p in patterns), axis=1)
    hits = df.loc[mask & ~df.index.to_series().isin(used)]
    if hits.empty:
        return []
    return [int(hits.sort_values(["_abs", "parameter_label"], kind="stable").index[0])]


def select_oat(boundaries: pd.DataFrame, max_params: int = 8) -> pd.DataFrame:
    d = boundaries.copy()
    d["deviation_from_central_pct"] = pd.to_numeric(d["deviation_from_central_pct"], errors="coerce")
    d = d.loc[d["deviation_from_central_pct"].notna()].copy()
    d["_abs"] = d["deviation_from_central_pct"].abs()
    d = d.sort_values(["_abs", "parameter_id"], kind="stable").drop_duplicates("parameter_id").copy()
    d["group"] = d.get("group", "").astype(str)
    d["_group_order"] = d["group"].map(OAT_GROUP_ORDER).fillna(99)

    used: set[int] = set()
    selected: list[int] = []
    preferred_buckets = [
        ["renewable cppa", "cppa premium", "cppa discount", "renewable premium", "renewable discount", "wind premium", "renewable electricity"],
        ["electric hgv transport cost", "bev transport cost", "electric hgv cost", "battery electric hgv cost"],
        ["autoclave energy", "autoclave electricity", "autoclave"],
        ["polyolefin recycling capacity", "tpe recycling capacity", "recycling capacity"],
        ["treated-water recovery", "treated water recovery", "wastewater generation", "utility-water demand", "utility water demand"],
        ["tpe fraction", "ldpe fraction", "lldpe fraction", "pp fraction", "evoh fraction", "composition"],
    ]
    for patterns in preferred_buckets:
        for idx in first_match(d, patterns, used):
            selected.append(idx)
            used.add(idx)

    quotas = {
        "plastics": 2, "water": 2, "electricity": 1, "transport": 1,
        "material_composition": 2, "network": 1, "costs": 1, "policy": 1,
        "process_variability": 1, "throughput": 1, "discrete_availability": 1,
    }
    counts = pd.Series(d.loc[selected, "group"]).value_counts().to_dict() if selected else {}
    for _, row in d.sort_values(["_group_order", "_abs", "parameter_label"], kind="stable").iterrows():
        idx = int(row.name)
        if idx in used:
            continue
        grp = str(row.get("group", ""))
        if counts.get(grp, 0) >= quotas.get(grp, 1):
            continue
        selected.append(idx)
        used.add(idx)
        counts[grp] = counts.get(grp, 0) + 1
        if len(selected) >= max_params:
            break

    out = d.loc[selected].copy()
    out = out.sort_values(["_group_order", "_abs", "parameter_label"], kind="stable").head(max_params).reset_index(drop=True)
    out.insert(0, "display_order", np.arange(1, len(out) + 1))
    out["parameter_label"] = out["parameter_label"].map(clean_text)
    out["display_group"] = out["group"].replace({"material_composition": "Composition"}).str.replace("_", " ").str.title()
    out["unit_display"] = out["unit"].astype(str).replace({
        "fraction of input": "fraction",
        "× current LCIA factor": "x current LCIA factor",
        "GBP/vehicle-route-km": "GBP/vehicle-route-km",
        "m3/batch": "m3/batch",
        "fraction of eligible central feed": "fraction",
    })
    return out[[
        "display_order", "parameter_id", "parameter_label", "display_group", "group", "unit", "unit_display",
        "direction", "deviation_from_central_pct", "boundary_estimate", "central_value",
        "lower_bracket", "upper_bracket", "boundary_bracket_width", "boundary_status",
    ]]


# -----------------------------------------------------------------------------
# Sobol archetype processing — exact reproduction of the loader grouping rules
# -----------------------------------------------------------------------------
def classify_share(value: float, zero: str, full: str, mixed: str) -> str:
    if value <= ARTICLE_SHARE_LOW:
        return zero
    if value >= ARTICLE_SHARE_HIGH:
        return full
    return mixed


def short_route(route: str) -> str:
    mapping = {
        "mechanical_recycling_polyolefins": "polyolefin_recycling",
        "mechanical_recycling_tpe_tubing": "tpe_recycling",
        "incineration_efw_heat_electricity": "efw",
        "landfill_all": "landfill",
        "chemical_recycling_strap_hypothetical": "strap",
    }
    return mapping.get(str(route), str(route))


def polymer_route_state(row: dict[str, Any], polymer: str) -> str:
    prefix = f"decision__polymer_route_share__{polymer}__"
    items: list[tuple[float, str]] = []
    for key, value in row.items():
        key = str(key)
        if key.startswith(prefix):
            share = safe_float(value)
            if np.isfinite(share) and share > 1e-9:
                items.append((share, key[len(prefix):]))
    if not items:
        return "none"
    items.sort(reverse=True)
    top_share, top_route = items[0]
    second_share, second_route = items[1] if len(items) > 1 else (0.0, "")
    if top_share >= ARTICLE_POLYMER_DOMINANT or second_share < ARTICLE_POLYMER_SECONDARY:
        return short_route(top_route)
    return "+".join(sorted([short_route(top_route), short_route(second_route)]))


def grouped_payload(row: dict[str, Any]) -> dict[str, Any]:
    # Preserve the loader's alphabetical discovery order, then Figure 3 later uses POLYMERS.
    polymers = sorted({
        str(key).split("__", 3)[2]
        for key in row
        if str(key).startswith("decision__polymer_route_share__") and len(str(key).split("__", 3)) >= 4
    })
    return {
        "low_batch_location": str(row.get("decision__low_batch_location", "")),
        "electricity": classify_share(safe_float(row.get("decision__wind_share"), 0.0), "grid", "wind", "mixed"),
        "water": classify_share(safe_float(row.get("decision__water_reuse_share"), 0.0), "no_reuse", "full_reuse", "partial_reuse"),
        "transport": classify_share(safe_float(row.get("decision__bev_vehicle_km_share"), 0.0), "diesel", "battery_electric", "mixed"),
        "polymer_routes": [(p, polymer_route_state(row, p)) for p in polymers],
    }


def ensure_grouped_archetypes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    for c in ["grouped_archetype", "grouped_archetype_json", "grouped_archetype_snapshot"]:
        if c not in out.columns:
            out[c] = ""
    success = bool_series(out["success"]) if "success" in out else pd.Series(True, index=out.index)
    for idx in out.index[success]:
        existing = str(out.at[idx, "grouped_archetype"]).strip()
        if existing not in {"", "nan", "None"}:
            continue
        payload = grouped_payload(out.loc[idx].to_dict())
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        key = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
        out.at[idx, "grouped_archetype"] = key
        out.at[idx, "grouped_archetype_json"] = raw
        route_text = ", ".join(f"{p}:{r}" for p, r in payload["polymer_routes"])
        out.at[idx, "grouped_archetype_snapshot"] = (
            f"low node={payload['low_batch_location']}; electricity={payload['electricity']}; "
            f"water={payload['water']}; transport={payload['transport']}; {route_text}"
        )
    return out


def group_summary(base: pd.DataFrame, central: pd.DataFrame) -> pd.DataFrame:
    if base.empty:
        return pd.DataFrame()
    central_key = str(central.iloc[0].get("grouped_archetype", "")) if not central.empty else ""
    rows: list[dict[str, Any]] = []
    counts = base["grouped_archetype"].astype(str).value_counts()
    numeric_decision_cols = [
        c for c in base.columns if c.startswith("decision__") and pd.api.types.is_numeric_dtype(base[c])
    ]
    for key, count in counts.items():
        g = base.loc[base["grouped_archetype"].astype(str).eq(str(key))]
        payload_text = str(g["grouped_archetype_json"].mode().iloc[0]) if "grouped_archetype_json" in g and len(g) else "{}"
        try:
            payload = json.loads(payload_text)
        except Exception:
            payload = grouped_payload(g.iloc[0].to_dict())
        row: dict[str, Any] = {
            "grouped_archetype": key,
            "sample_count": int(count),
            "sample_share_feasible": float(count / len(base)),
            "n_exact_macro_regimes": int(g["macro_regime"].nunique()) if "macro_regime" in g else 0,
            "is_central_archetype": str(key) == central_key,
            "low_batch_location": payload.get("low_batch_location", ""),
            "electricity_state": payload.get("electricity", ""),
            "water_state": payload.get("water", ""),
            "transport_state": payload.get("transport", ""),
            "grouped_archetype_json": payload_text,
        }
        route_dict = dict(payload.get("polymer_routes", []))
        for polymer in POLYMERS:
            row[f"route_state__{polymer}"] = route_dict.get(polymer, "none")
        for metric in ["gwp", "fossil_resource_use", "water_scarcity", "human_health_daly", "ecosystem_quality", "cost"]:
            if metric in g:
                x = pd.to_numeric(g[metric], errors="coerce").dropna()
                if len(x):
                    row[f"{metric}__p05"] = float(x.quantile(0.05))
                    row[f"{metric}__median"] = float(x.median())
                    row[f"{metric}__p95"] = float(x.quantile(0.95))
        for col in numeric_decision_cols:
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            if len(vals):
                row[f"median__{col}"] = float(vals.median())
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["sample_count", "grouped_archetype"], ascending=[False, True]).reset_index(drop=True)
    out.insert(0, "archetype_label", [f"A{i}" for i in range(1, len(out) + 1)])
    out["cumulative_sample_share_feasible"] = out["sample_share_feasible"].cumsum()
    return out


def build_sobol_display(groups: pd.DataFrame, base: pd.DataFrame, central: pd.DataFrame, top_n: int = 4) -> pd.DataFrame:
    g = groups.sort_values("sample_count", ascending=False).reset_index(drop=True)
    shown = g.head(top_n).copy()
    central_group = g.loc[g.get("is_central_archetype", False).astype(bool)] if "is_central_archetype" in g else pd.DataFrame()
    if len(central_group) and not shown["grouped_archetype"].astype(str).eq(str(central_group.iloc[0]["grouped_archetype"])).any():
        shown = pd.concat([shown, central_group.head(1)], ignore_index=True)
    shown = shown.drop_duplicates("grouped_archetype").reset_index(drop=True)

    keys = set(shown["grouped_archetype"].astype(str))
    hidden = g.loc[~g["grouped_archetype"].astype(str).isin(keys)]
    if len(hidden):
        other = base.loc[base["grouped_archetype"].astype(str).isin(hidden["grouped_archetype"].astype(str))]
        row: dict[str, Any] = {
            "archetype_label": "Other",
            "grouped_archetype": "__OTHER__",
            "sample_count": len(other),
            "sample_share_feasible": len(other) / len(base) if len(base) else np.nan,
            "n_exact_macro_regimes": int(other["macro_regime"].nunique()) if "macro_regime" in other else 0,
            "is_central_archetype": False,
            "low_batch_location": "varied",
            "electricity_state": "varied",
            "water_state": "varied",
            "transport_state": "varied",
        }
        for p in POLYMERS:
            row[f"route_state__{p}"] = "varied"
        for m in ["gwp", "fossil_resource_use", "water_scarcity", "human_health_daly", "ecosystem_quality", "cost"]:
            x = pd.to_numeric(other.get(m), errors="coerce").dropna()
            if len(x):
                row[f"{m}__p05"] = x.quantile(0.05)
                row[f"{m}__median"] = x.median()
                row[f"{m}__p95"] = x.quantile(0.95)
        shown = pd.concat([shown, pd.DataFrame([row])], ignore_index=True, sort=False)

    # Display labels 1..N for concrete archetypes; central is flagged separately.
    concrete = shown["grouped_archetype"].astype(str).ne("__OTHER__")
    labels = []
    j = 1
    for is_concrete in concrete:
        if is_concrete:
            labels.append(str(j))
            j += 1
        else:
            labels.append("Other")
    shown["display_label"] = labels

    # Add compact codes specifically for LaTeX/TikZ.
    shown["site_code"] = shown["low_batch_location"].map(lambda x: "VAR" if str(x) == "varied" else str(x)[:3].upper())
    shown["electricity_code"] = shown["electricity_state"].map({"wind": "W", "grid": "G", "mixed": "Mix", "varied": "var."}).fillna("-")
    shown["water_code"] = shown.apply(
        lambda r: (
            "var." if r.get("water_state") == "varied" else
            (f"{100*safe_float(r.get('median__decision__water_reuse_share')):.0f}%" if np.isfinite(safe_float(r.get("median__decision__water_reuse_share"))) else
             {"full_reuse": "100%", "partial_reuse": "Part.", "no_reuse": "0%"}.get(str(r.get("water_state")), "-"))
        ), axis=1
    )
    shown["hgv_code"] = shown.apply(
        lambda r: (
            "var." if r.get("transport_state") == "varied" else
            ("D" if safe_float(r.get("median__decision__bev_vehicle_km_share")) < 0.05 else
             "BEV" if safe_float(r.get("median__decision__bev_vehicle_km_share")) > 0.95 else
             f"{100*safe_float(r.get('median__decision__bev_vehicle_km_share')):.0f}%B")
            if np.isfinite(safe_float(r.get("median__decision__bev_vehicle_km_share"))) else
            {"diesel": "D", "battery_electric": "BEV", "mixed": "Mix"}.get(str(r.get("transport_state")), "-")
        ), axis=1
    )
    route_code = {
        "polyolefin_recycling": "MR",
        "tpe_recycling": "TPE-R",
        "efw": "EfW",
        "landfill": "LF",
        "strap": "STRAP",
        "none": "-",
        "varied": "var.",
    }
    for p in POLYMERS:
        shown[f"route_code__{p}"] = shown[f"route_state__{p}"].map(lambda s: route_code.get(str(s).split("+")[0], str(s)[:5]))

    shown["sample_share_feasible_pct"] = 100.0 * pd.to_numeric(shown["sample_share_feasible"], errors="coerce")

    # Precompute changes vs the deterministic central solution so PGFPlots does
    # not need to repeat scientific arithmetic in LaTeX.
    if central.empty:
        raise ValueError("central_solution.csv is empty; Sobol KPI changes cannot be computed.")
    c0 = central.iloc[0]
    for metric in ["gwp", "fossil_resource_use", "water_scarcity", "human_health_daly", "ecosystem_quality", "cost"]:
        c = safe_float(c0.get(metric))
        if not np.isfinite(c) or abs(c) <= 1e-12:
            continue
        for stat in ["p05", "median", "p95"]:
            col = f"{metric}__{stat}"
            shown[f"{metric}__{stat}_change_pct"] = 100.0 * (pd.to_numeric(shown[col], errors="coerce") - c) / abs(c)
        shown[f"{metric}__median_delta"] = pd.to_numeric(shown[f"{metric}__median"], errors="coerce") - c

    # Put central flag near the front and keep only useful display/audit fields.
    front = [
        "display_label", "archetype_label", "grouped_archetype", "is_central_archetype",
        "sample_count", "sample_share_feasible", "sample_share_feasible_pct", "n_exact_macro_regimes",
        "low_batch_location", "site_code", "electricity_state", "electricity_code",
        "water_state", "water_code", "transport_state", "hgv_code",
    ]
    route_cols = [x for p in POLYMERS for x in (f"route_state__{p}", f"route_code__{p}")]
    kpi_cols = [
        x
        for m in ["gwp", "fossil_resource_use", "water_scarcity", "human_health_daly", "ecosystem_quality", "cost"]
        for x in (
            f"{m}__p05", f"{m}__median", f"{m}__p95",
            f"{m}__p05_change_pct", f"{m}__median_change_pct", f"{m}__p95_change_pct",
            f"{m}__median_delta",
        )
    ]
    return shown[[c for c in front + route_cols + kpi_cols if c in shown.columns]].reset_index(drop=True)



# -----------------------------------------------------------------------------
# LaTeX/PGFPlots-specific compact Sobol tables
# -----------------------------------------------------------------------------
def compact_sobol_display(display: pd.DataFrame) -> pd.DataFrame:
    """Keep only the decision/prevalence fields needed by the TikZ fingerprint.

    The returned table also contains display-friendly fields used by the final
    LaTeX panel, while avoiding literal percent signs that would otherwise be
    awkward inside pgfplotstable-loaded CSVs.
    """
    rename = {
        "is_central_archetype": "is_central",
        "sample_share_feasible_pct": "feasible_share_pct",
        "n_exact_macro_regimes": "exact_regimes",
        "site_code": "site",
        "electricity_code": "electricity",
        "water_code": "water_reuse",
        "hgv_code": "hgv",
        "route_code__LDPE": "route_ldpe",
        "route_code__LLDPE": "route_lldpe",
        "route_code__PP": "route_pp",
        "route_code__TPE": "route_tpe",
        "route_code__EVOH": "route_evoh",
        "route_code__Other": "route_other",
    }
    cols = [
        "display_label", "grouped_archetype", "is_central_archetype", "sample_count",
        "sample_share_feasible_pct", "n_exact_macro_regimes", "site_code", "electricity_code",
        "water_code", "hgv_code", "route_code__LDPE", "route_code__LLDPE", "route_code__PP",
        "route_code__TPE", "route_code__EVOH", "route_code__Other",
    ]
    out = display[[c for c in cols if c in display.columns]].copy().rename(columns=rename)
    out.insert(0, "display_order", np.arange(1, len(out) + 1))

    def _water_disp(v: Any) -> str:
        s = clean_text(v)
        if not s or s.lower() == "nan":
            return "var."
        return s

    def _hgv_disp(v: Any) -> str:
        s = clean_text(v)
        if not s or s.lower() == "nan":
            return "var."
        if s == "D":
            return "Diesel"
        if s == "BEV":
            return "BEV"
        if s.endswith("B") and s[:-1].replace(".", "", 1).isdigit():
            return s[:-1]
        return s

    out["water_reuse_display"] = out.get("water_reuse", "").map(_water_disp)
    out["hgv_display"] = out.get("hgv", "").map(_hgv_disp)
    return out


def sobol_kpi_long(display: pd.DataFrame) -> pd.DataFrame:
    """Long-format KPI table so each Figure 3(c) envelope is trivial in PGFPlots."""
    metric_meta = [
        ("gwp", "GWP", "kg CO2e/yr"),
        ("fossil_resource_use", "Fossil resources", "kg oil-eq/yr"),
        ("water_scarcity", "Water scarcity", "m3 world-eq/yr"),
        ("human_health_daly", "Human health", "DALY/yr"),
        ("ecosystem_quality", "Ecosystem quality", "species·yr/yr"),
        ("cost", "Annual cost", "GBP/yr"),
    ]
    rows: list[dict[str, Any]] = []
    for order, (_, r) in enumerate(display.iterrows(), start=1):
        for metric, label, unit in metric_meta:
            rows.append({
                "display_order": order,
                "display_label": r.get("display_label", ""),
                "is_central": bool(r.get("is_central_archetype", False)),
                "metric": metric,
                "metric_label": label,
                "unit": unit,
                "p05": r.get(f"{metric}__p05", np.nan),
                "median": r.get(f"{metric}__median", np.nan),
                "p95": r.get(f"{metric}__p95", np.nan),
                "p05_change_pct": r.get(f"{metric}__p05_change_pct", np.nan),
                "median_change_pct": r.get(f"{metric}__median_change_pct", np.nan),
                "p95_change_pct": r.get(f"{metric}__p95_change_pct", np.nan),
                "median_delta": r.get(f"{metric}__median_delta", np.nan),
            })
    return pd.DataFrame(rows)


def make_overleaf_safe_tables(sobol_display: pd.DataFrame, sobol_kpis: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return TeX-safe Figure 3(c) tables for direct pgfplotstable use.

    The safe variants retain the same scientific content but avoid fragile string
    formatting inside CSVs, for example literal percent signs in categorical
    fields. The final LaTeX Figure 3 panels should read these safe files.
    """
    display_safe = sobol_display.copy()
    kpi_safe = sobol_kpis.copy()

    # Keep text columns tidy and free from repeated whitespace.
    for col in display_safe.columns:
        if display_safe[col].dtype == object:
            display_safe[col] = display_safe[col].map(clean_text)
    for col in kpi_safe.columns:
        if kpi_safe[col].dtype == object:
            kpi_safe[col] = kpi_safe[col].map(clean_text)

    # Add the scaled delta used by the existing Figure 3(c) LaTeX implementation.
    delta_map = {
        "gwp": "t",
        "fossil_resource_use": "t",
        "water_scarcity": "k_m3",
        "human_health_daly": "DALY",
        "ecosystem_quality": "species.yr",
        "cost": "kGBP",
    }
    kpi_safe["median_delta_scaled"] = kpi_safe["median_delta"]
    kpi_safe.loc[kpi_safe["metric"].eq("gwp"), "median_delta_scaled"] = pd.to_numeric(
        kpi_safe.loc[kpi_safe["metric"].eq("gwp"), "median_delta"], errors="coerce"
    ) / 1000.0
    kpi_safe.loc[kpi_safe["metric"].eq("fossil_resource_use"), "median_delta_scaled"] = pd.to_numeric(
        kpi_safe.loc[kpi_safe["metric"].eq("fossil_resource_use"), "median_delta"], errors="coerce"
    ) / 1000.0
    kpi_safe.loc[kpi_safe["metric"].eq("water_scarcity"), "median_delta_scaled"] = pd.to_numeric(
        kpi_safe.loc[kpi_safe["metric"].eq("water_scarcity"), "median_delta"], errors="coerce"
    ) / 1000.0
    kpi_safe.loc[kpi_safe["metric"].eq("cost"), "median_delta_scaled"] = pd.to_numeric(
        kpi_safe.loc[kpi_safe["metric"].eq("cost"), "median_delta"], errors="coerce"
    ) / 1000.0
    kpi_safe["delta_unit_code"] = kpi_safe["metric"].map(delta_map).fillna("")

    return display_safe, kpi_safe


def source_manifest(inputs: dict[str, Path], root: Path) -> pd.DataFrame:
    """Human-readable provenance table; retained with analysis outputs, not needed by Overleaf."""
    rows = []
    for key, path in inputs.items():
        if key in {"sobol_quality", "sobol_manifest", "existing_grouped_archetypes"} and not path.exists():
            continue
        rows.append({
            "input_role": key,
            "path_relative_to_project": str(path.resolve().relative_to(root.resolve())) if path.exists() else str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else np.nan,
        })
    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# Summary statistics
# -----------------------------------------------------------------------------
def build_summary(
    scenario_decisions: pd.DataFrame,
    oat_boundaries: pd.DataFrame,
    oat_display: pd.DataFrame,
    evals: pd.DataFrame,
    base_all: pd.DataFrame,
    base: pd.DataFrame,
    groups: pd.DataFrame,
    central: pd.DataFrame,
) -> pd.DataFrame:
    scen_nonbase = scenario_decisions.loc[~scenario_decisions["case_id"].astype(str).eq(SCENARIO_BASELINE_CASE_ID)]
    changed = int((pd.to_numeric(scen_nonbase.get("physical_decision_change_count", 0), errors="coerce").fillna(0) > 0).sum())
    n_params = int(oat_boundaries["parameter_id"].nunique()) if "parameter_id" in oat_boundaries else len(oat_boundaries)
    # The Figure 3 headline follows the legacy display ordering, i.e. the first
    # displayed OAT boundary, not the globally smallest absolute boundary.
    nearest = oat_display.iloc[0] if len(oat_display) else pd.Series(dtype=object)

    n_attempted = len(base_all)
    n_feasible = len(base)
    n_exact_regimes = int(base["macro_regime"].nunique()) if "macro_regime" in base else np.nan
    return pd.DataFrame([
        {
            "scenario_cases_nonbaseline": len(scen_nonbase),
            "scenario_cases_with_physical_change": changed,
            "oat_unique_parameters": n_params,
            "oat_nearest_parameter": nearest.get("parameter_label", ""),
            "oat_nearest_deviation_pct": nearest.get("deviation_from_central_pct", np.nan),
            "sobol_attempted": n_attempted,
            "sobol_feasible": n_feasible,
            "sobol_feasible_pct": 100 * n_feasible / n_attempted if n_attempted else np.nan,
            "sobol_exact_regimes": n_exact_regimes,
            "sobol_grouped_archetypes": len(groups),
            "central_gwp": safe_float(central.iloc[0].get("gwp")) if len(central) else np.nan,
            "central_fossil_resource_use": safe_float(central.iloc[0].get("fossil_resource_use")) if len(central) else np.nan,
            "central_water_scarcity": safe_float(central.iloc[0].get("water_scarcity")) if len(central) else np.nan,
            "central_human_health_daly": safe_float(central.iloc[0].get("human_health_daly")) if len(central) else np.nan,
            "central_ecosystem_quality": safe_float(central.iloc[0].get("ecosystem_quality")) if len(central) else np.nan,
            "central_cost": safe_float(central.iloc[0].get("cost")) if len(central) else np.nan,
        }
    ])


# -----------------------------------------------------------------------------
# Main — zero arguments; discover everything automatically
# -----------------------------------------------------------------------------
def main() -> None:
    root = resolve_project_root()
    inputs = discover_inputs(root)
    validate_sobol_run(inputs)

    outdir = root / "results" / "data" / "figure_03"
    outdir.mkdir(parents=True, exist_ok=True)

    scenario_summary = pd.read_csv(inputs["scenario_summary"])
    scenario_routing = pd.read_csv(inputs["scenario_routing"])
    scenario_mix = pd.read_csv(inputs["scenario_route_mix"])
    oat_boundaries = pd.read_csv(inputs["oat_boundaries"])
    evaluations = pd.read_csv(inputs["sobol_evaluations"])
    central = pd.read_csv(inputs["sobol_central"])

    # (a) Deterministic scenarios — same six-case display selection as Figure 3.
    scenario_decisions = scenario_decision_table(scenario_summary, scenario_routing, scenario_mix)
    scenario_display = select_scenarios(scenario_decisions, max_rows=6)

    # (b) OAT — same eight-parameter publication subset as Figure 3.
    oat_display = select_oat(oat_boundaries, max_params=8)

    # (c) Sobol-QMC — reproduce the authoritative grouped-archetype logic.
    evaluations = ensure_grouped_archetypes(evaluations)
    central = ensure_grouped_archetypes(central)
    base_all = (
        evaluations.loc[evaluations["analysis_case"].astype(str).eq("BASE")].copy()
        if "analysis_case" in evaluations else evaluations.copy()
    )
    base = base_all.loc[bool_series(base_all["success"])].copy() if "success" in base_all else base_all.copy()
    groups = group_summary(base, central)
    sobol_display_wide = build_sobol_display(groups, base, central, top_n=4)
    sobol_display = compact_sobol_display(sobol_display_wide)
    sobol_kpis = sobol_kpi_long(sobol_display_wide)
    sobol_display_safe, sobol_kpis_safe = make_overleaf_safe_tables(sobol_display, sobol_kpis)

    summary = build_summary(
        scenario_decisions, oat_boundaries, oat_display, evaluations, base_all, base, groups, central
    )
    manifest = source_manifest(inputs, root)

    # Stable outputs. The first five are the only files the final Overleaf Figure 3 needs.
    paths = {
        "scenarios": outdir / "figure_03_scenarios.csv",
        "oat": outdir / "figure_03_oat.csv",
        "sobol_display": outdir / "figure_03_sobol_display.csv",
        "sobol_display_safe": outdir / "figure_03_sobol_display_safe.csv",
        "sobol_kpi": outdir / "figure_03_sobol_kpi.csv",
        "sobol_kpi_safe": outdir / "figure_03_sobol_kpi_safe.csv",
        "summary": outdir / "figure_03_summary.csv",
        # Audit/provenance outputs retained with the analysis but not required by LaTeX.
        "sobol_archetypes_audit": outdir / "figure_03_sobol_archetypes.csv",
        "source_manifest": outdir / "figure_03_source_manifest.csv",
    }
    write_csv(scenario_display, paths["scenarios"])
    write_csv(oat_display, paths["oat"])
    write_csv(sobol_display, paths["sobol_display"])
    write_csv(sobol_display_safe, paths["sobol_display_safe"])
    write_csv(sobol_kpis, paths["sobol_kpi"])
    write_csv(sobol_kpis_safe, paths["sobol_kpi_safe"])
    write_csv(summary, paths["summary"])
    write_csv(groups, paths["sobol_archetypes_audit"])
    write_csv(manifest, paths["source_manifest"])

    # The box plot uses the same five displayed archetypes and central baseline.
    box_rows = []
    for _, shown in sobol_display.iterrows():
        group_id = str(shown["grouped_archetype"])
        sample = base if group_id == "__OTHER__" else base.loc[base["grouped_archetype"].astype(str).eq(group_id)]
        if group_id == "__OTHER__":
            explicit = set(sobol_display.loc[sobol_display["grouped_archetype"].astype(str).ne("__OTHER__"), "grouped_archetype"].astype(str))
            sample = base.loc[~base["grouped_archetype"].astype(str).isin(explicit)]
        for metric in ("gwp", "fossil_resource_use", "water_scarcity", "cost"):
            central_value = safe_float(central.iloc[0].get(metric))
            values = pd.to_numeric(sample.get(metric, pd.Series(dtype=float)), errors="coerce").dropna()
            if values.empty or not np.isfinite(central_value) or abs(central_value) <= 1e-12:
                continue
            pct = 100 * (values - central_value) / abs(central_value)
            box_rows.append({"display_label": shown["display_label"], "metric": metric,
                             "sample_count": len(values),
                             **{f"{q}_change_pct": pct.quantile(frac) for q, frac in
                                (("p05", .05), ("q25", .25), ("median", .5), ("q75", .75), ("p95", .95))}})
    write_csv(pd.DataFrame(box_rows), outdir / "figure_03_sobol_kpi_box_safe.csv")

    # Cross-check against an existing legacy grouped-archetype table if present.
    legacy = inputs["existing_grouped_archetypes"]
    if legacy.is_file() and legacy.stat().st_size:
        old = pd.read_csv(legacy)
        if "sample_count" in old.columns:
            old_counts = sorted(pd.to_numeric(old["sample_count"], errors="coerce").dropna().astype(int).tolist(), reverse=True)
            new_counts = sorted(pd.to_numeric(groups["sample_count"], errors="coerce").dropna().astype(int).tolist(), reverse=True)
            if old_counts != new_counts:
                raise ValueError(
                    "Rebuilt Sobol grouped-archetype counts do not match the persisted legacy article grouping table. "
                    "Refusing to write a silently inconsistent Figure 3."
                )

    print(f"\nFIGURE 3 DATA BUILD COMPLETE — {SCRIPT_VERSION}")
    print(f"  project root: {root}")
    print(f"  output dir:   {outdir}")
    print(f"  scenario rows: {len(scenario_display)}")
    print("  overleaf data inputs: figure_03_scenarios.csv, figure_03_oat.csv, figure_03_sobol_display_safe.csv, figure_03_sobol_kpi_safe.csv, figure_03_summary.csv")
    print(f"  OAT rows:      {len(oat_display)}")
    print(f"  Sobol attempted / feasible: {len(base_all):,} / {len(base):,}")
    print(f"  exact regimes: {int(base['macro_regime'].nunique()) if 'macro_regime' in base else 0:,}")
    print(f"  grouped archetypes: {len(groups):,}")
    print("  displayed Sobol prevalence:")
    for _, r in sobol_display.iterrows():
        mark = " [central]" if bool(r.get("is_central", False)) else ""
        print(f"    {str(r['display_label']):>5s}: n={int(r['sample_count']):4d}, {float(r['feasible_share_pct']):5.1f}%{mark}")

    print("\nOverleaf data inputs written:")
    for key in ["scenarios", "oat", "sobol_display", "sobol_kpi", "summary"]:
        print(f"  {paths[key]}")
    print("\nAudit/provenance outputs also written:")
    for key in ["sobol_archetypes_audit", "source_manifest"]:
        print(f"  {paths[key]}")


if __name__ == "__main__":
    main()
