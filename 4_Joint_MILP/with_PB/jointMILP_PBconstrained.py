"""Minimise the joint model's annual cost subject to the nine status quo PB caps."""

import argparse
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import pandas as pd
import pyomo.environ as pyo
from openpyxl import load_workbook

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "without_LCA"))
import testjoint_PB as joint

WORKBOOK = joint.CODE_DIR / "Planetary_Boundary_impacts.xlsx"


def load_limits(path, sheet_name, indicators):
    """Match each D cell to its indicator label on the same row."""
    names = {" ".join(name.casefold().split()): name for name in indicators}
    names["phosphorus cycle (freshwater- ocean)"] = "Phosphorus Cycle"
    book = load_workbook(path, data_only=True, read_only=True)
    try:
        sheet = book[sheet_name] if sheet_name else book.active
        limits = {}
        for row in range(23, 32):
            cap = sheet[f"D{row}"].value
            if type(cap) not in (int, float) or not math.isfinite(cap) or cap <= 0:
                raise ValueError(f"{path}, {sheet.title}!D{row}: expected a positive numeric "
                                 "status quo limit. Save calculated Excel formulas before running.")
            labels = {names[key] for cell in sheet[f"A{row}:C{row}"][0]
                      if (key := " ".join(str(cell.value).casefold().split())) in names}
            if len(labels) != 1 or labels & limits.keys():
                raise ValueError(f"{path}, row {row}: expected one unique PB indicator in A:C.")
            limits[labels.pop()] = (cap, f"{sheet.title}!D{row}")
        return limits
    finally:
        book.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=WORKBOOK)
    parser.add_argument("--sheet", help="Worksheet containing D23:D31 (default: active sheet)")
    parser.add_argument("--output-dir", type=Path, help="Optionally keep solver files and full reports.")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    indicators = pd.read_excel(joint.PB_BUDGET_FILE, sheet_name="PB budgets",
                               header=6, usecols="A", nrows=9)["Indicator"].tolist()
    limits = load_limits(args.workbook, args.sheet, indicators)
    model, data = joint.build_joint_model()
    impacts = joint.calculate_planetary_boundary_impacts(model, data, as_expressions=True)
    model.Impact = pyo.Expression(indicators, initialize=impacts)
    model.PB_local = pyo.Param(indicators, initialize={n: cap for n, (cap, _) in limits.items()})
    model.status_quo_limit = pyo.Constraint(
        indicators, rule=lambda m, n: m.Impact[n] <= m.PB_local[n])
    # Scale both sides for GAMS because some caps are below its feasibility tolerance.
    for n, row in model.status_quo_limit.items():
        scale = model.PB_local[n]
        row.set_value(row.body / scale <= row.upper / scale)
    if args.export_only:
        path, _ = joint.export_gams(model, args.output_dir or HERE)
        print(f"GAMS input: {path}")
        return
    with TemporaryDirectory(prefix="joint_pb_") as scratch:
        joint.solve_joint_model(model, args.output_dir or scratch)
    report, _, _ = joint.calculate_planetary_boundary_impacts(model, data)
    report = report.set_index("Indicator")
    rows = []
    for name, (cap, cell) in limits.items():
        actual = report.at[name, "raw_impact"]
        fraction = actual / cap
        if abs(fraction - pyo.value(impacts[name]) / cap) > 1e-7 or fraction > 1 + 1e-7:
            raise ValueError(f"PB check failed for {name}: impact {actual}, limit {cap}")
        rows.append((name, report.at[name, "Indicator unit"], str(args.workbook), cell, cap, actual, fraction))
    comparison = pd.DataFrame(rows, columns=["indicator", "unit", "workbook", "cell", "limit",
                                             "impact", "fraction_used"])
    if args.output_dir:
        joint.write_joint_results(model, data, args.output_dir)
        comparison.to_csv(args.output_dir / "status_quo_limits.csv", index=False)
    print(comparison[["indicator", "cell", "impact", "limit", "fraction_used"]].to_string(index=False))
    print(f"Annual cost: GBP {pyo.value(model.total_cost_gbp):,.2f}")
    print(f"Unmet demand: {sum(pyo.value(v) for v in model.Slack.values()):,.2f} kg")


if __name__ == "__main__":
    main()
