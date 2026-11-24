"""Write the final PB budgets, utilisation results, sensitivity ranges, charts and checks.

Run from the repository root (reads committed data only):
    python -B 2b_PB_Allocation/tools/finalise.py

Writes 2b_PB_Allocation/final_results/ and refreshes the budget workbook read by the economic-only joint model
('2a_PB-LCA/FYR_Economic_Allocation.xlsx', sheet 'PB budgets', header row 7, columns A-G).
"""
import csv
import hashlib
import itertools
import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pb_allocation as pa  # noqa: E402

OUT = ROOT / "final_results"
ALFIE_WORKBOOK = ROOT.parent / "2a_PB-LCA" / "FYR_Economic_Allocation.xlsx"
IMPACTS_SHA256 = "2a02c39fc43edd1ec200dc13bb49bdcb5b10493d8fabdfd6196409d85c8daf01"
LABEL = {"total_gva": "Total GVA", "ability_to_pay": "Ability to Pay"}
SHORT = {"Climate Change": "Climate change", "Ocean Acidification": "Ocean acidification",
         "Atmospheric Aerosol Loading": "Aerosol loading", "Phosphorus Cycle": "Phosphorus cycle",
         "Change in Biosphere Integrity": "Biosphere integrity", "Freshwater Use": "Freshwater use",
         "Stratospheric Ozone Depletion": "Ozone depletion", "Nitrogen Cycle": "Nitrogen cycle",
         "Land-system Change": "Land-system change"}
CASE_LABELS = {
    "reference": "reference", "ppi_C211": "value: basic-pharma price index", "ppi_C212": "value: pharma-preparations price index",
    "no_inflation": "value: no 2019-22 inflation", "margin_SIC21.1": "value: + basic-pharma margin",
    "margin_SIC21": "value: + pharma margin", "full_supply_chain": "origin: full supply chain",
    "network_costs": "origin: network cost structure", "network_costs_high_income": "origin: high-income suppliers",
    "all_purchases_imported": "origin: all purchases imported", "domestic_only": "origin: all UK (bound)",
    "hartmann_normalised": "weights: normalised population", "worldbank": "weights: World Bank population",
    "worldbank_EXIO3": "weights: World Bank, EXIO3 mapping", "validation_2017_MER": "data: 2017",
    "sensitivity_2022_MER_no_imputation": "data: no GDP imputation", "sensitivity_2022_PPP": "data: PPP GDP",
    "sensitivity_2017_PPP": "data: 2017 PPP GDP", "sensitivity_2022_MER_matched_PPP": "data: MER, PPP coverage",
    "sensitivity_2022_MER_EXIO3": "data: EXIO3 country mapping",
}
WORKBOOK_UNITS = {"Atmospheric Aerosol Loading": "AOD (dimensionless)", "Change in Biosphere Integrity": "BII percentage points",
                  "Climate Change": "W/m²", "Freshwater Use": "km³/year", "Land-system Change": "million km²",
                  "Nitrogen Cycle": "Tg N/year", "Ocean Acidification": "Ωarag", "Phosphorus Cycle": "Tg P/year",
                  "Stratospheric Ozone Depletion": "DU"}


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def sig(v, n=6):
    return float(f"{v:.{n}g}")


# ----------------------------------------------------------------------------- tables
def budget_rows(impacts):
    rows = []
    for m in pa.METHODS:
        b = pa.budget_vector(m)
        u = pa.utilisation(b, impacts)
        for c in pa.CATEGORIES:
            rows.append(dict(method=m, indicator=c, unit=impacts[c]["unit"], global_sos=pa.SOS[c], budget=sig(b[c]),
                             annual_impact=sig(impacts[c]["value"]), utilisation_pct=round(100 * u[c], 3),
                             exceeded="YES" if u[c] > 1 else "no"))
    return rows


def sensitivity_rows(impacts):
    """Climate utilisation with one assumption changed at a time."""
    climate = impacts["Climate Change"]["value"]
    pct = lambda m, **kw: round(100 * climate / pa.budget_vector(m, **kw)["Climate Change"], 2)
    rows = []
    groups = [("reference", "reference", {})]
    groups += [("output value", b, dict(value_basis=b)) for b in pa.VALUE_BASES[1:]]
    groups += [("input origin", o, dict(origin=o)) for o in pa.INPUT_ORIGINS[1:]]
    for m in pa.METHODS:
        for group, case, kw in groups:
            rows.append(dict(method=m, assumption=group, case=case, climate_pct=pct(m, **kw)))
        for case in pa.WEIGHT_CASES[m][1:]:
            rows.append(dict(method=m, assumption="weights data", case=case, climate_pct=pct(m, weight_case=case)))
        b = pa.budget_vector(m)["Climate Change"]
        for case, imp in pa.climate_impact_cases().items():
            rows.append(dict(method=m, assumption="inventory", case=case, climate_pct=round(100 * imp / b, 2)))
    return rows


def range_rows():
    """Min/max climate utilisation over every combination of the four assumption groups."""
    inventory = pa.climate_impact_cases()
    rows = []
    for m in pa.METHODS:
        results = []
        for b, o, w in itertools.product(pa.VALUE_BASES, pa.INPUT_ORIGINS, pa.WEIGHT_CASES[m]):
            budget = pa.budget_vector(m, b, o, w)["Climate Change"]
            results += [(100 * imp / budget, b, o, w, inv) for inv, imp in inventory.items()]
        lo, hi = min(results), max(results)
        n_over = sum(r[0] > 100 for r in results)
        for tag, r in (("min", lo), ("max", hi)):
            rows.append(dict(method=m, extreme=tag, climate_pct=round(r[0], 2), value_basis=r[1], input_origin=r[2],
                             weight_case=r[3], inventory_case=r[4], combinations=len(results), exceeding_100pct=n_over))
    return rows


def regional_rows():
    d = pa._exio()
    a = pa.input_origin()
    wg, wa = pa.weights("total_gva"), pa.weights("ability_to_pay")
    cg, ca = pa.regional_contributions("total_gva"), pa.regional_contributions("ability_to_pay")
    rows = [dict(region=r, input_share=sig(a[r]), population_weight=sig(wg[r]), ability_to_pay_weight=sig(wa[r]),
                 share_of_total_gva_budget=sig(cg[r]), share_of_ability_to_pay_budget=sig(ca[r])) for r in d["regions"]]
    return sorted(rows, key=lambda r: -r["share_of_ability_to_pay_budget"])


def milp_record(impacts):
    rec = dict(description=("Annual PB budgets for the saved UK mAb network (861 batches, 4,003.65 kg/yr bulk drug "
                            "substance), 2022, native PBAESA units. Use as fixed parameters; choose one method per run."),
               reference_impacts={c: impacts[c]["value"] for c in pa.CATEGORIES},
               units={c: impacts[c]["unit"] for c in pa.CATEGORIES}, global_sos=pa.SOS, methods={})
    for m in pa.METHODS:
        rec["methods"][m] = dict(label=LABEL[m], sos_fraction=pa.sos_fraction(m), budgets=pa.budget_vector(m))
    rec["ability_to_pay_over_total_gva"] = pa.sos_fraction("ability_to_pay") / pa.sos_fraction("total_gva")
    rec["note"] = "Both vectors are SOS x one scalar, so switching method rescales all nine budgets equally."
    return rec


def write_workbook(path):
    """Budget workbook in the layout read by 4_Joint_MILP/without_LCA/jointmodel_MILP (header row 7, A-G)."""
    from openpyxl import Workbook
    from openpyxl.styles import Font
    columns = [("Global SOS", None),
               ("Total GVA (primary)", pa.budget_vector("total_gva")),
               ("Ability to Pay (alternative)", pa.budget_vector("ability_to_pay")),
               ("Total GVA, full supply chain (sensitivity)", pa.budget_vector("total_gva", origin="full_supply_chain")),
               ("Ability to Pay, full supply chain (sensitivity)", pa.budget_vector("ability_to_pay", origin="full_supply_chain"))]
    wb = Workbook()
    ws = wb.active
    ws.title = "PB budgets"
    ws["A2"] = "FYR planetary-boundary budgets (final)"
    ws["A2"].font = Font(bold=True, size=13)
    ws["A3"] = "Generated by 2b_PB_Allocation/tools/finalise.py. Annual budgets for the 861-batch network, native PBAESA units."
    ws["A5"] = "Budget = global SOS x sector factor (Hartmann Eq. 7) x network share of sector output. Columns are alternatives, never added."
    header = ["Indicator", "Indicator unit"] + [name for name, _ in columns]
    for i, h in enumerate(header, 1):
        ws.cell(7, i, h).font = Font(bold=True)
    for k, c in enumerate(sorted(pa.CATEGORIES)):
        ws.cell(8 + k, 1, c)
        ws.cell(8 + k, 2, WORKBOOK_UNITS[c])
        for i, (_, vec) in enumerate(columns, 3):
            ws.cell(8 + k, i, pa.SOS[c] if vec is None else vec[c])
    ws["A19"] = "Method, data and assumptions: 2b_PB_Allocation/README.md and METHODOLOGY.md."
    for col, width in zip("ABCDEFG", (32, 22, 14, 22, 26, 34, 38)):
        ws.column_dimensions[col].width = width
    wb.save(path)


# ----------------------------------------------------------------------------- charts (inline SVG, light/dark)
STYLE = ("<style>.bg{fill:#fcfcfb}.t1{fill:#0b0b0b}.t2{fill:#52514e}.grid{stroke:#e4e3df}.ref{stroke:#0b0b0b}"
         ".over{fill:#f3f2ef}.s1{fill:#2a78d6}.s2{fill:#eb6834}.ring{stroke:#fcfcfb}"
         ".halo{stroke:#fcfcfb;stroke-width:4px;paint-order:stroke;stroke-linejoin:round}"
         "@media (prefers-color-scheme: dark){.bg{fill:#1a1a19}.t1{fill:#f0efec}.t2{fill:#c3c2b7}.grid{stroke:#34332f}"
         ".ref{stroke:#f0efec}.over{fill:#24231f}.s1{fill:#3987e5}.s2{fill:#d95926}.ring{stroke:#1a1a19}"
         ".halo{stroke:#1a1a19}}</style>")


def chart_boundaries(path, rows):
    data = {(r["method"], r["indicator"]): r["utilisation_pct"] for r in rows}
    order = sorted(pa.CATEGORIES, key=lambda c: -data[("total_gva", c)])
    left, right, top, row_h, lo, hi = 190, 840, 108, 34, -2.0, 3.0
    width, height = 900, top + len(order) * row_h + 64
    X = lambda v: left + (math.log10(max(v, 10 ** lo)) - lo) / (hi - lo) * (right - left)
    axis_y = top + len(order) * row_h
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
         'font-family="Helvetica, Arial, sans-serif" role="img" aria-labelledby="t d">', STYLE,
         f'<rect class="bg" width="{width}" height="{height}"/>',
         '<title id="t">Share of allocated PB budget used: Total GVA vs Ability to Pay</title>',
         '<desc id="d">Log-scale utilisation for nine PB indicators. Values in budgets.csv.</desc>',
         '<text class="t1" x="20" y="30" font-size="17" font-weight="600">Share of allocated PB budget used, by boundary</text>',
         '<text class="t2" x="20" y="50" font-size="12">Saved 861-batch network, 2022; characterised subtotal. Log scale; '
         '100% = allocated budget.</text>',
         '<circle class="s1" cx="26" cy="70" r="5"/><text class="t1" x="36" y="74" font-size="12">Total GVA (primary)</text>',
         '<path class="s2" d="M196 64 l6 6 l-6 6 l-6 -6 z"/><text class="t1" x="208" y="74" font-size="12">Ability to Pay (alternative)</text>',
         f'<rect class="over" x="{X(100):.1f}" y="{top - 8}" width="{right - X(100):.1f}" height="{axis_y - top + 8}"/>',
         f'<text class="t2" x="{(X(100) + right) / 2:.1f}" y="{top - 14}" font-size="11" text-anchor="middle">budget exceeded</text>']
    for e in range(int(lo), int(hi) + 1):
        p.append(f'<line class="grid" x1="{X(10 ** e):.1f}" y1="{top - 8}" x2="{X(10 ** e):.1f}" y2="{axis_y}"/>'
                 f'<text class="t2" x="{X(10 ** e):.1f}" y="{axis_y + 16}" font-size="11" text-anchor="middle">{10 ** e:g}%</text>')
    p.append(f'<line class="ref" x1="{X(100):.1f}" y1="{top - 8}" x2="{X(100):.1f}" y2="{axis_y}" stroke-width="2" stroke-dasharray="5 3"/>')
    for k, c in enumerate(order):
        y = top + k * row_h + row_h / 2
        g, a = data[("total_gva", c)], data[("ability_to_pay", c)]
        p.append(f'<text class="t1" x="{left - 12}" y="{y + 4:.1f}" font-size="12" text-anchor="end">{SHORT[c]}</text>')
        p.append(f'<g><title>{SHORT[c]}, Total GVA: {g:.3g}%</title><circle class="s1 ring" cx="{X(g):.1f}" cy="{y:.1f}" r="6" stroke-width="2"/></g>')
        p.append(f'<g><title>{SHORT[c]}, Ability to Pay: {a:.3g}%</title><path class="s2 ring" stroke-width="2" '
                 f'd="M{X(a):.1f} {y - 7:.1f} l7 7 l-7 7 l-7 -7 z"/></g>')
        if k == 0:
            p.append(f'<text class="t1 halo" x="{X(g):.1f}" y="{y - 11:.1f}" font-size="11" text-anchor="middle">{g:.1f}%</text>')
            p.append(f'<text class="t1 halo" x="{X(a) + 10:.1f}" y="{y + 4:.1f}" font-size="11">{a:.1f}%</text>')
    p.append(f'<text class="t2" x="20" y="{axis_y + 44}" font-size="11">Only climate change exceeds a budget (Ability to Pay). '
             'Both methods scale all nine budgets by one factor.</text></svg>')
    Path(path).write_text("\n".join(p) + "\n", encoding="utf-8")


def chart_sensitivity(path, rows):
    keep = [r for r in rows if r["assumption"] != "inventory"]
    cls = {"total_gva": "s1", "ability_to_pay": "s2"}
    left, right, top, bar_h, gap, vmax = 330, 810, 96, 16, 7, 250.0
    width, height = 900, top + len(keep) * (bar_h + gap) + 60
    X = lambda v: left + min(v, vmax) / vmax * (right - left)
    axis_y = top + len(keep) * (bar_h + gap)
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
         'font-family="Helvetica, Arial, sans-serif" role="img" aria-labelledby="t d">', STYLE,
         f'<rect class="bg" width="{width}" height="{height}"/>',
         '<title id="t">Climate utilisation, one assumption changed at a time</title>',
         '<desc id="d">Bar length is climate utilisation, linear from zero, capped at 250%. Values in sensitivity_climate.csv.</desc>',
         '<text class="t1" x="20" y="30" font-size="17" font-weight="600">Climate utilisation: one assumption changed at a time</text>',
         '<text class="t2" x="20" y="50" font-size="12">Linear scale from 0, bars capped at 250% (value printed); 100% = allocated budget.</text>',
         '<rect class="s1" x="20" y="64" width="12" height="12" rx="2"/><text class="t1" x="38" y="74" font-size="12">Total GVA</text>',
         '<rect class="s2" x="190" y="64" width="12" height="12" rx="2"/><text class="t1" x="208" y="74" font-size="12">Ability to Pay</text>']
    for tick in (0, 50, 100, 150, 200, 250):
        p.append(f'<line class="grid" x1="{X(tick):.1f}" y1="{top - 6}" x2="{X(tick):.1f}" y2="{axis_y}"/>'
                 f'<text class="t2" x="{X(tick):.1f}" y="{axis_y + 16}" font-size="11" text-anchor="middle">{tick}%</text>')
    p.append(f'<line class="ref" x1="{X(100):.1f}" y1="{top - 10}" x2="{X(100):.1f}" y2="{axis_y}" stroke-width="2" stroke-dasharray="5 3"/>')
    for k, r in enumerate(keep):
        y = top + k * (bar_h + gap)
        v = r["climate_pct"]
        label = f'{LABEL[r["method"]]}: {CASE_LABELS.get(r["case"], r["case"].replace("_", " "))}'
        p.append(f'<text class="t1" x="{left - 10}" y="{y + bar_h / 2 + 4:.1f}" font-size="11" text-anchor="end">{label}</text>')
        p.append(f'<g><title>{label}: {v:.1f}%</title><rect class="{cls[r["method"]]}" x="{left}" y="{y}" '
                 f'width="{max(X(v) - left, 1):.1f}" height="{bar_h}" rx="4"/></g>')
        lx = X(100) + 6 if 85 < v < 100 else X(v) + 6
        p.append(f'<text class="t1 halo" x="{lx:.1f}" y="{y + bar_h / 2 + 4:.1f}" font-size="11">{v:,.1f}%{" ▸" if v > vmax else ""}</text>')
    p.append(f'<text class="t2" x="20" y="{axis_y + 40}" font-size="11">Combined extremes (all assumptions together): see climate_ranges.csv.</text></svg>')
    Path(path).write_text("\n".join(p) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- checks
def validate(impacts):
    checks = []

    def check(name, ok, detail):
        checks.append(dict(check=name, passed=bool(ok), detail=detail))

    d = pa._exio()
    t = d["target"]
    check("pbaesa_error_reproduces_published_factor",
          math.isclose(d["sector_factor_pbaesa_formula"], d["sector_factor_published"], rel_tol=1e-9),
          f"PBAESA formula {d['sector_factor_pbaesa_formula']:.10e} vs published {d['sector_factor_published']:.10e}")
    check("eq7_factor_used", math.isclose(pa.sector_factor("total_gva"), d["sector_factor_eq7"], rel_tol=1e-12),
          f"Eq. 7 factor {d['sector_factor_eq7']:.10e} (published / Eq. 7 = {d['sector_factor_published'] / d['sector_factor_eq7']:.3f})")
    check("total_gva_equals_gross_output", abs(t["embodied_gva_per_unit_output"] - 1) < 1e-6,
          f"g = {t['embodied_gva_per_unit_output']:.9f}")
    check("input_origins_sum_to_one", all(abs(sum(pa.input_origin(o).values()) - 1) < 1e-12 for o in pa.INPUT_ORIGINS),
          f"{len(pa.INPUT_ORIGINS)} origin cases")
    check("ability_to_pay_weights_sum_to_one",
          all(abs(sum(pa.weights("ability_to_pay", c).values()) - 1) < 1e-9 for c in pa.WEIGHT_CASES["ability_to_pay"]),
          f"{len(pa.WEIGHT_CASES['ability_to_pay'])} data cases")
    a17 = pa._json("ability_to_pay_weights.json")["cases"]["validation_2017_MER"]["alpha"]
    check("hjalsted_alpha_2017_reproduced", abs(a17 / 0.243 - 1) < 0.05, f"alpha 2017 = {a17:.4f} vs 0.243")
    hand = 1.0 * d["sector_factor_eq7"] * pa.network_value() / t["gross_output"]
    check("climate_budget_hand_arithmetic", math.isclose(pa.budget_vector()["Climate Change"], hand, rel_tol=1e-12),
          f"{hand:.6e} W/m2")
    ratio = [pa.budget_vector("ability_to_pay")[c] / pa.budget_vector("total_gva")[c] for c in pa.CATEGORIES]
    check("switching_method_only_rescales", max(ratio) - min(ratio) < 1e-15, f"ratio {ratio[0]:.6f} for all indicators")
    imp_sha = hashlib.sha256((pa.DATA / "annual_impacts.csv").read_bytes()).hexdigest()
    check("impacts_unchanged", imp_sha == IMPACTS_SHA256, "data/annual_impacts.csv hash")
    ref = pa.climate_impact_cases()["legacy_foam|batch_scaled"]
    check("inventory_reference_case_matches_impacts", math.isclose(ref, impacts["Climate Change"]["value"], rel_tol=1e-9),
          f"{ref:.10e} W/m2")
    failed = [c for c in checks if not c["passed"]]
    if failed:
        raise AssertionError("; ".join(f"{c['check']}: {c['detail']}" for c in failed))
    return checks


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    impacts = pa.load_impacts()
    checks = validate(impacts)
    budgets = budget_rows(impacts)
    sens = sensitivity_rows(impacts)
    write_csv(OUT / "budgets.csv", budgets)
    write_csv(OUT / "sensitivity_climate.csv", sens)
    write_csv(OUT / "climate_ranges.csv", range_rows())
    write_csv(OUT / "regional_contributions.csv", regional_rows())
    (OUT / "milp_budgets.json").write_text(json.dumps(milp_record(impacts), indent=1) + "\n")
    write_workbook(OUT / "FYR_PB_budgets.xlsx")
    if ALFIE_WORKBOOK.parent.exists():
        shutil.copyfile(OUT / "FYR_PB_budgets.xlsx", ALFIE_WORKBOOK)
    chart_boundaries(OUT / "utilisation_by_boundary.svg", budgets)
    chart_sensitivity(OUT / "climate_sensitivity.svg", sens)
    (OUT / "validation.json").write_text(json.dumps(dict(passed=True, checks=checks), indent=1) + "\n")
    print(f"{len(checks)} checks passed")
    for r in budgets:
        if r["indicator"] == "Climate Change":
            print(f"  {LABEL[r['method']]:15s} climate {r['utilisation_pct']:.2f}%")


if __name__ == "__main__":
    main()
