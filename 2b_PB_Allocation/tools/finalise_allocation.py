"""Final PB allocation results: total GVA (primary) and ability to pay (alternative).

Reads only committed files and writes 2b_PB_Allocation/results_allocation_final/:
  utilisation_summary.csv     nine indicators x four methods, % of budget used, exceeded flag
  budgets_long.csv            every method x value basis x indicator (budget, impact, utilisation)
  sensitivity_climate.csv     climate utilisation under every documented variant
  regional_contributions.csv  where each method's budget comes from (EXIOBASE region shares)
  milp_budgets.json           budget vectors for the joint MILP (select by method key)
  utilisation_by_boundary.svg, climate_sensitivity.svg
  validation.json, provenance.json

Run from the repository root:
    .venv/bin/python -B 2b_PB_Allocation/tools/finalise_allocation.py
"""
import csv
import hashlib
import json
import math
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pb_allocation as pa  # noqa: E402

OUT = ROOT / "results_allocation_final"
VERIFY = ROOT / "data/total_gva_verification_2022.json"
IMPACTS_SHA256 = "2a02c39fc43edd1ec200dc13bb49bdcb5b10493d8fabdfd6196409d85c8daf01"  # results_final/provenance.json
MAIN = ("total_gva", "ability_to_pay", "total_fce", "total_fce_corrected", "total_fce_corrected_f")
DECISION_METHODS = ("total_gva", "ability_to_pay")
INVENTORY = ROOT / "data/climate_inventory_scenarios.json"
AUDIT = ROOT / "data/audit_from_raw_2022.json"
SHORT = {"Climate Change": "Climate change", "Ocean Acidification": "Ocean acidification",
         "Atmospheric Aerosol Loading": "Aerosol loading", "Phosphorus Cycle": "Phosphorus cycle",
         "Change in Biosphere Integrity": "Biosphere integrity", "Freshwater Use": "Freshwater use",
         "Stratospheric Ozone Depletion": "Ozone depletion", "Nitrogen Cycle": "Nitrogen cycle",
         "Land-system Change": "Land-system change"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def fmt(v):
    return float(f"{v:.6g}")


# --------------------------------------------------------------------------- tables
def long_rows(impacts):
    rows = []
    for m in pa.METHODS:
        for basis in pa.VALUE_BASES:
            a = pa.allocate(m, basis)
            u = pa.utilisation(impacts, a.budgets)
            for c in pa.CATEGORIES:
                rows.append(dict(method=m, role=a.method.role, value_basis=basis, category=c,
                                 unit=impacts[c]["unit"], global_sos=pa.SOS[c], sector_factor=fmt(a.sector_factor),
                                 network_share=fmt(a.network_share), sos_fraction=fmt(a.sos_fraction),
                                 budget=fmt(a.budgets[c]), annual_impact=fmt(impacts[c]["value"]),
                                 utilisation_pct=fmt(100 * u[c]), exceeded="YES" if u[c] > 1 else "no"))
    return rows


def summary_rows(impacts):
    allocs = {m: pa.allocate(m) for m in MAIN}
    rows = []
    for c in sorted(pa.CATEGORIES, key=lambda c: -pa.utilisation(impacts, allocs["total_gva"].budgets)[c]):
        row = dict(category=c, unit=impacts[c]["unit"], annual_impact=fmt(impacts[c]["value"]))
        for m, a in allocs.items():
            u = 100 * pa.utilisation(impacts, a.budgets)[c]
            row[f"{m}_budget"] = fmt(a.budgets[c])
            row[f"{m}_pct"] = round(u, 4)
        row["exceeded_under"] = ";".join(m for m in MAIN if row[f"{m}_pct"] > 100) or "none"
        rows.append(row)
    return rows


def sensitivity_rows(impacts):
    climate = impacts["Climate Change"]["value"]
    out = []

    def add(group, method, variant, note, **kw):
        a = pa.allocate(method, **kw)
        out.append(dict(group=group, method=method, variant=variant, value_basis=a.value_basis, atp_case=a.atp_case,
                        input_origin=a.input_origin, population_case=a.population_case,
                        sos_fraction=fmt(a.sos_fraction),
                        climate_pct=round(100 * climate / a.budgets["Climate Change"], 3), note=note))

    for m, g in (("total_gva", "Total GVA"), ("ability_to_pay", "Ability to pay")):
        add(g, m, "central", "SuperPro cost, 2022 weights")
        add(g, m, "no 2019-2022 price inflation", "Flat price scenario", value_basis="cost_no_inflation")
        add(g, m, "cost + SIC 21.1 operating margin", "Illustrative gross margin uplift (basic pharmaceutical products)",
            value_basis="cost_plus_margin_SIC21.1")
        add(g, m, "cost + SIC 21 operating margin", "Illustrative gross margin uplift (all pharmaceuticals)",
            value_basis="cost_plus_margin_SIC21")
        for series in ("C211", "C212"):
            add(g, m, f"ONS {series} price index", "Domestic output price index proxy",
                value_basis=f"cost_ppi_{series}")
    for case, label in (("validation_2017_MER", "2017 population and GDP"),
                        ("sensitivity_2022_MER_no_imputation", "no GDP imputation (6 small economies dropped)"),
                        ("sensitivity_2022_PPP", "PPP GDP, complete cases"),
                        ("sensitivity_2022_MER_matched_PPP", "MER GDP, same PPP coverage"),
                        ("sensitivity_2022_MER_EXIO3", "EXIO3 concordance"),
                        ("sensitivity_2017_PPP", "2017 PPP comparison")):
        add("Ability to pay", "ability_to_pay", label, "Data choice for Hjalsted Eq. 5", atp_case=case)
    for case, label in (("hartmann_normalised", "Hartmann weights normalised"),
                        ("worldbank_hartmann", "World Bank, Hartmann mapping"),
                        ("worldbank_EXIO3", "World Bank, EXIO3 mapping")):
        add("Total GVA", "total_gva", label, "Population-weight scenario", population_case=case)
    origin_labels = {"network_costs": "inputs: network cost structure",
                     "network_costs_high_income_suppliers": "inputs: network costs, high-income suppliers",
                     "network_costs_all_purchases_imported": "inputs: all purchased goods imported",
                     "domestic_only": "inputs: all from UK (bound)",
                     "full_supply_chain": "inputs: full-supply-chain regional split"}
    for m, g in (("total_gva", "Total GVA"), ("ability_to_pay", "Ability to pay")):
        for origin, label in origin_labels.items():
            add(g, m, label, pa.INPUT_ORIGINS[origin], input_origin=origin)
    add("Legacy", "total_gva_published", "published factor (PBAESA error)", "Do not use")
    add("Legacy", "ability_to_pay_regional", "ATP on 49 aggregated regions", "Superseded")
    add("Supporting", "total_fce", "total FCE, published", "Unresolved supporting comparison")
    add("Supporting", "total_fce_corrected", "total FCE, corrected reading", "Unresolved supporting comparison")
    add("Supporting", "total_fce_corrected_f", "total FCE, corrected reading with f by product", "Unresolved supporting comparison")
    add("Supporting", "ability_to_pay_fce", "ability to pay x total FCE (corrected)", "Unresolved supporting comparison")
    return out


def envelope_rows(impacts):
    """Climate utilisation for every value basis x input origin combination (assumptions compound)."""
    climate = impacts["Climate Change"]["value"]
    rows = []
    for m in ("total_gva", "ability_to_pay"):
        combos = [(b, o, 100 * climate / pa.budget_vector(m, b, input_origin=o)["Climate Change"])
                  for b in pa.VALUE_BASES for o in pa.INPUT_ORIGINS]
        lo, hi = min(c[2] for c in combos), max(c[2] for c in combos)
        rows += [dict(method=m, value_basis=b, input_origin=o, climate_pct=round(u, 3),
                      exceeded="YES" if u > 100 else "no",
                      extreme="min" if u == lo else ("max" if u == hi else "")) for b, o, u in combos]
    return rows


def climate_inventory(impacts):
    reference = impacts["Climate Change"]["value"]
    contributions = json.loads(INVENTORY.read_text())["scenarios"]
    scenarios = {name: math.fsum(terms.values()) for name, terms in contributions.items()}
    if len(scenarios) != 9 or not math.isclose(scenarios["legacy_foam|batch_scaled"], reference, rel_tol=1e-9):
        raise ValueError("Physical replay does not match the saved reference")
    return {"reference": reference, **scenarios}


def crossed_climate_rows(impacts):
    """Discrete stress tests. Procurement and physical inventory are varied independently."""
    inventory = climate_inventory(impacts)
    weights = {
        "total_gva": [{"population_case": case} for case in pa.POPULATION_CASES],
        "ability_to_pay": [{"atp_case": case} for case in json.loads(pa.ATP_WEIGHTS.read_text())["cases"]],
    }
    rows = []
    shares = {basis: pa.network_share(basis) for basis in pa.VALUE_BASES}
    climate_sos = pa.SOS["Climate Change"]
    for method, cases in weights.items():
        for origin, case in product(pa.INPUT_ORIGINS, cases):
            factor = pa.allocate(method, input_origin=origin, **case).sector_factor
            for basis, share in shares.items():
                budget = climate_sos * factor * share
                for name, impact in inventory.items():
                    rows.append(dict(method=method, value_basis=basis, input_origin=origin,
                                     weight_case=next(iter(case.values())), inventory_case=name,
                                     budget_W_m2=budget, impact_W_m2=impact, climate_pct=100 * impact / budget))
    return rows


def crossed_climate_extrema(rows):
    summary = []
    for method in DECISION_METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        for pick in (min, max):
            row = pick(method_rows, key=lambda row: row["climate_pct"])
            summary.append(dict(extreme=pick.__name__, **row))
    return summary


def regional_rows():
    d = pa.exiobase_terms()
    alpha = pa.input_shares("sector_average")
    alpha_net = pa.input_shares("network_costs")
    eq, atp = pa.regional_contributions("total_gva"), pa.regional_contributions("ability_to_pay")
    atp_net = pa.regional_contributions("ability_to_pay", input_origin="network_costs")
    pop, aw = pa.population_weights(), pa.ability_to_pay_weights()
    gva = dict(zip(d["regions"], d["gva_by_region_eur_m"]))
    income = json.loads(pa.ATP_WEIGHTS.read_text())["region_income_group"]
    rows = [dict(region=r, income_group=income[r], input_share_sector_average=fmt(alpha[r]),
                 input_share_network_costs=fmt(alpha_net[r]), population_weight=fmt(pop[r]), atp_weight=fmt(aw[r]),
                 sos_per_eur_gva_vs_GB_equal_per_capita=fmt((pop[r] / gva[r]) / (pop["GB"] / gva["GB"])),
                 sos_per_eur_gva_vs_GB_ability_to_pay=fmt((aw[r] / gva[r]) / (aw["GB"] / gva["GB"])),
                 share_of_total_gva_budget=fmt(eq[r]), share_of_ability_to_pay_budget=fmt(atp[r]),
                 share_of_ability_to_pay_budget_network_costs=fmt(atp_net[r])) for r in d["regions"]]
    return sorted(rows, key=lambda r: -r["share_of_ability_to_pay_budget"])


def milp_record(impacts):
    rec = dict(
        description=("Annual PB budgets for the saved UK mAb network (4,003.65 kg/yr bulk drug substance), native "
                     "PBAESA control-variable units, 2022. Use as fixed parameters: never as decision variables."),
        default_method="total_gva", value_basis="cost", status="exploratory fixed-budget scenarios",
        reference=dict(
            annual_mass_kg=pa.annual_mass_kg(),
            product_gate="Bulk mAb drug substance",
            numerator="Partial characterised two-stage inventory",
            input_origin="sector_average",
            population_case="hartmann_reference",
            atp_case="primary_2022_MER",
            output_value_note=pa.VALUE_BASES["cost"],
        ),
        formula="budget_c = SOS_c x AF x (V_net / x_GB,Chemicals nec)",
        units={c: impacts[c]["unit"] for c in pa.CATEGORIES}, global_sos=pa.SOS, methods={})
    for m in DECISION_METHODS:
        a = pa.allocate(m)
        rec["methods"][m] = dict(label=a.method.label, role=a.method.role, geography=a.method.geography,
                                 value_perspective=a.method.value, sector_factor=a.sector_factor,
                                 network_share=a.network_share, sos_fraction=a.sos_fraction, budgets=a.budgets,
                                 epsilon_equivalent_vs_total_gva=a.sos_fraction / pa.allocate("total_gva").sos_fraction)
    rec["relative_importance_ability_to_pay_over_total_gva"] = pa.relative_importance()
    rec["sensitivity_sos_fractions"] = {
        m: {f"{basis}|{origin}": pa.allocate(m, basis, input_origin=origin).sos_fraction
            for basis in pa.VALUE_BASES for origin in pa.INPUT_ORIGINS}
        for m in ("total_gva", "ability_to_pay")}
    rec["sensitivity_note"] = ("Each sensitivity is the global SOS vector times the listed fraction "
                               "(key = value_basis|input_origin). Run MILP sensitivities by rescaling, not by "
                               "letting the solver choose a variant.")
    rec["note"] = ("All methods share one SOS fraction across the nine indicators, so the ratio "
                   "budget_c(method)/budget_c(total_gva) is the same for every c: switching method only rescales.")
    return rec


# --------------------------------------------------------------------------- validation
def validate(impacts):
    checks = []

    def check(name, ok, detail):
        checks.append(dict(check=name, passed=bool(ok), detail=detail))

    v = json.loads(VERIFY.read_text())["target"]
    eq7 = pa.allocate("total_gva").sector_factor
    pub = pa.allocate("total_gva_published").sector_factor
    check("eq7_factor_two_independent_routes", math.isclose(eq7, v["af_eq7"], rel_tol=1e-9),
          f"A-based extract {eq7:.10e} vs Z/solve-based {v['af_eq7']:.10e}")
    check("pbaesa_formula_reproduces_published", math.isclose(v["af_pbaesa_formula"], v["af_published"], rel_tol=1e-6),
          f"broadcasting formula {v['af_pbaesa_formula']:.10e} vs workbook {v['af_published']:.10e}")
    check("published_factor_differs_from_eq7", abs(v["af_eq7_over_published"] - 1) > 0.1,
          f"Eq.7 / published = {v['af_eq7_over_published']:.4f}")
    check("legacy_factor_matches_workbook", math.isclose(pub, v["af_published"], rel_tol=1e-9), f"{pub:.10e}")
    check("total_gva_equals_output_for_target", abs(v["vL"] - 1) < 1e-6,
          f"(vL)_j = {v['vL']:.9f}; m_j GVA_j = {v['total_gva_eq7_eur_m']:.2f} vs x_j {v['gross_output_eur_m']:.2f} M EUR")
    atp = json.loads(pa.ATP_WEIGHTS.read_text())
    a17 = atp["cases"]["validation_2017_MER"]["alpha"]
    ppp17 = atp["cases"]["sensitivity_2017_PPP"]["alpha"]
    check("hjalsted_alpha_2017_reproduced", abs(a17 / 0.243 - 1) < 0.05,
          f"country-level MER alpha 2017 = {a17:.4f} vs published 0.243 ({100 * (a17 / 0.243 - 1):+.1f}%)")
    check("hjalsted_used_market_exchange_rates", abs(ppp17 / 0.243 - 1) > 0.5,
          f"PPP alpha 2017 = {ppp17:.4f} is inconsistent with 0.243")
    pop_wb = atp["cases"]["primary_2022_MER"]["population_share_by_region"]
    pop_pb = pa.population_weights()
    worst = max(abs(pop_wb[r] / pop_pb[r] - 1) for r in pop_pb)
    check("country_mapping_matches_pbaesa_population", worst < 0.07 and abs(pop_wb["GB"] / pop_pb["GB"] - 1) < 0.01,
          f"worst regional population-share deviation {100 * worst:.1f}%, GB {100 * (pop_wb['GB'] / pop_pb['GB'] - 1):+.2f}%")
    for name, w in (("population", pop_pb), ("ability_to_pay", pa.ability_to_pay_weights())):
        check(f"{name}_weights_sum_to_one", abs(sum(w.values()) - 1) < 1e-3, f"sum {sum(w.values()):.9f}")
    ri = pa.relative_importance()
    check("switching_method_only_rescales", max(ri.values()) - min(ri.values()) < 1e-12,
          f"budget ratio ability_to_pay/total_gva = {next(iter(ri.values())):.6f} for all nine indicators")
    a = pa.allocate("total_gva")
    hand = pa.SOS["Climate Change"] * eq7 * pa.network_value_eur_m() / pa.exiobase_terms()["target"]["gross_output"]
    check("budget_equals_hand_arithmetic", math.isclose(a.budgets["Climate Change"], hand, rel_tol=1e-12), f"{hand:.6e} W/m2")
    check("network_value_matches_previous_review", math.isclose(pa.network_value_eur_m(), 378.4, rel_tol=1e-3),
          f"{pa.network_value_eur_m():.3f} M EUR2022")
    for origin in pa.INPUT_ORIGINS:
        s = pa.input_shares(origin)
        check(f"input_shares_sum_to_one_{origin}", abs(sum(s.values()) - 1) < 1e-12 and min(s.values()) >= 0,
              f"foreign share {1 - s['GB']:.4f}")
    check("default_input_origin_is_hartmann_sector_average",
          math.isclose(pa.allocate("total_gva", input_origin="sector_average").sector_factor, eq7, rel_tol=1e-15),
          "primary results use EXIOBASE sector-average input shares")
    if AUDIT.exists():
        aud = json.loads(AUDIT.read_text())["climate_utilisation_pct"]
        climate = impacts["Climate Change"]["value"]
        pairs = {"total_gva_eq7": "total_gva", "ability_to_pay_eq7": "ability_to_pay",
                 "total_fce_published_formula": "total_fce", "total_fce_corr_FR": "total_fce_corrected",
                 "total_fce_corr_FR_f": "total_fce_corrected_f", "ability_to_pay_fce_corr_FR": "ability_to_pay_fce",
                 "total_gva_pbaesa_formula": "total_gva_published"}
        worst = max(abs(100 * climate / pa.budget_vector(m)["Climate Change"] / aud[k] - 1) for k, m in pairs.items())
        check("independent_raw_audit_agrees", worst < 1e-9,
              f"7 climate results recomputed from raw EXIOBASE/World Bank/SuperPro; max rel. deviation {worst:.1e}")
        sos = json.loads(AUDIT.read_text())["checks"]["sos_code_vs_workbook"]
        check("sos_matches_pbaesa_code_and_cf_workbook",
              all(math.isclose(pa.SOS[c], code) and (wbv is None or math.isclose(code, wbv)) for c, (code, wbv) in sos.items()),
              "nine SOS values (nitrogen from PBAESA code; eight also in the CF workbook)")
    check("impacts_unchanged", sha256(pa.IMPACTS) == IMPACTS_SHA256, "results_final/annual_impacts.csv hash")
    failed = [c for c in checks if not c["passed"]]
    if failed:
        raise AssertionError("; ".join(f"{c['check']}: {c['detail']}" for c in failed))
    return checks


# --------------------------------------------------------------------------- charts (hand-written SVG)
STYLE = ("<style>.bg{fill:#fcfcfb}.t1{fill:#0b0b0b}.t2{fill:#52514e}.grid{stroke:#e4e3df}.ref{stroke:#0b0b0b}"
         ".over{fill:#f3f2ef}.s1{fill:#2a78d6}.s2{fill:#eb6834}.s3{fill:#1baf7a}.l3{stroke:#1baf7a}.sg{fill:#9b9a95}"
         ".ring{stroke:#fcfcfb}.halo{stroke:#fcfcfb;stroke-width:4px;paint-order:stroke;stroke-linejoin:round}"
         "@media (prefers-color-scheme: dark){.bg{fill:#1a1a19}.t1{fill:#f0efec}.t2{fill:#c3c2b7}.grid{stroke:#34332f}"
         ".ref{stroke:#f0efec}.over{fill:#24231f}.s1{fill:#3987e5}.s2{fill:#d95926}.s3{fill:#199e70}.l3{stroke:#199e70}"
         ".sg{fill:#8a8984}.ring{stroke:#1a1a19}.halo{stroke:#1a1a19}}</style>")


def chart_boundaries(path, rows):
    """Dot plot, log x (0.01%-1000%): total GVA (circle), ability to pay (diamond), total FCE range (segment)."""
    left, right, top, row_h = 190, 840, 108, 34
    width, height = 900, top + len(rows) * row_h + 70
    lo, hi = -2.0, 3.0
    X = lambda pct: left + (math.log10(max(pct, 10 ** lo)) - lo) / (hi - lo) * (right - left)
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
         'font-family="Helvetica, Arial, sans-serif" role="img" aria-labelledby="t d">', STYLE,
         f'<rect class="bg" width="{width}" height="{height}"/>',
         '<title id="t">PB budget utilisation by boundary: total GVA vs ability to pay</title>',
         '<desc id="d">Log-scale utilisation (% of allocated budget) for nine PBAESA indicators. Values in utilisation_summary.csv.</desc>',
         '<text class="t1" x="20" y="30" font-size="17" font-weight="600">Share of allocated PB budget used, by boundary</text>',
         '<text class="t2" x="20" y="50" font-size="12">Saved two-stage network, 2022, SuperPro cost valuation; '
         'incomplete characterised subtotal. Log scale; 100% = allocated budget.</text>']
    ly = 74
    p.append(f'<circle class="s1" cx="26" cy="{ly - 4}" r="5"/><text class="t1" x="36" y="{ly}" font-size="12">Total GVA (primary)</text>')
    p.append(f'<path class="s2" d="M196 {ly - 10} l6 6 l-6 6 l-6 -6 z"/><text class="t1" x="208" y="{ly}" font-size="12">Ability to pay (alternative)</text>')
    p.append(f'<line class="l3" x1="398" y1="{ly - 4}" x2="418" y2="{ly - 4}" stroke-width="4" stroke-linecap="round"/>'
             f'<text class="t1" x="426" y="{ly}" font-size="12">Total FCE, unresolved historical variants</text>')
    axis_y = top + len(rows) * row_h
    p.append(f'<rect class="over" x="{X(100):.1f}" y="{top - 8}" width="{right - X(100):.1f}" height="{axis_y - top + 8}"/>')
    p.append(f'<text class="t2" x="{(X(100) + right) / 2:.1f}" y="{top - 14}" font-size="11" text-anchor="middle">budget exceeded</text>')
    for e in range(int(lo), int(hi) + 1):
        v = 10 ** e
        lab = f"{v:g}%"
        p.append(f'<line class="grid" x1="{X(v):.1f}" y1="{top - 8}" x2="{X(v):.1f}" y2="{axis_y}"/>'
                 f'<text class="t2" x="{X(v):.1f}" y="{axis_y + 16}" font-size="11" text-anchor="middle">{lab}</text>')
    p.append(f'<line class="ref" x1="{X(100):.1f}" y1="{top - 8}" x2="{X(100):.1f}" y2="{axis_y}" stroke-width="2" stroke-dasharray="5 3"/>')
    for k, r in enumerate(rows):
        y = top + k * row_h + row_h / 2
        p.append(f'<text class="t1" x="{left - 12}" y="{y + 4:.1f}" font-size="12" text-anchor="end">{SHORT[r["category"]]}</text>')
        f0, f1 = min(r["total_fce_pct"], r["total_fce_corrected_pct"], r["total_fce_corrected_f_pct"]), \
            max(r["total_fce_pct"], r["total_fce_corrected_pct"], r["total_fce_corrected_f_pct"])
        p.append(f'<g><title>{SHORT[r["category"]]}, total FCE: {f0:.3g}–{f1:.3g}%</title>'
                 f'<line class="l3" x1="{X(f0):.1f}" y1="{y:.1f}" x2="{X(f1):.1f}" y2="{y:.1f}" stroke-width="4" stroke-linecap="round"/></g>')
        g, a = r["total_gva_pct"], r["ability_to_pay_pct"]
        p.append(f'<g><title>{SHORT[r["category"]]}, total GVA: {g:.3g}%</title>'
                 f'<circle class="s1 ring" cx="{X(g):.1f}" cy="{y:.1f}" r="6" stroke-width="2"/></g>')
        p.append(f'<g><title>{SHORT[r["category"]]}, ability to pay: {a:.3g}%</title>'
                 f'<path class="s2 ring" stroke-width="2" d="M{X(a):.1f} {y - 7:.1f} l7 7 l-7 7 l-7 -7 z"/></g>')
        if k == 0:
            p.append(f'<text class="t1" x="{X(g):.1f}" y="{y - 11:.1f}" font-size="11" text-anchor="middle">{g:.1f}%</text>')
            p.append(f'<text class="t1" x="{X(a) + 10:.1f}" y="{y + 4:.1f}" font-size="11">{a:.1f}%</text>')
    p.append(f'<text class="t2" x="20" y="{axis_y + 44}" font-size="11">Only climate change exceeds a budget (ability to pay). '
             'Both methods scale all nine budgets by one factor, so the ordering of boundaries is identical.</text></svg>')
    Path(path).write_text("\n".join(p) + "\n", encoding="utf-8")


def chart_sensitivity(path, rows):
    cls = {"Total GVA": "s1", "Ability to pay": "s2", "Legacy": "sg"}
    keep = [r for g in cls for r in rows if r["group"] == g]
    left, right, top, bar_h, gap, vmax = 330, 810, 96, 18, 9, 250.0
    width, height = 900, top + len(keep) * (bar_h + gap) + 72
    X = lambda v: left + min(v, vmax) / vmax * (right - left)
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
         'font-family="Helvetica, Arial, sans-serif" role="img" aria-labelledby="t d">', STYLE,
         f'<rect class="bg" width="{width}" height="{height}"/>',
         '<title id="t">Climate utilisation: sensitivity of total GVA and ability to pay</title>',
         '<desc id="d">Bar length is climate budget utilisation in percent, linear from zero, capped at 250%. '
         'Values in sensitivity_climate.csv.</desc>',
         '<text class="t1" x="20" y="30" font-size="17" font-weight="600">Climate utilisation under documented variants</text>',
         '<text class="t2" x="20" y="50" font-size="12">Linear scale from 0; bars capped at 250% (value printed). 100% = allocated budget.</text>']
    for i, (g, c) in enumerate(cls.items()):
        lx = 20 + i * 170
        p.append(f'<rect class="{c}" x="{lx}" y="64" width="12" height="12" rx="2"/><text class="t1" x="{lx + 18}" y="74" font-size="12">{g}</text>')
    axis_y = top + len(keep) * (bar_h + gap)
    for tick in (0, 50, 100, 150, 200, 250):
        p.append(f'<line class="grid" x1="{X(tick):.1f}" y1="{top - 6}" x2="{X(tick):.1f}" y2="{axis_y}"/>'
                 f'<text class="t2" x="{X(tick):.1f}" y="{axis_y + 16}" font-size="11" text-anchor="middle">{tick}%</text>')
    p.append(f'<line class="ref" x1="{X(100):.1f}" y1="{top - 10}" x2="{X(100):.1f}" y2="{axis_y}" stroke-width="2" stroke-dasharray="5 3"/>')
    for k, r in enumerate(keep):
        y = top + k * (bar_h + gap)
        v = r["climate_pct"]
        label = (r["group"].replace("Ability to pay", "ATP") + ": " + r["variant"]) if r["group"] != "Legacy" else "Legacy: " + r["variant"]
        p.append(f'<text class="t1" x="{left - 10}" y="{y + bar_h / 2 + 4:.1f}" font-size="11" text-anchor="end">{label}</text>')
        p.append(f'<g><title>{label}: {v:.1f}%</title><rect class="{cls[r["group"]]}" x="{left}" y="{y}" '
                 f'width="{max(X(v) - left, 1):.1f}" height="{bar_h}" rx="4"/></g>')
        tag = f"{v:,.1f}%" + (" ▸" if v > vmax else "")
        lx = X(100) + 6 if 85 < v < 100 else X(v) + 6   # keep labels clear of the 100% reference line
        p.append(f'<text class="t1 halo" x="{lx:.1f}" y="{y + bar_h / 2 + 4:.1f}" font-size="11">{tag}</text>')
    p.append(f'<text class="t2" x="20" y="{axis_y + 40}" font-size="11">One assumption changed at a time. "inputs:" rows replace the '
             'EXIOBASE sector-average input origins (Hartmann Eqs. 8-9).</text>')
    p.append(f'<text class="t2" x="20" y="{axis_y + 56}" font-size="11">Combined extremes are in envelope_climate.csv '
             'and crossed_climate_extrema.csv. Legacy rows: traceability only.</text></svg>')
    Path(path).write_text("\n".join(p) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    impacts = pa.load_impacts()
    checks = validate(impacts)
    summary = summary_rows(impacts)
    sens = sensitivity_rows(impacts)
    write_csv(OUT / "utilisation_summary.csv", summary)
    write_csv(OUT / "budgets_long.csv", long_rows(impacts))
    write_csv(OUT / "sensitivity_climate.csv", sens)
    write_csv(OUT / "regional_contributions.csv", regional_rows())
    write_csv(OUT / "envelope_climate.csv", envelope_rows(impacts))
    crossed = crossed_climate_rows(impacts)
    write_csv(OUT / "crossed_climate_sensitivity.csv", crossed)
    write_csv(OUT / "crossed_climate_extrema.csv", crossed_climate_extrema(crossed))
    (OUT / "milp_budgets.json").write_text(json.dumps(milp_record(impacts), indent=1) + "\n")
    chart_boundaries(OUT / "utilisation_by_boundary.svg", summary)
    chart_sensitivity(OUT / "climate_sensitivity.svg", sens)
    (OUT / "validation.json").write_text(json.dumps(dict(passed=True, checks=checks), indent=1) + "\n")
    inputs = [pa.EXIO_TERMS, pa.ATP_WEIGHTS, pa.ECON_INPUTS, pa.ONS_ABS, pa.IMPACTS, pa.NETWORK_COSTS, pa.AUDIT_TERMS, AUDIT, VERIFY, INVENTORY, ROOT / "data/exio_crosswalk_sensitivity.json",
              ROOT / "pb_allocation.py", Path(__file__)]
    outputs = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != "provenance.json")
    (OUT / "provenance.json").write_text(json.dumps(dict(
        inputs={str(p.resolve().relative_to(ROOT.parent)): sha256(p) for p in inputs},
        outputs={p.name: sha256(p) for p in outputs}), indent=1) + "\n")
    print(f"{len(checks)} checks passed")
    for r in summary:
        print(f"  {SHORT[r['category']]:20s} GVA {r['total_gva_pct']:9.4f}%  ATP {r['ability_to_pay_pct']:9.4f}%  "
              f"historical FCE {r['total_fce_pct']:.4g}-{r['total_fce_corrected_f_pct']:.4g}%  exceeded: {r['exceeded_under']}")


if __name__ == "__main__":
    main()
