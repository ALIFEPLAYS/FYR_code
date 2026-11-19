"""Independent end-to-end audit of the PB allocation results, recomputed from raw sources only.

Nothing here imports pb_allocation or reads the committed EXIOBASE/ATP extracts. Every central result is
rebuilt from: raw EXIOBASE 3.9.6 (2022, ixi), the Hartmann allocation workbook, the PBAESA source code
(population weights and SOS, parsed as literals), raw World Bank/IMF downloads, the SuperPro ICR report,
the saved MILP output (mass) and the saved annual impacts. Exchange-rate and deflator constants are
written out with their sources.

It also replicates PBAESA's total-FCE code in matrix form and evaluates two corrections, so that the
total-FCE discrepancy can be bounded:
  published   term_{a,r} = FR_{a,r} * (1 + (1 + f_a) * sum_b L_ab / cs_b)              (code as written)
  corr_FR     term_{a,r} = FR_{a,r} + (1 + f_a) * sum_b L_ab FR_{b,r} / cs_b            (weights final product b)
  corr_FR_f   term_{a,r} = FR_{a,r} + sum_b L_ab (1 + f_b) FR_{b,r} / cs_b              (also f indexed by column)
  oosterhoff  term_{a,r} = (1 + f_a) * sum_b L_ab FR_{b,r} / cs_b                         (source equations)
  with cs_b = sum_i L_ib, f_b = (cs_b - L_bb) / L_bb (PBAESA Eqs. 3-8 as coded).

The 'oosterhoff' reading follows Oosterhoff et al. (2023) Eqs. 3-7 as printed: S = S_hat + S_tilde with
S_hat = diag(L^T e) and S_tilde the OFF-diagonal entries L_ij (1 + f_i) (Eq. 4: f indexed by the supplier i,
Eq. 5); aSoSOS_{i,j,r} = FR_{j,r} S_ij / S_hat_jj (Eq. 7: weighted by the final product j). Because
L_ii (1 + f_i) = cs_i, every entry equals L_ij (1 + f_i) / cs_j, and the diagonal equals one. PBAESA adds
S_tilde on the diagonal as well, so a sector's own final sales count twice; corr_FR keeps that extra term
and corr_FR_f additionally indexes f by the final product, contrary to Eqs. 4-5.

It additionally exports two sensitivity inputs to data/exiobase_audit_terms_2022.json:
  * full-supply-chain regional origin of the value added embodied in GB Chemicals nec output,
    share_r = sum_{i in r} v_i L_ij / (v L)_j (an alternative to Hartmann's first-tier Eqs. 8-9);
  * the corr_FR_f total-FCE regional terms for GB Chemicals nec.

Run from the repository root (needs EXIOBASE locally, about a minute):
    .venv/bin/python -B 2b_PB_Allocation/tools/audit_from_raw.py --exiobase <EXIOBASE IOT_2022_ixi folder>
Writes 2b_PB_Allocation/data/audit_from_raw_2022.json and 2b_PB_Allocation/data/exiobase_audit_terms_2022.json.
"""
import argparse
import ast
import csv
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
OUT = ROOT / "data/audit_from_raw_2022.json"
TERMS_OUT = ROOT / "data/exiobase_audit_terms_2022.json"
TARGET = ("GB", "Chemicals nec")
GVA_ROWS = ["Other net taxes on production",
            "Compensation of employees; wages, salaries, & employers' social contributions: Low-skilled",
            "Compensation of employees; wages, salaries, & employers' social contributions: Medium-skilled",
            "Compensation of employees; wages, salaries, & employers' social contributions: High-skilled",
            "Operating surplus: Consumption of fixed capital", "Operating surplus: Remaining net operating surplus"]
FCE_COLS = ("Final consumption expenditure by households",
            "Final consumption expenditure by non-profit organisations serving households (NPISH)",
            "Final consumption expenditure by government")
# Currency path, USD2019 -> EUR2022 (UK production):
USD_PER_GBP_2019 = 1.2766            # Bank of England XUAAUSS, 2019 annual average
DEFLATOR_2019, DEFLATOR_2022 = 81.1, 90.6   # ONS YBGB, Blue Book vintage 30 Oct 2026
GBP_PER_EUR_2022 = 0.85276           # ECB euro reference rate, 2022 annual average
ROW = {"East Asia & Pacific": "WA", "South Asia": "WA", "Latin America & Caribbean": "WL", "North America": "WL",
       "Europe & Central Asia": "WE", "Sub-Saharan Africa": "WF", "Middle East & North Africa": "WM",
       "Middle East, North Africa, Afghanistan & Pakistan": "WM"}


def first_existing(*paths):
    return next(p for p in paths if p.exists())


def vendor_dict(fn_name, file):
    src = (ROOT / "vendor/pbaesa/pbaesa" / file).read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == fn_name)
    node = next(n for n in ast.walk(fn) if isinstance(n, ast.Dict))
    return {ast.literal_eval(k): (float(v.args[0].value) if isinstance(v, ast.Call) else ast.literal_eval(v))
            for k, v in zip(node.keys, node.values)}


def atp_weights(regions):
    """Hjalsted Eq. 5 per country (World Bank MER GDP, IMF gap fill, income-group median imputation)."""
    raw = ROOT / "data/worldbank"
    meta = {c["iso2Code"]: c for c in json.loads((raw / "countries.json").read_text())[1]
            if c["region"]["value"].strip() != "Aggregates"}
    get = lambda ind: {(r["country"]["id"], r["date"]): float(r["value"])
                       for r in json.loads((raw / f"{ind}_2017_2022.json").read_text())[1] if r["value"] is not None}
    pop, gdp = get("SP.POP.TOTL"), get("NY.GDP.MKTP.CD")
    imf_g = json.loads((raw / "imf_weo_NGDPD.json").read_text())["values"]["NGDPD"]
    imf_p = json.loads((raw / "imf_weo_LP.json").read_text())["values"]["LP"]
    rows = []
    for iso2, c in list(meta.items()) + [("TW", None)]:
        iso3, income = ("TWN", "High income") if c is None else (c["id"], c["incomeLevel"]["value"].strip())
        p = pop.get((iso2, "2022")) or (imf_p.get(iso3, {}).get("2022") or 0) * 1e6 or None
        if p is None:
            continue
        g = gdp.get((iso2, "2022")) or (imf_g.get(iso3, {}).get("2022") or 0) * 1e9 or None
        if iso2 in regions:
            reg = iso2
        elif iso2 in ("AF", "PK"):
            reg = "WA"
        else:
            reg = ROW[c["region"]["value"].strip()]
        rows.append([iso2, reg, income, p, g])
    med = {}
    for r in rows:
        if r[4]:
            med.setdefault(r[2], []).append(r[4] / r[3])
    for r in rows:
        if not r[4]:
            r[4] = statistics.median(med[r[2]]) * r[3]
    P, G = sum(r[3] for r in rows), sum(r[4] for r in rows)
    raw_w = [(r[3] / P) * (G / P) / (r[4] / r[3]) for r in rows]
    alpha = 1 / sum(raw_w)
    out = dict.fromkeys(regions, 0.0)
    for r, w in zip(rows, raw_w):
        out[r[1]] += w * alpha
    return out, alpha, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exiobase", type=Path, required=True)
    d = ap.parse_args().exiobase
    x = pd.read_csv(d / "x.txt", sep="\t", index_col=[0, 1]).iloc[:, 0]
    idx = list(x.index)
    n, j = len(idx), idx.index(TARGET)
    regions = sorted({r for r, _ in idx})
    reg_of = np.array([regions.index(r) for r, _ in idx])
    F = pd.read_csv(d / "factor_inputs/F.txt", sep="\t", index_col=0, header=[0, 1]).reindex(columns=x.index)
    V = F.loc[GVA_ROWS].sum().to_numpy(float)
    xv = x.to_numpy(float)
    A = pd.read_csv(d / "A.txt", sep="\t", index_col=[0, 1], header=[0, 1]).to_numpy(float)
    L = np.linalg.inv(np.eye(n) - A)
    v = np.divide(V, xv, out=np.zeros(n), where=xv > 0)
    # Eqs. 8-9 from A: column j of the regional input shares (intermediates by region + own GVA).
    col = np.array([A[reg_of == k, j].sum() for k in range(len(regions))])
    col[reg_of[j]] += v[j]
    alpha_j = col / col.sum()
    del A
    gva_r = np.array([V[reg_of == k].sum() for k in range(len(regions))])
    pop = vendor_dict("get_population_weights", "allocation.py")
    w_pop = np.array([pop[r] for r in regions])
    vL_j = float(v @ L[:, j])
    m_eq7 = vL_j / v[j]
    m_pbaesa = xv[j] * float(np.divide(1.0, xv, out=np.zeros(n), where=xv > 0) @ L[:, j])
    terms_eq7 = alpha_j * m_eq7 * V[j] / gva_r
    af = {"total_gva_eq7": float(w_pop @ terms_eq7),
          "total_gva_pbaesa_formula": float(w_pop @ (alpha_j * m_pbaesa * V[j] / gva_r))}
    w_atp, atp_alpha, n_countries = atp_weights(regions)
    af["ability_to_pay_eq7"] = float(np.array([w_atp[r] for r in regions]) @ terms_eq7)

    # Total FCE: PBAESA FR matrix (rows sorted by 'REGION_sector', columns by region) and three readings.
    Y = pd.read_csv(d / "Y.txt", sep="\t", index_col=[0, 1], header=[0, 1])
    fce = sum(Y.xs(c, axis=1, level=1) for c in FCE_COLS)[regions]
    FR = (fce / fce.sum()).to_numpy(float)                       # EXIOBASE order rows, sorted regions
    cs = L.sum(axis=0)
    Ld = np.diag(L).copy()
    f = (cs - Ld) / Ld
    Lrow = L[j]
    published = FR[j] * (1 + (1 + f[j]) * np.sum(Lrow / cs))
    corr_fr = FR[j] + (1 + f[j]) * (Lrow / cs) @ FR
    corr_fr_f = FR[j] + (Lrow * (1 + f) / cs) @ FR
    oosterhoff = (1 + f[j]) * (Lrow / cs) @ FR
    for k, t in (("total_fce_published_formula", published), ("total_fce_corr_FR", corr_fr),
                 ("total_fce_corr_FR_f", corr_fr_f), ("total_fce_oosterhoff", oosterhoff),
                 ("direct_fce", FR[j])):
        af[k] = float(w_pop @ t)
    w_atp_vec = np.array([w_atp[r] for r in regions])
    af["ability_to_pay_fce_corr_FR"] = float(w_atp_vec @ corr_fr)
    af["ability_to_pay_fce_oosterhoff"] = float(w_atp_vec @ oosterhoff)

    wb = pd.read_excel(ROOT / "data/Allocation Factors_2022.xlsx")
    wb.columns = ["region", "sector", "dFCE", "tFCE", "dGVA", "tGVA"]
    row = wb[(wb.region == TARGET[0]) & (wb.sector == TARGET[1])].iloc[0]
    wbx = wb.set_index(["region", "sector"]).reindex(x.index)
    b2b_zero = int(((wbx["dFCE"] == 0) & (wbx["tFCE"] == 0) & (xv > 0)).sum())

    icr = first_existing(REPO / "01_SuperPro/rawSP_reports/mab_fedbatch_1000_sut_ICR.xls",
                         REPO / "1_SuperPro_Results/rawSP_reports/mab_fedbatch_1000_sut_ICR.xls")
    sp = pd.read_excel(icr, sheet_name=0, header=None)
    k = sp.index[sp[1].astype(str).str.strip() == "SUMMARY PER COST ITEM (Entire Process)"][0]
    total_row = sp.iloc[k:k + 14]
    usd_per_kg = float(str(total_row[total_row[1].astype(str).str.strip() == "TOTAL"].iloc[0][7]).replace(",", ""))
    impacts_path = ROOT / "results_final/annual_impacts.csv"
    with impacts_path.open() as fh:
        imp = {r["category"]: float(r["annual_impact"]) for r in csv.DictReader(fh) if r["case_id"] == "two_stage"}
    mass_kg = 861 * 4.65   # 05_Joint_MILP/results/comparison.csv: 861 batches, 4,003.65 kg delivered
    eur_per_usd = (1 / USD_PER_GBP_2019) * (DEFLATOR_2022 / DEFLATOR_2019) / GBP_PER_EUR_2022
    v_net = mass_kg * usd_per_kg / 1e6 * eur_per_usd             # M EUR2022
    share = v_net / xv[j]
    sos = vendor_dict("calculate_exploitation_of_SOS", "utils.py")
    cf_path = ROOT / "vendor/pbaesa/pbaesa/data/Characterization factors_for_eco3101.xlsx"
    sos_wb = {}
    if cf_path.exists():   # optional: not shipped in every copy of this folder
        cf = pd.read_excel(cf_path, sheet_name=1)
        sos_wb = dict(zip(cf["Planetary Boundary Category"], cf["Safe Operating Space"]))
    util = {m: {c: 100 * imp[c] / (sos[c] * af[m] * share) for c in sos}
            for m in ("total_gva_eq7", "ability_to_pay_eq7", "total_fce_published_formula", "total_fce_corr_FR",
                      "total_fce_corr_FR_f", "ability_to_pay_fce_corr_FR", "total_gva_pbaesa_formula",
            "total_fce_oosterhoff", "ability_to_pay_fce_oosterhoff")}
    rec = dict(
        description="Independent recomputation from raw sources (no pb_allocation, no committed extracts).",
        sector_factors=af, workbook=dict(tGVA=float(row.tGVA), tFCE=float(row.tFCE), dFCE=float(row.dFCE)),
        checks=dict(vL_target=vL_j, m_eq7=m_eq7, m_pbaesa=m_pbaesa, alpha_GB=float(alpha_j[regions.index("GB")]),
                    pbaesa_tgva_formula_vs_workbook=af["total_gva_pbaesa_formula"] / row.tGVA - 1,
                    pbaesa_tfce_formula_vs_workbook=af["total_fce_published_formula"] / row.tFCE - 1,
                    direct_fce_vs_workbook=af["direct_fce"] / row.dFCE - 1,
                    corr_FR_minus_oosterhoff_over_direct_fce=(af["total_fce_corr_FR"] - af["total_fce_oosterhoff"])
                    / af["direct_fce"] - 1,
                    sectors_with_output_zero_direct_and_total_fce=b2b_zero,
                    sos_code_vs_workbook={c: (sos[c], sos_wb.get(c)) for c in sos},
                    atp_alpha_2022=atp_alpha, atp_countries=n_countries, atp_weight_GB=w_atp["GB"]),
        network=dict(usd_per_kg=usd_per_kg, mass_kg=mass_kg, eur2022_per_usd2019=eur_per_usd, value_eur_m=v_net,
                     share_of_sector_output=share),
        climate_utilisation_pct={m: u["Climate Change"] for m, u in util.items()},
        utilisation_pct=util,
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (d / "x.txt", d / "A.txt", d / "Y.txt", d / "factor_inputs/F.txt", icr, impacts_path)},
    )
    va_chain = v * L[:, j]
    full_chain = {r: float(va_chain[reg_of == k].sum() / va_chain.sum()) for k, r in enumerate(regions)}
    TERMS_OUT.write_text(json.dumps(dict(
        description=("Sensitivity inputs for GB Chemicals nec from raw EXIOBASE 3.9.6 2022 ixi "
                     "(tools/audit_from_raw.py)."),
        regions=regions,
        full_supply_chain_value_added_share=full_chain,
        full_supply_chain_note="share_r = sum_{i in r} v_i L_ij / (v L)_j; Hartmann Eqs. 8-9 use first-tier inputs instead.",
        tfce_corr_FR_f_by_region=dict(zip(regions, corr_fr_f.tolist())),
        tfce_oosterhoff_by_region=dict(zip(regions, oosterhoff.tolist())),
        tfce_oosterhoff_note="(1 + f_j) sum_k L_jk FR_k,r / cs_k: Oosterhoff et al. (2023) Eqs. 3-7 as printed.",
        tfce_corr_FR_f_note="FR_a + sum_b L_ab (1 + f_b) FR_b / cs_b: both index orientations corrected.",
        source_sha256=rec["source_sha256"]), indent=1) + "\n")
    rec["full_supply_chain_value_added_share_GB"] = full_chain["GB"]
    OUT.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec["climate_utilisation_pct"], indent=1))
    print(json.dumps(rec["checks"], indent=1, default=str)[:1500])


if __name__ == "__main__":
    main()
