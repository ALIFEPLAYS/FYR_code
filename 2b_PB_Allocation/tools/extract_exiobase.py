"""Extract the EXIOBASE terms needed for the GB Chemicals nec allocation factor (optional; needs raw data).

Writes data/exiobase_gb_chemicals_2022.json, which pb_allocation.py reads. Run from the repository root:

    python -B 2b_PB_Allocation/tools/extract_exiobase.py --exiobase <IOT_2022_ixi folder> \
        [--hartmann-workbook "<Allocation Factors_2022.xlsx>"]

Inputs: EXIOBASE 3.9.6, 2022, industry-by-industry (x.txt, A.txt, factor_inputs/F.txt), and the PBAESA
population weights (parsed from vendor/pbaesa/pbaesa/allocation.py). With the optional Hartmann workbook it
also checks, for every sector, that the published total-GVA factors equal PBAESA's (erroneous) formula.

Quantities for target sector j and regions r (Hartmann et al. 2026, Eqs. 1, 4, 6-9):
  v_i = V_i / x_i,  L = (I - A)^-1,  g_j = sum_i v_i L_ij             (embodied GVA per unit output)
  m_j = g_j / v_j                                                      (type-I GVA multiplier, Eq. 7)
  m_j(PBAESA) = x_j * sum_i L_ij / x_i                                 (broadcasting error; no value added)
  a_rj  = (sum_{i in r} A_ij + 1[r = home(j)] v_j) / (column total)    (first-tier input origin, Eqs. 8-9)
  c_rj  = sum_{i in r} v_i L_ij / g_j                                  (full-supply-chain origin; sensitivity)
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/exiobase_gb_chemicals_2022.json"
TARGET = ("GB", "Chemicals nec")
GVA_ROWS = ["Other net taxes on production",
            "Compensation of employees; wages, salaries, & employers' social contributions: Low-skilled",
            "Compensation of employees; wages, salaries, & employers' social contributions: Medium-skilled",
            "Compensation of employees; wages, salaries, & employers' social contributions: High-skilled",
            "Operating surplus: Consumption of fixed capital", "Operating surplus: Remaining net operating surplus"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pbaesa_population_weights():
    src = (ROOT / "vendor/pbaesa/pbaesa/allocation.py").read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "get_population_weights")
    return ast.literal_eval(next(n for n in ast.walk(fn) if isinstance(n, ast.Dict)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exiobase", type=Path, required=True)
    ap.add_argument("--hartmann-workbook", type=Path)
    args = ap.parse_args()
    d = args.exiobase
    x = pd.read_csv(d / "x.txt", sep="\t", index_col=[0, 1]).iloc[:, 0]
    idx = list(x.index)
    n, j = len(idx), idx.index(TARGET)
    regions = sorted({r for r, _ in idx})
    reg_of = np.array([regions.index(r) for r, _ in idx])
    F = pd.read_csv(d / "factor_inputs/F.txt", sep="\t", index_col=0, header=[0, 1]).reindex(columns=x.index)
    V = F.loc[GVA_ROWS].sum().to_numpy(float)
    xv = x.to_numpy(float)
    A = pd.read_csv(d / "A.txt", sep="\t", index_col=[0, 1], header=[0, 1]).to_numpy(float)
    v = np.divide(V, xv, out=np.zeros(n), where=xv > 0)
    inv_x = np.divide(1.0, xv, out=np.zeros(n), where=xv > 0)
    L = np.linalg.inv(np.eye(n) - A)     # explicit inverse, as PBAESA (pymrio.calc_L) computes it
    g = v @ L                            # g_k = sum_i v_i L_ik for every k
    u = inv_x @ L                        # u_k = sum_i L_ik / x_i
    l_j = L[:, j]
    col = np.array([A[reg_of == k, j].sum() for k in range(len(regions))])
    col[reg_of[j]] += v[j]
    first_tier = col / col.sum()
    va_chain = v * l_j
    full_chain = np.array([va_chain[reg_of == k].sum() for k in range(len(regions))]) / va_chain.sum()
    pop = pbaesa_population_weights()
    gva_r = np.array([V[reg_of == k].sum() for k in range(len(regions))])
    w = np.array([pop[r] for r in regions])
    m_eq7, m_pbaesa = g[j] / v[j], xv[j] * u[j]
    record = dict(
        description="EXIOBASE 3.9.6 (2022, ixi) terms for GB Chemicals nec; million EUR, current 2022 basic prices.",
        target=dict(region=TARGET[0], sector=TARGET[1], gross_output=float(xv[j]), direct_gva=float(V[j]),
                    embodied_gva_per_unit_output=float(g[j]), m_eq7=float(m_eq7), m_pbaesa=float(m_pbaesa)),
        regions=regions,
        population_weight_pbaesa=[float(pop[r]) for r in regions],
        gva_by_region=gva_r.tolist(),
        input_origin_first_tier=first_tier.tolist(),
        input_origin_full_supply_chain=full_chain.tolist(),
        sector_factor_eq7=float(w @ (first_tier * m_eq7 * V[j] / gva_r)),
        sector_factor_pbaesa_formula=float(w @ (first_tier * m_pbaesa * V[j] / gva_r)),
        sector_factor_published=4.161413565987638e-4,
        published_source="Hartmann et al. allocation workbook (Zenodo 10.5281/zenodo.17457702), GB Chemicals nec, total GVA",
        source_sha256={k: sha256(d / k) for k in ("x.txt", "A.txt", "factor_inputs/F.txt")},
    )
    if args.hartmann_workbook:
        wb = pd.read_excel(args.hartmann_workbook)
        wb.columns = ["region", "sector", "dFCE", "tFCE", "dGVA", "tGVA"]
        pub = wb.set_index(["region", "sector"]).reindex(x.index)["tGVA"].to_numpy(float)
        a_all = np.zeros((len(regions), n))
        for k in range(len(regions)):
            a_all[k] = A[reg_of == k].sum(axis=0)
        a_all[reg_of, np.arange(n)] += v
        tot = a_all.sum(axis=0)
        a_all = np.divide(a_all, tot, out=np.zeros_like(a_all), where=tot > 0)
        weight = (a_all * (w / gva_r)[:, None]).sum(axis=0)
        af_pbaesa, af_eq7 = weight * xv * u * V, weight * g * xv
        mask = pub > 0
        record["all_sector_check"] = dict(
            sectors=int(mask.sum()),
            pbaesa_formula_vs_published_max_rel_error=float(np.max(np.abs(af_pbaesa[mask] / pub[mask] - 1))),
            eq7_over_published_median=float(np.median(af_eq7[mask] / pub[mask])),
            workbook_sha256=sha256(args.hartmann_workbook))
        record["sector_factor_published"] = float(wb[(wb.region == "GB") & (wb.sector == "Chemicals nec")].tGVA.iloc[0])
    OUT.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: record[k] for k in ("sector_factor_eq7", "sector_factor_pbaesa_formula", "sector_factor_published")}))
    print(json.dumps(record.get("all_sector_check", {})))


if __name__ == "__main__":
    main()
