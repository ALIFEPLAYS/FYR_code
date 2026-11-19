"""Independent check of the Hartmann et al. (2026) total-GVA allocation factors from raw EXIOBASE.

This deliberately uses a different numerical route from tools/alternative_allocations.py:
  * no explicit Leontief inverse: row vectors y = v (I - A)^-1 are obtained by solving (I - A)^T y = v;
  * regional input shares (Eqs. 8-9) are built from the transaction matrix Z, exactly as PBAESA
    add_regional_resolution_to_total_GVA_of_sector does, not from A.

Equations (Hartmann et al. 2026, Scientific Data 13:970):
  Eq. 1   AF_j = sum_r SoSOS_r * GVA^total_{j,r} / GVA_r           SoSOS_r = population share of r
  Eq. 6   GVA^total_{j,r} = s_{j,r} * GVA^direct_j
  Eq. 7   s_{j,r} = alpha_{r,j} * m_j,     m_j = type-I GVA multiplier = (sum_i v_i L_ij) / v_j
  Eq. 8-9 alpha_{r,j} = share of all inputs into j (intermediate + own value added) supplied by region r

What PBAESA computes instead (allocation.py, calculate_total_GVA_per_sector):
  np.divide(V, x) with V of shape (n,) and x of shape (n, 1) broadcasts to an n x n matrix
  M[i, k] = V_k / x_i; .iloc[:, 0] keeps V_0 / x_i. Hence
  m_j^PBAESA = (sum_i L_ij V_0/x_i) / (V_0/x_j) = x_j * sum_i L_ij / x_i,
  which does not depend on value added at all.

Run from the repository root (needs the local EXIOBASE 3.9.6 2022 ixi folder, about a minute):
    .venv/bin/python -B 2b_PB_Allocation/tools/verify_total_gva.py --exiobase <EXIOBASE IOT_2022_ixi folder>
Writes 2b_PB_Allocation/data/total_gva_verification_2022.json.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/total_gva_verification_2022.json"
TARGET = ("GB", "Chemicals nec")
# Exactly the components PBAESA calculate_GVA_per_sector sums.
GVA_COMPONENTS = [
    "Other net taxes on production",
    "Compensation of employees; wages, salaries, & employers' social contributions: Low-skilled",
    "Compensation of employees; wages, salaries, & employers' social contributions: Medium-skilled",
    "Compensation of employees; wages, salaries, & employers' social contributions: High-skilled",
    "Operating surplus: Consumption of fixed capital",
    "Operating surplus: Remaining net operating surplus",
]


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
    d = ap.parse_args().exiobase
    x = pd.read_csv(d / "x.txt", sep="\t", index_col=[0, 1]).iloc[:, 0]
    idx = x.index
    F = pd.read_csv(d / "factor_inputs/F.txt", sep="\t", index_col=0, header=[0, 1]).reindex(columns=idx)
    V = F.loc[GVA_COMPONENTS].sum(axis=0).to_numpy(float)
    xv = x.to_numpy(float)
    n = len(xv)
    regions = sorted(idx.get_level_values(0).unique())
    reg_of = np.array([regions.index(r) for r in idx.get_level_values(0)])

    # Eqs. 8-9 from Z (PBAESA route): rows = Z summed by supplying region + own-region primary input V.
    Z = pd.read_csv(d / "Z.txt", sep="\t", index_col=[0, 1], header=[0, 1], low_memory=False)
    if list(Z.columns) != list(idx) or list(Z.index) != list(idx):
        raise ValueError("Z ordering differs from x")
    Z = Z.apply(pd.to_numeric, errors="coerce").to_numpy(float)
    A = pd.read_csv(d / "A.txt", sep="\t", index_col=[0, 1], header=[0, 1]).to_numpy(float)
    # Integrity check of the local Z file: Z must equal A diag(x). Non-numeric cells are reported and
    # replaced by A_ij x_j; the source file is not modified.
    Zax = A * xv[None, :]
    bad = np.argwhere(~np.isfinite(Z))
    z_anomalies = [dict(row=list(idx[i]), col=list(idx[k]), replaced_with=float(Zax[i, k])) for i, k in bad]
    Z[~np.isfinite(Z)] = Zax[~np.isfinite(Z)]
    z_vs_ax = float(np.max(np.abs(Z - Zax)) / np.max(np.abs(Zax)))
    inputs = np.stack([Z[reg_of == k].sum(axis=0) for k in range(len(regions))])
    del Z, Zax
    inputs[reg_of, np.arange(n)] += V
    col = inputs.sum(axis=0)
    alpha = np.divide(inputs, col, out=np.zeros_like(inputs), where=col > 0)

    IAT = (np.eye(n) - A).T
    del A
    pos = xv > 0
    v = np.where(pos, V / np.where(pos, xv, 1), 0.0)
    vL = np.linalg.solve(IAT, v)                          # (v L)_j for every j
    u = np.linalg.solve(IAT, np.where(pos, 1 / np.where(pos, xv, 1), 0.0))   # sum_i L_ij / x_i
    m_eq7 = np.divide(vL, v, out=np.zeros_like(vL), where=v != 0)
    m_pbaesa = xv * u
    tgva_eq7 = m_eq7 * V                                  # = (vL)_j * x_j
    tgva_pbaesa = m_pbaesa * V

    pop = pbaesa_population_weights()
    gva_r = np.array([V[reg_of == k].sum() for k in range(len(regions))])
    w_over_gva = np.array([pop[r] for r in regions]) / gva_r
    af_eq7 = (alpha * w_over_gva[:, None]).sum(axis=0) * tgva_eq7
    af_pbaesa = (alpha * w_over_gva[:, None]).sum(axis=0) * tgva_pbaesa

    wb = pd.read_excel(ROOT / "data/Allocation Factors_2022.xlsx")
    wb.columns = ["region", "sector", "dFCE", "tFCE", "dGVA", "tGVA"]
    pub = wb.set_index(["region", "sector"]).reindex(idx)["tGVA"].to_numpy(float)
    mask = pub > 0
    rel_pbaesa = np.abs(af_pbaesa[mask] / pub[mask] - 1)
    ratio_eq7 = af_eq7[mask] / pub[mask]
    j = list(idx).index(TARGET)
    rec = dict(
        description="Independent recomputation of Hartmann total-GVA factors (solve-based, Z-based alpha).",
        target=dict(region=TARGET[0], sector=TARGET[1], gross_output_eur_m=float(xv[j]), direct_gva_eur_m=float(V[j]),
                    vL=float(vL[j]), m_eq7=float(m_eq7[j]), m_pbaesa=float(m_pbaesa[j]),
                    total_gva_eq7_eur_m=float(tgva_eq7[j]), total_gva_pbaesa_eur_m=float(tgva_pbaesa[j]),
                    alpha_GB=float(alpha[regions.index("GB"), j]),
                    af_published=float(pub[j]), af_pbaesa_formula=float(af_pbaesa[j]), af_eq7=float(af_eq7[j]),
                    af_eq7_over_published=float(af_eq7[j] / pub[j])),
        all_sectors=dict(n_positive_published=int(mask.sum()),
                         pbaesa_formula_vs_published_max_rel_error=float(rel_pbaesa.max()),
                         eq7_over_published_median=float(np.median(ratio_eq7)),
                         eq7_over_published_p05_p95=[float(np.percentile(ratio_eq7, 5)), float(np.percentile(ratio_eq7, 95))],
                         vL_min_max_over_producing_sectors=[float(vL[v > 0].min()), float(vL[v > 0].max())],
                         sum_af_eq7=float(af_eq7.sum()), sum_af_published=float(np.nansum(pub)),
                         world_output_over_world_gva=float(xv.sum() / V.sum())),
        local_data_integrity=dict(z_non_numeric_cells=z_anomalies,
                                  z_vs_A_times_x_max_rel_deviation=z_vs_ax),
        source_sha256={k: sha256(d / k) for k in ("x.txt", "A.txt", "Z.txt", "factor_inputs/F.txt")},
    )
    OUT.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec["target"], indent=1))
    print(json.dumps(rec["all_sectors"], indent=1))


if __name__ == "__main__":
    main()
