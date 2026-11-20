"""Tests for the final Total GVA and Ability-to-Pay budgets. Run from the repository root:

    python -B -m unittest discover -s 2b_PB_Allocation/tests -v
"""
import hashlib
import importlib.util
import json
import math
import sys
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pb_allocation as pa  # noqa: E402

_spec = importlib.util.spec_from_file_location("atp_builder", ROOT / "tools/build_ability_to_pay_weights.py")
atp_builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(atp_builder)


def synthetic_economy():
    """Two regions x two sectors with balanced accounts (column sums of A plus v equal one)."""
    names = ["AA_s1", "AA_s2", "BB_s1", "BB_s2"]
    A = np.array([[0.10, 0.20, 0.05, 0.00], [0.05, 0.10, 0.10, 0.05],
                  [0.02, 0.05, 0.20, 0.10], [0.03, 0.05, 0.05, 0.15]])
    x = np.array([100.0, 50.0, 400.0, 20.0])
    return names, A, x, (1 - A.sum(axis=0)) * x


class TotalGVATests(unittest.TestCase):
    def test_eq7_total_gva_equals_gross_output(self):
        _, A, x, V = synthetic_economy()
        L = np.linalg.inv(np.eye(4) - A)
        v = V / x
        np.testing.assert_allclose((v @ L) / v * V, x, rtol=1e-12)     # m_j * V_j = x_j

    def test_pbaesa_code_has_the_broadcasting_error(self):
        """Run the original PBAESA function on synthetic data: it ignores value added."""
        names, A, x, V = synthetic_economy()
        L = np.linalg.inv(np.eye(4) - A)
        sys.modules.setdefault("pymrio", types.ModuleType("pymrio"))
        spec = importlib.util.spec_from_file_location("pbaesa_alloc", ROOT / "vendor/pbaesa/pbaesa/allocation.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        L_df = pd.DataFrame(L, index=names, columns=names)
        mod.calculate_GVA_per_sector = lambda year, exiobase_storage_path=None: pd.Series(V, index=names)
        mod.prepare_L_matrix = lambda year, exiobase_storage_path=None: L_df
        mod.get_index = lambda year: L_df.index
        mod.load_satellites = lambda year, **kw: pd.DataFrame(
            {"indout": x}, index=pd.MultiIndex.from_tuples([tuple(n.split("_")) for n in names]))
        got = mod.calculate_total_GVA_per_sector(2022).iloc[:, 0].to_numpy()
        np.testing.assert_allclose(got, x * ((1 / x) @ L) * V, rtol=1e-12)   # x_j sum_i L_ij / x_i * V_j
        self.assertGreater(np.max(np.abs(got / x - 1)), 0.1)                   # differs from Eq. 7 (= x_j)

    def test_extract_records_the_error_and_the_correction(self):
        d = pa._exio()
        self.assertTrue(math.isclose(d["sector_factor_pbaesa_formula"], d["sector_factor_published"], rel_tol=1e-9))
        self.assertAlmostEqual(d["sector_factor_eq7"], 1.972932187632e-4, places=15)
        self.assertLess(d["all_sector_check"]["pbaesa_formula_vs_published_max_rel_error"], 1e-9)
        self.assertAlmostEqual(d["target"]["embodied_gva_per_unit_output"], 1.0, places=6)
        self.assertTrue(math.isclose(pa.sector_factor("total_gva"), d["sector_factor_eq7"], rel_tol=1e-12))


class AbilityToPayTests(unittest.TestCase):
    def test_hjalsted_eq5(self):
        table = [dict(iso2="P", population=100.0, gdp_usd=100.0), dict(iso2="M", population=50.0, gdp_usd=500.0),
                 dict(iso2="R", population=10.0, gdp_usd=1000.0)]
        alpha, w, pop = atp_builder.hjalsted_weights(table)
        raw = {r["iso2"]: r["population"] / 160 * 10 / (r["gdp_usd"] / r["population"]) for r in table}
        self.assertAlmostEqual(alpha, 1 / sum(raw.values()))
        self.assertAlmostEqual(sum(w.values()), 1.0)
        self.assertGreater(w["P"], pop["P"])   # poorer than average: more than an equal share
        self.assertLess(w["R"], pop["R"])

    def test_equal_incomes_give_equal_per_capita(self):
        table = [dict(iso2=k, population=p, gdp_usd=3 * p) for k, p in (("A", 1.0), ("B", 7.0), ("C", 2.0))]
        alpha, w, pop = atp_builder.hjalsted_weights(table)
        self.assertAlmostEqual(alpha, 1.0)
        for k in w:
            self.assertAlmostEqual(w[k], pop[k])

    def test_country_weights_and_validation(self):
        rec = json.loads((pa.DATA / "ability_to_pay_weights.json").read_text())
        _, w, _ = atp_builder.hjalsted_weights(rec["country_rows_2022"])
        self.assertAlmostEqual(pa.weights("ability_to_pay")["GB"], w["GB"], places=15)
        self.assertLess(abs(rec["cases"]["validation_2017_MER"]["alpha"] / 0.243 - 1), 0.05)   # Hjalsted 2017
        self.assertGreater(abs(rec["cases"]["sensitivity_2017_PPP"]["alpha"] / 0.243 - 1), 0.5)
        wb = rec["cases"]["primary_2022_MER"]["population_share_by_region"]
        pb = pa.weights("total_gva")
        self.assertLess(max(abs(wb[r] / pb[r] - 1) for r in pb), 0.07)


class NetworkShareTests(unittest.TestCase):
    def test_reference_value(self):
        v = 861 * 4.65 * 92.09802 / 1000 / 1.2766 * (90.6 / 81.1) / 0.85276
        self.assertTrue(math.isclose(pa.network_value(), v, rel_tol=1e-12))
        self.assertAlmostEqual(pa.network_value(), 378.383, places=3)

    def test_value_bases(self):
        self.assertAlmostEqual(pa.network_value("ppi_C211"), 346.57, places=2)
        self.assertAlmostEqual(pa.network_value("ppi_C212"), 330.39, places=2)
        self.assertGreater(pa.network_value("margin_SIC21"), pa.network_value("margin_SIC21.1"))

    def test_input_origins(self):
        for o in pa.INPUT_ORIGINS:
            s = pa.input_origin(o)
            self.assertAlmostEqual(sum(s.values()), 1.0, places=12)
            self.assertGreaterEqual(min(s.values()), 0.0)
        self.assertAlmostEqual(1 - pa.input_origin()["GB"], 0.2555, places=4)
        self.assertAlmostEqual(1 - pa.input_origin("full_supply_chain")["GB"], 0.3159, places=4)


class ResultTests(unittest.TestCase):
    def climate(self, method, **kw):
        return 100 * pa.utilisation(pa.budget_vector(method, **kw))["Climate Change"]

    def test_central_results(self):
        self.assertAlmostEqual(self.climate("total_gva"), 64.8987, places=3)
        self.assertAlmostEqual(self.climate("ability_to_pay"), 153.4593, places=3)

    def test_sensitivity_values(self):
        self.assertAlmostEqual(self.climate("total_gva", origin="full_supply_chain"), 58.30, places=2)
        self.assertAlmostEqual(self.climate("ability_to_pay", origin="full_supply_chain"), 106.05, places=2)
        self.assertAlmostEqual(self.climate("total_gva", value_basis="ppi_C212"), 74.33, places=2)
        self.assertAlmostEqual(self.climate("ability_to_pay", weight_case="sensitivity_2022_MER_EXIO3"), 159.77, places=2)

    def test_only_climate_exceeds_and_only_under_ability_to_pay(self):
        for m in pa.METHODS:
            over = {c for c, u in pa.utilisation(pa.budget_vector(m)).items() if u > 1}
            self.assertEqual(over, {"Climate Change"} if m == "ability_to_pay" else set())

    def test_switching_method_only_rescales(self):
        r = [pa.budget_vector("ability_to_pay")[c] / pa.budget_vector("total_gva")[c] for c in pa.CATEGORIES]
        self.assertLess(max(r) - min(r), 1e-15)
        self.assertAlmostEqual(r[0], 0.422905, places=6)

    def test_ability_to_pay_attribution(self):
        c = pa.regional_contributions("ability_to_pay")
        a = pa.input_origin()
        four = ("IN", "WA", "WM", "WF")
        self.assertAlmostEqual(sum(a[r] for r in four), 0.0270, places=4)
        self.assertAlmostEqual(sum(c[r] for r in four), 0.7847, places=4)
        self.assertAlmostEqual(1 - c["GB"], 0.9258, places=4)

    def test_impacts_unchanged(self):
        sha = hashlib.sha256((pa.DATA / "annual_impacts.csv").read_bytes()).hexdigest()
        self.assertEqual(sha, "2a02c39fc43edd1ec200dc13bb49bdcb5b10493d8fabdfd6196409d85c8daf01")

    def test_invalid_options_raise(self):
        for kw in (dict(method="status_quo"), dict(value_basis="price"), dict(origin="moon")):
            with self.assertRaises(KeyError):
                pa.budget_vector(**kw)


class OutputTests(unittest.TestCase):
    OUT = ROOT / "final_results"

    def test_milp_file_matches_module(self):
        rec = json.loads((self.OUT / "milp_budgets.json").read_text())
        for m in pa.METHODS:
            for c, b in pa.budget_vector(m).items():
                self.assertTrue(math.isclose(rec["methods"][m]["budgets"][c], b, rel_tol=1e-12))

    def test_workbook_matches_the_joint_model_reader(self):
        b = pd.read_excel(self.OUT / "FYR_PB_budgets.xlsx", sheet_name="PB budgets", header=6, nrows=9, usecols="A:G")
        self.assertEqual(sorted(b["Indicator"]), sorted(pa.CATEGORIES))
        self.assertTrue((b.iloc[:, 2:] > 0).all().all())
        row = b[b["Indicator"] == "Climate Change"].iloc[0]
        self.assertTrue(math.isclose(row["Total GVA (primary)"], pa.budget_vector("total_gva")["Climate Change"], rel_tol=1e-12))

    def test_combined_ranges(self):
        r = pd.read_csv(self.OUT / "climate_ranges.csv").set_index(["method", "extreme"])["climate_pct"]
        self.assertAlmostEqual(r["total_gva", "min"], 39.54, places=2)
        self.assertAlmostEqual(r["total_gva", "max"], 108.22, places=2)
        self.assertAlmostEqual(r["ability_to_pay", "min"], 70.38, places=2)
        self.assertAlmostEqual(r["ability_to_pay", "max"], 1773.2, places=1)


if __name__ == "__main__":
    unittest.main()
