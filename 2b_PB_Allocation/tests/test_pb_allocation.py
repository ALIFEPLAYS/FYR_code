"""Validation tests for pb_allocation.py (total GVA and ability to pay).

The tests check the equations on small synthetic economies, the vendored PBAESA code itself, the
committed data extracts, and the exported MILP budgets. None needs EXIOBASE or Brightway.
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

SPEC = importlib.util.spec_from_file_location("atp_builder", ROOT / "tools/build_ability_to_pay_weights.py")
atp_builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(atp_builder)


def synthetic_economy():
    """Two regions x two sectors with balanced accounts: column sums of A plus v equal one."""
    names = ["AA_s1", "AA_s2", "BB_s1", "BB_s2"]
    A = np.array([[0.10, 0.20, 0.05, 0.00],
                  [0.05, 0.10, 0.10, 0.05],
                  [0.02, 0.05, 0.20, 0.10],
                  [0.03, 0.05, 0.05, 0.15]])
    x = np.array([100.0, 50.0, 400.0, 20.0])
    v = 1 - A.sum(axis=0)
    return names, A, x, v * x


class Equation7Tests(unittest.TestCase):
    def test_total_gva_equals_gross_output_when_accounts_balance(self):
        _, A, x, V = synthetic_economy()
        L = np.linalg.inv(np.eye(4) - A)
        v = V / x
        m = (v @ L) / v                        # type-I GVA multiplier (Hartmann Eq. 7, Kecek et al.)
        np.testing.assert_allclose(v @ L, np.ones(4), rtol=1e-12)
        np.testing.assert_allclose(m * V, x, rtol=1e-12)

    def test_vendored_pbaesa_function_contains_the_broadcasting_error(self):
        """Run the actual PBAESA calculate_total_GVA_per_sector on synthetic inputs."""
        names, A, x, V = synthetic_economy()
        L = np.linalg.inv(np.eye(4) - A)
        sys.modules.setdefault("pymrio", types.ModuleType("pymrio"))
        spec = importlib.util.spec_from_file_location("pbaesa_allocation_under_test",
                                                      ROOT / "vendor/pbaesa/pbaesa/allocation.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        L_df = pd.DataFrame(L, index=names, columns=names)
        x_df = pd.DataFrame({"indout": x}, index=pd.MultiIndex.from_tuples([tuple(n.split("_")) for n in names]))
        mod.calculate_GVA_per_sector = lambda year, exiobase_storage_path=None: pd.Series(V, index=names)
        mod.prepare_L_matrix = lambda year, exiobase_storage_path=None: L_df
        mod.get_index = lambda year: L_df.index
        mod.load_satellites = lambda year, **kw: x_df.copy()
        got = mod.calculate_total_GVA_per_sector(2022).iloc[:, 0].to_numpy()
        buggy = x * ((1 / x) @ L) * V           # m_j = x_j sum_i L_ij / x_i: independent of value added
        eq7 = (V / x) @ L / (V / x) * V          # = x
        np.testing.assert_allclose(got, buggy, rtol=1e-12)
        self.assertGreater(np.max(np.abs(got / eq7 - 1)), 0.1)
        # The code uses V_0/x_i in place of V_i/x_i, so it agrees with Eq. 7 only if every sector has the
        # same absolute GVA (V_i = V_0). Confirm that this is the only thing the error depends on:
        V_equal = np.full(4, 30.0)
        mod.calculate_GVA_per_sector = lambda year, exiobase_storage_path=None: pd.Series(V_equal, index=names)
        np.testing.assert_allclose(mod.calculate_total_GVA_per_sector(2022).iloc[:, 0].to_numpy(),
                                   (V_equal / x) @ L / (V_equal / x) * V_equal, rtol=1e-12)

    def test_two_independent_routes_give_the_same_eq7_factor(self):
        v = json.loads((ROOT / "data/total_gva_verification_2022.json").read_text())["target"]
        self.assertTrue(math.isclose(pa.allocate("total_gva").sector_factor, v["af_eq7"], rel_tol=1e-9))
        self.assertTrue(math.isclose(v["af_pbaesa_formula"], v["af_published"], rel_tol=1e-6))
        self.assertAlmostEqual(v["vL"], 1.0, places=6)

    def test_legacy_factor_reproduces_published_workbook(self):
        wb = pd.read_excel(ROOT / "data/Allocation Factors_2022.xlsx")
        wb.columns = ["region", "sector", "dFCE", "tFCE", "dGVA", "tGVA"]
        row = wb[(wb.region == "GB") & (wb.sector == "Chemicals nec")].iloc[0]
        self.assertTrue(math.isclose(pa.allocate("total_gva_published").sector_factor, row.tGVA, rel_tol=1e-9))
        self.assertTrue(math.isclose(pa.allocate("total_fce").sector_factor, row.tFCE, rel_tol=1e-9))


class AbilityToPayTests(unittest.TestCase):
    def test_hjalsted_eq5_closed_form(self):
        table = [dict(iso2="P", population=100.0, gdp_usd=100.0), dict(iso2="M", population=50.0, gdp_usd=500.0),
                 dict(iso2="R", population=10.0, gdp_usd=1000.0)]
        alpha, w, pop = atp_builder.hjalsted_weights(table)
        gdppc_w = 1600 / 160
        raw = {r["iso2"]: r["population"] / 160 * gdppc_w / (r["gdp_usd"] / r["population"]) for r in table}
        self.assertAlmostEqual(alpha, 1 / sum(raw.values()))
        self.assertAlmostEqual(sum(w.values()), 1.0)
        for k in w:
            self.assertAlmostEqual(w[k], raw[k] * alpha)
        self.assertGreater(w["P"] / pop["P"], 1)   # poorer than average: more than equal per capita
        self.assertLess(w["R"] / pop["R"], 1)

    def test_equal_incomes_reduce_to_equal_per_capita(self):
        table = [dict(iso2=k, population=p, gdp_usd=3 * p) for k, p in (("A", 1.0), ("B", 7.0), ("C", 2.0))]
        alpha, w, pop = atp_builder.hjalsted_weights(table)
        self.assertAlmostEqual(alpha, 1.0)
        for k in w:
            self.assertAlmostEqual(w[k], pop[k])

    def test_aggregation_to_regions_conserves_weight(self):
        rec = json.loads(pa.ATP_WEIGHTS.read_text())
        rows = rec["country_rows_2022"]
        alpha, w, _ = atp_builder.hjalsted_weights(rows)
        regional = rec["cases"]["primary_2022_MER"]["atp_weight_by_region"]
        self.assertAlmostEqual(sum(regional.values()), 1.0, places=12)
        self.assertAlmostEqual(regional["GB"], w["GB"], places=15)
        self.assertAlmostEqual(regional["WF"], sum(w[r["iso2"]] for r in rows if r["exio_region"] == "WF"), places=15)

    def test_reproduces_hjalsted_alpha_and_market_exchange_rate_basis(self):
        cases = json.loads(pa.ATP_WEIGHTS.read_text())["cases"]
        self.assertLess(abs(cases["validation_2017_MER"]["alpha"] / 0.243 - 1), 0.05)
        self.assertGreater(abs(cases["sensitivity_2017_PPP"]["alpha"] / 0.243 - 1), 0.5)

    def test_country_mapping_matches_pbaesa_population_weights(self):
        wb = json.loads(pa.ATP_WEIGHTS.read_text())["cases"]["primary_2022_MER"]["population_share_by_region"]
        pb = pa.population_weights()
        self.assertEqual(set(wb), set(pb))
        self.assertLess(abs(wb["GB"] / pb["GB"] - 1), 0.01)
        self.assertLess(max(abs(wb[r] / pb[r] - 1) for r in pb), 0.07)

    def test_independent_recomputation_of_climate_result(self):
        """Rebuild the ATP climate utilisation from country rows without using pb_allocation's factor code."""
        rec = json.loads(pa.ATP_WEIGHTS.read_text())
        rows = rec["country_rows_2022"]
        pop_w = sum(r["population"] for r in rows)
        gdppc_w = sum(r["gdp_usd"] for r in rows) / pop_w
        raw = {r["iso2"]: r["population"] / pop_w * gdppc_w / (r["gdp_usd"] / r["population"]) for r in rows}
        alpha = 1 / sum(raw.values())
        region_w = {}
        for r in rows:
            region_w[r["exio_region"]] = region_w.get(r["exio_region"], 0.0) + raw[r["iso2"]] * alpha
        ex = json.loads(pa.EXIO_TERMS.read_text())
        t = ex["target"]
        af = sum(region_w[r] * a * t["m_eq7"] * t["direct_gva"] / g
                 for r, a, g in zip(ex["regions"], t["alpha_by_region"], ex["gva_by_region_eur_m"]))
        value = 4003.65 * 92.09802 / 1000 / 1.2766 * (90.6 / 81.1) / 0.85276
        budget = 1.0 * af * value / t["gross_output"]
        pct = 100 * pa.load_impacts()["Climate Change"]["value"] / budget
        self.assertAlmostEqual(pct, 153.4593, places=3)

    def test_rich_region_weight_below_population_share(self):
        w, pop = pa.ability_to_pay_weights(), pa.population_weights()
        self.assertLess(w["GB"], 0.1 * pop["GB"])
        self.assertGreater(w["WF"], pop["WF"])


class NetworkShareTests(unittest.TestCase):
    def test_currency_multiplier(self):
        self.assertTrue(math.isclose(pa.usd2019_to_eur2022(), 1.0261850479108974, rel_tol=1e-12))

    def test_network_value_and_share(self):
        self.assertAlmostEqual(pa.annual_mass_kg(), 4003.65, places=6)
        v = 4003.65 * 92.09802 / 1000 * 1.0261850479108974
        self.assertTrue(math.isclose(pa.network_value_eur_m(), v, rel_tol=1e-12))
        self.assertTrue(math.isclose(pa.network_share(), v / 44744.2686398, rel_tol=1e-9))

    def test_value_bases_are_ordered(self):
        vals = {b: pa.network_value_eur_m(b) for b in pa.VALUE_BASES}
        self.assertLess(vals["cost_no_inflation"], vals["cost"])
        self.assertLess(vals["cost"], vals["cost_plus_margin_SIC21.1"])
        self.assertLess(vals["cost_plus_margin_SIC21.1"], vals["cost_plus_margin_SIC21"])

    def test_ons_margin(self):
        m = pa.ons_gross_operating_margin("21.1")
        self.assertEqual((m["turnover_gbp_m"], m["agva_gbp_m"], m["employment_gbp_m"]), (2834.0, 971.0, 557.0))
        self.assertAlmostEqual(m["margin"], (971 - 557) / 2834)


class InputOriginTests(unittest.TestCase):
    def test_shares_sum_to_one_and_are_non_negative(self):
        for origin in pa.INPUT_ORIGINS:
            s = pa.input_shares(origin)
            self.assertAlmostEqual(sum(s.values()), 1.0, places=12, msg=origin)
            self.assertGreaterEqual(min(s.values()), 0.0, msg=origin)

    def test_network_cost_structure(self):
        c = json.loads(pa.NETWORK_COSTS.read_text())
        self.assertAlmostEqual(c["total_usd2019_per_year"], 51_848_755)
        self.assertAlmostEqual(c["domestic_share"] + c["purchased_goods_share"], 1.0, places=12)
        self.assertAlmostEqual(c["purchased_goods_share"], (3_767_171 + 11_675_193) / 51_848_755, places=9)

    def test_scenario_definitions(self):
        sector = pa.input_shares("sector_average")
        t = pa.exiobase_terms()["target"]
        ratio = (1 - sector["GB"]) / (1 - t["direct_gva"] / t["gross_output"])
        goods = json.loads(pa.NETWORK_COSTS.read_text())["purchased_goods_share"]
        self.assertAlmostEqual(1 - pa.input_shares("network_costs")["GB"], goods * ratio, places=12)
        self.assertAlmostEqual(1 - pa.input_shares("network_costs_all_purchases_imported")["GB"], goods, places=12)
        self.assertEqual(pa.input_shares("domestic_only")["GB"], 1.0)
        income = json.loads(pa.ATP_WEIGHTS.read_text())["region_income_group"]
        hi = pa.input_shares("network_costs_high_income_suppliers")
        self.assertTrue(all(v == 0 for r, v in hi.items() if r != "GB" and income[r] != "High income"))
        # Within the imported part, the sector's regional mix is preserved.
        net = pa.input_shares("network_costs")
        self.assertAlmostEqual(net["CN"] / net["US"], sector["CN"] / sector["US"], places=12)

    def test_conclusions_hold_when_input_origin_alone_changes(self):
        imp = pa.load_impacts()["Climate Change"]["value"]
        for origin in pa.INPUT_ORIGINS:
            g = imp / pa.budget_vector("total_gva", input_origin=origin)["Climate Change"]
            a = imp / pa.budget_vector("ability_to_pay", input_origin=origin)["Climate Change"]
            self.assertLess(g, 1.0, origin)
            self.assertGreater(a, 1.0, origin)

    def test_full_supply_chain_shares(self):
        s = pa.input_shares("full_supply_chain")
        self.assertAlmostEqual(sum(s.values()), 1.0, places=12)
        self.assertAlmostEqual(s["GB"], 0.6841, places=4)

    def test_combined_assumption_envelope(self):
        """Assumptions compound: neither method's conclusion holds for every combination."""
        g_lo, g_hi = pa.envelope("total_gva")
        a_lo, a_hi = pa.envelope("ability_to_pay")
        self.assertAlmostEqual(100 * g_lo["utilisation"], 46.271, places=2)
        self.assertAlmostEqual(100 * g_hi["utilisation"], 106.891, places=2)
        self.assertAlmostEqual(100 * a_lo["utilisation"], 84.170, places=2)
        self.assertAlmostEqual(100 * a_hi["utilisation"], 1763.09, places=1)
        self.assertEqual((g_hi["value_basis"], g_hi["input_origin"]),
                         ("cost_ppi_C212", "network_costs_high_income_suppliers"))
        self.assertEqual((a_lo["value_basis"], a_lo["input_origin"]), ("cost_plus_margin_SIC21", "full_supply_chain"))

    def test_input_origin_rejected_for_fce(self):
        with self.assertRaises(ValueError):
            pa.allocate("total_fce", input_origin="network_costs")


def oosterhoff_literal(L, FR):
    """Oosterhoff et al. (2023) Eqs. 3-7 exactly as printed (matrix form), for one region of final demand."""
    n = L.shape[0]
    I = np.eye(n)
    e = np.ones(n)
    S_hat = np.diag(L.T @ e)                                    # Eq. 3: diag of column sums
    f = ((L - L * I).T @ e) / ((L * I) @ e)                     # Eq. 5: indexed by industry i
    S_tilde = (L + L * np.outer(f, e)) * (1 - I)                # Eq. 4: off-diagonal entries only
    S = S_hat + S_tilde                                         # Eq. 6
    share = FR[None, :] * S / np.diag(S_hat)[None, :]           # Eq. 7: weighted by final product j
    return share.sum(axis=1)                                    # Eq. 8: sum over j


def pbaesa_tfce_as_coded(L, FR):
    """PBAESA calculate_total_FCE_allocation_factor, one region: S_tilde on the diagonal, FR of the row."""
    n = L.shape[0]
    I = np.eye(n)
    e = np.ones((n, 1))
    S_roof = (L.T @ e) * I
    f = ((L - L * I).T @ e) / ((L * I) @ e)
    S = S_roof + L + L * (f @ e.T)
    S_marginal = S @ np.linalg.inv(S_roof)
    return (S_marginal * FR.reshape(n, 1)).sum(axis=1)


class TotalFCETests(unittest.TestCase):
    """Total FCE: Oosterhoff's printed equations against the PBAESA code and the module's closed form."""

    def test_one_sector_economy_receives_exactly_its_whole_share(self):
        L, FR = np.array([[1.0]]), np.array([1.0])
        self.assertAlmostEqual(oosterhoff_literal(L, FR)[0], 1.0)
        self.assertAlmostEqual(pbaesa_tfce_as_coded(L, FR)[0], 2.0)   # own sales counted twice

    def test_pure_final_sector_keeps_its_direct_share_and_b2b_supplier_gets_credit(self):
        # Sector 0 sells only to final demand and buys from sector 1 (pure B2B, no final demand).
        A = np.array([[0.0, 0.0], [0.4, 0.0]])
        L = np.linalg.inv(np.eye(2) - A)
        FR = np.array([1.0, 0.0])
        lit = oosterhoff_literal(L, FR)
        self.assertAlmostEqual(lit[0], 1.0)            # B2C sector retains its direct FCE (Hartmann text)
        self.assertGreater(lit[1], 0.0)                 # B2B supplier is credited (Oosterhoff's purpose)
        coded = pbaesa_tfce_as_coded(L, FR)
        self.assertAlmostEqual(coded[0], 2.0)
        self.assertEqual(coded[1], 0.0)                 # PBAESA: zero for a sector without own final demand

    def test_closed_form_used_in_the_audit_equals_the_literal_equations(self):
        _, A, _, _ = synthetic_economy()
        L = np.linalg.inv(np.eye(4) - A)
        FR = np.array([0.1, 0.5, 0.0, 0.4])
        cs, f = L.sum(axis=0), (L.sum(axis=0) - np.diag(L)) / np.diag(L)
        closed = (1 + f) * ((L / cs[None, :]) @ FR)
        np.testing.assert_allclose(closed, oosterhoff_literal(L, FR), rtol=1e-12)
        corr_fr = FR + closed                           # weighting fixed only (legacy 21.5%)
        np.testing.assert_allclose(corr_fr - closed, FR, rtol=1e-12)

    def test_module_value_and_double_count_identity(self):
        self.assertTrue(math.isclose(pa.allocate("total_fce_oosterhoff").sector_factor, 0.0004352631055, rel_tol=1e-9))
        wb = pd.read_excel(ROOT / "data/Allocation Factors_2022.xlsx")
        wb.columns = ["region", "sector", "dFCE", "tFCE", "dGVA", "tGVA"]
        dfce = wb[(wb.region == "GB") & (wb.sector == "Chemicals nec")].iloc[0].dFCE
        diff = pa.allocate("total_fce_corrected").sector_factor - pa.allocate("total_fce_oosterhoff").sector_factor
        self.assertTrue(math.isclose(diff, dfce, rel_tol=1e-9))
        self.assertEqual(pa.METHODS["total_fce_oosterhoff"].role, "supporting")
        self.assertEqual(pa.METHODS["total_fce"].role, "legacy")


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.impacts = pa.load_impacts()

    def test_budget_is_sos_times_factor_times_share(self):
        a = pa.allocate("ability_to_pay")
        for c, sos in pa.SOS.items():
            self.assertTrue(math.isclose(a.budgets[c], sos * a.sector_factor * a.network_share, rel_tol=1e-14))

    def test_hand_calculation_of_primary_climate_budget(self):
        budget = 1.0 * 1.972932187632152e-4 * (4003.65 * 92.09802 / 1000 * 1.0261850479108974) / 44744.2686398
        self.assertTrue(math.isclose(pa.budget_vector()["Climate Change"], budget, rel_tol=1e-9))
        self.assertTrue(math.isclose(self.impacts["Climate Change"]["value"] / budget, 0.648987, rel_tol=1e-5))

    def test_reference_results(self):
        expected = {"total_gva": 64.8987, "ability_to_pay": 153.4593, "total_fce": 15.0051,
                    "total_fce_corrected": 21.5093, "total_fce_corrected_f": 23.2311,
                    "total_fce_oosterhoff": 29.4169, "ability_to_pay_fce_oosterhoff": 28.8981,
                    "total_gva_published": 30.7687, "ability_to_pay_regional": 148.0237}
        for m, pct in expected.items():
            got = 100 * pa.utilisation(self.impacts, pa.budget_vector(m))["Climate Change"]
            self.assertAlmostEqual(got, pct, places=3, msg=m)

    def test_only_climate_exceeds_and_only_under_ability_to_pay(self):
        for m in ("total_gva", "ability_to_pay", "total_fce", "total_fce_corrected"):
            over = {c for c, u in pa.utilisation(self.impacts, pa.budget_vector(m)).items() if u > 1}
            self.assertEqual(over, {"Climate Change"} if m == "ability_to_pay" else set(), m)

    def test_switching_methods_only_rescales(self):
        for b in ("ability_to_pay", "total_fce", "total_fce_corrected"):
            ratios = list(pa.relative_importance("total_gva", b).values())
            self.assertLess(max(ratios) - min(ratios), 1e-12 * max(ratios))

    def test_regional_contributions_sum_to_one(self):
        for m in ("total_gva", "ability_to_pay"):
            self.assertAlmostEqual(sum(pa.regional_contributions(m).values()), 1.0, places=12)

    def test_invalid_inputs_raise(self):
        with self.assertRaises(KeyError):
            pa.allocate("grandfathering")
        with self.assertRaises(KeyError):
            pa.network_value_eur_m("market_price")
        with self.assertRaises(ValueError):
            pa.sector_factor({"GB": 1.0}, {"FR": 1.0})
        with self.assertRaises(ValueError):
            pa.utilisation(self.impacts, dict(pa.budget_vector(), **{"Climate Change": 0.0}))

    def test_impacts_file_unchanged(self):
        digest = hashlib.sha256(pa.IMPACTS.read_bytes()).hexdigest()
        self.assertEqual(digest, "2a02c39fc43edd1ec200dc13bb49bdcb5b10493d8fabdfd6196409d85c8daf01")


class RawAuditTests(unittest.TestCase):
    """Compare with tools/audit_from_raw.py, which rebuilds every result from raw sources."""

    def setUp(self):
        path = ROOT / "data/audit_from_raw_2022.json"
        if not path.exists():
            self.skipTest("run tools/audit_from_raw.py with local EXIOBASE first")
        self.audit = json.loads(path.read_text())

    def test_all_central_results_reproduced_from_raw_sources(self):
        imp = pa.load_impacts()
        pairs = {"total_gva_eq7": "total_gva", "ability_to_pay_eq7": "ability_to_pay",
                 "total_fce_published_formula": "total_fce", "total_fce_corr_FR": "total_fce_corrected",
                 "total_fce_corr_FR_f": "total_fce_corrected_f", "ability_to_pay_fce_corr_FR": "ability_to_pay_fce",
                 "total_gva_pbaesa_formula": "total_gva_published"}
        for k, m in pairs.items():
            mine = pa.utilisation(imp, pa.budget_vector(m))
            for c, pct in self.audit["utilisation_pct"][k].items():
                self.assertTrue(math.isclose(100 * mine[c], pct, rel_tol=1e-9), f"{m} {c}")

    def test_published_factors_reproduced_by_pbaesa_formulas(self):
        c = self.audit["checks"]
        self.assertLess(abs(c["pbaesa_tgva_formula_vs_workbook"]), 1e-9)
        self.assertLess(abs(c["pbaesa_tfce_formula_vs_workbook"]), 1e-9)
        self.assertEqual(c["sectors_with_output_zero_direct_and_total_fce"], 146)

    def test_sos_values(self):
        for cat, (code, workbook) in self.audit["checks"]["sos_code_vs_workbook"].items():
            self.assertEqual(pa.SOS[cat], code)
            if workbook is not None:
                self.assertEqual(code, workbook)


class ExportTests(unittest.TestCase):
    def test_milp_budget_file_matches_module(self):
        path = ROOT / "results_allocation_final/milp_budgets.json"
        if not path.exists():
            self.skipTest("run tools/finalise_allocation.py first")
        rec = json.loads(path.read_text())
        self.assertEqual(rec["default_method"], "total_gva")
        self.assertEqual(set(rec["methods"]), {"total_gva", "ability_to_pay"})
        for m in ("total_gva", "ability_to_pay"):
            self.assertEqual(set(rec["methods"][m]["budgets"]), set(pa.CATEGORIES))
            for c, b in pa.budget_vector(m).items():
                self.assertTrue(math.isclose(rec["methods"][m]["budgets"][c], b, rel_tol=1e-12))
        eps = rec["methods"]["ability_to_pay"]["epsilon_equivalent_vs_total_gva"]
        self.assertTrue(math.isclose(eps, 0.4229, rel_tol=1e-3))


class FinalisationSensitivityTests(unittest.TestCase):
    def test_pharmaceutical_output_valuation(self):
        self.assertAlmostEqual(pa.network_value_eur_m("cost_ppi_C211"), 346.57, places=2)
        self.assertAlmostEqual(pa.network_value_eur_m("cost_ppi_C212"), 330.39, places=2)

    def test_matched_coverage_and_concordance(self):
        cases = json.loads(pa.ATP_WEIGHTS.read_text())["cases"]
        self.assertEqual(cases["sensitivity_2022_MER_matched_PPP"]["n_countries"], 199)
        self.assertEqual(cases["sensitivity_2022_PPP"]["n_countries"], 199)
        climate = pa.load_impacts()["Climate Change"]["value"]
        for case, expected in (("sensitivity_2022_MER_matched_PPP", 160.31),
                               ("sensitivity_2022_PPP", 149.70),
                               ("sensitivity_2022_MER_EXIO3", 159.77)):
            b = pa.budget_vector("ability_to_pay", atp_case=case)["Climate Change"]
            self.assertAlmostEqual(100 * climate / b, expected, delta=0.01)

    def test_population_scenarios_are_normalised(self):
        for case in pa.POPULATION_CASES[1:]:
            weights = pa.geographical_weights("total_gva", population_case=case)
            self.assertAlmostEqual(sum(weights.values()), 1.0, places=12)

    def test_crossed_scenarios_reproduce_the_reference_and_extremes(self):
        import csv
        out = ROOT / "results_allocation_final"
        with (out / "crossed_climate_sensitivity.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        for method, case in (("total_gva", "hartmann_reference"), ("ability_to_pay", "primary_2022_MER")):
            references = [r for r in rows if (r["method"], r["weight_case"], r["value_basis"],
                          r["input_origin"], r["inventory_case"]) ==
                          (method, case, "cost", "sector_average", "reference")]
            self.assertEqual(len(references), 1)
            self.assertAlmostEqual(float(references[0]["budget_W_m2"]) /
                                   pa.budget_vector(method)["Climate Change"], 1.0, places=12)
        with (out / "crossed_climate_extrema.csv").open(newline="") as stream:
            extrema = list(csv.DictReader(stream))
        for row, expected in zip(extrema, (39.5421, 108.2168, 70.3777, 1773.1986), strict=True):
            self.assertAlmostEqual(float(row["climate_pct"]), expected, places=3)


if __name__ == "__main__":
    unittest.main()
