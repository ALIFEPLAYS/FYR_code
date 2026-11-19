"""Planetary-boundary budgets for the UK mAb network: total GVA (primary) and ability to pay (alternative).

This module is the single entry point for PB budgets. It reads only committed data files, so it runs
without EXIOBASE, Brightway or the MILP. The joint MILP should call ``budget_vector(method)``.

Three separate steps (never mix their keys):

1. Geography (who gets the SOS): a weight w_r for each of the 49 EXIOBASE regions.
   * equal per capita (Hartmann et al. 2026 Eq. 2):   w_r = Pop_r / Pop_world     (PBAESA 2022 weights)
   * ability to pay   (Hjalsted et al. 2021 Eq. 5):   w_r = sum_{c in r} (Pop_c/Pop_w)(GDPpc_w/GDPpc_c) alpha
     computed per country, then summed to regions (tools/build_ability_to_pay_weights.py).

2. Sector (which economic activity in r generates that value): Hartmann et al. (2026) Eqs. 1, 4, 6-9,
   production perspective with total value:
       AF_j = sum_r w_r * GVA^total_{j,r} / GVA_r,
       GVA^total_{j,r} = alpha_{r,j} * m_j * GVA^direct_j,      m_j = (v L)_j / v_j   (type-I multiplier)
   For j = GB Chemicals nec, (vL)_j = 0.99999995, so m_j GVA_j equals gross output x_j. The published
   workbook used a broadcasting error instead (see tools/verify_total_gva.py); it is kept only as a
   labelled legacy factor.

3. Network (our share of the sector): Hartmann's usage notes recommend disaggregating by the share of
   the sector's economic value attributable to the system (e.g. turnover). Under Eq. 7 the sector's total
   GVA is its gross output, so the matched key is the network's gross output:  s = V_net / x_j.

    budget_c = SOS_c * AF_j * s          utilisation_c = impact_c / budget_c

Steps 1 and 2 multiply into one scalar, so every budget vector here is the global SOS vector times a
single fraction f. Switching methods rescales all nine budgets by the same factor and never changes
their relative importance (checked by ``relative_importance``).
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
EXIO_TERMS = DATA / "exiobase_alternatives_2022.json"
ATP_WEIGHTS = DATA / "ability_to_pay_weights.json"
ECON_INPUTS = DATA / "allocation_inputs_20261119.json"
ONS_ABS = DATA / "ons/abssectionsas.xlsx"
IMPACTS = ROOT / "results_final/annual_impacts.csv"
TARGET = ("GB", "Chemicals nec")   # NACE 21 -> EXIOBASE p24.d / i24.d (public concordance)

# Global safe operating space, native PBAESA control-variable units (CF workbook, sheet 2; nitrogen from
# the pinned implementation). Retained so impacts and budgets share one method version.
SOS = {
    "Climate Change": 1.0,                     # W/m2 energy imbalance
    "Change in Biosphere Integrity": 10.0,     # % BII
    "Ocean Acidification": 0.688,              # Omega_arag
    "Stratospheric Ozone Depletion": 14.5,     # DU
    "Phosphorus Cycle": 10.0,                  # Tg P/yr
    "Nitrogen Cycle": 62.0,                    # Tg N/yr
    "Freshwater Use": 4000.0,                  # km3/yr
    "Atmospheric Aerosol Loading": 0.11,       # AOD
    "Land-system Change": 85.1,                # million km2
}
CATEGORIES = tuple(SOS)


# --------------------------------------------------------------------------- input loading
def _json(path):
    return json.loads(Path(path).read_text())


def _positive(value, label):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive, got {value!r}")
    return value


def exiobase_terms():
    """Committed EXIOBASE 3.9.6 (2022, ixi) extract for GB Chemicals nec and the 49 regions."""
    d = _json(EXIO_TERMS)
    if d["target"]["region"] != TARGET[0] or d["target"]["sector"] != TARGET[1]:
        raise ValueError("EXIOBASE extract is not for GB Chemicals nec")
    if len(d["regions"]) != 49 or abs(sum(d["target"]["alpha_by_region"]) - 1) > 1e-9:
        raise ValueError("EXIOBASE extract failed its region/input-share checks")
    return d


def population_weights():
    """PBAESA/Hartmann 2022 equal-per-capita weights (Eq. 2), by EXIOBASE region."""
    d = exiobase_terms()
    return dict(zip(d["regions"], d["population_weight"]))


def ability_to_pay_weights(case="primary_2022_MER"):
    """Hjalsted Eq. 5 weights by EXIOBASE region (country-level calculation, see the builder tool)."""
    rec = _json(ATP_WEIGHTS)
    w = rec["cases"][case]["atp_weight_by_region"]
    if abs(sum(w.values()) - 1) > 1e-9:
        raise ValueError(f"ATP weights for {case} do not sum to one")
    return w


def ability_to_pay_weights_regional():
    """Legacy: Hjalsted Eq. 5 applied directly to the 49 aggregated regions (EXIOBASE GVA as GDP).
    Understates income dispersion inside the Rest-of-World regions; kept to reproduce the earlier 148%."""
    d = exiobase_terms()
    raw = {r: p * p / g for r, p, g in zip(d["regions"], d["population_weight"], d["gva_by_region_eur_m"])}
    total = sum(raw.values())
    return {r: v / total for r, v in raw.items()}


# --------------------------------------------------------------------------- input origin (Eqs. 8-9)
NETWORK_COSTS = DATA / "network_cost_structure.json"
AUDIT_TERMS = DATA / "exiobase_audit_terms_2022.json"
INPUT_ORIGINS = {
    "sector_average": "EXIOBASE first-tier input shares of GB Chemicals nec (Hartmann default). PRIMARY.",
    "network_costs": ("SuperPro cost structure: facility, labour, QC, utilities and waste are UK (70.2%); purchased "
                      "raw materials and consumables (29.8%) are imported at the sector's intermediate-import ratio "
                      "and regional mix."),
    "network_costs_high_income_suppliers": ("As network_costs, but imported goods come only from high-income regions "
                                            "(pro rata to the sector mix), as for single-use consumables and media."),
    "network_costs_all_purchases_imported": ("SuperPro cost structure with all purchased goods imported in the "
                                             "sector's regional mix (largest foreign share; bound)."),
    "domestic_only": "All first-tier inputs from GB, i.e. only the GB term of Hartmann Eq. 1 (bound).",
    "full_supply_chain": ("Regional origin of all value added embodied in the sector's output (v L), instead of "
                          "first-tier inputs; tests Hartmann's Eqs. 8-9 simplification."),
}


def input_shares(origin="sector_average"):
    """alpha_r: share of first-tier inputs (intermediates + own value added) supplied by region r."""
    d = exiobase_terms()
    regions, t = d["regions"], d["target"]
    sector = dict(zip(regions, t["alpha_by_region"]))
    if origin == "sector_average":
        return sector
    foreign = {r: a for r, a in sector.items() if r != "GB"}
    if origin == "domestic_only":
        return {r: (1.0 if r == "GB" else 0.0) for r in regions}
    if origin == "full_supply_chain":
        shares = _json(AUDIT_TERMS)["full_supply_chain_value_added_share"]
        return {r: shares[r] for r in regions}
    costs = _json(NETWORK_COSTS)
    goods = costs["purchased_goods_share"]
    # Sector intermediate-import ratio: foreign first-tier inputs / all intermediate inputs.
    va_share = t["direct_gva"] / t["gross_output"]
    import_ratio = sum(foreign.values()) / (1 - va_share)
    if origin == "network_costs":
        imported, mix = goods * import_ratio, foreign
    elif origin == "network_costs_high_income_suppliers":
        income = _json(ATP_WEIGHTS)["region_income_group"]
        imported, mix = goods * import_ratio, {r: a for r, a in foreign.items() if income[r] == "High income"}
    elif origin == "network_costs_all_purchases_imported":
        imported, mix = goods, foreign
    else:
        raise KeyError(f"Unknown input origin {origin!r}; choose from {sorted(INPUT_ORIGINS)}")
    total = sum(mix.values())
    shares = {r: imported * mix.get(r, 0.0) / total for r in regions}
    shares["GB"] = 1 - imported
    return shares


# --------------------------------------------------------------------------- sector factor (step 2)
def total_gva_terms(multiplier="eq7", input_origin="sector_average"):
    """GVA^total_{j,r} / GVA_r for each region r (Hartmann Eqs. 4, 6-9).

    multiplier="eq7"       type-I GVA multiplier as defined in the paper (correct)
    multiplier="published" the multiplier PBAESA actually computed (broadcasting error; legacy only)
    input_origin           which first-tier input shares alpha_r to use (see INPUT_ORIGINS)
    """
    d = exiobase_terms()
    t = d["target"]
    m = {"eq7": t["m_eq7"], "published": t["m_implemented"]}[multiplier]
    alpha = input_shares(input_origin)
    return {r: alpha[r] * m * t["direct_gva"] / g for r, g in zip(d["regions"], d["gva_by_region_eur_m"])}


def total_fce_terms(variant="published"):
    """FCE^total_{j,r} / FCE_r by region (Hartmann Eqs. 3, 5; Oosterhoff 2023 scaling).

    "published": as PBAESA computes it; "corrected": the code-comment reading (sum over the final
    demand of the products sector j supplies); "corrected_f": additionally indexes the Oosterhoff
    scaling f by the final product. None is confirmed by the authors; report the range."""
    if variant == "corrected_f":
        return dict(_json(AUDIT_TERMS)["tfce_corr_FR_f_by_region"])
    t = exiobase_terms()["target"]
    key = {"published": "tfce_implemented_by_region", "corrected": "tfce_alternative_by_region"}[variant]
    return dict(zip(exiobase_terms()["regions"], t[key]))


def sector_factor(terms, weights):
    """Hartmann Eq. 1: AF_j = sum_r w_r * term_r."""
    if set(terms) != set(weights):
        raise ValueError("Region sets of value terms and geographical weights differ")
    return sum(weights[r] * terms[r] for r in terms)


# --------------------------------------------------------------------------- network share (step 3)
def ons_gross_operating_margin(sic="21.1", year=2022):
    """(aGVA - employment costs) / turnover from ONS ABS. Gross surplus can include
    depreciation already counted in SuperPro. Use this only as an illustrative uplift."""
    df = pd.read_excel(ONS_ABS, "Section C", header=None)
    row = df[(df[0].astype(str).str.strip() == sic) & (df[2].astype(str).str.strip() == str(year))]
    if len(row) != 1:
        raise ValueError(f"ONS ABS row SIC {sic} {year} not found exactly once")
    turnover, agva, employment = (float(row.iloc[0][k]) for k in (4, 5, 7))
    return dict(sic=sic, year=year, turnover_gbp_m=turnover, agva_gbp_m=agva, employment_gbp_m=employment,
                margin=(agva - employment) / turnover)


def usd2019_to_eur2022(deflate=True, *, pharma_index=None):
    """EUR2022 per USD2019 for UK production: USD->GBP (2019), UK GDP deflator 2019->2022, GBP->EUR (2022)."""
    c = _json(ECON_INPUTS)["currency_conversion"]
    usd_per_gbp = _positive(c["USD_per_GBP_2019"]["value"], "USD/GBP 2019")
    ratio = _positive(c["UK_GDP_deflator_2022"]["value"], "deflator") / _positive(c["UK_GDP_deflator_2019"]["value"], "deflator")
    if pharma_index is not None:
        ppi = c["pharma_output_price_indices"][pharma_index]
        ratio = _positive(ppi["2022"], "2022 PPI") / _positive(ppi["2019"], "2019 PPI")
    gbp_per_eur = _positive(c["GBP_per_EUR_2022"]["value"], "GBP/EUR 2022")
    return (1 / usd_per_gbp) * (ratio if deflate else 1.0) / gbp_per_eur


def annual_mass_kg():
    p = _json(ECON_INPUTS)["production"]
    return _positive(p["annual_batches"]["value"], "batches") * _positive(p["batch_size_kg"]["value"], "batch size")


VALUE_BASES = {
    "cost": "SuperPro full unit production cost (USD2019 92.10/g), GDP-deflated. Zero-margin output-value proxy. REFERENCE.",
    "cost_no_inflation": "As 'cost' without 2019->2022 price inflation (stand-in for a flat pharma producer price index).",
    "cost_plus_margin_SIC21.1": "Cost / (1 - ONS SIC 21.1 gross operating margin): illustrative uplift for basic pharmaceutical products.",
    "cost_ppi_C211": "SuperPro cost with ONS domestic basic pharmaceutical products price index.",
    "cost_ppi_C212": "SuperPro cost with ONS domestic pharmaceutical preparations price index.",
    "cost_plus_margin_SIC21": "Cost / (1 - ONS SIC 21 gross operating margin): illustrative uplift for all pharmaceuticals.",
}


def network_value_eur_m(basis="cost"):
    """Annual value of the network's output, million EUR2022 (the unit of EXIOBASE x)."""
    usd_per_g = _positive(_json(ECON_INPUTS)["unit_value_scenarios_USD2019_per_g"]["superpro_unit_production_cost"]["value"],
                          "unit cost")
    base = annual_mass_kg() * 1000 * usd_per_g / 1e6
    if basis == "cost":
        return base * usd2019_to_eur2022()
    if basis == "cost_no_inflation":
        return base * usd2019_to_eur2022(deflate=False)
    if basis in ("cost_ppi_C211", "cost_ppi_C212"):
        return base * usd2019_to_eur2022(pharma_index=basis.rsplit("_", 1)[1])
    if basis in VALUE_BASES and basis.startswith("cost_plus_margin_SIC"):
        return base * usd2019_to_eur2022() / (1 - ons_gross_operating_margin(basis.split("SIC")[1])["margin"])
    raise KeyError(f"Unknown value basis {basis!r}; choose from {sorted(VALUE_BASES)}")


def network_share(basis="cost"):
    """s = V_net / x_j, both million EUR2022 at basic prices."""
    return network_value_eur_m(basis) / _positive(exiobase_terms()["target"]["gross_output"], "sector output")


# --------------------------------------------------------------------------- methods
@dataclass(frozen=True)
class Method:
    key: str
    label: str
    role: str          # primary | alternative | supporting | legacy
    geography: str     # equal per capita | ability to pay (...)
    value: str         # total GVA | total FCE (...)
    note: str = ""


METHODS = {
    "total_gva": Method("total_gva", "Total GVA (Hartmann Eq. 7)", "primary", "equal per capita",
                        "total GVA, Eq. 7"),
    "ability_to_pay": Method("ability_to_pay", "Ability to pay x total GVA", "alternative",
                             "ability to pay, country level, GDP at market exchange rates, 2022", "total GVA, Eq. 7"),
    "total_fce": Method("total_fce", "Total FCE (published)", "supporting", "equal per capita", "total FCE, published"),
    "total_fce_corrected": Method("total_fce_corrected", "Total FCE (corrected reading)", "supporting",
                                  "equal per capita", "total FCE, comment-consistent reading"),
    "total_fce_corrected_f": Method("total_fce_corrected_f", "Total FCE (corrected reading, f by product)",
                                    "supporting", "equal per capita", "total FCE, both index orientations corrected"),
    "ability_to_pay_fce": Method("ability_to_pay_fce", "Ability to pay x total FCE (corrected)", "supporting",
                                 "ability to pay, country level, 2022", "total FCE, comment-consistent reading"),
    "total_gva_published": Method("total_gva_published", "Total GVA, published factor", "legacy",
                                  "equal per capita", "total GVA, PBAESA broadcasting error",
                                  "Do not use: factor not computed with Eq. 7."),
    "ability_to_pay_regional": Method("ability_to_pay_regional", "Ability to pay (49-region) x total GVA", "legacy",
                                      "ability to pay on aggregated EXIOBASE regions", "total GVA, Eq. 7",
                                      "Superseded: regional aggregation overstates the UK share."),
}


POPULATION_CASES = (
    "hartmann_reference", "hartmann_normalised", "worldbank_hartmann", "worldbank_EXIO3",
)


def geographical_weights(method, atp_case="primary_2022_MER", *, population_case="hartmann_reference"):
    if population_case not in POPULATION_CASES:
        raise KeyError(population_case)
    if population_case != "hartmann_reference":
        if method != "total_gva":
            raise ValueError("Population-weight sensitivities apply to Total GVA only")
        if population_case == "hartmann_normalised":
            weights = population_weights()
            return {r: share / sum(weights.values()) for r, share in weights.items()}
        case = {"worldbank_hartmann": "primary_2022_MER",
                "worldbank_EXIO3": "sensitivity_2022_MER_EXIO3"}[population_case]
        return _json(ATP_WEIGHTS)["cases"][case]["population_share_by_region"]
    if method in ("total_gva", "total_fce", "total_fce_corrected", "total_fce_corrected_f", "total_gva_published"):
        return population_weights()
    if method in ("ability_to_pay", "ability_to_pay_fce"):
        return ability_to_pay_weights(atp_case)
    if method == "ability_to_pay_regional":
        return ability_to_pay_weights_regional()
    raise KeyError(method)


def value_terms(method, input_origin="sector_average"):
    if method in ("total_gva", "ability_to_pay", "ability_to_pay_regional"):
        return total_gva_terms("eq7", input_origin)
    if input_origin != "sector_average":
        raise ValueError(f"input_origin applies to the Eq. 7 total-GVA methods only, not {method!r}")
    if method == "total_gva_published":
        return total_gva_terms("published")
    if method == "total_fce":
        return total_fce_terms("published")
    if method in ("total_fce_corrected", "ability_to_pay_fce"):
        return total_fce_terms("corrected")
    if method == "total_fce_corrected_f":
        return total_fce_terms("corrected_f")
    raise KeyError(method)


@dataclass(frozen=True)
class Allocation:
    method: Method
    value_basis: str
    atp_case: str
    input_origin: str
    sector_factor: float
    network_share: float
    network_value_eur_m: float
    budgets: dict = field(repr=False)
    population_case: str = "hartmann_reference"

    @property
    def sos_fraction(self):
        return self.sector_factor * self.network_share


def allocate(method="total_gva", value_basis="cost", atp_case="primary_2022_MER", input_origin="sector_average",
             *, population_case="hartmann_reference"):
    """Budgets for one method. Defaults are the documented primary choices."""
    if method not in METHODS:
        raise KeyError(f"Unknown allocation method {method!r}; choose from {sorted(METHODS)}")
    weights = geographical_weights(method, atp_case, population_case=population_case)
    af = sector_factor(value_terms(method, input_origin), weights)
    value = network_value_eur_m(value_basis)
    s = value / _positive(exiobase_terms()["target"]["gross_output"], "sector output")
    budgets = {c: sos * af * s for c, sos in SOS.items()}
    return Allocation(METHODS[method], value_basis, atp_case, input_origin, af, s,
                      value, budgets, population_case=population_case)


def budget_vector(method="total_gva", value_basis="cost", atp_case="primary_2022_MER", input_origin="sector_average",
                  *, population_case="hartmann_reference"):
    """MILP entry point: {PB category: allocated annual budget in native PBAESA units}."""
    return dict(allocate(method, value_basis, atp_case, input_origin, population_case=population_case).budgets)


def relative_importance(method_a="total_gva", method_b="ability_to_pay", **kw):
    """Budget ratio b/a per category. Identical ratios mean pure rescaling (no change in relative importance)."""
    a, b = budget_vector(method_a, **kw), budget_vector(method_b, **kw)
    return {c: b[c] / a[c] for c in CATEGORIES}


def envelope(method="total_gva", category="Climate Change", case="two_stage"):
    """Utilisation over every combination of value basis x input origin (assumptions compound
    multiplicatively, so one-at-a-time ranges understate the spread). Returns (min, max) dicts."""
    impact = load_impacts(case)[category]["value"]
    rows = [dict(value_basis=b, input_origin=o, utilisation=impact / budget_vector(method, b, input_origin=o)[category])
            for b in VALUE_BASES for o in INPUT_ORIGINS]
    return min(rows, key=lambda r: r["utilisation"]), max(rows, key=lambda r: r["utilisation"])


def regional_contributions(method="total_gva", atp_case="primary_2022_MER", input_origin="sector_average"):
    """Share of the sector factor that each region contributes (sums to one)."""
    terms, w = value_terms(method, input_origin), geographical_weights(method, atp_case)
    af = sector_factor(terms, w)
    return {r: w[r] * terms[r] / af for r in terms}


# --------------------------------------------------------------------------- impacts and utilisation
def load_impacts(case="two_stage"):
    """Annual characterised impacts of a saved network case (read-only; never modified here)."""
    with IMPACTS.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["case_id"] == case]
    out = {r["category"]: dict(value=float(r["annual_impact"]), unit=r["unit"]) for r in rows}
    if set(out) != set(SOS):
        raise ValueError(f"Impact categories for {case} do not match the nine PBAESA indicators")
    return out


def utilisation(impacts, budgets):
    """{category: impact / budget}. 1.0 means the allocated budget is fully used."""
    return {c: float(impacts[c]["value"] if isinstance(impacts[c], dict) else impacts[c]) / _positive(budgets[c], c)
            for c in CATEGORIES}
