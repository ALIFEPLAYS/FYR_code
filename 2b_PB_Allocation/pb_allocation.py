"""Planetary-boundary (PB) budgets for the UK mAb network: Total GVA (primary) and Ability to Pay (alternative).

    budget_c = SOS_c x F x s          utilisation_c = impact_c / budget_c

    F = sum_r w_r * a_r * H / G_r     sector factor for GB Chemicals nec (Hartmann et al. 2026, Eqs. 1, 4, 6-9)
    H = m * V = g * x                 total GVA, m = type-I GVA multiplier (Eq. 7)
    s = X / x                         network output value / sector gross output

The two methods differ only in the geographical weights w_r:
    total_gva       w_r = population share (Hartmann/PBAESA 2022)
    ability_to_pay  w_r = sum over countries c in r of (P_c^2 / D_c) / sum_k (P_k^2 / D_k)   (Hjalsted et al. 2021, Eq. 5)

So every budget vector is SOS x one scalar: switching method rescales all nine budgets by the same factor.
MILP entry point: budget_vector("total_gva") or budget_vector("ability_to_pay"). See METHODOLOGY.md.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
METHODS = ("total_gva", "ability_to_pay")

# Global safe operating space in the pinned PBAESA units (same method version as the impacts).
SOS = {
    "Climate Change": 1.0,                   # W/m2
    "Ocean Acidification": 0.688,            # aragonite saturation state
    "Change in Biosphere Integrity": 10.0,   # BII percentage points
    "Phosphorus Cycle": 10.0,                # Tg P/yr
    "Nitrogen Cycle": 62.0,                  # Tg N/yr
    "Atmospheric Aerosol Loading": 0.11,     # AOD
    "Freshwater Use": 4000.0,                # km3/yr
    "Stratospheric Ozone Depletion": 14.5,   # DU
    "Land-system Change": 85.1,              # million km2
}
CATEGORIES = tuple(SOS)

# Sensitivity options (first entry of each = reference)
VALUE_BASES = ("cost", "ppi_C211", "ppi_C212", "no_inflation", "margin_SIC21.1", "margin_SIC21")
INPUT_ORIGINS = ("sector_average", "full_supply_chain", "network_costs", "network_costs_high_income",
                 "all_purchases_imported", "domestic_only")
WEIGHT_CASES = {
    "total_gva": ("hartmann", "hartmann_normalised", "worldbank", "worldbank_EXIO3"),
    "ability_to_pay": ("primary_2022_MER", "validation_2017_MER", "sensitivity_2022_MER_no_imputation",
                       "sensitivity_2022_PPP", "sensitivity_2017_PPP", "sensitivity_2022_MER_matched_PPP",
                       "sensitivity_2022_MER_EXIO3"),
}


def _json(name):
    return json.loads((DATA / name).read_text())


def _exio():
    return _json("exiobase_gb_chemicals_2022.json")


# ----------------------------------------------------------------------------- step 1: geography
def weights(method="total_gva", case=None):
    """Geographical weights w_r over the 49 EXIOBASE regions."""
    case = case or WEIGHT_CASES[method][0]
    if case not in WEIGHT_CASES[method]:
        raise KeyError(f"Unknown weight case {case!r} for {method}")
    d = _exio()
    if method == "total_gva" and case.startswith("hartmann"):
        w = dict(zip(d["regions"], d["population_weight_pbaesa"]))
        if case == "hartmann_normalised":
            total = sum(w.values())
            w = {r: v / total for r, v in w.items()}
        return w
    atp = _json("ability_to_pay_weights.json")["cases"]
    if method == "total_gva":   # World Bank 2022 population, Hartmann (World Bank region) or EXIO3 mapping
        key = "primary_2022_MER" if case == "worldbank" else "sensitivity_2022_MER_EXIO3"
        return atp[key]["population_share_by_region"]
    return atp[case]["atp_weight_by_region"]


# ----------------------------------------------------------------------------- step 2: sector
def input_origin(origin="sector_average"):
    """a_r: regional origin used to split total GVA (Hartmann Eqs. 8-9 = 'sector_average')."""
    d = _exio()
    regions = d["regions"]
    if origin == "sector_average":
        return dict(zip(regions, d["input_origin_first_tier"]))
    if origin == "full_supply_chain":
        return dict(zip(regions, d["input_origin_full_supply_chain"]))
    if origin == "domestic_only":
        return {r: float(r == "GB") for r in regions}
    sector = dict(zip(regions, d["input_origin_first_tier"]))
    foreign = {r: a for r, a in sector.items() if r != "GB"}
    t = d["target"]
    import_ratio = sum(foreign.values()) / (1 - t["direct_gva"] / t["gross_output"])   # imports / intermediates
    c = _json("economic_inputs.json")["superpro_cost_structure_usd2019_per_year"]
    goods = sum(c["items"][k] for k in c["purchased_goods_items"]) / c["total"]
    if origin == "network_costs":
        imported, mix = goods * import_ratio, foreign
    elif origin == "network_costs_high_income":
        income = _json("ability_to_pay_weights.json")["region_income_group"]
        imported, mix = goods * import_ratio, {r: a for r, a in foreign.items() if income[r] == "High income"}
    elif origin == "all_purchases_imported":
        imported, mix = goods, foreign
    else:
        raise KeyError(f"Unknown input origin {origin!r}")
    total = sum(mix.values())
    shares = {r: imported * mix.get(r, 0.0) / total for r in regions}
    shares["GB"] = 1 - imported
    return shares


def sector_factor(method="total_gva", origin="sector_average", weight_case=None):
    """F = sum_r w_r a_r m V / G_r  (Hartmann Eqs. 1, 4, 6-9, with the Eq. 7 multiplier)."""
    if method not in METHODS:
        raise KeyError(f"Unknown method {method!r}; choose from {METHODS}")
    d = _exio()
    t = d["target"]
    w, a = weights(method, weight_case), input_origin(origin)
    total_gva = t["m_eq7"] * t["direct_gva"]
    return sum(w[r] * a[r] * total_gva / g for r, g in zip(d["regions"], d["gva_by_region"]))


# ----------------------------------------------------------------------------- step 3: network share
def eur2022_per_usd2019(index="gdp_deflator"):
    c = _json("economic_inputs.json")["currency"]
    if index == "gdp_deflator":
        p = c["UK_GDP_deflator"]
    elif index == "none":
        p = {"2019": 1.0, "2022": 1.0}
    else:
        p = c["pharma_output_price_index"][index]
    return (1 / c["USD_per_GBP_2019"]["value"]) * (p["2022"] / p["2019"]) / c["GBP_per_EUR_2022"]["value"]


def network_value(basis="cost"):
    """Annual network output value, million EUR 2022 (same units as EXIOBASE gross output)."""
    e = _json("economic_inputs.json")
    kg = e["production"]["annual_batches"]["value"] * e["production"]["batch_size_kg"]["value"]
    usd_m = kg * 1000 * e["unit_production_cost_usd2019_per_g"]["value"] / 1e6
    if basis == "cost":
        return usd_m * eur2022_per_usd2019()
    if basis in ("ppi_C211", "ppi_C212"):
        return usd_m * eur2022_per_usd2019(basis.split("_")[1])
    if basis == "no_inflation":
        return usd_m * eur2022_per_usd2019("none")
    if basis.startswith("margin_"):
        o = e["ons_abs_2022_gbp_m"][basis.replace("margin_", "").replace("SIC", "SIC ")]
        margin = (o["agva"] - o["employment_costs"]) / o["turnover"]
        return usd_m * eur2022_per_usd2019() / (1 - margin)
    raise KeyError(f"Unknown value basis {basis!r}; choose from {VALUE_BASES}")


def network_share(basis="cost"):
    return network_value(basis) / _exio()["target"]["gross_output"]


# ----------------------------------------------------------------------------- budgets and utilisation
def sos_fraction(method="total_gva", value_basis="cost", origin="sector_average", weight_case=None):
    return sector_factor(method, origin, weight_case) * network_share(value_basis)


def budget_vector(method="total_gva", value_basis="cost", origin="sector_average", weight_case=None):
    """MILP entry point: {PB category: annual budget in native PBAESA units}. Defaults = reference."""
    f = sos_fraction(method, value_basis, origin, weight_case)
    return {c: sos * f for c, sos in SOS.items()}


def load_impacts():
    """Reference annual impacts of the saved 861-batch network (characterised subtotal, native units)."""
    with (DATA / "annual_impacts.csv").open(newline="", encoding="utf-8") as fh:
        rows = {r["category"]: r for r in csv.DictReader(fh) if r["case_id"] == "two_stage"}
    return {c: dict(value=float(rows[c]["annual_impact"]), unit=rows[c]["unit"]) for c in CATEGORIES}


def climate_impact_cases():
    """Annual climate impact (W/m2) for the 9 saved material x electricity inventory cases."""
    def total(o):
        return o if isinstance(o, (int, float)) else sum(total(v) for v in (o.values() if isinstance(o, dict) else o))
    return {k: total(v) for k, v in _json("climate_inventory_scenarios.json")["scenarios"].items()}


def utilisation(budgets, impacts=None):
    """{category: impact / budget} (1.0 = budget fully used)."""
    impacts = impacts or load_impacts()
    return {c: impacts[c]["value"] / budgets[c] for c in CATEGORIES}


def regional_contributions(method="total_gva", origin="sector_average", weight_case=None):
    """Share of the sector factor contributed by each region (sums to one)."""
    d = _exio()
    t = d["target"]
    w, a = weights(method, weight_case), input_origin(origin)
    terms = {r: w[r] * a[r] * t["m_eq7"] * t["direct_gva"] / g for r, g in zip(d["regions"], d["gva_by_region"])}
    total = sum(terms.values())
    return {r: v / total for r, v in terms.items()}
