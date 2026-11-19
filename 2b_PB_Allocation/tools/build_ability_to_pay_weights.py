"""Country-level ability-to-pay (ATP) weights of Hjalsted et al. (2021), aggregated to EXIOBASE regions.

Hjalsted et al. (2021) Eq. 5 gives each individual in country c the share

    SoSOS_ind,c = SOS / Pop_world * (GDPpc_world / GDPpc_c) * alpha

so the share of the whole country is

    w_c = (Pop_c / Pop_world) * (GDPpc_world / GDPpc_c) * alpha,
    alpha = 1 / sum_c (Pop_c / Pop_world) * (GDPpc_world / GDPpc_c)      (Supporting Information Eq. S1)

so that sum_c w_c = 1. Hjalsted et al. report alpha = 0.243 for 2017. The principle is applied at
country level and the country weights are then summed into the 49 EXIOBASE regions. Doing it on the
49 aggregated regions instead understates income dispersion inside the five Rest-of-World regions, so it
overstates the share given to rich countries (w is convex in GDP per capita).

Data (raw downloads committed in 2b_PB_Allocation/data/worldbank/, World Bank CC BY 4.0; IMF WEO):
  population      World Bank SP.POP.TOTL; Taiwan from IMF WEO LP
  GDP             World Bank NY.GDP.MKTP.CD (current US$, market exchange rates); IMF WEO NGDPD gap fill
  PPP sensitivity World Bank NY.GDP.MKTP.PP.CD
  remaining gaps  GDP per capita imputed as the median of the country's World Bank income group

Country -> EXIOBASE mapping: the 44 EXIOBASE countries by ISO2; all other countries by World Bank
region, following Hartmann et al. (2026) Table 1 (RoW Asia and Pacific = East Asia & Pacific + South
Asia; RoW America = Latin America & Caribbean + North America; RoW Europe = Europe & Central Asia;
RoW Africa = Sub-Saharan Africa; RoW Middle East = Middle East & North Africa). The World Bank moved
Afghanistan and Pakistan into a new "Middle East, North Africa, Afghanistan & Pakistan" region; they are
mapped to RoW Asia and Pacific here, as in the classification Hartmann et al. used.

Run from the repository root (no EXIOBASE needed):
    .venv/bin/python -B 2b_PB_Allocation/tools/build_ability_to_pay_weights.py
Writes 2b_PB_Allocation/data/ability_to_pay_weights.json.
"""
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/worldbank"
OUT = ROOT / "data/ability_to_pay_weights.json"
CROSSWALK = ROOT / "data/exio_crosswalk_sensitivity.json"
EXIO_REGIONS = json.loads((ROOT / "data/exiobase_alternatives_2022.json").read_text())["regions"]
RoW = {"East Asia & Pacific": "WA", "South Asia": "WA", "Latin America & Caribbean": "WL",
       "North America": "WL", "Europe & Central Asia": "WE", "Sub-Saharan Africa": "WF",
       "Middle East & North Africa": "WM", "Middle East, North Africa, Afghanistan & Pakistan": "WM"}
MOVED_TO_SOUTH_ASIA = {"AF", "PK"}  # reclassified by the World Bank after Hartmann et al. (2026)
IMF_ISO3_FOR = {"TW": "TWN"}        # not a World Bank economy
TAIWAN = dict(iso2="TW", name="Taiwan", income="High income")
HJALSTED_ALPHA_2017 = 0.243


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_wb(indicator):
    rows = json.loads((RAW / f"{indicator}_2017_2022.json").read_text())[1]
    return {(r["country"]["id"], r["date"]): float(r["value"]) for r in rows if r["value"] is not None}


def load_imf(code):
    return json.loads((RAW / f"imf_weo_{code}.json").read_text())["values"][code]


def countries():
    meta = json.loads((RAW / "countries.json").read_text())[1]
    out = {}
    for c in meta:
        region = c["region"]["value"].strip()
        if region == "Aggregates":
            continue
        out[c["iso2Code"]] = dict(iso2=c["iso2Code"], iso3=c["id"], name=c["name"], wb_region=region,
                                  income=c["incomeLevel"]["value"].strip())
    out["TW"] = dict(TAIWAN, iso3="TWN", wb_region="East Asia & Pacific")
    return out


def exio_region(c):
    if c["iso2"] in EXIO_REGIONS:
        return c["iso2"]
    if c["iso2"] in MOVED_TO_SOUTH_ASIA:
        return "WA"
    return RoW[c["wb_region"]]


def country_table(year, gdp_indicator="NY.GDP.MKTP.CD", impute=True):
    """Population and GDP (current US$) per country, with the source of every value."""
    meta = countries()
    pop, gdp = load_wb("SP.POP.TOTL"), load_wb(gdp_indicator)
    imf_gdp, imf_pop = load_imf("NGDPD"), load_imf("LP")
    table = []
    for iso2, c in meta.items():
        iso3 = IMF_ISO3_FOR.get(iso2, c["iso3"])
        p = pop.get((iso2, year))
        p_src = "World Bank SP.POP.TOTL"
        if p is None and imf_pop.get(iso3, {}).get(year) is not None:
            p, p_src = imf_pop[iso3][year] * 1e6, "IMF WEO LP"
        if p is None:
            continue  # no population: the economy does not enter the world total either
        g = gdp.get((iso2, year))
        g_src = f"World Bank {gdp_indicator}"
        if g is None and gdp_indicator == "NY.GDP.MKTP.CD" and imf_gdp.get(iso3, {}).get(year) is not None:
            g, g_src = imf_gdp[iso3][year] * 1e9, "IMF WEO NGDPD"
        table.append(dict(c, exio_region=exio_region(c), population=p, population_source=p_src,
                          gdp_usd=g, gdp_source=g_src if g is not None else None))
    if impute:
        by_income = {}
        for r in table:
            if r["gdp_usd"] is not None:
                by_income.setdefault(r["income"], []).append(r["gdp_usd"] / r["population"])
        for r in table:
            if r["gdp_usd"] is None:
                r["gdp_usd"] = statistics.median(by_income[r["income"]]) * r["population"]
                r["gdp_source"] = f"imputed: median GDP per capita of {r['income']}"
    else:
        table = [r for r in table if r["gdp_usd"] is not None]
    return table


def hjalsted_weights(table):
    """Hjalsted Eq. 5 with alpha from Eq. S1. Returns (alpha, {iso2: w_c}, {iso2: pop share})."""
    pop_w = sum(r["population"] for r in table)
    gdppc_w = sum(r["gdp_usd"] for r in table) / pop_w
    raw = {r["iso2"]: (r["population"] / pop_w) * gdppc_w / (r["gdp_usd"] / r["population"]) for r in table}
    alpha = 1 / sum(raw.values())
    return alpha, {k: v * alpha for k, v in raw.items()}, {r["iso2"]: r["population"] / pop_w for r in table}


def aggregate(table, weights):
    out = {r: 0.0 for r in EXIO_REGIONS}
    for row in table:
        out[row["exio_region"]] += weights[row["iso2"]]
    return out


def build():
    primary = country_table("2022")
    ppp = country_table("2022", "NY.GDP.MKTP.PP.CD", False)
    ppp_countries = {r["iso2"] for r in ppp}
    overrides = json.loads(CROSSWALK.read_text())["overrides"]
    concordance = []
    for row in primary:
        region = row["exio_region"]
        if row["iso3"] in overrides:
            override = overrides[row["iso3"]]
            if override["reference"] != region:
                raise ValueError(f"Stale crosswalk override for {row['iso3']}")
            region = override["alternative"]
        concordance.append(dict(row, exio_region=region))

    cases = {}
    for name, year, ind, impute, table in (
        ("primary_2022_MER", "2022", "NY.GDP.MKTP.CD", True, primary),
        ("validation_2017_MER", "2017", "NY.GDP.MKTP.CD", True, country_table("2017")),
        ("sensitivity_2022_MER_no_imputation", "2022", "NY.GDP.MKTP.CD", False,
         country_table("2022", impute=False)),
        ("sensitivity_2022_PPP", "2022", "NY.GDP.MKTP.PP.CD", False, ppp),
        ("sensitivity_2017_PPP", "2017", "NY.GDP.MKTP.PP.CD", False,
         country_table("2017", "NY.GDP.MKTP.PP.CD", False)),
        ("sensitivity_2022_MER_matched_PPP", "2022", "NY.GDP.MKTP.CD", True,
         [row for row in primary if row["iso2"] in ppp_countries]),
        ("sensitivity_2022_MER_EXIO3", "2022", "NY.GDP.MKTP.CD", True, concordance),
    ):
        alpha, w, popshare = hjalsted_weights(table)
        cases[name] = dict(
            year=int(year), gdp_indicator=ind, imputation=impute, n_countries=len(table), alpha=alpha,
            atp_weight_by_region=aggregate(table, w), population_share_by_region=aggregate(table, popshare),
            imputed=[r["name"] for r in table if r["gdp_source"].startswith("imputed")],
            gap_filled_from_imf=[r["name"] for r in table if r["gdp_source"] == "IMF WEO NGDPD"
                                 or r["population_source"] == "IMF WEO LP"],
            UK=dict(population_share=popshare["GB"], atp_weight=w["GB"], ratio=w["GB"] / popshare["GB"]))
    meta = countries()
    income = {r: (meta[r]["income"] if r in meta else "aggregate (treated as not high income)") for r in EXIO_REGIONS}
    record = dict(
        description="Hjalsted et al. (2021) Eq. 5 ability-to-pay weights, country level, summed to EXIOBASE regions.",
        formula="w_c = (Pop_c/Pop_world) * (GDPpc_world/GDPpc_c) * alpha; alpha = 1/sum_c(...); w_r = sum_{c in r} w_c",
        hjalsted_alpha_2017_published=HJALSTED_ALPHA_2017,
        regions=EXIO_REGIONS, cases=cases,
        region_income_group=income,
        region_income_group_note=("World Bank income classification at retrieval for the 44 EXIOBASE countries "
                                  "(Taiwan: high income); the five Rest-of-World aggregates are mixed and treated "
                                  "as not high income. Used only for the input-origin sensitivity."),
        country_rows_2022=[{k: r[k] for k in ("iso2", "name", "exio_region", "income", "population",
                                               "population_source", "gdp_usd", "gdp_source")} for r in primary],
        source_sha256={p.name: sha256(p) for p in sorted(RAW.glob("*.json"))},
    )
    record["source_sha256"][CROSSWALK.name] = sha256(CROSSWALK)
    OUT.write_text(json.dumps(record, indent=1) + "\n")
    return record


if __name__ == "__main__":
    rec = build()
    for name, c in rec["cases"].items():
        print(f"{name:38s} n={c['n_countries']:3d} alpha={c['alpha']:.4f}  UK pop {c['UK']['population_share']:.5f}"
              f"  UK ATP {c['UK']['atp_weight']:.6f}  imputed={len(c['imputed'])}")
