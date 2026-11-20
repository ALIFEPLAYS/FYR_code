# Methodology: Total GVA and Ability-to-Pay budgets

## 1. What is calculated

For each of the nine PB indicators c, the network's annual budget and utilisation are

```
budget_c = SOS_c × F × s        utilisation_c = impact_c / budget_c
```

- **SOS_c** is the global safe operating space in the PBAESA units (climate 1 W/m², ocean acidification 0.688 Ωarag, biosphere integrity 10 BII %-points, phosphorus 10 Tg P/yr, nitrogen 62 Tg N/yr, aerosols 0.11 AOD, freshwater 4,000 km³/yr, ozone 14.5 DU, land 85.1 million km²).
- **F** is the share of the global SOS given to the UK pharmaceutical sector (EXIOBASE "Chemicals nec", GB, 2022).
- **s** is the network's share of that sector's output.

The two methods share s and the sector calculation, and differ only in how the SOS is first shared between countries.

## 2. Sector factor F (Hartmann et al. 2026)

```
F = Σ_r w_r · a_r · H / G_r                 (Eqs. 1, 4, 6–9)
H = m · V,   m = g / v,   g = Σ_i v_i L_ij    (Eq. 7: type-I GVA multiplier)
```

| Symbol | Meaning | Value for GB Chemicals nec (2022) |
|---|---|---|
| x, V | sector gross output, direct GVA | €44,744 M, €16,233 M |
| v_i = V_i/x_i, L = (I − A)⁻¹ | value-added intensity, Leontief inverse | |
| g | GVA embodied per € of output | 0.99999995, so H = total GVA ≈ x |
| a_r | share of the sector's first-tier inputs (incl. own GVA) from region r | GB 74.5%, rest of world 25.5% |
| G_r | total GVA of region r | EXIOBASE |
| w_r | region's share of the SOS (step 1, below) | |

**Correction of the published factor.** Hartmann's published workbook gives F = 4.161×10⁻⁴. It was produced by a NumPy broadcasting error in PBAESA (`calculate_total_GVA_per_sector`): dividing a GVA array of shape (n,) by an output array of shape (n, 1) creates an n×n matrix, and the code then uses the *first* sector's GVA for every sector. The resulting "multiplier" contains no value added and is not Eq. 7. Replaying the error reproduces all 6,161 published factors to 3×10⁻¹²; the test suite also runs the original function on a synthetic economy. **Eq. 7 gives F = 1.9729×10⁻⁴**, which we use (2.11× smaller budget than published). Two independent reviews (Claude, Astra) confirmed this; author confirmation is pending.

## 3. Step 1: sharing the SOS between countries

| Method | w_r |
|---|---|
| **Total GVA** (equal per capita) | population share of region r (PBAESA 2022 values; UK 0.853%) |
| **Ability to Pay** (Hjalsted et al. 2021, Eq. 5) | each person gets SOS/P × (world GDP per capita / national GDP per capita) × α, so country c gets (P_c²/D_c) / Σ_k (P_k²/D_k); summed into regions |

Ability to Pay is computed for 218 economies (World Bank 2022 population and GDP at market exchange rates; IMF for Taiwan, South Sudan, Yemen, Eritrea; income-group median for six economies without GDP) and then summed into the 49 EXIOBASE regions. This reproduces Hjalsted's published scaling factor (α = 0.234 for 2017 vs their 0.243); purchasing-power GDP would give 0.431, so market rates are correct. The UK receives 0.0512% of the SOS, 6% of its population share.

**What the Ability-to-Pay result means.** It applies Hjalsted's country weights inside Hartmann's *production* framework, so the budget reflects the income of the people in the regions where the network's value is produced. Because income enters twice (in the weight and in G_r), a euro of value created in India earns 414× the SOS of a euro created in the UK. India and the rest-of-world Asia, Middle East and Africa regions supply 2.7% of the sector's inputs but 78.5% of its Ability-to-Pay budget. The result therefore depends strongly on where inputs come from (§5).

## 4. Step 3: network share s

```
s = X / x,   X = 4,003.65 kg × $92.10/g (SuperPro, 2019) → €378.4 M (2022)
```

Conversion: ÷1.2766 USD/GBP (Bank of England 2019) × 90.6/81.1 UK GDP deflator (ONS) ÷ 0.85276 GBP/EUR (ECB 2022), giving s = 0.846%. Under Eq. 7 the sector's total GVA equals its gross output, so the matching network key is the network's output value. SuperPro's full production cost is used as a **zero-margin valuation**: the real factory-gate price is unknown. The product is bulk drug substance (including bulk filling into bags, without finished-dose fill-finish), which matches the cradle-to-gate impact inventory.

## 5. Results and sensitivity

| | Total GVA | Ability to Pay |
|---|---:|---:|
| Share of global SOS | 1.668×10⁻⁶ | 7.056×10⁻⁷ |
| **Climate (reference)** | **64.9%** | **153.5%** |
| Ocean acidification | 20.7% | 48.9% |
| Aerosol loading | 16.5% | 38.9% |
| Phosphorus | 11.1% | 26.2% |
| Biosphere integrity | 7.8% | 18.4% |
| Other four indicators | < 2.1% | < 4.9% |

Climate, one assumption changed at a time ([`sensitivity_climate.csv`](final_results/sensitivity_climate.csv)):

| Assumption | Cases | Total GVA | Ability to Pay |
|---|---|---:|---:|
| Output value | ONS pharma price indices, no inflation, + sector profit margins | 51.5–74.3% | 121.8–175.8% |
| Input origin | full supply chain, SuperPro cost structure, high-income suppliers, all-UK bound | 58.3–93.3% | 106.1–1,539% |
| Weight data | normalised / World Bank population, EXIO3 country mapping; 2017, PPP, imputation | 64.9–66.5% | 145.5–160.3% |
| Inventory | 3 material × 3 electricity cases | 55.5–65.3% | 131–154% |
| **All combined** ([`climate_ranges.csv`](final_results/climate_ranges.csv)) | 1,296 / 2,268 combinations | **39.5–108.2%** (93% within) | **70.4–1,773%** (91% exceeded) |

**Conclusion.** Under Total GVA the network is within all budgets in almost every scenario; under Ability to Pay it exceeds the climate budget in almost every scenario. The allocation principle, not the design, decides whether the climate boundary is met.

## 6. Use in the MILP

- Budgets are fixed parameters from `pb_allocation.budget_vector(method)` or `final_results/milp_budgets.json`; fix demand, valuation and geography outside the optimiser.
- Constraint rows: Σ_k a_ck q_k / B_c ≤ 1, including fixed impacts, counting each activity once.
- Ability to Pay = 0.4229 × Total GVA for all nine indicators, so ε-constraint and minimax runs give the same designs with relabelled ε. Under Total GVA no indicator is exceeded at the reference plan; under Ability to Pay only climate is, needing a ≥34.8% cut of the characterised climate impact.

## 7. Limitations

1. The impact inventory is a partial subtotal (missing media/buffers, some plastics fabrication, logistics); climate exceeds the Total GVA budget if the full footprint is ≥1.54× the current one.
2. The network's true output value is unknown; the cost comes from one 121-batch SuperPro plant at 2019 prices.
3. "Chemicals nec" is broader than biopharmaceuticals.
4. Ability to Pay depends on the (unknown) origin of purchased inputs and expresses one ethical view (producer-region income).
5. The PBAESA correction awaits author confirmation; the EXIOBASE rest-of-world country mapping (Hartmann vs EXIO3) is a small open question (+2% / +4%).
6. Total-value budgets overlap along supply chains, so "within budget" for one actor does not by itself imply global compliance.

## References

- Hartmann, J.M., Shu, D.Y., Bardow, A. & von der Assen, N. (2026). Allocation and characterisation factors for absolute environmental sustainability assessment. *Scientific Data* 13, 970. [doi:10.1038/s41597-026-07797-w](https://doi.org/10.1038/s41597-026-07797-w); code: [PBAESA d932921](https://github.com/RWTH-LTT/pbaesa/tree/d9329215521feba86005caac65bfa7d0404efb5a).
- Hjalsted, A.W. et al. (2021). Sharing the safe operating space. *J. Ind. Ecol.* 25, 6–19 (Table 2, Eq. 5). [Author copy](https://backend.orbit.dtu.dk/ws/files/234548766/Hjalsted_2020_.pdf)
- Keček, D., Brlek, P. & Buntak, K. (2022). Economic effects of transport sectors on Croatian economy: an input–output approach. *Econ. Res.-Ekon. Istraživanja* 35, 2023–2038 (type-I GVA multiplier).
- Stadler, K. et al. (2018). EXIOBASE 3. *J. Ind. Ecol.* 22, 502–515 (data 3.9.6, 2022).
- Data: World Bank WDI (CC BY 4.0); IMF WEO; ONS (YBGB deflator, PPI C211/C212, Annual Business Survey 2022); Bank of England; ECB; SuperPro `mab_fedbatch_1000_sut_ICR.xls`.

Earlier development versions (including the full audits by Claude and Astra) are kept on the branches `feature/pb-allocation-methods`, `codex/pb-allocation-final` and `feature/final-pb-allocation`.
