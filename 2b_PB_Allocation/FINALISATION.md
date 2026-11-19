# Finalisation of the GVA and Ability-to-Pay budgets

The reference remains the saved 861-batch UK network, delivering 4,003.65 kg of bulk mAb per year. Both central calculations reproduce independently from the underlying data. We retain their inputs for the reference comparison because the evidence does not identify a better mAb-specific producer price or supplier mix.

| Reference method | Sector factor | Network climate budget (W/m²) | Climate utilisation |
|---|---:|---:|---:|
| Total GVA, Hartmann first tier | 0.0001972932188 | 1.66843 × 10⁻⁶ | 64.90% |
| Country-level Ability-to-Pay × Total GVA, producer regions | 0.0000834363 | 7.05586 × 10⁻⁷ | 153.46% |

The characterised reference climate subtotal is 1.0827868785 × 10⁻⁶ W/m². All nine native-unit budgets are in [the budget export](results_allocation_final/milp_budgets.json). The [methodology](METHODOLOGY.md) derives the equations and preserves Claude Code’s audit. The [separate Astra audit](audit/INDEPENDENT_AUDIT_ASTRA.md) records the independent raw-data reconstruction. Its FCE analysis is retained as research evidence. The inherited FCE calculations remain unresolved supporting comparisons and are excluded from the decision budget export.

## Valuation and system boundary

Use the current-2022 basic-price output share of UK Chemicals n.e.c. as the matched scaling variable. The NACE mapping supports the sector selection, while the broad industry mix and unusual zero compensation entries in the UK sector make its upstream mix a weak proxy for biotechnology. The SuperPro $92.09802/g figure is average production cost for a 121-batch case. Facility costs represent 53.36% of this total. The extrapolation assumes the same cost per gram at 861 batches and a zero-margin bulk-output valuation. Actual basic-price output can vary in either direction. Gross operating margin cases are illustrative uplifts and can overlap with depreciation already included in SuperPro.

The branch reference converts the US-dollar 2019 estimate through GBP, the UK GDP deflator and 2022 GBP/EUR to €378.38 million. The two domestic pharmaceutical output price series are separate scenarios. C211 uses 107.7 and 110.2 (2019 and 2022), yielding €346.57 million. C212 uses 109.9 and 107.2, yielding €330.39 million. Neither measures the contract price of this product. [ONS C211 series](https://www.ons.gov.uk/economy/inflationandpriceindices/timeseries/g32a/ppi) · [ONS C212 series](https://www.ons.gov.uk/economy/inflationandpriceindices/timeseries/g32b/ppi)

The bulk-drug-substance gate excludes retail fill–finish and downstream logistics. Detailed SuperPro ICR entries assign labour, cartridges and 20-L bags to filling even though the summary reports zero. Reconcile bulk handling and packaging against the inventory before treating the economic and environmental gates as equivalent. The impact input remains a partial characterised subtotal. Unresolved inputs include media and buffers, components of plastics fabrication and resin, and some elementary-flow mappings. The climate result is radiative forcing in W/m² for annual activity. The nine outputs are the pinned method’s indicators, with its own definitions, including global freshwater volume.

## Geography and ethics

The first-tier reference allocates total chain GVA geographically using purchases plus own GVA. It reproduces Hartmann’s rule. The full-chain sensitivity instead traces the location of embodied GVA. It raises the UK sector’s attributed foreign share from 25.55% to 31.59% and moves climate utilisation to **58.30% GVA** and **106.05% ATP**. The SuperPro cost structure with assumed import shares yields **294.83% ATP**. These are scenarios with no measured biopharma procurement mix. India and three rest-of-world groups account for 78.47% of the reference ATP budget from 2.70% of first-tier inputs. All foreign regions account for 92.58% of the budget from 25.55% of inputs.

The ATP scenario substitutes Hjalsted’s normalised country entitlements into Hartmann’s producer-region allocation. That applies a capacity-based principle to production regions. The original Hjalsted product step distributes by final consumption. Dividing by regional GVA can compound the inverse-income effect. Normalising country shares does not prevent overlaps among product-level supply chains.

The alternate EXIO3 mapping moves 14 of the reference economies between rest-of-world groups and gives **159.77% ATP**. It is a concordance diagnostic pending confirmation of the release-specific aggregation. The equal-per-capita diagnostics use normalised Hartmann weights or common World Bank population inputs under either mapping. Normalised Hartmann changes the reference GVA result to about 64.9045%. Matched coverage isolates the income measure among the same 199 economies, yielding **160.31% MER** and **149.70% PPP** compared with 153.46% for the 218-economy reference. [Concordance overrides](data/exio_crosswalk_sensitivity.json) · [Country weight snapshots](data/ability_to_pay_weights.json)

## Sensitivity

One-at-a-time results are in [sensitivity_climate.csv](results_allocation_final/sensitivity_climate.csv). The [economic envelope](results_allocation_final/envelope_climate.csv) varies valuation and sourcing at reference weights and inventory. Its extrema are 46.27–106.89% for GVA and 84.17–1,763.09% for ATP. The [crossed cases](results_allocation_final/crossed_climate_sensitivity.csv) also vary income or population cases and nine saved material and electricity combinations. They give **39.54–108.22% GVA** and **70.38–1,773.20% ATP**. See [extrema with the responsible assumptions](results_allocation_final/crossed_climate_extrema.csv). These ranges describe the selected discrete scenarios. They provide neither probabilities nor physical bounds, and procurement changes would require environmental factors and freight that vary with actual supplier origins.

The physical climate scenarios are reproduced from [committed attributed contribution snapshots](data/climate_inventory_scenarios.json) derived from the saved joint plan and assessed coefficients. The joint replay agrees with the frozen two-stage reference climate subtotal to numerical precision. The snapshot records hashes of the local replay inputs. The export sums these terms in W/m² from the committed snapshot and validates the reference replay. This reproducibility check does not regenerate background ecoinvent exchanges.

## Integration status

Use the GVA budget as an exploratory fixed cap and ATP as a separate ethical sensitivity. Fix demand, valuation, scope and geography outside the optimiser. Scale the left-hand side by the selected budget and use a dimensionless right-hand side of one. Include fixed burdens and check that each activity contributes once. Check the reference design against the exported vector, then minimise utilisation within the admissible domain to test achievability. The saved ATP exceedance establishes an exceedance for that characterised reference plan. Whether the cap is feasible depends on the attainable designs and the completeness of the impact model.

## Reproduce

Run from the repository root with the project’s Python dependencies installed. The reference export used CPython 3.11.17, pandas 3.1.0 and openpyxl 3.1.5.

```bash
python -B 2b_PB_Allocation/tools/build_ability_to_pay_weights.py
python -B 2b_PB_Allocation/tools/finalise_allocation.py
python -B -m unittest discover -s 2b_PB_Allocation/tests -v
git diff --exit-code -- 2b_PB_Allocation/data/ability_to_pay_weights.json 2b_PB_Allocation/results_allocation_final
```

The last command checks a clean checkout for byte-for-byte regeneration with the reference environment. Newer CPython releases use a different floating-point summation algorithm and can change the least significant digits of the exported weights and budgets. The optional raw EXIOBASE rebuilds and their dependency paths are in [the README](README.md).
