# 2b. Planetary-boundary allocation (total GVA and ability to pay)

This folder turns the global planetary-boundary (PB) safe operating space into **annual budgets for our UK mAb network**. It does this in two ways:

| Method | Role | Basis | Climate utilisation (reference and expanded scenario extrema) | Audit verdict |
|---|---|---|---:|---|
| `total_gva` | **Primary** | Hartmann et al. (2026), total gross value added, Eq. 7, recomputed to fix a PBAESA code error | **64.90%** (39.54–108.22%) | Exploratory fixed cap |
| `ability_to_pay` | **Alternative** | Hjalsted et al. (2021) Eq. 5, applied per country, inside the same Hartmann framework | **153.46%** (70.38–1,773.20%) | Producer-region ethical scenario |
| `total_fce`, `total_fce_corrected`, `total_fce_corrected_f` | Supporting | Hartmann total final consumption expenditure | 15.0–23.2% historical variants | Unresolved supporting comparison |

No other PB indicator is exceeded under either method. The impacts used are an incomplete characterised subtotal, so "within budget" is conditional. Every central number has been recomputed independently from raw sources (`tools/audit_from_raw.py`; METHODOLOGY.md §12). See [finalisation and interpretation](FINALISATION.md), [the methodology and Claude Code audit](METHODOLOGY.md), and [the separate Astra audit](audit/INDEPENDENT_AUDIT_ASTRA.md).

## Use in the joint MILP

```python
import sys; sys.path.insert(0, "2b_PB_Allocation")
import pb_allocation as pa

budgets = pa.budget_vector("total_gva")        # or "ability_to_pay"
# {"Climate Change": 1.668e-06, "Ocean Acidification": 1.148e-06, ...}  native PBAESA units for annual activity
```

The same numbers are in [`results_allocation_final/milp_budgets.json`](results_allocation_final/milp_budgets.json). Rules:

- Treat budgets as **fixed parameters**. Choose the method once per run; never let the solver choose it.
- Switching method **only rescales** all nine budgets: ability to pay = 0.4229 × total GVA. ε-constraint and minimax designs therefore coincide up to relabelling; At the reference plan, none of the modelled indicators exceeds the Total GVA budget.
- Divide each PB row by its budget (r_c = I_c/B_c); native budgets are about 10⁻⁶.
- Sensitivity variants (value basis × input origin) are scalar SOS fractions under `sensitivity_sos_fractions`.

The PB optimisation formulations themselves are **not** implemented here; see `4_Joint_MILP/with_PB/`.

## Reproduce (from the repository root)

```bash
.venv/bin/python -B 2b_PB_Allocation/tools/finalise_allocation.py
```

```bash
.venv/bin/python -B -m unittest discover -s 2b_PB_Allocation/tests -v
```

Both run from committed files only: no EXIOBASE, Brightway or ecoinvent needed. The exporter prints its validation count. Tests compare both central budgets to the pinned raw-data audit and exercise the added scenario outputs.

Optional rebuilds of the committed inputs:

| Command | Needs | Rebuilds |
|---|---|---|
| `tools/build_ability_to_pay_weights.py` | `data/worldbank/` (committed) | `data/ability_to_pay_weights.json` |
| `tools/extract_network_cost_structure.py` | `1_SuperPro_Results/rawSP_reports/mab_fedbatch_1000_sut_ICR.xls` | `data/network_cost_structure.json` |
| `tools/verify_total_gva.py --exiobase <IOT_2022_ixi folder>` | EXIOBASE 3.9.6 2022 ixi (about 1.3 GB, not committed) | `data/total_gva_verification_2022.json` |
| `tools/audit_from_raw.py --exiobase <IOT_2022_ixi folder>` | EXIOBASE, plus the files above | `data/audit_from_raw_2022.json`, `data/exiobase_audit_terms_2022.json` |

## Contents

| Path | What it is |
|---|---|
| `pb_allocation.py` | All allocation logic and the MILP entry point `budget_vector()` |
| `tools/` | Results exporter and the three input builders above |
| `tests/test_pb_allocation.py` | Tests cover equations on synthetic economies, the vendored PBAESA function, Hjalsted α, data mapping, budgets, raw-data audit, exports |
| `results_allocation_final/` | Final budgets, utilisation tables, sensitivity and combined envelope (`envelope_climate.csv`), regional decomposition, two SVG charts, `validation.json`, `provenance.json`; `pre_audit_20261119/` keeps the first pushed results |
| `results_final/annual_impacts.csv` | Reference annual PB impacts of the saved two-stage network (copied unchanged from Ahmed's PB assessment; hash-checked) |
| `data/` | EXIOBASE-derived regional terms, ability-to-pay weights, economic inputs, SuperPro cost structure, ONS ABS workbook, World Bank/IMF downloads (see `data/worldbank/SOURCES.txt`) |
| `data/Allocation Factors_2022.xlsx` | Hartmann et al. published allocation factors (CC BY-NC-SA, custom; see `allocation_LICENSE.txt`, non-commercial academic use) |
| `vendor/pbaesa/` | `allocation.py` and `utils.py` from PBAESA commit d932921 (MIT), kept so the tests and audit can use the original code |

## Relation to `2a_PB-LCA /`

`2a_PB-LCA /FYR_Economic_Allocation.xlsx` reproduces the earlier 14 November allocation: the published total-GVA factor, a $200/g price and network-GVA shares. This folder supersedes those inputs. The published factor contains the PBAESA error, $200/g has no source, and network-GVA shares do not match total factors. See METHODOLOGY.md §4 and §9. Nothing in `2a_PB-LCA /` was changed.
