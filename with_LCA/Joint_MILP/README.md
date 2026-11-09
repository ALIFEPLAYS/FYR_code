# Joint manufacturing and industrial symbiosis MILP

[Read the results manual](RESULTS_MANUAL.md) for the two-stage comparison, electricity cost correction and changes to the constraints.

Run from the project root with its virtual environment and existing input workbooks, including the local ecoinvent 3.12 route-factor workbook.

```sh
.venv/bin/python -B Joint_MILP/run_joint_milp.py
.venv/bin/python -B Joint_MILP/check_allocations.py
```

The first command solves the original two stages and the fully integrated model in memory. It writes comparison, site, manufacturing line, polymer, route and validation files under `Joint_MILP/results`. The second command independently enumerates the annual batch allocations and writes `allocation_check.json`. Run both commands when inputs change.

## Construction

The run starts with the original supply-chain Pyomo model and solves its original objective. It fixes the selected sites, scales and product-assigned lines at their optimum. Enabled lines in its exported CSV can include idle equipment. Annual production, line time and product shipments remain decisions. Both cases deliver 861 batches and meet every demand-zone minimum with zero shortfall.

The original IS objective and matrix are rebuilt from the central input files. The only IS constraints replaced are the one-low-site rule and its three high/low batch equalities. The model checks those rows by their variables, coefficients and limits. New equality constraints link each site's IS batch count to its completed fill-and-finish batches. Total batches, polymer balances and eligibility, vehicle payloads and integer journeys, biomass, electricity, water and treatment constraints come from the original IS matrix. The original production, inventory, activation and distribution equations remain in the solve.

The central IS inputs have no measured recycler capacity. For each shared mechanical recycler without a supplied limit, its benchmark annual intake is used as the maximum network intake. The benchmark is replayed under those identical limits. These ceilings represent no increase above modelled baseline throughput. They do not establish the facilities' commercial or physical capacities. The site-processing limits of 328, 328 and 242 batches are verified by maximising throughput at each site with its assigned lines fixed.

## Annual objective and accounting

Minimise **supply-chain cost in 2026 GBP − embedded manufacturing power charge + net IS operating and investment cost + carbon appraisal value**. There are no PB caps. The carbon valuation uses the unrounded central model coefficient of £328.714056/tCO₂e. Cash costs are reported separately and include annualised capital.

* The original supply-chain objective covers production, capital, site charges, handling and both product-freight legs. The project convention converts 2019 USD into 2026 GBP at 1.022785. The same price basis is assumed for the illustrative upstream freight, storage and site rates.
* The original SuperPro section costs include purchased power and exclude waste treatment/disposal in the upstream variable coefficients. The routine sums the detailed 2019 USD `Std Power` cost in the manufacturing workbook and checks the kWh against IS demand. That £6.454m annual overlap is removed before adding the IS electricity bill. Other process utilities and WFI remain in upstream costs. IS prices the non-contact water and wastewater services separately.
* The original IS equations include revenue on saleable recovered polymer, waste transport and treatment, biomass, water and purchased electricity. Net route LCA coefficients already include avoided virgin production and applicable energy or material substitution credits. Each coefficient enters the carbon calculation once. Manufacturing background is included once. The route breakdown reports the embedded credits for interpretation.
* Product freight contributes to cost on the original illustrative distances. The supplied product shipment model has no emissions factor for distribution or cold storage. The carbon result covers the central IS life-cycle inventory, including waste transport. It is a modelled footprint, not a full delivered-product footprint.

## Benchmark and validation

The benchmark reruns Cian's two-stage *code* with the current workbooks and price parameters, including the historical freedom to swap the low site in stage two. Under these inputs the two stages agree on Slough. The benchmark is its optimised central case, not the landfill/grid business-as-usual case. The run requires the two stages' site totals to agree before replay. The current workbook and the September paper use different polymer inventories. The paper reports about 996 t of annual plastics, £11.113m IS cost and 3.531 ktCO₂e for its central optimum. The current code benchmark has about 752 t, £11.502m and 4.630 kt. Changes between those input vintages are excluded from the measured integration saving.

Validation checks optimal termination, constraint residuals, variable bounds, integer decisions, common site totals, product deliveries, zero shortfall, objective reconciliation and the solver's global bound. The fixed two-stage plan must reproduce its separately computed objective to the penny in the joint equations. `is_breakdown.csv` independently reconciles the IS cost, GWP and water components. The validation JSON identifies the inputs by SHA-256.

The second script uses the upstream model to establish each site's maximum. For every possible batch split it solves the original IS matrix with that split imposed. With fixed production, assigned lines, customer demand and one storage node, only factory-to-storage freight varies on the upstream side. The checker recovers the per-batch freight rates from the original upstream solve and compares the exhaustive minimum with the full joint solution. These are numerical checks of the model, not evidence of actual plant or recycler performance.
