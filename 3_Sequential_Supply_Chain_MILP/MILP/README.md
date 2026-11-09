# Industrial-symbiosis MILP

`code/run_master_MILP.py` loads the supplied supply-chain CSV, SuperPro workbook, model parameters, an LCA route-factor workbook, and a polymer-route eligibility workbook. It writes case summaries, decisions, solver reports, and an Excel result workbook to `results/`. The mathematical model is implemented in Python and solved with SciPy HiGHS or GAMS/CPLEX.

Default inputs are:

| Input | Path |
| --- | --- |
| Supply-chain allocation | `../mAb_SupplyChain_model/mab_supply_chain_superpro_v2_results.csv` |
| SuperPro inventory | `../../1_SuperPro_Results/data/processed/SP_master_dataset.xlsx` |
| Central parameters | `../parameters/MILP_central_params.json` |
| LCA factors | `../../2_LCA/outputs/master_lca_route_factor_database_ecoinvent_3_12.xlsx` |
| Route eligibility | `inputs/polymer_route_eligibility_20260722.xlsx` |

The original eligibility workbook is present at the path above. It was copied unchanged from the thesis MILP source folder. The LCA workbook must be produced by the LCA pipeline or selected explicitly with `IS_LCA_ROUTE_FACTOR_FILE`.

After the inputs exist, run from the `3_Sequential_Supply_Chain_MILP` directory:

```bash
IS_MILP_BACKEND=scipy python MILP/code/run_master_MILP.py
IS_MILP_BACKEND=scipy python MILP/code/routing_comparison.py
IS_MILP_BACKEND=scipy python MILP/code/run_scenario_MILP.py
```

Other data-only analyses are `code/run_epsilon_constraint_pareto.py`, `code/run_oat_decision_boundaries.py`, `code/run_stochastic_optimisation.py`, and the QMC diagnostic scripts. The scenario runner uses `../parameters/MILP_scenarios.json`. The stochastic runner uses `../parameters/MILP_stochastic_optimisation_parameters.json`. OAT reads `../parameters/MILP_OAT_sensitivity_analysis_parameters.json`, which is present.

The hypothetical STRAP scenario uses its supplied GWP proxy. Its DALY and ecosystem-quality results are left blank because no STRAP factors for those categories were supplied. Scenarios whose constraints are infeasible are recorded as such; unexpected execution errors cause the runner to fail after writing its audit tables.

The scripts use the scientific assumptions in the supplied JSONs; they do not revise them. `run_sobol_infeasibility_diagnostics.py` replays a completed stochastic run and requires its `results/stochastic_optimisation_solution_space/run_plan.json` and `sobol_qmc_design.csv`; those reference results are not supplied. Some analyses can require hundreds or thousands of MILP solves. The core runner checks inputs before opening its results directory. On reruns, it retains existing result tables until a successful solve and archives earlier solver logs and reports under `results/ecosystem_master_20261015/previous_solver_runs/`. Review solver status and diagnostic tables before interpreting results.
