# Joint MILP with LCA — Ahmed

This variant jointly optimises production and IS choices within site and assigned-line capacities fixed from an upstream optimum. Its objective combines economic cost with life-cycle carbon valuation, and it reports environmental results. It currently has no planetary-boundary caps.

The upstream and IS implementations are shared from section 3, foreground inputs from section 1, and route factors from section 2. Mathematical equations and scientific parameters are unchanged by the folder reorganisation; only file locations were adapted.

From the repository root, with the shared environment and permitted route-factor workbook available:

```sh
.venv/bin/python -B 4_Joint_MILP/with_LCA/run_joint_milp.py
.venv/bin/python -B 4_Joint_MILP/with_LCA/check_allocations.py
```

Run the checker after a successful solve with the same inputs. Results are written to `4_Joint_MILP/with_LCA/results/` and stay local. `IS_LCA_ROUTE_FACTOR_FILE` can select another permitted route workbook.

[Model details](MODEL_DETAILS.md) cover the equations and accounting. The [results manual](RESULTS_MANUAL.md) describes the previously verified local results; its linked generated files appear after a successful run.
