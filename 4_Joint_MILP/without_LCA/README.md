# Joint MILP without LCA — Alfie

The launcher uses the shared project inputs and the joint cost formulation in `testjoint.py`. In the combined FYR checkout it also loads the annual PB expressions in `07_PB/lcia/joint_pb.py`. For a separate installation, set `IS_PB_LCIA_DIR` to that directory. A standalone collaboration checkout can still run the economic model.

The model decides sites, batches and IS choices jointly. Environmental impact factors and carbon valuation do not enter its objective. It still reads physical inventory quantities and recovery yields from the route workbook in section 2.

From the repository root:

```sh
# Options work without loading input data:
.venv/bin/python -B 4_Joint_MILP/without_LCA/run_without_lca.py --help

# Build and export the model without invoking GAMS:
.venv/bin/python -B 4_Joint_MILP/without_LCA/run_without_lca.py --export-only

# Export, solve and report:
.venv/bin/python -B 4_Joint_MILP/without_LCA/run_without_lca.py
```

The launcher reads `1_SuperPro_Results/data/processed/itemised_costs_regenerated_20261123.csv` for the embedded electricity deduction. The shared route workbook is also required, either at its default section-2 location or via `IS_LCA_ROUTE_FACTOR_FILE`.

Solving uses GAMS with CPLEX by default. Set `IS_GAMS_SOLVER` or pass `--gams-executable` if needed. Local results default to `4_Joint_MILP/without_LCA/results/`; `--output-dir` selects another location. Run the launcher rather than executing the original source directly, because its original internal paths describe the previous folder layout.


PB-enabled runs write `joint_pb_impacts.csv` and vehicle-km for every site, route and mode in
`joint_waste_transport.csv`. Use `--pb-caps budgets.json` for optional annual limits, with
exact category names and native units from the PB workbook. Each cap is scaled by its budget.
For capped CPLEX solves the export supplies `epint` of 1e-9 and `eprhs` of 1e-8 through `cplex.o81`
in the run directory. The existing feasibility, objective reconciliation and gap checks still run. The
large shortage-penalty coefficient makes very close cost rankings sensitive to numerical tolerances. Use a separate output directory for sensitivity runs.
