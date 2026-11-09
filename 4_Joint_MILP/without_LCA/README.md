# Joint MILP without LCA — Alfie

`testjoint.py` is Alfie's original economic-only model, preserved byte for byte from GitHub. `run_without_lca.py` is the entry point for this layout: it redirects file locations to shared inputs before calling the original model. No equations or solver choices are changed.

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

Building or solving requires `1_SuperPro_Results/data/processed/itemised_costs_20261026.csv`, which was absent from the supplied files. Obtain that exact input before running; another vintage is not substituted automatically. The shared route workbook is also required, either at its default section-2 location or via `IS_LCA_ROUTE_FACTOR_FILE`.

Solving uses GAMS with CPLEX by default. Set `IS_GAMS_SOLVER` or pass `--gams-executable` if needed. Local results default to `4_Joint_MILP/without_LCA/results/`; `--output-dir` selects another location. Run the launcher rather than executing the original source directly, because its original internal paths describe the previous folder layout.
