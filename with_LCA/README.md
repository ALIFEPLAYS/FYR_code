# With LCA — Ahmed's joint MILP

This section contains the existing joint model, its allocation checker, the upstream supply-chain model, the IS model, the LCA pipeline and the foreground inputs those workflows require. Model code and supplied inputs are unchanged from Ahmed's local project. Generated results are not bundled.

The annual objective combines manufacturing and supply-chain cost, removes the overlapping manufacturing electricity charge, adds net IS cost and applies life-cycle carbon valuation. [Joint model details](Joint_MILP/README.md) and the [results manual](Joint_MILP/RESULTS_MANUAL.md) describe the formulation and the previously obtained local results. Rerun to generate the result files referenced by that manual.

## Setup

From this `with_LCA` directory, with Python 3.11 or newer:

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The optimiser uses HiGHS. A full LCA generation also requires your own permitted ecoinvent 3.12 cut-off database, Brightway project and configured LCIA methods. See [LCA/README.md](LCA/README.md) and [parameters/lca_parameters.json](parameters/lca_parameters.json).

## Route-factor input

The default required workbook is:

```text
LCA/outputs/master_lca_route_factor_database_ecoinvent_3_12.xlsx
```

It is deliberately excluded from Git. To use an existing permitted local workbook, set `IS_LCA_ROUTE_FACTOR_FILE` to its absolute path before running the joint model and checker. Otherwise validate the supplied foreground inputs and generate the workbook:

```sh
.venv/bin/python -B LCA/run_lca_pipeline.py --validate-inputs \
  --superpro-workbook SuperPro_Reports_Data_Extraction/data/processed/SP_master_dataset.xlsx \
  --manufacturing-workbook SuperPro_Reports_Data_Extraction/data/processed/manufacturing_footprint_inputs_and_costs_20260703.xlsx

.venv/bin/python -B LCA/run_lca_pipeline.py \
  --superpro-workbook SuperPro_Reports_Data_Extraction/data/processed/SP_master_dataset.xlsx \
  --manufacturing-workbook SuperPro_Reports_Data_Extraction/data/processed/manufacturing_footprint_inputs_and_costs_20260703.xlsx
```

## Run the model

```sh
.venv/bin/python -B Joint_MILP/run_joint_milp.py
.venv/bin/python -B Joint_MILP/check_allocations.py
```

Outputs are written under `Joint_MILP/results/` and remain local. The allocation checker must follow a successful joint-model run with the same inputs. Its enumeration can take longer than the joint solve.

## Interpretation

This model fixes sites and product-assigned lines at an upstream optimum before jointly optimising production and IS choices. Alfie's version optimises site opening and production together. Treat them as separately documented formulations, rather than a controlled experiment in which only carbon valuation changes.
