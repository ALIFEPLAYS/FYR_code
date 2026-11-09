# Without LCA — Alfie's economic-only joint MILP

[MILP/code/testjoint.py](MILP/code/testjoint.py) is Alfie's original GitHub code, moved without changing its contents. It decides manufacturing sites, batches and industrial-symbiosis choices jointly and minimises annual economic cost. Environmental impact factors and carbon valuation do not enter its objective.

The `MILP/code/` location matches the relative paths already expected by the original script. The central parameter JSON and route-eligibility workbook are supplied alongside it.

## Setup

From this `without_LCA` directory, with Python 3.11 or newer:

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Solving uses a local GAMS installation with CPLEX by default. `IS_GAMS_SOLVER` selects the GAMS solver, and `--gams-executable` accepts the executable path.

## Required local inputs

The original script requires two files that are not bundled here:

| Input | Expected location |
| --- | --- |
| Route workbook, used for physical per-batch quantities and recovery yields | `LCA/outputs/master_lca_route_factor_database_ecoinvent_3_12.xlsx` |
| Itemised manufacturing costs from Alfie's input vintage | `SuperPro_Reports_Data_Extraction/data/processed/itemised_costs_20261026.csv` |

`IS_LCA_ROUTE_FACTOR_FILE` can point to a permitted route workbook elsewhere on your machine. Although the objective has no LCA impacts, this implementation still reads physical inventory from an LCA-generated workbook. The [with-LCA section](../with_LCA/README.md) contains its generation pipeline. Do not upload licensed route-factor workbooks.

The exact `itemised_costs_20261026.csv` was absent from the supplied GitHub repository and Ahmed's local project at packaging time. Obtain that input from Alfie and place it at the path above before building or solving. Another dated cost file is not silently substituted.

## Run

```sh
# Inspect options without loading inputs:
.venv/bin/python -B MILP/code/testjoint.py --help

# Requires the local inputs above; exports without calling a solver:
.venv/bin/python -B MILP/code/testjoint.py --export-only

# Export, solve with GAMS, and write reports:
.venv/bin/python -B MILP/code/testjoint.py
```

Results default to `MILP/results/testjoint/joint_cost/`. Use `--output-dir` to choose another directory. See the original module documentation and code for formulation details.
