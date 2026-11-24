# Status quo PB-constrained joint MILP

Run from the repository root:

```sh
python 4_Joint_MILP/with_PB/jointMILP_PBconstrained.py
```

Install the dependencies in `4_Joint_MILP/without_LCA/requirements.txt` and make GAMS/CPLEX available. Supply the matching local route-factor workbook at `2_LCA/outputs/master_lca_route_factor_database_ecoinvent_3_12.xlsx` or set `IS_LCA_ROUTE_FACTOR_FILE` to its path.

The model minimises the joint cost objective, including the existing shortage penalty, subject to nine annual impact limits from D23:D31 of `4_Joint_MILP/without_LCA/Planetary_Boundary_impacts.xlsx`. The terminal summary reports impacts, limits, cost and unmet demand. Use `--output-dir PATH` to retain solver files and full reports.

Other allocation methods are documented in [2b_PB_Allocation](../../2b_PB_Allocation/README.md).
