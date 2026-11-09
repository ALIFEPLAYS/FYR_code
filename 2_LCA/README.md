# 2. LCA

`run_lca_pipeline.py` reads the shared processed SuperPro inputs in section 1, builds the SUT inventory, and scores plastics, wastewater, manufacturing utilities and biomass. Settings live in `parameters/lca_parameters.json` and point to the shared input workbooks by relative paths.

From the repository root:

```sh
# Check foreground workbook structure and inventory processing:
.venv/bin/python -B 2_LCA/run_lca_pipeline.py --validate-inputs

# Generate route factors with a configured, permitted Brightway database:
.venv/bin/python -B 2_LCA/run_lca_pipeline.py
```

The full calculation requires the configured local `bw_ecoinvent_3_12_cutoff` Brightway project, ecoinvent 3.12 cut-off databases and LCIA methods. See the parameter JSON for scientific settings and source references. The default output is `outputs/master_lca_route_factor_database_ecoinvent_3_12.xlsx`, with `route_factor_matrix`, `input_output_table` and `lca_stage_breakdown` sheets. It remains local and is ignored by Git.

The functional unit is one 1,000 L single-use mAb batch. Positive factors are burdens and negative values are credits. The stage breakdown must not be added to net factors again. The inherited foreground validation identifies three unmapped SUT rows; complete inventory coverage is not established.

Use `--settings`, `--superpro-workbook`, `--manufacturing-workbook` or `--set dotted.path=value` for explicit input changes. The sequential and joint models read this output by default; `IS_LCA_ROUTE_FACTOR_FILE` selects another permitted local workbook.
