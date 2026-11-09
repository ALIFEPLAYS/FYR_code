# LCA data pipeline

`run_lca_pipeline.py` reads the supplied processed SuperPro and manufacturing workbooks, builds the SUT polymer inventory, scores plastics, wastewater, manufacturing utilities, and biomass, and writes a route-factor workbook for the MILP. The output has `route_factor_matrix`, `input_output_table`, and `lca_stage_breakdown` worksheets. No chart code is included.

Run the two commands in the root README. The runner loads `../parameters/lca_parameters.json` by default. Use `--settings` for another file and `--set dotted.path=value` for a recorded sensitivity run. `--superpro-workbook` and `--manufacturing-workbook` override only the two workbook paths at runtime; they do not rewrite the supplied JSON. The runner stops when a required sheet, column, database, method, or activity mapping is absent.

The functional unit is one 1,000 L single-use mAb batch. The background model is the configured ecoinvent 3.12 cut-off project. The parameter JSON retains the source references and scientific settings. Positive route factors are burdens and negative values are credits. The stage breakdown must not be added to the net factors again.

`--validate-inputs` passed on the supplied workbooks during packaging. It found 873.168 kg modelled SUT polymer per batch, 12.83 kg residual biomass per batch, and three unmapped SUT rows. Those rows need scientific review before claiming complete inventory coverage.
