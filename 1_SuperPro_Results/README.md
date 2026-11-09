# 1. SuperPro results

This section holds the manufacturing foreground data shared by LCA and both optimisation workflows.

| Folder | Contents |
| --- | --- |
| `rawSP_reports/` | Original SuperPro reports and the supplied process file |
| `notebooks/` | SR/ICR and manufacturing-cost extraction notebooks |
| `data/processed/` | Reviewed master workbook, manufacturing workbooks and dated extraction results |
| `data/interim/` | Extraction tables and inventory-review workbooks |
| `data/templates/` | Supplied mapping and optimisation templates |

The current LCA and Ahmed joint-model inputs are `data/processed/SP_master_dataset.xlsx` and `data/processed/manufacturing_footprint_inputs_and_costs_20260703.xlsx`. Later dated outputs are retained for provenance; moving the folders does not select a new input vintage.

The notebooks use their working directory to locate reports. Run them with the notebook directory or this section as the working directory. Uploaded notebook source comes from the original project's committed revision, with execution counts and saved outputs cleared. Local uncommitted notebooks in the original project have not been changed.

Alfie's economic-only joint model requests `data/processed/itemised_costs_20261026.csv`. That exact file was not available when this repository was organised; the supplied July and October files are not silently substituted.
