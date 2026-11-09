# FYR: SuperPro, LCA and supply-chain optimisation

The project has four main sections, shown in the same order on GitHub and in VS Code.

```text
1_SuperPro_Results/
  rawSP_reports/          Original SuperPro reports and process file
  notebooks/             Extraction notebook source, with saved outputs cleared
  data/                  Processed results, intermediate tables and templates
2_LCA/
  lca_pipeline/          Life-cycle inventory and impact calculations
  parameters/            LCA settings
  outputs/               Local generated route factors (not uploaded)
3_Sequential_Supply_Chain_MILP/
  mAb_SupplyChain_model/  Old first-stage manufacturing/supply-chain model
  MILP/                  Old second-stage IS model and numerical analyses
  parameters/            Shared economic, eligibility and scenario settings
4_Joint_MILP/
  without_LCA/           Alfie's original economic-only joint model
  with_LCA/              Ahmed's joint model with carbon valuation and LCA reporting
  with_PB/               Reserved for a joint planetary-boundary model
```

| Section | Start here |
| --- | --- |
| 1. SuperPro results | [Reports, extraction and data vintages](1_SuperPro_Results/README.md) |
| 2. LCA | [Validate inputs and generate route factors](2_LCA/README.md) |
| 3. Old sequential supply chain + MILP | [Run the original two-stage workflow](3_Sequential_Supply_Chain_MILP/README.md) |
| 4. Joint MILP | [Choose a joint-model variant](4_Joint_MILP/README.md) |

## Setup and VS Code

Open `FYR_code.code-workspace` for the four main Explorer sections. Create one Python environment at the repository root and share it across the workflows:

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Python 3.11 or newer is recommended. HiGHS is used for Ahmed's joint model; Alfie's original solver interface uses GAMS/CPLEX. Each section documents its commands. Run the examples from the repository root.

## Shared inputs and model scope

Foreground manufacturing inputs live once in section 1, LCA generation and settings in section 2, and common optimisation inputs in section 3. `project_paths.py` connects those locations. Each workflow writes its own local results directory.

The joint variants currently have different formulations. Ahmed's version fixes sites and assigned-line capacity after an upstream solve; Alfie's version decides sites and batches jointly. Differences between their results cannot be attributed solely to LCA. The `with_PB` folder is a planned variant, with no solver or PB budgets implemented yet.

Alfie's `testjoint.py` is preserved byte for byte. Its new `run_without_lca.py` launcher adapts paths to shared inputs. The missing `itemised_costs_20261026.csv` is still required for that variant; older dated inputs are kept under section 1 without substituting one automatically.

Generated model outputs, environments and ecoinvent-derived route-factor workbooks remain excluded from Git. Supply or generate a permitted workbook at `2_LCA/outputs/master_lca_route_factor_database_ecoinvent_3_12.xlsx`, or set `IS_LCA_ROUTE_FACTOR_FILE` to a local path. The economic-only model still uses physical inventory from this workbook, although it does not optimise LCA impacts.

[SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) records source provenance and packaged file hashes. The original project outside this checkout is preserved.
