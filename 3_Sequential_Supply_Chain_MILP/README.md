# 3. Old sequential supply chain + MILP

This section retains the original two-stage workflow:

1. `mAb_SupplyChain_model/mAb_SupplyChain_v1.py` selects a manufacturing/supply-chain plan.
2. `MILP/code/run_master_MILP.py` reads that allocation and optimises industrial-symbiosis decisions separately.

The source formulation is retained. Shared SuperPro inputs are in section 1, route factors and LCA settings in section 2, and optimisation parameters here in `parameters/`. The route-eligibility workbook is in `MILP/inputs/`.

From the repository root, after the LCA route-factor workbook is available:

```sh
IS_SUPPLY_SOLVER=highs .venv/bin/python -B 3_Sequential_Supply_Chain_MILP/mAb_SupplyChain_model/mAb_SupplyChain_v1.py

IS_SUPPLY_CHAIN_FILE="$PWD/3_Sequential_Supply_Chain_MILP/mAb_SupplyChain_model/results/mab_supply_chain_superpro_v2_results.csv" \
IS_MILP_BACKEND=scipy .venv/bin/python -B 3_Sequential_Supply_Chain_MILP/MILP/code/run_master_MILP.py
```

The first command writes a fresh allocation under the supply-chain model's `results/` folder. The second explicitly uses that fresh allocation. Without `IS_SUPPLY_CHAIN_FILE`, the IS model uses the retained supplied canonical CSV in `mAb_SupplyChain_model/`. These are distinct input choices; compare a fresh allocation before replacing a reporting baseline.

`MILP/code/` also retains routing comparisons, scenarios, Pareto analysis, OAT, stochastic/QMC assessment, diagnostic replays and figure-data export. Their parameters and result subdirectories stay within this section. Some analyses require prior generated results and substantial solve time. [MILP details](MILP/README.md) describe these entry points.

This is the legacy sequential workflow. Joint variants are grouped separately in [section 4](../4_Joint_MILP/README.md).
