# Upstream mAb supply-chain model

`mAb_SupplyChain_v1.py` solves the supplied deterministic supply-chain formulation. Its embedded SuperPro coefficients and assumptions are unchanged. Unused Matplotlib setup was removed.

The existing `mab_supply_chain_superpro_v2_results.csv` is a supplied input to the downstream MILP. A fresh solve writes to `results/mab_supply_chain_superpro_v2_results.csv`, leaving that supplied CSV untouched. Compare the two files before choosing a new downstream allocation.

From the `3_Sequential_Supply_Chain_MILP` directory:

```bash
python mAb_SupplyChain_model/mAb_SupplyChain_v1.py
```

A fresh run uses the existing Pyomo formulation. It selects GAMS/CPLEX when that interface is available, otherwise HiGHS through `highspy`. The HiGHS solve uses a relative MIP gap of `1e-9` and reproduces the supplied 328/328/205 batch allocation. Set `IS_SUPPLY_SOLVER=gams:cplex` or `IS_SUPPLY_SOLVER=highs` to require one explicitly.
