# 4. Joint MILP

| Variant | Status | Entry point |
| --- | --- | --- |
| [Without LCA — Alfie](without_LCA/README.md) | Economic-only formulation; exact October 26 itemised-cost input still missing | `without_LCA/run_without_lca.py` |
| [With LCA — Ahmed](with_LCA/README.md) | Existing model with carbon valuation and environmental reporting | `with_LCA/run_joint_milp.py` |
| [With PB](with_PB/README.md) | Folder reserved; PB constraints and budgets are not implemented | No runner yet |

All variants share the foreground data in section 1, LCA workbook in section 2 and optimisation inputs in section 3. New variants can be added here without copying those shared inputs.

These are separately documented formulations. Ahmed's existing version fixes site/assigned-line capacity after an upstream solve, whereas Alfie's code decides sites and batches jointly. The names describe how environmental information is used; they do not imply an otherwise identical controlled comparison.
