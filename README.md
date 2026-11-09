# Joint MILP: with and without LCA

Two versions are kept in separate, self-contained project layouts:

| Section | Model | Entry point |
| --- | --- | --- |
| [With LCA — Ahmed's version](with_LCA/README.md) | Joint manufacturing and industrial-symbiosis optimisation with economic cost plus life-cycle carbon valuation, and environmental reporting | `with_LCA/Joint_MILP/run_joint_milp.py` |
| [Without LCA — Alfie's version](without_LCA/README.md) | Alfie's original economic-only joint MILP; environmental impacts do not enter the objective | `without_LCA/MILP/code/testjoint.py` |

Alfie's original `testjoint.py` has been moved unchanged into the second section. Ahmed's model and its supporting source and foreground inputs have been copied unchanged into the first. [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) records source commits and file hashes.

These versions have different formulations: Ahmed's run fixes site and assigned-line capacity after an upstream solve, while Alfie's code decides sites and batches jointly. Their output differences therefore cannot be attributed solely to including LCA. See each section's README for setup and input requirements.

## Open in VS Code

Open `FYR_code.code-workspace` to see both sections side by side. Each section has its own requirements and results directories. Run commands from the section directory specified in its README.

## Data boundary

Generated results, Python environments and licensed ecoinvent-derived route-factor workbooks are excluded. Both versions currently read that workbook: the economic-only version uses physical inventory quantities rather than impact factors. Supply your own permitted workbook or generate it through the included LCA pipeline. Neither a GitHub download nor this repository supplies an ecoinvent licence.
