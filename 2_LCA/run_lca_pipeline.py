from __future__ import annotations

import argparse
from pathlib import Path

from lca_pipeline.pipeline import run_pipeline, validate_inputs


def main():
    parser = argparse.ArgumentParser(description="Run the processed-input Brightway LCA pipeline.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="PATH=VALUE")
    parser.add_argument("--validate-inputs", action="store_true")
    parser.add_argument("--superpro-workbook", type=Path)
    parser.add_argument("--manufacturing-workbook", type=Path)
    args = parser.parse_args()
    overrides = list(args.overrides)
    if args.superpro_workbook:
        overrides.append(f"inputs.superpro_workbook={args.superpro_workbook.resolve()}")
    if args.manufacturing_workbook:
        overrides.append(f"inputs.manufacturing_workbook={args.manufacturing_workbook.resolve()}")

    if args.validate_inputs:
        summary = validate_inputs(args.project_root, args.settings, overrides)
        print("Input validation completed")
        print(f'Source SUT polymer inventory: {summary["source_polymer_kg_per_batch"]:,.3f} kg per batch')
        print(f'After tubing correction: {summary["polymer_kg_per_batch_after_tubing_adjustment"]:,.3f} kg per batch')
        print(f'SUT polymer inventory: {summary["total_polymer_kg_per_batch"]:,.3f} kg per batch')
        print(f'Tubing-driven change: {summary["tubing_adjustment_delta_kg_per_batch"]:,.3f} kg per batch')
        print(f'Defect-driven change: {summary["defect_uplift_delta_kg_per_batch"]:,.3f} kg per batch')
        print('Additional packaging uplift: 0.000%')
        print(f'Unmapped SUT rows: {summary["sut_rows_unmapped"]}')
        print(f'Residual biomass: {summary["biomass_kg_per_batch"]:,.2f} kg per batch')
        return

    run_pipeline(
        args.project_root,
        settings_path=args.settings,
        overrides=overrides,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
