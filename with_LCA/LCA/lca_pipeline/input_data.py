from __future__ import annotations

from pathlib import Path

import openpyxl
import pandas as pd

from .config import resolve_project_path


SUPERPRO_REQUIRED_COLUMNS = [
    "inventory_id",
    "source_report",
    "source_table",
    "broad_process_area",
    "superpro_section",
    "unit_operation",
    "source_procedure",
    "destination",
    "flow_name_superpro",
    "component",
    "amount_per_batch",
    "unit_per_batch",
    "amount_per_year",
    "unit_per_year",
    "stream_volume_L_per_batch",
    "stream_volume_L_per_year",
    "stream_component_sum_kg_per_batch",
    "stream_total_minus_component_sum_kg_per_batch",
    "source_row",
    "source_column",
]


def _read_nonblank_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        if sheet_name not in workbook.sheetnames:
            raise ValueError(
                f"Sheet {sheet_name!r} was not found in {path.name}. "
                f"Available sheets: {workbook.sheetnames}"
            )
        rows = [
            row
            for row in workbook[sheet_name].iter_rows(values_only=True)
            if any(value is not None for value in row)
        ]
    finally:
        workbook.close()

    if len(rows) < 2:
        raise ValueError(f"Sheet {sheet_name!r} in {path.name} contains no data rows.")
    return pd.DataFrame(rows[1:], columns=rows[0])


def _require_columns(frame: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def load_input_data(project_root: Path, settings: dict) -> dict[str, pd.DataFrame | Path]:
    inputs = settings["inputs"]
    superpro_path = resolve_project_path(project_root, inputs["superpro_workbook"])
    manufacturing_path = resolve_project_path(project_root, inputs["manufacturing_workbook"])

    if not superpro_path.exists():
        raise FileNotFoundError(superpro_path)
    if not manufacturing_path.exists():
        raise FileNotFoundError(manufacturing_path)

    superpro_inventory = _read_nonblank_sheet(
        superpro_path,
        inputs["superpro_inventory_sheet"],
    )
    _require_columns(superpro_inventory, SUPERPRO_REQUIRED_COLUMNS, "SuperPro inventory")

    sut_overall = pd.read_excel(
        superpro_path,
        sheet_name=inputs["sut_overall_sheet"],
    )
    sut_composition = pd.read_excel(
        superpro_path,
        sheet_name=inputs["sut_composition_sheet"],
        header=1,
    )

    manufacturing_inputs = pd.read_excel(
        manufacturing_path,
        sheet_name=inputs["manufacturing_inputs_sheet"],
    )
    itemised_costs = pd.read_excel(
        manufacturing_path,
        sheet_name=inputs["itemised_costs_sheet"],
    )

    _require_columns(
        manufacturing_inputs,
        ["inventory_id", "record_type", "component", "amount_per_batch"],
        "manufacturing_inputs",
    )
    _require_columns(
        itemised_costs,
        ["cost_id", "cost_group", "item_name", "amount_per_batch", "cost_per_batch_usd"],
        "itemised_costs",
    )

    return {
        "superpro_path": superpro_path,
        "manufacturing_path": manufacturing_path,
        "superpro_inventory": superpro_inventory,
        "sut_overall": sut_overall,
        "sut_composition": sut_composition,
        "manufacturing_inputs": manufacturing_inputs,
        "itemised_costs": itemised_costs,
    }
