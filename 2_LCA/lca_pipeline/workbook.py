from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill


HEADER_FILL = PatternFill("solid", fgColor="1F4E78")
HEADER_FONT = Font(color="FFFFFF", bold=True)


def _format_sheet(worksheet) -> None:
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions
    worksheet.sheet_view.showGridLines = False
    worksheet.row_dimensions[1].height = 34

    for cell in worksheet[1]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(wrap_text=True, vertical="center")

    for column_cells in worksheet.columns:
        values = [str(cell.value) for cell in list(column_cells)[:250] if cell.value is not None]
        width = min(max([len(value) for value in values] + [10]) + 2, 42)
        worksheet.column_dimensions[column_cells[0].column_letter].width = width


def write_model_workbook(
    route_factor_matrix: pd.DataFrame,
    input_output_table: pd.DataFrame,
    path: Path,
    lca_stage_breakdown: pd.DataFrame | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        route_factor_matrix.to_excel(writer, sheet_name="route_factor_matrix", index=False)
        input_output_table.to_excel(writer, sheet_name="input_output_table", index=False)
        if lca_stage_breakdown is not None:
            lca_stage_breakdown.to_excel(
                writer,
                sheet_name="lca_stage_breakdown",
                index=False,
            )
        _format_sheet(writer.book["route_factor_matrix"])
        _format_sheet(writer.book["input_output_table"])
        if lca_stage_breakdown is not None:
            _format_sheet(writer.book["lca_stage_breakdown"])

    return path
