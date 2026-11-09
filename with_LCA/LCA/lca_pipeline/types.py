from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class PipelineResult:
    output_workbook: Path | None
    tables: dict[str, Any]
    settings: dict[str, Any]
