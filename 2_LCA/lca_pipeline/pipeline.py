from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from .biomass_lca import extract_biomass_inventory, run_biomass_lca
from .combine import combine_lca_results
from .config import apply_overrides, load_settings
from .input_data import load_input_data
from .manufacturing_lca import run_manufacturing_lca
from .plastics_lca import run_plastics_lca
from .sut_inventory import build_sut_polymer_inventory
from .types import PipelineResult
from .wastewater_lca import run_wastewater_lca
from .workbook import write_model_workbook


def _quiet_call(function, *args, **kwargs):
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        return function(*args, **kwargs)


def run_pipeline(
    project_root: Path,
    settings_path: Path | None = None,
    overrides: list[str] | None = None,
    output_path: Path | None = None,
    write_output: bool = True,
    progress=print,
) -> PipelineResult:
    root = Path(project_root).expanduser().resolve()
    settings = apply_overrides(load_settings(root, settings_path), overrides)

    progress("Started loading processed SuperPro datasets")
    inputs = _quiet_call(load_input_data, root, settings)

    progress("Started processing SUT plastics composition")
    sut_inventory, sut_validation = _quiet_call(
        build_sut_polymer_inventory,
        inputs["superpro_inventory"],
        inputs["sut_overall"],
        inputs["sut_composition"],
        settings,
    )

    progress("Started LCA for SUT plastics")
    plastics = _quiet_call(run_plastics_lca, sut_inventory, settings, root)

    progress("Started LCA for wastewater")
    wastewater = _quiet_call(run_wastewater_lca, inputs["superpro_inventory"], settings)

    progress("Started LCA for manufacturing footprint")
    manufacturing = _quiet_call(
        run_manufacturing_lca,
        inputs["manufacturing_inputs"],
        inputs["itemised_costs"],
        settings,
    )

    progress("Started LCA for residual biomass")
    biomass = _quiet_call(
        run_biomass_lca,
        inputs["superpro_inventory"],
        settings,
    )

    progress("Started final route-factor processing")
    combined = _quiet_call(
        combine_lca_results,
        plastics,
        wastewater,
        manufacturing,
        biomass,
        settings,
    )

    workbook = None
    if write_output:
        if output_path is None:
            output = settings["output"]
            output_path = root / output["directory"] / f'{output["filename_prefix"]}.xlsx'
        workbook = write_model_workbook(
            combined["route_factor_matrix"],
            combined["input_output_table"],
            output_path,
            lca_stage_breakdown=combined["route_factor_long"],
        )
        progress(f"Completed: {workbook}")

    tables = {
        "inputs": inputs,
        "sut_inventory": sut_inventory,
        "sut_validation": sut_validation,
        "plastics": plastics,
        "wastewater": wastewater,
        "manufacturing": manufacturing,
        "biomass": biomass,
        "combined": combined,
    }
    return PipelineResult(workbook, tables, settings)


def validate_inputs(
    project_root: Path,
    settings_path: Path | None = None,
    overrides: list[str] | None = None,
) -> dict[str, object]:
    root = Path(project_root).expanduser().resolve()
    settings = apply_overrides(load_settings(root, settings_path), overrides)
    inputs = load_input_data(root, settings)
    inventory, validation = build_sut_polymer_inventory(
        inputs["superpro_inventory"],
        inputs["sut_overall"],
        inputs["sut_composition"],
        settings,
    )
    biomass_inventory = extract_biomass_inventory(
        inputs["superpro_inventory"],
        settings,
    )
    return {
        "superpro_inventory_rows": int(len(inputs["superpro_inventory"])),
        "manufacturing_input_rows": int(len(inputs["manufacturing_inputs"])),
        "itemised_cost_rows": int(len(inputs["itemised_costs"])),
        "sut_inventory_rows": int(len(inventory)),
        "biomass_inventory_rows": int(len(biomass_inventory)),
        "biomass_kg_per_batch": float(biomass_inventory.iloc[0]["biomass_kg_per_batch"]),
        **{key: value for key, value in validation.items() if key != "composition_matrix_check"},
    }
