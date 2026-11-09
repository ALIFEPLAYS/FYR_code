from __future__ import annotations

import numpy as np
import pandas as pd

from .scientific_parameters import (
    BIOREACTOR_DEFECT_RATE,
    MIXING_BAG_DEFECT_RATE,
    PACKAGING_UPLIFT_RATE,
    TUBING_TO_ASSEMBLY_MASS_RATIO,
    consumable_defect_rate,
)


POLYMER_COLUMNS = ["LLDPE", "LDPE", "HDPE", "PP", "TPE", "EVOH", "Other"]
DETAIL_COLUMNS = [
    "inventory_id",
    "process_area",
    "superpro_section",
    "unit_operation",
    "source_procedure",
    "destination",
    "flow_name_superpro",
    "component",
    "sut_subcomponent",
    "item_count_per_batch",
    "item_count_per_year",
    "assembly_kg_per_item_reference",
    "assembly_kg_per_batch_reference",
    "assembly_kg_per_year_reference",
    "source_subcomponent_kg_per_item",
    "mass_after_tubing_adjustment_kg_per_item",
    "tubing_adjustment_factor",
    "defect_rate",
    "defect_multiplier",
    "subcomponent_kg_per_item",
    "subcomponent_kg_per_batch",
    "subcomponent_kg_per_year",
    "source_sut_kg_per_item",
    "sut_kg_per_item_after_tubing_adjustment",
    "sut_kg_per_item",
    "sut_kg_per_batch",
    "sut_kg_per_year",
    "flow_key",
    "source",
    "link",
    "notes",
    "composition_assumptions",
    "composition_sources",
    "source_polymer_kg_per_item",
    "polymer_kg_per_item_after_tubing_adjustment",
    "polymer_kg_per_item",
    "polymer_kg_per_batch",
    "polymer",
    "polymer_kg_per_year",
]


def _process_area(value) -> str:
    key = str(value).strip().lower().replace(" ", "_")
    mapping = {
        "upstream": "Upstream",
        "downstream": "Downstream",
        "formulation_fill": "Fill finish",
        "fill_finish": "Fill finish",
    }
    return mapping.get(key, str(value).replace("_", " ").title())


def _prepare_overall_table(frame: pd.DataFrame) -> pd.DataFrame:
    overall = frame.copy()
    overall = overall.rename(columns={
        "kg material/item": "assembly_kg_per_item_reference",
        "report/proudct": "report_or_product",
    })
    required = ["flow_name_superpro", "assembly_kg_per_item_reference"]
    missing = [column for column in required if column not in overall.columns]
    if missing:
        raise ValueError(f"SUT_overall_plastic is missing columns: {missing}")
    overall["assembly_kg_per_item_reference"] = pd.to_numeric(
        overall["assembly_kg_per_item_reference"], errors="coerce"
    )
    return overall


def _prepare_composition_table(frame: pd.DataFrame) -> pd.DataFrame:
    composition = frame.copy()
    first_column = composition.columns[0]
    composition = composition.rename(columns={
        first_column: "flow_name_superpro",
        "Overall plastic assembly qty (kg)": "subcomponent_kg_per_item",
        "Assumptions": "composition_assumptions",
        "Composition Source(s)": "composition_sources",
    })
    required = ["flow_name_superpro", "Category", "subcomponent_kg_per_item"]
    missing = [column for column in required if column not in composition.columns]
    if missing:
        raise ValueError(f"SUT_plastic_comp_matrix is missing columns: {missing}")

    composition = composition[
        composition["Category"].isin(["Assembly", "Tubing"])
        & composition["flow_name_superpro"].notna()
    ].copy()
    composition["subcomponent_kg_per_item"] = pd.to_numeric(
        composition["subcomponent_kg_per_item"], errors="coerce"
    )
    for polymer in POLYMER_COLUMNS:
        if polymer not in composition.columns:
            composition[polymer] = 0.0
        composition[polymer] = pd.to_numeric(composition[polymer], errors="coerce").fillna(0.0)
    return composition


def _apply_journal_paper_mass_assumptions(
    composition: pd.DataFrame,
    settings: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply tubing and defect assumptions while retaining every mass basis.

    The source composition table is never modified on disk. Packaging is already
    embedded in the underlying assembly weights, so the configured packaging
    uplift is required to remain zero to prevent double counting.
    """
    adjusted = composition.copy().reset_index(drop=True)
    plastics = settings["plastics"]
    tubing_ratio = float(
        plastics.get("tubing_to_assembly_mass_ratio", TUBING_TO_ASSEMBLY_MASS_RATIO)
    )
    bioreactor_rate = float(
        plastics.get("bioreactor_defect_rate", BIOREACTOR_DEFECT_RATE)
    )
    mixing_bag_rate = float(
        plastics.get("mixing_bag_defect_rate", MIXING_BAG_DEFECT_RATE)
    )
    packaging_uplift = float(
        plastics.get("packaging_uplift_rate", PACKAGING_UPLIFT_RATE)
    )
    if tubing_ratio < 0:
        raise ValueError("plastics.tubing_to_assembly_mass_ratio must be non-negative.")
    if not 0 <= bioreactor_rate <= 1 or not 0 <= mixing_bag_rate <= 1:
        raise ValueError("Consumable defect rates must be between 0 and 1.")
    if packaging_uplift != 0.0:
        raise ValueError(
            "No standalone packaging uplift is permitted: packaging is already "
            "represented in the source assembly masses."
        )

    adjusted["source_subcomponent_kg_per_item"] = adjusted["subcomponent_kg_per_item"]
    for polymer in POLYMER_COLUMNS:
        adjusted[f"source_{polymer}"] = adjusted[polymer]

    assembly_mass = (
        adjusted.loc[adjusted["Category"].eq("Assembly")]
        .groupby("flow_name_superpro")["source_subcomponent_kg_per_item"]
        .sum()
    )
    adjusted["tubing_adjustment_factor"] = 1.0
    tubing_mask = adjusted["Category"].eq("Tubing")
    for index in adjusted.index[tubing_mask]:
        flow = adjusted.at[index, "flow_name_superpro"]
        source_tubing_mass = float(adjusted.at[index, "source_subcomponent_kg_per_item"])
        if flow not in assembly_mass:
            raise ValueError(f"Tubing row has no matching assembly mass: {flow}")
        target_tubing_mass = float(assembly_mass[flow]) * tubing_ratio
        if source_tubing_mass == 0.0:
            # A zero tubing row is an explicit source statement that the assembly
            # has no represented tubing (for example, filter cartridges). Do not
            # fabricate tubing merely because a matching assembly mass exists.
            factor = 1.0
        else:
            factor = target_tubing_mass / source_tubing_mass
        adjusted.at[index, "tubing_adjustment_factor"] = factor

    adjusted["mass_after_tubing_adjustment_kg_per_item"] = (
        adjusted["source_subcomponent_kg_per_item"]
        * adjusted["tubing_adjustment_factor"]
    )
    for polymer in POLYMER_COLUMNS:
        adjusted[f"{polymer}_after_tubing_adjustment"] = (
            adjusted[f"source_{polymer}"] * adjusted["tubing_adjustment_factor"]
        )

    adjusted["defect_rate"] = adjusted["flow_name_superpro"].map(
        lambda name: consumable_defect_rate(
            name,
            bioreactor_rate=bioreactor_rate,
            mixing_bag_rate=mixing_bag_rate,
        )
    )
    adjusted["defect_multiplier"] = 1.0 + adjusted["defect_rate"]
    adjusted["subcomponent_kg_per_item"] = (
        adjusted["mass_after_tubing_adjustment_kg_per_item"]
        * adjusted["defect_multiplier"]
    )
    for polymer in POLYMER_COLUMNS:
        adjusted[polymer] = (
            adjusted[f"{polymer}_after_tubing_adjustment"]
            * adjusted["defect_multiplier"]
        )

    reconciliation = (
        adjusted.groupby("flow_name_superpro", as_index=False)
        .agg(
            source_mass_kg_per_item=("source_subcomponent_kg_per_item", "sum"),
            mass_after_tubing_adjustment_kg_per_item=(
                "mass_after_tubing_adjustment_kg_per_item",
                "sum",
            ),
            defect_rate=("defect_rate", "max"),
            final_modelled_mass_kg_per_item=("subcomponent_kg_per_item", "sum"),
        )
    )
    reconciliation["tubing_mass_change_kg_per_item"] = (
        reconciliation["mass_after_tubing_adjustment_kg_per_item"]
        - reconciliation["source_mass_kg_per_item"]
    )
    reconciliation["defect_uplift_kg_per_item"] = (
        reconciliation["final_modelled_mass_kg_per_item"]
        - reconciliation["mass_after_tubing_adjustment_kg_per_item"]
    )
    reconciliation["packaging_uplift_kg_per_item"] = 0.0
    return adjusted, reconciliation


def build_sut_polymer_inventory(
    superpro_inventory: pd.DataFrame,
    sut_overall: pd.DataFrame,
    sut_composition: pd.DataFrame,
    settings: dict,
) -> tuple[pd.DataFrame, dict[str, object]]:
    sut = superpro_inventory[
        superpro_inventory["source_report"].astype(str).str.strip().eq("ICR")
        & superpro_inventory["destination"].astype(str).str.strip().eq("END_OF_LIFE_OUTPUT")
    ].copy()
    if sut.empty:
        raise ValueError("No SUT end-of-life rows were found in the processed SuperPro dataset.")

    for column in ["amount_per_batch", "amount_per_year"]:
        sut[column] = pd.to_numeric(sut[column], errors="coerce")
    if sut[["amount_per_batch", "amount_per_year"]].isna().any().any():
        raise ValueError("SUT item counts contain missing or non-numeric values.")

    overall = _prepare_overall_table(sut_overall)
    composition = _prepare_composition_table(sut_composition)
    composition, mass_reconciliation = _apply_journal_paper_mass_assumptions(
        composition,
        settings,
    )

    composition["_composition_row_id"] = np.arange(len(composition))
    trace_id_vars = [
        "_composition_row_id",
        "flow_name_superpro",
        "Category",
        "source_subcomponent_kg_per_item",
        "mass_after_tubing_adjustment_kg_per_item",
        "tubing_adjustment_factor",
        "defect_rate",
        "defect_multiplier",
        "subcomponent_kg_per_item",
        "composition_assumptions",
        "composition_sources",
    ]

    polymer_long = composition.melt(
        id_vars=trace_id_vars,
        value_vars=POLYMER_COLUMNS,
        var_name="polymer",
        value_name="polymer_kg_per_item",
    )
    source_polymer_long = composition.melt(
        id_vars=["_composition_row_id"],
        value_vars=[f"source_{polymer}" for polymer in POLYMER_COLUMNS],
        var_name="polymer",
        value_name="source_polymer_kg_per_item",
    )
    source_polymer_long["polymer"] = source_polymer_long["polymer"].str.removeprefix("source_")
    tubing_adjusted_polymer_long = composition.melt(
        id_vars=["_composition_row_id"],
        value_vars=[f"{polymer}_after_tubing_adjustment" for polymer in POLYMER_COLUMNS],
        var_name="polymer",
        value_name="polymer_kg_per_item_after_tubing_adjustment",
    )
    tubing_adjusted_polymer_long["polymer"] = tubing_adjusted_polymer_long[
        "polymer"
    ].str.removesuffix("_after_tubing_adjustment")
    polymer_long = polymer_long.merge(
        source_polymer_long,
        on=["_composition_row_id", "polymer"],
        how="left",
        validate="one_to_one",
    ).merge(
        tubing_adjusted_polymer_long,
        on=["_composition_row_id", "polymer"],
        how="left",
        validate="one_to_one",
    )
    polymer_long = polymer_long[polymer_long["polymer_kg_per_item"] > 0].copy()

    mapped_flows = set(polymer_long["flow_name_superpro"].astype(str))
    unmatched = sut[~sut["flow_name_superpro"].astype(str).isin(mapped_flows)].copy()
    policy = settings["plastics"].get("unmapped_composition_policy", "exclude")
    if policy not in {"exclude", "error"}:
        raise ValueError("plastics.unmapped_composition_policy must be 'exclude' or 'error'.")
    if policy == "error" and not unmatched.empty:
        names = sorted(unmatched["flow_name_superpro"].dropna().astype(str).unique())
        raise ValueError(f"SUT composition is missing for: {names}")

    mapped = sut.merge(
        polymer_long,
        on="flow_name_superpro",
        how="inner",
        validate="many_to_many",
    )
    mapped = mapped.merge(
        overall[
            [
                "flow_name_superpro",
                "assembly_kg_per_item_reference",
                "source",
                "link",
                "notes",
            ]
        ],
        on="flow_name_superpro",
        how="left",
        validate="many_to_one",
    )

    total_sut_per_item = composition.groupby("flow_name_superpro", as_index=False).agg(
        source_sut_kg_per_item=("source_subcomponent_kg_per_item", "sum"),
        sut_kg_per_item_after_tubing_adjustment=(
            "mass_after_tubing_adjustment_kg_per_item",
            "sum",
        ),
        sut_kg_per_item=("subcomponent_kg_per_item", "sum"),
    )
    mapped = mapped.merge(total_sut_per_item, on="flow_name_superpro", how="left", validate="many_to_one")

    mapped["process_area"] = mapped["broad_process_area"].map(_process_area)
    mapped["sut_subcomponent"] = mapped["Category"]
    mapped["item_count_per_batch"] = mapped["amount_per_batch"]
    mapped["item_count_per_year"] = mapped["amount_per_year"]
    mapped["assembly_kg_per_batch_reference"] = (
        mapped["item_count_per_batch"] * mapped["assembly_kg_per_item_reference"]
    )
    mapped["assembly_kg_per_year_reference"] = (
        mapped["item_count_per_year"] * mapped["assembly_kg_per_item_reference"]
    )
    mapped["subcomponent_kg_per_batch"] = (
        mapped["item_count_per_batch"] * mapped["subcomponent_kg_per_item"]
    )
    mapped["subcomponent_kg_per_year"] = (
        mapped["item_count_per_year"] * mapped["subcomponent_kg_per_item"]
    )
    mapped["sut_kg_per_batch"] = mapped["item_count_per_batch"] * mapped["sut_kg_per_item"]
    mapped["sut_kg_per_year"] = mapped["item_count_per_year"] * mapped["sut_kg_per_item"]
    mapped["polymer_kg_per_batch"] = (
        mapped["item_count_per_batch"] * mapped["polymer_kg_per_item"]
    )
    mapped["polymer_kg_per_year"] = (
        mapped["item_count_per_year"] * mapped["polymer_kg_per_item"]
    )
    mapped["flow_key"] = mapped["flow_name_superpro"].astype(str).str.strip().str.lower()

    if mapped["polymer_kg_per_batch"].isna().any() or (mapped["polymer_kg_per_batch"] < 0).any():
        raise ValueError("The processed SUT polymer inventory contains invalid masses.")

    detail = mapped[DETAIL_COLUMNS].sort_values(
        ["process_area", "superpro_section", "flow_name_superpro", "sut_subcomponent", "polymer"],
        kind="stable",
    ).reset_index(drop=True)

    matrix_check = composition.copy()
    matrix_check["polymer_sum_kg_per_item"] = matrix_check[POLYMER_COLUMNS].sum(axis=1)
    matrix_check["composition_gap_kg_per_item"] = (
        matrix_check["polymer_sum_kg_per_item"] - matrix_check["subcomponent_kg_per_item"]
    )

    source_polymer_kg_per_batch = float(
        (detail["source_polymer_kg_per_item"] * detail["item_count_per_batch"]).sum()
    )
    polymer_kg_per_batch_after_tubing = float(
        (
            detail["polymer_kg_per_item_after_tubing_adjustment"]
            * detail["item_count_per_batch"]
        ).sum()
    )
    total_polymer_kg_per_batch = float(detail["polymer_kg_per_batch"].sum())
    validation = {
        "sut_rows_total": int(len(sut)),
        "sut_rows_mapped": int(sut["flow_name_superpro"].isin(mapped_flows).sum()),
        "sut_rows_unmapped": int(len(unmatched)),
        "unmapped_flows": sorted(unmatched["flow_name_superpro"].dropna().astype(str).unique().tolist()),
        "detail_rows": int(len(detail)),
        "total_polymer_kg_per_batch": total_polymer_kg_per_batch,
        "source_polymer_kg_per_batch": source_polymer_kg_per_batch,
        "polymer_kg_per_batch_after_tubing_adjustment": polymer_kg_per_batch_after_tubing,
        "tubing_adjustment_delta_kg_per_batch": (
            polymer_kg_per_batch_after_tubing - source_polymer_kg_per_batch
        ),
        "defect_uplift_delta_kg_per_batch": (
            total_polymer_kg_per_batch - polymer_kg_per_batch_after_tubing
        ),
        "packaging_uplift_rate": 0.0,
        "mass_adjustment_reconciliation": mass_reconciliation,
        "polymer_totals_kg_per_batch": detail.groupby("polymer")["polymer_kg_per_batch"].sum().to_dict(),
        "composition_matrix_check": matrix_check,
    }
    return detail, validation
