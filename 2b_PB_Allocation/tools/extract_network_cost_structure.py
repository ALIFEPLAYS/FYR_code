"""Extract the SuperPro cost structure used for the network-specific input-origin scenarios.

Hartmann et al. (2026) Eqs. 8-9 split total GVA between regions by alpha_{r,j}: the share of all
first-tier inputs to sector j (intermediate purchases + own value added) supplied by region r. The
default uses EXIOBASE's sector average for GB Chemicals nec. For a sensitivity, the network's own
cost structure is used instead: SuperPro 'SUMMARY PER COST ITEM (Entire Process)'.

Classification (first-tier supplier region, input-output accounting):
  domestic value added / UK services   Facility (depreciation = consumption of fixed capital, maintenance,
                                       insurance, local taxes), Labor, Lab/QC/QA, Utilities, Waste Trtmt/Disp
  purchased goods, origin uncertain    Raw Materials (media, buffers, chemicals), Consumables (single-use
                                       bags, filters, membranes, chromatography resins)

Run from the repository root:
    .venv/bin/python -B 2b_PB_Allocation/tools/extract_network_cost_structure.py --icr <path to mab_fedbatch_1000_sut_ICR.xls>
Writes 2b_PB_Allocation/data/network_cost_structure.json (committed, so the allocation runs without SuperPro files).
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/network_cost_structure.json"
ICR_NAME = "rawSP_reports/mab_fedbatch_1000_sut_ICR.xls"
# Ahmed's project layout first, then the shared GitHub layout.
DEFAULT_ICR = next((p for p in (ROOT.parent / "01_SuperPro" / ICR_NAME, ROOT.parent / "1_SuperPro_Results" / ICR_NAME)
                    if p.exists()), ROOT.parent / "01_SuperPro" / ICR_NAME)
DOMESTIC = ("Facility", "Labor", "Lab/QC/QA", "Utilities", "Waste Trtmt/Disp")
PURCHASED_GOODS = ("Raw Materials", "Consumables")
ZERO_ITEMS = ("Transportation", "Miscellaneous")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--icr", type=Path, default=DEFAULT_ICR)
    icr = ap.parse_args().icr
    df = pd.read_excel(icr, sheet_name=0, header=None)
    start = df.index[df[1].astype(str).str.strip() == "SUMMARY PER COST ITEM (Entire Process)"]
    if len(start) != 1:
        raise ValueError("Entire-process cost summary not found exactly once")
    block = df.iloc[start[0] + 2:start[0] + 14]
    items = {}
    for _, row in block.iterrows():
        name = str(row[1]).strip()
        if name == "TOTAL":
            total = float(str(row[21]).replace(",", ""))
            break
        items[name] = float(str(row[21]).replace(",", ""))
    if abs(sum(items.values()) - total) > 1.0:
        raise ValueError("Cost items do not add up to the SuperPro total")
    if set(items) != set(DOMESTIC + PURCHASED_GOODS + ZERO_ITEMS):
        raise ValueError(f"Unexpected cost items {sorted(items)}")
    record = dict(
        description="SuperPro entire-process annual cost by item, one 121-batch plant, USD2019/year.",
        source=str(icr.relative_to(ROOT.parent)) if icr.is_relative_to(ROOT.parent) else icr.name,
        source_sha256=hashlib.sha256(icr.read_bytes()).hexdigest(),
        cost_usd2019_per_year=items, total_usd2019_per_year=total,
        domestic_items=list(DOMESTIC), purchased_goods_items=list(PURCHASED_GOODS),
        domestic_share=sum(items[k] for k in DOMESTIC) / total,
        purchased_goods_share=sum(items[k] for k in PURCHASED_GOODS) / total,
        status="verified_model_output (classification of items is an assumption)",
    )
    OUT.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: record[k] for k in ("domestic_share", "purchased_goods_share", "total_usd2019_per_year")}))


if __name__ == "__main__":
    main()
