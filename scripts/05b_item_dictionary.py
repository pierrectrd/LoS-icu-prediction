#!/usr/bin/env python3
"""
Step 05b - Build the item dictionary (itemid -> human meaning).

The text renderer (06) needs to turn opaque itemids into "Creatinine ... mg/dL".
This step assembles one lookup from MIMIC's two dictionary tables:
  - d_labitems (hosp)  : labevents items -> label, fluid, category
  - d_items    (icu)   : chartevents items -> label, unit (unitname), category

OUTPUT: 05b_item_dictionary.parquet (+ manifest)
Columns: itemid, source ('lab'|'chart'), label, unit, category

Note: labevents has no per-item unit column in d_labitems (units live per-row in
labevents.valueuom). We capture the most common valueuom per lab separately only
if needed; for now unit is NaN for labs and resolved at render time from valueuom.
"""
from pathlib import Path
import json
import datetime as dt
import pandas as pd

RAW_ROOT = Path("/home/pierrectrd/LoS project/mimic-iv-3.1")
OUT_ROOT = Path("/home/pierrectrd/LoS project/data")
STEP_NAME = "05b_item_dictionary"

def log(msg): print(f"[{dt.datetime.now():%H:%M:%S}] {msg}")

def main():
    # --- d_labitems (labevents) ---
    dlab = pd.read_csv(RAW_ROOT / "hosp" / "d_labitems.csv.gz", dtype=str)
    log(f"d_labitems: {dlab.shape}, cols={list(dlab.columns)}")
    lab = pd.DataFrame({
        "itemid": dlab["itemid"],
        "source": "lab",
        "label": dlab["label"],
        "unit": pd.NA,                      # labs carry unit per-row (valueuom)
        "category": dlab.get("category", pd.NA),
        "fluid": dlab.get("fluid", pd.NA),
    })

    # --- d_items (chartevents) ---
    ditem = pd.read_csv(RAW_ROOT / "icu" / "d_items.csv.gz", dtype=str)
    log(f"d_items: {ditem.shape}, cols={list(ditem.columns)}")
    chart = pd.DataFrame({
        "itemid": ditem["itemid"],
        "source": "chart",
        "label": ditem["label"],
        "unit": ditem.get("unitname", pd.NA),
        "category": ditem.get("category", pd.NA),
        "fluid": pd.NA,
    })

    dictionary = pd.concat([lab, chart], ignore_index=True)
    # An itemid is unique within a table; guard against accidental dup.
    dup = int(dictionary["itemid"].duplicated().sum())

    out_path = OUT_ROOT / f"{STEP_NAME}.parquet"
    dictionary.to_parquet(out_path, index=False)
    log(f"  combined dictionary: {dictionary.shape} (duplicate itemids: {dup})")
    log(f"  wrote -> {out_path}")

    manifest = {
        "step": STEP_NAME,
        "run_at": dt.datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "d_labitems": str(RAW_ROOT / "hosp" / "d_labitems.csv.gz"),
            "d_items": str(RAW_ROOT / "icu" / "d_items.csv.gz"),
        },
        "output": str(out_path),
        "rows": int(len(dictionary)),
        "lab_items": int(len(lab)),
        "chart_items": int(len(chart)),
        "duplicate_itemids": dup,
        "columns": list(dictionary.columns),
        "unit_note": "lab units are per-row (labevents.valueuom), resolved at render time; chart units from d_items.unitname",
        "purpose": "itemid -> label/unit/category for the text renderer (06) and tabular interpretation",
    }
    (OUT_ROOT / f"{STEP_NAME}__manifest.json").write_text(json.dumps(manifest, indent=2))
    log("  wrote manifest"); log("DONE.")

if __name__ == "__main__":
    main()
