#!/usr/bin/env python3
"""
Step 09b - Keep only stays with strictly positive remaining LoS.

The >=24h cohort + (los - 1) retarget (05a) leaves 130 stays at exactly
remaining_los_days == 0 (discharged at ~24h). A zero target is legitimate but
awkward for the Box-Cox transform (and ambiguous for "remaining" LoS), so we
drop them here into their own step, archiving the dropped rows per project
convention. Input is the live text head (09); output mirrors its schema.

INPUT : 09_clinical_text_sofa.parquet
OUTPUT: 09b_los_strictly_positive.parquet (+ manifest); zeros -> _dropped_rows/
"""
from pathlib import Path
import json
import datetime as dt
import pandas as pd

OUT_ROOT = Path("/home/pierrectrd/LoS project/data")
IN_NAME = "09_clinical_text_sofa"
STEP_NAME = "09b_los_strictly_positive"

def log(msg): print(f"[{dt.datetime.now():%H:%M:%S}] {msg}")

def main():
    in_path = OUT_ROOT / f"{IN_NAME}.parquet"
    df = pd.read_parquet(in_path)
    n_in = len(df)

    # strictly positive == drop <= 0 (05a clips to >= 0, so this is just the zeros)
    keep = df["remaining_los_days"] > 0
    dropped, out = df[~keep].copy(), df[keep].copy()
    assert (out["remaining_los_days"] > 0).all(), "non-positive target survived the filter"

    archive = OUT_ROOT / "_dropped_rows"
    archive.mkdir(parents=True, exist_ok=True)
    dropped.to_parquet(archive / f"{STEP_NAME}__zero_remaining_los.parquet", index=False)

    out_path = OUT_ROOT / f"{STEP_NAME}.parquet"
    out.to_parquet(out_path, index=False)
    log(f"dropped {len(dropped)} zero-LoS stays; {n_in} -> {len(out)}")

    manifest = {
        "step": STEP_NAME,
        "run_at": dt.datetime.now().isoformat(timespec="seconds"),
        "input": str(in_path),
        "output": str(out_path),
        "dropped_archive": str(archive / f"{STEP_NAME}__zero_remaining_los.parquet"),
        "rows_in": n_in,
        "rows_dropped_zero_remaining_los": int(len(dropped)),
        "rows_out": int(len(out)),
        "filter": "remaining_los_days > 0",
        "notes": "Zeros (~24h stays) dropped for a strictly positive target. Reversible via archive.",
    }
    (OUT_ROOT / f"{STEP_NAME}__manifest.json").write_text(json.dumps(manifest, indent=2))
    log("wrote manifest"); log("DONE.")

if __name__ == "__main__":
    main()
