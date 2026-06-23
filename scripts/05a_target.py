#!/usr/bin/env python3
"""
Step 05a - Retarget to REMAINING LoS after the 24h observation window.

Rationale (director): predicting total icu_los_days from the first 24h, on a
>=24h cohort, bakes 1 day into every target and imposes a floor of 1. The
clinically actionable quantity is "given the first 24h, how much longer?".

  remaining_los_days = icu_los_days - 1.0        (window = 24h = 1 day)

Defined on the 04a cohort (already first-ICU, >=24h), so remaining_los_days >= 0
always. This becomes THE target for both modalities (tabular and LLM).

INPUT : 04a_icu_los_cohort.parquet   (note: your file may be named
        04_icu_los_cohort.parquet -- set IN_NAME accordingly)
OUTPUT: 05a_target.parquet (+ 05a_target__manifest.json)
"""
from pathlib import Path
import json
import datetime as dt
import pandas as pd

OUT_ROOT = Path("/home/pierrectrd/LoS project/data")
# Your existing cohort file from step 04a (adjust if your filename differs):
IN_NAME = "04_icu_los_cohort"
STEP_NAME = "05a_target"
WINDOW_DAYS = 1.0   # 24h observation window

def log(msg): print(f"[{dt.datetime.now():%H:%M:%S}] {msg}")

def main():
    in_path = OUT_ROOT / f"{IN_NAME}.parquet"
    log(f"Reading {in_path}")
    df = pd.read_parquet(in_path)
    df["icu_los_days"] = pd.to_numeric(df["icu_los_days"], errors="coerce")

    df["remaining_los_days"] = df["icu_los_days"] - WINDOW_DAYS
    # On a >=24h cohort this is >= 0; guard against tiny float negatives.
    n_neg = int((df["remaining_los_days"] < 0).sum())
    df["remaining_los_days"] = df["remaining_los_days"].clip(lower=0.0)

    # Round both LoS columns to 2 decimals -- sub-minute precision is meaningless
    # (0.000006 days is noise) and clutters the LLM text and the manifests.
    df["icu_los_days"] = df["icu_los_days"].round(2)
    df["remaining_los_days"] = df["remaining_los_days"].round(2)

    out_path = OUT_ROOT / f"{STEP_NAME}.parquet"
    df.to_parquet(out_path, index=False)
    log(f"  added remaining_los_days; clipped {n_neg} tiny negatives to 0")
    log(f"  wrote -> {out_path}  shape={df.shape}")

    manifest = {
        "step": STEP_NAME,
        "run_at": dt.datetime.now().isoformat(timespec="seconds"),
        "input": str(in_path),
        "output": str(out_path),
        "new_target": "remaining_los_days = icu_los_days - 1.0",
        "old_target_retained": "icu_los_days (kept for reference/comparison)",
        "window_days_subtracted": WINDOW_DAYS,
        "tiny_negatives_clipped": n_neg,
        "remaining_los_summary_days": {
            "mean": float(df["remaining_los_days"].mean()),
            "median": float(df["remaining_los_days"].median()),
            "min": float(df["remaining_los_days"].min()),
            "p95": float(df["remaining_los_days"].quantile(0.95)),
            "max": float(df["remaining_los_days"].max()),
        },
        "notes": "Target for BOTH modalities. Right-skewed -> Box-Cox later (fit on train only).",
    }
    (OUT_ROOT / f"{STEP_NAME}__manifest.json").write_text(json.dumps(manifest, indent=2))
    log("  wrote manifest"); log("DONE.")

if __name__ == "__main__":
    main()
