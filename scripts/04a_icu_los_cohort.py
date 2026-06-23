#!/usr/bin/env python3
"""
Step 04a - ICU-LoS cohort (subtask 2), MedM2T Task-3 style.

INPUT : 03_icu_stays_clean.parquet  (one row per ICU stay)
OUTPUT: 04_icu_los_cohort.parquet   (+ manifest)

Reproduces MedM2T's ICU cohort exactly:
  - FIRST ICU stay per patient (min intime per subject_id), like their
    mimic_first_icu.py (MIN(intime) GROUP BY subject_id).
  - EXCLUDE stays shorter than 24h (their <24h exclusion). Order matters:
    take the first stay first, THEN drop it if it is <24h (a patient whose
    *first* ICU stay is short is excluded, matching their SQL semantics).
  - Defines the feature window [intime, intime + 24h] used by 04b/04c.

The target is icu_los_days (full ICU stay length). Features (built later) come
ONLY from the first 24h -> prospective, non-leaky. Dropped rows are archived.
"""
from pathlib import Path
import json
import datetime as dt
import pandas as pd

OUT_ROOT = Path("/home/pierrectrd/LoS project/data")
IN_NAME = "03_icu_stays_clean"
STEP_NAME = "04_icu_los_cohort"
WINDOW_HOURS = 24

def log(msg): print(f"[{dt.datetime.now():%H:%M:%S}] {msg}")

def main():
    in_path = OUT_ROOT / f"{IN_NAME}.parquet"
    log(f"Reading {in_path}")
    df = pd.read_parquet(in_path)
    log(f"  input shape: {df.shape}")
    n_in = len(df)

    df["intime"] = pd.to_datetime(df["intime"], errors="coerce")
    df["icu_los_days"] = pd.to_numeric(df["icu_los_days"], errors="coerce")

    archive = OUT_ROOT / "_dropped_rows"
    archive.mkdir(parents=True, exist_ok=True)

    # --- 1. First ICU stay per patient (min intime per subject_id). ---
    df_sorted = df.sort_values("intime")
    first_idx = df_sorted.groupby("subject_id")["intime"].idxmin()
    first = df_sorted.loc[first_idx].copy()
    non_first = df_sorted.drop(index=first_idx)
    non_first.to_parquet(archive / f"{STEP_NAME}__non_first_icu_stays.parquet", index=False)
    log(f"  first ICU stay per patient: {len(first)} (dropped {len(non_first)} later stays)")

    # --- 2. Exclude first stays shorter than 24h. ---
    short_mask = first["icu_los_days"] < (WINDOW_HOURS / 24.0)
    short = first[short_mask].copy()
    cohort = first[~short_mask].copy()
    short.to_parquet(archive / f"{STEP_NAME}__first_stay_under_24h.parquet", index=False)
    log(f"  excluded first stays <24h: {len(short)}; cohort now {len(cohort)}")

    # --- 3. Define the feature window. ---
    cohort["window_start"] = cohort["intime"]
    cohort["window_end"] = cohort["intime"] + pd.Timedelta(hours=WINDOW_HOURS)

    out_path = OUT_ROOT / f"{STEP_NAME}.parquet"
    cohort.to_parquet(out_path, index=False)
    log(f"  wrote -> {out_path}  shape={cohort.shape}")

    manifest = {
        "step": STEP_NAME,
        "run_at": dt.datetime.now().isoformat(timespec="seconds"),
        "input": str(in_path),
        "output": str(out_path),
        "approach": "MedM2T Task-3: first ICU stay per patient, stay >= 24h",
        "window_hours": WINDOW_HOURS,
        "rows_in_all_stays": n_in,
        "after_first_per_patient": int(len(first)),
        "dropped_non_first_stays": int(len(non_first)),
        "dropped_first_stay_under_24h": int(len(short)),
        "cohort_out": int(len(cohort)),
        "target_column": "icu_los_days (full ICU stay length)",
        "feature_window": "[intime, intime + 24h] (features built in 04b/04c)",
        "columns_out": list(cohort.columns),
        "notes": "Features come ONLY from first 24h -> prospective. Full-stay target. Raw untouched.",
    }
    (OUT_ROOT / f"{STEP_NAME}__manifest.json").write_text(json.dumps(manifest, indent=2))
    log("  wrote manifest"); log("DONE.")

if __name__ == "__main__":
    main()
