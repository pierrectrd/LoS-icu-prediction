#!/usr/bin/env python3
"""
Step 03b - ICU stays standalone base table (subtask 2: ICU LoS prediction).

INPUT : icu/icustays.csv.gz                 (one row per ICU stay)
        02_cohort_features.parquet           (for patient demographics to attach)
OUTPUT: 03_icu_stays.parquet                 (+ manifest)

GOAL: a clean ONE-ROW-PER-ICU-STAY base for subtask 2. The target (ICU LoS) is
already provided as `los` in icustays, so we just rename it `icu_los_days`.
We KEEP intime/outtime so a "first 24h" observation window can be defined later
(NOT applied here -- that is a modeling decision, like the <24h stay exclusion).

We attach a few admission/patient attributes (age, gender, admission_type) by
joining on hadm_id, so subtask-2 models have basic demographics without
re-reading raw tables.

NOTE: no row filtering here (no <24h cut). Extraction stays faithful; modeling
decisions stay separate and reversible.
"""

from pathlib import Path
import json
import datetime as dt
import pandas as pd

RAW_ROOT = Path("/home/pierrectrd/LoS project/mimic-iv-3.1")
OUT_ROOT = Path("/home/pierrectrd/LoS project/data")
ICU = RAW_ROOT / "icu"
COHORT_NAME = "02_cohort_features"
STEP_NAME = "03_icu_stays"

# Admission/patient columns to attach to each ICU stay (demographics only -- no
# hospital-stay outcome columns, to avoid dragging hospital-LoS leakage in).
ATTACH_COLS = ["hadm_id", "subject_id", "age_at_admission", "gender",
               "admission_type", "insurance", "marital_status", "race",
               "in_hospital_death"]

def log(msg: str) -> None:
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}")

def main() -> None:
    icu_path = ICU / "icustays.csv.gz"
    cohort_path = OUT_ROOT / f"{COHORT_NAME}.parquet"

    log(f"Reading {icu_path}")
    icu = pd.read_csv(icu_path, dtype=str)
    log(f"  icustays raw shape: {icu.shape}")

    icu["los"] = pd.to_numeric(icu["los"], errors="coerce")
    icu["intime"] = pd.to_datetime(icu["intime"], errors="coerce")
    icu["outtime"] = pd.to_datetime(icu["outtime"], errors="coerce")
    icu = icu.rename(columns={"los": "icu_los_days"})   # the subtask-2 target

    # Flag invalid ICU LoS (mirror step 01's hygiene), archive + drop.
    invalid = icu["icu_los_days"].isna() | (icu["icu_los_days"] <= 0)
    n_invalid = int(invalid.sum())
    archive_dir = OUT_ROOT / "_dropped_rows"
    archive_dir.mkdir(parents=True, exist_ok=True)
    if n_invalid:
        icu[invalid].to_parquet(archive_dir / f"{STEP_NAME}__invalid_icu_los.parquet",
                                index=False)
    icu = icu[~invalid].copy()
    log(f"  dropped invalid icu_los rows: {n_invalid}")

    # Attach demographics from cohort (one row per hadm_id) onto each stay.
    log(f"Reading cohort {cohort_path} for demographics")
    cohort = pd.read_parquet(cohort_path)
    attach = [c for c in ATTACH_COLS if c in cohort.columns]
    # cohort is one row per hadm_id; drop dup hadm just in case before merge.
    demo = cohort[attach].drop_duplicates(subset="hadm_id")
    stays = icu.merge(demo, on="hadm_id", how="left")
    # subject_id appears in both; resolve any suffix collision.
    if "subject_id_x" in stays.columns:
        stays = stays.rename(columns={"subject_id_x": "subject_id"}).drop(
            columns=[c for c in ["subject_id_y"] if c in stays.columns])
    log(f"  stays with demographics: {stays.shape}")

    matched = int(stays["age_at_admission"].notna().sum()) if "age_at_admission" in stays else 0
    log(f"  stays matched to a cohort admission: {matched}/{len(stays)}")

    out_path = OUT_ROOT / f"{STEP_NAME}.parquet"
    stays.to_parquet(out_path, index=False)
    log(f"  wrote -> {out_path}  shape={stays.shape}")

    manifest = {
        "step": STEP_NAME,
        "run_at": dt.datetime.now().isoformat(timespec="seconds"),
        "inputs": {"icustays": str(icu_path), "cohort_for_demographics": str(cohort_path)},
        "output": str(out_path),
        "grain": "one row per ICU stay (stay_id)",
        "target_column": "icu_los_days",
        "rows_after_invalid_drop": int(len(stays)),
        "invalid_icu_los_dropped": n_invalid,
        "stays_matched_to_cohort": matched,
        "kept_time_columns": ["intime", "outtime"],
        "window_note": "intime/outtime kept so a first-24h window can be defined later; NOT applied here",
        "filters_NOT_applied": ["<24h stay exclusion", "first-ICU-only restriction"],
        "demographics_attached": attach,
        "columns_out": list(stays.columns),
        "notes": (
            "Standalone ICU-stay table for subtask 2. Target icu_los_days already "
            "provided by MIMIC. No window/length filtering (modeling decisions). "
            "Raw data untouched."
        ),
    }
    (OUT_ROOT / f"{STEP_NAME}__manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"  wrote manifest")
    log("DONE.")

if __name__ == "__main__":
    main()
