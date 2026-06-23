#!/usr/bin/env python3
"""
Step 03a - ICU features at ADMISSION grain (subtask 1: general-ward LoS).

INPUT : 02_cohort_features.parquet  (one row per hospital admission)
        icu/icustays.csv.gz          (one row per ICU stay)
OUTPUT: 03_cohort_with_icu.parquet   (+ manifest)

GOAL: give every hospital admission an ICU feature block, so a single model can
serve ICU and non-ICU patients. icustays is aggregated to one row per hadm_id
(a hadm may contain several ICU stays), then LEFT-joined onto the cohort.

LEAKAGE TAGGING (important): some aggregated ICU features are "future
information" relative to an AT-ADMISSION prediction, because they summarise a
duration that overlaps the very stay we predict. We BUILD them (per director's
"extract ICU features for now"), but tag each in the manifest by when it becomes
known, so they can be included/excluded consciously at modeling time.

This step uses ONLY icustays (small). The giant event tables (chartevents,
inputevents, ...) are deferred to later, windowed feature steps.
"""

from pathlib import Path
import json
import datetime as dt
import pandas as pd

RAW_ROOT = Path("/home/pierrectrd/LoS project/mimic-iv-3.1")
OUT_ROOT = Path("/home/pierrectrd/LoS project/data")
ICU = RAW_ROOT / "icu"
IN_NAME = "02_cohort_features"
STEP_NAME = "03_cohort_with_icu"

def log(msg: str) -> None:
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}")

def main() -> None:
    in_path = OUT_ROOT / f"{IN_NAME}.parquet"
    icu_path = ICU / "icustays.csv.gz"

    log(f"Reading cohort {in_path}")
    cohort = pd.read_parquet(in_path)
    log(f"  cohort shape: {cohort.shape}")

    log(f"Reading {icu_path}")
    icu = pd.read_csv(icu_path, dtype=str)
    log(f"  icustays raw shape: {icu.shape}")

    # Cast the numeric / time columns we need.
    icu["los"] = pd.to_numeric(icu["los"], errors="coerce")
    icu["intime"] = pd.to_datetime(icu["intime"], errors="coerce")
    icu["outtime"] = pd.to_datetime(icu["outtime"], errors="coerce")

    # --- Aggregate icustays to ONE ROW PER hadm_id. ---
    # A hadm can have multiple ICU stays; collapse them to STRUCTURAL features
    # that describe WHETHER/HOW-OFTEN/WHERE ICU contact happened -- deliberately
    # NOT how long it lasted or how it ended (those are leaky and are deferred to
    # the windowed feature stage, where they return BOUNDED by the prediction
    # horizon and therefore non-leaky).
    grp = icu.groupby("hadm_id")
    icu_agg = pd.DataFrame({
        "n_icu_stays":         grp.size(),                       # multi-stay signal
        "icu_first_intime":    grp["intime"].min(),              # for later windowing
        "first_careunit":      grp["first_careunit"].first(),    # which unit first
        "n_distinct_careunits": grp["first_careunit"].nunique(), # escalation/transfers
    }).reset_index()
    log(f"  unique hadm_ids with ICU: {len(icu_agg)}")

    # --- LEFT join onto cohort (keeps ALL admissions, ICU or not). ---
    merged = cohort.merge(icu_agg, on="hadm_id", how="left")

    # had_icu_stay flag: 1 if this admission appears in icustays, else 0.
    merged["had_icu_stay"] = merged["n_icu_stays"].notna().astype(int)
    # was_readmitted_to_icu: 1 if >1 ICU stay this admission, else 0.
    merged["was_readmitted_to_icu"] = (merged["n_icu_stays"].fillna(0) > 1).astype(int)

    # hours_from_admit_to_first_icu: time on the ward before first ICU entry.
    # WINDOW-DEPENDENT leakage: knowable only if prediction horizon is AT/AFTER
    # first ICU entry. Built here; gate at modeling time.
    merged["hours_from_admit_to_first_icu"] = (
        (merged["icu_first_intime"] - merged["admittime"]).dt.total_seconds() / 3600.0
    )

    # Structural-missing fill: for non-ICU admissions, counts = 0 ("no exposure").
    # The had_icu_stay flag lets the model tell a real 0 from "not applicable".
    for col in ["n_icu_stays", "n_distinct_careunits"]:
        merged[col] = merged[col].fillna(0)
    # icu_first_intime / first_careunit / hours_from_admit_to_first_icu left NaN
    # for non-ICU admissions (no defensible 0 exists for a timestamp/unit).

    n_icu = int(merged["had_icu_stay"].sum())
    log(f"  admissions flagged had_icu_stay=1: {n_icu} ({n_icu/len(merged):.1%})")

    out_path = OUT_ROOT / f"{STEP_NAME}.parquet"
    merged.to_parquet(out_path, index=False)
    log(f"  wrote -> {out_path}  shape={merged.shape}")

    manifest = {
        "step": STEP_NAME,
        "run_at": dt.datetime.now().isoformat(timespec="seconds"),
        "inputs": {"cohort": str(in_path), "icustays": str(icu_path)},
        "output": str(out_path),
        "rows_in": int(len(cohort)),
        "rows_out": int(len(merged)),       # unchanged: left join, no row growth
        "admissions_with_icu": n_icu,
        "icu_features_added": [
            "had_icu_stay", "n_icu_stays", "was_readmitted_to_icu",
            "icu_first_intime", "first_careunit", "n_distinct_careunits",
            "hours_from_admit_to_first_icu",
        ],
        "leakage_tags": {
            "had_icu_stay": "SAFE-ish (whether ICU happened; usable)",
            "n_icu_stays": "SAFE (count, structural)",
            "was_readmitted_to_icu": "SAFE (derived from count)",
            "n_distinct_careunits": "SAFE (structural)",
            "first_careunit": "SAFE-ish (known at/after first ICU entry)",
            "icu_first_intime": "SAFE (timestamp; needed to define windows)",
            "hours_from_admit_to_first_icu": "WINDOW-DEPENDENT (safe iff horizon >= first ICU entry)",
        },
        "deferred_leaky_features": [
            "icu_los_sum/mean/max/min/std (duration -> overlaps target)",
            "icu_last_outtime / any ICU-exit info",
            "(return later BOUNDED by the prediction window, non-leaky then)",
        ],
        "do_not_feed_to_model": {
            "in_hospital_death": "LEAKY (derived from deathtime; outcome from the future). Keep for cohort analysis/stratification only.",
            "los_days": "this is the TARGET, not a feature",
            "dod": "date of death; future information",
        },
        "fill_policy": "counts filled 0 for non-ICU; timestamps/careunit/hours left NaN",
        "columns_out": list(merged.columns),
        "notes": (
            "Only icustays used. Event tables deferred. LEFT join preserves all "
            "admissions. Leaky features built but tagged; gate at modeling time. "
            "Raw data untouched."
        ),
    }
    (OUT_ROOT / f"{STEP_NAME}__manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"  wrote manifest")
    log("DONE.")

if __name__ == "__main__":
    main()
