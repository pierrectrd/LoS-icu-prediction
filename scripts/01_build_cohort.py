#!/usr/bin/env python3
"""

ONE SENTENCE SUMMARY: this is just a admissions left join on patient (roughly) 

Step 01 - Build the cohort table for hospital Length-of-Stay (LoS) prediction.

WHAT THIS DOES (and only this):
  - Reads admissions + patients from the RAW MIMIC-IV hosp module (read-only).
  - Parses timestamps, computes the regression target `los_days` (float days).
  - Adds informative flags (in_hospital_death, invalid_los) WITHOUT deleting
    clinically valid rows.
  - Removes ONLY genuinely invalid rows (null / non-positive LoS = entry errors),
    and ARCHIVES those removed rows so the step is reversible.
  - Keeps EVERY original column. No feature selection happens here.
  - Writes a versioned Parquet output + a JSON manifest describing the run.

WHAT THIS DOES NOT DO (deliberately, for later steps):
  - No dropping of "irrelevant" columns (language, etc.) -> feature stage.
  - No removal of in-hospital deaths -> that is a modeling choice, flagged here.
  - No 24-hour feature window -> that belongs to the feature-extraction stage.
  - No train/test split.

The RAW folder is never written to. All outputs go to a separate data folder.
"""

from pathlib import Path
import json
import datetime as dt
import pandas as pd

# ----------------------------------------------------------------------
# CONFIG -- edit these two paths only.
# (WSL paths. From inside WSL these are ordinary Linux paths.)
# ----------------------------------------------------------------------
RAW_ROOT = Path("/home/pierrectrd/LoS project/mimic-iv-3.1")
OUT_ROOT = Path("/home/pierrectrd/LoS project/data")

HOSP = RAW_ROOT / "hosp"
STEP_NAME = "01_cohort"

# ----------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}")

def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    archive_dir = OUT_ROOT / "_dropped_rows"
    archive_dir.mkdir(parents=True, exist_ok=True)

    # --- Load. admissions is small (~430k rows) so a full read is fine. ---
    adm_path = HOSP / "admissions.csv.gz"
    pat_path = HOSP / "patients.csv.gz"
    log(f"Reading {adm_path}")
    # Read timestamps as plain strings first, then parse explicitly. This avoids
    # surprises from pandas guessing types on a messy file.
    adm = pd.read_csv(adm_path, dtype=str)
    log(f"  admissions raw shape: {adm.shape}")

    log(f"Reading {pat_path}")
    pat = pd.read_csv(pat_path, dtype=str)
    log(f"  patients raw shape:   {pat.shape}")

    n_raw = len(adm)

    # --- Parse the timestamps we need for the target. ---
    for col in ["admittime", "dischtime", "deathtime"]:
        adm[col] = pd.to_datetime(adm[col], errors="coerce")

    # --- Compute the target: hospital LoS in (fractional) days. ---
    adm["los_days"] = (adm["dischtime"] - adm["admittime"]).dt.total_seconds() / 86400.0

    # --- Add flags (information, not deletion). ---
    adm["in_hospital_death"] = adm["deathtime"].notna().astype(int)
    adm["invalid_los"] = (adm["los_days"].isna() | (adm["los_days"] <= 0)).astype(int)

    # --- Attach a couple of demographics from patients (age, gender). ---
    # patients has one row per subject_id; anchor_age is age at anchor_year.
    adm = adm.merge(pat, on="subject_id", how="left")
    log(f"  merged patient demographics: basically admissions LEFT JOIN patients ON subject.id")

    # --- Remove ONLY invalid rows, and archive them. ---
    invalid_mask = adm["invalid_los"] == 1
    dropped = adm[invalid_mask].copy()
    cohort = adm[~invalid_mask].copy()
    log(f"  invalid LoS rows removed: {len(dropped)} "
        f"({len(dropped)/n_raw:.2%} of raw)")

    # Archive dropped rows so removal is reversible / inspectable.
    dropped_path = archive_dir / f"{STEP_NAME}__invalid_los.parquet"
    dropped.to_parquet(dropped_path, index=False)
    log(f"  archived dropped rows -> {dropped_path}")

    # --- Write the versioned output. ---
    out_path = OUT_ROOT / f"{STEP_NAME}.parquet"
    cohort.to_parquet(out_path, index=False)
    log(f"  wrote cohort -> {out_path}  shape={cohort.shape}")

    # --- Write a manifest describing exactly what happened. ---
    manifest = {
        "step": STEP_NAME,
        "run_at": dt.datetime.now().isoformat(timespec="seconds"),
        "inputs": {"admissions": str(adm_path), "patients": str(pat_path)},
        "output": str(out_path),
        "dropped_rows_archive": str(dropped_path),
        "rows_in_raw_admissions": n_raw,
        "rows_dropped_invalid_los": int(len(dropped)),
        "rows_out": int(len(cohort)),
        "target_column": "los_days",
        "target_units": "days (float, fractional)",
        "flags_added": ["in_hospital_death", "invalid_los"],
        "columns_out": list(cohort.columns),
        "los_summary_days": {
            "mean": float(cohort["los_days"].mean()),
            "median": float(cohort["los_days"].median()),
            "p95": float(cohort["los_days"].quantile(0.95)),
            "max": float(cohort["los_days"].max()),
        },
        "in_hospital_death_count": int(cohort["in_hospital_death"].sum()),
        "notes": (
            "No feature columns dropped. In-hospital deaths KEPT and flagged. "
            "No 24h window applied. No split applied. Raw data not modified."
        ),
    }
    manifest_path = OUT_ROOT / f"{STEP_NAME}__manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log(f"  wrote manifest -> {manifest_path}")
    log("DONE.")

if __name__ == "__main__":
    main()
