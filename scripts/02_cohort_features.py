#!/usr/bin/env python3
"""
Step 02 - Derive admission-level fields.

INPUT : 01_cohort.parquet  (one row per valid hospital admission)
OUTPUT: 02_cohort_features.parquet  (+ manifest)

WHAT THIS DOES (and only this):
  - Drops the now-useless `invalid_los` flag (all rows are valid after step 01).
  - Drops leaky / no-longer-needed timestamps: dischtime, deathtime,
    edregtime, edouttime, hospital_expire_flag's redundant partners are KEPT
    only where informative (see notes).  IMPORTANT: `admittime` is KEPT,
    because age-at-admission and the future 24h window both depend on it.
  - Derives `age_at_admission` using the anchor correction:
        age_at_admission = anchor_age + (year(admittime) - anchor_year)
    Top: clipped to 91, the MIMIC sentinel for the HIPAA-permitted
      "age 90 or older" aggregated category.
    Bottom: rows with computed age < 18 are DROPPED (and archived) -- MIMIC
      is adults-only; these arise from negative anchor-year offsets.

WHAT THIS DOES NOT DO (deliberately):
  - No categorical dummy/one-hot encoding  -> must happen AFTER the train/test
    split to avoid leakage; done at the modeling stage.
  - No dropping of "irrelevant" feature columns (language, etc.).
  - No ICU join, no 24h window, no row filtering by stay length.
  - `admittime` is NOT dropped here; it is load-bearing for later steps.

Raw data is never touched. Output is a new versioned file; reversible by
deleting it (see backtrack.py).
"""

from pathlib import Path
import json
import datetime as dt
import pandas as pd

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
OUT_ROOT = Path("/home/pierrectrd/LoS project/data")
IN_NAME = "01_cohort"
STEP_NAME = "02_cohort_features"

# Timestamps that are leaky (computed-from for target) or otherwise not needed
# as features. NOTE: admittime is intentionally absent from this list.
DROP_COLS = ["dischtime", "deathtime", "edregtime", "edouttime", "invalid_los"]

# Upper bound: MIMIC stores all "90 or older" patients as anchor_age 91 (the
# HIPAA-permitted aggregated category). We do NOT clip to 90 (that would collide
# with genuine 90-year-olds); we clip the anchor-corrected value back to 91 so
# the censored group stays marked by its sentinel and inflation is undone.
AGE_SENTINEL_90PLUS = 91
# Lower bound: MIMIC is adults-only. Negative anchor-year offsets can compute an
# age < 18. These are artifacts AND under-age records we won't handle -> DROP them.
AGE_MIN = 18

# ----------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}")

def main() -> None:
    in_path = OUT_ROOT / f"{IN_NAME}.parquet"
    log(f"Reading {in_path}")
    df = pd.read_parquet(in_path)
    n_in = len(df)
    cols_in = list(df.columns)
    log(f"  input shape: {df.shape}")

    # --- Derive age at admission (anchor correction). ---
    # admittime is datetime (parsed in step 01 and preserved through parquet).
    # anchor_age / anchor_year arrived as strings (read with dtype=str); cast.
    df["anchor_age"] = pd.to_numeric(df["anchor_age"], errors="coerce")
    df["anchor_year"] = pd.to_numeric(df["anchor_year"], errors="coerce")
    admit_year = df["admittime"].dt.year

    raw_age = df["anchor_age"] + (admit_year - df["anchor_year"])
    # Undo anchor-offset inflation at the top, back to the 90+ sentinel (NOT 90).
    df["age_at_admission"] = raw_age.clip(upper=AGE_SENTINEL_90PLUS)

    n_clipped_high = int((raw_age > AGE_SENTINEL_90PLUS).sum())
    log(f"  age clipped down to {AGE_SENTINEL_90PLUS} (90+ sentinel): "
        f"{n_clipped_high} rows")

    # --- Drop (and archive) under-age records. ---
    under_mask = df["age_at_admission"] < AGE_MIN
    n_under = int(under_mask.sum())
    under_rows = df[under_mask].copy()
    df = df[~under_mask].copy()
    log(f"  dropped under-{AGE_MIN} rows: {n_under}")
    log(f"  age_at_admission range now: "
        f"{df['age_at_admission'].min():.0f}–{df['age_at_admission'].max():.0f}")

    archive_dir = OUT_ROOT / "_dropped_rows"
    archive_dir.mkdir(parents=True, exist_ok=True)
    under_path = archive_dir / f"{STEP_NAME}__under_{AGE_MIN}.parquet"
    under_rows.to_parquet(under_path, index=False)
    log(f"  archived under-age rows -> {under_path}")

    # --- Drop the leaky / unneeded columns that are present. ---
    present_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=present_drop)
    log(f"  dropped columns: {present_drop}")
    log(f"  KEPT admittime (needed for age + future 24h window)")

    # --- Write output. ---
    out_path = OUT_ROOT / f"{STEP_NAME}.parquet"
    df.to_parquet(out_path, index=False)
    log(f"  wrote -> {out_path}  shape={df.shape}")

    # --- Manifest. ---
    manifest = {
        "step": STEP_NAME,
        "run_at": dt.datetime.now().isoformat(timespec="seconds"),
        "input": str(in_path),
        "output": str(out_path),
        "rows_in": n_in,
        "rows_out": int(len(df)),
        "rows_dropped_under_18": n_under,
        "under_18_archive": str(under_path),
        "columns_dropped": present_drop,
        "columns_added": ["age_at_admission"],
        "admittime_kept": True,
        "age_min_kept": AGE_MIN,
        "age_sentinel_90plus": AGE_SENTINEL_90PLUS,
        "age_clipped_to_sentinel_count": n_clipped_high,
        "age_derivation": (
            "anchor_age + (year(admittime) - anchor_year); top clipped to 91 "
            "(HIPAA 90+ aggregated category, MIMIC sentinel); rows <18 dropped"
        ),
        "columns_in": cols_in,
        "columns_out": list(df.columns),
        "notes": (
            "Under-18 rows dropped and archived. Top age clipped to 91 sentinel "
            "(not 90). No dummy/one-hot encoding (must follow split). "
            "admittime intentionally retained. Raw data untouched."
        ),
    }
    manifest_path = OUT_ROOT / f"{STEP_NAME}__manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log(f"  wrote manifest -> {manifest_path}")
    log("DONE.")

if __name__ == "__main__":
    main()
