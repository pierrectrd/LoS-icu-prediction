#!/usr/bin/env python3
"""
Step 03c - Integrity cleaning after the ICU joins.

Does three things, all reversible:
  1. ASSERTS key uniqueness:
       - 02_cohort_features: one row per hadm_id
       - 03_cohort_with_icu: one row per hadm_id (join did not multiply rows)
       - 03_icu_stays:       one row per stay_id
  2. Drops from 03_icu_stays the ICU stays whose hadm_id is NOT in the cohort
     (the 81 bucket-(b) stays whose admissions we removed in step 1/2). They
     currently sit in 03b with NaN demographics. Archived before removal.
  3. Confirms 03_cohort_with_icu needs NO drop (left join never admitted them).

OUTPUT: 03_icu_stays_clean.parquet (+ manifest). 03a output is re-validated
        but unchanged, so no new file is written for it.
"""
from pathlib import Path
import json
import datetime as dt
import pandas as pd

OUT_ROOT = Path("/home/pierrectrd/LoS project/data")
STEP_NAME = "03c_integrity"

def log(msg): print(f"[{dt.datetime.now():%H:%M:%S}] {msg}")

def main():
    cohort = pd.read_parquet(OUT_ROOT / "02_cohort_features.parquet")
    icu_adm = pd.read_parquet(OUT_ROOT / "03_cohort_with_icu.parquet")
    stays = pd.read_parquet(OUT_ROOT / "03_icu_stays.parquet")

    # --- 1. Uniqueness asserts ---
    assert cohort["hadm_id"].is_unique, "02_cohort_features has duplicate hadm_id!"
    assert icu_adm["hadm_id"].is_unique, "03_cohort_with_icu has duplicate hadm_id (join multiplied rows)!"
    assert stays["stay_id"].is_unique, "03_icu_stays has duplicate stay_id!"
    log("uniqueness asserts PASSED (hadm_id unique in cohort & 03a; stay_id unique in 03b)")

    cohort_hadm = set(cohort["hadm_id"].astype(str))

    # --- 2. Drop unmatched ICU stays from 03b ---
    stays["hadm_id"] = stays["hadm_id"].astype(str)
    unmatched_mask = ~stays["hadm_id"].isin(cohort_hadm)
    n_unmatched = int(unmatched_mask.sum())
    archive_dir = OUT_ROOT / "_dropped_rows"
    archive_dir.mkdir(parents=True, exist_ok=True)
    if n_unmatched:
        stays[unmatched_mask].to_parquet(
            archive_dir / f"{STEP_NAME}__icu_stays_unmatched_hadm.parquet", index=False)
    stays_clean = stays[~unmatched_mask].copy()
    log(f"03b: dropped {n_unmatched} ICU stays whose hadm_id not in cohort "
        f"(archived). rows {len(stays)} -> {len(stays_clean)}")

    # --- 3. Confirm 03a unaffected ---
    icu_adm_hadm_in_cohort = icu_adm["hadm_id"].isin(cohort_hadm).all()
    log(f"03a: every hadm_id already in cohort? {icu_adm_hadm_in_cohort} (expected True)")

    out_path = OUT_ROOT / "03_icu_stays_clean.parquet"
    stays_clean.to_parquet(out_path, index=False)
    log(f"wrote -> {out_path}  shape={stays_clean.shape}")

    manifest = {
        "step": STEP_NAME,
        "run_at": dt.datetime.now().isoformat(timespec="seconds"),
        "asserts_passed": ["cohort hadm_id unique", "03a hadm_id unique", "03b stay_id unique"],
        "icu_stays_in": int(len(stays)),
        "icu_stays_unmatched_dropped": n_unmatched,
        "icu_stays_out": int(len(stays_clean)),
        "unmatched_archive": str(archive_dir / f"{STEP_NAME}__icu_stays_unmatched_hadm.parquet"),
        "03a_needed_drop": (not icu_adm_hadm_in_cohort),
        "output": str(out_path),
        "notes": (
            "All unmatched ICU stays were bucket-(b): belonged to admissions "
            "removed in step1/2. 03a unchanged (left join never admitted them). "
            "Raw data untouched."
        ),
    }
    (OUT_ROOT / f"{STEP_NAME}__manifest.json").write_text(json.dumps(manifest, indent=2))
    log("wrote manifest")
    log("DONE.")

if __name__ == "__main__":
    main()
