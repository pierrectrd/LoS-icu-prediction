#!/usr/bin/env python3
"""
Step 07 - Patient-level train/val/test split.

WHY patient-level: 44.8% of patients have repeat admissions/stays. A row-level
split would put the same patient in train AND test, letting the model memorise
individuals -> inflated, dishonest scores. We split by subject_id so every stay
of a patient lands in ONE fold. (MedM2T calls this cross-subject partitioning.)

Ratios: 0.64 / 0.16 / 0.20 (train/val/test), matching MedM2T. Fixed seed for
reproducibility. We split groups (patients), not rows.

CRITICAL DOWNSTREAM RULE (director's feedback 4): everything "fitted" -- the
text-rendering recipe, Box-Cox lambda, imputation values, one-hot categories,
scaling -- must be decided on the TRAIN fold ONLY, then applied unchanged to
val/test. This file just assigns the folds; it learns nothing.

INPUT : 05a_target.parquet   (base cohort + target; one row per stay)
OUTPUT: 07_split.parquet (+ manifest)  -- same rows + a `split` column
        plus 07_split__train_subjects.parquet etc. as explicit id lists.
"""
from pathlib import Path
import json
import datetime as dt
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

OUT_ROOT = Path("/home/pierrectrd/LoS project/data")
IN_NAME = "05a_target"
STEP_NAME = "07_split"
SEED = 42
TEST_FRAC = 0.20
VAL_FRAC_OF_REMAINDER = 0.20   # 0.20 of the 0.80 remainder = 0.16 overall

def log(msg): print(f"[{dt.datetime.now():%H:%M:%S}] {msg}")

def main():
    df = pd.read_parquet(OUT_ROOT / f"{IN_NAME}.parquet")
    df["subject_id"] = df["subject_id"].astype(str)
    n = len(df)
    log(f"cohort: {n} stays, {df['subject_id'].nunique()} unique patients")

    groups = df["subject_id"].values

    # 1) carve TEST off the full set, grouped by patient
    gss1 = GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=SEED)
    trainval_idx, test_idx = next(gss1.split(df, groups=groups))

    # 2) carve VAL off the remaining train+val, again grouped by patient
    tv = df.iloc[trainval_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=VAL_FRAC_OF_REMAINDER, random_state=SEED)
    tr_rel, val_rel = next(gss2.split(tv, groups=tv["subject_id"].values))
    train_idx = tv.iloc[tr_rel].index
    val_idx   = tv.iloc[val_rel].index
    test_idx_abs = df.iloc[test_idx].index

    df["split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx_abs, "split"] = "test"

    # --- VERIFY zero patient overlap across folds (the whole point) ---
    sets = {s: set(df.loc[df["split"] == s, "subject_id"]) for s in ["train","val","test"]}
    overlap_tv = sets["train"] & sets["val"]
    overlap_tt = sets["train"] & sets["test"]
    overlap_vt = sets["val"] & sets["test"]
    assert not overlap_tv and not overlap_tt and not overlap_vt, \
        f"PATIENT OVERLAP DETECTED: tv={len(overlap_tv)} tt={len(overlap_tt)} vt={len(overlap_vt)}"
    log("verified: ZERO patient overlap across train/val/test")

    counts = df["split"].value_counts()
    log(f"stays  -> train {counts.get('train',0)}, val {counts.get('val',0)}, test {counts.get('test',0)}")
    log(f"patients-> train {len(sets['train'])}, val {len(sets['val'])}, test {len(sets['test'])}")

    out_path = OUT_ROOT / f"{STEP_NAME}.parquet"
    df.to_parquet(out_path, index=False)
    # explicit id lists, handy for downstream train-only fitting
    for s in ["train","val","test"]:
        pd.DataFrame({"subject_id": sorted(sets[s])}).to_parquet(
            OUT_ROOT / f"{STEP_NAME}__{s}_subjects.parquet", index=False)
    log(f"wrote -> {out_path}")

    manifest = {
        "step": STEP_NAME,
        "run_at": dt.datetime.now().isoformat(timespec="seconds"),
        "input": str(OUT_ROOT / f"{IN_NAME}.parquet"),
        "output": str(out_path),
        "method": "GroupShuffleSplit on subject_id (patient-level)",
        "ratios": {"train": 0.64, "val": 0.16, "test": 0.20},
        "seed": SEED,
        "stays": {s: int(counts.get(s, 0)) for s in ["train","val","test"]},
        "patients": {s: len(sets[s]) for s in ["train","val","test"]},
        "patient_overlap_verified_zero": True,
        "downstream_rule": "Fit EVERYTHING (text recipe, Box-Cox lambda, imputation, one-hot, scaling) on TRAIN ONLY; apply unchanged to val/test.",
        "id_lists": {s: str(OUT_ROOT / f"{STEP_NAME}__{s}_subjects.parquet") for s in ["train","val","test"]},
    }
    (OUT_ROOT / f"{STEP_NAME}__manifest.json").write_text(json.dumps(manifest, indent=2))
    log("wrote manifest"); log("DONE.")

if __name__ == "__main__":
    main()
