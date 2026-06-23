#!/usr/bin/env python3
"""
Diagnostic only - explain WHY some ICU stays don't match a cohort admission.
Classifies unmatched icustays.hadm_id into:
  (a) null hadm_id
  (b) hadm_id present in RAW admissions but dropped by our step 1/2 hygiene
  (c) hadm_id not in RAW admissions at all (genuinely orphaned)
Writes nothing.
"""
from pathlib import Path
import pandas as pd

RAW_ROOT = Path("/home/pierrectrd/LoS project/mimic-iv-3.1")
OUT_ROOT = Path("/home/pierrectrd/LoS project/data")

# ICU stays (keys only)
icu = pd.read_csv(RAW_ROOT / "icu" / "icustays.csv.gz",
                  dtype=str, usecols=["hadm_id", "stay_id"])

# Our current cohort hadm_ids (post step 1+2)
cohort = pd.read_parquet(OUT_ROOT / "02_cohort_features.parquet")
cohort_hadm = set(cohort["hadm_id"].astype(str))

# Raw admissions hadm_ids (what existed BEFORE our hygiene)
raw_adm = pd.read_csv(RAW_ROOT / "hosp" / "admissions.csv.gz",
                      dtype=str, usecols=["hadm_id"])
raw_hadm = set(raw_adm["hadm_id"].astype(str))

# Unmatched = ICU hadm not in our cohort
icu["in_cohort"] = icu["hadm_id"].isin(cohort_hadm)
unmatched = icu[~icu["in_cohort"]].copy()
n_un = len(unmatched)

print(f"total ICU stays:                       {len(icu)}")
print(f"ICU stays NOT matching cohort hadm:    {n_un}")
print(f"  distinct unmatched hadm_ids:         {unmatched['hadm_id'].nunique(dropna=True)}")
print()

# Classify
null_hadm = unmatched["hadm_id"].isna().sum()
present_in_raw = unmatched["hadm_id"].isin(raw_hadm) & unmatched["hadm_id"].notna()
n_present_raw = int(present_in_raw.sum())          # -> dropped by OUR hygiene
n_orphan = int((~unmatched["hadm_id"].isin(raw_hadm) & unmatched["hadm_id"].notna()).sum())

print("Why are they unmatched?")
print(f"  (a) null hadm_id in icustays:                       {int(null_hadm)}")
print(f"  (b) hadm in RAW admissions but dropped by step1/2:  {n_present_raw} ")
print(f"  (c) hadm NOT in raw admissions (true orphan):       {n_orphan}")
