#!/usr/bin/env python3
"""
Diagnostic only - reads cohort + icustays, reports:
  1. how many PATIENTS (subject_id) have >1 hospital admission (hadm_id)
  2. how many ADMISSIONS (hadm_id) have >1 ICU stay (stay_id)
Writes nothing.
"""
from pathlib import Path
import pandas as pd

RAW_ROOT = Path("/home/pierrectrd/LoS project/mimic-iv-3.1")
OUT_ROOT = Path("/home/pierrectrd/LoS project/data")

# --- 1. Multiple hospital admissions per patient ---
cohort = pd.read_parquet(OUT_ROOT / "02_cohort_features.parquet")
adm_per_patient = cohort.groupby("subject_id")["hadm_id"].nunique()

n_patients = len(adm_per_patient)
n_multi_adm = int((adm_per_patient > 1).sum())
print("=== 1. Hospital admissions per patient ===")
print(f"total unique patients:                 {n_patients}")
print(f"patients with >1 hospital admission:   {n_multi_adm}  "
      f"({n_multi_adm/n_patients:.1%})")
print(f"patients with exactly 1 admission:     {n_patients - n_multi_adm}  "
      f"({(n_patients - n_multi_adm)/n_patients:.1%})")
print(f"max admissions for a single patient:   {int(adm_per_patient.max())}")
print(f"mean admissions per patient:           {adm_per_patient.mean():.2f}")
print()
print("distribution (admissions per patient, capped view):")
print(adm_per_patient.value_counts().sort_index().head(10).to_string())
print()

# --- 2. Multiple ICU stays per hospital admission ---
icu = pd.read_csv(RAW_ROOT / "icu" / "icustays.csv.gz",
                  dtype=str, usecols=["hadm_id", "stay_id"])
stays_per_hadm = icu.groupby("hadm_id")["stay_id"].nunique()

n_hadm_with_icu = len(stays_per_hadm)
n_hadm_multi_icu = int((stays_per_hadm > 1).sum())
n_total_hadm = cohort["hadm_id"].nunique()

print("=== 2. ICU stays per hospital admission ===")
print(f"total hospital admissions (cohort):        {n_total_hadm}")
print(f"admissions with >=1 ICU stay:              {n_hadm_with_icu}  "
      f"({n_hadm_with_icu/n_total_hadm:.1%} of all admissions)")
print(f"admissions with >1 ICU stay:               {n_hadm_multi_icu}")
print(f"  - as % of all admissions:                {n_hadm_multi_icu/n_total_hadm:.2%}")
print(f"  - as % of ICU admissions only:           {n_hadm_multi_icu/n_hadm_with_icu:.2%}")
print(f"max ICU stays for a single admission:      {int(stays_per_hadm.max())}")
print()
print("distribution (ICU stays per ICU admission):")
print(stays_per_hadm.value_counts().sort_index().head(10).to_string())
