#!/usr/bin/env python3
"""
Diagnostic only - distribution of ICU stay LENGTHS (icustays.los, in days).
Helps decide: is a 'first 24h' observation window viable, or does it discard
too many short stays? Writes nothing.
"""
from pathlib import Path
import pandas as pd

RAW_ROOT = Path("/home/pierrectrd/LoS project/mimic-iv-3.1")

icu = pd.read_csv(RAW_ROOT / "icu" / "icustays.csv.gz",
                  dtype=str, usecols=["los"])
los = pd.to_numeric(icu["los"], errors="coerce").dropna()
hours = los * 24.0
n = len(los)

print(f"total ICU stays (non-null los): {n}\n")
print("ICU LoS in DAYS - summary:")
print(los.describe(percentiles=[.1, .25, .5, .75, .9, .95, .99]).to_string())
print()

print("Share of stays SHORTER than a candidate observation window:")
for h in [6, 12, 18, 24, 36, 48, 72]:
    k = int((hours < h).sum())
    print(f"  < {h:>2}h : {k:>7}  ({k/n:.1%})")
print()

print("los >= 24h (MedM2T-style)")
keep = int((hours >= 24).sum())
print(f"  {keep} of {n} stays  ({keep/n:.1%})")
print()

print("ICU LoS day-bin distribution:")
bins = [0, 0.5, 1, 2, 3, 5, 7, 10, 14, 21, 1000]
labels = ["<0.5","0.5-1","1-2","2-3","3-5","5-7","7-10","10-14","14-21","21+"]
print(pd.cut(los, bins=bins, labels=labels, right=False)
      .value_counts().sort_index().to_string())
