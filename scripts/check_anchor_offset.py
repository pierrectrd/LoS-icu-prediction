#!/usr/bin/env python3
"""
Diagnostic only - reads 01_cohort.parquet, compares admit year vs anchor_year.
Writes nothing. Just prints.
"""
from pathlib import Path
import pandas as pd

OUT_ROOT = Path("/home/pierrectrd/LoS project/data")
df = pd.read_parquet(OUT_ROOT / "01_cohort.parquet")

df["anchor_year"] = pd.to_numeric(df["anchor_year"], errors="coerce")
admit_year = df["admittime"].dt.year
offset = admit_year - df["anchor_year"]          # 0 if same year

n = len(df)
n_diff = int((offset != 0).sum())
print(f"total admissions:                 {n}")
print(f"admit year != anchor_year:        {n_diff}  ({n_diff/n:.1%})")
print(f"admit year == anchor_year:        {n - n_diff}  ({(n-n_diff)/n:.1%})")
print()
print("offset (admit_year - anchor_year) distribution:")
print(offset.value_counts().sort_index().to_string())
print()
print(f"offset range: {int(offset.min())} to {int(offset.max())}")
print(f"unique patients affected (subject_id): "
      f"{df.loc[offset != 0, 'subject_id'].nunique()}")