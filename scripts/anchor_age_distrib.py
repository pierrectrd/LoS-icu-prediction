#!/usr/bin/env python3
"""
Diagnostic only - reads 01_cohort.parquet, shows raw anchor_age distribution.
Writes nothing.
"""
from pathlib import Path
import pandas as pd

OUT_ROOT = Path("/home/pierrectrd/LoS project/data")
df = pd.read_parquet(OUT_ROOT / "01_cohort.parquet")

age = pd.to_numeric(df["anchor_age"], errors="coerce")

print(f"rows: {len(df)}")
print(f"non-null anchor_age: {age.notna().sum()}")
print()
print("summary:")
print(age.describe().to_string())
print()
print(f"count at exactly 91 (HIPAA over-89 censor): {(age == 91).sum()} "
      f"({(age == 91).mean():.2%})")
print(f"count below 18:                             {(age < 18).sum()}")
print()
print("distribution in 10-year bins:")
bins = list(range(0, 101, 10)) + [200]
print(pd.cut(age, bins=bins, right=False).value_counts().sort_index().to_string())
print()
print("note: anchor_age is per-PATIENT (constant across that patient's rows),")
print("so these counts are over admissions, not unique patients.")