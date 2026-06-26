#!/usr/bin/env python3
"""
showtext.py - print a text column legibly (line breaks rendered).

Usage:
    python3 showtext.py 06_clinical_text.parquet clinical_text          # first row
    python3 showtext.py 06_clinical_text.parquet clinical_text 3        # row index 3
    python3 showtext.py 06_clinical_text.parquet clinical_text 0 5      # rows 0..4
"""
from pathlib import Path
import sys
import pandas as pd

OUT_ROOT = Path("/home/pierrectrd/LoS project/data")

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 showtext.py <file.parquet> <column> [start] [end]")
        sys.exit(1)
    arg, col = sys.argv[1], sys.argv[2]
    start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    end = int(sys.argv[4]) if len(sys.argv) > 4 else start + 1

    path = Path(arg)
    if not path.exists():
        path = OUT_ROOT / arg
    df = pd.read_parquet(path)
    if col not in df.columns:
        print(f"Column '{col}' not found. Available: {list(df.columns)[:20]}...")
        sys.exit(1)

    for i in range(start, min(end, len(df))):
        row = df.iloc[i]
        # show a couple of id columns if present, for context
        ids = [f"{c}={row[c]}" for c in ("stay_id", "subject_id", "remaining_los_days")
               if c in df.columns]
        print(f"===== row {i}  {'  '.join(ids)} =====")
        print(row[col])
        print()

if __name__ == "__main__":
    main()
