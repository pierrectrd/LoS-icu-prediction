#!/usr/bin/env python3
"""
peek.py - quickly inspect a parquet file without loading it all into RAM.

Usage:
    python3 peek.py 03_icu_stays_clean.parquet
    python3 peek.py 03_icu_stays_clean.parquet 10      # show 10 rows
    python3 peek.py /full/path/to/file.parquet

Prints: row count, column count, schema (names + types), and the first N rows.
"""
from pathlib import Path
import sys
import pyarrow.parquet as pq

OUT_ROOT = Path("/home/pierrectrd/LoS project/data")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 peek.py <file.parquet> [n_rows]")
        sys.exit(1)

    arg = sys.argv[1]
    n_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    # Accept either a bare filename (resolved against OUT_ROOT) or a full path.
    path = Path(arg)
    if not path.exists():
        path = OUT_ROOT / arg
    if not path.exists():
        print(f"File not found: {arg}")
        sys.exit(1)

    pf = pq.ParquetFile(path)
    print(f"file: {path}")
    print(f"rows: {pf.metadata.num_rows}   columns: {pf.metadata.num_columns}")
    print("\nschema (column : type):")
    for field in pf.schema_arrow:
        print(f"  {field.name} : {field.type}")

    # Read only the first batch -> first rows, without loading the whole file.
    print(f"\nfirst {n_rows} rows:")
    batch = next(pf.iter_batches(batch_size=n_rows))
    print(batch.to_pandas().to_string(index=False))

if __name__ == "__main__":
    main()
