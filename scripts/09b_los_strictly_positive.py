#!/usr/bin/env python3
"""
Step 09b - Carve the LoS-task modelling cohort: strictly positive remaining LoS
AND survivors only (in-hospital deaths removed).

Two filters, both belonging to the LoS task specifically (NOT to cohort
construction, which is shared with the planned mortality task):

  1. STRICTLY POSITIVE: the >=24h cohort + (los - 1) retarget (05a) leaves 130
     stays at exactly remaining_los_days == 0 (discharged at ~24h). A zero target
     is legitimate but awkward for Box-Cox and ambiguous for "remaining" LoS.

  2. SURVIVORS ONLY (in_hospital_death == 0): a patient who dies in hospital has a
     length of stay truncated by death -- time-to-death, not time-to-discharge.
     LoS and mortality are SEPARATE tasks (CLAUDE.md s1, s4); the deaths are the
     mortality cohort and must not contaminate LoS training/eval. The flag is not
     carried in 09's text frame, so we re-join it from 05a on stay_id. This is the
     single source-of-truth exclusion: every downstream LoS step (11 reads 09b;
     12 reads 09b then 13/14/14b/16 read 12's parquet) inherits a death-free cohort.

Earlier this exclusion was only ever a FEATURE leak-guard (the death flag was kept
out of X), never a ROW filter -- so ~10.8% of every prior LoS result was computed
on a cohort that still contained deaths. This step closes that hole at the source.

Both dropped sets are archived per project convention (reversible).

INPUT : 09_clinical_text_sofa.parquet  +  05a_target.parquet (death flag)
OUTPUT: 09b_los_strictly_positive.parquet (+ manifest); drops -> _dropped_rows/
"""
from pathlib import Path
import json
import datetime as dt
import pandas as pd

OUT_ROOT = Path("/home/pierrectrd/LoS project/data")
IN_NAME = "09_clinical_text_sofa"
FLAG_NAME = "05a_target"
STEP_NAME = "09b_los_strictly_positive"

def log(msg): print(f"[{dt.datetime.now():%H:%M:%S}] {msg}")

def main():
    in_path = OUT_ROOT / f"{IN_NAME}.parquet"
    df = pd.read_parquet(in_path)
    n_in = len(df)

    # re-join the in-hospital-death flag (not present in 09's text frame) on stay_id
    flag = pd.read_parquet(OUT_ROOT / f"{FLAG_NAME}.parquet")[["stay_id", "in_hospital_death"]].copy()
    df["_sid"] = df["stay_id"].astype(str)
    flag["_sid"] = flag["stay_id"].astype(str)
    df = df.merge(flag[["_sid", "in_hospital_death"]], on="_sid", how="left")
    assert df["in_hospital_death"].notna().all(), "a stay had no death flag in 05a (join gap)"
    df["in_hospital_death"] = df["in_hospital_death"].astype(int)

    positive = df["remaining_los_days"] > 0
    survived = df["in_hospital_death"] == 0
    keep = positive & survived

    # archive the two drop reasons separately (a zero-LoS death counts as zero-LoS)
    dropped_zero  = df[~positive].copy()
    dropped_death = df[positive & ~survived].copy()
    out = df[keep].drop(columns=["_sid", "in_hospital_death"]).copy()

    assert (out["remaining_los_days"] > 0).all(), "non-positive target survived the filter"

    archive = OUT_ROOT / "_dropped_rows"
    archive.mkdir(parents=True, exist_ok=True)
    dropped_zero.drop(columns="_sid").to_parquet(
        archive / f"{STEP_NAME}__zero_remaining_los.parquet", index=False)
    dropped_death.drop(columns="_sid").to_parquet(
        archive / f"{STEP_NAME}__in_hospital_death.parquet", index=False)

    out_path = OUT_ROOT / f"{STEP_NAME}.parquet"
    out.to_parquet(out_path, index=False)
    log(f"{n_in} in -> dropped {len(dropped_zero)} zero-LoS + {len(dropped_death)} deaths "
        f"-> {len(out)} survivors with positive LoS")

    manifest = {
        "step": STEP_NAME,
        "run_at": dt.datetime.now().isoformat(timespec="seconds"),
        "inputs": {"text": str(in_path), "death_flag": str(OUT_ROOT / f"{FLAG_NAME}.parquet")},
        "output": str(out_path),
        "dropped_archives": {
            "zero_remaining_los": str(archive / f"{STEP_NAME}__zero_remaining_los.parquet"),
            "in_hospital_death": str(archive / f"{STEP_NAME}__in_hospital_death.parquet"),
        },
        "rows_in": n_in,
        "rows_dropped_zero_remaining_los": int(len(dropped_zero)),
        "rows_dropped_in_hospital_death": int(len(dropped_death)),
        "rows_out": int(len(out)),
        "filter": "remaining_los_days > 0 AND in_hospital_death == 0",
        "notes": "LoS-task cohort = strictly-positive survivors. Deaths removed here "
                 "(not at cohort construction, which is shared with the mortality task). "
                 "Both drop reasons archived; reversible.",
    }
    (OUT_ROOT / f"{STEP_NAME}__manifest.json").write_text(json.dumps(manifest, indent=2))
    log("wrote manifest"); log("DONE.")

if __name__ == "__main__":
    main()
