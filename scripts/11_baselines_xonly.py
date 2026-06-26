#!/usr/bin/env python3
"""
Step 11 - X-only baselines (Case 3 reference): demographics -> remaining ICU LoS.

WHAT: predict remaining_los_days from 7 structured demographic features only,
with LinearRegression and a small-grid RandomForest, to set the bar the LLM
embedding (Case 1) must beat. MAE reported in REAL DAYS (Box-Cox inverted).

WHY THIS WAY: target is right-skewed -> Box-Cox (lambda fit on TRAIN ONLY).
Everything fitted (lambda, one-hot categories, models) is fit on train and
applied unchanged to val/test = leakage discipline (CLAUDE.md s3). The shared
machinery (assemble, one-hot, Box-Cox, MAE-in-days, RF grid) lives in _common.py.

DELIBERATELY DOES NOT:
  - use clinical_text / embeddings (that is Case 1/4, a separate script);
  - merge the full 05a frame (it carries icu_los_days, outtime, window_*,
    in_hospital_death, last_careunit = outcome/leaky; we pull 8 cols only);
  - impute age (spec: pass numeric through) or scale features (LR/RF don't
    need scaling here);
  - tune LinearRegression, or refit RF on train+val (fit-on-train only);
  - write a parquet (no per-row predictions requested).

Inputs : data/09b_los_strictly_positive.parquet (target + split, 51,704 rows)
         data/05a_target.parquet (source of the demographic columns)
Output : data/11_baselines_xonly__manifest.json  (no parquet)
"""
import json
import argparse
import datetime as dt
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

from _common import (OUT_ROOT, MEDM2T_MAE, NUM_COL, CAT_COLS, DEMO_COLS,
                     log, assemble, split_parts, make_demo, boxcox_target,
                     mae_real, best_rf_by_val, synthetic_demographics)

TARGET_NAME = "09b_los_strictly_positive"
DEMO_NAME   = "05a_target"
STEP_NAME   = "11_baselines_xonly"

def evaluate(df):
    """Full leakage-ordered pipeline on an assembled frame. Returns a result dict."""
    parts = split_parts(df)
    assert parts["train"][NUM_COL].notna().all(), f"{NUM_COL} has NaN in train (spec: pass-through)"

    # --- target: Box-Cox, lambda from TRAIN ONLY, applied to val/test ---
    y_t, y_real, lam = boxcox_target(parts)

    # --- features: OneHot fit on TRAIN ONLY ---
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X = {"train": make_demo(parts["train"], ohe, fit=True)}
    X["val"]  = make_demo(parts["val"],  ohe)
    X["test"] = make_demo(parts["test"], ohe)
    feat_names = [NUM_COL] + list(ohe.get_feature_names_out(CAT_COLS))

    # --- LinearRegression ---
    lr = LinearRegression().fit(X["train"], y_t["train"])
    lr_mae = {s: mae_real(lr, X[s], y_real[s], lam) for s in parts}

    # --- RandomForest: pick (n_estimators, max_depth) by best val MAE-in-days ---
    rf, n_est, depth = best_rf_by_val(X, y_t["train"], y_real, lam)
    rf_mae = {s: mae_real(rf, X[s], y_real[s], lam) for s in parts}

    return {
        "boxcox_lambda": float(lam),
        "rf_best": {"n_estimators": n_est, "max_depth": depth},
        "n_features_after_onehot": len(feat_names),
        "feature_list": feat_names,
        "rows_per_split": {s: int(len(parts[s])) for s in parts},
        "mae_real_days": {"LinearRegression": lr_mae, "RandomForest": rf_mae},
    }

# ---------------------------- the one runnable check ----------------------------

def selftest():
    """Tiny synthetic frames matching the real schema; asserts the logic holds."""
    rng = np.random.default_rng(0)
    n = 90
    sid = [f"s{i}" for i in range(n)]
    split = (["train"] * 50) + (["val"] * 20) + (["test"] * 20)
    df09b = pd.DataFrame({
        "stay_id": sid, "subject_id": sid, "split": split,
        "remaining_los_days": np.round(rng.exponential(2.0, n) + 0.01, 2),  # >0, skewed
        "clinical_text": ["..."] * n,
    })
    df05a = synthetic_demographics(rng, sid, careunits=("MICU", "SICU", "CCU"),
                                   races=("WHITE", "BLACK", "ASIAN"),
                                   extra_leaky=("window_start",))
    # unseen category in val/test exercises handle_unknown='ignore'
    df05a.loc[df05a["stay_id"] == "s60", "first_careunit"] = "NICU"

    asm = assemble(df09b, df05a)
    assert list(asm.columns) == list(df09b.columns) + DEMO_COLS[1:], "wrong assembled columns"
    assert not {"icu_los_days", "outtime", "in_hospital_death",
                "last_careunit", "window_start"} & set(asm.columns), "leaky column leaked in!"
    assert len(asm) == len(df09b)

    res = evaluate(asm)
    assert isinstance(res["boxcox_lambda"], float)
    assert res["n_features_after_onehot"] >= 1
    for m in res["mae_real_days"].values():
        assert all(np.isfinite(v) and v >= 0 for v in m.values()), "bad MAE"
    log("selftest OK (assemble drops leaky cols, pipeline runs, MAEs finite)")

# --------------------------------- real run ---------------------------------

def main(max_stays=None):
    selftest()  # run the check before touching real data

    df09b = pd.read_parquet(OUT_ROOT / f"{TARGET_NAME}.parquet")
    df05a = pd.read_parquet(OUT_ROOT / f"{DEMO_NAME}.parquet")
    df = assemble(df09b, df05a)
    if max_stays:
        # cap to the first N stays, in 09b row order, so the cohort matches step 12
        df = df.head(max_stays)
    log(f"assembled feature frame: {df.shape}" + (f" (capped to {max_stays})" if max_stays else ""))

    res = evaluate(df)

    # --- print MAE table (real days) ---
    print("\nMAE in real days (remaining ICU LoS)")
    print(f"{'model':<18}{'train':>9}{'val':>9}{'test':>9}")
    for name, m in res["mae_real_days"].items():
        print(f"{name:<18}{m['train']:>9.3f}{m['val']:>9.3f}{m['test']:>9.3f}")
    print(f"\nMedM2T benchmark (full multimodal): {MEDM2T_MAE} days")
    print(f"RF chosen: {res['rf_best']} | Box-Cox lambda: {res['boxcox_lambda']:.4f} "
          f"| features: {res['n_features_after_onehot']}")

    manifest = {
        "step": STEP_NAME,
        "run_at": dt.datetime.now().isoformat(timespec="seconds"),
        "inputs": {"target_split": str(OUT_ROOT / f"{TARGET_NAME}.parquet"),
                   "demographics": str(OUT_ROOT / f"{DEMO_NAME}.parquet")},
        "case": "Case 3 reference: X-only (7 demographics) -> remaining_los_days",
        "max_stays": max_stays,
        "features_used": DEMO_COLS[1:],
        "medm2t_benchmark_mae_days": MEDM2T_MAE,
        **res,
        "leakage_note": "Box-Cox lambda + OneHot categories fit on TRAIN ONLY, "
                        "applied to val/test. No clinical_text/leaky 05a columns used.",
    }
    (OUT_ROOT / f"{STEP_NAME}__manifest.json").write_text(json.dumps(manifest, indent=2))
    log("wrote manifest"); log("DONE.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-stays", type=int, default=None,
                    help="fit/evaluate on the first N stays only (default: all; use 10000 to match step 12)")
    args = ap.parse_args()
    main(max_stays=args.max_stays)
