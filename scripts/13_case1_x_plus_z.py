#!/usr/bin/env python3
"""
Step 13 - Case 1: predict remaining_los_days from demographics X + embedding Z.

WHAT: concatenate the 7 structured demographic features (X) with the 1536-dim
LLM answer-embedding (Z) and fit Ridge + RandomForest, to test whether X+Z beats
either modality alone -- X-only (step 11, RF test MAE ~2.59-2.67) and Z-alone
(step 12, ~2.57). This is the primary modelling case (CLAUDE.md s7).

WHY THIS WAY: target is right-skewed -> Box-Cox (lambda on TRAIN only). Z is
1536 wide columns -> Ridge needs regularization + standardized emb (scale-
sensitive); RF needs neither. Everything fitted (lambda, one-hot categories, emb
scaler, model hyperparameters) is fit on train and applied unchanged to val/test.
Shared machinery (assemble, one-hot, Box-Cox, MAE-in-days, RF grid) is in _common.py.

DELIBERATELY DOES NOT:
  - use llm_point_estimate / llm_answer from the base (that is step 12's job);
  - merge the full 05a frame (it carries icu_los_days, last_careunit, outtime,
    in_hospital_death = outcome/leaky); only 8 columns are pulled;
  - scale features for RF, or scale the one-hot / age columns for Ridge (spec:
    standardize the emb block only);
  - write a parquet (no per-row predictions requested).

NOTE ON SCALE: results on the ~1074-stay capped cohort are DIRECTIONAL ONLY --
the test fold is tiny and RF overfits at this size. The decisive X+Z-vs-baseline
verdict needs the full cohort; this script is the apparatus, not the answer.

Inputs : data/12_baseline_llm_float.parquet  (base: target, split, emb_0..1535)
         data/05a_target.parquet             (source of the 7 demographics)
Output : data/13_case1_x_plus_z__manifest.json  (no parquet)
"""
import json
import datetime as dt
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge

from _common import (OUT_ROOT, MEDM2T_MAE, NUM_COL, DEMO_COLS,
                     log, assemble, split_parts, make_demo, boxcox_target,
                     mae_real, best_rf_by_val, synthetic_demographics)

BASE_NAME = "12_baseline_llm_float"
DEMO_NAME = "05a_target"
STEP_NAME = "13_case1_x_plus_z"

RIDGE_ALPHAS = [1, 10, 100, 1000]

# reference bars for the printed table (days)
REF_XONLY_RF = "step 11 X-only RF ~2.59-2.67"
REF_Z_ALONE  = "step 12 Z-alone ~2.57"

def evaluate(df):
    """Full leakage-ordered pipeline on an assembled X+Z frame. Returns result dict."""
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    assert emb_cols, "no emb_* columns found in base"
    parts = split_parts(df)
    assert parts["train"][NUM_COL].notna().all(), f"{NUM_COL} has NaN in train"

    # --- target: Box-Cox, lambda from TRAIN ONLY ---
    y_t, y_real, lam = boxcox_target(parts)

    # --- demographics: OneHot fit on TRAIN ONLY (same block for both models) ---
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    demo = {"train": make_demo(parts["train"], ohe, fit=True)}
    demo["val"]  = make_demo(parts["val"],  ohe)
    demo["test"] = make_demo(parts["test"], ohe)
    n_demo = demo["train"].shape[1]

    # --- embedding: raw (for RF) and standardized (for Ridge), scaler on TRAIN ONLY ---
    emb = {s: parts[s][emb_cols].to_numpy(float) for s in parts}
    scaler = StandardScaler().fit(emb["train"])
    emb_z = {s: scaler.transform(emb[s]) for s in parts}

    X_rf    = {s: np.hstack([demo[s], emb[s]])   for s in parts}   # RF: raw emb
    X_ridge = {s: np.hstack([demo[s], emb_z[s]]) for s in parts}   # Ridge: scaled emb

    # --- Ridge: pick alpha by best val MAE-in-days ---
    best = None
    for a in RIDGE_ALPHAS:
        m = Ridge(alpha=a).fit(X_ridge["train"], y_t["train"])
        v = mae_real(m, X_ridge["val"], y_real["val"], lam)
        if best is None or v < best[0]:
            best = (v, a, m)
    _, ridge_alpha, ridge = best
    ridge_mae = {s: mae_real(ridge, X_ridge[s], y_real[s], lam) for s in parts}

    # --- RandomForest: pick (n_estimators, max_depth) by best val MAE-in-days ---
    rf, n_est, depth = best_rf_by_val(X_rf, y_t["train"], y_real, lam)
    rf_mae = {s: mae_real(rf, X_rf[s], y_real[s], lam) for s in parts}

    return {
        "boxcox_lambda": float(lam),
        "ridge_best_alpha": ridge_alpha,
        "rf_best": {"n_estimators": n_est, "max_depth": depth},
        "n_demographic_features": int(n_demo),
        "n_embedding_features": len(emb_cols),
        "n_features_total": int(n_demo + len(emb_cols)),
        "rows_per_split": {s: int(len(parts[s])) for s in parts},
        "mae_real_days": {"Ridge": ridge_mae, "RandomForest": rf_mae},
    }

# ---------------------------- the one runnable check ----------------------------

def selftest():
    """Tiny synthetic base+demographics matching the real schema; asserts the logic."""
    rng = np.random.default_rng(0)
    n, dim = 90, 8
    sid = [f"s{i}" for i in range(n)]
    split = (["train"] * 50) + (["val"] * 20) + (["test"] * 20)
    base = pd.DataFrame({
        "stay_id": sid, "subject_id": sid, "split": split,
        "remaining_los_days": np.round(rng.exponential(2.0, n) + 0.01, 2),
        "llm_point_estimate": rng.random(n), "llm_answer": ["..."] * n,
    })
    for i in range(dim):
        base[f"emb_{i}"] = rng.standard_normal(n)
    base["emb_0"] = 0.0  # constant column -> exercises StandardScaler zero-variance

    df05a = synthetic_demographics(rng, sid)

    asm = assemble(base, df05a)
    assert not {"icu_los_days", "outtime", "in_hospital_death",
                "last_careunit"} & set(asm.columns), "leaky column leaked in!"
    assert len(asm) == len(base)

    res = evaluate(asm)
    assert isinstance(res["boxcox_lambda"], float)
    assert res["n_embedding_features"] == dim
    assert res["n_features_total"] == res["n_demographic_features"] + dim
    for m in res["mae_real_days"].values():
        assert all(np.isfinite(v) and v >= 0 for v in m.values()), "bad MAE"
    log("selftest OK (assemble drops leaky cols, X+Z pipeline runs, MAEs finite)")

# --------------------------------- real run ---------------------------------

def main():
    selftest()  # run the check before touching real data

    base = pd.read_parquet(OUT_ROOT / f"{BASE_NAME}.parquet")
    base["stay_id"] = base["stay_id"].astype(str)
    df05a = pd.read_parquet(OUT_ROOT / f"{DEMO_NAME}.parquet")
    df05a["stay_id"] = df05a["stay_id"].astype(str)
    df = assemble(base, df05a)
    log(f"assembled X+Z frame: {df.shape}")

    res = evaluate(df)

    print("\nMAE in real days (remaining ICU LoS) -- X+Z")
    print(f"{'model':<16}{'train':>9}{'val':>9}{'test':>9}")
    for name, m in res["mae_real_days"].items():
        print(f"{name:<16}{m['train']:>9.3f}{m['val']:>9.3f}{m['test']:>9.3f}")
    print(f"\nreference: {REF_XONLY_RF} | {REF_Z_ALONE} | MedM2T {MEDM2T_MAE} d")
    print(f"Ridge alpha: {res['ridge_best_alpha']} | RF: {res['rf_best']} | "
          f"Box-Cox lambda: {res['boxcox_lambda']:.4f}")
    print(f"features: {res['n_demographic_features']} demo + "
          f"{res['n_embedding_features']} emb = {res['n_features_total']}")

    manifest = {
        "step": STEP_NAME,
        "run_at": dt.datetime.now().isoformat(timespec="seconds"),
        "inputs": {"base_embedding": str(OUT_ROOT / f"{BASE_NAME}.parquet"),
                   "demographics": str(OUT_ROOT / f"{DEMO_NAME}.parquet")},
        "case": "Case 1: X (7 demographics) + Z (1536 emb) -> remaining_los_days",
        "demographic_features": DEMO_COLS[1:],
        "references": {"x_only": REF_XONLY_RF, "z_alone": REF_Z_ALONE,
                       "medm2t_mae_days": MEDM2T_MAE},
        **res,
        "scale_caveat": "~1074-stay cap: directional only (tiny test fold, RF overfits); "
                        "decisive verdict needs the full cohort.",
        "leakage_note": "Box-Cox lambda, OneHot categories, and emb StandardScaler all "
                        "fit on TRAIN only, applied to val/test. No leaky 05a cols, "
                        "no llm_point_estimate/llm_answer used.",
    }
    (OUT_ROOT / f"{STEP_NAME}__manifest.json").write_text(json.dumps(manifest, indent=2))
    log("wrote manifest"); log("DONE.")

if __name__ == "__main__":
    main()
