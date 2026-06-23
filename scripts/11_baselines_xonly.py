#!/usr/bin/env python3
"""
Step 11 - X-only baselines (Case 3 reference): demographics -> remaining ICU LoS.

WHAT: predict remaining_los_days from 7 structured demographic features only,
with LinearRegression and a small-grid RandomForest, to set the bar the LLM
embedding (Case 1) must beat. MAE reported in REAL DAYS (Box-Cox inverted).

WHY THIS WAY: target is right-skewed -> Box-Cox (lambda fit on TRAIN ONLY).
Everything fitted (lambda, one-hot categories, models) is fit on train and
applied unchanged to val/test = leakage discipline (CLAUDE.md s3).

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
from pathlib import Path
import json
import argparse
import datetime as dt
import numpy as np
import pandas as pd
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

OUT_ROOT = Path("/home/pierrectrd/LoS project/data")
TARGET_NAME = "09b_los_strictly_positive"
DEMO_NAME   = "05a_target"
STEP_NAME   = "11_baselines_xonly"
SEED = 42
MEDM2T_MAE = 2.31  # their full-multimodal LoS MAE, days; our reference bar

NUM_COL  = "age_at_admission"
CAT_COLS = ["first_careunit", "gender", "admission_type",
            "insurance", "marital_status", "race"]
# the ONLY columns we take from 05a (stay_id key + the 7 features) -> leak guard
DEMO_COLS = ["stay_id", NUM_COL] + CAT_COLS
RF_GRID = [(n, d) for n in (200, 500) for d in (None, 10, 20)]

def log(msg): print(f"[{dt.datetime.now():%H:%M:%S}] {msg}")

# ---- pure helpers (so the self-test exercises the real logic, no disk I/O) ----

def assemble(df09b, df05a):
    """Left-join target/split frame <- demographics, on stay_id. 1:1, no inflation."""
    demo = df05a[DEMO_COLS]                       # drop every leaky 05a column here
    out = df09b.merge(demo, on="stay_id", how="left")
    assert len(out) == len(df09b), "row inflation on join (stay_id not 1:1)"
    return out

def make_X(frame, ohe, fit=False):
    cat = frame[CAT_COLS].fillna("Unknown")       # missing categ -> explicit level
    if fit:
        ohe.fit(cat)
    cat_arr = ohe.transform(cat)
    age = frame[[NUM_COL]].to_numpy(dtype=float)  # numeric, passed through
    return np.hstack([age, cat_arr])

def mae_real(model, X, y_real, lam):
    """Predict in Box-Cox space, invert to days, MAE vs raw real-day target."""
    pred = inv_boxcox(model.predict(X), lam)
    # ponytail: inv_boxcox is nan outside its domain (LR can extrapolate past it);
    # floor non-finite + negatives to 0 days (remaining LoS can't be <0).
    # ceiling = crude tail clip; upgrade = clip the transformed pred to the domain.
    pred = np.clip(np.where(np.isfinite(pred), pred, 0.0), 0.0, None)
    return float(np.mean(np.abs(pred - y_real)))

def evaluate(df):
    """Full leakage-ordered pipeline on an assembled frame. Returns a result dict."""
    parts = {s: df[df["split"] == s] for s in ("train", "val", "test")}
    assert all(len(p) for p in parts.values()), "a split is empty"
    assert parts["train"][NUM_COL].notna().all(), f"{NUM_COL} has NaN in train (spec: pass-through)"

    # --- target: Box-Cox, lambda from TRAIN ONLY, applied to val/test ---
    y_tr_real = parts["train"]["remaining_los_days"].to_numpy(float)
    assert y_tr_real.min() > 0, "train target not strictly positive (Box-Cox needs >0)"
    y_tr_t, lam = boxcox(y_tr_real)
    y_real = {s: parts[s]["remaining_los_days"].to_numpy(float) for s in parts}
    y_t = {"train": y_tr_t,
           "val":  boxcox(y_real["val"],  lmbda=lam),
           "test": boxcox(y_real["test"], lmbda=lam)}

    # --- features: OneHot fit on TRAIN ONLY ---
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X = {"train": make_X(parts["train"], ohe, fit=True)}
    X["val"]  = make_X(parts["val"],  ohe)
    X["test"] = make_X(parts["test"], ohe)
    feat_names = [NUM_COL] + list(ohe.get_feature_names_out(CAT_COLS))

    # --- LinearRegression ---
    lr = LinearRegression().fit(X["train"], y_t["train"])
    lr_mae = {s: mae_real(lr, X[s], y_real[s], lam) for s in parts}

    # --- RandomForest: pick (n_estimators, max_depth) by best val MAE-in-days ---
    best = None
    for n_est, depth in RF_GRID:
        rf = RandomForestRegressor(n_estimators=n_est, max_depth=depth,
                                   random_state=SEED, n_jobs=-1).fit(X["train"], y_t["train"])
        v = mae_real(rf, X["val"], y_real["val"], lam)
        if best is None or v < best[0]:
            best = (v, n_est, depth, rf)
    _, n_est, depth, rf = best
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
    cats = lambda opts: rng.choice(opts, n)
    df05a = pd.DataFrame({
        "stay_id": sid,
        NUM_COL: rng.integers(18, 92, n).astype(float),
        "first_careunit": cats(["MICU", "SICU", "CCU"]),
        "gender": cats(["M", "F"]),
        "admission_type": cats(["EW EMER.", "ELECTIVE"]),
        "insurance": cats(["Medicare", "Other"]),
        "marital_status": cats(["SINGLE", "MARRIED"]),
        "race": cats(["WHITE", "BLACK", "ASIAN"]),
        # leaky columns that assemble() must NOT carry through:
        "icu_los_days": rng.random(n), "outtime": "x", "in_hospital_death": 0,
        "last_careunit": "z", "window_start": "w",
    })
    # unseen category in val/test exercises handle_unknown='ignore'
    df05a.loc[df05a["stay_id"] == "s60", "first_careunit"] = "NICU"
    # a missing categorical exercises the "Unknown" fill
    df05a.loc[df05a["stay_id"] == "s0", "gender"] = np.nan

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
