#!/usr/bin/env python3
"""
Step 14 - Case 2: Stein-Encoder-INSPIRED supervised compression of Z.

WHAT: 3-stage architecture to predict remaining_los_days from demographics X1 +
a SINGLE supervised score t distilled from the 1536-dim LLM embedding Z, instead
of dumping all 1536 Z columns next to X (that is Case 1 / step 13, which overfits:
RF train MAE 1.33 vs test 2.44). Stages:
  1. RESIDUALIZE Z against X1  -> Z' = the narrative info beyond demographics;
  2. COMPRESS Z' -> one score t = gamma . Z'  (gamma = supervised weights);
  3. DOWNSTREAM model on (X1, t)  -> remaining_los_days.

WHY INSPIRED-NOT-LITERAL: Stein's closed-form gamma estimator is only valid if
Z|X is multivariate-Gaussian. Our LLM embeddings are NOT Gaussian (high-dim,
correlated, ~unit-norm), so that formula has no guarantee here. We keep the
architecture but replace ONLY the fragile weight-finding step with a robust
proxy: a ridge regression of the (transformed) target on Z', whose coefficients
ARE gamma. Ridge is distribution-agnostic -- it just finds predictive weights --
so the Gaussian assumption is sidestepped entirely.

WHY THIS WAY (leakage): target is right-skewed -> Box-Cox (lambda on TRAIN only).
The gamma fit in stage 2 is the one place this can leak; gamma is fit on TRAIN
ONLY and FROZEN before t is computed on val/test. Stage-1 residualizer, emb
scaler, one-hot categories: all train-only too.

DELIBERATELY DOES NOT:
  - tune the stage-1 residualizer alpha (fixed RESID_ALPHA; tuning a 1536-output
    nuisance model is cost the method doesn't need);
  - use NONLINEAR residualization (RF/XGB per emb dim, DML flexible-nuisance
    style) -- it would orthogonalize tighter but is deferred for cost; so Z still
    carries demographics nonlinearly and the orthogonalization is APPROXIMATE;
  - use llm_point_estimate / llm_answer (step 12's job);
  - merge the full 05a frame (icu_los_days, outtime, in_hospital_death,
    last_careunit = outcome/leaky); only 8 columns are pulled;
  - scale t or X1 for the downstream models (LR/RF don't need it here);
  - write a parquet (no per-row predictions requested).

POINT OF THIS SCRIPT vs CASE 1: not necessarily a lower MAE. It tests whether
compressing Z to ONE supervised score (vs 1536 raw dims) preserves the signal
while KILLING the overfitting Case-1 RF showed (watch the train-vs-test gap).

NOTE ON SCALE: ~1074-stay cap -> DIRECTIONAL ONLY (tiny test fold, RF overfits).
The decisive verdict needs the full cohort; this script is the apparatus.

Inputs : data/12_baseline_llm_float.parquet  (base: target, split, emb_0..1535)
         data/05a_target.parquet             (source of the 7 demographics)
Output : data/14_case2_stein_inspired__manifest.json  (no parquet)
"""
from pathlib import Path
import json
import datetime as dt
import numpy as np
import pandas as pd
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor

OUT_ROOT = Path("/home/pierrectrd/LoS project/data")
BASE_NAME = "12_baseline_llm_float"
DEMO_NAME = "05a_target"
STEP_NAME = "14_case2_stein_inspired"
SEED = 42

NUM_COL  = "age_at_admission"
CAT_COLS = ["first_careunit", "gender", "admission_type",
            "insurance", "marital_status", "race"]
DEMO_COLS = ["stay_id", NUM_COL] + CAT_COLS   # the ONLY cols taken from 05a (leak guard)

RESID_ALPHA   = 1.0                 # stage-1 nuisance ridge, FIXED (not tuned)
STAGE2_ALPHAS = [1, 10, 100, 1000]  # gamma ridge, selected by val MAE-in-days
RF_GRID = [(n, d) for n in (200, 500) for d in (None, 10, 20)]

# reference bars for the printed table (days)
REF_CASE1  = "step 13 X+Z  Ridge ~2.49 / RF ~2.44"
REF_XONLY  = "step 11 X-only ~2.59-2.67"
REF_ZALONE = "step 12 Z-alone ~2.57"
MEDM2T_MAE = 2.31

def log(msg): print(f"[{dt.datetime.now():%H:%M:%S}] {msg}")

# ---- pure helpers (self-test exercises these without disk I/O) ----

def assemble(df_base, df05a):
    """Left-join demographics onto the embedding base, on stay_id. 1:1, no inflation."""
    demo = df05a[DEMO_COLS]                         # drop every leaky 05a column here
    out = df_base.merge(demo, on="stay_id", how="left")
    assert len(out) == len(df_base), "row inflation on join (stay_id not 1:1)"
    return out

def make_demo(frame, ohe, fit=False):
    """One-hot demographics + raw numeric age -> dense block (X1)."""
    cat = frame[CAT_COLS].fillna("Unknown")        # missing categ -> explicit level
    if fit:
        ohe.fit(cat)
    age = frame[[NUM_COL]].to_numpy(dtype=float)   # numeric, passed through (unscaled)
    return np.hstack([age, ohe.transform(cat)])

def mae_real(model, X, y_real, lam):
    """Predict in Box-Cox space, invert to days, MAE vs raw real-day target."""
    pred = inv_boxcox(model.predict(X), lam)
    # ponytail: inv_boxcox is nan outside its domain (linear models extrapolate);
    # floor non-finite + negatives to 0 days. ceiling = crude tail clip; upgrade =
    # clip the transformed pred to the Box-Cox domain before inverting.
    pred = np.clip(np.where(np.isfinite(pred), pred, 0.0), 0.0, None)
    return float(np.mean(np.abs(pred - y_real)))

def evaluate(df):
    """Full leakage-ordered Stein-inspired pipeline. Returns a result dict."""
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    assert emb_cols, "no emb_* columns found in base"
    parts = {s: df[df["split"] == s] for s in ("train", "val", "test")}
    assert all(len(p) for p in parts.values()), "a split is empty"
    assert parts["train"][NUM_COL].notna().all(), f"{NUM_COL} has NaN in train"

    # --- target: Box-Cox, lambda from TRAIN ONLY ---
    y_tr_real = parts["train"]["remaining_los_days"].to_numpy(float)
    assert y_tr_real.min() > 0, "train target not strictly positive (Box-Cox needs >0)"
    y_tr_t, lam = boxcox(y_tr_real)
    y_real = {s: parts[s]["remaining_los_days"].to_numpy(float) for s in parts}
    y_t = {"train": y_tr_t,
           "val":  boxcox(y_real["val"],  lmbda=lam),
           "test": boxcox(y_real["test"], lmbda=lam)}

    # --- X1 demographics: OneHot fit on TRAIN ONLY ---
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    demo = {"train": make_demo(parts["train"], ohe, fit=True)}
    demo["val"]  = make_demo(parts["val"],  ohe)
    demo["test"] = make_demo(parts["test"], ohe)
    n_demo = demo["train"].shape[1]

    # --- Stage 1: residualize Z against X1 (everything fit on TRAIN ONLY) ---
    emb = {s: parts[s][emb_cols].to_numpy(float) for s in parts}
    scaler = StandardScaler().fit(emb["train"])
    emb_z = {s: scaler.transform(emb[s]) for s in parts}
    # ponytail: linear residualizer, first-order demographic removal only. Nonlinear
    # (RF/XGB per emb dim, DML flexible-nuisance) orthogonalizes tighter; deferred for cost.
    resid = Ridge(alpha=RESID_ALPHA, random_state=SEED).fit(demo["train"], emb_z["train"])
    Zprime = {s: emb_z[s] - resid.predict(demo[s]) for s in parts}   # demographics-stripped Z

    # --- Stage 2: compress Z' -> score t. gamma fit on TRAIN ONLY, then FROZEN. ---
    # Select stage-2 alpha by best val MAE-days of a cheap LinearRegression on (X1, t).
    def with_t(s, t):  # stack one-hot X1 + the single score column
        return np.hstack([demo[s], t[s][:, None]])
    best = None
    for a in STAGE2_ALPHAS:
        gamma = Ridge(alpha=a, random_state=SEED).fit(Zprime["train"], y_t["train"]).coef_
        t = {s: Zprime[s] @ gamma for s in parts}                   # frozen gamma -> t per split
        proxy = LinearRegression().fit(with_t("train", t), y_t["train"])
        v = mae_real(proxy, with_t("val", t), y_real["val"], lam)
        if best is None or v < best[0]:
            best = (v, a, t)
    _, stage2_alpha, t = best   # winning gamma's score, frozen

    Xds = {s: with_t(s, t) for s in parts}   # downstream design: X1 + t (single col)

    # --- Stage 3a: LinearRegression on (X1, t) ---
    lr = LinearRegression().fit(Xds["train"], y_t["train"])
    lr_mae = {s: mae_real(lr, Xds[s], y_real[s], lam) for s in parts}

    # --- Stage 3b: RandomForest on (X1, t), pick (n,d) by val MAE-days ---
    best = None
    for n_est, depth in RF_GRID:
        m = RandomForestRegressor(n_estimators=n_est, max_depth=depth,
                                  random_state=SEED, n_jobs=-1).fit(Xds["train"], y_t["train"])
        v = mae_real(m, Xds["val"], y_real["val"], lam)
        if best is None or v < best[0]:
            best = (v, n_est, depth, m)
    _, n_est, depth, rf = best
    rf_mae = {s: mae_real(rf, Xds[s], y_real[s], lam) for s in parts}

    return {
        "boxcox_lambda": float(lam),
        "residualizer": "linear",
        "resid_alpha": RESID_ALPHA,
        "stage2_ridge_alpha": stage2_alpha,
        "rf_best": {"n_estimators": n_est, "max_depth": depth},
        "n_demographic_features": int(n_demo),
        "n_score_features": 1,
        "n_features_total": int(n_demo + 1),
        "n_embedding_dims_compressed": len(emb_cols),
        "rows_per_split": {s: int(len(parts[s])) for s in parts},
        "mae_real_days": {"LinearRegression": lr_mae, "RandomForest": rf_mae},
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

    cats = lambda opts: rng.choice(opts, n)
    df05a = pd.DataFrame({
        "stay_id": sid,
        NUM_COL: rng.integers(18, 92, n).astype(float),
        "first_careunit": cats(["MICU", "SICU"]), "gender": cats(["M", "F"]),
        "admission_type": cats(["EW EMER.", "ELECTIVE"]), "insurance": cats(["Medicare", "Other"]),
        "marital_status": cats(["SINGLE", "MARRIED"]), "race": cats(["WHITE", "BLACK"]),
        "icu_los_days": rng.random(n), "outtime": "x",  # leaky cols assemble must drop
        "in_hospital_death": 0, "last_careunit": "z",
    })
    df05a.loc[df05a["stay_id"] == "s0", "gender"] = np.nan  # exercises "Unknown" fill

    asm = assemble(base, df05a)
    assert not {"icu_los_days", "outtime", "in_hospital_death",
                "last_careunit"} & set(asm.columns), "leaky column leaked in!"
    assert len(asm) == len(base)

    res = evaluate(asm)
    assert isinstance(res["boxcox_lambda"], float)
    assert res["n_score_features"] == 1, "score must be exactly one column"
    assert res["n_embedding_dims_compressed"] == dim
    assert res["n_features_total"] == res["n_demographic_features"] + 1
    for m in res["mae_real_days"].values():
        assert all(np.isfinite(v) and v >= 0 for v in m.values()), "bad MAE"
    log("selftest OK (assemble drops leaky cols, Z->1 score, pipeline runs, MAEs finite)")

# --------------------------------- real run ---------------------------------

def main():
    selftest()  # run the check before touching real data

    base = pd.read_parquet(OUT_ROOT / f"{BASE_NAME}.parquet")
    base["stay_id"] = base["stay_id"].astype(str)
    df05a = pd.read_parquet(OUT_ROOT / f"{DEMO_NAME}.parquet")
    df05a["stay_id"] = df05a["stay_id"].astype(str)
    df = assemble(base, df05a)
    log(f"assembled X1+Z frame: {df.shape}")

    res = evaluate(df)

    print("\nMAE in real days (remaining ICU LoS) -- Case 2 (X1 + 1 Stein-score t)")
    print(f"{'model':<18}{'train':>9}{'val':>9}{'test':>9}")
    for name, m in res["mae_real_days"].items():
        print(f"{name:<18}{m['train']:>9.3f}{m['val']:>9.3f}{m['test']:>9.3f}")
    print(f"\nreference: {REF_CASE1} | {REF_XONLY} | {REF_ZALONE} | MedM2T {MEDM2T_MAE} d")
    print(f"stage-2 ridge alpha: {res['stage2_ridge_alpha']} | RF: {res['rf_best']} | "
          f"Box-Cox lambda: {res['boxcox_lambda']:.4f}")
    print(f"features: {res['n_demographic_features']} demo + 1 score "
          f"(compressed from {res['n_embedding_dims_compressed']} emb dims)")

    manifest = {
        "step": STEP_NAME,
        "run_at": dt.datetime.now().isoformat(timespec="seconds"),
        "inputs": {"base_embedding": str(OUT_ROOT / f"{BASE_NAME}.parquet"),
                   "demographics": str(OUT_ROOT / f"{DEMO_NAME}.parquet")},
        "case": "Case 2: X1 (demographics) + t (1-D supervised compression of Z) "
                "-> remaining_los_days; Stein-Encoder-inspired",
        "demographic_features": DEMO_COLS[1:],
        "references": {"case1_x_plus_z": REF_CASE1, "x_only": REF_XONLY,
                       "z_alone": REF_ZALONE, "medm2t_mae_days": MEDM2T_MAE},
        **res,
        "method_note": "Stein architecture (residualize -> supervised-compress -> "
                       "downstream) preserved; Gaussian closed-form gamma REPLACED by a "
                       "ridge regression of the target on Z' (distribution-agnostic).",
        "residualization_note": "linear (first-order demographic removal only; nonlinear "
                                "deferred). Z still contains demographics nonlinearly, so "
                                "the orthogonalization is APPROXIMATE.",
        "scale_caveat": "~1074-stay cap: directional only (tiny test fold, RF overfits); "
                        "decisive verdict needs the full cohort.",
        "leakage_note": "Box-Cox lambda, OneHot categories, emb StandardScaler, stage-1 "
                        "residualizer, and stage-2 gamma ALL fit on TRAIN only, applied to "
                        "val/test. gamma frozen before t is computed off-train. No leaky "
                        "05a cols, no llm_point_estimate/llm_answer used.",
    }
    (OUT_ROOT / f"{STEP_NAME}__manifest.json").write_text(json.dumps(manifest, indent=2))
    log("wrote manifest"); log("DONE.")

if __name__ == "__main__":
    main()
