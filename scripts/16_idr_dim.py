#!/usr/bin/env python3
"""
Step 16 - Case 5: Distributional Index Model (DIM) + Isotonic Distributional
Regression (IDR) for remaining ICU LoS.

WHAT: the project's first DISTRIBUTIONAL method. Every prior step (11-14b) emits a
point prediction; this one emits, per patient, a full predictive CDF of remaining
LoS, from which we read a CRPS (distributional accuracy) and a predictive median
(MAE in days, comparable to the point baselines and MedM2T). Two stages, per
Henzi-Kleger-Ziegel (2021) DIM + Henzi-Ziegel-Gneiting (2021) IDR:
  Stage 1 (index): fit Ridge to predict (Box-Cox) remaining-LoS from the features;
    the index g(x) is its prediction. The index only needs to RANK patients -- IDR
    is invariant under increasing transforms of the index, so absolute values and
    the Box-Cox choice are immaterial to the fit, only the ordering matters.
  Stage 2 (IDR): feed (index_i, real_remaining_los_i) to IDR; for each LoS
    threshold z it returns P(LoS <= z | g(x)), monotone non-increasing in the index
    (a higher index never implies more short-stay probability -> stochastic order).

TWO SETTINGS (run both, report both):
  A - EMBEDDING ONLY : index fit on the 1536 emb_* columns (Z) alone.
  B - CONCATENATION  : index fit on Z concatenated with structured demographics X
                       (age numeric; 6 categoricals one-hot, fit on TRAIN only).

WHY sklearn, not the repo's Rust IDR: the vendored isodistrreg Python binding is a
compiled Rust extension needing Python>=3.13 + cargo + maturin; this .venv is 3.10
with none of those, no wheel, no .so. AND its CRPS + DIM (`dindexm`) live only in
the R binding -- they must be hand-written in Python under any engine. For a 1-D
index, IDR provably reduces to per-threshold isotonic regression (PAVA), and
sklearn's IsotonicRegression IS a PAVA implementation: it reuses a library PAVA
(not a hand-rolled one) and returns the identical CDF to the Rust core (the
isotonic least-squares problem is convex with a unique solution). This equivalence
holds BECAUSE the index is 1-D -- DIM's one-number compression is what makes the
reduction exact; it would NOT hold for a multivariate covariate.

DELIBERATELY DOES NOT:
  - claim CRPS is comparable to the point-forecast baselines / MedM2T: CRPS scores
    whole distributions, MAE scores points -- different scales, not the same ruler.
    Only the median-MAE column is cross-comparable to those.
  - pool in-hospital deaths: they are EXCLUDED upstream at 09b (survivors only).
    A death gives a stay truncated by death, not discharge -- a short stay that
    would violate IDR's "higher index => longer LoS" stochastic ordering. Removing
    them at the cohort source (09b) is what keeps that ordering meaningful here.
  - tune anything on test: alpha is picked by val CRPS only; Box-Cox lambda, OHE,
    StandardScaler and Ridge are all fit on TRAIN and applied unchanged to val/test.

Inputs : data/12_baseline_llm_float.parquet  (target, split, emb_0..1535)
         data/05a_target.parquet             (source of the demographics, Setting B)
Output : data/16_idr_dim__manifest.json  (no parquet)
         RESULTS_10k.md is updated by hand from the manifest (kept idempotent --
         the script does not append, to avoid duplicate sections on re-run).
"""
import json
import datetime as dt
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.isotonic import IsotonicRegression

from _common import (OUT_ROOT, MEDM2T_MAE, NUM_COL, DEMO_COLS,
                     log, assemble, split_parts, make_demo, boxcox_target,
                     synthetic_demographics)

BASE_NAME = "12_baseline_llm_float"
DEMO_NAME = "05a_target"
STEP_NAME = "16_idr_dim"

RIDGE_ALPHAS = [1, 10, 100, 1000]

# reference bars for the printed table (median-MAE days; CRPS has no such bar)
REF_XONLY_RF = "step 11 X-only RF ~2.64"
REF_XZ_RF    = "step 13 X+Z RF ~2.52"

# ------------------------------ the IDR core (1-D) ------------------------------
# For a single ordered index, IDR = per-threshold isotonic regression of the
# threshold-exceedance indicator on the index. sklearn's IsotonicRegression runs
# PAVA; the family of fits across thresholds is rearranged into a valid CDF.

def fit_idr(index_train, y_train):
    """Fit 1-D IDR. Returns (thresholds, iso_models): one IsotonicRegression per
    unique training response value, each modelling P(Y <= z) as a NON-INCREASING
    function of the index (higher index -> stochastically longer LoS)."""
    thresholds = np.unique(y_train)
    models = []
    for z in thresholds:
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
        iso.fit(index_train, (y_train <= z).astype(float))
        models.append(iso)
    return thresholds, models


def predict_cdf(iso_models, index_new):
    """Predictive CDF for new index values: (n_new, n_thresholds). Each column is
    one isotonic model's prediction; rearrange per row into a valid CDF (monotone
    non-decreasing in the threshold z, clipped to [0, 1])."""
    cdf = np.column_stack([m.predict(index_new) for m in iso_models])
    cdf = np.maximum.accumulate(cdf, axis=1)
    return np.clip(cdf, 0.0, 1.0)


def crps_idr(cdf, thresholds, y):
    """CRPS of the discrete IDR forecast, ported verbatim from the repo's R
    crps.idr (isodistrreg/bindings/R/R/evaluation.R). w = per-row CDF jumps;
    CRPS_i = 2 * sum_j w_ij * ((y_i < p_j) - cdf_ij + 0.5*w_ij) * (p_j - y_i).
    Returns the per-observation CRPS array (real-day units)."""
    p = thresholds
    y = np.asarray(y, float)
    w = np.diff(cdf, axis=1, prepend=0.0)          # w[:,0]=cdf[:,0]; else cdf_j-cdf_{j-1}
    ind = (y[:, None] < p[None, :]).astype(float)
    contrib = w * (ind - cdf + 0.5 * w) * (p[None, :] - y[:, None])
    return 2.0 * contrib.sum(axis=1)


def median_from_cdf(cdf, thresholds):
    """Predictive median = smallest threshold whose CDF >= 0.5; if a row never
    reaches 0.5, fall back to the largest threshold."""
    reached = cdf >= 0.5
    idx = np.argmax(reached, axis=1)               # first True per row (0 if none)
    idx[~reached.any(axis=1)] = len(thresholds) - 1
    return thresholds[idx]


# ------------------------------- the DIM per setting -------------------------------

def run_setting(parts, y_t, y_real, lam, features):
    """One DIM/IDR setting given a per-split design dict `features`. Ridge alpha
    chosen by val CRPS (train-only fit); reports CRPS + median-MAE on all splits."""
    best = None
    for a in RIDGE_ALPHAS:
        ridge = Ridge(alpha=a).fit(features["train"], y_t["train"])
        index = {s: ridge.predict(features[s]) for s in parts}
        thr, models = fit_idr(index["train"], y_real["train"])
        vcrps = crps_idr(predict_cdf(models, index["val"]), thr, y_real["val"]).mean()
        if best is None or vcrps < best[0]:
            best = (vcrps, a, index, thr, models)
    _, alpha, index, thr, models = best

    crps, mae = {}, {}
    for s in parts:
        cdf = predict_cdf(models, index[s])
        crps[s] = float(crps_idr(cdf, thr, y_real[s]).mean())
        mae[s] = float(np.mean(np.abs(median_from_cdf(cdf, thr) - y_real[s])))
    return {"ridge_best_alpha": alpha, "n_thresholds": int(len(thr)),
            "crps_days": crps, "median_mae_days": mae}


def evaluate(df):
    """Both settings on an assembled X+Z frame. Returns a result dict."""
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    assert emb_cols, "no emb_* columns found in base"
    parts = split_parts(df)
    assert parts["train"][NUM_COL].notna().all(), f"{NUM_COL} has NaN in train"

    y_t, y_real, lam = boxcox_target(parts)        # Box-Cox lambda from TRAIN only

    # embedding Z: standardized, scaler on TRAIN only (Ridge is scale-sensitive)
    emb = {s: parts[s][emb_cols].to_numpy(float) for s in parts}
    scaler = StandardScaler().fit(emb["train"])
    emb_z = {s: scaler.transform(emb[s]) for s in parts}

    # demographics X: one-hot fit on TRAIN only (Setting B only)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    demo = {"train": make_demo(parts["train"], ohe, fit=True),
            "val":   make_demo(parts["val"],   ohe),
            "test":  make_demo(parts["test"],  ohe)}

    featA = emb_z                                                  # Z only
    featB = {s: np.hstack([demo[s], emb_z[s]]) for s in parts}     # X + Z

    return {
        "boxcox_lambda": float(lam),
        "n_demographic_features": int(demo["train"].shape[1]),
        "n_embedding_features": len(emb_cols),
        "rows_per_split": {s: int(len(parts[s])) for s in parts},
        "settings": {
            "A_embedding_only": run_setting(parts, y_t, y_real, lam, featA),
            "B_concat_X_Z":     run_setting(parts, y_t, y_real, lam, featB),
        },
    }

# ---------------------------- the one runnable check ----------------------------

def selftest():
    """Exercises the IDR core on cases with known closed-form answers, then the
    full DIM pipeline on synthetic data -- before touching real data."""
    rng = np.random.default_rng(0)

    # 1. CRPS of a point-mass forecast (CDF jumps 0->1 at m) must equal |y - m|.
    p = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    m = 3.0
    cdf_pm = (p[None, :] >= m).astype(float)
    for yv in (1.0, 2.5, 3.0, 4.0, 5.0):
        c = crps_idr(cdf_pm, p, np.array([yv]))[0]
        assert abs(c - abs(yv - m)) < 1e-9, (yv, c)

    # 2. IDR on synthetic data where y increases with the index.
    n = 300
    idx = rng.random(n)
    y = np.round(10 * idx + rng.random(n) + 0.01, 2)
    thr, models = fit_idr(idx, y)
    grid = np.array([0.1, 0.5, 0.9])               # ascending index values
    cdf = predict_cdf(models, grid)
    assert (cdf >= -1e-9).all() and (cdf <= 1 + 1e-9).all(), "CDF out of [0,1]"
    assert (np.diff(cdf, axis=1) >= -1e-9).all(), "CDF not monotone in threshold"
    # higher index -> CDF not higher at any threshold (stochastic ordering)
    assert (np.diff(cdf, axis=0) <= 1e-9).all(), "stochastic ordering violated"

    # 3. Median recovery on a toy CDF (crosses 0.5 at p=3).
    p2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    cdf2 = np.array([[0.1, 0.3, 0.6, 0.8, 1.0]])
    assert median_from_cdf(cdf2, p2)[0] == 3.0, "median extraction wrong"

    # 4. Full DIM pipeline on synthetic base+demographics matching the real schema.
    nb, dim = 90, 8
    sid = [f"s{i}" for i in range(nb)]
    split = (["train"] * 50) + (["val"] * 20) + (["test"] * 20)
    base = pd.DataFrame({
        "stay_id": sid, "subject_id": sid, "split": split,
        "remaining_los_days": np.round(rng.exponential(2.0, nb) + 0.01, 2),
        "llm_point_estimate": rng.random(nb), "llm_answer": ["..."] * nb,
    })
    for i in range(dim):
        base[f"emb_{i}"] = rng.standard_normal(nb)
    base["emb_0"] = 0.0                            # constant col -> StandardScaler edge
    asm = assemble(base, synthetic_demographics(rng, sid))
    assert not {"icu_los_days", "outtime", "in_hospital_death",
                "last_careunit"} & set(asm.columns), "leaky column leaked in!"
    res = evaluate(asm)
    for st in res["settings"].values():
        for d in (st["crps_days"], st["median_mae_days"]):
            assert all(np.isfinite(v) and v >= 0 for v in d.values()), "bad metric"
    log("selftest OK (CRPS=|y-m| on point mass, valid+ordered CDFs, median, DIM runs)")

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

    print("\nDIM + IDR -- CRPS and median-MAE in real days (remaining ICU LoS)")
    print(f"{'setting':<20}{'alpha':>7}{'  ':>2}"
          f"{'CRPS tr':>9}{'CRPS val':>9}{'CRPS te':>9}"
          f"{'MAE tr':>9}{'MAE val':>9}{'MAE te':>9}")
    for key, st in res["settings"].items():
        c, m = st["crps_days"], st["median_mae_days"]
        print(f"{key:<20}{st['ridge_best_alpha']:>7}{'  ':>2}"
              f"{c['train']:>9.3f}{c['val']:>9.3f}{c['test']:>9.3f}"
              f"{m['train']:>9.3f}{m['val']:>9.3f}{m['test']:>9.3f}")
    print(f"\nmedian-MAE refs: {REF_XONLY_RF} | {REF_XZ_RF} | MedM2T {MEDM2T_MAE} d "
          f"(CRPS is NOT comparable to these point scores)")
    print(f"Box-Cox lambda: {res['boxcox_lambda']:.4f} | "
          f"{res['n_demographic_features']} demo + {res['n_embedding_features']} emb")

    manifest = {
        "step": STEP_NAME,
        "run_at": dt.datetime.now().isoformat(timespec="seconds"),
        "inputs": {"base_embedding": str(OUT_ROOT / f"{BASE_NAME}.parquet"),
                   "demographics": str(OUT_ROOT / f"{DEMO_NAME}.parquet")},
        "case": "Case 5: DIM (Ridge index) + IDR -> distributional remaining_los_days",
        "method": "Henzi-Kleger-Ziegel (2021) DIM + Henzi-Ziegel-Gneiting (2021) IDR",
        "demographic_features": DEMO_COLS[1:],
        "index_model": "Ridge on Box-Cox target; alpha chosen by val CRPS",
        "idr_engine": "sklearn.isotonic.IsotonicRegression (1-D PAVA); per-threshold "
                      "fit + rearrangement. Equivalent to the repo's Rust IDR for a "
                      "1-D index (unique convex isotonic solution).",
        "metrics": "CRPS (distributional) and MAE of the IDR predictive median, "
                   "both in real days, per split.",
        "references": {"x_only_rf": REF_XONLY_RF, "x_plus_z_rf": REF_XZ_RF,
                       "medm2t_mae_days": MEDM2T_MAE},
        **res,
        "comparability_caveat": "CRPS scores whole distributions; only the median-MAE "
                                "column is comparable to the point baselines / MedM2T.",
        "stochastic_order_caveat": "in-hospital deaths are EXCLUDED at 09b (survivors "
                                   "only); their death-truncated short stays would "
                                   "violate the 'higher index => longer LoS' ordering. "
                                   "Removed at the cohort source, not here.",
        "leakage_note": "Box-Cox lambda, OneHot categories, emb StandardScaler, Ridge, "
                        "and the IDR fit are all TRAIN-only; alpha picked by val CRPS; "
                        "test never used in selection.",
    }
    (OUT_ROOT / f"{STEP_NAME}__manifest.json").write_text(json.dumps(manifest, indent=2))
    log("wrote manifest"); log("DONE.")

if __name__ == "__main__":
    main()
