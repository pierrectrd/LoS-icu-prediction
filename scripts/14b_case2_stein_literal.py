#!/usr/bin/env python3
"""
Step 14b - Case 2 (LITERAL Stein-Encoder variant): recover the index direction
gamma from Stein MOMENTS over a set of probe functions T(Y), instead of the
ridge-regression stand-in used in step 14.

WHAT (the literal Stein machinery step 14 deliberately skipped):
  1. RESIDUALIZE the embedding Z against demographics X1 (remove what X explains)
     -> Z'  (linear, train-fit; same first-order removal as step 14).
  2. WHITEN Z' so its covariance is ~I  -> W ("whitened residual"). We do this by
     PCA-whitening (train-fit) on a capped number of components, NOT by inverting
     the full 1536x1536 covariance (singular/unstable -- the very thing step 14's
     ridge avoided). The Stein identities below assume Cov(W)=I, so whitening is
     mandatory here, unlike step 14.
  3. For each probe T(Y) and each ORDER, form the Stein moment on TRAIN:
       first order : g = E[ T(Y) * W ]                    (a d-vector)
                     -> gamma_hat = g / ||g||
       second order: M = E[ T(Y) * (W W^T - I) ]          (a d x d matrix)
                     -> gamma_hat = leading eigenvector (largest |eigenvalue|)
     These are the closed-form Stein estimators of the single-index direction in
     Y = f(gamma^T W) + noise: the first identity says E[T(Y)W] ∝ gamma when the
     link's first Stein coefficient is non-zero; the second recovers gamma even
     when that first coefficient vanishes (e.g. symmetric links) via the Hessian.
  4. PICK the probe/order with the strongest signal. We REPORT each candidate's
     train signal |corr(t, Y)|, but SELECT the winner by VALIDATION MAE-in-days
     of a cheap LinearRegression on (X1, t) -- leakage-safe, and consistent with
     how step 14 selects its ridge alpha. (Pure-train "strongest |corr|" is also
     printed, so you can see if the two criteria agree.)
  5. Output the scalar score t = gamma_hat^T W for EVERY patient, then run the
     same downstream LR + RF on (X1, t) as step 14, so MAEs are comparable.
     Closed-form throughout: no iterative training of gamma.

PROBES (from the Stein-Encoder paper): T(Y) in { y, y^2, arctan(a*y),
a*y^2/(1+a*y^2) }, with a in A_VALUES. The two bounded probes (arctan, rational)
saturate, so they resist the heavy LoS tail. Probes are applied to the
Box-Cox-transformed target STANDARDIZED on train (zero-mean/unit-var) so that
"a" lives on a comparable scale and y^2 doesn't explode on raw day counts.

WHY THIS vs step 14: step 14 replaced the Stein moment with a ridge regression of
Y on Z' (distribution-agnostic, robust). This script does the textbook thing
instead -- explicit moments + probe/order selection -- to see whether the literal
estimator buys anything on real (non-Gaussian) embeddings. Expectation: the
Gaussian assumption behind the moments does NOT hold for LLM embeddings, so this
is a curiosity/benchmark, not necessarily an improvement over step 14's ridge.

DELIBERATELY DOES NOT:
  - whiten the FULL 1536-dim residual (singular covariance); caps at PCA_COMPONENTS
    high-variance directions -> whitening is on the stable subspace only;
  - use a NONLINEAR residualizer (deferred, as in step 14) -> orthogonalization
    against X1 is APPROXIMATE; Z' still carries demographics nonlinearly;
  - tune the residualizer alpha or PCA_COMPONENTS (both FIXED);
  - use llm_point_estimate / llm_answer, or any leaky 05a column;
  - write a parquet (no per-row predictions requested).

LEAKAGE: Box-Cox lambda, target standardization, OneHot categories, emb scaler,
residualizer, PCA-whitener, and the chosen gamma are ALL fit on TRAIN ONLY;
gamma is frozen before t is computed on val/test; probe/order picked on val.

Inputs : data/12_baseline_llm_float.parquet  (base: target, split, emb_0..1535)
         data/05a_target.parquet             (source of the 7 demographics)
Output : data/14b_case2_stein_literal__manifest.json  (no parquet)
"""
from pathlib import Path
import json
import datetime as dt
import numpy as np
import pandas as pd
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor

OUT_ROOT = Path("/home/pierrectrd/LoS project/data")
BASE_NAME = "12_baseline_llm_float"
DEMO_NAME = "05a_target"
STEP_NAME = "14b_case2_stein_literal"
SEED = 42

NUM_COL  = "age_at_admission"
CAT_COLS = ["first_careunit", "gender", "admission_type",
            "insurance", "marital_status", "race"]
DEMO_COLS = ["stay_id", NUM_COL] + CAT_COLS   # ONLY cols taken from 05a (leak guard)

RESID_ALPHA    = 1.0      # stage-1 nuisance ridge, FIXED (not tuned)
PCA_COMPONENTS = 100      # whitened subspace size, FIXED (stability vs full 1536-dim)
A_VALUES       = [0.5, 1.0, 2.0]   # scale param for the two bounded probes
RF_GRID = [(n, d) for n in (200, 500) for d in (None, 10, 20)]

REF_CASE2  = "step 14 X1+ridge-score: LR ~2.53 / RF ~2.58 (10k)"
REF_CASE1  = "step 13 X+Z: Ridge ~2.53 / RF ~2.52 (10k)"
MEDM2T_MAE = 2.31

def log(msg): print(f"[{dt.datetime.now():%H:%M:%S}] {msg}")

# ----------------------------- probe functions ------------------------------

def build_probes():
    """T(Y) candidates from the Stein-Encoder paper. Returns [(name, func), ...].
    y and y^2 do not depend on a, so they appear once; the two bounded probes
    appear once per a in A_VALUES."""
    cand = [("y", lambda y: y),
            ("y^2", lambda y: y**2)]
    for a in A_VALUES:
        cand.append((f"arctan({a}y)",        lambda y, a=a: np.arctan(a * y)))
        cand.append((f"{a}y^2/(1+{a}y^2)",   lambda y, a=a: (a * y**2) / (1.0 + a * y**2)))
    return cand

# ---- pure helpers (shared shape with step 14; self-test exercises them) ----

def assemble(df_base, df05a):
    """Left-join demographics onto the embedding base on stay_id. 1:1, no inflation."""
    demo = df05a[DEMO_COLS]                         # drop every leaky 05a column here
    out = df_base.merge(demo, on="stay_id", how="left")
    assert len(out) == len(df_base), "row inflation on join (stay_id not 1:1)"
    return out

def make_demo(frame, ohe, fit=False):
    """One-hot demographics + raw numeric age -> dense block (X1)."""
    cat = frame[CAT_COLS].fillna("Unknown")
    if fit:
        ohe.fit(cat)
    age = frame[[NUM_COL]].to_numpy(dtype=float)
    return np.hstack([age, ohe.transform(cat)])

def mae_real(model, X, y_real, lam):
    """Predict in Box-Cox space, invert to days, MAE vs raw real-day target."""
    pred = inv_boxcox(model.predict(X), lam)
    pred = np.clip(np.where(np.isfinite(pred), pred, 0.0), 0.0, None)
    return float(np.mean(np.abs(pred - y_real)))

def stein_direction(W_tr, Tvals, order):
    """Closed-form Stein direction from TRAIN whitened residual W_tr (Cov~I, mean~0)
    and probe values Tvals=T(Y_tr). order 1 -> moment vector; order 2 -> Hessian
    moment's leading eigenvector. Returns a unit d-vector gamma_hat."""
    Tc = Tvals - Tvals.mean()                       # center the probe (W has mean ~0)
    n = len(Tc)
    if order == 1:
        g = (W_tr * Tc[:, None]).sum(0) / n         # g = E[T(Y) W]
    else:
        # M = E[T(Y)(W W^T - I)]; with Tc centered the -I term vanishes in expectation
        M = (W_tr * Tc[:, None]).T @ W_tr / n       # d x d symmetric
        w, V = np.linalg.eigh(M)
        g = V[:, np.argmax(np.abs(w))]              # eigenvector of largest |eigenvalue|
    nrm = np.linalg.norm(g)
    return g / nrm if nrm > 0 else g

def evaluate(df):
    """Literal-Stein pipeline (leakage-ordered). Returns a result dict."""
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    assert emb_cols, "no emb_* columns found in base"
    parts = {s: df[df["split"] == s] for s in ("train", "val", "test")}
    assert all(len(p) for p in parts.values()), "a split is empty"

    # --- target: Box-Cox (lambda TRAIN-only), then standardize (TRAIN-only) ---
    y_tr_real = parts["train"]["remaining_los_days"].to_numpy(float)
    assert y_tr_real.min() > 0, "train target not strictly positive (Box-Cox needs >0)"
    y_tr_t, lam = boxcox(y_tr_real)
    y_real = {s: parts[s]["remaining_los_days"].to_numpy(float) for s in parts}
    y_t = {"train": y_tr_t,
           "val":  boxcox(y_real["val"],  lmbda=lam),
           "test": boxcox(y_real["test"], lmbda=lam)}
    ymu, ysd = y_t["train"].mean(), y_t["train"].std()
    ys = {s: (y_t[s] - ymu) / ysd for s in parts}   # standardized target -> probe input

    # --- X1 demographics: OneHot fit TRAIN-only ---
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    demo = {"train": make_demo(parts["train"], ohe, fit=True)}
    demo["val"]  = make_demo(parts["val"],  ohe)
    demo["test"] = make_demo(parts["test"], ohe)
    n_demo = demo["train"].shape[1]

    # --- Stage 1: residualize Z against X1 (TRAIN-only fit) ---
    emb = {s: parts[s][emb_cols].to_numpy(float) for s in parts}
    scaler = StandardScaler().fit(emb["train"])
    emb_z = {s: scaler.transform(emb[s]) for s in parts}
    resid = Ridge(alpha=RESID_ALPHA, random_state=SEED).fit(demo["train"], emb_z["train"])
    Zprime = {s: emb_z[s] - resid.predict(demo[s]) for s in parts}

    # --- Stage 2: WHITEN the residual (PCA-whiten, TRAIN-only) -> W, Cov(W_train)~I ---
    k = min(PCA_COMPONENTS, len(emb_cols), len(parts["train"]) - 1)
    pca = PCA(n_components=k, whiten=True, random_state=SEED).fit(Zprime["train"])
    W = {s: pca.transform(Zprime[s]) for s in parts}

    # --- Stage 3: Stein moments over probes x orders; gamma TRAIN-only, then frozen ---
    def with_t(s, t):
        return np.hstack([demo[s], t[s][:, None]])
    cands = []
    for pname, T in build_probes():
        Tvals = {s: T(ys[s]) for s in parts}
        for order in (1, 2):
            gamma = stein_direction(W["train"], Tvals["train"], order)
            t = {s: W[s] @ gamma for s in parts}
            # align sign so the score correlates positively with the target (cosmetic)
            if np.corrcoef(t["train"], ys["train"])[0, 1] < 0:
                gamma = -gamma
                t = {s: -t[s] for s in parts}
            signal = abs(np.corrcoef(t["train"], ys["train"])[0, 1])   # train "strength"
            proxy = LinearRegression().fit(with_t("train", t), y_t["train"])
            val_mae = mae_real(proxy, with_t("val", t), y_real["val"], lam)
            cands.append({"probe": pname, "order": order,
                          "train_signal_abs_corr": float(signal),
                          "val_mae_days": float(val_mae), "t": t})
    # winner by VAL MAE (leakage-safe); also note the strongest-train-signal one
    best = min(cands, key=lambda c: c["val_mae_days"])
    strongest = max(cands, key=lambda c: c["train_signal_abs_corr"])
    t = best["t"]
    Xds = {s: with_t(s, t) for s in parts}

    # --- Downstream a: LinearRegression on (X1, t) ---
    lr = LinearRegression().fit(Xds["train"], y_t["train"])
    lr_mae = {s: mae_real(lr, Xds[s], y_real[s], lam) for s in parts}

    # --- Downstream b: RandomForest on (X1, t), pick (n,d) by val MAE ---
    rfbest = None
    for n_est, depth in RF_GRID:
        m = RandomForestRegressor(n_estimators=n_est, max_depth=depth,
                                  random_state=SEED, n_jobs=-1).fit(Xds["train"], y_t["train"])
        v = mae_real(m, Xds["val"], y_real["val"], lam)
        if rfbest is None or v < rfbest[0]:
            rfbest = (v, n_est, depth, m)
    _, n_est, depth, rf = rfbest
    rf_mae = {s: mae_real(rf, Xds[s], y_real[s], lam) for s in parts}

    # strip the heavy 't' arrays before returning the candidate table
    table = [{kk: c[kk] for kk in ("probe", "order", "train_signal_abs_corr", "val_mae_days")}
             for c in cands]
    return {
        "boxcox_lambda": float(lam),
        "pca_components": int(k),
        "resid_alpha": RESID_ALPHA,
        "selected_probe": best["probe"], "selected_order": best["order"],
        "selected_by": "val_mae_days",
        "strongest_train_signal_probe": strongest["probe"],
        "strongest_train_signal_order": strongest["order"],
        "rf_best": {"n_estimators": n_est, "max_depth": depth},
        "n_demographic_features": int(n_demo),
        "n_score_features": 1,
        "n_embedding_dims_compressed": len(emb_cols),
        "rows_per_split": {s: int(len(parts[s])) for s in parts},
        "candidates": table,
        "mae_real_days": {"LinearRegression": lr_mae, "RandomForest": rf_mae},
    }

# ---------------------------- the one runnable check ----------------------------

def selftest():
    """Synthetic base+demographics with a PLANTED single-index signal, so the Stein
    moment has something real to recover. Asserts schema, 1-D score, finite MAEs."""
    rng = np.random.default_rng(0)
    n, dim = 120, 12
    sid = [f"s{i}" for i in range(n)]
    split = (["train"] * 70) + (["val"] * 25) + (["test"] * 25)
    Z = rng.standard_normal((n, dim))
    direction = rng.standard_normal(dim)
    idx = Z @ direction
    los = np.clip(0.5 + 0.8 * idx + rng.standard_normal(n) * 0.5, 0.01, None)  # single-index
    base = pd.DataFrame({"stay_id": sid, "subject_id": sid, "split": split,
                         "remaining_los_days": np.round(los, 2),
                         "llm_point_estimate": rng.random(n), "llm_answer": ["..."] * n})
    for i in range(dim):
        base[f"emb_{i}"] = Z[:, i]

    cats = lambda opts: rng.choice(opts, n)
    df05a = pd.DataFrame({
        "stay_id": sid, NUM_COL: rng.integers(18, 92, n).astype(float),
        "first_careunit": cats(["MICU", "SICU"]), "gender": cats(["M", "F"]),
        "admission_type": cats(["EW EMER.", "ELECTIVE"]), "insurance": cats(["Medicare", "Other"]),
        "marital_status": cats(["SINGLE", "MARRIED"]), "race": cats(["WHITE", "BLACK"]),
        "icu_los_days": rng.random(n), "outtime": "x",
        "in_hospital_death": 0, "last_careunit": "z"})
    df05a.loc[df05a["stay_id"] == "s0", "gender"] = np.nan

    asm = assemble(base, df05a)
    assert not {"icu_los_days", "outtime", "in_hospital_death",
                "last_careunit"} & set(asm.columns), "leaky column leaked in!"
    res = evaluate(asm)
    assert res["n_score_features"] == 1, "score must be exactly one column"
    assert res["n_embedding_dims_compressed"] == dim
    assert len(res["candidates"]) == len(build_probes()) * 2, "probe x order count off"
    for m in res["mae_real_days"].values():
        assert all(np.isfinite(v) and v >= 0 for v in m.values()), "bad MAE"
    log(f"selftest OK (probes={len(build_probes())}, picked {res['selected_probe']} "
        f"order {res['selected_order']}, MAEs finite)")

# --------------------------------- real run ---------------------------------

def main():
    selftest()

    base = pd.read_parquet(OUT_ROOT / f"{BASE_NAME}.parquet")
    base["stay_id"] = base["stay_id"].astype(str)
    df05a = pd.read_parquet(OUT_ROOT / f"{DEMO_NAME}.parquet")
    df05a["stay_id"] = df05a["stay_id"].astype(str)
    df = assemble(base, df05a)
    log(f"assembled X1+Z frame: {df.shape}")

    res = evaluate(df)

    print("\nStein moments over probes x orders (selected by val MAE-days):")
    print(f"{'probe':<18}{'order':>6}{'train|corr|':>13}{'val MAE':>10}")
    for c in sorted(res["candidates"], key=lambda x: x["val_mae_days"]):
        mark = "  <- picked" if (c["probe"] == res["selected_probe"]
                                 and c["order"] == res["selected_order"]) else ""
        print(f"{c['probe']:<18}{c['order']:>6}{c['train_signal_abs_corr']:>13.3f}"
              f"{c['val_mae_days']:>10.3f}{mark}")

    print("\nMAE in real days (remaining ICU LoS) -- Case 2 LITERAL Stein "
          "(X1 + 1 Stein-moment score t)")
    print(f"{'model':<18}{'train':>9}{'val':>9}{'test':>9}")
    for name, m in res["mae_real_days"].items():
        print(f"{name:<18}{m['train']:>9.3f}{m['val']:>9.3f}{m['test']:>9.3f}")
    print(f"\nselected probe/order: {res['selected_probe']} / {res['selected_order']} "
          f"(by val MAE) | strongest train signal: "
          f"{res['strongest_train_signal_probe']} / {res['strongest_train_signal_order']}")
    print(f"reference: {REF_CASE2} | {REF_CASE1} | MedM2T {MEDM2T_MAE} d")
    print(f"PCA-whitened dims: {res['pca_components']} | RF: {res['rf_best']} | "
          f"Box-Cox lambda: {res['boxcox_lambda']:.4f}")

    manifest = {
        "step": STEP_NAME,
        "run_at": dt.datetime.now().isoformat(timespec="seconds"),
        "inputs": {"base_embedding": str(OUT_ROOT / f"{BASE_NAME}.parquet"),
                   "demographics": str(OUT_ROOT / f"{DEMO_NAME}.parquet")},
        "case": "Case 2 LITERAL: X1 (demographics) + t (Stein-moment 1-D compression "
                "of Z over probes {y, y^2, arctan(ay), ay^2/(1+ay^2)}, orders 1&2)",
        "demographic_features": DEMO_COLS[1:],
        "probes": [p for p, _ in build_probes()],
        "a_values": A_VALUES,
        "references": {"case2_ridge_score": REF_CASE2, "case1_x_plus_z": REF_CASE1,
                       "medm2t_mae_days": MEDM2T_MAE},
        **res,
        "method_note": "Literal Stein-Encoder: explicit first/second-order moments "
                       "E[T(Y)W] and E[T(Y)(WW^T-I)] on PCA-whitened residual W; "
                       "gamma=closed-form (moment vector / leading eigenvector); "
                       "probe&order picked by val MAE. Contrast step 14, which "
                       "replaces the moment with a ridge regression of Y on Z'.",
        "whitening_note": f"PCA-whiten on {res['pca_components']} train components "
                          "(full 1536-dim covariance is singular/unstable to invert).",
        "leakage_note": "Box-Cox lambda, target standardization, OneHot, emb scaler, "
                        "residualizer, PCA-whitener, gamma ALL train-only; gamma frozen "
                        "before t on val/test; probe/order picked on val. No leaky 05a "
                        "cols, no llm_point_estimate/llm_answer.",
    }
    (OUT_ROOT / f"{STEP_NAME}__manifest.json").write_text(json.dumps(manifest, indent=2))
    log("wrote manifest"); log("DONE.")

if __name__ == "__main__":
    main()
