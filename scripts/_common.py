#!/usr/bin/env python3
"""
_common.py - shared scaffolding for the modelling scripts (steps 11, 13, 14, 14b).

Those four scripts all do the SAME things around their differing core: pull the
same 7 demographics from 05a (leak-guarded), one-hot them, Box-Cox the target
(lambda on TRAIN only), invert predictions back to real days for MAE, and tune a
small RF grid by val MAE. That machinery used to be copy-pasted into each script;
it lives here once so the leakage discipline is defined in EXACTLY ONE place.

Every function is pure (no disk I/O) so each script's selftest still exercises
the real logic on synthetic data. Imported only by the modelling steps -- the
lightweight utilities (peek.py, showtext.py) deliberately do NOT import this, to
avoid dragging sklearn into a parquet viewer.
"""
from pathlib import Path
import datetime as dt
import numpy as np
import pandas as pd
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.ensemble import RandomForestRegressor

OUT_ROOT = Path("/home/pierrectrd/LoS project/data")
SEED = 42
MEDM2T_MAE = 2.31  # MedM2T full-multimodal LoS MAE (days); our reference bar

NUM_COL  = "age_at_admission"
CAT_COLS = ["first_careunit", "gender", "admission_type",
            "insurance", "marital_status", "race"]
# the ONLY columns taken from 05a (stay_id key + the 7 features) -> leak guard
DEMO_COLS = ["stay_id", NUM_COL] + CAT_COLS
RF_GRID = [(n, d) for n in (200, 500) for d in (None, 10, 20)]


def log(msg): print(f"[{dt.datetime.now():%H:%M:%S}] {msg}")


def assemble(df_base, df05a):
    """Left-join the 7 demographics onto a base frame, on stay_id. 1:1, no inflation.
    Selecting DEMO_COLS drops every other (leaky) 05a column here."""
    out = df_base.merge(df05a[DEMO_COLS], on="stay_id", how="left")
    assert len(out) == len(df_base), "row inflation on join (stay_id not 1:1)"
    return out


def split_parts(df):
    """Split a frame into train/val/test sub-frames; assert none is empty."""
    parts = {s: df[df["split"] == s] for s in ("train", "val", "test")}
    assert all(len(p) for p in parts.values()), "a split is empty"
    return parts


def make_demo(frame, ohe, fit=False):
    """One-hot the 6 categoricals + raw numeric age -> dense X1 block. Missing
    categorical -> explicit 'Unknown' level; age passed through unscaled."""
    cat = frame[CAT_COLS].fillna("Unknown")
    if fit:
        ohe.fit(cat)
    age = frame[[NUM_COL]].to_numpy(dtype=float)
    return np.hstack([age, ohe.transform(cat)])


def boxcox_target(parts):
    """Box-Cox the target with lambda fit on TRAIN ONLY, applied to val/test.
    Returns (y_t, y_real, lam): transformed-space dict, real-day dict, lambda."""
    y_real = {s: parts[s]["remaining_los_days"].to_numpy(float) for s in parts}
    assert y_real["train"].min() > 0, "train target not strictly positive (Box-Cox needs >0)"
    y_tr_t, lam = boxcox(y_real["train"])
    y_t = {"train": y_tr_t,
           "val":  boxcox(y_real["val"],  lmbda=lam),
           "test": boxcox(y_real["test"], lmbda=lam)}
    return y_t, y_real, lam


def mae_real(model, X, y_real, lam):
    """Predict in Box-Cox space, invert to days, MAE vs raw real-day target."""
    pred = inv_boxcox(model.predict(X), lam)
    # ponytail: inv_boxcox is nan outside its domain (linear models extrapolate);
    # floor non-finite + negatives to 0 days. ceiling: crude tail clip; upgrade:
    # clip the transformed pred to the Box-Cox domain before inverting.
    pred = np.clip(np.where(np.isfinite(pred), pred, 0.0), 0.0, None)
    return float(np.mean(np.abs(pred - y_real)))


def with_score(demo, t, s):
    """Downstream design for one split: stack the one-hot X1 block + single score t."""
    return np.hstack([demo[s], t[s][:, None]])


def best_rf_by_val(X, y_t_train, y_real, lam):
    """Fit the RF_GRID on X['train'], pick by val MAE-days. Returns (rf, n_est, depth).
    X is a {split: design} dict; only train/val are used to select."""
    best = None
    for n_est, depth in RF_GRID:
        m = RandomForestRegressor(n_estimators=n_est, max_depth=depth,
                                  random_state=SEED, n_jobs=-1).fit(X["train"], y_t_train)
        v = mae_real(m, X["val"], y_real["val"], lam)
        if best is None or v < best[0]:
            best = (v, n_est, depth, m)
    _, n_est, depth, rf = best
    return rf, n_est, depth


def synthetic_demographics(rng, sid, careunits=("MICU", "SICU"),
                           races=("WHITE", "BLACK"), extra_leaky=()):
    """Synthetic 05a-shaped demographics for selftests: the 7 real features plus
    the leaky columns assemble() must drop. Plants a NaN gender on sid[0] to
    exercise the 'Unknown' fill. `extra_leaky` adds further drop-me columns."""
    n = len(sid)
    pick = lambda opts: rng.choice(opts, n)
    df = pd.DataFrame({
        "stay_id": sid,
        NUM_COL: rng.integers(18, 92, n).astype(float),
        "first_careunit": pick(list(careunits)),
        "gender": pick(["M", "F"]),
        "admission_type": pick(["EW EMER.", "ELECTIVE"]),
        "insurance": pick(["Medicare", "Other"]),
        "marital_status": pick(["SINGLE", "MARRIED"]),
        "race": pick(list(races)),
        # leaky columns assemble() must NOT carry through:
        "icu_los_days": rng.random(n), "outtime": "x",
        "in_hospital_death": 0, "last_careunit": "z",
    })
    for col in extra_leaky:
        df[col] = "w"
    df.loc[df["stay_id"] == sid[0], "gender"] = np.nan
    return df
