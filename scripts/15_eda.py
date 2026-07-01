#!/usr/bin/env python3
"""
Step 15 - Exploratory data analysis of the ICU LoS cohort (report, not a pipeline step).

WHAT: characterise the MIMIC-IV cohort and the effect of our preprocessing, for
the master's report. Headline = the LoS distribution (heavily right-skewed). Also
covers the cohort funnel / selection bias, age, categorical demographics, repeat-
patient multiplicity, missingness (tabular + SOFA-narrative coverage), mortality /
split sanity, and a light LoS-vs-covariate bivariate pass.

Replaces the five one-off diagnostics (check_icu_los_dist, anchor_age_distrib,
check_multiplicity, check_anchor_offset, check_unmatched_icu) with one organised
script. Self-contained: pandas / numpy / scipy.stats / matplotlib (Agg) only --
no sklearn, no _common import (analysis is a different concern from modelling).

OUTPUTS (all under data/, which is gitignored -- aggregate, but kept local until a
scrubbed report is deliberately promoted):
  data/eda/15_eda_report.md   - stats tables + embedded figures
  data/eda/figs/*.png         - figures
  data/15_eda__manifest.json  - inputs, headline stats, caveats

INPUTS:
  data/07_split.parquet              - 51,834 stays: demographics + icu_los_days +
                                       remaining_los_days + split (primary frame)
  data/09b_los_strictly_positive.parquet - 51,704: + clinical_text (target + SOFA coverage)
  data/02_cohort_features.parquet    - 545,848 admissions (subject->hadm multiplicity)
  mimic-iv-3.1/icu/icustays.csv.gz   - raw ICU stays (hadm->stay, subject->stay)
  data/*__manifest.json, data/_dropped_rows/*.parquet - cohort funnel counts

DELIBERATELY DOES NOT: model anything; suppress small cells (not needed locally --
only required if an aggregate report is later committed); analyse ID columns.
"""
import json
import re
import datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/pierrectrd/LoS project")
OUT_ROOT = ROOT / "data"
RAW_ROOT = ROOT / "mimic-iv-3.1"
EDA = OUT_ROOT / "eda"
FIGS = EDA / "figs"
STEP_NAME = "15_eda"

CAT_COLS = ["gender", "race", "insurance", "marital_status",
            "admission_type", "first_careunit"]

# SOFA variables as they are spelled in the rendered narrative (step 09). Presence
# is detected by substring/regex; a variable not measured in 24h is simply absent.
SOFA_VARS = {
    "FiO2":         r"FiO2",
    "SpO2":         r"SpO2",
    "ventilation":  r"mechanically ventilated",
    "MAP":          r"\bMAP\b",
    "vasopressors": r"vasopressor",
    "Creatinine":   r"Creatinine",
    "Bilirubin":    r"Bilirubin",
    "Platelets":    r"Platelets",
}

def log(msg): print(f"[{dt.datetime.now():%H:%M:%S}] {msg}")

# --------------------------- report / figure helpers ---------------------------

_R = []  # accumulated markdown
def h(text, lvl=2): _R.append(f"\n{'#'*lvl} {text}\n")
def p(text):        _R.append(text + "\n")
def fig_md(name):   _R.append(f"\n![{name}](figs/{name})\n")

def fmt(x):
    if isinstance(x, float):
        return f"{x:,.2f}" if abs(x) < 1e4 else f"{x:,.0f}"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    return str(x)

def md_table(df, index=True):
    """Minimal markdown table (avoids a tabulate dependency)."""
    df = df.copy()
    if index:
        df = df.reset_index()
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |",
           "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt(v) for v in r) + " |")
    return "\n".join(out) + "\n"

def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(FIGS / name, dpi=110, bbox_inches="tight")
    plt.close(fig)
    fig_md(name)

# --------------------------------- sofa parser ---------------------------------

def sofa_presence(texts):
    """Boolean DataFrame: one column per SOFA variable, True if it appears in the
    narrative. Vectorised str.contains over the whole column."""
    s = pd.Series(texts).fillna("")
    return pd.DataFrame({v: s.str.contains(pat, regex=True) for v, pat in SOFA_VARS.items()})

# ----------------------------------- sections -----------------------------------

def sec_funnel():
    """Cohort funnel from manifests + parquet lengths + _dropped_rows counts."""
    h("1. Cohort funnel & selection bias")
    man = lambda n: json.load(open(OUT_ROOT / f"{n}__manifest.json"))
    nrows = lambda n: pq.ParquetFile(OUT_ROOT / f"{n}.parquet").metadata.num_rows
    dropn = lambda n: pq.ParquetFile(OUT_ROOT / "_dropped_rows" / f"{n}.parquet").metadata.num_rows

    raw_adm = man("01_cohort")["rows_in_raw_admissions"]
    stages = [
        ("raw hospital admissions",              raw_adm,                      ""),
        ("valid hospital LoS",                   nrows("01_cohort"),           f"-{dropn('01_cohort__invalid_los')} invalid/neg LoS"),
        ("adults (age >= 18)",                   nrows("02_cohort_features"),  f"-{dropn('02_cohort_features__under_18')} under 18"),
        ("cleaned ICU stays",                    nrows("03_icu_stays_clean"),  f"-{dropn('03_icu_stays__invalid_icu_los')} invalid ICU LoS, -{dropn('03c_integrity__icu_stays_unmatched_hadm')} orphan hadm"),
        ("first ICU stay/patient, >= 24h",       nrows("04_icu_los_cohort"),   f"-{dropn('04_icu_los_cohort__non_first_icu_stays')} non-first, -{dropn('04_icu_los_cohort__first_stay_under_24h')} under 24h"),
        ("strictly-positive remaining LoS",      nrows("04_icu_los_cohort") - dropn('09b_los_strictly_positive__zero_remaining_los'), f"-{dropn('09b_los_strictly_positive__zero_remaining_los')} zero remaining LoS"),
        ("survivors (LoS-task cohort, 09b)",     nrows("09b_los_strictly_positive"), f"-{dropn('09b_los_strictly_positive__in_hospital_death')} in-hospital deaths (mortality cohort)"),
    ]
    tab = pd.DataFrame(stages, columns=["stage", "rows", "drop reason"])
    p(md_table(tab, index=False))
    p("**Selection biases introduced:** first-ICU-stay-per-patient only (repeat "
      "stays discarded — see §5); the **>= 24h floor** removes short stays and "
      "truncates the LoS left tail; adults only; age censored at 91 (HIPAA, see §3); "
      "**in-hospital deaths removed at 09b** from the LoS cohort (LoS is truncated by "
      "death, not discharge — they are the separate mortality-task cohort, §7).")

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh([s[0] for s in stages][::-1], [s[1] for s in stages][::-1], color="#4878a8")
    ax.set_xlabel("rows"); ax.set_title("Cohort funnel (admissions -> modelling cohort)")
    for i, s in enumerate(stages[::-1]):
        ax.text(s[1], i, f" {s[1]:,}", va="center", fontsize=8)
    save_fig(fig, "01_funnel.png")
    return {"raw_admissions": int(raw_adm), "final_cohort": int(nrows("07_split")),
            "strictly_positive": int(nrows("09b_los_strictly_positive"))}

def describe_series(x, pcts=(1, 5, 10, 25, 50, 75, 90, 95, 99)):
    x = pd.Series(x).dropna()
    d = {"n": len(x), "mean": x.mean(), "std": x.std(), "min": x.min(), "max": x.max(),
         "skew": stats.skew(x), "kurtosis": stats.kurtosis(x)}
    for q in pcts:
        d[f"p{q}"] = np.percentile(x, q)
    return pd.Series(d)

def sec_los(df, df09b):
    """The headline: ICU LoS and the remaining-LoS target, raw + transformed."""
    h("2. ICU length-of-stay distribution (headline)")
    icu = df["icu_los_days"].to_numpy(float)
    rem = df09b["remaining_los_days"].to_numpy(float)   # strictly positive (51,704)

    tab = pd.concat([describe_series(icu).rename("icu_los_days (51,834)"),
                     describe_series(rem).rename("remaining_los_days (51,704)")], axis=1)
    p(md_table(tab))
    p(f"**Right-skew confirmed:** icu_los_days skew = {stats.skew(icu):.2f}, "
      f"median {np.median(icu):.2f} d < mean {icu.mean():.2f} d. "
      "`remaining_los_days = icu_los_days − 1`; the >= 24h floor means icu_los >= 1, "
      "so the target starts at 0 (130 exact-zero stays dropped at step 09b).")

    rows = []
    bd = icu.sum()
    for thr in (7, 14, 21, 30):
        share_stays = (icu > thr).mean()
        share_beddays = icu[icu > thr].sum() / bd
        rows.append((f"> {thr} d", f"{share_stays:.1%}", f"{share_beddays:.1%}"))
    p("**Tail concentration (icu_los_days):**\n")
    p(md_table(pd.DataFrame(rows, columns=["stay length", "% of stays", "% of ICU bed-days"]), index=False))
    top10_share = icu[icu >= np.percentile(icu, 90)].sum() / bd
    p(f"The longest-staying **10%** of patients consume **{top10_share:.0%}** of all "
      "ICU bed-days — the operational cost of the tail.")

    # Box-Cox (descriptive lambda on the FULL strictly-positive target; NOT the
    # train-fold lambda used for modelling -- labelled as such).
    rem_bc, lam = stats.boxcox(rem)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].hist(np.clip(icu, 0, 30), bins=60, color="#4878a8")
    axes[0, 0].set_title("icu_los_days (clipped at 30 d)"); axes[0, 0].set_xlabel("days")
    axes[0, 1].hist(np.log1p(rem), bins=60, color="#6acc64")
    axes[0, 1].set_title("log1p(remaining_los_days)"); axes[0, 1].set_xlabel("log1p days")
    axes[1, 0].hist(rem_bc, bins=60, color="#ee854a")
    axes[1, 0].set_title(f"Box-Cox(remaining_los_days), descriptive lambda={lam:.3f}")
    xs = np.sort(rem); axes[1, 1].plot(xs, np.linspace(0, 1, len(xs)), color="#956cb4")
    axes[1, 1].set_title("ECDF remaining_los_days"); axes[1, 1].set_xlabel("days")
    axes[1, 1].set_xlim(0, 30)
    save_fig(fig, "02_los_distribution.png")
    return {"icu_los_skew": float(stats.skew(icu)), "icu_los_median": float(np.median(icu)),
            "remaining_los_median": float(np.median(rem)),
            "top_decile_beddays_share": float(top10_share),
            "descriptive_boxcox_lambda": float(lam)}

def sec_age(df):
    h("3. Age distribution")
    age = df["age_at_admission"].to_numpy(float)
    p(md_table(describe_series(age).to_frame("age_at_admission")))
    bins = list(range(10, 100, 10)) + [200]
    labels = [f"{b}-{b+9}" for b in range(10, 90, 10)] + ["90+"]
    binned = pd.cut(age, bins=bins, right=False, labels=labels).value_counts().sort_index()
    p(md_table(binned.to_frame("count")))
    share91 = float((age == 91).mean())
    p(f"**Age-91 censor spike:** {share91:.1%} of stays sit at exactly 91 — the HIPAA "
      "over-89 sentinel (all ages > 89 collapsed to 91), not a real modal age.")
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.hist(age, bins=np.arange(15, 96, 1), color="#4878a8")
    ax.axvline(91, color="crimson", ls="--", lw=1, label="91 = >89 censor")
    ax.set_xlabel("age at admission"); ax.set_title("Age distribution"); ax.legend()
    save_fig(fig, "03_age.png")
    return {"age_median": float(np.median(age)), "share_at_91_censor": share91}

def sec_categoricals(df):
    h("4. Categorical demographics")
    n = len(df)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.ravel(), CAT_COLS):
        vc = df[col].fillna("(missing)").value_counts()
        top = vc.head(8)
        tab = pd.DataFrame({"count": top, "%": (top / n * 100)})
        p(f"**{col}** ({df[col].isna().mean():.1%} missing, {df[col].nunique()} levels):\n")
        p(md_table(tab))
        ax.barh(top.index[::-1].astype(str), top.values[::-1], color="#4878a8")
        ax.set_title(col, fontsize=10)
    save_fig(fig, "04_categoricals.png")

def sec_multiplicity(df):
    """How many patients have several stays -- framed as what first-stay-only drops."""
    h("5. Repeat patients / multiplicity")
    adm = pd.read_parquet(OUT_ROOT / "02_cohort_features.parquet", columns=["subject_id", "hadm_id"])
    icu = pd.read_csv(RAW_ROOT / "icu" / "icustays.csv.gz", dtype=str,
                      usecols=["subject_id", "hadm_id", "stay_id"])

    def summarise(counts, unit, per):
        tot = len(counts); multi = int((counts > 1).sum())
        p(f"**{unit} per {per}:** {tot:,} {per}s; {multi:,} ({multi/tot:.1%}) have >1; "
          f"max {int(counts.max())}, mean {counts.mean():.2f}.\n")
        dist = counts.value_counts().sort_index().head(8)
        return dist.rename(f"{unit}/{per}")

    d1 = summarise(adm.groupby("subject_id")["hadm_id"].nunique(), "hospital admissions", "patient")
    d2 = summarise(icu.groupby("subject_id")["stay_id"].nunique(), "ICU stays", "patient")
    d3 = summarise(icu.groupby("hadm_id")["stay_id"].nunique(), "ICU stays", "admission")
    p(md_table(pd.concat([d1, d2, d3], axis=1).fillna(0).astype(int)))
    p("Our cohort keeps the **first ICU stay per patient only**, so every >1 above is "
      "data the model never sees — a deliberate independence choice with a recall cost.")

    fig, ax = plt.subplots(figsize=(7, 3.2))
    spc = icu.groupby("subject_id")["stay_id"].nunique().value_counts().sort_index().head(8)
    ax.bar(spc.index.astype(str), spc.values, color="#4878a8")
    ax.set_xlabel("ICU stays per patient"); ax.set_ylabel("patients")
    ax.set_title("ICU stays per patient (raw, pre-first-stay filter)")
    save_fig(fig, "05_multiplicity.png")
    spc_full = icu.groupby("subject_id")["stay_id"].nunique()
    return {"pct_patients_multi_icu_stay": float((spc_full > 1).mean())}

def sec_missingness(df, df09b):
    h("6. Missingness")
    miss = (df[CAT_COLS + ["age_at_admission"]].isna().mean() * 100).sort_values(ascending=False)
    p("**Tabular missingness (cohort columns, % NaN):**\n")
    p(md_table(miss.round(2).to_frame("% missing")))

    pres = sofa_presence(df09b["clinical_text"])
    rate = (pres.mean() * 100).sort_values(ascending=False)
    p("\n**SOFA-narrative coverage** — a variable not measured in the first 24h is "
      "simply absent from the text (MNAR: absence is informative). Per-variable "
      "presence rate across 51,704 narratives:\n")
    p(md_table(rate.round(1).to_frame("% of stays present")))
    per_stay = pres.sum(axis=1)
    p(f"\nVariables present per stay: mean {per_stay.mean():.2f} of {len(SOFA_VARS)} "
      f"(min {per_stay.min()}, max {per_stay.max()}).")

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    axes[0].barh(rate.index[::-1], rate.values[::-1], color="#6acc64")
    axes[0].set_xlabel("% of stays present"); axes[0].set_title("SOFA variable presence")
    vc = per_stay.value_counts().sort_index()
    axes[1].bar(vc.index.astype(str), vc.values, color="#4878a8")
    axes[1].set_xlabel("# SOFA variables present"); axes[1].set_ylabel("stays")
    axes[1].set_title("Narrative richness per stay")
    save_fig(fig, "06_missingness.png")
    return {"mean_sofa_vars_present": float(per_stay.mean())}

def sec_mortality_split(df):
    h("7. Mortality & split sanity")
    death = df["in_hospital_death"].fillna(0).astype(float)
    p(f"**In-hospital death rate:** {death.mean():.1%} of the full cohort "
      f"({int(death.sum()):,} stays). LoS is truncated by death (time-to-death, not "
      "time-to-discharge), so these are **removed at 09b** from the LoS-task cohort "
      "and reserved as the separate mortality-task cohort. (This figure characterises "
      "the full 51,834 cohort; the LoS model trains on the death-free 09b subset.)\n")
    g = df.groupby("split")["remaining_los_days"]
    sp = pd.DataFrame({"n": g.size(), "%": g.size() / len(df) * 100,
                       "median": g.median(), "p25": g.quantile(.25), "p75": g.quantile(.75)})
    p("**Split proportions + target spread per split** (comparable medians => the "
      "patient-level split did not skew the target):\n")
    p(md_table(sp.loc[["train", "val", "test"]]))
    return {"in_hospital_death_rate": float(death.mean())}

def sec_bivariate(df):
    h("8. LoS vs key covariates (median remaining days)")
    df = df.copy()
    df["age_band"] = pd.cut(df["age_at_admission"], [18, 40, 55, 70, 85, 200],
                            right=False, labels=["18-39", "40-54", "55-69", "70-84", "85+"])
    df["died"] = df["in_hospital_death"].fillna(0).astype(int).map({0: "survived", 1: "died"})
    groups = [("first_careunit", 8), ("age_band", None), ("admission_type", 6),
              ("died", None), ("gender", None)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (col, topn) in zip(axes.ravel(), groups):
        med = df.groupby(col, observed=True)["remaining_los_days"].median().sort_values(ascending=False)
        if topn: med = med.head(topn)
        p(f"**median remaining_los_days by {col}:**\n")
        p(md_table(med.round(2).to_frame("median days")))
        ax.barh(med.index[::-1].astype(str), med.values[::-1], color="#ee854a")
        ax.set_title(f"by {col}", fontsize=10)
    axes.ravel()[-1].axis("off")
    save_fig(fig, "08_los_by_covariate.png")

# ----------------------------- the one runnable check ----------------------------

def selftest():
    """Exercise the SOFA parser on a synthetic narrative with a known subset present."""
    txt = ("70-year-old man, ew emer. admission, MICU.\n"
           "Respiratory: FiO2 60% (min 40, max 80); SpO2 98->93%.\n"
           "Cardiovascular: MAP 78->64 mmHg.\n"
           "Renal: Creatinine 1.4 mg/dL.\n"
           "Coagulation: Platelets 150 K/uL.")  # NO vent, NO vasopressor, NO bilirubin
    flags = sofa_presence([txt]).iloc[0]
    for v in ("FiO2", "SpO2", "MAP", "Creatinine", "Platelets"):
        assert flags[v], f"{v} should be present"
    for v in ("ventilation", "vasopressors", "Bilirubin"):
        assert not flags[v], f"{v} should be absent"
    assert int(flags.sum()) == 5, "expected exactly 5 SOFA variables present"
    log("selftest OK (SOFA parser detects present/absent variables correctly)")

# --------------------------------- real run ---------------------------------

def main():
    selftest()
    FIGS.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(OUT_ROOT / "07_split.parquet")           # 51,834: demo + target + split
    df09b = pd.read_parquet(OUT_ROOT / "09b_los_strictly_positive.parquet")  # 51,704: + clinical_text
    log(f"loaded cohort {df.shape}, strictly-positive {df09b.shape}")

    _R.append(f"# EDA — ICU LoS cohort (MIMIC-IV v3.1)\n")
    _R.append(f"_generated {dt.datetime.now():%Y-%m-%d %H:%M}_\n")
    stats_out = {}
    stats_out.update(sec_funnel())
    stats_out.update(sec_los(df, df09b))
    stats_out.update(sec_age(df))
    sec_categoricals(df)
    stats_out.update(sec_multiplicity(df))
    stats_out.update(sec_missingness(df, df09b))
    stats_out.update(sec_mortality_split(df))
    sec_bivariate(df)

    (EDA / "15_eda_report.md").write_text("\n".join(_R))
    log(f"wrote report -> {EDA / '15_eda_report.md'}")

    manifest = {
        "step": STEP_NAME,
        "run_at": dt.datetime.now().isoformat(timespec="seconds"),
        "inputs": {"cohort": str(OUT_ROOT / "07_split.parquet"),
                   "strictly_positive": str(OUT_ROOT / "09b_los_strictly_positive.parquet"),
                   "admissions": str(OUT_ROOT / "02_cohort_features.parquet"),
                   "raw_icustays": str(RAW_ROOT / "icu" / "icustays.csv.gz")},
        "outputs": {"report": str(EDA / "15_eda_report.md"), "figures_dir": str(FIGS)},
        "headline_stats": stats_out,
        "caveats": "Aggregate only; outputs kept in gitignored data/eda/. No small-cell "
                   "suppression (needed only if promoted to git). Box-Cox lambda here is "
                   "descriptive (full cohort), NOT the train-fold modelling lambda.",
    }
    (OUT_ROOT / f"{STEP_NAME}__manifest.json").write_text(json.dumps(manifest, indent=2))
    log("wrote manifest"); log("DONE.")

if __name__ == "__main__":
    main()
