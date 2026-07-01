# Results — death-free survivor cohort (≈8.9k)

> **Cohort correction (this supersedes the earlier numbers in this file).** Until
> now, in-hospital deaths were never removed from the LoS cohort — only kept out of
> the feature matrix. LoS for a patient who dies is truncated by *death*, not
> *discharge*, so ~10.8% of every prior result mixed a time-to-death outcome into a
> time-to-discharge task. Deaths are now excluded at the source (`09b`, survivors
> only). Every number below is **re-run on the death-free cohort**; the old
> death-contaminated values are shown in the last column for reference.

Cohort = the **8,923 survivors** among the first 10,000 stays of the LLM cache
(10,000 − 1,077 deaths). Patient-level split **5,618 train / 1,489 val / 1,816
test**. Primary metric: **test MAE in days** (lower = better); Case 5 adds CRPS.
Benchmark: MedM2T **2.31** (see comparability caveat).

## Test MAE — death-free vs death-contaminated

| Case | Method | Model | **death-free test** | old (w/ deaths) |
|------|--------|-------|:---:|:---:|
| 3 | X only (demographics) | LR | 2.49 | 2.65 |
| 3 | X only | RF | 2.47 | 2.64 |
| 4 | Z only (LLM's own day estimate) | — | 2.66 | 2.82 |
| **1** | **X + Z (1536-dim embedding)** | **Ridge** | **2.356** | 2.53 |
| 1 | X + Z | RF | 2.357 | 2.52 |
| 2 | X + 1 ridge-compressed score | LR | 2.367 | 2.53 |
| 2 | X + 1 ridge-compressed score | RF | 2.370 | 2.58 |
| **2b** | **X + 1 literal Stein-moment score** | **LR** | **2.337** | 2.50 |
| 2b | X + 1 literal Stein-moment score | RF | 2.354 | 2.51 |
| 5 | DIM + IDR, Z only (median) | IDR | 2.409 | 2.58 |
| 5 | DIM + IDR, X + Z (median) | IDR | 2.415 | 2.57 |

**Best: Case 2b (literal Stein) — 2.337d**, with Case 1 (X+Z, 2.356) and Case 2b RF
(2.354) a hair behind; all clustered within fold noise and now **essentially at the
MedM2T 2.31 bar** (best gap 0.03d). Case 5 (distributional) trails on point MAE
(2.41) but is the only method giving calibrated CDFs and the least-overfit — its own
section below.

## What removing the deaths changed
- **Every method improved by ~0.15–0.20 d.** Deaths are short, death-truncated stays
  the LoS models could never place correctly; they were pure noise/bias in the
  target. Removing them is the single biggest accuracy jump in the project so far —
  bigger than any modelling change — and it came from a **cohort-definition fix, not
  a better model.**
- **The ranking held.** X+Z (Case 1) still beats X-only (2.36 vs 2.47) and the LLM's
  own number (2.66); the Stein compressions (Case 2/2b) remain marginally best on
  point MAE. The embedding still carries signal demographics don't.
- **Case 4 (Z-alone) is still the weakest** (2.66): the LLM's raw point estimate
  hedges to the middle; using the embedding *through a model* is better.
- **We are now at the MedM2T bar, not above it.** 2.34 vs 2.31. The remaining gap is
  within fold noise on a 1,816-stay test fold (see caveats on bar comparability).

## Overfitting
Train-vs-test gap for the random forests / IDR (death-free):

| Case | Model | Train | Test | Gap |
|------|-------|:---:|:---:|:---:|
| 1 | RF | 1.16 | 2.36 | **1.19** |
| 2 | RF | 1.83 | 2.37 | **0.54** |
| 2b | RF | 1.98 | 2.35 | **0.38** |
| 5 | IDR (Z-only) | 2.10 | 2.41 | **0.31** |

- **Case 1 RF still overfits hardest** (gap 1.19): the 1536 embedding columns +
  demographic one-hots give the forest room to memorise training stays, undiminished
  by the cleaner cohort.
- **Compression controls it** (Case 2 → 0.54, Case 2b → 0.38) and **IDR controls it
  most** (0.31): squeezing to a 1-D index + a tuning-free isotonic fit is the
  strongest regulariser of the lot.

## Case 5 — distributional forecasts (DIM + IDR)
First **distributional** method (`scripts/16_idr_dim.py`): a Distributional Index
Model + Isotonic Distributional Regression wrapper around the LLM embedding. Stage 1
fits a Ridge **index** g(x) (α picked by val CRPS); stage 2 feeds (index, real days)
to IDR, which returns a full predictive CDF per patient, monotone in the index. Two
settings: **A = embedding only (Z)**, **B = Z + demographics (X)**. Metrics: CRPS
(distributional) and MAE of the predictive **median** (point, comparable above). IDR
via `sklearn.isotonic` (1-D PAVA) — equivalent to the repo's Rust IDR for a scalar
index; the vendored binding is unbuildable in this Py-3.10 venv.

| Setting | α | CRPS train | CRPS val | **CRPS test** | MAE train | MAE val | **MAE test** |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| A — Z only | 1000 | 1.61 | 1.69 | **1.87** | 2.10 | 2.18 | **2.41** |
| B — X + Z | 1000 | 1.60 | 1.70 | **1.87** | 2.09 | 2.19 | **2.42** |

- **Embedding-only ≈ embedding+demographics.** A and B are tied (CRPS 1.870 vs 1.866;
  MAE 2.409 vs 2.415). Adding structured demographics X on top of Z buys nothing —
  the demographics were already fed through the LLM and are **baked into Z**. Setting
  A is not as information-poor as it looks.
- **Least-overfit method we have** (MAE gap ~0.31, CRPS gap ~0.23).
- **Competitive point predictor.** Median-MAE 2.41 beats X-only (2.47) and the LLM's
  own number (2.66), just behind the best point models (Case 1/2b ≈ 2.34–2.36). The
  win is *calibrated distributions*, not a lower MAE.
- **CRPS is not on the MAE ruler.** The 1.87 CRPS scores whole distributions; not
  comparable to the 2.31 bar or the point-MAE rows. Only the median-MAE column is.

## Caveats
- **Cohort is a contiguous head, not a random sample**: the 8,923 survivors of the
  first 10,000 stays of `09b` (kept comparable to earlier runs). A full-cohort run
  (46,203 survivors) is a separate objective.
- **MedM2T 2.31 bar comparability**: their cohort's death handling and sampling may
  differ from ours (now death-free) — treat the 2.31 as an indicative bar, not an
  exact apples-to-apples target.
- **LLM contamination** unaddressed: the model may have seen MIMIC in pretraining.
- **The death fix is the headline**: the ~0.18 d gain is a data-correctness gain, not
  a modelling gain. It also retroactively explains some earlier oddities (e.g. Case 4
  looking unusually weak with deaths in).
- Death-contaminated 10k manifests/numbers are superseded; 1,074-stay manifests
  preserved in `data/_benchmark_1074/`; `RESULTS.md` (1,074-era) is also
  death-contaminated — see its banner.
