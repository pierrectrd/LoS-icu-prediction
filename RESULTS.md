# Results so far — ICU remaining-LoS prediction

> ⚠️ **Superseded & cohort-contaminated.** These 1,074-stay numbers were computed
> *with in-hospital deaths still in the cohort* (a bug fixed later — deaths are
> truncated by death, not discharge, and belong to the separate mortality task; they
> are now removed at `09b`). They are also a tiny directional-only test fold. See
> `RESULTS_10k.md` for the corrected, death-free results. Kept here for history only.

**Cohort:** first 1,074 stays of the cohort (capped run — *directional only*, small
test fold). Target: `remaining_los_days` after first 24h. Metric: **MAE in real
days** (lower = better). Benchmark: MedM2T **2.31d**.

## Test MAE by case

| Case | What it uses | Model | Test MAE |
|------|--------------|-------|----------|
| 3 (baseline) | X only (7 demographics) | RF | 2.67 |
| 4 (baseline) | Z only (LLM's own day estimate) | — | 2.57 |
| **1 (main)** | X + Z (1536-dim embedding) | **RF** | **2.44** |
| 1 | X + Z | Ridge | 2.49 |
| 2 | X + 1 score `t` (Stein-style compression of Z) | LR | 2.68 |
| 2 | X + 1 score `t` | RF | 2.72 |

**Best so far: Case 1 (X+Z), RF — 2.44d.** Still above the 2.31 benchmark, but the
embedding clearly adds signal over demographics alone (2.67 → 2.44).

## What worked / what didn't
- **Adding the LLM embedding helps** (Case 1 beats both baselines). The clinical
  narrative carries information demographics don't.
- **Compressing the embedding to one number hurts** (Case 2 worse than Case 1).
  Squeezing 1536 dims into a single supervised score threw away signal — and it
  did *not* fix the overfitting (see below). The fancy architecture lost to the
  simple "just concatenate everything" of Case 1.

## On overfitting
A model **overfits** when it memorizes the training data instead of learning
patterns that generalize. The tell is a **large gap between train and test error**:
great on data it has seen, poor on data it hasn't.

We see exactly that here:

| Case | Model | Train MAE | Test MAE | Gap |
|------|-------|-----------|----------|-----|
| 1 | RF | 1.33 | 2.44 | **1.11** |
| 2 | RF | 1.24 | 2.72 | **1.48** |

Both random forests predict training stays ~1.3 days off but test stays ~2.5 days
off — they've partly memorized. Crucially, **Case 2's compression did not shrink
the gap.** That tells us the overfitting wasn't caused by the 1536 wide embedding
columns; it comes from the **55 one-hot demographic columns**, which give the
forest enough room to memorize individual stays regardless of the embedding.

**Implication:** the fix isn't compressing Z — it's more data (the full ~52k
cohort instead of 1,074) and/or regularizing the demographics, not the embedding.

## Caveats
- All numbers are on a **1,074-stay slice** → directional, not decisive. The test
  fold is tiny, which is itself why the RF gap looks large.
- **LLM contamination:** the model may have seen MIMIC during pretraining; its
  estimates could be optimistic. Known risk, not yet mitigated.

## Next
Run the full cohort (steps 12 → 11/13/14) for a verdict that actually counts.
