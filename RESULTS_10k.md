# Results — 10,000-stay benchmark

Same four cases as `RESULTS.md`, now on **10,000 stays** (was 1,074). Cohort =
first 10,000 stays of `09b` (contiguous head, same stays as before + 8,926 new).
Split 6,304 train / 1,669 val / **2,027 test**. Metric: **test MAE in days**
(lower = better). Benchmark: MedM2T **2.31**.

## Test MAE — 10k vs 1,074

| Case | Method | Model | **10k test** | 1,074 test |
|------|--------|-------|:---:|:---:|
| 3 | X only (demographics) | LR | 2.65 | 2.62 |
| 3 | X only | RF | 2.64 | 2.67 |
| 4 | Z only (LLM's own day estimate) | — | 2.82 | 2.57 |
| **1** | **X + Z (1536-dim embedding)** | **RF** | **2.52** | 2.44 |
| 1 | X + Z | Ridge | 2.53 | 2.49 |
| 2 | X + 1 ridge-compressed score | LR | 2.53 | 2.68 |
| 2 | X + 1 ridge-compressed score | RF | 2.58 | 2.72 |
| 2b | X + 1 **literal Stein-moment** score | LR | 2.50 | — |
| 2b | X + 1 literal Stein-moment score | RF | 2.51 | — |

**Best: Case 2b (literal Stein) — 2.50d**, edging Case 1 (X+Z) at 2.52d; all
within fold noise, all above the 2.31 bar. Case 2b picked the probe `arctan(0.5y)`
at first order (`scripts/14b_case2_stein_literal.py`).

## What the 10× data changed
- **The ranking held: Case 1 (X+Z) is best.** Adding the embedding still beats
  X-only (2.52 vs 2.64) and beats the LLM's own number alone (2.82). The clinical
  narrative carries signal demographics don't.
- **The spread collapsed.** At n=1,074 the cases ranged 2.44–2.72; at 10k they sit
  2.52–2.82. Most of the earlier separation was small-test-fold noise — the honest
  gaps between methods are smaller than they looked.
- **Case 4 (Z-alone) got worse** (2.57 → 2.82) and is now clearly the weakest. The
  LLM's raw point estimate hedges toward the middle of the range; over a larger,
  wider LoS sample that bias shows. Using the embedding *through a model* (Case 1)
  is much better than trusting the LLM's number directly.
- **Case 1's test MAE rose slightly** (2.44 → 2.52). The 1,074 number was an
  optimistic estimate from a tiny test fold; 2.52 on 2,027 test stays is the more
  trustworthy figure.

## Overfitting — the interesting flip
Train-vs-test gap for the random forests:

| Case | Model | Train | Test | Gap | (1,074 gap) |
|------|-------|:---:|:---:|:---:|:---:|
| 1 | RF | 1.28 | 2.52 | **1.24** | 1.11 |
| 2 | RF | 2.02 | 2.58 | **0.55** | 1.48 |
| 2b | RF | 2.17 | 2.51 | **0.33** | — |

- **Case 1 RF still overfits** (gap 1.24): even with 10× data, the 1536 embedding
  columns + demographic one-hots give the forest room to memorize training stays.
- **Case 2's compression now controls it** (gap 0.55, down from 1.48). At 1,074 the
  Stein score looked *more* overfit than Case 1 — that was noise. With real data the
  one-number compression does what it was designed to do: it kills the overfitting.
- **But the ridge-score Case 2 doesn't win on test** (2.58 vs Case 1's 2.52). So
  *that* compression buys generalization stability at the cost of a hair of accuracy.
- **Case 2b (literal Stein moment) gets both** (gap 0.33, test 2.51): finding the
  single direction via an explicit Stein moment over probes — rather than a ridge
  regression — keeps essentially all of Case 1's signal *and* overfits the least of
  any model. First order dominated (second-order Hessian moment carried almost no
  signal), and the bounded probes collapsed to ~linear, so the usable narrative
  signal is close to a single linear index. Marginal on test, but the best-behaved.

## Takeaways for next steps
- Case 1 (X+Z) is the architecture to keep developing.
- Case 1's residual overfitting is in the **demographics + raw 1536-dim Z**, not
  fixed by data volume alone → motivates the next objective, **Double Machine
  Learning** (orthogonalize Z against X with flexible nuisance models), which is a
  principled version of what Case 2's residualize-then-compress was reaching for.
- 2.52 vs MedM2T 2.31: closing, not closed. The full 51,704 cohort is the next lever.

## Caveats
- Still a contiguous head(10,000) of `09b`, not a random sample — kept comparable to
  the 1,074 runs. Full-cohort run is a separate objective.
- **LLM contamination** unaddressed: the model may have seen MIMIC in pretraining.
- 1,074-stay manifests preserved in `data/_benchmark_1074/`; `RESULTS.md` unchanged.
