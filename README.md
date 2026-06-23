# ICU Length-of-Stay prediction from clinical narratives (MIMIC-IV v3.1)

Predicting **remaining ICU length of stay** after a patient's first 24 hours,
from a human-readable **SOFA clinical narrative** rendered per stay and fed to a
large language model. The central question: does an LLM's medical world-knowledge,
activated on a meaning-rich narrative, add predictive signal over structured
demographics alone?

Master's research project. The deliverable is a *prediction* model; parameter
inference is pursued only where it is honestly available.

> ⚠️ **No patient data is included in this repository.** All results were
> produced on **MIMIC-IV v3.1**, a **credentialed-access** dataset distributed by
> [PhysioNet](https://physionet.org/content/mimiciv/) under a Data Use Agreement
> that prohibits redistribution. To reproduce, you must complete your own
> PhysioNet credentialing, download MIMIC-IV yourself, and point the scripts at
> it. The `data/` outputs, the raw dataset, and any patient-level derivative are
> intentionally **not** published here.

## Approach in one picture

```
first 24h of ICU  ──09_clinical_text_sofa──▶  clinical narrative (text, X)
                                                      │
                              ┌───────────────────────┤
                              ▼                        ▼
                   structured demographics       LLM (gpt-5.4-mini)
                          (X, one-hot)            ├─ point estimate of remaining days
                              │                   └─ free-text reasoning
                              │                        │ text-embedding-3-small
                              │                        ▼
                              │                   Z = 1536-dim embedding
                              └──────────┬─────────────┘
                                         ▼
                           Ridge / RandomForest  →  remaining_los_days
```

**Target:** `remaining_los_days = icu_los_days − 1` (days remaining after the
first 24h), Box-Cox transformed at modelling time (λ fit on the train fold only).
**Metric:** mean absolute error (MAE) in days; the MAE-optimal point estimate is
the median, so the LLM is explicitly asked for the median.

## The clinical narrative

Each ICU stay is rendered to one short paragraph: a demographic header followed
by up to five SOFA organ systems (Respiratory, Cardiovascular, Renal, Liver,
Coagulation), organ-support first, each variable condensed to a `first→last`
trend with `min/max` rather than raw timestamps. A variable not measured in the
first 24h is simply **absent** from the narrative (informative, not-at-random
missingness). Variables are sourced from MIMIC `chartevents`, `inputevents`, and
`labevents` (labs cleaned with the MedM2T value-mapping), all windowed to the
first 24h of the ICU stay.

A **synthetic, fully fictional** example (invented numbers, no real patient — for
format illustration only):

```
67-year-old man, ew emer. admission, Medical Intensive Care Unit (MICU).
Respiratory: FiO2 60% (min 40, max 80); SpO2 98→93% (min 88, max 99); mechanically ventilated.
Cardiovascular: MAP 78→64 mmHg (min 52, max 88); on norepinephrine (1 vasopressor, up to 5.0h).
Renal: Creatinine 1.4→2.1 mg/dL (min 1.4, max 2.1).
Liver: Bilirubin 1.1 mg/dL.
Coagulation: Platelets 150→120 K/uL (min 120, max 160).
```

## The LLM prompt

System: *"You are an experienced intensivist."*
User (abridged):

> A patient's first 24 hours in the ICU are summarized below. Estimate how many
> additional days they will remain in the ICU after this point, and explain your
> reasoning. Your estimate is scored by mean absolute error, which is minimized by
> the median outcome — so report the median number of additional days, not the
> mean. End your answer with a final line in exactly this format:
> `ESTIMATE_DAYS: <number>` … *{clinical_text}*

The trailing `ESTIMATE_DAYS:` number is parsed as a standalone baseline; the full
answer is embedded into the 1536-dim vector `Z`.

## Pipeline (`scripts/`)

| Step | Script | What it does |
|---|---|---|
| 01 | `01_build_cohort.py` | one row per hospital admission; length-of-stay; drop invalid |
| 02 | `02_cohort_features.py` | age correction; drop leaky timestamps |
| 03 | `03a/03b/03c_*.py` | ICU stays, cleaned; integrity asserts |
| 04 | `04a_icu_los_cohort.py` | first ICU stay per patient, ≥24h, 24h window |
| 05 | `05a_target.py`, `05b_item_dictionary.py` | add target; itemid → label/unit dictionary |
| 07 | `07_split.py` | patient-level train/val/test split (no subject overlap) |
| 09 | `09_clinical_text_sofa.py` | **render the SOFA clinical narrative** (the live text) |
| 09b | `09b_los_strictly_positive.py` | drop zero-LoS stays → strictly positive target |
| 11 | `11_baselines_xonly.py` | **Case 3** — demographics only → LR / RF |
| 12 | `12_baseline_llm_float.py` | **Case 4** — LLM point estimate + answer embedding `Z` |
| 13 | `13_case1_x_plus_z.py` | **Case 1** — X + Z (1536-dim) → Ridge / RF (primary) |
| 14 | `14_case2_stein_inspired.py` | **Case 2** — residualize Z against X, compress to one score |

Utilities: `peek.py` (parquet schema/head), `showtext.py` (read a text column);
`check_*.py` / `*_distrib.py` are exploratory sanity checks.

**Leakage discipline:** the split is by patient; everything fitted (Box-Cox λ,
one-hot encoders, scalers, embedding compression) is fit on the **train fold
only** and applied unchanged to val/test. No information past hour 24 is used.

## Results

See [`RESULTS.md`](RESULTS.md) (1,074-stay directional run) and
[`RESULTS_10k.md`](RESULTS_10k.md) (10,000-stay benchmark). Headline: adding the
LLM embedding (Case 1, X+Z) is the best configuration at **2.52d** test MAE on
10,000 stays, beating demographics-only (2.64d) and the LLM's own number alone
(2.82d); the reference benchmark (MedM2T) is 2.31d.

## Reproducing

1. Obtain MIMIC-IV v3.1 from PhysioNet (credentialed) and place it at
   `mimic-iv-3.1/`.
2. Create a virtualenv and install dependencies
   (`pandas`, `pyarrow`, `scikit-learn`, `scipy`, `openai`, `python-dotenv`).
3. Provide an OpenAI API key in a `.env` file at the repo root
   (`OPENAI_API_KEY=...`). **`.env` is git-ignored and must never be committed.**
4. Run the scripts in numeric order; each reads a numbered input and writes a
   numbered `.parquet` + a `__manifest.json` recording inputs, row counts, and
   decisions.

## License

Code released under the [MIT License](LICENSE). This license covers only the code
in this repository; MIMIC-IV's own access terms are separate and unaffected.
