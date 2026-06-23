#!/usr/bin/env python3
"""
Step 12 - LLM-as-predictor float baseline (Case 4-ish, no embedding model yet).

Per ICU stay, two-stage (same shape as step 10):
  1. GENERATE: clinical_text + prompt -> gpt-5.4-mini -> answer that ENDS with a
     line "ESTIMATE_DAYS: <number>". We regex that float out = llm_point_estimate.
  2. EMBED   : the full answer -> text-embedding-3-small -> 1536-dim vector.

WHAT THIS MEASURES: can the LLM's own point estimate (zero-shot, real days)
predict remaining ICU LoS? MAE is compared directly to the X-only RF baseline
(test 2.59 d, step 11) and MedM2T (2.31 d). The embedding is cached too, but is
NOT used for scoring here -- it is the ingredient for a later case (1/2).

DELIBERATELY DOES NOT:
  - transform the target: the LLM emits real days, so MAE is raw days, no Box-Cox
    (unlike step 11);
  - fit/train anything: this is zero-shot, so there is no train-only discipline
    to enforce -- BUT note the contamination caveat (the model may have seen
    MIMIC); that is a known project risk, not handled here;
  - crash on an unparseable answer: it stores null and logs the stay_id;
  - dedupe, clip, or sanity-bound the estimate.

RESUMABILITY: one JSONL line per stay, keyed by stay_id; cached stays are skipped
on restart. Record = {stay_id, llm_answer, llm_point_estimate, embedding}.

WORKFLOW:
  python3 12_baseline_llm_float.py --selftest   # offline logic check (no API)
  python3 12_baseline_llm_float.py --sample 5   # eyeball answers + ESTIMATE parse
  python3 12_baseline_llm_float.py              # step-12 cohort = first N_STAYS (10k), resumes
  python3 12_baseline_llm_float.py --assemble   # build parquet + score MAE
If 09b's text OR this prompt changed since a cached run, `rm` the cache first
(stale answers would otherwise be reused).

Key: .env -> OPENAI_API_KEY via load_dotenv; never written here.
"""
from pathlib import Path
import json
import re
import time
import argparse
import datetime as dt
import numpy as np
import pandas as pd

OUT_ROOT  = Path("/home/pierrectrd/LoS project/data")
IN_NAME   = "09b_los_strictly_positive"
STEP_NAME = "12_baseline_llm_float"
CACHE     = OUT_ROOT / f"{STEP_NAME}__cache.jsonl"
RUNSTATE  = OUT_ROOT / f"{STEP_NAME}__runstate.json"   # carries temp-drop fact to assemble

CHAT_MODEL  = "gpt-5.4-mini"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM   = 1536
N_STAYS     = 10_000   # token-discipline cap: step-12 cohort = first N stays of 09b

SYSTEM_PROMPT = "You are an experienced intensivist."
# Loss-aware elicitation: we score with MEAN ABSOLUTE error, whose optimal point
# estimate is the MEDIAN of the outcome (squared error would call for the mean).
# So we explicitly ask for the median, aligning the elicited number with the loss.
USER_TEMPLATE = (
    "A patient's first 24 hours in the ICU are summarized below. Estimate how "
    "many additional days they will remain in the ICU after this point, and "
    "explain your reasoning.\n\n"
    "Your estimate is scored by MEAN ABSOLUTE ERROR, which is minimized by the "
    "MEDIAN outcome -- so report the median number of additional days (the value "
    "equally likely to be too high or too low), not the mean.\n\n"
    "End your answer with a final line in exactly this format:\n"
    "ESTIMATE_DAYS: <number>\nwhere <number> is a single number of days "
    "(not a range).\n\n{clinical_text}"
)

# reference bars for the printed table (days)
RF_TEST_MAE = 2.59
MEDM2T_MAE  = 2.31

MAX_RETRIES = 5
ESTIMATE_RE = re.compile(r"ESTIMATE_DAYS:\s*([-+]?\d*\.?\d+)")
STATE = {"drop_temp": False}   # set once if the API rejects temperature=0

def log(msg): print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)

# ---------------------------- pure, testable logic ----------------------------

def parse_estimate(text):
    """Return the float from the LAST 'ESTIMATE_DAYS: <n>' line, or None."""
    if not text:
        return None
    matches = ESTIMATE_RE.findall(text)
    return float(matches[-1]) if matches else None

def build_table(records, meta):
    """Join cached records to target/split metadata; expand embedding wide.

    records: list of {stay_id, llm_answer, llm_point_estimate, embedding}.
    meta:    df with stay_id, subject_id, split, remaining_los_days.
    """
    dim = len(records[0]["embedding"])
    base = pd.DataFrame({
        "stay_id": [r["stay_id"] for r in records],
        "llm_answer": [r["llm_answer"] for r in records],
        "llm_point_estimate": [r["llm_point_estimate"] for r in records],
    })
    emb = pd.DataFrame([r["embedding"] for r in records],
                       columns=[f"emb_{i}" for i in range(dim)])
    feat = pd.concat([base, emb], axis=1)
    meta = meta.copy()
    meta["stay_id"] = meta["stay_id"].astype(str)
    out = meta.merge(feat, on="stay_id", how="inner")
    front = ["stay_id", "subject_id", "split", "remaining_los_days",
             "llm_point_estimate", "llm_answer"]
    return out[front + [f"emb_{i}" for i in range(dim)]]

def score(df):
    """MAE of llm_point_estimate vs remaining_los_days, real days, nulls dropped."""
    valid = df[df["llm_point_estimate"].notna()]
    mae = lambda s: float((s["llm_point_estimate"] - s["remaining_los_days"]).abs().mean())
    test = valid[valid["split"] == "test"]
    return {
        "n_total": int(len(df)),
        "n_null_dropped": int(len(df) - len(valid)),
        "n_test_scored": int(len(test)),
        "test_mae": mae(test) if len(test) else None,
        "whole_mae": mae(valid) if len(valid) else None,
    }

# ------------------------------- API plumbing -------------------------------

def call_with_retry(fn, what):
    """Retry transient API errors with exponential backoff."""
    for attempt in range(MAX_RETRIES):
        try:
            return fn()
        except Exception as e:
            wait = 2 ** attempt
            log(f"  {what} failed (attempt {attempt+1}/{MAX_RETRIES}): {e}; retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"{what} failed after {MAX_RETRIES} retries")

def chat_create(client, messages):
    """Create a chat completion at temperature=0, dropping the param if rejected."""
    kwargs = dict(model=CHAT_MODEL, messages=messages)
    if not STATE["drop_temp"]:
        kwargs["temperature"] = 0
    try:
        return client.chat.completions.create(**kwargs)
    except Exception as e:
        # ponytail: detect the one known param-rejection by message text; anything
        # else propagates to call_with_retry. Ceiling = string match; upgrade =
        # catch the SDK's typed BadRequestError on 'temperature'.
        if not STATE["drop_temp"] and "temperature" in str(e).lower():
            STATE["drop_temp"] = True
            log("API rejected temperature=0; retrying WITHOUT temperature (logged once).")
            kwargs.pop("temperature", None)
            return client.chat.completions.create(**kwargs)
        raise

def process_stay(client, stay_id, clinical_text):
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(clinical_text=clinical_text)}]
    chat = call_with_retry(lambda: chat_create(client, messages), "chat")
    answer = chat.choices[0].message.content
    est = parse_estimate(answer)
    if est is None:
        log(f"  no ESTIMATE_DAYS parsed for stay {stay_id} -> null")
    emb = call_with_retry(lambda: client.embeddings.create(
        model=EMBED_MODEL, input=answer), "embed")
    return {"stay_id": stay_id, "llm_answer": answer,
            "llm_point_estimate": est, "embedding": emb.data[0].embedding}

def load_done_ids():
    done = set()
    if CACHE.exists():
        with open(CACHE) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["stay_id"])
                except Exception:
                    pass
    return done

def run(sample=None):
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv()
    client = OpenAI()
    df = pd.read_parquet(OUT_ROOT / f"{IN_NAME}.parquet")
    df["stay_id"] = df["stay_id"].astype(str)
    df = df.head(N_STAYS)                       # token-discipline cap (first 10k stays)
    log(f"cohort capped to first {len(df)} stays (N_STAYS={N_STAYS})")
    if sample:
        df = df.head(sample)
        log(f"SAMPLE MODE: first {sample} stays")

    done = load_done_ids()
    todo = df[~df["stay_id"].isin(done)]
    log(f"{len(df)} target stays; {len(done)} cached; {len(todo)} to do")

    with open(CACHE, "a") as cache:
        for i, row in enumerate(todo.itertuples(), 1):
            rec = process_stay(client, row.stay_id, row.clinical_text)
            cache.write(json.dumps(rec) + "\n"); cache.flush()
            if sample:
                log(f"--- stay {row.stay_id} (true remaining={row.remaining_los_days}d, "
                    f"parsed estimate={rec['llm_point_estimate']}) ---")
                print(rec["llm_answer"]); print()
            elif i % 50 == 0:
                log(f"  ...{i}/{len(todo)} done")
    RUNSTATE.write_text(json.dumps({"temperature_dropped": STATE["drop_temp"],
                                    "last_run_at": dt.datetime.now().isoformat(timespec="seconds")}))
    log("run complete.")

def assemble():
    if not CACHE.exists():
        log("no cache yet."); return
    records = [json.loads(l) for l in open(CACHE)]
    log(f"assembling {len(records)} cached records")
    meta = pd.read_parquet(OUT_ROOT / f"{IN_NAME}.parquet")[
        ["stay_id", "subject_id", "split", "remaining_los_days"]]
    out = build_table(records, meta)
    out_path = OUT_ROOT / f"{STEP_NAME}.parquet"
    out.to_parquet(out_path, index=False)
    log(f"wrote -> {out_path}  shape={out.shape}")

    s = score(out)
    log(f"dropped {s['n_null_dropped']} null-estimate rows from MAE")
    print("\nMAE in real days (LLM point estimate vs remaining LoS)")
    print(f"{'split':<14}{'n':>8}{'MAE':>9}")
    print(f"{'test':<14}{s['n_test_scored']:>8}{(s['test_mae'] if s['test_mae'] is not None else float('nan')):>9.3f}")
    print(f"{'whole':<14}{s['n_total']-s['n_null_dropped']:>8}{(s['whole_mae'] if s['whole_mae'] is not None else float('nan')):>9.3f}")
    print(f"\nreference: X-only RF test MAE {RF_TEST_MAE} d (step 11) | MedM2T {MEDM2T_MAE} d")

    temp_dropped = json.loads(RUNSTATE.read_text())["temperature_dropped"] if RUNSTATE.exists() else "unknown"
    manifest = {
        "step": STEP_NAME, "run_at": dt.datetime.now().isoformat(timespec="seconds"),
        "input": str(OUT_ROOT / f"{IN_NAME}.parquet"), "output": str(out_path),
        "chat_model": CHAT_MODEL, "embed_model": EMBED_MODEL, "embedding_dim": EMBED_DIM,
        "system_prompt": SYSTEM_PROMPT, "user_template": USER_TEMPLATE,
        "temperature": 0, "temperature_dropped": temp_dropped,
        "cohort_cap_first_n": N_STAYS,
        "n_stays_processed": int(len(records)),
        "n_null_estimates": s["n_null_dropped"],
        "test_mae_days": s["test_mae"], "whole_mae_days": s["whole_mae"],
        "scoring": "MAE in real days, no transform (LLM emits days); null estimates dropped",
        "cache": str(CACHE),
        "contamination_caveat": "zero-shot; the LLM may have seen MIMIC (project-known risk).",
    }
    (OUT_ROOT / f"{STEP_NAME}__manifest.json").write_text(json.dumps(manifest, indent=2))
    log("wrote manifest")

# ---------------------------- the one runnable check ----------------------------

def selftest():
    """Offline: exercise parse, table build, and scoring. No API calls."""
    assert parse_estimate("blah\nESTIMATE_DAYS: 3.5") == 3.5
    assert parse_estimate("ESTIMATE_DAYS: 2\n...\nESTIMATE_DAYS: 4") == 4.0   # last wins
    assert parse_estimate("ESTIMATE_DAYS: .5") == 0.5
    assert parse_estimate("no estimate here") is None
    assert parse_estimate("") is None

    recs = [
        {"stay_id": "a", "llm_answer": "x\nESTIMATE_DAYS: 2", "llm_point_estimate": 2.0,
         "embedding": [0.1, 0.2, 0.3, 0.4]},
        {"stay_id": "b", "llm_answer": "y", "llm_point_estimate": None,
         "embedding": [0.0, 0.0, 0.0, 0.0]},
        {"stay_id": "c", "llm_answer": "z\nESTIMATE_DAYS: 5", "llm_point_estimate": 5.0,
         "embedding": [0.5, 0.5, 0.5, 0.5]},
    ]
    meta = pd.DataFrame({"stay_id": ["a", "b", "c"], "subject_id": ["p", "q", "r"],
                         "split": ["test", "test", "train"],
                         "remaining_los_days": [3.0, 1.0, 4.0]})
    tbl = build_table(recs, meta)
    assert list(tbl.columns)[:6] == ["stay_id", "subject_id", "split",
                                     "remaining_los_days", "llm_point_estimate", "llm_answer"]
    assert [c for c in tbl.columns if c.startswith("emb_")] == ["emb_0","emb_1","emb_2","emb_3"]
    assert len(tbl) == 3

    s = score(tbl)
    assert s["n_null_dropped"] == 1                 # stay b dropped
    assert s["n_test_scored"] == 1                  # only a (b is null) in test
    assert abs(s["test_mae"] - 1.0) < 1e-9          # |2-3| = 1
    assert abs(s["whole_mae"] - 1.0) < 1e-9         # mean(|2-3|,|5-4|) = 1
    log("selftest OK (parse, build_table, score all correct; no API used)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None, help="run on first N stays only")
    ap.add_argument("--assemble", action="store_true", help="build parquet + score from cache")
    ap.add_argument("--selftest", action="store_true", help="offline logic check, no API")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    elif args.assemble:
        assemble()
    else:
        run(sample=args.sample)
