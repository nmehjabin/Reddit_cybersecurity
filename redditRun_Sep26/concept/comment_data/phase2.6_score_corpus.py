#!/usr/bin/env python3
"""
score_corpus.py

Phase 2.6: score a much larger sample of the corpus against all frozen
labels (LLooM concepts + hand-written taxonomy) using a hand-rolled,
concurrency-limited, checkpointed batch loop -- not l.score(), which is
bound to the 2,176-comment generation sample and has no retry/crash
recovery for a run at this scale.

Run from your activated venv:
    python score_corpus.py --test      # 2.6.2's required first step: 10 calls, check usage
    python score_corpus.py             # full run

Resumable: results are appended to JSONL after every single batch, and
already-completed (label, chunk) pairs are skipped on restart. Safe to
interrupt and rerun at any point.
"""

import os
import sys
import json
import time
import math
import glob
import asyncio
import random
import argparse

import pandas as pd
from dotenv import load_dotenv

# ── CONFIG ───────────────────────────────────────────────────────────────
OUTPUT_DIR = "/Users/nadia/Desktop/redditRun_june/comment_data/"
ENV_PATH = os.path.join(OUTPUT_DIR, ".env")

TEXT_COL = "body"
ID_COL = "id"
SUBREDDIT_COL = "subreddit_source"

# 2.6.1 -- census the small strata, subsample only sysadmin
SCORE_N = {"ciso": None, "asknetsec": None, "SecurityCareerAdvice": None,
           "cybersecurity": None, "sysadmin": 25_000}
SAMPLE_RANDOM_STATE = 7

SCORE_SAMPLE_PATH = os.path.join(OUTPUT_DIR, "score_sample.parquet")
SCORE_RESULTS_DIR = os.path.join(OUTPUT_DIR, "score_results")
SCORE_JSONL_PATH = os.path.join(SCORE_RESULTS_DIR, "scores.jsonl")

JUDGE_MODEL = "openai/gpt-5.6-luna-pro"   # matches the model used for generation, per your consistency preference
COMMENTS_PER_CALL = 10
CONCURRENCY = 12
MAX_OUTPUT_TOKENS = 400   # raised from 200 -- headroom even with reasoning minimized (see 2.6.2 note)
MAX_RETRIES = 5
BASE_DELAY = 5


# ── MODEL SETUP ──────────────────────────────────────────────────────────
def setup_llm_fn(api_key):
    from openai import AsyncOpenAI
    import httpx
    return AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        timeout=httpx.Timeout(connect=15.0, read=60.0, write=60.0, pool=60.0),
    )


async def call_model(client, semaphore, prompt, on_credits_exhausted=None):
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                res = await client.chat.completions.create(
                    model=JUDGE_MODEL,
                    temperature=0,
                    max_tokens=MAX_OUTPUT_TOKENS,
                    response_format={"type": "json_object"},
                    extra_body={"reasoning": {"effort": "minimal"}},  # cut reasoning-token overhead -- see 2.6.2 note
                    messages=[
                        {"role": "system", "content": "You are a careful annotator. Follow the instructions exactly. No explanation."},
                        {"role": "user", "content": prompt},
                    ],
                )
                text = res.choices[0].message.content if res and res.choices else None
                usage = getattr(res, "usage", None)
                return text, usage
            except Exception as e:
                err_str = str(e).lower()
                is_credits = "402" in str(e) or "requires more credits" in err_str
                is_last_attempt = attempt == MAX_RETRIES - 1
                is_retryable = (
                    "429" in str(e) or "rate limit" in err_str or "timed out" in err_str
                    or "timeout" in err_str or "connection" in err_str
                )
                if is_credits:
                    print(f"  [402 -- out of credits] {e}")
                    if on_credits_exhausted is not None:
                        on_credits_exhausted()
                    return None, None
                if is_retryable and not is_last_attempt:
                    delay = BASE_DELAY * (2 ** attempt) + random.uniform(0, 2)
                    print(f"  [{type(e).__name__}] retrying in {delay:.1f}s (attempt {attempt+1}/{MAX_RETRIES})...")
                    await asyncio.sleep(delay)
                    continue
                print(f"  [error, giving up after {attempt+1} attempt(s)]: {e}")
                return None, None
        return None, None


def batch_prompt(label, docs):
    numbered = "\n".join(f"{i+1}. {d}" for i, d in enumerate(docs))
    return (f"Concept: {label['name']}\n"
            f"Criteria: {label['prompt']}\n\n"
            f"For each numbered comment, answer 1 if it matches the concept, 0 if not.\n"
            f'Return only JSON, no explanation: {{"1": 0, "2": 1, ...}}\n\n{numbered}')


def parse_batch_response(text):
    try:
        return {int(k): int(v) for k, v in json.loads(text).items()}
    except Exception as e:
        print(f"    ERROR parsing response: {e}")
        return {}


# ── 2.6.1: DRAW THE SCORING SAMPLE ───────────────────────────────────────
def draw_scoring_sample():
    if os.path.exists(SCORE_SAMPLE_PATH):
        print(f"{SCORE_SAMPLE_PATH} already exists -- loading it instead of re-drawing.")
        print("Delete it first if you want to redraw.")
        return pd.read_parquet(SCORE_SAMPLE_PATH)

    c = pd.read_csv(os.path.join(OUTPUT_DIR, "master_comments_filtered.csv"))
    print(f"Full corpus: {len(c):,} comments")
    print(c[SUBREDDIT_COL].value_counts())

    missing = set(k for k in SCORE_N if SCORE_N[k] is not None) - set(c[SUBREDDIT_COL].unique())
    unknown = set(SCORE_N.keys()) - set(c[SUBREDDIT_COL].unique())
    if unknown:
        raise ValueError(f"SCORE_N references subreddit(s) not found in data: {unknown}\n"
                          f"Actual values: {sorted(c[SUBREDDIT_COL].unique())}")

    def draw(g):
        n = SCORE_N.get(g.name)
        if n is None:
            out = g.copy()
        else:
            out = g.sample(min(n, len(g)), random_state=SAMPLE_RANDOM_STATE).copy()
        out[SUBREDDIT_COL] = g.name  # re-attach -- newer pandas strips the groupby column
        return out

    score_sample = c.groupby(SUBREDDIT_COL, group_keys=False).apply(draw)
    score_sample = score_sample.reset_index(drop=True)

    print(f"\nScoring sample: {len(score_sample):,} comments")
    print(score_sample[SUBREDDIT_COL].value_counts())

    score_sample.to_parquet(SCORE_SAMPLE_PATH)
    print(f"Saved: {SCORE_SAMPLE_PATH}")
    return score_sample


# ── LOAD LABELS (2.3 frozen concepts + hand-written taxonomy) ───────────
def load_labels():
    frozen_path = os.path.join(OUTPUT_DIR, "frozen_concepts.json")
    assert os.path.exists(frozen_path), f"frozen_concepts.json not found -- finish 2.3 first."
    lloom_labels = json.load(open(frozen_path))

    TAXONOMY = [
        {"id": "T1", "source": "taxonomy", "name": "informational_support",
         "prompt": "The commenter provides advice, factual information, instruction, referral to "
                   "a source of knowledge, or an assessment of the poster's situation. Includes "
                   "suggesting a course of action, explaining how something works, teaching a "
                   "skill, or offering an interpretation of what is happening to the poster."},
        {"id": "T2", "source": "taxonomy", "name": "emotional_support",
         "prompt": "The commenter expresses care, concern, sympathy, understanding, encouragement, "
                   "or reassurance. Includes acknowledging the poster's feelings, expressing "
                   "sorrow for their situation, or offering comfort."},
        {"id": "T3", "source": "taxonomy", "name": "esteem_support",
         "prompt": "The commenter validates the poster's worth, competence, or judgment. Includes "
                   "expressing confidence in their abilities, telling them their reaction is "
                   "reasonable or justified, complimenting them, or relieving them of blame."},
        {"id": "T4", "source": "taxonomy", "name": "tangible_support",
         "prompt": "The commenter offers concrete assistance or resources. Includes offering to "
                   "help directly, offering to review a resume or make an introduction, or "
                   "pointing to a specific service, document, template, or tool the poster can use."},
        {"id": "T5", "source": "taxonomy", "name": "network_support",
         "prompt": "The commenter conveys belonging or connection to others in the same situation. "
                   "Includes stating that the poster is not alone, describing the experience as "
                   "widely shared among peers, or directing them to a community or group."},
        {"id": "T6", "source": "taxonomy", "name": "unsupportive_response",
         "prompt": "The commenter minimizes, dismisses, criticizes, blames, or mocks the poster. "
                   "Includes stating the problem is normal and not worth raising, attributing it "
                   "to the poster's own failings or weakness, or responding with derision."},
    ]

    labels = lloom_labels + TAXONOMY
    print(f"Labels: {len(lloom_labels)} LLooM concepts + {len(TAXONOMY)} taxonomy = {len(labels)} total")
    return labels


# ── 2.6.2's REQUIRED FIRST STEP: TEST 10 CALLS, CHECK USAGE ─────────────
async def test_ten_calls(client, labels, score_sample):
    print("Running 10 test calls -- checking whether reasoning tokens are being billed...\n")
    semaphore = asyncio.Semaphore(CONCURRENCY)
    label = labels[0]
    docs = score_sample[TEXT_COL].tolist()[:COMMENTS_PER_CALL]

    total_prompt = 0
    total_completion = 0
    for i in range(10):
        prompt = batch_prompt(label, docs)
        text, usage = await call_model(client, semaphore, prompt)
        if usage is None:
            print(f"  call {i+1}: FAILED")
            continue
        print(f"  call {i+1}: prompt_tokens={usage.prompt_tokens}, "
              f"completion_tokens={usage.completion_tokens}")
        total_prompt += usage.prompt_tokens
        total_completion += usage.completion_tokens

    print(f"\nAverage completion_tokens per call: {total_completion/10:.1f}")
    print(f"(For {COMMENTS_PER_CALL} yes/no answers with max_tokens={MAX_OUTPUT_TOKENS}, "
          f"expect roughly 20-60 completion tokens for the JSON answer itself.")
    print(f"If completion_tokens is consistently much higher than that -- e.g. close to "
          f"{MAX_OUTPUT_TOKENS} -- the model is likely spending tokens on hidden reasoning "
          f"before the JSON, which would silently eat your max_tokens budget and could "
          f"truncate real responses at scale. Consider raising MAX_OUTPUT_TOKENS or switching "
          f"judge model if so.)")

    n_labels_est = len(labels)
    n_chunks_est = math.ceil(len(score_sample) / COMMENTS_PER_CALL)
    total_calls_est = n_labels_est * n_chunks_est
    avg_tokens_per_call = (total_prompt / 10) + (total_completion / 10)
    print(f"\nEstimated full run: {n_labels_est} labels x {n_chunks_est:,} chunks "
          f"= {total_calls_est:,} calls")


# ── 2.6.2: MAIN BATCHED SCORING LOOP ─────────────────────────────────────
def load_completed_keys():
    """Scan the JSONL for (label_id, chunk_idx) pairs already written."""
    completed = set()
    if not os.path.exists(SCORE_JSONL_PATH):
        return completed
    with open(SCORE_JSONL_PATH, "r") as f:
        for line in f:
            try:
                row = json.loads(line)
                completed.add((row["label_id"], row["chunk_idx"]))
            except Exception:
                continue
    return completed


async def run_full_scoring(client, labels, score_sample):
    os.makedirs(SCORE_RESULTS_DIR, exist_ok=True)
    semaphore = asyncio.Semaphore(CONCURRENCY)

    docs = score_sample[TEXT_COL].tolist()
    doc_ids = score_sample[ID_COL].tolist()
    n_chunks = math.ceil(len(docs) / COMMENTS_PER_CALL)

    completed = load_completed_keys()
    print(f"{len(completed):,} (label, chunk) pairs already done -- resuming from there.")

    total_units = len(labels) * n_chunks
    done_units = len(completed)
    t_start = time.time()
    credits_exhausted = asyncio.Event()

    jsonl_file = open(SCORE_JSONL_PATH, "a")

    async def score_one_chunk(label, chunk_idx):
        nonlocal done_units
        if credits_exhausted.is_set():
            return
        key = (label["id"], chunk_idx)
        if key in completed:
            return

        start = chunk_idx * COMMENTS_PER_CALL
        end = min(start + COMMENTS_PER_CALL, len(docs))
        chunk_docs = docs[start:end]
        chunk_ids = doc_ids[start:end]

        prompt = batch_prompt(label, chunk_docs)
        text, usage = await call_model(client, semaphore, prompt, on_credits_exhausted=credits_exhausted.set)

        if text is None:
            # Failed (e.g. out of credits, or exhausted retries) -- write
            # nothing, so a future run retries this batch instead of
            # treating an empty result as "done".
            done_units += 1
            return

        parsed = parse_batch_response(text)
        scores = {chunk_ids[i - 1]: v for i, v in parsed.items() if 1 <= i <= len(chunk_ids)}

        row = {"label_id": label["id"], "label_name": label["name"],
               "chunk_idx": chunk_idx, "scores": scores}
        jsonl_file.write(json.dumps(row) + "\n")
        jsonl_file.flush()

        done_units += 1
        if done_units % 100 == 0:
            elapsed = time.time() - t_start
            rate = done_units / elapsed if elapsed > 0 else 0
            remaining = total_units - done_units
            eta_min = (remaining / rate) / 60 if rate > 0 else float("inf")
            print(f"  {done_units:,}/{total_units:,} batches done "
                  f"({elapsed/60:.1f}m elapsed, ~{eta_min:.0f}m remaining)")

    tasks = [score_one_chunk(label, chunk_idx)
             for label in labels
             for chunk_idx in range(n_chunks)]

    print(f"Total batches to process: {total_units:,} (concurrency={CONCURRENCY})")
    await asyncio.gather(*tasks)

    jsonl_file.close()
    if credits_exhausted.is_set():
        print(f"\n⚠️  Run stopped early -- ran out of credits. Add more at "
              f"https://openrouter.ai/settings/credits, then run again to pick up where this left off.")
    else:
        print(f"\n✅ Done. Results in {SCORE_JSONL_PATH}")
    return credits_exhausted.is_set()


# ── 2.6.3: RESHAPE + ATTACH WEIGHTS ──────────────────────────────────────
def reshape_and_weight(labels, score_sample):
    print("\nReshaping results into wide table...")
    rows_by_label = {}
    with open(SCORE_JSONL_PATH, "r") as f:
        for line in f:
            row = json.loads(line)
            label_name = row["label_name"]
            rows_by_label.setdefault(label_name, {}).update(row["scores"])

    label_names = [l["name"] for l in labels]
    wide = pd.DataFrame({name: pd.Series(rows_by_label.get(name, {})) for name in label_names})
    wide.index.name = ID_COL
    wide = wide.reset_index()

    wide = wide.merge(
        score_sample[[ID_COL, SUBREDDIT_COL, "post_id"]],
        on=ID_COL, how="left"
    )

    # Compute weights from ACTUAL sample/population sizes, not hardcoded
    # plan numbers -- your real corpus may not match the plan's example exactly.
    full_corpus = pd.read_csv(os.path.join(OUTPUT_DIR, "master_comments_filtered.csv"),
                               usecols=[SUBREDDIT_COL])
    population_counts = full_corpus[SUBREDDIT_COL].value_counts()
    sample_counts = score_sample[SUBREDDIT_COL].value_counts()

    W = {}
    for sub in population_counts.index:
        if SCORE_N.get(sub) is None:
            W[sub] = 1.0  # censused -- no weighting needed
        else:
            W[sub] = population_counts[sub] / sample_counts[sub]

    print("\nWeights (population / sample, 1.0 = censused):")
    for sub, w in W.items():
        print(f"  {sub}: {w:.2f}")

    wide["w"] = wide[SUBREDDIT_COL].map(W)

    out_path = os.path.join(OUTPUT_DIR, "score_sample_wide.parquet")
    wide.to_parquet(out_path)
    print(f"\nSaved: {out_path}  ({len(wide):,} rows x {len(label_names)} labels)")
    print("\nReminder: per-subreddit numbers need no weighting. Any corpus-wide number does.")
    return wide


# ── MAIN ──────────────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                         help="Run only the 10-call usage check (2.6.2's required first step), then exit.")
    parser.add_argument("--reshape-only", action="store_true",
                         help="Skip scoring, just reshape existing scores.jsonl into the wide table.")
    args = parser.parse_args()

    loaded = load_dotenv(ENV_PATH)
    print(f".env loaded: {loaded}")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(f"OPENROUTER_API_KEY not found after loading {ENV_PATH}")

    client = setup_llm_fn(api_key)
    labels = load_labels()
    score_sample = draw_scoring_sample()

    if args.reshape_only:
        reshape_and_weight(labels, score_sample)
        return

    if args.test:
        await test_ten_calls(client, labels, score_sample)
        print("\nReview the numbers above. If they look right, run again without --test "
              "to start the full scoring run.")
        return

    print(f"\n About to run the FULL scoring pass: {len(labels)} labels x "
          f"~{math.ceil(len(score_sample)/COMMENTS_PER_CALL):,} chunks. "
          f"Estimated 3-5 hours, roughly $60 per the plan (verify against your own "
          f"--test run numbers, not this template's estimate).")
    confirm = input("Proceed? (y/n): ")
    if confirm.strip().lower() != "y":
        print("Aborted.")
        return

    stopped_early = await run_full_scoring(client, labels, score_sample)
    reshape_and_weight(labels, score_sample)
    if stopped_early:
        print("\n  Reminder: this run stopped early due to credits. The wide table above "
              "reflects only what was successfully scored -- add credits and run again "
              "before treating this as final.")


if __name__ == "__main__":
    asyncio.run(main())