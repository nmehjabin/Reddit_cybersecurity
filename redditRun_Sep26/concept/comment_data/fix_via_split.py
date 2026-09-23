#!/usr/bin/env python3
"""
fix_via_split.py

For (label, chunk) pairs still failing even after a plain retry_problematic.py
pass -- splits just those specific pairs into smaller sub-batches instead of
asking about all 10 comments at once. This is what actually fixed 5212/5328
when plain retry couldn't.

Reads directly from problematic_batches.csv (produced by inspect_scores.py),
so it targets exactly what's still broken -- no manually maintained list.

Run from your activated venv, AFTER running inspect_scores.py to get a
current problematic_batches.csv:
    python fix_via_split.py
Then:
    python phase2.6_score_corpus.py --reshape-only
"""

import os
import json
import asyncio
import random

import pandas as pd
from dotenv import load_dotenv

OUTPUT_DIR = "/Users/nadia/Desktop/redditRun_june/comment_data/"
ENV_PATH = os.path.join(OUTPUT_DIR, ".env")
PROBLEM_CSV = os.path.join(OUTPUT_DIR, "score_results", "problematic_batches.csv")
SCORE_JSONL_PATH = os.path.join(OUTPUT_DIR, "score_results", "scores.jsonl")
SCORE_SAMPLE_PATH = os.path.join(OUTPUT_DIR, "score_sample.parquet")

TEXT_COL = "body"
ID_COL = "id"
JUDGE_MODEL = "openai/gpt-5.6-luna-pro"
COMMENTS_PER_CALL = 10
SPLIT_SIZE = 3
CONCURRENCY = 12
MAX_OUTPUT_TOKENS = 600
MAX_RETRIES = 5
BASE_DELAY = 5


def setup_llm_fn(api_key):
    from openai import AsyncOpenAI
    import httpx
    return AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        timeout=httpx.Timeout(connect=15.0, read=60.0, write=60.0, pool=60.0),
    )


async def call_model(client, semaphore, prompt):
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                res = await client.chat.completions.create(
                    model=JUDGE_MODEL, temperature=0, max_tokens=MAX_OUTPUT_TOKENS,
                    response_format={"type": "json_object"},
                    extra_body={"reasoning": {"effort": "minimal"}},
                    messages=[
                        {"role": "system", "content": "You are a careful annotator. Follow the instructions exactly. No explanation."},
                        {"role": "user", "content": prompt},
                    ],
                )
                return res.choices[0].message.content if res and res.choices else None
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
                    return None
                if is_retryable and not is_last_attempt:
                    delay = BASE_DELAY * (2 ** attempt) + random.uniform(0, 2)
                    await asyncio.sleep(delay)
                    continue
                print(f"  [error, giving up]: {e}")
                return None
        return None


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
        print(f"    still failed to parse: {e}")
        return {}


def load_labels_by_id():
    frozen = json.load(open(os.path.join(OUTPUT_DIR, "frozen_concepts.json")))
    TAXONOMY = [
        {"id": "T1", "source": "taxonomy", "name": "informational_support", "prompt": "The commenter provides advice, factual information, instruction, referral to a source of knowledge, or an assessment of the poster's situation. Includes suggesting a course of action, explaining how something works, teaching a skill, or offering an interpretation of what is happening to the poster."},
        {"id": "T2", "source": "taxonomy", "name": "emotional_support", "prompt": "The commenter expresses care, concern, sympathy, understanding, encouragement, or reassurance. Includes acknowledging the poster's feelings, expressing sorrow for their situation, or offering comfort."},
        {"id": "T3", "source": "taxonomy", "name": "esteem_support", "prompt": "The commenter validates the poster's worth, competence, or judgment. Includes expressing confidence in their abilities, telling them their reaction is reasonable or justified, complimenting them, or relieving them of blame."},
        {"id": "T4", "source": "taxonomy", "name": "tangible_support", "prompt": "The commenter offers concrete assistance or resources. Includes offering to help directly, offering to review a resume or make an introduction, or pointing to a specific service, document, template, or tool the poster can use."},
        {"id": "T5", "source": "taxonomy", "name": "network_support", "prompt": "The commenter conveys belonging or connection to others in the same situation. Includes stating that the poster is not alone, describing the experience as widely shared among peers, or directing them to a community or group."},
        {"id": "T6", "source": "taxonomy", "name": "unsupportive_response", "prompt": "The commenter minimizes, dismisses, criticizes, blames, or mocks the poster. Includes stating the problem is normal and not worth raising, attributing it to the poster's own failings or weakness, or responding with derision."},
    ]
    return {l["id"]: l for l in frozen + TAXONOMY}


async def main():
    loaded = load_dotenv(ENV_PATH)
    print(f".env loaded: {loaded}")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not found")

    client = setup_llm_fn(api_key)
    labels_by_id = load_labels_by_id()
    score_sample = pd.read_parquet(SCORE_SAMPLE_PATH)
    docs = score_sample[TEXT_COL].tolist()
    doc_ids = score_sample[ID_COL].tolist()

    problem_df = pd.read_csv(PROBLEM_CSV)
    print(f"Splitting {len(problem_df):,} still-problematic batches into sub-batches of {SPLIT_SIZE}...\n")

    semaphore = asyncio.Semaphore(CONCURRENCY)
    jsonl_file = open(SCORE_JSONL_PATH, "a")

    async def score_sub_batch(label, sub_start, sub_end):
        sub_docs = docs[sub_start:sub_end]
        sub_ids = doc_ids[sub_start:sub_end]
        prompt = batch_prompt(label, sub_docs)
        text = await call_model(client, semaphore, prompt)
        if text is None:
            return {}
        parsed = parse_batch_response(text)
        return {sub_ids[i - 1]: v for i, v in parsed.items() if 1 <= i <= len(sub_ids)}

    async def fix_one(label_id, chunk_idx):
        label = labels_by_id.get(label_id)
        if label is None:
            print(f"  Unknown label_id {label_id!r} -- skipping")
            return

        chunk_start = chunk_idx * COMMENTS_PER_CALL
        chunk_end = min(chunk_start + COMMENTS_PER_CALL, len(docs))

        all_scores = {}
        for sub_start in range(chunk_start, chunk_end, SPLIT_SIZE):
            sub_end = min(sub_start + SPLIT_SIZE, chunk_end)
            sub_scores = await score_sub_batch(label, sub_start, sub_end)
            all_scores.update(sub_scores)

        row = {"label_id": label_id, "label_name": label["name"],
               "chunk_idx": chunk_idx, "scores": all_scores}
        jsonl_file.write(json.dumps(row) + "\n")
        jsonl_file.flush()
        n_expected = chunk_end - chunk_start
        print(f"  [{label['name']} / chunk {chunk_idx}] {len(all_scores)}/{n_expected} answered via split")

    tasks = [fix_one(row["label_id"], row["chunk_idx"]) for _, row in problem_df.iterrows()]
    await asyncio.gather(*tasks)

    jsonl_file.close()
    print(f"\nDone. Now run: python phase2.6_score_corpus.py --reshape-only")


if __name__ == "__main__":
    asyncio.run(main())