#!/usr/bin/env python3
"""
orphan_check.py

Phase 2.7: check whether the frozen LLooM concept list actually covers the
comments it was built from. Prints the orphan rate; if it's over 15%,
samples up to 500 uncovered comments and runs a follow-up gen() pass to
surface what's missing.

The final manual-review step (deciding which of the follow-up concepts are
genuinely new vs. redundant with the existing frozen list) is intentionally
NOT automated here -- same reasoning as the original 2.3 review. This script
prints the candidate concepts for you to read; merging them into
frozen_concepts.json is a separate, deliberate edit you make by hand.

Run from your activated venv:
    python orphan_check.py
"""

import os
import json
import pickle
import asyncio
import random

import pandas as pd
from dotenv import load_dotenv

# Fix SSL cert verification for nltk's download inside text_lloom's import.
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['SSL_CERT_DIR'] = os.path.dirname(certifi.where())

# Patch `display` into builtins -- LLooM's debug=True path calls it directly
# (assumes a Jupyter/IPython kernel), which doesn't exist in a plain script.
import builtins
if not hasattr(builtins, "display"):
    builtins.display = print

import text_lloom.workbench as wb
from text_lloom.llm import Model, EmbedModel

# ── CONFIG ───────────────────────────────────────────────────────────────
OUTPUT_DIR = "/Users/nadia/Desktop/redditRun_june/comment_data/"
ENV_PATH = os.path.join(OUTPUT_DIR, ".env")

TEXT_COL = "body"
ID_COL = "id"

OPENROUTER_MODEL = "openai/gpt-5.6-luna-pro"   # match whatever generated your original concepts
MODEL_COST = (0.20 / 1_000_000, 1.20 / 1_000_000)
CONTEXT_WINDOW = 1_050_000
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

ORPHAN_THRESHOLD = 0.15
ORPHAN_SAMPLE_N = 500

MAX_RETRIES = 5
BASE_DELAY = 5
MAX_OUTPUT_TOKENS = 4096


# ── MODEL SETUP (same as phase2_generate.py) ─────────────────────────────
def setup_llm_fn(api_key):
    from openai import AsyncOpenAI
    import httpx
    return AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        timeout=httpx.Timeout(connect=15.0, read=120.0, write=120.0, pool=120.0),
    )


def setup_embed_fn(api_key):
    from fastembed import TextEmbedding
    return TextEmbedding(model_name=EMBED_MODEL_NAME)


async def call_llm_fn(model, prompt):
    if "system_prompt" not in model.args:
        model.args["system_prompt"] = (
            "You are a helpful assistant who helps with identifying patterns in text examples."
        )
    if "temperature" not in model.args:
        model.args["temperature"] = 0

    for attempt in range(MAX_RETRIES):
        try:
            res = await model.client.chat.completions.create(
                model=model.name,
                temperature=model.args["temperature"],
                max_tokens=MAX_OUTPUT_TOKENS,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": model.args["system_prompt"]},
                    {"role": "user", "content": prompt},
                ],
            )
            text = res.choices[0].message.content if res and res.choices else None
            tokens = (res.usage.prompt_tokens, res.usage.completion_tokens) if res and getattr(res, "usage", None) else (0, 0)
            return text, tokens
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
                return None, None
            if is_retryable and not is_last_attempt:
                delay = BASE_DELAY * (2 ** attempt) + random.uniform(0, 2)
                print(f"  [{type(e).__name__}] retrying in {delay:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})...")
                await asyncio.sleep(delay)
                continue
            print(f"  [error, giving up after {attempt + 1} attempt(s)]: {e}")
            return None, None
    return None, None


def call_embed_fn(model, text_arr):
    embeddings = [e.tolist() for e in model.client.embed(text_arr)]
    return embeddings, (0, 0)


def build_models(api_key):
    return dict(
        distill_model=Model(setup_fn=setup_llm_fn, fn=call_llm_fn, name=OPENROUTER_MODEL,
                             cost=MODEL_COST, rate_limit=(15, 10), context_window=CONTEXT_WINDOW, api_key=api_key),
        cluster_model=EmbedModel(setup_fn=setup_embed_fn, fn=call_embed_fn, name=EMBED_MODEL_NAME,
                                  cost=0, batch_size=64, api_key=api_key),
        synth_model=Model(setup_fn=setup_llm_fn, fn=call_llm_fn, name=OPENROUTER_MODEL,
                           cost=MODEL_COST, rate_limit=(10, 10), context_window=CONTEXT_WINDOW, api_key=api_key),
        score_model=Model(setup_fn=setup_llm_fn, fn=call_llm_fn, name=OPENROUTER_MODEL,
                           cost=MODEL_COST, rate_limit=(10, 10), context_window=CONTEXT_WINDOW, api_key=api_key),
    )


# ── 2.7 ORPHAN CHECK ──────────────────────────────────────────────────────
def run_orphan_check(gen_sample, sc, frozen):
    lloom_cols = [x["name"] for x in frozen]
    print(f"Checking coverage across {len(lloom_cols)} frozen LLooM concepts:")
    for name in lloom_cols:
        print(f"  - {name}")

    wide = sc.pivot(index="doc_id", columns="concept_name", values="score")

    missing_docs = set(gen_sample[ID_COL]) - set(wide.index)
    if missing_docs:
        print(f"\n⚠️  {len(missing_docs)} comments from gen_sample have NO scores in sc at all "
              f"(likely silent scoring failures, not orphans) -- investigate separately "
              f"before trusting the rate below.")

    wide = wide.reindex(columns=lloom_cols, fill_value=0).fillna(0)

    orphan_mask = (wide[lloom_cols].sum(axis=1) == 0)
    orphan_rate = orphan_mask.mean()

    print(f"\norphan rate: {orphan_rate:.1%} "
          f"({orphan_mask.sum():,} / {len(wide):,} comments match none of the {len(lloom_cols)} concepts)")

    if orphan_rate <= ORPHAN_THRESHOLD:
        print(f"✅ Under {ORPHAN_THRESHOLD:.0%} -- no action needed, per the decision rule.")
    else:
        print(f"⚠️  Above {ORPHAN_THRESHOLD:.0%} -- concept list likely has a real gap. Running follow-up generation pass...")

    return wide, orphan_mask, orphan_rate


# ── FOLLOW-UP GENERATION ON ORPHANS ──────────────────────────────────────
async def run_orphan_generation(gen_sample, wide, orphan_mask, api_key):
    orphan_doc_ids = wide.index[orphan_mask]
    orphan_sample = gen_sample[gen_sample[ID_COL].isin(orphan_doc_ids)].sample(
        min(ORPHAN_SAMPLE_N, orphan_mask.sum()), random_state=42
    )
    print(f"Sampled {len(orphan_sample)} of {orphan_mask.sum()} orphan comments for follow-up generation")

    ckpt_dir = os.path.join(OUTPUT_DIR, "ckpt_orphan")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "orphan_gen.pkl")

    if os.path.exists(ckpt_path):
        print("Checkpoint found -- loading instead of re-running gen()")
        with open(ckpt_path, "rb") as f:
            l_orphan = pickle.load(f)
        for k, v in build_models(api_key).items():
            setattr(l_orphan, k, v)
    else:
        l_orphan = wb.lloom(df=orphan_sample, text_col=TEXT_COL, id_col=ID_COL, **build_models(api_key))
        params = l_orphan.auto_suggest_parameters(target_n_concepts=20)
        print(f"Auto-suggested params: {params}")
        await l_orphan.gen(params=params, n_synth=2, auto_review=True, debug=True)
        l_orphan.save(folder=ckpt_dir, file_name="orphan_gen")

    print(f"\nOrphan-pass concepts ({len(l_orphan.concepts)}):")
    for c in l_orphan.concepts.values():
        print(f"- {c.name}: {c.prompt}")

    print(f"\n{'='*70}")
    print("NEXT STEP (manual, not automated):")
    print("Read the concepts above against the same four questions from 2.3 --")
    print("coherent? action not topic? distinct from your existing frozen list?")
    print("reasonable prevalence? Then append genuinely new ones to frozen_concepts.json")
    print("by hand, rescore gen_sample against just the new concept(s), and re-run this")
    print("script to confirm the orphan rate dropped.")


# ── MAIN ──────────────────────────────────────────────────────────────────
async def main():
    loaded = load_dotenv(ENV_PATH)
    print(f".env loaded: {loaded}")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(f"OPENROUTER_API_KEY not found after loading {ENV_PATH}")

    gen_sample = pd.read_parquet(os.path.join(OUTPUT_DIR, "gen_sample.parquet"))
    print(f"gen_sample: {len(gen_sample):,} rows")

    sc = pd.read_parquet(os.path.join(OUTPUT_DIR, "gen_scores.parquet"))
    print(f"sc: {len(sc):,} rows, {sc['concept_id'].nunique()} concepts")

    frozen_path = os.path.join(OUTPUT_DIR, "frozen_concepts.json")
    assert os.path.exists(frozen_path), f"frozen_concepts.json not found at {frozen_path} -- finish 2.3 first."
    frozen = json.load(open(frozen_path))
    print(f"frozen: {len(frozen)} concepts")

    wide, orphan_mask, orphan_rate = run_orphan_check(gen_sample, sc, frozen)

    if orphan_rate > ORPHAN_THRESHOLD:
        await run_orphan_generation(gen_sample, wide, orphan_mask, api_key)


if __name__ == "__main__":
    asyncio.run(main())
