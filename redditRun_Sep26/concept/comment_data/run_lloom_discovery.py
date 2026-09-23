"""
run_lloom_discovery.py
STAGE 1: Concept discovery on sample_posts_with_comments.csv (150 comments)

This is the pilot/discovery run. Goal: see what support-type concepts LLooM
surfaces from a small sample before committing to a full run on
master_comments_filtered.csv. Uses l.gen() + l.select() (NOT gen_auto) so you
can review concepts interactively before spending money on scoring.

Setup:
    pip install text_lloom openai sentence-transformers --break-system-packages
    export OPENROUTER_API_KEY="sk-or-..."
    python run_lloom_discovery.py

Docs:
  - LLooM custom models: https://stanfordhci.github.io/lloom/about/custom-models.html
  - LLooM get started:   https://stanfordhci.github.io/lloom/about/get-started.html
"""

import asyncio
import os
import pandas as pd

import text_lloom.workbench as wb
from text_lloom.llm import Model, EmbedModel

INPUT_CSV = "discovery_sample.csv"
OUTPUT_DIR = "/Users/nadia/Desktop/redditRun_june/comment_data/"
TEXT_COL = "cleaned_body"
ID_COL = "id"

OPENROUTER_MODEL = "moonshotai/kimi-k2.5"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Double-check current pricing before trusting the cost estimate --
# OpenRouter pricing for this model has moved around ($0.375-0.58 in /
# $2.00-3.41 out per 1M tokens depending on date/provider):
# https://openrouter.ai/moonshotai/kimi-k2.5/pricing
KIMI_COST = (0.60 / 1_000_000, 3.41 / 1_000_000)  # (input_cost, output_cost) per token

# --- Embeddings use fastembed (ONNX-based, no PyTorch/numba conflict, runs
# locally, no API cost).
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"


# ---------------------------------------------------------------------------
# Setup / call functions
# ---------------------------------------------------------------------------

def setup_llm_fn(api_key):
    from openai import AsyncOpenAI
    return AsyncOpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)


def setup_embed_fn(api_key):
    # api_key unused -- fastembed runs locally, no API call, no cost
    from fastembed import TextEmbedding
    return TextEmbedding(model_name=EMBED_MODEL_NAME)


async def call_llm_fn(model, prompt):
    if "system_prompt" not in model.args:
        model.args["system_prompt"] = (
            "You are a helpful assistant who helps with identifying patterns "
            "in text examples."
        )
    if "temperature" not in model.args:
        model.args["temperature"] = 0

    res = await model.client.chat.completions.create(
        model=model.name,
        temperature=model.args["temperature"],
        messages=[
            {"role": "system", "content": model.args["system_prompt"]},
            {"role": "user", "content": prompt},
        ],
    )
    text = res.choices[0].message.content if res and res.choices else None

    if res and getattr(res, "usage", None):
        tokens = (res.usage.prompt_tokens, res.usage.completion_tokens)
    else:
        tokens = (0, 0)

    return text, tokens


def call_embed_fn(model, text_arr):
    embeddings = list(model.client.embed(text_arr))
    embeddings = [e.tolist() for e in embeddings]
    tokens = (0, 0)  # local model -- no token cost
    return embeddings, tokens


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Run:\n"
            "  export OPENROUTER_API_KEY='sk-or-...'\n"
            "before running this script."
        )

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows from {INPUT_CSV}")

    df = df[df[TEXT_COL].notna()]
    df = df[df[TEXT_COL].str.strip().str.len() > 0]
    print(f"{len(df)} rows remain after dropping empty {TEXT_COL} values")
    print(f"NOTE: this sample spans only {df['post_id'].nunique()} posts and "
          f"{df['subreddit_source'].nunique()} subreddits ({sorted(df['subreddit_source'].unique())}). "
          f"Treat concepts from this run as a pilot, not a final taxonomy.")

    l = wb.lloom(
        df=df,
        text_col=TEXT_COL,
        id_col=ID_COL,
        distill_model=Model(
            setup_fn=setup_llm_fn, fn=call_llm_fn, name=OPENROUTER_MODEL,
            cost=KIMI_COST, rate_limit=(20, 10), context_window=262_144, api_key=api_key,
        ),
        cluster_model=EmbedModel(
            setup_fn=setup_embed_fn, fn=call_embed_fn, name=EMBED_MODEL_NAME,
            cost=0, batch_size=64, api_key=api_key,  # api_key unused, but required as a positional/kw arg
        ),
        synth_model=Model(
            setup_fn=setup_llm_fn, fn=call_llm_fn, name=OPENROUTER_MODEL,
            cost=KIMI_COST, rate_limit=(10, 10), context_window=262_144, api_key=api_key,
        ),
        score_model=Model(
            setup_fn=setup_llm_fn, fn=call_llm_fn, name=OPENROUTER_MODEL,
            cost=KIMI_COST, rate_limit=(20, 10), context_window=262_144, api_key=api_key,
        ),
    )

    l.estimate_gen_cost(verbose=True)

    # Discovery only -- NOT gen_auto -- so you can review concepts via
    # l.select() before spending anything on the scoring pass.
    # debug=False skips the interactive y/n confirmation prompt, which
    # would otherwise hang a non-notebook script waiting on stdin.
    await l.gen(
        seed="the kind of support or response being offered to the poster",
        debug=False,
    )

    # ---- STOP AND REVIEW HERE ----
    # Run this in a notebook (or add a breakpoint) to inspect concepts
    # before proceeding to score():
    #     l.select()
    #
    # Once you're happy with the concept set, score them:
    score_df = await l.score(score_all=True, debug=False)

    score_path = os.path.join(OUTPUT_DIR, "lloom_pilot_score_df_discovery800.csv")
    score_df.to_csv(score_path, index=False)
    print(f"Saved concept scores: {score_path}")

    summary_df = l.export_df()
    summary_path = os.path.join(OUTPUT_DIR, "lloom_pilot_concept_summary_discovery800.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved concept summary: {summary_path}")

    l.summary(verbose=True)

    l.save(folder=OUTPUT_DIR, file_name="lloom_pilot_session_discovery800")
    print(f"Saved session pickle to {OUTPUT_DIR}/lloom_pilot_session_discovery800.pkl")


if __name__ == "__main__":
    asyncio.run(main())