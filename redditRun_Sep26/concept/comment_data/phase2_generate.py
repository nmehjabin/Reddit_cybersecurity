#!/usr/bin/env python3
"""
phase2_generate.py

Covers Phase 2 steps 1-8: load + merge + cap + stratify the generation
sample, then run gen() twice (unseeded, seeded) to produce candidate
concept lists. Saves everything needed for the interactive review notebook
to pick up from -- gen_sample.parquet, ckpt/final_unseeded.pkl,
ckpt/final_seeded.pkl.

Run from your activated venv:
    python phase2_generate.py

Resumable: if a checkpoint already exists for a run, that run is skipped
and reloaded instead of regenerated. Safe to re-run after an interruption.
"""

import os
import pickle
import asyncio
import random

import pandas as pd
from dotenv import load_dotenv

import text_lloom.workbench as wb
from text_lloom.llm import Model, EmbedModel

# Patch `display` into builtins -- LLooM's debug=True path calls it directly
# (assumes a Jupyter/IPython kernel), which doesn't exist in a plain script.
import builtins
if not hasattr(builtins, "display"):
    builtins.display = print

# ── CONFIG ───────────────────────────────────────────────────────────────
OUTPUT_DIR = "/Users/nadia/Desktop/redditRun_june/comment_data/"
ENV_PATH = "/Users/nadia/Desktop/redditRun_june/comment_data/.env"  # EDIT ME if it's elsewhere

TEXT_COL = "body"
ID_COL = "id"
SUBREDDIT_COL = "subreddit_source"

GEN_N = {"sysadmin": 700, "cybersecurity": 500,
         "SecurityCareerAdvice": 400, "asknetsec": 400, "ciso": 176}

CHOSEN_SEED = "what the commenter is doing in relation to the person who wrote the post"  # Seed A, from Phase 1

OPENROUTER_MODEL = "openai/gpt-5.6-luna-pro"   # switch back to "google/gemini-3.7-flash" + GEMINI_COST below if needed
MODEL_COST = (0.20 / 1_000_000, 1.20 / 1_000_000)
CONTEXT_WINDOW = 1_050_000
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

MAX_RETRIES = 5
BASE_DELAY = 5
MAX_OUTPUT_TOKENS = 4096


# ── STEP 1-2: LOAD + MERGE ───────────────────────────────────────────────
def load_and_merge():
    assert os.path.isdir(OUTPUT_DIR), f"OUTPUT_DIR not found: {OUTPUT_DIR}"

    c = pd.read_csv(os.path.join(OUTPUT_DIR, "master_comments_filtered.csv"))
    bat = pd.read_csv(os.path.join(OUTPUT_DIR, "bat_score_pos.csv"), usecols=["post_id", "bat_score"])

    print(f"Comments: {len(c):,} rows, columns: {c.columns.tolist()}")
    print(f"BAT scores: {len(bat):,} rows")

    c["post_id"] = c["post_id"].astype(str)
    bat["post_id"] = bat["post_id"].astype(str)
    c = c.merge(bat, on="post_id", how="left")

    n_missing = c["bat_score"].isna().sum()
    print(f"After merge: {n_missing:,} comments have no matching bat_score (expected 0)")

    print(f"bat_score dtype: {c['bat_score'].dtype}")
    print(f"bat_score unique values (first 20): {sorted(c['bat_score'].dropna().unique())[:20]}")
    return c


# ── STEP 3: CAP PER POST ─────────────────────────────────────────────────
def cap_per_post(c):
    def sample_post(t):
        # Explicitly re-attach post_id -- newer pandas strips the groupby
        # column from what the function receives, and t.sample() alone
        # would silently lose it otherwise.
        sampled = t.sample(min(len(t), 3), random_state=42).copy()
        sampled["post_id"] = t.name
        return sampled

    capped = c.groupby("post_id", group_keys=False).apply(sample_post)
    print(f"{len(c):,} comments -> {len(capped):,} after capping at 3/post")
    return capped


# ── STEP 4: SEVERITY BAND ────────────────────────────────────────────────
def add_severity_band(capped):
    capped = capped.copy()
    capped["sev_band"] = capped.bat_score.clip(upper=3).astype(str)  # "0" won't appear

    print("sev_band value counts:")
    print(capped["sev_band"].value_counts().sort_index())

    n_bands = capped["sev_band"].nunique()
    if n_bands > 10:
        print(f"WARNING: {n_bands} distinct severity bands found -- expected ~3-4. "
              f"bat_score may be continuous; check before trusting downstream sampling.")
    return capped


# ── STEP 5: STRATIFIED DRAW ──────────────────────────────────────────────
def draw_generation_sample(capped):
    print(f"GEN_N total: {sum(GEN_N.values())}")

    missing = set(GEN_N.keys()) - set(capped[SUBREDDIT_COL].unique())
    if missing:
        raise ValueError(
            f"GEN_N references subreddit(s) not found in the data: {missing}\n"
            f"Actual values are: {sorted(capped[SUBREDDIT_COL].unique())}"
        )

    def draw(g, n):
        per = max(1, n // g["sev_band"].nunique())

        def sample_band(b):
            sampled = b.sample(min(len(b), per), random_state=42).copy()
            sampled["sev_band"] = b.name  # re-attach -- same reason as sample_post above
            return sampled

        out = g.groupby("sev_band", group_keys=False).apply(sample_band)
        if len(out) < n:
            rest = g.drop(out.index, errors="ignore")
            out = pd.concat([out, rest.sample(min(len(rest), n - len(out)), random_state=42)])
        return out

    def draw_wrapper(g):
        sub_out = draw(g, min(GEN_N[g.name], len(g))).copy()
        sub_out[SUBREDDIT_COL] = g.name  # re-attach -- same reason
        return sub_out

    gen_sample = capped.groupby(SUBREDDIT_COL, group_keys=False).apply(draw_wrapper)

    print(f"Generation sample: {len(gen_sample):,} comments")
    print(gen_sample.groupby([SUBREDDIT_COL, "sev_band"]).size())
    return gen_sample


# ── STEP 6: MODEL SETUP ──────────────────────────────────────────────────
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


# ── STEP 7: DUAL GENERATION ──────────────────────────────────────────────
async def run_generation(gen_sample, api_key, ckpt_dir):
    os.makedirs(ckpt_dir, exist_ok=True)

    unseeded_path = os.path.join(ckpt_dir, "final_unseeded.pkl")
    if os.path.exists(unseeded_path):
        print("[unseeded] checkpoint found -- loading instead of re-running gen()")
        with open(unseeded_path, "rb") as f:
            l_unseeded = pickle.load(f)
        for k, v in build_models(api_key).items():
            setattr(l_unseeded, k, v)
    else:
        print("[unseeded] running gen()...")
        l_unseeded = wb.lloom(df=gen_sample, text_col=TEXT_COL, id_col=ID_COL, **build_models(api_key))
        params = l_unseeded.auto_suggest_parameters(target_n_concepts=20)
        print(f"Auto-suggested params: {params}")
        await l_unseeded.gen(params=params, n_synth=2, auto_review=True, debug=True)
        l_unseeded.save(folder=ckpt_dir, file_name="final_unseeded")

    seeded_path = os.path.join(ckpt_dir, "final_seeded.pkl")
    if os.path.exists(seeded_path):
        print("\n[seeded] checkpoint found -- loading instead of re-running gen()")
        with open(seeded_path, "rb") as f:
            l_seeded = pickle.load(f)
        for k, v in build_models(api_key).items():
            setattr(l_seeded, k, v)
    else:
        print("\n[seeded] running gen()...")
        l_seeded = wb.lloom(df=gen_sample, text_col=TEXT_COL, id_col=ID_COL, **build_models(api_key))
        params = l_seeded.auto_suggest_parameters(target_n_concepts=20)
        print(f"Auto-suggested params: {params}")
        await l_seeded.gen(params=params, seed=CHOSEN_SEED, n_synth=2, auto_review=True, debug=True)
        l_seeded.save(folder=ckpt_dir, file_name="final_seeded")

    print(f"\nUnseeded: {len(l_unseeded.concepts)} concepts")
    print(f"Seeded: {len(l_seeded.concepts)} concepts")

    print("\n===== UNSEEDED =====")
    for cpt in l_unseeded.concepts.values():
        print(f"- {cpt.name}: {cpt.prompt}")
    print("\n===== SEEDED =====")
    for cpt in l_seeded.concepts.values():
        print(f"- {cpt.name}: {cpt.prompt}")

    return l_unseeded, l_seeded


# ── MAIN ──────────────────────────────────────────────────────────────────
async def main():
    loaded = load_dotenv(ENV_PATH)
    print(f".env loaded: {loaded}")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(f"OPENROUTER_API_KEY not found after loading {ENV_PATH} -- check the file.")

    gen_sample_path = os.path.join(OUTPUT_DIR, "gen_sample.parquet")
    if os.path.exists(gen_sample_path):
        print(f"gen_sample.parquet already exists -- loading it instead of re-drawing.")
        print("Delete it first if you want to redraw with different GEN_N/parameters.")
        gen_sample = pd.read_parquet(gen_sample_path)
    else:
        c = load_and_merge()
        capped = cap_per_post(c)
        capped = add_severity_band(capped)
        gen_sample = draw_generation_sample(capped)
        gen_sample.to_parquet(gen_sample_path)
        print(f"Saved: {gen_sample_path}")

    ckpt_dir = os.path.join(OUTPUT_DIR, "ckpt")
    await run_generation(gen_sample, api_key, ckpt_dir)

    print("\n✅ Done. Open the review notebook next to run l.select() and score concepts --"
          " that part needs a Jupyter widget and can't run from this script.")


if __name__ == "__main__":
    asyncio.run(main())