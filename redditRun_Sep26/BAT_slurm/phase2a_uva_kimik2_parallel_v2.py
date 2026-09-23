# -*- coding: utf-8 -*-
"""phase2a_uva_kimik2_parallel.py
Sharded + concurrent version of phase2a_uva_kimik2.py for running the full
144,652-post BAT annotation as a Slurm ARRAY job on UVA HPC.

Two layers of parallelism, since ~54s/call is dominated by network/model
wait time, not CPU:
  1. CONCURRENCY  — within one job, --concurrency threads fire API calls
                     at once via a shared httpx.Client with a larger
                     connection pool.
  2. SHARDING     — the 144,652 posts are split into --num_shards
                     contiguous chunks; each Slurm array task processes
                     one chunk (--shard_id) and writes its own output file.
                     Run run_bat_array.sbatch (submitted with sbatch) to
                     launch all shards as one array job.

After all shards finish, run merge_shards.py to combine them into one CSV.

Usage (usually launched by the sbatch array script, not by hand):
  python phase2a_uva_kimik2_parallel.py \\
      --input master_posts_burnout_only.csv \\
      --output bat_posts_results.csv \\
      --num_shards 20 --shard_id 0 \\
      --concurrency 8 --resume

Environment:
  .env file with UVARC_GenAI_API=sk-... in the same folder.

IMPORTANT — concurrency and rate limits:
  UVA's GenAI endpoint is a shared resource. Start conservative
  (--concurrency 4-8) and watch for repeated "rate"/"503" errors in the
  logs before scaling up. If you see a lot of retries, LOWER concurrency
  rather than raising it — thrashing against a rate limit is slower than
  just respecting it.
"""

import httpx
import pandas as pd
import json
import time
import argparse
import os
import sys
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Argument parsing ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Sharded/concurrent BAT annotation — UVA Kimi K2.5")
parser.add_argument("--input", default="master_posts_burnout_only.csv")
parser.add_argument("--output", default="bat_posts_results.csv",
                     help="Base output filename — shard suffix is added automatically, e.g. bat_posts_results_shard03.csv")
parser.add_argument("--model", default="Kimi K2.5")
parser.add_argument("--limit", type=int, default=None,
                     help="Cap total posts BEFORE sharding (mainly for testing the sharding logic itself)")
parser.add_argument("--delay", type=float, default=0.0,
                     help="Seconds to sleep after each completed call (usually 0 when using concurrency — the concurrency itself is the rate control)")
parser.add_argument("--temp", type=float, default=0.0)
parser.add_argument("--max_tokens", type=int, default=4000)
parser.add_argument("--debug_n", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=200,
                     help="Flush to CSV every N completed posts")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--num_shards", type=int, default=1,
                     help="Total number of parallel shards (= Slurm array size)")
parser.add_argument("--shard_id", type=int, default=0,
                     help="This job's shard index, 0-indexed (= $SLURM_ARRAY_TASK_ID)")
parser.add_argument("--concurrency", type=int, default=8,
                     help="Number of concurrent API calls within this shard")
parser.add_argument("--max_text_chars", type=int, default=6000,
                     help="Truncate post text beyond this length before sending — guards against context-window overflow on unusually long Reddit posts")

args, _unknown = parser.parse_known_args()

if args.shard_id < 0 or args.shard_id >= args.num_shards:
    sys.exit(f"ERROR: --shard_id must be in [0, {args.num_shards - 1}], got {args.shard_id}")

# Derive per-shard output path: bat_posts_results.csv -> bat_posts_results_shard03.csv
_base, _ext = os.path.splitext(args.output)
SHARD_OUTPUT = f"{_base}_shard{args.shard_id:03d}{_ext}"

print(f"Input       : {args.input}")
print(f"Shard output: {SHARD_OUTPUT}")
print(f"Model       : {args.model}")
print(f"Shard       : {args.shard_id}/{args.num_shards}")
print(f"Concurrency : {args.concurrency}")
print(f"Batch       : {args.batch_size}")
print(f"Resume      : {args.resume}")
print(f"Max tokens  : {args.max_tokens}")

# ── Prompts (identical to phase2a_uva_kimik2.py) ──────────────────────────────
BAT_INSTRUCTIONS = """You are a researcher applying the Burnout Assessment Tool (BAT) to Reddit text.
Your job is to decide YES or NO for each of four burnout dimensions.  We will consider any form of burnout
and stress signal written in any tense such as past, present and future. For example: "I will feel stress..",
"I have gone through a lot of stress or I was stressed or burnout", "i am having sleep trouble or having
trouble balancing my work and personal life".

IMPORTANT RULES BEFORE YOU START:
- Read the text cold, with no assumptions about whether burnout is present.
- This is a SENSITIVITY-FIRST task. When in doubt, lean YES.
  Missing a true burnout signal is worse than flagging a stress-adjacent one.
- A post asking others about their experience is NOT the same as expressing it yourself.
- Naturalistic Reddit language rarely uses clinical terms — look for the meaning, not exact words.
- The word "burnout" alone with no other signal = NO. Any supporting signal alongside it = YES.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EX — Exhaustion
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Definition: Energy loss from work — physical (tiredness, feeling weak) AND/OR mental
(feeling drained, worn-out). Includes sustained overload that implies depletion even
without the exact word "drained".

YES if any of:
  - Explicit depletion: "drained", "nothing left", "used up", "running on empty",
    "physically broken", "can't decompress", "sleep doesn't help"
  - Sustained overload personally described over weeks or months:
    "tough couple of years", "impossible deadlines", "never ending [workload]",
    "always something left to do", chronic on-call or work pressure as personal cost
  - Physical or health deterioration attributed to work:
    "health has been deteriorating [since this job]", "getting sick from work"
  - Persistent misery tied to the job: "been miserable since [starting this role]"
  - Lack of energy to start work, feeling completely used up after working

NO if:
  - A single mentioned bad day with no sustained element
  - Asking others if they experience exhaustion (not expressing it personally)
  - Only boredom or dissatisfaction with no energy or health cost described

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EMO — Emotional Impairment
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Definition: Intense, persistent, or disproportionate emotional reactions tied to work.
Does NOT require explicit "loss of control" — strong sustained negative emotion qualifies.

YES if any of:
  - Strong hate or intense aversion toward the work situation:
    "I HATE [job/role/situation]", "this job is making me miserable"
  - Repeated or stacked emotional signals in the same post indicating sustained distress
    (e.g., "punch in the face" used twice, or multiple frustration phrases together)
  - Snapping, crying unexpectedly, or overreacting at work
  - Feeling upset, sad, or angry without a clear single cause
  - Persistent irritability or frustration tied to work beyond one incident
  - Feeling frustrated and angry at work, feeling upset or sad without knowing why
  - Feeling unable to control one's emotions at work, irritability, overreacting

NO if:
  - A single proportionate frustration about one event mentioned once and calmly
  - Mild annoyance described without intensity or repetition
  - Asking others if they feel frustrated (not expressing it personally)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COG — Cognitive Impairment
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Definition: Difficulty with memory, focus, or decision-making at work.
Includes feeling cognitively overwhelmed by job demands (volume, complexity, pace).

YES if any of:
  - Feeling overwhelmed by cognitive demands: volume of alerts, tasks, or complexity
    (even in a newer role, if the overwhelm goes beyond normal new-job adjustment)
  - Brain fog, forgetting procedures or tasks, trouble concentrating
  - Indecision or inability to make decisions that would normally be easy
  - Difficulty learning or keeping up with what the job demands
  - Being absent-minded, forgetful, or mentally scattered at work
  - Difficulties thinking clearly, poor memory, attention or concentration at work

NO if:
  - Brand new to role AND describes normal learning difficulty with no signs of distress
  - Asking others about cognitive difficulty without expressing it personally
  - Feeling generally confused with no work-specific cognitive symptom

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MD — Mental Distance
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Definition: Persistent psychological withdrawal — indifference, cynicism, aversion, autopilot.

YES if any of:
  - Explicit loss of meaning or interest: "what's the point", "don't care anymore",
    "I used to love this but now feel nothing", "no longer want to"
  - Going through the motions or autopilot described personally
  - Active avoidance of work tasks or colleagues
  - Persistent cynical or resentful tone throughout the post
  - Persistent dread of work: "I dread going in / this role / these tasks"
  - Wanting to escape driven by disengagement rather than career ambition
  - Withdrawing mentally or physically from work, avoiding contact with colleagues

NO if:
  - Asking others about engagement (not expressing own detachment)
  - Single bad day or one-off complaint
  - Considering career change out of ambition or curiosity (active agency, not withdrawal)
  - Mild boredom mentioned once without sustained withdrawal signals

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Respond with JSON only. No markdown fences. No explanation outside the JSON.

For each category provide a reasoning field explaining the decision whether YES or NO:
  - If YES: quote the exact phrase from the text that triggered YES.
  - If NO:  write one sentence explaining why the text did not meet the threshold.

{
  "EX":          "YES" or "NO",
  "EMO":         "YES" or "NO",
  "COG":         "YES" or "NO",
  "MD":          "YES" or "NO",
  "EX_reasoning":  "quoted phrase if YES  /  one-sentence explanation if NO",
  "EMO_reasoning": "quoted phrase if YES  /  one-sentence explanation if NO",
  "COG_reasoning": "quoted phrase if YES  /  one-sentence explanation if NO",
  "MD_reasoning":  "quoted phrase if YES  /  one-sentence explanation if NO"
}"""

POST_SYSTEM = (
    "You are annotating Reddit POSTS from cybersecurity communities for a burnout study.\n"
    + BAT_INSTRUCTIONS
)

# ── .env loader ─────────────────────────────────────────────────────────────
def _load_env_file():
    for candidate in [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env") if "__file__" in globals() else None,
        os.path.join(os.getcwd(), ".env"),
    ]:
        if candidate and os.path.exists(candidate):
            print(f"  Loading .env from: {candidate}")
            with open(candidate) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    line = re.sub(r"^export\s+", "", line)
                    if "=" in line:
                        k, _, v = line.partition("=")
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k and k not in os.environ:
                            os.environ[k] = v
            return candidate
    return None

_found = _load_env_file()
if not _found:
    print("  (No .env file found — falling back to shell environment)")

UVARC_API_KEY = os.environ.get("UVARC_GenAI_API")
if not UVARC_API_KEY:
    sys.exit("\nERROR: UVARC_GenAI_API not found. Create a .env file with UVARC_GenAI_API=sk-...\n")
print("API key loaded (starts with):", UVARC_API_KEY[:8] + "...")

# ── UVA Kimi K2.5 client (shared across threads, larger connection pool) ─────
UVARC_CHAT_ENDPOINT = "https://open-webui.rc.virginia.edu/api/chat/completions"
_httpx_client = httpx.Client(
    timeout=180.0,
    limits=httpx.Limits(
        max_connections=args.concurrency * 2,
        max_keepalive_connections=args.concurrency,
    ),
)
_call_counter = {"n": 0}
_counter_lock = threading.Lock()
DEBUG_FIRST_N = args.debug_n


def _extract_text_from_response(raw_text):
    raw_text = raw_text.strip()
    try:
        data = json.loads(raw_text)
        choice = data["choices"][0]
        content = choice.get("message", {}).get("content") or choice.get("text")
        if content:
            return content, "", choice.get("finish_reason")
    except Exception:
        pass

    if "data:" in raw_text:
        content_pieces, reasoning_pieces, finish_reason = [], [], None
        for line in raw_text.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]" or not payload:
                continue
            try:
                chunk = json.loads(payload)
                choice = chunk["choices"][0]
                delta = choice.get("delta", {})
                if delta.get("content"):
                    content_pieces.append(delta["content"])
                if delta.get("reasoning"):
                    reasoning_pieces.append(delta["reasoning"])
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
            except Exception:
                continue
        if content_pieces or reasoning_pieces:
            return "".join(content_pieces), "".join(reasoning_pieces), finish_reason

    return None, "", None


def call_kimi(system: str, user: str, post_id: str, max_tokens: int = None, max_retries: int = 5) -> dict:
    headers = {"Authorization": f"Bearer {UVARC_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": args.model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": args.temp,
        "max_tokens": max_tokens or args.max_tokens,
        "stream": False,
    }

    for attempt in range(max_retries):
        try:
            resp = _httpx_client.post(UVARC_CHAT_ENDPOINT, headers=headers, json=body)

            with _counter_lock:
                _call_counter["n"] += 1
                call_n = _call_counter["n"]

            content, reasoning, finish_reason = _extract_text_from_response(resp.text)

            if call_n <= DEBUG_FIRST_N:
                print(f"\n--- RAW RESPONSE (call {call_n}, status {resp.status_code}) ---")
                print(f"  reasoning chars: {len(reasoning)} | content chars: {len(content or '')} | finish_reason: {finish_reason}")
                print(f"  content: {(content or '')[:1500]}")
                print("--- END RAW RESPONSE ---\n")

            if resp.status_code == 401:
                sys.exit("ERROR: 401 Unauthorized — check that UVARC_GenAI_API is a valid, non-expired key.")

            if resp.status_code == 400:
                print(f"    [!] {post_id}: 400 Bad Request — server said: {resp.text[:500]}")

            resp.raise_for_status()

            if finish_reason == "length":
                print(f"    [!] {post_id}: hit max_tokens ({max_tokens or args.max_tokens}) — "
                      f"{len(reasoning)} reasoning chars, {len(content or '')} content chars "
                      f"(possibly truncated mid-JSON) — raise --max_tokens.")
                return {"_error": "truncated at max_tokens — raise --max_tokens"}

            if not content:
                print(f"    [!] {post_id}: no content field found.")
                return {"_error": "unparseable response"}

            raw = content.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            s = raw.find("{"); e = raw.rfind("}") + 1
            if s != -1 and e > s:
                raw = raw[s:e]
            return json.loads(raw)

        except Exception as ex:
            err = str(ex).lower()
            wait = 10 * (2 ** attempt) if ("rate" in err or "limit" in err or "503" in err) else 10
            print(f"    [!] API error for {post_id}: {ex} — waiting {wait}s (attempt {attempt+1}/{max_retries})")
            time.sleep(wait)

    return {"_error": "max retries exceeded"}


def bat_score(result: dict) -> int:
    return sum(1 for k in ["EX", "EMO", "COG", "MD"] if result.get(k) == "YES")


_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _sanitize_text(text: str) -> str:
    """
    Strip characters that can break JSON encoding/decoding on the way to the
    API: lone Unicode surrogates (common in scraped Reddit text with bad
    emoji/encoding artifacts) and control characters other than \\n and \\t.
    Dropping via utf-8 encode/decode with errors='ignore' removes anything
    that can't survive a clean utf-8 round trip.
    """
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    text = _CONTROL_CHAR_RE.sub("", text)
    return text


def empty_bat() -> dict:
    return {
        "EX": "NO", "EMO": "NO", "COG": "NO", "MD": "NO",
        "EX_reasoning": "out-of-scope, BAT not run",
        "EMO_reasoning": "out-of-scope, BAT not run",
        "COG_reasoning": "out-of-scope, BAT not run",
        "MD_reasoning": "out-of-scope, BAT not run",
    }


_save_lock = threading.Lock()


def save_batch(batch_data: list, output_path: str) -> None:
    if not batch_data:
        return
    with _save_lock:
        batch_df = pd.DataFrame(batch_data)
        file_is_new = not os.path.exists(output_path) or os.path.getsize(output_path) == 0
        batch_df.to_csv(output_path, mode="w" if file_is_new else "a", index=False, header=file_is_new)
        print(f"  -> Saved batch of {len(batch_data)} rows to {output_path}")


def annotate_one(row) -> dict:
    """Runs in a worker thread. Calls the model and returns the result row dict."""
    pid = str(row["id"])
    ptext = str(row["text"])
    ptext_for_prompt = _sanitize_text(ptext)
    if len(ptext_for_prompt) > args.max_text_chars:
        print(f"    [i] {pid}: text is {len(ptext_for_prompt)} chars, truncating to {args.max_text_chars}")
    ptext_for_prompt = ptext_for_prompt[:args.max_text_chars]
    prompt = f'Reddit post to annotate:\n"""{ptext_for_prompt.strip()}"""\n\nJSON only.'

    t0 = time.time()
    result = call_kimi(POST_SYSTEM, prompt, pid, max_tokens=args.max_tokens)
    elapsed = time.time() - t0

    if "_error" in result:
        bat = empty_bat()
        bat.update({"EX": "ERROR", "EMO": "ERROR", "COG": "ERROR", "MD": "ERROR"})
    else:
        bat = result

    if args.delay:
        time.sleep(args.delay)

    return {
        "row_type": "post", "post_id": pid, "comment_id": "", "text": ptext,
        "triage": "N/A", "na_subtype": "", "triage_reason": "",
        "EX": bat.get("EX", "ERROR"), "EMO": bat.get("EMO", "ERROR"),
        "COG": bat.get("COG", "ERROR"), "MD": bat.get("MD", "ERROR"),
        "bat_score": bat_score(bat),
        "EX_reasoning": bat.get("EX_reasoning", ""), "EMO_reasoning": bat.get("EMO_reasoning", ""),
        "COG_reasoning": bat.get("COG_reasoning", ""), "MD_reasoning": bat.get("MD_reasoning", ""),
        "_elapsed": elapsed,
    }


# ── STEP 1 — Load, filter, shard ──────────────────────────────────────────────
print("\n" + "=" * 60); print("STEP 1 — Load posts CSV"); print("=" * 60)

df = pd.read_csv(args.input)
for col in ["id", "text", "predicted_label"]:
    if col not in df.columns:
        sys.exit(f"ERROR: input CSV missing column '{col}'. Found: {df.columns.tolist()}")

df["id"] = df["id"].astype(str).str.strip()
df["Label"] = pd.to_numeric(df["predicted_label"], errors="coerce").fillna(0).astype(int)
label1_df = df[df["Label"] == 1].copy().reset_index(drop=True)

if args.limit:
    label1_df = label1_df.head(args.limit)

print(f"Total label=1 posts        : {len(label1_df)}")

# Contiguous sharding: shard S gets rows [S*chunk, (S+1)*chunk)
n = len(label1_df)
chunk = -(-n // args.num_shards)  # ceil division
start = args.shard_id * chunk
end = min(start + chunk, n)
shard_df = label1_df.iloc[start:end].copy()

print(f"Shard {args.shard_id}/{args.num_shards} covers rows [{start}:{end}) -> {len(shard_df)} posts")

# ── Resume ────────────────────────────────────────────────────────────────────
processed_ids = set()
if args.resume and os.path.exists(SHARD_OUTPUT):
    try:
        existing_df = pd.read_csv(SHARD_OUTPUT)
        processed_ids = set(existing_df["post_id"].astype(str).tolist())
        print(f"Resuming shard: {len(processed_ids)} posts already done in this shard.")
    except Exception as e:
        print(f"Could not read existing shard output ({e}) — starting fresh.")

to_process = shard_df[~shard_df["id"].isin(processed_ids)].copy()
print(f"Posts to process in this shard: {len(to_process)}")

# ── STEP 2 — Concurrent annotation ────────────────────────────────────────────
print("\n" + "=" * 60); print(f"STEP 2 — Annotating with {args.concurrency} concurrent workers"); print("=" * 60)

current_batch = []
all_timings = []
n_done = 0
n_error = 0
_progress_lock = threading.Lock()
run_t0 = time.time()

with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
    futures = {pool.submit(annotate_one, row): row["id"] for _, row in to_process.iterrows()}

    for fut in as_completed(futures):
        pid = futures[fut]
        try:
            result_row = fut.result()
        except Exception as ex:
            print(f"    [!] Worker crashed for {pid}: {ex}")
            continue

        elapsed = result_row.pop("_elapsed")
        with _progress_lock:
            all_timings.append(elapsed)
            n_done += 1
            if result_row["EX"] == "ERROR":
                n_error += 1
            current_batch.append(result_row)
            if n_done % 10 == 0 or n_done == len(to_process):
                rate = n_done / (time.time() - run_t0)
                remaining = len(to_process) - n_done
                eta_min = (remaining / rate) / 60 if rate > 0 else float("nan")
                print(f"  [{n_done}/{len(to_process)}] done | {n_error} errors | "
                      f"{rate:.2f} posts/sec | ETA {eta_min:.1f} min")
            if len(current_batch) >= args.batch_size:
                batch_to_save, current_batch = current_batch, []
                save_batch(batch_to_save, SHARD_OUTPUT)

if current_batch:
    save_batch(current_batch, SHARD_OUTPUT)

# ── Summary ───────────────────────────────────────────────────────────────────
total_elapsed = time.time() - run_t0
print("\n" + "=" * 60); print(f"SHARD {args.shard_id} DONE"); print("=" * 60)
print(f"Processed  : {n_done} posts ({n_error} errors)")
print(f"Wall time  : {total_elapsed/60:.1f} min")
if all_timings:
    print(f"Avg call   : {sum(all_timings)/len(all_timings):.1f}s (effective throughput with concurrency={args.concurrency}: "
          f"{len(all_timings)/total_elapsed:.2f} posts/sec)")
print(f"Output     : {SHARD_OUTPUT}")
