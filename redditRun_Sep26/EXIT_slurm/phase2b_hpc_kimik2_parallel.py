# -*- coding: utf-8 -*-
"""
phase2b_hpc_kimik2_parallel.py — Exit intention pipeline (two-pass), UVA HPC / Slurm version.

Mirrors the phase2a BAT annotation harness:
  - UVA RC GenAI endpoint (open-webui.rc.virginia.edu), model "Kimi K2.5"
  - Raw httpx calls (NOT the openai SDK — UVA's endpoint always streams SSE
    regardless of the "stream" flag, which breaks SDK response parsing)
  - SSE parser that separates `reasoning` deltas (chain-of-thought, discarded)
    from `content` deltas (the actual JSON answer we want)
  - Two layers of parallelism:
      1. CONCURRENCY — --concurrency worker threads fire API calls at once
         within one shard, via a shared httpx.Client connection pool.
      2. SHARDING    — the input is split into --num_shards contiguous
         chunks; each Slurm array task handles one --shard_id and writes
         its own output + checkpoint files. Run merge_shards_exit.py after
         all shards finish.
  - Unicode sanitization before every API call (Reddit scrape artifacts
    otherwise trigger HTTP 400s).
  - MAX_TOKENS = 4000 (Kimi K2.5 burns tokens on internal reasoning first;
    lower values silently truncate before any content is produced).
  - JSON checkpointing per shard, keyed by post id, so a killed/timed-out
    job resumes without reprocessing.

Pipeline (unchanged from phase2b_local.py):
  PASS 1 (all posts in this shard)   – Call 1: exit_explicit | exit_contemplating | no_exit
  PASS 2 (exit-positive rows only)   – Call 2: exit level (job_exit | field_exit | both | ambiguous)
                                      – Call 3: exit reason primary + secondary

Usage (normally launched by run_exit_array.sbatch, not by hand):
  python phase2b_hpc_kimik2_parallel.py \\
      --input bat_posts_results_final_patched.csv \\
      --output exit_annotated \\
      --num_shards 20 --shard_id 0 \\
      --concurrency 4 --resume

Environment:
  .env file (same folder or /scratch/jqc7gj/) with UVARC_GenAI_API=sk-...

IMPORTANT — concurrency and rate limits:
  The UVA GenAI endpoint is shared infrastructure. Start conservative
  (--concurrency 4) and watch logs for repeated "rate"/"503"/timeout
  errors before scaling up. If you see a lot of retries, LOWER
  concurrency rather than raising it.
"""

import argparse
import json
import os
import re
import syss
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
import pandas as pd
from dotenv import load_dotenv

# Look for .env next to the script AND in /scratch/jqc7gj/ (matches BAT harness)
load_dotenv()
load_dotenv("/scratch/jqc7gj/.env")

# ── Constants ──────────────────────────────────────────────────────────────
BASE_URL   = "https://open-webui.rc.virginia.edu/api/"
CHAT_URL   = BASE_URL + "chat/completions"
MODEL      = "Kimi K2.5"
MAX_TOKENS = 4000
TEMPERATURE = 0.0
MAX_TEXT_CHARS_DEFAULT = 2500

VALID_EXIT_CLASSES = {"exit_explicit", "exit_contemplating", "no_exit"}
VALID_LEVELS       = {"job_exit", "field_exit", "both", "ambiguous", "N/A"}
VALID_REASONS      = {
    "MANAGEMENT_FAILURE",
    "RESOURCE_INADEQUACY",
    "COMPENSATION",
    "TOXIC_CULTURE",
    "VALUE_MISALIGNMENT",
    "PERSONAL_LIMITS",
    "SYSTEMIC_FUTILITY",
    "TECHNICAL_DISILLUSIONMENT",
    "N/A",
}

PASS1_COLS = ["exit_intention_clas", "exit_confidence", "exit_evidence_span", "exit_reasoning"]
PASS2_COLS = ["exit_level", "exit_reason_primary", "exit_reason_secondary"]
EXIT_POSITIVE = {"exit_explicit", "exit_contemplating"}

# ── System prompts (identical to phase2b_local.py) ──────────────────────────
SYSTEM_EXIT_CLASS = """You are an expert qualitative researcher studying burnout and career exit intention in cybersecurity professionals. You are analyzing Reddit posts from
r/ciso, r/cybersecurity, r/SecurityCareerAdvice, r/sysadmin, r/asknetsec.

Your task: classify whether this post expresses exit intention — the poster's desire, plan, or consideration of leaving their job or the cybersecurity field.

CRITICAL CALIBRATION — be SENSITIVE. These are naturalistic posts, not clinical reports:
- "I can't keep doing this" → exit_contemplating (implicit)
- "Starting to look at what else is out there" → exit_contemplating
- "I'm done" / "putting in my notice" → exit_explicit
- "Something has to change or I'm gone" → exit_contemplating
- "I hate my job but I'm staying" / "venting but not leaving" → no_exit
- "Alert fatigue is real" (no leaving language) → no_exit

CLASSES:
- exit_explicit: Clear statement of intent, plan, or completed action to leave. No ambiguity.
- exit_contemplating: Considering leaving; hedged/conditional language; passive job-searching signals.
- no_exit: Burnout/frustration expressed but no directional language toward leaving.

Respond ONLY with valid JSON (no markdown, no extra text):
{
  "exit_intention_class": "<exit_explicit|exit_contemplating|no_exit>",
  "confidence": <float 0.0-1.0>,
  "evidence_span": "<direct quote from post, max 60 words, that most supports your classification>",
  "reasoning": "<1-2 sentence explanation of your classification>"
}"""

SYSTEM_EXIT_LEVEL = """You are analyzing a cybersecurity professional's Reddit post
that has already been flagged as containing exit intention.
Your task: determine WHAT LEVEL the poster wants to exit.

LEVELS:
- job_exit: The exit referent is their current employer, team, role, or specific company. They want to find a different job IN cybersecurity.
- field_exit: The exit referent is cybersecurity as a profession/industry. They want to leave the field entirely.
- both: The post clearly contains markers at BOTH levels.
- ambiguous: Cannot be determined from the text alone.

Linguistic signals for field_exit: "leaving cybersecurity", "done with security", "switching fields", "going back to dev", "the industry is broken everywhere", "no company is different", "considering finance/teaching/other field".
Linguistic signals for job_exit: "new job", "updating my resume", "leaving this company/org/team", "looking at other employers", references to a specific employer.

Respond ONLY with valid JSON:
{
  "exit_level": "<job_exit|field_exit|both|ambiguous>",
  "level_reasoning": "<1 sentence explanation>"
}"""

SYSTEM_REASON = """You are analyzing a cybersecurity professional's Reddit post that
expresses exit intention. Your task: identify the PRIMARY reason(s) cited for wanting to leave.

REASON TAXONOMY (use these exact labels):
- MANAGEMENT_FAILURE: Upper management doesn't care, leadership is incompetent, CISO ignores risks, "higher-ups" not supportive
- RESOURCE_INADEQUACY: Understaffed, no budget/tools/support, expected to do too much with too little
- COMPENSATION: Pay doesn't match stress, financial instability, inconsistent hours, underpaid
- TOXIC_CULTURE: Gaslighting, toxic positivity, drama, hostile team dynamics, unhealthy workplace culture
- VALUE_MISALIGNMENT: Ethical concerns about employer's practices, policies poster disagrees with, difficulty finding ethical employers
- PERSONAL_LIMITS: Mental health, family suffering, "I'm just tired", unsustainable for the individual, burnout at the body/identity level
- SYSTEMIC_FUTILITY: Industry-level disillusionment — "security doesn't matter to anyone", "all companies are the same", structural futility
- TECHNICAL_DISILLUSIONMENT: Alert fatigue, tool/tech debt frustration, the technical work itself has become alienating
- N/A: Exit is mentioned but no specific reason is given

Select the PRIMARY reason and, if clearly present, a SECONDARY reason. If no secondary reason is evident, use null.

Respond ONLY with valid JSON:
{
  "exit_reason_primary": "<REASON_LABEL>",
  "exit_reason_secondary": "<REASON_LABEL or null>",
  "reason_reasoning": "<1-2 sentence explanation>"
}"""

# ── Thread-safe print (workers run concurrently) ────────────────────────────
_print_lock = threading.Lock()
def log(msg: str):
    with _print_lock:
        print(msg, flush=True)

# ── Text sanitization (Reddit scrape artifacts cause HTTP 400s) ────────────
_CONTROL_CHAR_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
def _sanitize_text(text: str) -> str:
    if not text:
        return ""
    text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    text = _CONTROL_CHAR_RE.sub(" ", text)
    return text

# ── SSE parsing (UVA endpoint always streams, regardless of stream=False) ──
def _parse_sse_stream(raw_string: str) -> str:
    """Collect only `content` deltas; discard `reasoning` (chain-of-thought) deltas."""
    content_parts = []
    for line in raw_string.split("\n"):
        line = line.strip()
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
            delta = chunk["choices"][0]["delta"]
            if delta.get("content"):
                content_parts.append(delta["content"])
        except (json.JSONDecodeError, KeyError, IndexError):
            continue
    return "".join(content_parts).strip()

# ── API call helper (raw httpx, shared client for connection pooling) ──────
def call_kimi(client: httpx.Client, api_key: str, system_prompt: str, user_content: str,
              retries: int = 4, base_delay: float = 5.0):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }

    for attempt in range(1, retries + 1):
        try:
            resp = client.post(CHAT_URL, headers=headers, json=payload, timeout=120.0)
            if resp.status_code != 200:
                log(f"    [!] HTTP {resp.status_code} (attempt {attempt}): {resp.text[:300]}")
                time.sleep(base_delay * attempt)
                continue

            raw = resp.text
            # Endpoint always streams SSE regardless of request params.
            if raw.lstrip().startswith("data:"):
                content = _parse_sse_stream(raw)
            else:
                # Fallback: plain JSON response (rare, but handle it)
                try:
                    data = json.loads(raw)
                    content = data["choices"][0]["message"]["content"]
                except Exception:
                    content = _parse_sse_stream(raw)

            if not content:
                log(f"    [!] Empty content after SSE parse (attempt {attempt}); "
                    f"raw preview: {raw[:150]}")
                time.sleep(base_delay * attempt)
                continue

            match = re.search(r"\{.*\}", content, re.DOTALL)
            if not match:
                log(f"    [!] No JSON object in content (attempt {attempt}): {content[:150]}")
                time.sleep(base_delay * attempt)
                continue

            json_str = match.group(0)
            json_str = re.sub(r'\\([^"\\/bfnrtu])', r'\1', json_str)
            return json.loads(json_str)

        except json.JSONDecodeError as e:
            log(f"    [!] JSON parse error (attempt {attempt}): {e}")
            time.sleep(base_delay * attempt)
        except (httpx.TimeoutException, httpx.TransportError) as e:
            log(f"    [!] Network error (attempt {attempt}): {e}")
            time.sleep(base_delay * attempt)
        except Exception as e:
            log(f"    [!] Unexpected error (attempt {attempt}): {e}")
            time.sleep(base_delay * attempt)

    return None

# ── Pass 1: exit intention class ─────────────────────────────────────────
def classify_exit_intention(client, api_key, post_text, post_id, max_text_chars):
    result = {col: None for col in PASS1_COLS}
    text = _sanitize_text(post_text)[:max_text_chars]

    if not text.strip():
        result["exit_intention_class"] = "no_exit"
        result["exit_confidence"] = 0.0
        result["exit_evidence_span"] = ""
        result["exit_reasoning"] = "Empty post text."
        return result

    r1 = call_kimi(client, api_key, SYSTEM_EXIT_CLASS, f'POST TEXT:\n"""{text}"""\n\nJSON only.')

    if r1 is None:
        result["exit_intention_class"] = "no_exit"
        result["exit_confidence"] = 0.0
        result["exit_evidence_span"] = "ANNOTATION_ERROR"
        result["exit_reasoning"] = "API call failed after retries."
        return result

    exit_class = r1.get("exit_intention_class", "no_exit")
    if exit_class not in VALID_EXIT_CLASSES:
        exit_class = "no_exit"

    result["exit_intention_class"] = exit_class
    result["exit_confidence"] = float(r1.get("confidence", 0.5))
    result["exit_evidence_span"] = str(r1.get("evidence_span", ""))[:500]
    result["exit_reasoning"] = str(r1.get("reasoning", ""))[:1000]
    return result

# ── Pass 2: exit level + reasons (exit-positive rows only) ─────────────────
def classify_exit_level_and_reasons(client, api_key, post_text, existing_reasoning, max_text_chars):
    result = {
        "exit_level": "ambiguous",
        "exit_reason_primary": "N/A",
        "exit_reason_secondary": None,
        "exit_reasoning": existing_reasoning,
    }
    text = _sanitize_text(post_text)[:max_text_chars]
    prompt = f'POST TEXT:\n"""{text}"""\n\nJSON only.'

    r2 = call_kimi(client, api_key, SYSTEM_EXIT_LEVEL, prompt)
    if r2 is not None:
        lvl = r2.get("exit_level", "ambiguous")
        result["exit_level"] = lvl if lvl in VALID_LEVELS else "ambiguous"
        if r2.get("level_reasoning"):
            result["exit_reasoning"] = (result["exit_reasoning"] + " | Level: " + str(r2["level_reasoning"]))[:1500]

    r3 = call_kimi(client, api_key, SYSTEM_REASON, prompt)
    if r3 is not None:
        primary = r3.get("exit_reason_primary", "N/A")
        secondary = r3.get("exit_reason_secondary")
        result["exit_reason_primary"] = primary if primary in VALID_REASONS else "N/A"
        result["exit_reason_secondary"] = (
            secondary if (secondary and secondary in VALID_REASONS and secondary != "N/A") else None
        )
        if r3.get("reason_reasoning"):
            result["exit_reasoning"] = (result["exit_reasoning"] + " | Reason: " + str(r3["reason_reasoning"]))[:1500]

    return result

# ── Checkpoint helpers (thread-safe, save-on-every-completion) ─────────────
_ckpt_lock = threading.Lock()

def load_checkpoint(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        log(f"[i] Checkpoint loaded: {len(data)} entries from {path}")
        return data
    return {}

def save_checkpoint(path: str, data: dict):
    # Snapshot the dict INSIDE the lock so json.dump never iterates a dict
    # that another thread is simultaneously inserting into (this was the
    # cause of "RuntimeError: dictionary changed size during iteration").
    # The actual file write happens outside the lock so it doesn't block
    # other threads' checkpoint updates while disk I/O completes.
    with _ckpt_lock:
        snapshot = dict(data)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(snapshot, f)
    os.replace(tmp, path)


def update_checkpoint(checkpoint: dict, post_id: str, ann: dict):
    """Thread-safe insert into a shared checkpoint dict."""
    with _ckpt_lock:
        checkpoint[post_id] = ann

# ── Main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Sharded/concurrent exit-intention annotation — UVA Kimi K2.5")
    parser.add_argument("--input", default="bat_posts_results_final_patched.csv")
    parser.add_argument("--output", default="exit_annotated",
                         help="Base output filename (no extension) — shard suffix added automatically, "
                              "e.g. exit_annotated_pass1_shard0.csv / exit_annotated_pass2_shard0.csv")
    parser.add_argument("--num_shards", type=int, default=20)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--max_text_chars", type=int, default=MAX_TEXT_CHARS_DEFAULT)
    parser.add_argument("--limit", type=int, default=None, help="Process only first N rows of this shard (testing).")
    parser.add_argument("--debug_n", type=int, default=0, help="Print extra debug info for first N calls.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint (default behavior).")
    args = parser.parse_args()

    api_key = os.environ.get("UVARC_GenAI_API")
    if not api_key:
        log("[X] No UVARC_GenAI_API found. Put it in a .env file (this folder or /scratch/jqc7gj/.env) "
            "or export it as an env var.")
        sys.exit(1)

    in_path = Path(args.input)
    if not in_path.exists():
        log(f"[X] Input file not found: {in_path}")
        sys.exit(1)

    log(f"[i] Loading input: {in_path}")
    df_full = pd.read_csv(in_path, dtype=str)
    log(f"[i] Full input: {len(df_full)} rows")

    if args.id_col not in df_full.columns:
        log(f"[!] ID column '{args.id_col}' not found. Using row index as id.")
        df_full["_row_id"] = df_full.index.astype(str)
        id_col = "_row_id"
    else:
        id_col = args.id_col

    if args.text_col not in df_full.columns:
        log(f"[X] Text column '{args.text_col}' not found. Available: {list(df_full.columns)}")
        sys.exit(1)

    # ── Sharding: contiguous chunk assigned to this array task ────────────
    n = len(df_full)
    shard_size = -(-n // args.num_shards)  # ceil division
    start = args.shard_id * shard_size
    end = min(start + shard_size, n)
    df = df_full.iloc[start:end].reset_index(drop=True)
    log(f"[i] Shard {args.shard_id}/{args.num_shards - 1}: rows [{start}:{end}) -> {len(df)} rows")

    if args.limit:
        df = df.head(args.limit)
        log(f"[i] --limit set: processing first {len(df)} rows of this shard.")

    for col in PASS1_COLS + PASS2_COLS:
        if col not in df.columns:
            df[col] = None

    out_base = f"{args.output}_shard{args.shard_id}"
    out_pass1 = f"{out_base}_pass1.csv"
    out_pass2 = f"{out_base}_pass2.csv"
    ckpt_pass1 = f"{out_base}_ckpt_pass1.json"
    ckpt_pass2 = f"{out_base}_ckpt_pass2.json"

    client = httpx.Client(limits=httpx.Limits(max_connections=args.concurrency + 2,
                                               max_keepalive_connections=args.concurrency))

    # ══════════════════════════════════════════════════════════════════════
    # PASS 1 — exit intention class for every post in this shard
    # ══════════════════════════════════════════════════════════════════════
    checkpoint1 = load_checkpoint(ckpt_pass1)
    todo = []
    for idx, row in df.iterrows():
        post_id = str(row[id_col])
        if post_id in checkpoint1:
            ann = checkpoint1[post_id]
            for col in PASS1_COLS:
                df.at[idx, col] = ann.get(col)
        else:
            todo.append((idx, post_id, str(row[args.text_col]) if pd.notna(row[args.text_col]) else ""))

    log(f"\n{'='*70}\nPASS 1: {len(df)} rows total | {len(todo)} to process | "
        f"{len(checkpoint1)} from checkpoint | concurrency={args.concurrency}\n{'='*70}")

    def _pass1_worker(item):
        idx, post_id, post_text = item
        ann = classify_exit_intention(client, api_key, post_text, post_id, args.max_text_chars)
        update_checkpoint(checkpoint1, post_id, ann)
        save_checkpoint(ckpt_pass1, checkpoint1)
        return idx, post_id, ann

    done_count = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(_pass1_worker, item): item for item in todo}
        for fut in as_completed(futures):
            idx, post_id, ann = fut.result()
            for col in PASS1_COLS:
                df.at[idx, col] = ann.get(col)
            done_count += 1
            if done_count % 25 == 0 or done_count == len(todo):
                log(f"[P1] {done_count}/{len(todo)} done | last id={post_id} -> {ann['exit_intention_class']}")

    df.to_csv(out_pass1, index=False)
    log(f"[i] Saved: {out_pass1}")

    log("\n── PASS 1 SHARD SUMMARY ──────────────────────────────")
    vc = df["exit_intention_class"].value_counts(dropna=False)
    for label, count in vc.items():
        log(f"  {str(label):<25} {count:>5}  ({count/len(df)*100:.1f}%)")

    # ══════════════════════════════════════════════════════════════════════
    # PASS 2 — exit level + reasons for exit-positive rows in this shard
    # ══════════════════════════════════════════════════════════════════════
    df_exit = df[df["exit_intention_class"].isin(EXIT_POSITIVE)]
    log(f"\n{'='*70}\nPASS 2: {len(df_exit)} exit-positive rows in this shard\n{'='*70}")

    checkpoint2 = load_checkpoint(ckpt_pass2)
    todo2 = []
    for idx, row in df_exit.iterrows():
        post_id = str(row[id_col])
        if post_id in checkpoint2:
            ann2 = checkpoint2[post_id]
            for col in PASS2_COLS:
                df.at[idx, col] = ann2.get(col)
            if ann2.get("exit_reasoning"):
                df.at[idx, "exit_reasoning"] = ann2["exit_reasoning"]
        else:
            existing_reasoning = str(row.get("exit_reasoning", "") or "")
            post_text = str(row[args.text_col]) if pd.notna(row[args.text_col]) else ""
            todo2.append((idx, post_id, post_text, existing_reasoning))

    log(f"[i] {len(todo2)} to process | {len(checkpoint2)} from checkpoint")

    def _pass2_worker(item):
        idx, post_id, post_text, existing_reasoning = item
        ann2 = classify_exit_level_and_reasons(client, api_key, post_text, existing_reasoning, args.max_text_chars)
        update_checkpoint(checkpoint2, post_id, ann2)
        save_checkpoint(ckpt_pass2, checkpoint2)
        return idx, post_id, ann2

    done_count2 = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(_pass2_worker, item): item for item in todo2}
        for fut in as_completed(futures):
            idx, post_id, ann2 = fut.result()
            for col in PASS2_COLS:
                df.at[idx, col] = ann2.get(col)
            if ann2.get("exit_reasoning"):
                df.at[idx, "exit_reasoning"] = ann2["exit_reasoning"]
            done_count2 += 1
            if done_count2 % 25 == 0 or done_count2 == len(todo2):
                log(f"[P2] {done_count2}/{len(todo2)} done | last id={post_id} -> "
                    f"level={ann2['exit_level']} primary={ann2['exit_reason_primary']}")

    df.to_csv(out_pass2, index=False)
    log(f"[i] Saved: {out_pass2}")

    log("\n── PASS 2 SHARD SUMMARY (exit-positive rows) ─────────")
    df_ep = df[df["exit_intention_class"].isin(EXIT_POSITIVE)]
    vc2 = df_ep["exit_level"].value_counts(dropna=False)
    for label, count in vc2.items():
        log(f"  {str(label):<25} {count:>5}")
    vc3 = df_ep["exit_reason_primary"].value_counts(dropna=False)
    log("  --- primary reason ---")
    for label, count in vc3.items():
        log(f"  {str(label):<30} {count:>5}")

    client.close()
    log(f"\n[✓] Shard {args.shard_id} complete.")


if __name__ == "__main__":
    main()
