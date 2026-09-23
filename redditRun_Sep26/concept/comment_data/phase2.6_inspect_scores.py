#!/usr/bin/env python3
"""
inspect_scores.py

Diagnoses section 2.6's output at two levels:
1. Batch-level (scores.jsonl): full (10/10 answered) vs partial vs empty batches.
2. Cell-level (score_sample_wide.parquet): missing values per label, since a
   "successful" batch can still be missing individual comments if the model's
   JSON was malformed but partially parseable.

Prints 10 examples of problematic (partial/empty) raw batches and saves all
of them to a CSV for closer inspection.

Run from your activated venv:
    python inspect_scores.py
"""

import os
import json
import pandas as pd

OUTPUT_DIR = "/Users/nadia/Desktop/redditRun_june/comment_data/"
SCORES_PATH = os.path.join(OUTPUT_DIR, "score_results", "scores.jsonl")
WIDE_PATH = os.path.join(OUTPUT_DIR, "score_sample_wide.parquet")
COMMENTS_PER_CALL = 10  # matches score_corpus.py's config

NON_LABEL_COLS = {"id", "subreddit_source", "post_id", "w"}


def batch_level_check():
    print("=" * 70)
    print("1. BATCH-LEVEL CHECK (scores.jsonl)")
    print("=" * 70)

    # Retries APPEND rather than overwrite, so a given (label_id, chunk_idx)
    # can have multiple lines in the file -- keep only the LAST one per key,
    # matching exactly what reshape_and_weight()'s dict.update() logic does
    # when building the final table. Without this dedup, retried batches get
    # double-counted (once for the old failed attempt, once for the new one).
    latest_by_key = {}
    raw_line_count = 0

    with open(SCORES_PATH, "r") as f:
        for line in f:
            row = json.loads(line)
            raw_line_count += 1
            key = (row["label_id"], row["chunk_idx"])
            latest_by_key[key] = row  # later lines overwrite earlier ones for the same key

    print(f"Raw lines in file: {raw_line_count:,} (includes retries -- not the real batch count)")
    print(f"Distinct (label, chunk) batches: {len(latest_by_key):,}\n")

    total = 0
    full = 0
    partial = 0
    empty = 0
    problem_rows = []

    for row in latest_by_key.values():
        total += 1
        n_scores = len(row["scores"])

        if n_scores == 0:
            empty += 1
            row["_status"] = "empty"
            row["_n_scores"] = 0
            problem_rows.append(row)
        elif n_scores < COMMENTS_PER_CALL:
            partial += 1
            row["_status"] = "partial"
            row["_n_scores"] = n_scores
            problem_rows.append(row)
        else:
            full += 1

    print(f"Total distinct batches: {total:,}")
    print(f"Full ({COMMENTS_PER_CALL}/{COMMENTS_PER_CALL} answered): {full:,} ({full/total:.1%})")
    print(f"Partial (some but not all answered): {partial:,} ({partial/total:.1%})")
    print(f"Empty (0 answered): {empty:,} ({empty/total:.1%})")

    return problem_rows


def cell_level_check():
    print("\n" + "=" * 70)
    print("2. CELL-LEVEL CHECK (score_sample_wide.parquet)")
    print("=" * 70)

    wide = pd.read_parquet(WIDE_PATH)
    label_cols = [c for c in wide.columns if c not in NON_LABEL_COLS]

    print(f"{len(wide):,} comments x {len(label_cols)} labels = "
          f"{len(wide) * len(label_cols):,} total cells expected\n")

    total_missing = 0
    for col in label_cols:
        n_missing = wide[col].isna().sum()
        total_missing += n_missing
        marker = "  ⚠️" if n_missing / len(wide) > 0.02 else "  "
        print(f"{marker} {col}: {n_missing:,} missing ({n_missing/len(wide):.2%})")

    total_cells = len(wide) * len(label_cols)
    print(f"\nOverall: {total_missing:,} / {total_cells:,} cells missing "
          f"({total_missing/total_cells:.2%})")


def show_and_save_examples(problem_rows):
    print("\n" + "=" * 70)
    print("3. EXAMPLES OF PROBLEMATIC BATCHES")
    print("=" * 70)

    if not problem_rows:
        print("None found -- every batch fully succeeded.")
        return

    print(f"\nFirst 10 of {len(problem_rows):,} problematic batches:\n")
    for row in problem_rows[:10]:
        print(f"[{row['_status']}] label={row['label_name']!r} chunk_idx={row['chunk_idx']} "
              f"({row['_n_scores']}/{COMMENTS_PER_CALL} answered)")
        print(f"  raw scores dict: {row['scores']}")
        print()

    out_path = os.path.join(OUTPUT_DIR, "score_results", "problematic_batches.csv")
    df = pd.DataFrame(problem_rows)
    df.to_csv(out_path, index=False)
    print(f"Saved all {len(problem_rows):,} problematic batches to: {out_path}")


if __name__ == "__main__":
    problem_rows = batch_level_check()
    cell_level_check()
    show_and_save_examples(problem_rows)