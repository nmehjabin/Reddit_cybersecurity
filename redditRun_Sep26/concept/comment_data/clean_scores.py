#!/usr/bin/env python3
"""
clean_scores.py

Removes empty/failed entries from scores.jsonl (written by the pre-patch
version of score_corpus.py, which recorded a line even when a batch failed
due to running out of credits). Run this once, before adding credits and
rerunning score_corpus.py -- otherwise the resume logic will treat those
failed batches as already done and skip them forever.

Backs up the original file before touching it.

Run from your activated venv:
    python clean_scores.py
"""

import json
import shutil
import os

OUTPUT_DIR = "/Users/nadia/Desktop/redditRun_june/comment_data/"
SCORES_PATH = os.path.join(OUTPUT_DIR, "score_results", "scores.jsonl")


def main():
    assert os.path.exists(SCORES_PATH), f"Not found: {SCORES_PATH}"

    backup_path = SCORES_PATH + ".backup"
    shutil.copy(SCORES_PATH, backup_path)
    print(f"Backed up to: {backup_path}")

    kept, dropped = 0, 0
    by_label_dropped = {}

    with open(SCORES_PATH, "r") as f, open(SCORES_PATH + ".clean", "w") as out:
        for line in f:
            row = json.loads(line)
            if row["scores"]:
                out.write(line)
                kept += 1
            else:
                dropped += 1
                by_label_dropped[row["label_name"]] = by_label_dropped.get(row["label_name"], 0) + 1

    os.replace(SCORES_PATH + ".clean", SCORES_PATH)

    print(f"\nKept: {kept:,} (genuinely scored batches)")
    print(f"Dropped: {dropped:,} (empty/failed -- will retry on next score_corpus.py run)")
    if by_label_dropped:
        print("\nDropped by label:")
        for name, count in sorted(by_label_dropped.items(), key=lambda x: -x[1]):
            print(f"  {name}: {count:,}")

    print(f"\n{SCORES_PATH} is now clean. Add credits, then run:")
    print("  python score_corpus.py")


if __name__ == "__main__":
    main()
