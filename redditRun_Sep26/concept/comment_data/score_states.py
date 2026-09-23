#!/usr/bin/env python3
"""
score_stats.py

Descriptive overview of score_sample_wide.parquet -- the final 53,283-comment
x 14-label scored table. Answers: how many comments, how many match each
label (unweighted and corpus-weighted), how many labels does a typical
comment match, and confirms remaining missingness per label.

Run from your activated venv:
    python score_stats.py
"""

import os
import pandas as pd
import numpy as np

OUTPUT_DIR = "/Users/nadia/Desktop/redditRun_june/comment_data/"
WIDE_PATH = os.path.join(OUTPUT_DIR, "score_sample_wide.parquet")
STATS_DIR = os.path.join(OUTPUT_DIR, "score_results")
os.makedirs(STATS_DIR, exist_ok=True)

NON_LABEL_COLS = {"id", "subreddit_source", "post_id", "w"}


def main():
    wide = pd.read_parquet(WIDE_PATH)
    label_cols = [c for c in wide.columns if c not in NON_LABEL_COLS]

    print("=" * 70)
    print("1. OVERALL SHAPE")
    print("=" * 70)
    print(f"Total comments: {len(wide):,}")
    print(f"Total labels: {len(label_cols)}")
    print(f"Labels: {label_cols}\n")

    print("Comments per subreddit:")
    print(wide["subreddit_source"].value_counts())

    print("\n" + "=" * 70)
    print("2. PER-LABEL: MATCHES, NON-MATCHES, MISSING")
    print("=" * 70)

    rows = []
    for col in label_cols:
        n_total = len(wide)
        n_missing = wide[col].isna().sum()
        n_scored = n_total - n_missing
        n_matched = (wide[col] == 1).sum()
        n_not_matched = (wide[col] == 0).sum()
        prevalence_unweighted = n_matched / n_scored if n_scored > 0 else float("nan")

        # Corpus-weighted prevalence -- accounts for sysadmin subsampling
        valid = wide[wide[col].notna()]
        prevalence_weighted = np.average(valid[col], weights=valid["w"]) if len(valid) > 0 else float("nan")

        rows.append({
            "label": col,
            "n_matched": n_matched,
            "n_not_matched": n_not_matched,
            "n_missing": n_missing,
            "prevalence_unweighted": prevalence_unweighted,
            "prevalence_weighted": prevalence_weighted,
        })

    label_stats = pd.DataFrame(rows).sort_values("prevalence_weighted", ascending=False)
    pd.set_option("display.float_format", lambda x: f"{x:.2%}" if 0 <= x <= 1 else f"{x:.4f}")
    print(label_stats.to_string(index=False))

    label_stats.to_csv(os.path.join(STATS_DIR, "label_prevalence.csv"), index=False)
    print(f"\nSaved: {os.path.join(STATS_DIR, 'label_prevalence.csv')}")

    print("\n" + "=" * 70)
    print("3. DISTRIBUTION: HOW MANY LABELS DOES EACH COMMENT MATCH?")
    print("=" * 70)
    print("(missing cells treated as 0/no-match for this count -- see note below)\n")

    labels_matched_per_comment = wide[label_cols].fillna(0).sum(axis=1)
    dist = labels_matched_per_comment.value_counts().sort_index()
    dist_pct = (dist / len(wide) * 100).round(2)

    dist_df = pd.DataFrame({"n_labels_matched": dist.index, "n_comments": dist.values, "pct": dist_pct.values})
    print(dist_df.to_string(index=False))

    print(f"\nComments matching ZERO labels (orphans, full-corpus scale): "
          f"{dist_df.loc[dist_df['n_labels_matched']==0, 'n_comments'].sum():,} "
          f"({dist_df.loc[dist_df['n_labels_matched']==0, 'pct'].sum():.1f}%)")
    print(f"Mean labels matched per comment: {labels_matched_per_comment.mean():.2f}")
    print(f"Median labels matched per comment: {labels_matched_per_comment.median():.0f}")

    dist_df.to_csv(os.path.join(STATS_DIR, "labels_per_comment_distribution.csv"), index=False)
    print(f"\nSaved: {os.path.join(STATS_DIR, 'labels_per_comment_distribution.csv')}")

    print("\n" + "=" * 70)
    print("4. REMAINING MISSINGNESS SUMMARY")
    print("=" * 70)
    total_cells = len(wide) * len(label_cols)
    total_missing = wide[label_cols].isna().sum().sum()
    comments_with_any_missing = wide[label_cols].isna().any(axis=1).sum()
    print(f"Total cells missing: {total_missing:,} / {total_cells:,} ({total_missing/total_cells:.3%})")
    print(f"Comments with at least one missing label: {comments_with_any_missing:,} "
          f"({comments_with_any_missing/len(wide):.3%})")
    print("\nNote: section 3's distribution treats missing as 0/no-match, which slightly "
          "undercounts a comment's true label matches if it has a missing cell (currently "
          "affects a tiny fraction of comments -- see section 4).")


if __name__ == "__main__":
    main()