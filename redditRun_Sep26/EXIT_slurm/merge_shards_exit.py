# -*- coding: utf-8 -*-
"""
merge_shards_exit.py — combine per-shard exit-intention outputs into final CSVs.

Usage:
  python merge_shards_exit.py \
      --input bat_posts_results_final_patched.csv \
      --output-base exit_annotated \
      --num_shards 20 \
      --id-col id

Produces:
  exit_annotated_pass1_merged.csv   (all rows, exit_intention_class only)
  exit_annotated_pass2_merged.csv   (all rows; exit-positive rows also have
                                      exit_level / exit_reason_primary / exit_reason_secondary)

Also flags:
  - any post_id present in the original input but missing from the merged
    output (so you know which shard to rerun)
  - duplicate post_ids across shards (kept: first occurrence)
"""

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd


def merge_pass(pattern: str, id_col: str, expected_ids: set, label: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[X] No files matched pattern: {pattern}")
        sys.exit(1)

    print(f"\n[i] {label}: merging {len(files)} shard files")
    frames = []
    for f in files:
        d = pd.read_csv(f, dtype=str)
        frames.append(d)
        print(f"    {f}: {len(d)} rows")

    merged = pd.concat(frames, ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates(subset=[id_col], keep="first")
    after = len(merged)
    if before != after:
        print(f"[!] {label}: dropped {before - after} duplicate post_ids across shards")

    got_ids = set(merged[id_col].astype(str))
    missing = expected_ids - got_ids
    if missing:
        print(f"[!] {label}: {len(missing)} post_ids from input are MISSING from merged output.")
        print(f"    First few missing: {list(missing)[:10]}")
    else:
        print(f"[✓] {label}: full coverage — all {len(expected_ids)} input rows accounted for.")

    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="bat_posts_results_final_patched.csv")
    parser.add_argument("--output-base", default="exit_annotated")
    parser.add_argument("--num_shards", type=int, default=20)
    parser.add_argument("--id-col", default="id")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"[X] Input file not found: {in_path}")
        sys.exit(1)

    df_in = pd.read_csv(in_path, dtype=str)
    id_col = args.id_col if args.id_col in df_in.columns else None
    if id_col is None:
        print(f"[!] ID column '{args.id_col}' not in input; falling back to row index as id "
              f"(this only works if shards were also run with default row-index ids).")
        df_in["_row_id"] = df_in.index.astype(str)
        id_col = "_row_id"
    expected_ids = set(df_in[id_col].astype(str))
    print(f"[i] Input: {len(df_in)} rows, {len(expected_ids)} unique ids")

    pass1_pattern = f"{args.output_base}_shard*_pass1.csv"
    pass2_pattern = f"{args.output_base}_shard*_pass2.csv"

    merged1 = merge_pass(pass1_pattern, id_col, expected_ids, "PASS 1")
    merged1.to_csv(f"{args.output_base}_pass1_merged.csv", index=False)
    print(f"[i] Saved: {args.output_base}_pass1_merged.csv ({len(merged1)} rows)")

    merged2 = merge_pass(pass2_pattern, id_col, expected_ids, "PASS 2")
    merged2.to_csv(f"{args.output_base}_pass2_merged.csv", index=False)
    print(f"[i] Saved: {args.output_base}_pass2_merged.csv ({len(merged2)} rows)")

    print("\n── FINAL SUMMARY (merged Pass 2 file) ──────────────────")
    vc = merged2["exit_intention_class"].value_counts(dropna=False)
    for label, count in vc.items():
        print(f"  {str(label):<25} {count:>5}  ({count/len(merged2)*100:.1f}%)")

    ep = merged2[merged2["exit_intention_class"].isin({"exit_explicit", "exit_contemplating"})]
    print(f"\nExit-positive rows: {len(ep)}")
    print("Primary reason breakdown:")
    vc3 = ep["exit_reason_primary"].value_counts(dropna=False)
    for label, count in vc3.items():
        print(f"  {str(label):<30} {count:>5}")


if __name__ == "__main__":
    main()
