"""
Merge all shard output CSVs (bat_posts_results_shard000.csv, _shard001.csv, ...)
into a single final CSV, and report basic sanity checks (row count, duplicate
post_ids across shards, missing post_ids vs. the master input).
"""

import pandas as pd
import glob
import os

SHARD_PATTERN = "bat_posts_results_shard*.csv"
MASTER_INPUT  = "master_posts_burnout_only.csv"   # to check completeness against
FINAL_OUTPUT  = "bat_posts_results_final.csv"

shard_files = sorted(glob.glob(SHARD_PATTERN))
print(f"Found {len(shard_files)} shard files:")
for f in shard_files:
    print(f"  - {f}")

if not shard_files:
    raise SystemExit(f"No files matched pattern '{SHARD_PATTERN}' in {os.getcwd()}")

dfs = [pd.read_csv(f) for f in shard_files]
merged = pd.concat(dfs, ignore_index=True)
print(f"\nTotal rows across shards (before dedup): {len(merged)}")

dupes = merged["post_id"].duplicated().sum()
if dupes:
    print(f"[!] {dupes} duplicate post_id rows found — keeping first occurrence.")
    merged = merged.drop_duplicates(subset="post_id", keep="first")

print(f"Total rows after dedup: {len(merged)}")

if os.path.exists(MASTER_INPUT):
    master_df = pd.read_csv(MASTER_INPUT)
    master_ids = set(master_df["id"].astype(str).str.strip())
    merged_ids = set(merged["post_id"].astype(str).str.strip())
    missing = master_ids - merged_ids
    print(f"\nMaster burnout-only posts : {len(master_ids)}")
    print(f"Present in merged output  : {len(merged_ids & master_ids)}")
    if missing:
        print(f"[!] {len(missing)} posts from master are missing from merged output "
              f"(likely still-running or failed shards).")
    else:
        print("All master posts accounted for. ✓")

merged.to_csv(FINAL_OUTPUT, index=False)
print(f"\nSaved: {FINAL_OUTPUT}")

error_n = (merged["EX"] == "ERROR").sum()
if error_n:
    print(f"\n[!] {error_n} rows have EX=ERROR (failed API calls) — consider a cleanup re-run on just those post_ids.")
