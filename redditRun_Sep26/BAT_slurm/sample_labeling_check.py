import pandas as pd
import glob

SHARD_PATTERN = "bat_posts_results_shard*.csv"
OUTPUT_PATH   = "labeling_check.csv"
N_POSITIVE    = 12   # bat_score >= 1
N_ZERO        = 8    # bat_score == 0
RANDOM_STATE  = 42

shard_files = sorted(glob.glob(SHARD_PATTERN))
print(f"Found {len(shard_files)} shard files")

dfs = []
for f in shard_files:
    try:
        df = pd.read_csv(f)
        dfs.append(df)
    except Exception as e:
        # Skip a file caught mid-write by the still-running job
        print(f"  [!] Skipped {f} (likely being written right now): {e}")

merged = pd.concat(dfs, ignore_index=True)
print(f"Total processed rows so far: {len(merged)}")

positive_pool = merged[merged["bat_score"] >= 1]
zero_pool = merged[merged["bat_score"] == 0]
print(f"Available: {len(positive_pool)} with bat_score>=1, {len(zero_pool)} with bat_score==0")

n_pos = min(N_POSITIVE, len(positive_pool))
n_zero = min(N_ZERO, len(zero_pool))
if n_pos < N_POSITIVE:
    print(f"  [!] Only {n_pos} bat_score>=1 rows available (wanted {N_POSITIVE})")
if n_zero < N_ZERO:
    print(f"  [!] Only {n_zero} bat_score==0 rows available (wanted {N_ZERO})")

positive_sample = positive_pool.sample(n=n_pos, random_state=RANDOM_STATE)
zero_sample = zero_pool.sample(n=n_zero, random_state=RANDOM_STATE)

sample = pd.concat([positive_sample, zero_sample], ignore_index=True)
sample = sample.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)  # shuffle so they're not grouped

sample.to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved {len(sample)} sampled rows to {OUTPUT_PATH} "
      f"({n_pos} with bat_score>=1, {n_zero} with bat_score==0)")
print(f"post_ids sampled: {sample['post_id'].tolist()}")