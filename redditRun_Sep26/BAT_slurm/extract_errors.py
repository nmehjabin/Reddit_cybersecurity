import pandas as pd

FINAL_PATH  = "bat_posts_results_final.csv"
MASTER_PATH = "master_posts_burnout_only.csv"
OUTPUT_PATH = "cleanup_input.csv"

final_df = pd.read_csv(FINAL_PATH)
error_ids = final_df[final_df["EX"] == "ERROR"]["post_id"].astype(str).str.strip().tolist()
print(f"Found {len(error_ids)} posts with EX=ERROR")

master_df = pd.read_csv(MASTER_PATH)
master_df["id"] = master_df["id"].astype(str).str.strip()

cleanup_df = master_df[master_df["id"].isin(error_ids)].copy()
print(f"Matched {len(cleanup_df)}/{len(error_ids)} error post_ids in master file")

missing = set(error_ids) - set(cleanup_df["id"])
if missing:
    print(f"  [!] {len(missing)} error post_ids not found in master file (unexpected): {sorted(missing)[:10]}")

cleanup_df["predicted_label"] = 1
out_cols = ["id", "text", "predicted_label"] + [
    c for c in cleanup_df.columns if c not in ("id", "text", "predicted_label")
]
cleanup_df = cleanup_df[out_cols]

cleanup_df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved: {OUTPUT_PATH} ({len(cleanup_df)} rows) — ready to feed into the annotation script")
