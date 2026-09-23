import pandas as pd

FINAL_PATH   = "bat_posts_results_final.csv"
CLEANUP_PATH = "cleanup_results_shard000.csv"   # output from the cleanup re-run
OUTPUT_PATH  = "bat_posts_results_final_patched.csv"

final_df = pd.read_csv(FINAL_PATH)
cleanup_df = pd.read_csv(CLEANUP_PATH)

final_df["post_id"] = final_df["post_id"].astype(str).str.strip()
cleanup_df["post_id"] = cleanup_df["post_id"].astype(str).str.strip()

n_errors_before = (final_df["EX"] == "ERROR").sum()
print(f"EX=ERROR rows before patch: {n_errors_before}")

# Only patch in cleanup rows that actually succeeded this time (not still ERROR)
successful_cleanup = cleanup_df[cleanup_df["EX"] != "ERROR"]
print(f"Cleanup run produced {len(successful_cleanup)}/{len(cleanup_df)} newly-successful rows")

cleanup_lookup = successful_cleanup.set_index("post_id")

patched_count = 0
for idx, row in final_df.iterrows():
    if row["EX"] == "ERROR" and row["post_id"] in cleanup_lookup.index:
        new_row = cleanup_lookup.loc[row["post_id"]]
        for col in ["EX", "EMO", "COG", "MD", "bat_score",
                    "EX_reasoning", "EMO_reasoning", "COG_reasoning", "MD_reasoning"]:
            final_df.at[idx, col] = new_row[col]
        patched_count += 1

print(f"Patched {patched_count} rows")

n_errors_after = (final_df["EX"] == "ERROR").sum()
print(f"EX=ERROR rows after patch: {n_errors_after}")

final_df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved: {OUTPUT_PATH}")

if n_errors_after > 0:
    print(f"\n{n_errors_after} posts still failed even in the cleanup run — these consistently "
          f"fail the API call. Worth inspecting their text directly, or accepting them as "
          f"permanent gaps in the dataset if the count is small.")
