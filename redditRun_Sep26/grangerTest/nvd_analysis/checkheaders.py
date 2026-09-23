"""
Header/structure check for the three files going into the merge:
  1. bat_posts_results_with_dates.csv   (BAT scores: EX, EMO, COG, MD + dates)
  2. exit_annotated_pass1.csv           (exit intention class)
  3. nvd_merged_full.csv                (merged CVE data, all chunks combined)

Adjust the three paths below to match where each file lives on your Mac.
"""

import pandas as pd
import os

# ----------------------------------------------------------------------
# CONFIG - adjust these paths
# ----------------------------------------------------------------------
FILES = {
    "bat_with_dates": "/Users/nadia/Desktop/redditRun_june/bat_posts_results_with_dates.csv",
    "exit_pass1":     "/Users/nadia/Desktop/redditRun_june/exit_annotated_pass1.csv",
    "nvd_merged":     "/Users/nadia/Desktop/redditRun_june/nvd_chunks/nvd_merged_full.csv",
}


def fast_row_count(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for _ in f) - 1
    except Exception as e:
        return f"[count failed: {e}]"


def inspect_csv(label, path, nrows=3):
    print(f"\n{'=' * 75}")
    print(f"{label}  ->  {path}")
    print("=" * 75)

    if not os.path.exists(path):
        print("  [NOT FOUND -- check the path]")
        return None

    try:
        df_head = pd.read_csv(path, nrows=nrows)
    except Exception as e:
        print(f"  [FAILED TO READ: {e}]")
        return None

    n_rows = fast_row_count(path)
    print(f"  Row count (excl. header): {n_rows}")
    print(f"  Column count: {len(df_head.columns)}")
    print(f"  Columns: {list(df_head.columns)}")
    print(f"\n  Dtypes (from first {nrows} rows):")
    print("  " + df_head.dtypes.to_string().replace("\n", "\n  "))
    print(f"\n  Sample rows:")
    print(df_head.to_string(max_colwidth=40))

    return df_head


def flag_relevant_columns(label, df):
    if df is None:
        return
    cols_lower = {c: c.lower() for c in df.columns}

    id_cols = [c for c, lc in cols_lower.items() if lc in ("id", "cve_id", "post_id")]
    date_cols = [c for c, lc in cols_lower.items()
                 if any(k in lc for k in ("date", "time", "created", "published", "modified"))]
    cvss_cols = [c for c, lc in cols_lower.items() if "cvss" in lc or "basescore" in lc or "base_score" in lc]
    bat_cols = [c for c, lc in cols_lower.items() if lc in ("ex", "emo", "cog", "md", "is_burnout", "burnout")]
    exit_cols = [c for c, lc in cols_lower.items() if "exit" in lc]
    subreddit_cols = [c for c, lc in cols_lower.items() if "subreddit" in lc]

    print(f"\n  --- relevant columns for {label} ---")
    print(f"  id columns:        {id_cols if id_cols else 'NONE FOUND'}")
    print(f"  date/time columns: {date_cols if date_cols else 'NONE FOUND'}")
    print(f"  cvss columns:      {cvss_cols if cvss_cols else 'n/a'}")
    print(f"  BAT columns:       {bat_cols if bat_cols else 'n/a'}")
    print(f"  exit columns:      {exit_cols if exit_cols else 'n/a'}")
    print(f"  subreddit columns: {subreddit_cols if subreddit_cols else 'n/a'}")


def main():
    dfs = {}
    for label, path in FILES.items():
        df = inspect_csv(label, path)
        dfs[label] = df
        flag_relevant_columns(label, df)

    # Cross-file checks relevant to the planned merge
    print(f"\n{'=' * 75}")
    print("MERGE DIAGNOSTICS")
    print("=" * 75)

    if dfs.get("exit_pass1") is not None:
        exit_cols_lower = [c.lower() for c in dfs["exit_pass1"].columns]
        has_date = any(k in c for c in exit_cols_lower for k in ("date", "time", "created"))
        print(f"exit_pass1 has its own date column: {has_date}")
        if not has_date:
            print("  -> Will need to join exit_pass1 on 'id' to bat_posts_results_with_dates.csv "
                  "to recover dates.")

    if dfs.get("bat_with_dates") is not None and dfs.get("nvd_merged") is not None:
        bat_date_cols = [c for c in dfs["bat_with_dates"].columns
                          if any(k in c.lower() for k in ("date", "time", "created"))]
        nvd_date_cols = [c for c in dfs["nvd_merged"].columns
                          if any(k in c.lower() for k in ("published", "date", "time"))]
        print(f"bat_with_dates date column(s): {bat_date_cols}")
        print(f"nvd_merged date column(s):     {nvd_date_cols}")
        print("  -> These need to be parsed to the same dtype/granularity (day, likely) "
              "before time-bucketing both series for cross-correlation.")

    print("\nDone. Review the columns above before writing the merge/cross-correlation script.")


if __name__ == "__main__":
    main()