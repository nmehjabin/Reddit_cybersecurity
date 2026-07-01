"""
Scope diagnostics before merging bat_with_dates + exit_pass1 + nvd_merged_full.

Answers three specific questions raised by the header check:
  1. In bat_with_dates: are EX/EMO/COG/MD only populated for triage-positive
     rows, or are they populated everywhere? What does row_type split look like?
  2. In exit_pass1: what is predicted_label's value distribution, and is
     exit_intention_class only populated where predicted_label indicates burnout?
  3. Do exit_pass1['id'] and bat_with_dates['post_id'] share the same ID
     format/scheme, and how much do they actually overlap?

Uses usecols to keep memory reasonable given these are 1.5-2.9M row files.
"""

import pandas as pd
import os

# ----------------------------------------------------------------------
# CONFIG - adjust paths
# ----------------------------------------------------------------------
BAT_PATH = "/Users/nadia/Desktop/redditRun_june/bat_posts_results_with_dates.csv"
EXIT_PATH = "/Users/nadia/Desktop/redditRun_june/exit_annotated_pass1.csv"
NVD_PATH = "/Users/nadia/Desktop/redditRun_june/nvd_chunks/nvd_merged_full.csv"


def section(title):
    print(f"\n{'=' * 75}")
    print(title)
    print("=" * 75)


def check_bat_file():
    section("BAT FILE: row_type / triage / BAT-score population")

    if not os.path.exists(BAT_PATH):
        print(f"  [NOT FOUND: {BAT_PATH}]")
        return

    usecols = ["row_type", "post_id", "comment_id", "triage", "na_subtype", "EX", "EMO", "COG", "MD"]
    df = pd.read_csv(BAT_PATH, usecols=usecols)

    print(f"Total rows: {len(df)}")

    print("\nrow_type value counts:")
    print(df["row_type"].value_counts(dropna=False).to_string())

    print("\ntriage value counts:")
    print(df["triage"].value_counts(dropna=False).to_string())

    print("\nna_subtype value counts (top 15):")
    print(df["na_subtype"].value_counts(dropna=False).head(15).to_string())

    # Are EX/EMO/COG/MD null only for certain triage values?
    print("\nEX null count by triage value:")
    print(df.groupby("triage", dropna=False)["EX"].apply(lambda s: s.isna().sum()).to_string())

    print("\nEX non-null count by triage value:")
    print(df.groupby("triage", dropna=False)["EX"].apply(lambda s: s.notna().sum()).to_string())

    # post_id vs comment_id null pattern by row_type
    print("\npost_id null count by row_type:")
    print(df.groupby("row_type", dropna=False)["post_id"].apply(lambda s: s.isna().sum()).to_string())
    print("\ncomment_id null count by row_type:")
    print(df.groupby("row_type", dropna=False)["comment_id"].apply(lambda s: s.isna().sum()).to_string())

    # Sample ID formats
    print("\nSample post_id values:", df["post_id"].dropna().unique()[:5].tolist())
    print("Sample comment_id values:", df["comment_id"].dropna().unique()[:5].tolist())

    return df


def check_exit_file():
    section("EXIT FILE: predicted_label / exit_intention_class relationship")

    if not os.path.exists(EXIT_PATH):
        print(f"  [NOT FOUND: {EXIT_PATH}]")
        return

    usecols = ["id", "predicted_label", "prediction_confidence", "exit_intention_class", "exit_confidence"]
    df = pd.read_csv(EXIT_PATH, usecols=usecols)

    print(f"Total rows: {len(df)}")

    print("\npredicted_label value counts:")
    print(df["predicted_label"].value_counts(dropna=False).to_string())

    print("\nexit_intention_class value counts (including null):")
    print(df["exit_intention_class"].value_counts(dropna=False).to_string())

    print("\nCross-tab: predicted_label x (exit_intention_class is populated):")
    df["exit_populated"] = df["exit_intention_class"].notna()
    print(pd.crosstab(df["predicted_label"], df["exit_populated"]).to_string())

    print("\nSample id values:", df["id"].dropna().unique()[:5].tolist())

    return df


def check_id_overlap(bat_df, exit_df):
    section("ID OVERLAP: exit_pass1.id vs bat_with_dates.post_id")

    if bat_df is None or exit_df is None:
        print("  Skipping -- one or both files failed to load above.")
        return

    bat_ids = set(bat_df["post_id"].dropna().astype(str))
    exit_ids = set(exit_df["id"].dropna().astype(str))

    print(f"Unique post_id in bat file:  {len(bat_ids)}")
    print(f"Unique id in exit file:      {len(exit_ids)}")

    overlap = bat_ids & exit_ids
    print(f"Overlap (exact string match): {len(overlap)}")
    print(f"  In bat but not exit: {len(bat_ids - exit_ids)}")
    print(f"  In exit but not bat: {len(exit_ids - bat_ids)}")

    if len(overlap) == 0:
        print("\n  WARNING: zero overlap on exact match. Checking for a prefix mismatch")
        print("  (e.g. Reddit-style 't3_xxxxx' vs bare 'xxxxx')...")
        bat_sample = list(bat_ids)[:5]
        exit_sample = list(exit_ids)[:5]
        print(f"  bat post_id sample:  {bat_sample}")
        print(f"  exit id sample:      {exit_sample}")

        # Try stripping a possible t3_ prefix and re-checking
        def strip_prefix(s):
            return s[3:] if s.startswith("t3_") else s

        bat_stripped = {strip_prefix(s) for s in bat_ids}
        exit_stripped = {strip_prefix(s) for s in exit_ids}
        overlap_stripped = bat_stripped & exit_stripped
        print(f"\n  Overlap after stripping any 't3_' prefix: {len(overlap_stripped)}")


def check_nvd_file():
    section("NVD FILE: date parsing check")

    if not os.path.exists(NVD_PATH):
        print(f"  [NOT FOUND: {NVD_PATH}]")
        return

    df = pd.read_csv(NVD_PATH, usecols=["cve_id", "published", "last_modified", "cvss_base_score"])
    print(f"Total rows: {len(df)}")

    df["published_parsed"] = pd.to_datetime(df["published"], errors="coerce")
    n_bad = df["published_parsed"].isna().sum()
    print(f"published: {n_bad} unparseable row(s) out of {len(df)}")
    print(f"published date range: {df['published_parsed'].min()} to {df['published_parsed'].max()}")

    print(f"\ncvss_base_score range: {df['cvss_base_score'].min()} to {df['cvss_base_score'].max()}")
    print(f"Rows with score > 9.5: {(df['cvss_base_score'] > 9.5).sum()}")
    print(f"Rows with score == 10: {(df['cvss_base_score'] == 10).sum()}")


if __name__ == "__main__":
    bat_df = check_bat_file()
    exit_df = check_exit_file()
    check_id_overlap(bat_df, exit_df)
    check_nvd_file()

    section("DONE")
    print("Review the triage/EX-null relationship, the predicted_label/exit_intention_class")
    print("relationship, and the ID overlap numbers above before writing the merge script.")