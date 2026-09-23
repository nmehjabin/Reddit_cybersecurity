"""
Merge NVD chunk files (nvd_chunk_YYYYMMDD_YYYYMMDD.csv) into:
  1. nvd_merged_full.csv          -- everything, deduped, sorted by date
  2. nvd_run1_gte9_5.csv          -- CVSS > 9.5
  3. nvd_run2_eq10.csv            -- CVSS == 10
  4. missing_date_ranges.csv      -- exact gaps to hand back to the NVD downloader

Coverage is checked against your full study period (STUDY_START to STUDY_END
below), not just gaps between existing chunks -- so a missing first month or
a missing trailing month would also get caught, not just internal holes.
This only reads filenames, not file contents, so it runs instantly.
"""

import pandas as pd
import glob
import os
import re
from datetime import datetime

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
BASE_DIR = "/Users/nadia/Desktop/redditRun_june/nvd_chunks/"  # adjust if your folder name/path differs
CHUNK_PATTERN = "nvd_chunk_*.csv"

# Your study period -- adjust STUDY_END if your data should stop earlier than today
STUDY_START = datetime(2018, 1, 1)
STUDY_END = datetime(2026,4,26)

# Leave these as None to auto-detect. Set manually if auto-detect picks wrong
# (check the "Detected columns" line in the output, then hardcode here and re-run).
ID_COL_OVERRIDE = None      # e.g. "cve_id"
CVSS_COL_OVERRIDE = None    # e.g. "cvss_base_score"
DATE_COL_OVERRIDE = None    # e.g. "published"


def parse_chunk_dates(filename):
    """Extract (start, end) datetimes from nvd_chunk_YYYYMMDD_YYYYMMDD.csv"""
    m = re.search(r"(\d{8})_(\d{8})", os.path.basename(filename))
    if not m:
        return None, None
    start = datetime.strptime(m.group(1), "%Y%m%d")
    end = datetime.strptime(m.group(2), "%Y%m%d")
    return start, end


def check_coverage_gaps(files, study_start=STUDY_START, study_end=STUDY_END):
    """
    Detect gaps in date coverage using filename-encoded ranges only (no data load needed).
    Checks against the full study period, so a missing leading or trailing
    span is caught too, not just gaps between existing chunks.
    """
    ranges = []
    for f in files:
        start, end = parse_chunk_dates(f)
        if start is not None:
            ranges.append((start, end, f))
    ranges.sort(key=lambda x: x[0])

    print(f"\n{'=' * 70}")
    print("COVERAGE GAP CHECK (from filenames, vs. study period)")
    print("=" * 70)
    print(f"Study period:  {study_start.date()} to {study_end.date()}")
    print(f"Total chunks found: {len(ranges)}")
    if ranges:
        print(f"Data actually spans: {ranges[0][0].date()} to {ranges[-1][1].date()}")

    gaps = []

    # Leading gap: study starts before the first chunk
    if ranges and ranges[0][0] > study_start:
        gaps.append((study_start, ranges[0][0]))

    # Internal gaps: between consecutive chunks
    for i in range(len(ranges) - 1):
        end_i = ranges[i][1]
        start_next = ranges[i + 1][0]
        if start_next > end_i:
            gaps.append((end_i, start_next))

    # Trailing gap: last chunk ends before study end
    if ranges and ranges[-1][1] < study_end:
        gaps.append((ranges[-1][1], study_end))

    if gaps:
        print(f"\nFound {len(gaps)} gap(s) in coverage:")
        gap_rows = []
        for start, end in gaps:
            days = (end - start).days
            months = days / 30.44
            label_start = start.strftime("%b %Y")
            label_end = end.strftime("%b %Y")
            print(f"  MISSING: {start.date()} -> {end.date()}   "
                  f"({label_start} -> {label_end}, ~{months:.1f} months / {days} days)")
            gap_rows.append({
                "gap_start": start.date(),
                "gap_end": end.date(),
                "days_missing": days,
                "approx_months_missing": round(months, 1),
            })

        gap_df = pd.DataFrame(gap_rows)
        gap_csv_path = os.path.join(BASE_DIR, "missing_date_ranges.csv")
        gap_df.to_csv(gap_csv_path, index=False)
        total_days = sum(r["days_missing"] for r in gap_rows)
        print(f"\nTotal days missing across study period: {total_days} (~{total_days/30.44:.1f} months)")
        print(f"Saved -> {gap_csv_path}  (feed these exact ranges back into your NVD downloader)")
    else:
        print("\nNo gaps -- continuous coverage across the full study period.")

    return gaps, ranges


def detect_column(columns, keywords, label):
    cols_lower = {c.lower(): c for c in columns}
    for kw in keywords:
        for lc, orig in cols_lower.items():
            if kw in lc:
                return orig
    print(f"  WARNING: could not auto-detect {label} column from {list(columns)}")
    return None


def main():
    files = sorted(glob.glob(os.path.join(BASE_DIR, CHUNK_PATTERN)))
    print(f"Found {len(files)} chunk file(s) matching '{CHUNK_PATTERN}' in {BASE_DIR}")
    if not files:
        print("No files found -- check BASE_DIR / CHUNK_PATTERN.")
        return

    # 1) Gap check (filename-based, instant)
    gaps, ranges = check_coverage_gaps(files)

    # 2) Schema consistency check
    print(f"\n{'=' * 70}")
    print("SCHEMA CHECK")
    print("=" * 70)
    schemas = {}
    for f in files:
        cols = tuple(pd.read_csv(f, nrows=0).columns)
        schemas.setdefault(cols, []).append(f)
    if len(schemas) > 1:
        print(f"WARNING: {len(schemas)} distinct schemas found across chunks:")
        for cols, fs in schemas.items():
            print(f"  {len(fs)} file(s): {cols}")
        print("Resolve schema mismatch before merging. Stopping here.")
        return
    columns = list(schemas.keys())[0]
    print(f"Schema consistent across all {len(files)} files: {columns}")

    # 3) Auto-detect key columns
    id_col = ID_COL_OVERRIDE or detect_column(columns, ["cve_id", "cveid", "id"], "ID")
    cvss_col = CVSS_COL_OVERRIDE or detect_column(columns, ["cvss", "basescore", "base_score"], "CVSS")
    date_col = DATE_COL_OVERRIDE or detect_column(columns, ["published", "publishdate", "date", "time"], "date")

    print(f"\nDetected -> id: '{id_col}'  |  cvss: '{cvss_col}'  |  date: '{date_col}'")
    print("If any of these look wrong, set the *_OVERRIDE variables at the top and re-run.")

    if not all([id_col, cvss_col, date_col]):
        print("\nMissing a required column -- stopping before merge.")
        return

    # 4) Concatenate
    print(f"\n{'=' * 70}")
    print("MERGING")
    print("=" * 70)
    dfs = [pd.read_csv(f) for f in files]
    merged = pd.concat(dfs, ignore_index=True)
    print(f"Total rows before dedup: {len(merged)}")

    before = len(merged)
    merged = merged.drop_duplicates(subset=[id_col], keep="first")
    print(f"Removed {before - len(merged)} duplicate row(s) on '{id_col}'")
    print(f"Total rows after dedup: {len(merged)}")

    # 5) Parse + sort dates
    merged[date_col] = pd.to_datetime(merged[date_col], errors="coerce")
    n_bad_date = merged[date_col].isna().sum()
    if n_bad_date:
        print(f"WARNING: {n_bad_date} row(s) have unparseable dates in '{date_col}'")
    merged = merged.sort_values(date_col).reset_index(drop=True)
    print(f"Date range in data: {merged[date_col].min()} to {merged[date_col].max()}")

    # 6) Save full merged file
    full_path = os.path.join(BASE_DIR, "nvd_merged_full.csv")
    merged.to_csv(full_path, index=False)
    print(f"\nSaved -> {full_path}")

    # 7) Split into Run1 (>9.5) and Run2 (==10)
    merged[cvss_col] = pd.to_numeric(merged[cvss_col], errors="coerce")
    n_bad_score = merged[cvss_col].isna().sum()
    if n_bad_score:
        print(f"WARNING: {n_bad_score} row(s) have unparseable CVSS scores")

    run1 = merged[merged[cvss_col] > 9.5].copy()
    run2 = merged[merged[cvss_col] == 10].copy()

    run1_path = os.path.join(BASE_DIR, "nvd_run1_gte9_5.csv")
    run2_path = os.path.join(BASE_DIR, "nvd_run2_eq10.csv")
    run1.to_csv(run1_path, index=False)
    run2.to_csv(run2_path, index=False)

    print(f"\n{'=' * 70}")
    print("RUN SPLITS")
    print("=" * 70)
    print(f"Run1 (CVSS > 9.5): {len(run1)} rows -> {run1_path}")
    print(f"Run2 (CVSS == 10): {len(run2)} rows -> {run2_path}")

    # 8) Sanity check: every Run2 CVE should also be in Run1
    not_in_run1 = set(run2[id_col]) - set(run1[id_col])
    print(f"\nSanity check -- Run2 CVEs missing from Run1: {len(not_in_run1)} (expected: 0)")
    if not_in_run1:
        print(f"  Sample: {list(not_in_run1)[:10]}")

    # 9) Restate the gap warning in context of the final files
    if gaps:
        print(f"\n{'=' * 70}")
        print(f"REMINDER: {len(gaps)} coverage gap(s) remain relative to your "
              f"{STUDY_START.date()}-{STUDY_END.date()} study period (see top of output,")
        print("and missing_date_ranges.csv for the exact list).")
        print("Both nvd_run1_gte9_5.csv and nvd_run2_eq10.csv inherit these blind spots.")
        print("Backfill these ranges and re-run this script before using these files in")
        print("the Analysis 3a/3b cross-correlation or event-synchronized averaging.")
        print("=" * 70)


if __name__ == "__main__":
    main()