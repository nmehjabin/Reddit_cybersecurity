"""
Analysis 3b, Step 1b: Backfill the gaps found by the merge/gap-check script.

This is a targeted variant of the original full-range downloader -- same
fetch_chunk() / parse_cve_record() logic, same retry/backoff behavior, same
output format -- but instead of walking the entire 2018-2026 span, it reads
missing_date_ranges.csv (produced by merge_nvd_full.py's gap check) and only
fetches those specific windows.

Run this, then re-run merge_nvd_full.py to regenerate the merged file and
confirm the gaps are closed.

Requirements: same as the original script (requests, pandas).
"""

import requests
import pandas as pd
import time
import os
from datetime import datetime, timedelta

# ============================================================
# CONFIG
# ============================================================
API_KEY = None   # <-- paste your free NVD API key here as a string, or leave None
MIN_CVSS = 9.0
CHUNK_DAYS = 119          # stay safely under the 120-day API limit (used only as a
                          # defensive sub-chunk size, in case any gap exceeds 120 days)
OUT_DIR = "nvd_chunks"    # same folder your existing chunks and missing_date_ranges.csv live in
MISSING_RANGES_CSV = os.path.join(OUT_DIR, "missing_date_ranges.csv")
BASE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"

# Sleep between requests: 6s without a key (5 req / 30s), 0.65s with a key
# (50 req / 30s), per NVD's documented best practice.
SLEEP_SECONDS = 0.65 if API_KEY else 6.0


def fetch_chunk(start, end, min_cvss, max_retries=5):
    """Fetch all CVEs published in [start, end) with CVSS v3 base score
    >= min_cvss, handling pagination within the chunk.

    Transient errors (503 Service Unavailable, connection timeouts) are
    common on NVD's infrastructure and are retried with exponential
    backoff. A 403 (rate limit) is NOT retried automatically here -- it
    raises immediately so the caller can stop and let the rolling rate
    limit window clear before the next manual re-run.
    """
    headers = {"apiKey": API_KEY} if API_KEY else {}
    all_cves = []
    start_index = 0
    results_per_page = 2000

    while True:
        params = {
            "pubStartDate": start.strftime("%Y-%m-%dT%H:%M:%S.000"),
            "pubEndDate": end.strftime("%Y-%m-%dT%H:%M:%S.000"),
            "cvssV3Severity": "CRITICAL",  # pre-filter server-side (CRITICAL = 9.0-10.0 in v3)
            "resultsPerPage": results_per_page,
            "startIndex": start_index,
        }

        resp = None
        last_error = None
        for attempt in range(max_retries):
            try:
                resp = requests.get(BASE_URL, params=params, headers=headers, timeout=60)
            except requests.exceptions.RequestException as e:
                last_error = e
                resp = None

            if resp is not None and resp.status_code == 403:
                raise RuntimeError(
                    "403 Forbidden -- you're likely rate-limited. Wait a few "
                    "minutes and re-run (completed chunks will be skipped)."
                )
            if resp is not None and resp.status_code == 200:
                break

            # Transient failure (503, connection error, etc.) -- back off and retry
            wait = min(60, 5 * (2 ** attempt))
            status = resp.status_code if resp is not None else f"connection error ({last_error})"
            print(f"    Transient error ({status}) on attempt {attempt + 1}/{max_retries}, "
                  f"retrying in {wait}s...")
            time.sleep(wait)
        else:
            # Exhausted retries
            status = resp.status_code if resp is not None else last_error
            raise RuntimeError(
                f"NVD API still failing after {max_retries} retries (last status: {status}). "
                "This chunk was NOT saved -- re-run the script later to resume."
            )

        data = resp.json()
        vulns = data.get("vulnerabilities", [])
        all_cves.extend(vulns)

        total_results = data.get("totalResults", 0)
        start_index += results_per_page
        if start_index >= total_results:
            break
        time.sleep(SLEEP_SECONDS)

    return all_cves


def parse_cve_record(item):
    """Extract the fields we need from one NVD API CVE record."""
    cve = item.get("cve", {})
    cve_id = cve.get("id")
    published = cve.get("published")
    last_modified = cve.get("lastModified")
    source_identifier = cve.get("sourceIdentifier")

    # CVSS v3.1 preferred, fall back to v3.0
    base_score = None
    severity = None
    metrics = cve.get("metrics", {})
    for key in ("cvssMetricV31", "cvssMetricV30"):
        if key in metrics and metrics[key]:
            cvss_data = metrics[key][0].get("cvssData", {})
            base_score = cvss_data.get("baseScore")
            severity = cvss_data.get("baseSeverity")
            break

    return {
        "cve_id": cve_id,
        "published": published,
        "last_modified": last_modified,
        "source_identifier": source_identifier,
        "cvss_base_score": base_score,
        "cvss_severity": severity,
    }


def daterange_chunks(start, end, chunk_days):
    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days), end)
        yield current, chunk_end
        current = chunk_end


def load_missing_ranges(csv_path, chunk_days=CHUNK_DAYS):
    """
    Read missing_date_ranges.csv (produced by the gap-check step) and yield
    (start, end) pairs to fetch. Any single gap that happens to exceed the
    NVD API's 120-day limit is sub-chunked defensively -- shouldn't occur
    for the standard ~119-day gaps this pipeline produces, since the
    original downloader's CHUNK_DAYS=119 keeps every gap under the limit,
    but this guards against a gap that spans multiple missed chunks.
    """
    gaps_df = pd.read_csv(csv_path)
    for _, row in gaps_df.iterrows():
        start = datetime.strptime(str(row["gap_start"]), "%Y-%m-%d")
        end = datetime.strptime(str(row["gap_end"]), "%Y-%m-%d")
        if (end - start).days <= 120:
            yield start, end
        else:
            yield from daterange_chunks(start, end, chunk_days)


if __name__ == "__main__":
    if not os.path.exists(MISSING_RANGES_CSV):
        print(f"Could not find {MISSING_RANGES_CSV}.")
        print("Run the merge/gap-check script first to generate it.")
        raise SystemExit(1)

    chunks = list(load_missing_ranges(MISSING_RANGES_CSV))
    print(f"Loaded {len(chunks)} gap window(s) to backfill from {MISSING_RANGES_CSV}:")
    for s, e in chunks:
        print(f"  {s.date()} to {e.date()}  ({(e - s).days} days)")
    print(f"\nUsing API key: {'YES' if API_KEY else 'NO'} "
          f"(sleep = {SLEEP_SECONDS}s between requests)\n")

    failed_chunks = []
    for i, (chunk_start, chunk_end) in enumerate(chunks, 1):
        fname = os.path.join(
            OUT_DIR,
            f"nvd_chunk_{chunk_start.strftime('%Y%m%d')}_{chunk_end.strftime('%Y%m%d')}.csv"
        )
        if os.path.exists(fname):
            print(f"[{i}/{len(chunks)}] SKIP (already exists): {fname}")
            continue

        print(f"[{i}/{len(chunks)}] Fetching {chunk_start.date()} to {chunk_end.date()} ...")
        try:
            raw_items = fetch_chunk(chunk_start, chunk_end, MIN_CVSS)
        except RuntimeError as e:
            print(f"  ERROR: {e}")
            if "403" in str(e):
                print("  Rate-limited -- stopping the whole run now. Wait a few minutes "
                      "and re-run to resume.")
                break
            print("  Skipping this chunk for now, continuing to the next one. "
                  "Re-run later to fill in this gap.")
            failed_chunks.append((chunk_start.date(), chunk_end.date()))
            time.sleep(SLEEP_SECONDS)
            continue

        records = [parse_cve_record(item) for item in raw_items]
        chunk_df = pd.DataFrame(records)

        # Double-check the score filter client-side too (CRITICAL severity
        # band is 9.0-10.0 in CVSS v3, which matches MIN_CVSS=9.0, but this
        # guards against any edge cases / future CVSS version changes)
        if len(chunk_df) > 0 and "cvss_base_score" in chunk_df.columns:
            chunk_df = chunk_df[chunk_df["cvss_base_score"] >= MIN_CVSS]

        chunk_df.to_csv(fname, index=False)
        print(f"  Saved {len(chunk_df)} CVEs (CVSS >= {MIN_CVSS}) -> {fname}")
        time.sleep(SLEEP_SECONDS)

    print("\nBackfill run complete. If it stopped early due to rate limiting, "
          "just re-run -- it will resume from the next incomplete chunk.")
    if failed_chunks:
        print(f"\n{len(failed_chunks)} chunk(s) failed after retries and were SKIPPED:")
        for s, e in failed_chunks:
            print(f"  {s} to {e}")
        print("Re-run this script again to retry just these (completed chunks "
              "will be skipped automatically).")
    print("\nOnce done, re-run merge_nvd_full.py to regenerate nvd_merged_full.csv, "
          "nvd_run1_gte9_5.csv, nvd_run2_eq10.csv, and confirm the gap check is clean.")