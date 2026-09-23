"""
inspect_sysadmin_comments.py

Purpose:
    Stream through a raw r/sysadmin comment JSON(L) file and print basic
    inspection stats: line count, malformed-line count, date coverage,
    field presence, and a sample record. Does NOT filter or write anything.
    Run this first, on its own, before filtering.

Usage:
    python3 -u inspect_sysadmin_comments.py --raw-json r_sysadmin_comments.jsonl

    (-u = unbuffered stdout, so you see progress prints immediately
     instead of them being held back by Python's output buffering)
"""

import argparse
import json
import sys
from datetime import datetime, timezone

PROGRESS_EVERY = 100_000  # print a progress line every N records


def inspect(raw_json_path: str):
    total_lines = 0
    malformed_lines = 0
    min_ts, max_ts = None, None
    sample_record = None
    field_counts = {}

    print(f"Opening {raw_json_path} ...", flush=True)

    with open(raw_json_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_lines += 1

            if total_lines % PROGRESS_EVERY == 0:
                print(f"  ...{total_lines:,} lines read so far", flush=True)

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                malformed_lines += 1
                continue

            if sample_record is None:
                sample_record = obj

            for k in obj.keys():
                field_counts[k] = field_counts.get(k, 0) + 1

            ts = obj.get("created_utc")
            if ts is not None:
                try:
                    ts = int(ts)
                    if min_ts is None or ts < min_ts:
                        min_ts = ts
                    if max_ts is None or ts > max_ts:
                        max_ts = ts
                except (ValueError, TypeError):
                    pass

    print("\n=== Inspection summary ===", flush=True)
    print(f"File: {raw_json_path}")
    print(f"Total lines read: {total_lines:,}")
    print(f"Malformed/unparseable lines: {malformed_lines:,}")

    if min_ts and max_ts:
        print(
            f"Date coverage: "
            f"{datetime.fromtimestamp(min_ts, tz=timezone.utc).strftime('%Y-%m-%d')} "
            f"to "
            f"{datetime.fromtimestamp(max_ts, tz=timezone.utc).strftime('%Y-%m-%d')}"
        )
    else:
        print("Date coverage: could not determine (no created_utc values found)")

    if total_lines > 0:
        print(f"\nField presence (out of {total_lines:,} records):")
        for k, v in sorted(field_counts.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v:,} ({v/total_lines:.1%})")

    if sample_record:
        print("\nSample record:")
        print(json.dumps(sample_record, indent=2)[:2000])

    if total_lines == 0:
        print("\nWARNING: no lines were read. Check the file path and that it's not empty/gzipped.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-json", required=True, help="Path to raw sysadmin comment JSONL")
    args = parser.parse_args()
    inspect(args.raw_json)


if __name__ == "__main__":
    main()


# python3 -u inspect_sysadmin_comments.py --raw-json r_sysadmin_comments2021Q10.jsonl