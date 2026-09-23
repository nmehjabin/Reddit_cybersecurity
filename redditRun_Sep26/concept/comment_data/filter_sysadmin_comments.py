"""
filter_sysadmin_comments.py

Purpose:
    Filter a raw r/sysadmin comment JSON(L) file down to top-level
    comments (parent_id == link_id) that belong to posts in your target
    posts corpus, and write the matches to a CSV.

    Run inspect_sysadmin_comments.py first to sanity-check the raw file's
    field names and date coverage before running this.

    Accepts either a plain .jsonl/.json file, or a .zst-compressed ndjson
    file (e.g. sysadmin_comments.zst from the Watchful1 Academic Torrents
    dump) -- the .zst is streamed and decompressed on the fly, never
    written to disk in full.

Usage:
    python3 -u filter_sysadmin_comments.py \
        --raw-json r_sysadmin_comments.jsonl \
        --posts-file bat_score_pos.csv \
        --out-csv sysadmin_comments_18_filtered.csv \
        --post-id-col post_id \
        --subreddit-filter sysadmin

    python3 -u filter_sysadmin_comments.py \
        --raw-json sysadmin_comments.zst \
        --posts-file bat_score_pos.csv \
        --out-csv sysadmin_comments_filtered.csv

    (-u = unbuffered stdout, so progress prints show up immediately)

Requirements for .zst input:
    pip install zstandard --break-system-packages
"""

import argparse
import csv
import io
import json
import sys
from datetime import datetime, timezone

try:
    import zstandard
except ImportError:
    zstandard = None

PROGRESS_EVERY = 100_000  # print a progress line every N raw records scanned

# Pushshift zst dumps use a very large window size; the default zstandard
# decompressor will refuse to read them without this bumped up.
ZSTD_MAX_WINDOW_SIZE = 2**31


def open_lines(path: str):
    """
    Yield decoded text lines from either a plain .jsonl/.json file or a
    .zst-compressed ndjson file (streaming, never fully decompressed to disk).
    """
    if path.endswith(".zst"):
        if zstandard is None:
            sys.exit(
                "ERROR: reading .zst files requires the 'zstandard' package.\n"
                "Install it with: pip install zstandard --break-system-packages"
            )
        with open(path, "rb") as fh:
            dctx = zstandard.ZstdDecompressor(max_window_size=ZSTD_MAX_WINDOW_SIZE)
            with dctx.stream_reader(fh) as reader:
                text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
                for line in text_stream:
                    yield line
    else:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                yield line

COMMENT_FIELD_MAP = {
    "id": "id",
    "link_id": "link_id",
    "parent_id": "parent_id",
    "created_utc": "created_utc",
    "body": "body",
    "author": "author",
    "score": "score",
    "subreddit": "subreddit",
}

OUTPUT_COLUMNS = [
    "comment_id",
    "post_id",
    "created_utc",
    "created_date",
    "author",
    "score",
    "subreddit",
    "body",
]


def load_target_post_ids(posts_file: str, post_id_col: str) -> set:
    target_ids = set()
    print(f"Loading target post_ids from {posts_file} ...", flush=True)
    with open(posts_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if post_id_col not in reader.fieldnames:
            sys.exit(
                f"ERROR: column '{post_id_col}' not found in {posts_file}. "
                f"Available columns: {reader.fieldnames}"
            )
        for row in reader:
            pid = row[post_id_col].strip()
            if pid:
                target_ids.add(pid)
    print(f"Loaded {len(target_ids):,} target post_ids", flush=True)
    return target_ids


def strip_prefix(reddit_id: str) -> str:
    if reddit_id and "_" in reddit_id[:3]:
        return reddit_id.split("_", 1)[1]
    return reddit_id


def filter_comments(raw_json_path, target_post_ids, subreddit_filter, out_csv):
    total_lines = 0
    malformed_lines = 0
    matched_post = 0
    matched_top_level = 0
    written = 0

    print(f"Scanning {raw_json_path} ...", flush=True)

    with open(out_csv, "w", encoding="utf-8", newline="") as outfile:

        writer = csv.DictWriter(outfile, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for line in open_lines(raw_json_path):
            line = line.strip()
            if not line:
                continue
            total_lines += 1

            if total_lines % PROGRESS_EVERY == 0:
                print(
                    f"  ...{total_lines:,} scanned | "
                    f"{matched_post:,} post matches | "
                    f"{matched_top_level:,} top-level | "
                    f"{written:,} written",
                    flush=True,
                )

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                malformed_lines += 1
                continue

            if subreddit_filter:
                sub = obj.get(COMMENT_FIELD_MAP["subreddit"], "")
                if sub.lower() != subreddit_filter.lower():
                    continue

            link_id_raw = obj.get(COMMENT_FIELD_MAP["link_id"], "")
            parent_id_raw = obj.get(COMMENT_FIELD_MAP["parent_id"], "")
            post_id = strip_prefix(link_id_raw)

            if post_id not in target_post_ids:
                continue
            matched_post += 1

            is_top_level = link_id_raw == parent_id_raw
            if not is_top_level:
                continue
            matched_top_level += 1

            created_utc = obj.get(COMMENT_FIELD_MAP["created_utc"])
            created_date = (
                datetime.fromtimestamp(int(created_utc), tz=timezone.utc).strftime("%Y-%m-%d")
                if created_utc is not None else ""
            )

            writer.writerow({
                "comment_id": strip_prefix(obj.get(COMMENT_FIELD_MAP["id"], "")),
                "post_id": post_id,
                "created_utc": created_utc,
                "created_date": created_date,
                "author": obj.get(COMMENT_FIELD_MAP["author"], ""),
                "score": obj.get(COMMENT_FIELD_MAP["score"], ""),
                "subreddit": obj.get(COMMENT_FIELD_MAP["subreddit"], ""),
                "body": obj.get(COMMENT_FIELD_MAP["body"], ""),
            })
            written += 1

    print("\n=== Filtering results ===", flush=True)
    print(f"Total raw lines scanned: {total_lines:,}")
    print(f"Malformed/unparseable lines: {malformed_lines:,}")
    print(f"Comments matching target post_ids: {matched_post:,}")
    print(f"Of those, top-level comments: {matched_top_level:,}")
    print(f"Rows written to CSV: {written:,}")
    print(f"Output: {out_csv}")

    if total_lines == 0:
        print("\nWARNING: no lines were read from the raw file. Check the path.")
    elif matched_post == 0:
        print(
            "\nWARNING: 0 comments matched any target post_id. Likely causes:\n"
            "  - post_id prefix mismatch (raw file's link_id carries 't3_', "
            "posts file doesn't, or vice versa)\n"
            "  - wrong --post-id-col\n"
            "  - subreddit filter excluding everything (check --subreddit-filter)\n"
            "Run inspect_sysadmin_comments.py and check a sample record's "
            "link_id format against your posts file's post_id format."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-json", required=True, help="Path to raw sysadmin comment JSONL")
    parser.add_argument("--posts-file", required=True, help="Path to posts corpus CSV with target post_ids")
    parser.add_argument("--out-csv", required=True, help="Path to write filtered top-level comments CSV")
    parser.add_argument("--post-id-col", default="post_id", help="Column name for post_id in posts file (default: post_id)")
    parser.add_argument("--subreddit-filter", default="sysadmin", help="Only keep comments from this subreddit (default: sysadmin). Pass '' to disable.")
    args = parser.parse_args()

    target_post_ids = load_target_post_ids(args.posts_file, args.post_id_col)
    filter_comments(
        raw_json_path=args.raw_json,
        target_post_ids=target_post_ids,
        subreddit_filter=args.subreddit_filter,
        out_csv=args.out_csv,
    )


if __name__ == "__main__":
    main()
