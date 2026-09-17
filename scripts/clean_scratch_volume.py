"""
Dry-run-first cleanup for stray `.tmp_*` files on the `rag-scratch` RunPod
network volume's results/ directory.

The atomic-write pattern used everywhere results are written (see
run_baseline.py's `_atomic_open`) writes to a `.tmp_*` name first and
os.replace()s it onto the final name only once the write completes without
raising. That means a `.tmp_*` file is BY DEFINITION an incomplete/abandoned
write (a crashed run, a killed pod mid-write) -- never a live result under
any naming convention, current or future. So the safety property this script
relies on is an INCLUSION filter: only names starting with `.tmp_` are ever
candidates in the first place. The NEVER_DELETE_PATTERNS list below is a
belt-and-suspenders check on top of that, not the primary safety mechanism --
new result-file prefixes will keep appearing as more attacks/phases get
built, and this script must not need updating every time one does.

NEVER deletes anything by default. Every invocation without --confirm-delete
lists candidates only. This script is meant to be run by hand, on demand --
see docs/archive/task-briefs/cleanup_tooling_task.md's "What NOT to build": no cron, no automatic
invocation on pod start, ever.

Talks to the volume via RunPod's S3-compatible API -- a Network Volume is
reachable as an S3 bucket (keyed by volume ID) without a pod attached; see
https://docs.runpod.io/pods/storage/s3-api. Needs `pip install boto3`
(deliberately NOT added to requirements.txt -- that file is the Docker
image's ML dependency set, and this is a local ops tool that never runs
inside the container) plus:

    RUNPOD_S3_ENDPOINT_URL   e.g. https://s3api-eu-ro-1.runpod.io
    RUNPOD_S3_BUCKET         the network volume ID (the "bucket" name)
    AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY   a RunPod S3 API key pair
                             (create one in the RunPod console; boto3 picks
                             these up from the environment or ~/.aws/credentials
                             automatically -- nothing else to configure)

Usage:
    python -m scripts.clean_scratch_volume                     # dry run, 24h+
    python -m scripts.clean_scratch_volume --min-age-hours 48   # dry run, 48h+
    python -m scripts.clean_scratch_volume --confirm-delete     # actually delete
"""

import argparse
import fnmatch
import os
import sys
from datetime import datetime, timezone

# Redundant safety net, not the primary filter (see module docstring) -- any
# real result/doc naming convention that exists now or gets added later.
NEVER_DELETE_PATTERNS = [
    "attack_raw_*", "attack_summary_*",
    "baseline_raw_*", "baseline_summary_*",
    "phase1_results_complete/*", "phase1_results_partial/*",
    "PHASE2_PREP_LOG.md",
    "*.md", "*.log",
]

# Confirmed 2026-09-06 against the real cplemvuitj bucket: objects are
# nested under a "rag-scratch/" folder (the volume's own name), not at the
# bucket root -- "results/" alone silently found nothing.
DEFAULT_PREFIX = "rag-scratch/results/"


def _is_never_delete(key: str) -> bool:
    name = key.rsplit("/", 1)[-1]
    return any(
        fnmatch.fnmatch(key, pattern) or fnmatch.fnmatch(name, pattern)
        for pattern in NEVER_DELETE_PATTERNS
    )


def find_candidates(s3, bucket: str, prefix: str, min_age_hours: float):
    """
    Yield (key, size_bytes, age_hours) for every `.tmp_*` object under
    prefix at least min_age_hours old. Paginates -- a multi-cell sweep's
    results/ dir can hold hundreds of files.
    """
    now = datetime.now(timezone.utc)
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            name = key.rsplit("/", 1)[-1]
            if not name.startswith(".tmp_"):
                continue  # inclusion filter -- see module docstring
            if _is_never_delete(key):
                # Should be unreachable given the .tmp_ prefix check above
                # (e.g. a name like ".tmp_notes.md" would trip this) -- if it
                # ever fires, that's a bug worth surfacing loudly, not a
                # silent skip-and-move-on.
                print(f"  [SAFETY] {key} matched a .tmp_* prefix AND a "
                      f"never-delete pattern -- skipping, please investigate.")
                continue
            age_hours = (now - obj["LastModified"]).total_seconds() / 3600
            if age_hours >= min_age_hours:
                yield key, obj["Size"], age_hours


def _human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bucket", default=os.environ.get("RUNPOD_S3_BUCKET"),
        help="network volume ID / S3 bucket name (default: RUNPOD_S3_BUCKET env var)")
    parser.add_argument(
        "--endpoint-url", default=os.environ.get("RUNPOD_S3_ENDPOINT_URL"),
        help="RunPod S3 API endpoint (default: RUNPOD_S3_ENDPOINT_URL env var)")
    parser.add_argument(
        "--prefix", default=DEFAULT_PREFIX,
        help=f"key prefix to scan (default: {DEFAULT_PREFIX!r})")
    parser.add_argument(
        "--min-age-hours", type=float, default=24.0,
        help="only list/delete .tmp_* files at least this old (default: 24)")
    parser.add_argument(
        "--confirm-delete", action="store_true",
        help="actually delete candidates (default: dry run, lists only)")
    args = parser.parse_args(argv)

    if not args.bucket or not args.endpoint_url:
        parser.error(
            "--bucket/--endpoint-url (or RUNPOD_S3_BUCKET/RUNPOD_S3_ENDPOINT_URL "
            "env vars) are required -- this script refuses to guess which "
            "volume to touch."
        )

    import boto3  # imported here, not at module top -- not a project dependency
    s3 = boto3.client("s3", endpoint_url=args.endpoint_url)

    candidates = list(find_candidates(s3, args.bucket, args.prefix, args.min_age_hours))

    if not candidates:
        print(f"No .tmp_* files >= {args.min_age_hours}h old under "
              f"{args.bucket}/{args.prefix}. Nothing to do.")
        return 0

    print(f"{'MODE: DELETING' if args.confirm_delete else 'DRY RUN (no files touched)'} "
          f"-- {len(candidates)} candidate(s) under {args.bucket}/{args.prefix}:\n")
    for key, size, age_hours in candidates:
        print(f"  {key:<70} {_human_size(size):>8}  {age_hours:>7.1f}h old")

    if not args.confirm_delete:
        print("\nDry run only -- nothing deleted. Re-run with --confirm-delete to delete these.")
        return 0

    deleted = []
    for key, _, _ in candidates:
        s3.delete_object(Bucket=args.bucket, Key=key)
        deleted.append(key)

    print(f"\nDeleted {len(deleted)} file(s):")
    for key in deleted:
        print(f"  {key}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
