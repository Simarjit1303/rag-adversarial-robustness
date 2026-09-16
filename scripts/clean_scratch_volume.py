import argparse
import fnmatch
import os
import sys
from datetime import datetime, timezone
NEVER_DELETE_PATTERNS = ['attack_raw_*', 'attack_summary_*', 'baseline_raw_*', 'baseline_summary_*', 'phase1_results_complete/*', 'phase1_results_partial/*', 'PHASE2_PREP_LOG.md', '*.md', '*.log']
DEFAULT_PREFIX = 'rag-scratch/results/'

def _is_never_delete(key: str) -> bool:
    name = key.rsplit('/', 1)[-1]
    return any((fnmatch.fnmatch(key, pattern) or fnmatch.fnmatch(name, pattern) for pattern in NEVER_DELETE_PATTERNS))

def find_candidates(s3, bucket: str, prefix: str, min_age_hours: float):
    now = datetime.now(timezone.utc)
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            name = key.rsplit('/', 1)[-1]
            if not name.startswith('.tmp_'):
                continue
            if _is_never_delete(key):
                print(f'  [SAFETY] {key} matched a .tmp_* prefix AND a never-delete pattern -- skipping, please investigate.')
                continue
            age_hours = (now - obj['LastModified']).total_seconds() / 3600
            if age_hours >= min_age_hours:
                yield (key, obj['Size'], age_hours)

def _human_size(n: float) -> str:
    for unit in ('B', 'KB', 'MB', 'GB'):
        if n < 1024:
            return f'{n:.0f}{unit}'
        n /= 1024
    return f'{n:.1f}TB'

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bucket', default=os.environ.get('RUNPOD_S3_BUCKET'), help='network volume ID / S3 bucket name (default: RUNPOD_S3_BUCKET env var)')
    parser.add_argument('--endpoint-url', default=os.environ.get('RUNPOD_S3_ENDPOINT_URL'), help='RunPod S3 API endpoint (default: RUNPOD_S3_ENDPOINT_URL env var)')
    parser.add_argument('--prefix', default=DEFAULT_PREFIX, help=f'key prefix to scan (default: {DEFAULT_PREFIX!r})')
    parser.add_argument('--min-age-hours', type=float, default=24.0, help='only list/delete .tmp_* files at least this old (default: 24)')
    parser.add_argument('--confirm-delete', action='store_true', help='actually delete candidates (default: dry run, lists only)')
    args = parser.parse_args(argv)
    if not args.bucket or not args.endpoint_url:
        parser.error('--bucket/--endpoint-url (or RUNPOD_S3_BUCKET/RUNPOD_S3_ENDPOINT_URL env vars) are required -- this script refuses to guess which volume to touch.')
    import boto3
    s3 = boto3.client('s3', endpoint_url=args.endpoint_url)
    candidates = list(find_candidates(s3, args.bucket, args.prefix, args.min_age_hours))
    if not candidates:
        print(f'No .tmp_* files >= {args.min_age_hours}h old under {args.bucket}/{args.prefix}. Nothing to do.')
        return 0
    print(f"{('MODE: DELETING' if args.confirm_delete else 'DRY RUN (no files touched)')} -- {len(candidates)} candidate(s) under {args.bucket}/{args.prefix}:\n")
    for key, size, age_hours in candidates:
        print(f'  {key:<70} {_human_size(size):>8}  {age_hours:>7.1f}h old')
    if not args.confirm_delete:
        print('\nDry run only -- nothing deleted. Re-run with --confirm-delete to delete these.')
        return 0
    deleted = []
    for key, _, _ in candidates:
        s3.delete_object(Bucket=args.bucket, Key=key)
        deleted.append(key)
    print(f'\nDeleted {len(deleted)} file(s):')
    for key in deleted:
        print(f'  {key}')
    return 0
if __name__ == '__main__':
    sys.exit(main())
