# PR #9 follow-up — marker file for two-file cache consistency

Add as a new commit on the existing `cache-integrity-fix` branch. Do not
open a new PR — this corrects work already in PR #9, and the PR
description should end up accurate for the final state, not require
reading a second PR to understand what actually happened.

## Why the current fix is incomplete

The metadata-before-index reordering doesn't close the crash-window problem
— it relocates which file is stale when a crash lands between the two
`os.replace()` calls. Old order left (new index, old metadata) on crash;
new order leaves (new metadata, old index) instead. Both are a mismatched
pair. Whether that mismatch is actually harmless depends on whether metadata
is deterministic across rebuilds of the same corpus (same chunking, same
order, every time) — if so, old and new metadata are identical and the
mismatch never bites in practice. **Confirm whether that determinism
actually holds for this codebase** before relying on it implicitly; if it's
true, say so explicitly in the PR with the reasoning, rather than let the
reordering imply a guarantee it doesn't independently provide.

## The actual fix — marker file

1. Keep the existing `_atomic_write_index`/`_atomic_write_pickle` functions
   from Task 2 — they're still correct for each file individually. Order
   between them no longer matters once step 3 exists.
2. After both files are successfully written and renamed to their final
   paths, atomically create a small marker file — e.g. `.build_complete` in
   the same cache directory — using the same temp-file-then-`os.replace()`
   pattern.
3. Change the cache-validity check in `data/build_index.py` (wherever it
   currently checks for the index file's existence) to check for the
   **marker file's** existence instead. The marker existing is now the one
   fact that guarantees both the index and metadata are present and belong
   to the same completed build — there's no window where a crash can leave
   something that looks valid but isn't, because the marker is the last
   thing written, after everything else has already succeeded.

```python
def _mark_build_complete(cache_dir):
    marker_path = os.path.join(cache_dir, ".build_complete")
    fd, tmp_path = tempfile.mkstemp(dir=cache_dir, prefix=".tmp_", suffix=".marker")
    os.close(fd)
    os.replace(tmp_path, marker_path)

def _is_build_complete(cache_dir):
    return os.path.exists(os.path.join(cache_dir, ".build_complete"))
```

If `force_rebuild=True` or the marker is missing, treat the cache as invalid
regardless of whether the index/metadata files happen to exist — a
leftover index or metadata file from an interrupted build is not evidence
of anything, only the marker is.

## Update PR #9's description

Correct the claim about what the reordering achieves — replace "closes the
misalignment window" with an accurate description: the reordering is now a
minor, non-load-bearing detail (or can be reverted to whichever order is
simpler, given it's no longer doing the actual work), and the marker file
is what provides the real guarantee. The PR's final state should be
readable on its own and not require a second PR to understand correctly.

## Verification

Repeat Task 4's interruption test from the original brief, but this time
kill the process at three points instead of one: mid-index-write,
mid-metadata-write, and between metadata/index completing but before the
marker gets created. All three should result in **no marker file present**,
meaning the next run correctly treats the cache as invalid and rebuilds —
regardless of what partial state the index/metadata files are in. Confirm
a normal successful build still produces a working marker and is loaded
correctly on the next run without an unnecessary rebuild.
