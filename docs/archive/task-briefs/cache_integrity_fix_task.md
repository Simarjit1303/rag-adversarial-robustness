# Cache-integrity fix — atomic writes for index/metadata files

## Why this matters now, specifically

PR #8's A2 audit was correct that the current I/O (whole-file JSONL/pickle/
faiss writes) has no locking or partial-write handling — and flagged that as
pre-existing, not mount-specific. That's true, but Part B's
`--replica-retry-limit 0` changes the stakes: with no automatic retry, an
interrupted build (crash, OOM, manual stop, replica timeout) leaves whatever
partial file was mid-write sitting there permanently, and the cache check in
`data/build_index.py` trusts `exists()`, not completeness. The next trigger
will find that file and treat it as valid.

**What actually happens on a truncated file is not yet known — test it, don't
assume it.** It could be:
- A hard crash on load (annoying — forces someone to notice and manually
  diagnose — but safe, nothing wrong gets used).
- A silent partial/corrupted load that produces wrong retrieval results
  without erroring (dangerous — a full sweep could run to completion on bad
  data with no signal anything's wrong).

Task 1 below determines which one this codebase actually has, before
deciding how urgently this needs to ship.

---

## Task 1 — reproduce the failure mode locally, first

Before writing the fix, confirm what's actually broken:

1. Run a normal index build for one corpus, let it complete successfully.
   Confirm it loads and works.
2. Simulate an interrupted write: either (a) kill the build process with
   `SIGKILL` partway through a fresh `faiss.write_index()` call (add a
   `time.sleep()` right before it to make the timing easy to hit), or
   (b) manually truncate a completed index file to ~50% of its size with a
   script, to approximate what a partial write would leave behind.
3. Run the pipeline again without `force_rebuild=True`. Observe: does it
   crash on load, or does it silently proceed with something? Document
   whichever happens — this determines the actual severity, not a guess.

## Task 2 — atomic write for the FAISS index + pickle metadata

In `data/build_index.py`, wherever `faiss.write_index(index, path)` and the
metadata `pickle.dump(...)` calls happen:

- Write to a temp file **in the same directory** as the final destination,
  not `/tmp` or any other filesystem — `os.replace()` is only atomic within
  a single filesystem; across filesystems it silently falls back to
  copy+delete, which reintroduces the exact problem this is meant to fix.
- Only call `os.replace(tmp_path, final_path)` after the write function
  returns successfully.
- If anything raises during the write, the temp file is left behind
  (harmless — it's not at the path anything checks) and the final path
  either doesn't exist yet or still holds the last known-good version.

```python
import os
import tempfile

def atomic_write_index(index, final_path):
    final_path = str(final_path)
    dir_name = os.path.dirname(final_path)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, prefix=".tmp_", suffix=".faiss")
    os.close(fd)
    try:
        faiss.write_index(index, tmp_path)
        os.replace(tmp_path, final_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
```

Same pattern for the pickle metadata write — temp file in the same
directory, `os.replace()` only after `pickle.dump()` completes and the file
handle is closed.

## Task 3 — apply the same pattern to JSONL results, at the same time

Lower urgency (a truncated JSONL is self-evident on load — short row count,
not silently wrong), but cheap to fix consistently while already in this
code, and worth doing for the same underlying discipline. Same
temp-file-then-`os.replace()` approach in `evaluation/run_baseline.py`
wherever results get written.

## Task 4 — re-run Task 1's interruption test, post-fix

Repeat the same kill/truncate test from Task 1. Confirm now: the final path
either has the complete prior version (if one existed) or doesn't exist at
all — never a partial file at the path the cache check looks at. Confirm
normal successful builds are unaffected (same output, same load behavior)
before and after the change.

---

## What stays open until Part B actually exists — don't close this out yet

`os.replace()`'s atomicity guarantee is a POSIX/local-filesystem guarantee.
**It has not been confirmed to hold the same way over an Azure Files SMB
mount** — SMB's rename semantics can differ from local ext4/similar in ways
that matter here. Once Part B is done and the share is actually mounted,
repeat Task 1 and Task 4's tests directly against a path under
`/mnt/rag-scratch` rather than local disk, and confirm atomic rename still
behaves as expected there. Do not assume it transfers just because it
passed locally — that's the whole reason this was sequenced local-first
rather than skipped.

---

## Verification summary before calling this done

1. Task 1: documented, with evidence, whether the pre-fix failure mode is a
   hard crash or a silent bad load.
2. Task 2 + 3: atomic write pattern applied to index, metadata, and results
   files.
3. Task 4: interruption test re-run post-fix, confirms no partial file ever
   sits at a path the cache check trusts.
4. Explicitly flagged as NOT yet verified: same behavior over the Azure
   Files mount — that's a Part-B-dependent follow-up, not something to fake
   confidence about now.
