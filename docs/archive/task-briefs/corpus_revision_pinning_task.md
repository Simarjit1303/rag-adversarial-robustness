# Corpus revision pinning — close the gap next to model pinning

New branch `corpus-revision-pinning`, stacked on `cache-integrity-fix`.
Same standing rule: open the PR, don't merge to main.

## Why

`config.py`'s `MODELS` dict pins every model to an exact commit hash, with
an explicit comment: "every phase must load byte-identical weights."
`CORPORA` has no equivalent — `load_dataset()` calls aren't revision-pinned,
only the sampling seed is. The cache-integrity audit on PR #9 surfaced this
directly: the "does old cached metadata still match a freshly regenerated
one" question turned out to be conditional on the upstream HF dataset
snapshot not changing, which isn't guaranteed. This closes that gap the
same way model pinning was already closed — same pattern, same rigor.

Scope: `nq_open`, `hotpot_qa`, `ms_marco` — the three corpora Phase 1's
`build_index.py` actually touches. The adversarial/utility datasets
(JBB-Behaviors, HarmBench, XSTest) aren't in scope yet since Phase 3 hasn't
started; pin those when that work begins, using the identical pattern, not
now.

## Task 1 — get the real commit hashes

Same method already used for `MODELS`' revisions (per `config.py`'s own
comment: "the hash from each repo's 'Files and versions' tab / API `sha`
field"). Use `huggingface_hub`:

```python
from huggingface_hub import HfApi
api = HfApi()
for repo_id in ["google-research-datasets/nq_open", "hotpotqa/hotpot_qa", "microsoft/ms_marco"]:
    info = api.dataset_info(repo_id)
    print(repo_id, info.sha)
```

Record the actual hashes returned — don't guess or reuse a hash from
memory.

## Task 2 — add `revision` to `CORPORA`, thread it through

```python
CORPORA = {
    "nq_open": {
        "hf_id": "google-research-datasets/nq_open",
        "revision": "<actual sha from Task 1>",
        "dev_n": 1000,
        "eval_n": 10000,
    },
    # same for hotpot_qa, ms_marco
}
```

Wherever `data/loader.py` (or wherever `load_dataset()` is actually called)
builds the corpus, pass `revision=cfg["revision"]` through, same shape as
`harness/model_loader.py` already does for models.

## Task 3 — compose with the PR #9 marker, don't just coexist with it

This is the part that makes the pin actually mean something over time. The
`.build_complete` marker from PR #9 currently just exists or doesn't. Change
it to carry the revision hash it was built with:

```python
def _mark_build_complete(cache_dir, revision):
    marker_path = os.path.join(cache_dir, ".build_complete")
    fd, tmp_path = tempfile.mkstemp(dir=cache_dir, prefix=".tmp_", suffix=".marker")
    with os.fdopen(fd, "w") as f:
        f.write(revision)
    os.replace(tmp_path, marker_path)

def _is_build_complete(cache_dir, expected_revision):
    marker_path = os.path.join(cache_dir, ".build_complete")
    if not os.path.exists(marker_path):
        return False
    with open(marker_path) as f:
        return f.read().strip() == expected_revision
```

If `config.py`'s pinned revision ever changes, every cache built under the
old revision fails this check automatically and rebuilds — no manual
`force_rebuild=True` required, no silent staleness. This is a small change
on top of what already exists, not a new mechanism; keep the rest of PR #9's
per-(corpus, split) marker logic and pre-rebuild marker deletion as-is.

## Verification

1. Confirm Task 1's hashes are real, current, actually fetched — not
   placeholders.
2. Build a cache normally, confirm the marker now contains the revision
   string, confirm a second run is served from cache (no rebuild).
3. Manually change the pinned revision in `config.py` to a different valid
   value, run again, confirm it correctly detects the mismatch and rebuilds
   — this is the actual point of Task 3, test it directly rather than
   assume the logic is correct from reading it.
4. Confirm normal operation (matching revision, valid cache) is unaffected
   — same behavior as before this PR, just with the added safety.
