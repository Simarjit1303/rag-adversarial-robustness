# Fix: result filenames collide across models, corpora, and engines

## What was found, and why it's urgent

`baseline_raw.jsonl` and `baseline_summary.csv` are written to the same
fixed path regardless of which model, corpus, or engine produced them.
Confirmed directly via the Network Volume's S3 API: after this session's
final vLLM run, only one `baseline_raw.jsonl` existed, timestamped to
match that run exactly — the prior HF-path run's data was silently
overwritten with no error, no warning, nothing in the logs to indicate
data was lost.

**This is a launch-blocker for the real Phase 1 sweep, not a minor
cleanup item.** The planned sweep covers 4 models × 3 corpora — if every
cell writes to the same filename, only the very last cell to finish
survives. The other eleven would be silently destroyed, while the
pipeline reports success at every step, because nothing currently checks
for this. This needs to be fixed before `RAG_MODELS`/`RAG_CORPORA` are
ever removed to run the full matrix.

## The fix

Wherever `baseline_raw.jsonl` and `baseline_summary.csv` paths are
constructed (likely `evaluation/run_baseline.py`), include the model key,
corpus name, and engine in the filename:

```python
engine = os.environ.get("INFERENCE_ENGINE", "hf")
raw_path = results_dir / f"baseline_raw_{model_key}_{corpus_name}_{engine}.jsonl"
summary_path = results_dir / f"baseline_summary_{model_key}_{corpus_name}_{engine}.csv"
```

Apply the same atomic-write discipline already used elsewhere (temp file
in the same directory, `os.replace()` on completion) — this fix should
not touch or weaken that pattern, only the naming.

**Decide explicitly whether the summary CSV should also have an
aggregating, cross-cell view** (e.g., a single `baseline_summary_all.csv`
that appends one row per completed cell) — useful for actually reading
results across the full matrix without opening twelve separate files.
If added, this aggregate file needs its own append-safe write pattern
(not a plain overwrite) so concurrent or sequential runs don't collide
with each other the same way this bug just did.

## Testing (no GPU needed)

- Run the pipeline's result-writing logic (mocked generation, no real
  model calls) for two different `(model, corpus, engine)` combinations
  in sequence, using the same `RAG_SCRATCH_DIR`. Confirm both sets of
  output files exist afterward, neither overwritten.
- Confirm existing tests referencing the old fixed filenames are updated
  to the new pattern, not just left pointing at strings that no longer
  match reality.

## Verification

Before trusting this fix for the real sweep, confirm on the Network
Volume itself (via the S3 API, same method used to discover this bug)
that two different model/corpus combinations run back to back produce
two distinct sets of files, not one overwriting the other.
