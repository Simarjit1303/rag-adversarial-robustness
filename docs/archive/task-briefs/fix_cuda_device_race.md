# Fix: intermittent CUDA device-busy race between model load and embedder load

## What changed in understanding

Previously treated as a possible restart-loop symptom. That's now disproven
directly: a genuinely fresh pod (no restart history) still hit
`cudaErrorDevicesUnavailable` on its very first attempt, after two prior
clean runs on the identical code path. Two clean, one failed, same
sequence each time — that's the signature of a real, intermittent race
condition, not leftover state from a killed process.

**Likely mechanism:** `phi-4-mini` loads via `device_map='auto'`
(accelerate). The progress bar reaching 100% doesn't guarantee every
underlying CUDA operation accelerate kicked off has actually finished
settling on the device. `sentence-transformers` then immediately tries to
claim the same GPU for a completely separate model, in the same process.
Sometimes there's enough of a gap for this to work cleanly; sometimes
there isn't.

## Fix 1 — close the race directly

In `harness/model_loader.py` (or wherever `phi-4-mini`'s load call
returns), add an explicit CUDA sync before control passes to anything that
loads a second model on the same device:

```python
import torch
# ... after the model finishes loading via device_map='auto' ...
if torch.cuda.is_available():
    torch.cuda.synchronize()
```

This forces all pending CUDA work from the first model's load to actually
complete before the embedder gets a chance to claim the device — removes
the timing gap rather than working around its symptom.

## Fix 2 — retry-with-backoff as a safety net

Given this has now been directly observed as intermittent (not
deterministic), a synchronize call reduces the race window but a defensive
retry is cheap insurance in case the window isn't fully closed. In
`data/build_index.py`'s `_get_embedder()`:

```python
import time

def _get_embedder():
    global _embedder
    if _embedder is not None:
        return _embedder
    last_exc = None
    for attempt in range(3):
        try:
            _embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)
            return _embedder
        except Exception as e:
            last_exc = e
            print(f"[index] embedder init attempt {attempt + 1} failed: {e!r} "
                  f"-- retrying" if attempt < 2 else " -- giving up",
                  file=sys.stderr)
            time.sleep(2)
    raise last_exc
```

Same pattern already used for pip's network retries earlier in this
project — transient failures get a bounded number of retries with a short
backoff, not treated as immediately fatal.

## Testing (no GPU needed for the logic, GPU needed to confirm the actual fix)

- `torch.cuda.synchronize()` is a no-op on CPU/no-GPU environments — safe
  to call unconditionally behind the `torch.cuda.is_available()` check,
  confirm this doesn't break anything on the Windows/CPU-only dev machine.
- Mock `SentenceTransformer` to fail twice then succeed on the third
  attempt — confirm the retry loop actually retries the right number of
  times and eventually returns the successful instance, not silently
  swallowing all three failures.
- Mock it to fail all three times — confirm the original exception
  propagates (not swallowed), so `_fail_and_idle()` still catches it
  correctly upstream.

## What this doesn't resolve yet

The `RUNPOD_TERMINATE_KEY` fix from the previous PR has still never been
exercised end to end — this run's pipeline failed before ever reaching the
termination step. Once this CUDA race is closed, the next clean run
naturally re-tests that fix too; no separate action needed for it.
