# Two bugs from the real RunPod smoke test

Both surfaced on real infrastructure, not in testing — the smoke test did
its job. Fix both before the next paid run, in any order, ideally same
branch/PR since they're both small and both came from the same test.

Branch off `runpod-self-termination` (still open, unmerged — same standing
rule as every other PR this session).

---

## Bug 1 — a hang never triggers termination, only a clean exit does

**What happened:** `phi-4-mini` finished loading, then the pod sat at 0%
CPU, 0% GPU utilization, 4 MiB VRAM, for 12+ minutes with zero telemetry
movement — no download, no computation, nothing. `run_and_terminate.py`'s
success/failure logic only runs *after* the pipeline subprocess exits.
A genuine hang never exits, so that check never fires. The pod would have
billed indefinitely if not manually stopped — the exact failure mode the
whole self-termination effort exists to prevent, just from a different
angle (a hang instead of a restart loop).

### Fix: bound the subprocess with a real timeout

```python
import subprocess
import signal
import os

PIPELINE_TIMEOUT_SECONDS = int(os.environ.get("RAG_PIPELINE_TIMEOUT_SECONDS", 21600))
# default 6h, generous for a real sweep -- override to something short
# (e.g. 1800 = 30 min) for smoke tests specifically, via the pod's env vars

def run_pipeline():
    proc = subprocess.Popen(
        [...],  # existing pipeline invocation
        start_new_session=True,  # own process group, so a timeout kill
                                   # takes any child processes with it --
                                   # a plain proc.kill() only kills the
                                   # direct child, not grandchildren
    )
    try:
        proc.wait(timeout=PIPELINE_TIMEOUT_SECONDS)
        return proc.returncode
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        print(f"[run_and_terminate] pipeline exceeded {PIPELINE_TIMEOUT_SECONDS}s "
              f"-- killed. NOT terminating pod (same as any other failure), "
              f"but this bounds the hang instead of leaving it unbounded.",
              file=sys.stderr)
        return None  # distinct from a real exit code, so the caller can
                      # log "TIMEOUT" specifically, not just "exit != 0"
```

**Explicit design decision, stated so it's not silently assumed:** on
timeout, still do NOT terminate the pod — same principle as any other
failure, preserve the ability to inspect. This means a timeout still
requires manual intervention to actually stop billing, same as what just
happened. What changes is the hang becomes *bounded* (a known ceiling)
instead of *unbounded* (could run forever unnoticed). Full automatic
cleanup on timeout was considered and rejected here: a hang rarely
produces useful error output the way a crash does, so there's less
inspection value in staying alive than after a normal failure — but
terminating unconditionally on timeout would be inconsistent with the
project's established "never terminate except on verified success" rule
without a much stronger justification than convenience. If you disagree
with this call, say so explicitly rather than silently changing it.

### Testing (no GPU/RunPod needed)

Simulate a hang locally with a stand-in subprocess (e.g.
`python -c "import time; time.sleep(999)"`) and a short timeout (a few
seconds). Confirm:
- It's actually killed within roughly the timeout window, not left running.
- The process group is killed, not just the direct child (test with a
  stand-in that itself spawns a child, confirm the grandchild also dies).
- `terminate_pod`/the REST call is never reached on timeout — same mocking
  approach as the existing failure-path tests.
- A normal, fast-completing subprocess is unaffected by the timeout logic.

---

## Bug 2 — `RAG_CORPORA` doesn't actually filter anything

**What happened:** the pod was launched with `RAG_CORPORA=nq_open`, but the
logs show indices being built for both `nq_open` and `ms_marco` — the
filter had no effect. Find where `RAG_CORPORA` is supposed to be read
(check `data/build_index.py` and `evaluation/run_baseline.py` — PR #2 added
an equivalent `RAG_MODELS` filter, confirm whether `RAG_CORPORA` was
actually wired up the same way or only ever documented/assumed).

### Fix

Whatever the actual gap turns out to be — filter never implemented, wrong
variable name, applied in one file but not the other, string-parsing bug —
fix it so `RAG_CORPORA=nq_open` genuinely restricts the corpus loop to
just `nq_open`, the same way `RAG_MODELS` already restricts models.

### Testing (no GPU needed)

This is pure filtering logic — test it without building real indices.
Mock or stub the corpus-processing loop, set `RAG_CORPORA` to a single
value, confirm only that corpus gets processed and the others are
genuinely skipped (not just skipped in output but never touched at all).
Test the unset case too (no `RAG_CORPORA` set → all corpora process,
confirming this fix doesn't break the default full-sweep behavior).

---

## Verification before calling this done

1. Both bugs fixed, both covered by local tests, all green.
2. Confirm existing tests (from PR #11's REST-termination fix) still pass
   — this branch shouldn't regress anything already working.
3. State plainly in the PR that neither fix has been proven on a real
   RunPod pod yet — that's the next actual smoke test, once this merges
   into the branch, and it should use a SHORT `RAG_PIPELINE_TIMEOUT_SECONDS`
   override specifically so a repeat hang bounds out in minutes, not hours.
