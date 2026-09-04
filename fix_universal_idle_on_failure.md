# Fix: idle-on-failure must be universal, not applied per-site

## What the last smoke test proved

Bug 3's fix idles on termination-call failure specifically. It does NOT
cover the original, more common case: the pipeline itself fails. Direct
evidence from the last real run — pipeline crashes, logs
`"NOT terminating. Pod stays alive for inspection"`, then the script still
exits, and 23 seconds later the whole pipeline restarts from scratch.
That log message describes intent, not actual behavior. RunPod restarts
on the exit itself, regardless of what got printed first.

**Root cause of the gap:** Bug 3 was fixed at its one specific failure
site (the `terminate_pod()` call). Every *other* failure path in this
script — a pipeline stage exiting non-zero, a pipeline timeout (Bug 1),
`verify_success()` returning False, a missing required env var — still
exits normally after logging. Each of those is a separate, independent
opportunity for the same restart-loop bug, and fixing them one at a time
as they're discovered (the way this has gone so far) means there's no
guarantee the next one won't be missed the same way this one was.

## The fix: one choke point, not scattered fixes

Refactor so every failure path funnels through a single function. Don't
call `sys.exit()` or let an exception propagate from ANY failure branch —
route it here instead:

```python
def _fail_and_idle(message: str) -> None:
    """
    The only place any failure path in this script should end up. Never
    sys.exit() or let an exception propagate from a failure branch directly
    -- call this instead. Exists because idling was previously applied at
    one failure site only (termination-call failure), and every other
    failure site still exited normally, which was directly observed to
    trigger a full paid pipeline re-run via RunPod's restart behavior --
    repeatedly, not just once. This is now the universal rule for every
    failure, enforced structurally rather than remembered per-site.
    """
    print(
        f"[run_and_terminate] FAILURE: {message} -- idling instead of "
        f"exiting. Any exit here has been directly observed to trigger a "
        f"full pipeline re-run rather than a clean stop. Stop this pod "
        f"manually once you've finished inspecting.",
        file=sys.stderr,
    )
    _idle_forever()  # reuse Bug 3's existing, already-tested idle loop
```

**Then find and fix every existing failure path** — this needs to be
exhaustive, not just the ones observed so far. At minimum:

1. Missing `RAG_SCRATCH_DIR` (the very first check in `main()`).
2. A pipeline stage exiting non-zero (the one that just failed in
   practice).
3. Pipeline timeout from Bug 1 (`run_pipeline()` returning `None`).
4. `verify_success()` returning False (exit 0 but missing/empty output).
5. Termination call failure (Bug 3's existing branch — refactor it to call
   the same shared `_fail_and_idle()` rather than its own separate inline
   idle loop, so there's one implementation, not two that could drift
   apart later).

**Do a real audit, not a guess at completeness.** Search the whole file
for every `sys.exit(`, every early `return` from `main()`, and every
`raise` that isn't already caught — for each one, either confirm it's on
the genuine success path (pipeline succeeded AND termination succeeded,
where exiting is correct because the pod is already being deleted by
RunPod's own systems) or route it through `_fail_and_idle()`. State in the
PR that this was done as an actual line-by-line audit, not inferred from
the cases already known about.

## Testing (no GPU/RunPod needed)

For each of the 5 failure paths above, individually: simulate the
triggering condition, confirm `_fail_and_idle()` (and therefore
`_idle_forever()` / `time.sleep`) is reached, and confirm the process does
NOT exit — same mocking pattern as Bug 3's existing tests. Also confirm
the genuine success path is unaffected: pipeline succeeds, termination
succeeds, process exits normally, `_fail_and_idle` never called.

## What's explicitly NOT this task

The `cudaErrorDevicesUnavailable` error itself isn't being chased here.
Working hypothesis: it may be a symptom of the restart loop (a killed
process leaving a dirty CUDA context that the next attempt then collides
with), not an independent bug — worth re-checking only after this fix
lands and a genuinely fresh, non-restarted pod either does or doesn't hit
it again. Don't spend time on it in this PR.
