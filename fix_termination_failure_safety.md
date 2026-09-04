# Fix: a failed termination call must never crash the script

## Root cause, for context (don't re-litigate, just fix it)

Confirmed by direct observation, twice, on real RunPod pods: pipeline
succeeds fully (results written correctly to the Network Volume) →
`terminate_pod()`'s REST call fails (403, credential issue still being
diagnosed separately, not this task's concern) → `raise_for_status()`
throws an unhandled `HTTPError` → the whole script crashes → something in
RunPod's pod orchestration reads the crash-exit as "this needs retrying" →
the entire pipeline (model load, embedding, full sweep) runs again from
scratch. Not confirmed whether RunPod restarts on any exit or only
non-zero ones — build the fix assuming the worse case (any exit) rather
than betting on the better one.

This is worse than Bug 1's hang: a hang bills at a steady rate. This
compounds — full paid re-runs, repeatedly, for as long as the credential
issue persists.

## The fix

Wrap the `terminate_pod()` call site. On success, nothing changes — the
pod is being deleted by RunPod's own systems regardless of what the script
does after. On failure, **do not exit the process at all** — log clearly,
then idle:

```python
def main():
    # ... existing pipeline run + verify_success() logic, unchanged ...

    try:
        terminate_pod(pod_id, api_key)
        print("[run_and_terminate] pod terminated successfully.")
    except Exception as e:
        print(
            f"[run_and_terminate] PIPELINE SUCCEEDED, results are safe on "
            f"the Network Volume. Termination call FAILED: {e!r}. "
            f"This pod may still be billing -- check manually and stop it "
            f"if so. Idling instead of exiting, since an exit here has "
            f"already been observed to trigger a full pipeline re-run "
            f"rather than a clean stop.",
            file=sys.stderr,
        )
        while True:
            time.sleep(3600)
```

The specific log message matters — it needs to be unmistakable in a log
stream, since this branch is the only thing standing between "billing
continues but is at least visible and bounded to the same idle pod" and
another expensive re-run loop.

## Testing (no GPU/RunPod needed)

- Mock `requests.delete` to raise/return a non-2xx status, same pattern as
  the existing failure-path tests. Confirm: no unhandled exception
  propagates out of `main()`, the process does not exit, and the correct
  log message is printed.
- Confirm the success path is unchanged — mock a 204, confirm no idle loop
  triggers, confirm the existing "terminated successfully" log line still
  prints.
- Confirm this doesn't interact badly with Bug 1's pipeline-timeout logic
  — the timeout wraps the pipeline subprocess itself; this wraps a later,
  separate step in `main()`, after `verify_success()`. They shouldn't
  overlap, but state explicitly in the PR that this was checked, not
  assumed.
- For the idle loop specifically: don't actually sleep for real hours in
  a test. Mock `time.sleep` and assert it was called, or refactor so the
  loop condition is testable without a real wait.

## One thing to flag in the PR, not fix here

This idle-on-failure branch has no timeout of its own — unlike Bug 1's
hang, which resolves on its own once the fix lands. That's a deliberate,
asymmetric choice: this branch only triggers *after* a confirmed
successful run, so the actual work and results are never at risk, only
additional idle billing time until a human notices the log message and
stops the pod manually. State this tradeoff plainly in the PR rather than
silently leaving it — if a bounded idle timeout is wanted here too later,
that's a deliberate follow-up decision, not something to add unasked.

## Separate from this task, for the student to handle directly (not code)

The actual 403 root cause — `RUNPOD_API_KEY` working from a local machine
but failing from inside the pod — is still unresolved and isn't something
this fix touches. Recommend re-verifying the Secret's exact value once
more before the next paid attempt, ideally by checking it from inside a
running pod's web terminal this time (`echo ${#RUNPOD_API_KEY}` and a
truncated prefix), rather than assuming a recreation fixed it without
confirming.
