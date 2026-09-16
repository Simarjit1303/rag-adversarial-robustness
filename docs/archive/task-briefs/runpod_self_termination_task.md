# RunPod self-termination — new wrapper script, not a Dockerfile change

## Why this is a new file, not an edit to the existing CMD

The Dockerfile's current `CMD` (run sweep, then `sleep infinity`) was the
correct fix for Azure Container Apps back in PR #4 — it stopped the
platform from restart-looping the container. On RunPod, that same
`sleep infinity` is the opposite of correct: the pod would finish, sleep
forever, and never reach a termination call, billing indefinitely. Same
underlying lesson as the Stage 2 brief's rule about not porting one
platform's fix to another platform reflexively.

So: **do not change the existing Dockerfile `CMD`.** Add a new script.
RunPod's pod template has its own "Start command" override field — set
that to invoke the new script instead, leaving the image's default
behavior (and the Azure path) completely untouched.

## Task 1 — add the `runpod` dependency

Add `runpod` to `requirements.txt`, pinned to a specific version — same
discipline as every other dependency in this project (`pip show runpod`
after installing, or check PyPI for the current stable release before
pinning; don't guess a version number).

## Task 2 — new script: `scripts/run_and_terminate.py`

```python
"""
RunPod entrypoint wrapper. Runs the sweep, then self-terminates the pod
ONLY on confirmed success -- never on failure, so a crashed run stays
alive for log inspection instead of erasing its own evidence.

This is NOT used by the Azure path. It's invoked only via RunPod's
template "Start command" override.
"""
import os
import sys
import subprocess
from pathlib import Path

def verify_success(scratch_dir: str) -> bool:
    """
    Confirms results actually landed, not just that the sweep process
    exited 0. Check the actual expected output files exist and are
    non-empty -- an exit code alone doesn't prove correct output, given
    this project's own history of silent-looking failures elsewhere.
    """
    results_dir = Path(scratch_dir) / "results"
    if not results_dir.exists():
        return False
    # Adjust the glob pattern to match whatever run_baseline.py actually
    # writes (JSONL rows + summary CSV per PR #9's atomic-write pattern) --
    # confirm the real filename pattern in evaluation/run_baseline.py
    # rather than guess it here.
    jsonl_files = list(results_dir.glob("*.jsonl"))
    csv_files = list(results_dir.glob("*.csv"))
    if not jsonl_files or not csv_files:
        return False
    return all(f.stat().st_size > 0 for f in jsonl_files + csv_files)


def main():
    scratch_dir = os.environ.get("RAG_SCRATCH_DIR")
    if not scratch_dir:
        print("RAG_SCRATCH_DIR not set -- refusing to run.", file=sys.stderr)
        sys.exit(1)

    result = subprocess.run(
        [sys.executable, "-m", "evaluation.run_baseline"],
        # or whatever the actual current entrypoint invocation is --
        # confirm against the Dockerfile's existing CMD rather than guess
    )

    if result.returncode != 0:
        print(f"Sweep exited with code {result.returncode} -- NOT "
              f"terminating. Pod stays alive for inspection.", file=sys.stderr)
        sys.exit(result.returncode)

    if not verify_success(scratch_dir):
        print("Sweep exited 0 but expected output files are missing or "
              "empty -- NOT terminating. Investigate before assuming this "
              "run succeeded.", file=sys.stderr)
        sys.exit(1)

    print("Sweep confirmed successful. Terminating pod.")
    import runpod
    runpod.api_key = os.environ["RUNPOD_API_KEY"]
    pod_id = os.environ["RUNPOD_POD_ID"]
    runpod.terminate_pod(pod_id)


if __name__ == "__main__":
    main()
```

Fix the `subprocess.run` invocation to match however `evaluation/run_baseline.py`
is actually currently invoked (check the Dockerfile's existing `CMD` for
the real command) — don't assume the placeholder above is exactly right.

## Task 3 — testing without spending real RunPod money

This needs to be testable locally before ever running on a real pod:

1. **Unit-test `verify_success()` directly** against a temp directory with
   fake JSONL/CSV files (present + non-empty → True; missing → False;
   present but empty → False). No RunPod API involved, no cost.
2. **Test the failure path** by having the subprocess call return a
   non-zero exit code (mock or a deliberately-failing dummy command) and
   confirm `runpod.terminate_pod` is never reached — assert this via
   mocking the `runpod` import so the test can't accidentally make a real
   API call even if the logic is wrong.
3. Only after both pass locally does this get tried on an actual RunPod
   pod, and even then, start with the smoke-test filters
   (`RAG_MODELS`/`RAG_CORPORA` restricted to one model/corpus) before a
   real full-cost run.

## Verification before calling this done

- Confirm the Azure Dockerfile `CMD` is byte-for-byte unchanged.
- Confirm `verify_success()` unit tests pass for all three cases (success,
  missing files, empty files).
- Confirm the mocked-failure test proves `terminate_pod` is never called
  on a non-zero exit.
- State clearly in the PR that this has NOT yet been tested against a
  real RunPod pod — that's a separate, explicit step the student does
  directly, same as Part B was for Azure, not something to claim as done
  from local testing alone.
