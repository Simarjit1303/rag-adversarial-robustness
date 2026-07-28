"""
RunPod entrypoint wrapper. Runs the Phase 1 sweep, then self-terminates the
pod ONLY on confirmed success -- never on failure, so a crashed run stays
alive for log inspection instead of erasing its own evidence.

This is NOT used by the Azure path and MUST NOT be wired into the Dockerfile
CMD. The image's default CMD (run sweep, then `sleep infinity`) is the
correct behavior for Azure Container Apps, where `sleep infinity` stops the
platform from restart-looping the container (PR #4). On RunPod that same idle
would bill forever, so RunPod's pod-template "Start command" override points
at this script instead:

    python -m scripts.run_and_terminate

Why terminate only on *verified* success: an exit code of 0 does not prove
the sweep wrote correct output -- this project has a documented history of
silent-looking failures (truncated caches, wiped results on restart). So the
pod is only destroyed once the actual expected result files exist and are
non-empty. Anything less leaves the pod alive for inspection.
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

# 6h default is generous for a real sweep; override to something short (e.g.
# 1800 = 30 min) for smoke tests via the pod's env vars. Exists because a
# genuine hang (phi-4-mini loaded, then 0% CPU/GPU for 12+ minutes with zero
# telemetry movement) sat unbounded on a real RunPod run: the success/failure
# check below only runs AFTER the subprocess exits, so a hang that never
# exits never reaches it, and the pod billed indefinitely until stopped by
# hand -- the exact failure mode self-termination exists to prevent, from a
# different angle (a hang instead of a restart loop).
PIPELINE_TIMEOUT_SECONDS = int(os.environ.get("RAG_PIPELINE_TIMEOUT_SECONDS", 21600))

# The full pipeline the Azure Dockerfile CMD runs, minus the trailing
# `sleep infinity` (which is exactly the part that must NOT happen on RunPod).
# Cache the corpora, build the FAISS indices, then run the sweep -- same three
# stages, same order, so a fresh pod with empty scratch reaches a real result.
# run_baseline can build indices lazily on its own, but running the stages
# explicitly matches the deployed Azure sequence and pins the blame to a
# specific stage when one fails. RAG_MODE=debug is deliberately NOT mirrored
# here: a debug sample run has no results to verify and nothing to terminate
# for.
PIPELINE_STAGES = (
    [sys.executable, "-m", "data.loader"],
    [sys.executable, "-m", "data.build_index"],
    [sys.executable, "-m", "evaluation.run_baseline"],
)

# The exact files run_baseline.py writes (confirmed in evaluation/run_baseline.py:
# RESULTS_DIR / "baseline_raw.jsonl" and RESULTS_DIR / "baseline_summary.csv").
# Checked by name, not a *.jsonl / *.csv glob, so a stray leftover file can
# never be mistaken for a real result.
EXPECTED_RESULT_FILES = ("baseline_raw.jsonl", "baseline_summary.csv")


def verify_success(scratch_dir: str) -> bool:
    """
    Confirm results actually landed, not just that the sweep process exited 0.

    Both expected output files (the raw per-question JSONL and the summary
    CSV, written via PR #9's atomic-write pattern) must exist and be
    non-empty. An exit code alone doesn't prove correct output.
    """
    results_dir = Path(scratch_dir) / "results"
    if not results_dir.exists():
        return False
    for name in EXPECTED_RESULT_FILES:
        f = results_dir / name
        if not f.exists() or f.stat().st_size == 0:
            return False
    return True


def run_pipeline():
    """
    Run every pipeline stage in order, each bounded by PIPELINE_TIMEOUT_SECONDS.
    Return 0 only if all stages exit 0; the first non-zero exit code and stop
    (a failed stage makes every later stage meaningless); or None if a stage
    hung past the timeout and had to be killed -- distinct from a real exit
    code so the caller can log "TIMEOUT" specifically, not just "exit != 0".

    Each stage runs in its own process group (start_new_session=True) so a
    timeout kill takes any child processes with it -- a plain proc.kill()
    only kills the direct child, not grandchildren a hung stage may have
    spawned.
    """
    for cmd in PIPELINE_STAGES:
        proc = subprocess.Popen(cmd, start_new_session=True)
        try:
            returncode = proc.wait(timeout=PIPELINE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            # SIGKILL doesn't exist on Windows (this runs in Linux Docker on
            # RunPod/Azure, but the getattr fallback keeps the module
            # importable/testable on a Windows dev machine too).
            os.killpg(os.getpgid(proc.pid), getattr(signal, "SIGKILL", signal.SIGTERM))
            proc.wait()
            print(
                f"[run_and_terminate] stage {' '.join(cmd)} exceeded "
                f"{PIPELINE_TIMEOUT_SECONDS}s -- killed. NOT terminating pod "
                f"(same as any other failure), but this bounds the hang "
                f"instead of leaving it unbounded.",
                file=sys.stderr,
            )
            return None
        if returncode != 0:
            print(
                f"[run_and_terminate] stage {' '.join(cmd)} exited "
                f"{returncode} -- stopping pipeline.",
                file=sys.stderr,
            )
            return returncode
    return 0


def terminate_pod(pod_id: str, api_key: str) -> None:
    """
    Destroy the current pod via RunPod's REST API.

    This deliberately does NOT use the `runpod` SDK. That package hard-depends
    on fastapi>=0.139.2 while vllm 0.25.1 caps fastapi<0.137.0, so the two
    cannot be installed together at all -- `pip install -r requirements.txt`
    fails outright on the non-overlapping ranges. Terminate is a plain HTTP
    DELETE, so issuing it directly needs only `requests`, which vllm already
    pulls in transitively. Returns 204 on success; raise_for_status() turns
    anything else into a loud exception rather than a silently-unterminated
    (still billing) pod.
    """
    response = requests.delete(
        f"https://rest.runpod.io/v1/pods/{pod_id}",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    response.raise_for_status()  # raises on anything other than 2xx


def _fail_and_idle(message: str) -> None:
    """
    The only place any failure path in this script should end up. Never
    sys.exit() or let an exception propagate from a failure branch directly
    -- call this instead.

    Exists because idling was previously applied at one failure site only
    (termination-call failure, Bug 3), and every other failure site still
    exited normally after logging intent ("NOT terminating. Pod stays alive
    for inspection") -- which was directly observed to be false: RunPod
    restarts on the exit itself regardless of what got printed first,
    triggering a full paid pipeline re-run, repeatedly, not just once. This
    is now the universal rule for every failure, enforced structurally
    (one choke point) rather than remembered per-site.
    """
    print(
        f"[run_and_terminate] FAILURE: {message} -- idling instead of "
        f"exiting. Any exit here has been directly observed to trigger a "
        f"full pipeline re-run rather than a clean stop. Stop this pod "
        f"manually once you've finished inspecting.",
        file=sys.stderr,
    )
    _idle_forever()  # reuse Bug 3's existing, already-tested idle loop


def _idle_forever() -> None:
    """
    Deliberately has no timeout of its own, unlike Bug 1's pipeline hang
    (which resolves on its own once that fix lands). Every path that reaches
    this only does so AFTER either a confirmed successful run or a failure
    whose evidence (partial results, logs) is worth preserving, so the
    actual work is never at risk -- only additional idle billing time until
    a human notices the log message above and stops the pod manually. A
    bounded idle timeout here too is a deliberate future decision, not
    something to add unasked.
    """
    while True:
        time.sleep(3600)


def main() -> None:
    scratch_dir = os.environ.get("RAG_SCRATCH_DIR")
    if not scratch_dir:
        _fail_and_idle("RAG_SCRATCH_DIR not set -- refusing to run")
        return

    returncode = run_pipeline()
    if returncode is None:
        # None (not a real exit code) is what run_pipeline() returns on a
        # pipeline timeout (Bug 1).
        _fail_and_idle("pipeline TIMED OUT")
        return
    if returncode != 0:
        _fail_and_idle(f"pipeline stage exited {returncode}")
        return

    if not verify_success(scratch_dir):
        _fail_and_idle(
            "pipeline exited 0 but expected output files are missing or "
            "empty -- investigate before assuming this run succeeded"
        )
        return

    print("[run_and_terminate] sweep confirmed successful. Terminating pod.")
    pod_id = os.environ["RUNPOD_POD_ID"]
    # NOT RUNPOD_API_KEY: RunPod auto-injects its own RUNPOD_API_KEY into
    # every pod, scoped/restricted in a way that causes exactly the 403 seen
    # in Bug 3 -- confirmed as a known, unresolved RunPod limitation by a
    # RunPod team member. RUNPOD_TERMINATE_KEY is this project's own
    # credential, set explicitly on the pod template, so it never collides
    # with RunPod's auto-injected one.
    api_key = os.environ["RUNPOD_TERMINATE_KEY"]
    try:
        terminate_pod(pod_id, api_key)
        print("[run_and_terminate] pod terminated successfully.")
    except Exception as e:
        # Genuine success path up to this point -- pipeline succeeded and
        # results are already safe on the Network Volume -- so this is the
        # one failure site where nothing is lost by idling instead of
        # exiting; still routed through the same shared choke point as
        # every other failure, not a separate inline idle loop, so there is
        # one implementation instead of two that could drift apart later.
        _fail_and_idle(
            f"PIPELINE SUCCEEDED, results are safe on the Network Volume. "
            f"Termination call FAILED: {e!r}. This pod may still be "
            f"billing -- check manually"
        )
        return


def _main_with_backstop() -> None:
    """
    The final backstop, not a replacement for main()'s 5 specific failure
    sites (missing scratch dir, non-zero stage exit, pipeline timeout,
    verify_success()-False, termination-call failure) -- those still matter
    because they give a clear, specific error message at each known point
    rather than a generic one. Bug 4's audit covered every failure path
    CURRENTLY WRITTEN into this file, but it can't cover an exception type
    nobody's hit yet: a KeyError from an env var that isn't one of the ones
    explicitly checked, an OSError if the Network Volume runs low on space,
    something raised deep inside torch or sentence-transformers that's
    never been triggered before. Any of those would crash exactly the same
    way -- process exits, RunPod restarts, full paid re-run -- without ever
    touching one of the 5 known sites, purely because it isn't one of them.

    `Exception` (not a bare `except:`) deliberately still lets
    KeyboardInterrupt/SystemExit propagate normally, rather than swallowing
    genuine intentional interrupts too.
    """
    try:
        main()
    except Exception as e:
        _fail_and_idle(f"unhandled exception: {e!r}")


if __name__ == "__main__":
    _main_with_backstop()
