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


def main() -> None:
    scratch_dir = os.environ.get("RAG_SCRATCH_DIR")
    if not scratch_dir:
        print(
            "[run_and_terminate] RAG_SCRATCH_DIR not set -- refusing to run.",
            file=sys.stderr,
        )
        sys.exit(1)

    returncode = run_pipeline()
    if returncode is None:
        # Timeout, not a real exit code -- sys.exit(None) would exit 0, which
        # would look like success. Exit 1 instead, same as any other failure.
        print(
            "[run_and_terminate] sweep TIMED OUT -- NOT terminating. Pod "
            "stays alive for inspection.",
            file=sys.stderr,
        )
        sys.exit(1)
    if returncode != 0:
        print(
            f"[run_and_terminate] sweep exited {returncode} -- NOT "
            f"terminating. Pod stays alive for inspection.",
            file=sys.stderr,
        )
        sys.exit(returncode)

    if not verify_success(scratch_dir):
        print(
            "[run_and_terminate] sweep exited 0 but expected output files are "
            "missing or empty -- NOT terminating. Investigate before assuming "
            "this run succeeded.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("[run_and_terminate] sweep confirmed successful. Terminating pod.")
    try:
        terminate_pod(os.environ["RUNPOD_POD_ID"], os.environ["RUNPOD_API_KEY"])
        print("[run_and_terminate] pod terminated successfully.")
    except Exception as e:
        # Confirmed by direct observation, twice, on real RunPod pods: an
        # unhandled exception here crashes the process, and something in
        # RunPod's pod orchestration reads that crash-exit as "needs
        # retrying" -- the ENTIRE pipeline (model load, embedding, full
        # sweep) then re-runs from scratch. Not confirmed whether RunPod
        # restarts on any exit or only non-zero ones, so this assumes the
        # worse case (any exit) rather than betting on the better one.
        # A hang (Bug 1) bills at a steady rate; this compounds -- full paid
        # re-runs, repeatedly, for as long as the failure persists. So on
        # failure here, do NOT exit at all: idle instead. The pipeline
        # already succeeded and results are already safe on the Network
        # Volume, so nothing is lost by staying alive -- only additional
        # idle billing until a human notices and stops the pod by hand.
        print(
            f"[run_and_terminate] PIPELINE SUCCEEDED, results are safe on "
            f"the Network Volume. Termination call FAILED: {e!r}. "
            f"This pod may still be billing -- check manually and stop it "
            f"if so. Idling instead of exiting, since an exit here has "
            f"already been observed to trigger a full pipeline re-run "
            f"rather than a clean stop.",
            file=sys.stderr,
        )
        _idle_forever()


def _idle_forever() -> None:
    """
    Deliberately has no timeout of its own, unlike Bug 1's pipeline hang
    (which resolves on its own once that fix lands). This branch only
    triggers AFTER a confirmed successful run, so the actual work and
    results are never at risk -- only additional idle billing time until a
    human notices the log message above and stops the pod manually. A
    bounded idle timeout here too is a deliberate future decision, not
    something to add unasked.
    """
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
