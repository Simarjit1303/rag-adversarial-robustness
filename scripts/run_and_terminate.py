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
import subprocess
import sys
from pathlib import Path

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


def run_pipeline() -> int:
    """
    Run every pipeline stage in order. Return 0 only if all stages exit 0;
    otherwise return the first non-zero exit code and stop (a failed stage
    makes every later stage meaningless).
    """
    for cmd in PIPELINE_STAGES:
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(
                f"[run_and_terminate] stage {' '.join(cmd)} exited "
                f"{result.returncode} -- stopping pipeline.",
                file=sys.stderr,
            )
            return result.returncode
    return 0


def terminate_pod() -> None:
    """
    Destroy the current pod via the RunPod API. Imported lazily so the
    failure paths above (and the unit tests) never need the runpod SDK or a
    real API key just to decide NOT to terminate.
    """
    import runpod

    runpod.api_key = os.environ["RUNPOD_API_KEY"]
    pod_id = os.environ["RUNPOD_POD_ID"]
    runpod.terminate_pod(pod_id)


def main() -> None:
    scratch_dir = os.environ.get("RAG_SCRATCH_DIR")
    if not scratch_dir:
        print(
            "[run_and_terminate] RAG_SCRATCH_DIR not set -- refusing to run.",
            file=sys.stderr,
        )
        sys.exit(1)

    returncode = run_pipeline()
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
    terminate_pod()


if __name__ == "__main__":
    main()
