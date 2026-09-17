#!/usr/bin/env bash
#
# Phase 3 defense wiring: real end-to-end smoke test on the RunPod GPU pod.
# Runs the 8 defense x runner cells from defense/wire-sweep-runners (commit
# ea5edc1) SEQUENTIALLY -- never parallel, since output_filter cells load a
# second ~12B model (Llama-Guard-4-12B) alongside the qwen3-8b target and
# concurrent cells would risk OOM on shared GPU memory.
#
# Run this FROM THE REPO ROOT, inside the pod's container, on the image
# built from defense/wire-sweep-runners (NOT the main-branch image -- see
# this session's report for the workflow_dispatch command that builds it).
# Nothing in this script downloads anything on its own; it only invokes the
# existing sweep runners, which do their own model/dataset loading exactly
# as they always have.
#
# Wall-clock estimate: ~45-75 minutes total. This is dominated by repeated
# COLD model loads, not generation -- each `python -m ...` below is a fresh
# process, so nothing is cached across cells within this script:
#   - 7 separate qwen3-8b loads (one per command below) at ~2-4 min each
#     on a real GPU pod
#   - 3 of those cells (output_filter) additionally cold-load
#     Llama-Guard-4-12B (~24GB bf16) at ~3-6 min each
#   - generation itself is small at n=3, but cells 1-3 default to ALL 5
#     injection templates (RAG_INJECTION_TEMPLATES is not set) -- so cells
#     1-3 actually run 5 templates x 3 questions = 15 generations each, not
#     3; cells 4-6 default to the single adv5 poison config (3 generations
#     each); cell 7 samples 3 Crescendo behaviors x up to 5 turns
#   - final pytest run: ~1-2 min (no GPU needed, matches the ~30-75s
#     observed on a laptop CPU)
# Actual time depends entirely on the pod's GPU and whether weights are
# already warm in the HF cache on the Network Volume.

set -uo pipefail

cd "$(dirname "$0")/.." || { echo "[smoke-test] could not cd to repo root"; exit 1; }

# run_cell <label> <result-path-lister-python-snippet> <command...> -- env
# vars for the actual sweep command are set by the caller via `export`
# immediately before calling this function, so both the sweep run and the
# path-lister below see the identical RAG_* configuration (same source of
# truth as the runners themselves use:
# evaluation.result_paths.resolve_*_sweep_selection()).
run_cell() {
    local label="$1"
    shift
    local path_lister="$1"
    shift

    echo ""
    echo "=== [smoke-test] running: $label ==="
    echo "    command: $*"

    "$@"
    local rc=$?

    if [ "$rc" -ne 0 ]; then
        echo ""
        echo "!!! [smoke-test] FAILED (exit $rc): $label"
        echo "!!! [smoke-test] command: $*"
        echo "!!! [smoke-test] stopping here -- not running remaining cells against a broken foundation."
        exit 1
    fi

    echo "--- [smoke-test] $label succeeded. Result files:"
    python -c "$path_lister"
}

# --- Attack injection (run_attack_injection.py): instruction_detection, spotlighting, output_filter ---

export RAG_SAMPLE_N=3 RAG_MODELS=qwen3-8b RAG_CORPORA=hotpot_qa RAG_DEFENSE=instruction_detection
run_cell "attack_injection / instruction_detection" '
from config import RESULTS_DIR
from evaluation.result_paths import expected_attack_result_files
for p in expected_attack_result_files(RESULTS_DIR):
    print(f"    {p}")
' python -m evaluation.run_attack_injection

export RAG_SAMPLE_N=3 RAG_MODELS=qwen3-8b RAG_CORPORA=hotpot_qa RAG_DEFENSE=spotlighting
run_cell "attack_injection / spotlighting" '
from config import RESULTS_DIR
from evaluation.result_paths import expected_attack_result_files
for p in expected_attack_result_files(RESULTS_DIR):
    print(f"    {p}")
' python -m evaluation.run_attack_injection

export RAG_SAMPLE_N=3 RAG_MODELS=qwen3-8b RAG_CORPORA=hotpot_qa RAG_DEFENSE=output_filter
run_cell "attack_injection / output_filter" '
from config import RESULTS_DIR
from evaluation.result_paths import expected_attack_result_files
for p in expected_attack_result_files(RESULTS_DIR):
    print(f"    {p}")
' python -m evaluation.run_attack_injection

# --- PoisonedRAG (run_poisonedrag.py): instruction_detection, spotlighting, output_filter ---

export RAG_SAMPLE_N=3 RAG_MODELS=qwen3-8b RAG_CORPORA=hotpot_qa RAG_DEFENSE=instruction_detection
run_cell "poisonedrag / instruction_detection" '
from config import RESULTS_DIR
from evaluation.result_paths import expected_poison_result_files
for p in expected_poison_result_files(RESULTS_DIR):
    print(f"    {p}")
' python -m evaluation.run_poisonedrag

export RAG_SAMPLE_N=3 RAG_MODELS=qwen3-8b RAG_CORPORA=hotpot_qa RAG_DEFENSE=spotlighting
run_cell "poisonedrag / spotlighting" '
from config import RESULTS_DIR
from evaluation.result_paths import expected_poison_result_files
for p in expected_poison_result_files(RESULTS_DIR):
    print(f"    {p}")
' python -m evaluation.run_poisonedrag

export RAG_SAMPLE_N=3 RAG_MODELS=qwen3-8b RAG_CORPORA=hotpot_qa RAG_DEFENSE=output_filter
run_cell "poisonedrag / output_filter" '
from config import RESULTS_DIR
from evaluation.result_paths import expected_poison_result_files
for p in expected_poison_result_files(RESULTS_DIR):
    print(f"    {p}")
' python -m evaluation.run_poisonedrag

# --- Crescendo (run_crescendo.py): output_filter only (instruction_detection/spotlighting don't apply) ---

export RAG_SAMPLE_N=3 RAG_MODELS=qwen3-8b RAG_DEFENSE=output_filter
unset RAG_CORPORA  # Crescendo has no corpus axis -- a stale export from the cells above would be misleading, even though run_crescendo.py never reads it
run_cell "crescendo / output_filter" '
from config import RESULTS_DIR
from evaluation.result_paths import expected_crescendo_result_files
for p in expected_crescendo_result_files(RESULTS_DIR):
    print(f"    {p}")
' python -m evaluation.run_crescendo

# --- Final sanity check: full pytest suite, same "verify on real hardware" discipline as everything above ---

echo ""
echo "=== [smoke-test] all 8 cells succeeded. Running full pytest suite as a final sanity check ==="
python -m pytest -q
rc=$?
if [ "$rc" -ne 0 ]; then
    echo "!!! [smoke-test] pytest FAILED (exit $rc) after real-hardware smoke test succeeded -- investigate before merging."
    exit 1
fi

echo ""
echo "=== [smoke-test] ALL 8 CELLS + FULL PYTEST SUITE PASSED. Phase 3 defense wiring verified end-to-end. ==="
exit 0
