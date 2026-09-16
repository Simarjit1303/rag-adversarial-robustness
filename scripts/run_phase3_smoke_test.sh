#!/usr/bin/env bash

set -uo pipefail

cd "$(dirname "$0")/.." || { echo "[smoke-test] could not cd to repo root"; exit 1; }

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


export RAG_SAMPLE_N=3 RAG_MODELS=qwen3-8b RAG_DEFENSE=output_filter
unset RAG_CORPORA
run_cell "crescendo / output_filter" '
from config import RESULTS_DIR
from evaluation.result_paths import expected_crescendo_result_files
for p in expected_crescendo_result_files(RESULTS_DIR):
    print(f"    {p}")
' python -m evaluation.run_crescendo


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
