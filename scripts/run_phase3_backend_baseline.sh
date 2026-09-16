#!/usr/bin/env bash

set -uo pipefail

cd "$(dirname "$0")/.." || { echo "[backend-baseline] could not cd to repo root"; exit 1; }

run_cell() {
    local label="$1"
    shift
    local path_lister="$1"
    shift

    echo ""
    echo "=== [backend-baseline] running: $label ==="
    echo "    command: $*"

    "$@"
    local rc=$?

    if [ "$rc" -ne 0 ]; then
        echo ""
        echo "!!! [backend-baseline] FAILED (exit $rc): $label"
        echo "!!! [backend-baseline] command: $*"
        exit 1
    fi

    echo "--- [backend-baseline] $label succeeded. Result files:"
    python -c "$path_lister"
}


export RAG_SAMPLE_N=1000 RAG_DEFENSE=none
export RAG_MODELS=llama-3.1-8b,ministral-3-8b,phi-4-mini,qwen3-8b
export RAG_CORPORA=hotpot_qa,ms_marco
export RAG_INJECTION_TEMPLATES=ignore,fake_completion
run_cell "no-defense hf baseline / 4 models x 2 corpora x {ignore,fake_completion}" '
from config import RESULTS_DIR
from evaluation.result_paths import expected_attack_result_files
for p in expected_attack_result_files(RESULTS_DIR):
    print(f"    {p}")
' python -m evaluation.run_attack_injection


export RAG_SAMPLE_N=1000 RAG_DEFENSE=none
export RAG_MODELS=qwen3-8b
export RAG_CORPORA=hotpot_qa
export RAG_INJECTION_TEMPLATES=combined,escape_char,naive
run_cell "no-defense hf baseline / qwen3-8b x hotpot_qa x {combined,escape_char,naive}" '
from config import RESULTS_DIR
from evaluation.result_paths import expected_attack_result_files
for p in expected_attack_result_files(RESULTS_DIR):
    print(f"    {p}")
' python -m evaluation.run_attack_injection

echo ""
echo "=== [backend-baseline] both cells succeeded. Output files: attack_raw_{model}_{corpus}_{template}_hf.jsonl (no _defense- suffix, defense=none) ==="
echo "=== [backend-baseline] These files pair item-for-item against the corresponding phase3_defense_results/attack_raw_*_hf_defense-{output_filter,spotlighting}.jsonl cells for a McNemar test isolating the backend effect from the defense effect."
exit 0
