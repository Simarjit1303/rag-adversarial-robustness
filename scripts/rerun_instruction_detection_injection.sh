#!/usr/bin/env bash

set -uo pipefail

cd "$(dirname "$0")/.." || { echo "[id-rerun] could not cd to repo root"; exit 1; }

export RAG_SAMPLE_N=40 RAG_DEFENSE=instruction_detection
export RAG_MODELS=llama-3.1-8b,ministral-3-8b,phi-4-mini,qwen3-8b
export RAG_CORPORA=hotpot_qa,ms_marco
export RAG_INJECTION_TEMPLATES=ignore,fake_completion

echo "=== [id-rerun] running: instruction_detection rerun / 4 models x 2 corpora x {ignore,fake_completion}, n=40 ==="
python -m evaluation.run_attack_injection
rc=$?
if [ "$rc" -ne 0 ]; then
    echo "!!! [id-rerun] FAILED (exit $rc)"
    exit 1
fi

echo "--- [id-rerun] succeeded. Result files:"
python -c "
from config import RESULTS_DIR
from evaluation.result_paths import expected_attack_result_files
for p in expected_attack_result_files(RESULTS_DIR):
    print(f'    {p}')
"
echo "--- [id-rerun] plus 8 new instruction_detection_log_attack_{model}_{corpus}.jsonl files (one per model x corpus, accumulating both templates rows)."

echo ""
echo "=== [id-rerun] final sanity check: full pytest suite ==="
python -m pytest -q
rc=$?
if [ "$rc" -ne 0 ]; then
    echo "!!! [id-rerun] pytest FAILED (exit $rc) after the real-hardware rerun succeeded -- investigate before committing."
    exit 1
fi

echo ""
echo "=== [id-rerun] rerun + full pytest suite passed. ==="
exit 0
