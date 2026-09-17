#!/usr/bin/env bash
#
# Rerun injection/instruction_detection with the now-fixed per-passage
# mechanism logging (defenses/instruction_detection.py's
# log_passage_detection_event, wired into evaluation/run_attack_injection.py's
# _build_defended_attack_prompt call sites). Previously, detect_injection()'s
# DetectionResult was computed per passage and used only inline to filter the
# context -- never persisted -- making the mechanism question ("did blocking
# actually happen, or did the model just resist anyway") unanswerable from
# the committed data (see docs/PHASE3_DEFENSE_INSIGHTS.md's
# mechanism_instruction_detection() finding). This produces the missing
# instruction_detection_log_attack_{model}_{corpus}.jsonl files.
#
# NOTE on scope: the real coverage is 16 cells (4 models x 2 corpora x
# {ignore, fake_completion}), not 8 -- confirmed by
# scripts/analyze_phase3_defense_stats.py's discover_injection_cells(), the
# same "verify the real count, don't trust the assumed one" discipline this
# whole Phase 3 analysis has used throughout (Task 1 corrected 52 assumed
# cells to 79 real; this is the same class of correction, scoped to one
# defense). qwen3-8b's 3 extra hotpot_qa templates (naive/escape_char/
# combined) are n=3 smoke-test fragments, not real cells -- excluded here,
# same as everywhere else in this analysis.
#
# THIS OVERWRITES the 16 already-committed
# attack_raw/summary_*_hf_defense-instruction_detection.jsonl/csv files at
# their existing filenames (defense=instruction_detection always writes the
# same names). This is intentional -- it is a rerun, not a fresh cell -- and
# should not change attack_success/ASR numbers (do_sample=False, same items,
# same model/classifier behavior, only the logging side-effect is new); if
# any number DOES change, that's worth investigating before trusting the
# rerun. Commit the new files (raw+summary+log) as their own snapshot,
# following this session's existing precedent, once run.
#
# Run this FROM THE REPO ROOT, on a GPU pod, on the same image Phase 3's
# defended sweep used (defense/wire-sweep-runners) -- built AFTER this
# session's instruction_detection logging fix is merged into that image.
# Do NOT run until a pod is up; this script is prepared, not launched.
#
# Wall-clock estimate: 16 cells x n=40 = 640 generations total, plus one
# DeBERTa classifier forward pass per retrieved passage (TOP_K=5 -> up to
# 5 per item, ~3200 classifier calls, each sub-second per this session's
# CPU-thread-capping fix). Per the task's own estimate, this should be fast:
# ~10-20 minutes total, dominated by the 4 cold model loads (~2-4 min each,
# same order as run_phase3_smoke_test.sh's estimate) more than generation
# itself at this small n.

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
echo "--- [id-rerun] plus 8 new instruction_detection_log_attack_{model}_{corpus}.jsonl files (one per model x corpus, accumulating both templates' rows)."

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
