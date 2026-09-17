#!/usr/bin/env bash
#
# Backend-matched, NO-DEFENSE, HF-backend baseline for injection/output_filter
# and injection/spotlighting -- isolates whether those two defenses' reported
# ASR reductions are caused by the _hf-vs-_vllm backend switch between Phase
# 2's baseline and Phase 3's defended runs, independent of any defense.
# See PHASE3_DEFENSE_INSIGHTS.md headline finding 2 and Methodology item (f)
# for why this confound matters: output_filter/injection's guard only fired
# on 0.8% of blocked items, so most of its measured ASR reduction is more
# likely this backend switch than the guard.
#
# Item selection is a TRUE paired re-baseline, not a fresh random sample:
# evaluation/run_attack_injection.py's hf-engine item-selection loop depends
# only on (corpus_name, split, sample_n) -- never on model_key, template, or
# defense -- so RAG_DEFENSE=none with the same corpus + RAG_SAMPLE_N=1000
# deterministically re-selects the exact same 1000 questions, in the exact
# same order, that every real output_filter/spotlighting cell already used.
# Verified empirically (not just reasoned from code) by
# scripts/verify_phase3_backend_baseline_items.py, which loads no LLM/guard
# model and touches no GPU -- run it FIRST if you haven't already:
#     python -m scripts.verify_phase3_backend_baseline_items
# Result as of this writing: 35/35 real output_filter/spotlighting cells
# (19 unique model/corpus/template combos, since output_filter's coverage is
# a superset of spotlighting's -- see the two invocations below) match
# exactly, item-for-item, corpus-for-corpus.
#
# Run this FROM THE REPO ROOT, on a GPU pod, on the same image Phase 3's
# defended sweep used (defense/wire-sweep-runners). Do NOT run this script
# until a pod is up -- it is prepared, not launched, per the task instruction.
#
# Wall-clock estimate (see report for the full derivation): 19 (model,
# corpus, template) combos x n=1000 = 19,000 generations total. Tonight's
# observed HF-backend, no-guard-model pace for spotlighting was ~0.6-2s/item
# on real hardware; this no-defense baseline goes through the SAME
# generation path output_filter/injection uses (run_attack_query, see
# evaluation/run_attack_injection.py's `else` branch) minus the guard-model
# classification step output_filter adds and minus spotlighting's
# base64-lengthened context, so it should be at or below that pace, not
# above it. At 0.6-2.0s/item: 19,000 x 0.6s = 11,400s (~3.2h) to
# 19,000 x 2.0s = 38,000s (~10.6h), plus ~15-20 min of cumulative cold
# model-load overhead across the 4 models involved (same order of magnitude
# as run_phase3_smoke_test.sh's own per-model load estimate). Realistic
# range: ~3.5-11 hours wall-clock, most likely toward the lower half of
# that range given this path has less per-item overhead than either
# defended path it's being compared against.

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

# --- Cell 1: all 4 models x both corpora x {ignore, fake_completion} ---
# Covers all 16 real spotlighting cells and 16 of output_filter's 19.
# INFERENCE_ENGINE intentionally left unset -- "hf" is already the default
# (evaluation/result_paths.py resolve_attack_sweep_selection), same as every
# Phase 3 defended cell.

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

# --- Cell 2: qwen3-8b's 3 extra hotpot_qa templates (output_filter-only combos) ---

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
