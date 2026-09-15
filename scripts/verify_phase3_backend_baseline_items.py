"""
Dry-run item-selection check for scripts/run_phase3_backend_baseline.sh --
confirms, WITHOUT loading any LLM/guard model and WITHOUT touching a GPU,
that a fresh no-defense HF-backend sweep (RAG_DEFENSE=none, RAG_SAMPLE_N=1000)
would select the exact same questions, in the exact same order, that Phase
3's already-committed injection/output_filter and injection/spotlighting
cells used.

Why this works without a model: evaluation/run_attack_injection.py's item-
selection loop (run_attack_sweep, the "hf" engine path) iterates
`records = load_corpus(corpus_name, split)` and keeps the first `sample_n`
records for which `extract_question`/`extract_gold_answers` both succeed --
this loop does not depend on model_key, injection_template, or defense at
all, only on (corpus_name, split, sample_n). load_corpus() itself reads a
locally cached, already-sampled-with-a-fixed-seed JSONL
(C:\\rag-data-local\\cache\\{corpus}_dev.jsonl) when present -- confirmed
present for hotpot_qa and ms_marco on this machine -- so this script
reproduces the selection with zero network access, zero embedding model,
zero LLM.

Run: python -m scripts.verify_phase3_backend_baseline_items
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.loader import load_corpus  # noqa: E402
from data.normalize import extract_gold_answers, extract_question  # noqa: E402
from scripts.analyze_phase3_defense_stats import (  # noqa: E402
    P3_DIR,
    discover_injection_cells,
)

RE_BASELINE_DEFENSES = ("output_filter", "spotlighting")
SAMPLE_N = 1000


def would_select(corpus_name, sample_n, split="dev"):
    """Mirrors evaluation/run_attack_injection.py's hf-engine item-selection
    loop exactly (lines ~244-250), minus everything after item selection."""
    records = load_corpus(corpus_name, split=split)
    selected = []
    n_processed = 0
    for record in records:
        question = extract_question(corpus_name, record)
        gold = extract_gold_answers(corpus_name, record)
        if not question or not gold:
            continue
        if sample_n and n_processed >= sample_n:
            break
        selected.append(question)
        n_processed += 1
    return selected


def actual_questions(path):
    questions = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            questions.append(json.loads(line)["question"])
    return questions


def main():
    cells, _fragments = discover_injection_cells()
    target = {k: v for k, v in cells.items() if k[3] in RE_BASELINE_DEFENSES}
    print(f"=== {len(target)} real output_filter/spotlighting cells need a "
          f"backend-matched no-defense baseline ===")

    by_corpus = {}
    for (model, corpus, template, defense) in target:
        by_corpus.setdefault(corpus, set()).add((model, template, defense))

    all_match = True
    for corpus in sorted(by_corpus):
        predicted = would_select(corpus, sample_n=SAMPLE_N)
        print(f"\n--- corpus={corpus}: dry-run predicts {len(predicted)} items "
              f"(no model loaded, no GPU touched) ---")
        print(f"    first 3: {predicted[:3]}")
        print(f"    last 1:  {predicted[-1:]}")

        for model, template, defense in sorted(by_corpus[corpus]):
            path = P3_DIR / f"attack_raw_{model}_{corpus}_{template}_hf_defense-{defense}.jsonl"
            actual = actual_questions(path)
            match = actual == predicted
            all_match = all_match and match
            print(f"    {path.name}: n={len(actual)} match={match}")
            if not match:
                for i, (a, p) in enumerate(zip(actual, predicted)):
                    if a != p:
                        print(f"      FIRST MISMATCH at index {i}: "
                              f"actual={a!r} predicted={p!r}")
                        break
                if len(actual) != len(predicted):
                    print(f"      LENGTH MISMATCH: actual={len(actual)} "
                          f"predicted={len(predicted)}")

    print(f"\n=== overall item-selection match across every real "
          f"output_filter/spotlighting cell: {all_match} ===")
    return 0 if all_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
