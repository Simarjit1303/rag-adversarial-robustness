"""
Dry-run item-selection check for instruction_detection's backend-matched
no-defense baseline -- confirms, WITHOUT loading any LLM/GPU, whether the
n=40 items each instruction_detection/injection cell already used are a
PREFIX of the n=1000 items scripts/run_phase3_backend_baseline.sh's
Task B sweep already produced (RAG_DEFENSE=none, RAG_SAMPLE_N=1000,
committed at phase3_defense_results/attack_raw_{model}_{corpus}_{template}_hf.jsonl).

Why this can be answered without a new sweep: evaluation/run_attack_
injection.py's hf-engine item-selection loop (run_attack_sweep) iterates
`records = load_corpus(corpus_name, split)` and keeps the first `sample_n`
records for which extract_question/extract_gold_answers both succeed --
this loop depends only on (corpus_name, split), never on sample_n itself
beyond WHEN to stop, and never on model_key/injection_template/defense
(scripts/verify_phase3_backend_baseline_items.py already confirmed this
empirically for output_filter/spotlighting at n=1000). Consequently, the
first 40 items processed at sample_n=1000 MUST be identical, in the same
order, to the full item set processed at sample_n=40 for the same
(corpus, split) -- IF the two runs used the same defense-independent
selection loop. Both instruction_detection's original n=40 sweep and
Task B's n=1000 sweep did (same function, same loop, only sample_n and
defense differ). This script verifies that prediction directly against
two sets of already-committed files -- no load_corpus() call, no model,
no GPU -- which is stronger evidence than re-deriving the corpus order,
since it also implicitly catches any corpus-cache drift between the two
original runs.

If every cell matches: no new pod time is needed for Task 1 at all --
the matched no-defense baseline for instruction_detection's 3-way test is
just the first 40 rows of Task B's already-committed file, per cell.
If any cell mismatches: Task 1's launch script (prepared alongside this
script) is the fallback -- run it for exactly the mismatching cells.

Run: python -m scripts.verify_phase3_instruction_detection_baseline_items
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_phase3_defense_stats import (  # noqa: E402
    P3_DIR,
    discover_injection_cells,
)

TARGET_DEFENSE = "instruction_detection"
SAMPLE_N = 40


def _questions(path):
    questions = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            questions.append(json.loads(line)["question"])
    return questions


def main():
    cells, _fragments = discover_injection_cells()
    target = sorted(k for k in cells if k[3] == TARGET_DEFENSE)
    print(f"=== {len(target)} real {TARGET_DEFENSE} cells need a backend-matched "
          f"no-defense baseline ===")

    all_match = True
    already_available = []
    needs_launch = []

    for (model, corpus, template, defense) in target:
        id_path = P3_DIR / f"attack_raw_{model}_{corpus}_{template}_hf_defense-{TARGET_DEFENSE}.jsonl"
        nodef_path = P3_DIR / f"attack_raw_{model}_{corpus}_{template}_hf.jsonl"

        id_questions = _questions(id_path)
        n_id = len(id_questions)

        if not nodef_path.exists():
            print(f"  {model}/{corpus}/{template}: NO Task B file at {nodef_path.name} "
                  f"-- needs launch")
            all_match = False
            needs_launch.append((model, corpus, template))
            continue

        nodef_questions = _questions(nodef_path)
        prefix = nodef_questions[:SAMPLE_N]
        match = (n_id == SAMPLE_N) and (id_questions == prefix)

        print(f"  {model}/{corpus}/{template}: n_instruction_detection={n_id} "
              f"n_task_b_total={len(nodef_questions)} prefix_match={match}")
        print(f"      instruction_detection first 3: {id_questions[:3]}")
        print(f"      task_b prefix     first 3: {prefix[:3]}")

        if match:
            already_available.append((model, corpus, template))
        else:
            all_match = False
            needs_launch.append((model, corpus, template))
            if n_id != len(prefix):
                print(f"      LENGTH MISMATCH: instruction_detection n={n_id} vs "
                      f"task_b prefix n={len(prefix)}")
            else:
                for i, (a, p) in enumerate(zip(id_questions, prefix)):
                    if a != p:
                        print(f"      FIRST MISMATCH at index {i}: "
                              f"instruction_detection={a!r} task_b_prefix={p!r}")
                        break

    print(f"\n=== {len(already_available)}/{len(target)} cells: matched no-defense "
          f"baseline already exists (first {SAMPLE_N} rows of the committed Task B "
          f"file) -- zero new GPU time needed for these ===")
    for model, corpus, template in already_available:
        print(f"    OK   {model}/{corpus}/{template}")
    print(f"\n=== {len(needs_launch)}/{len(target)} cells: need "
          f"run_phase3_instruction_detection_baseline.sh ===")
    for model, corpus, template in needs_launch:
        print(f"    RUN  {model}/{corpus}/{template}")

    print(f"\n=== overall: all cells covered without a new sweep: {all_match} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
