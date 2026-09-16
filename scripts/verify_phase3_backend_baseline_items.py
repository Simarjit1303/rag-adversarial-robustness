import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.loader import load_corpus
from data.normalize import extract_gold_answers, extract_question
from scripts.analyze_phase3_defense_stats import P3_DIR, discover_injection_cells
RE_BASELINE_DEFENSES = ('output_filter', 'spotlighting')
SAMPLE_N = 1000

def would_select(corpus_name, sample_n, split='dev'):
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
    with path.open(encoding='utf-8') as f:
        for line in f:
            questions.append(json.loads(line)['question'])
    return questions

def main():
    cells, _fragments = discover_injection_cells()
    target = {k: v for k, v in cells.items() if k[3] in RE_BASELINE_DEFENSES}
    print(f'=== {len(target)} real output_filter/spotlighting cells need a backend-matched no-defense baseline ===')
    by_corpus = {}
    for model, corpus, template, defense in target:
        by_corpus.setdefault(corpus, set()).add((model, template, defense))
    all_match = True
    for corpus in sorted(by_corpus):
        predicted = would_select(corpus, sample_n=SAMPLE_N)
        print(f'\n--- corpus={corpus}: dry-run predicts {len(predicted)} items (no model loaded, no GPU touched) ---')
        print(f'    first 3: {predicted[:3]}')
        print(f'    last 1:  {predicted[-1:]}')
        for model, template, defense in sorted(by_corpus[corpus]):
            path = P3_DIR / f'attack_raw_{model}_{corpus}_{template}_hf_defense-{defense}.jsonl'
            actual = actual_questions(path)
            match = actual == predicted
            all_match = all_match and match
            print(f'    {path.name}: n={len(actual)} match={match}')
            if not match:
                for i, (a, p) in enumerate(zip(actual, predicted)):
                    if a != p:
                        print(f'      FIRST MISMATCH at index {i}: actual={a!r} predicted={p!r}')
                        break
                if len(actual) != len(predicted):
                    print(f'      LENGTH MISMATCH: actual={len(actual)} predicted={len(predicted)}')
    print(f'\n=== overall item-selection match across every real output_filter/spotlighting cell: {all_match} ===')
    return 0 if all_match else 1
if __name__ == '__main__':
    raise SystemExit(main())
