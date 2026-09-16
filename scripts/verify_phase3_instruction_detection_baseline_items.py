import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_phase3_defense_stats import P3_DIR, discover_injection_cells
TARGET_DEFENSE = 'instruction_detection'
SAMPLE_N = 40

def _questions(path):
    questions = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            questions.append(json.loads(line)['question'])
    return questions

def main():
    cells, _fragments = discover_injection_cells()
    target = sorted((k for k in cells if k[3] == TARGET_DEFENSE))
    print(f'=== {len(target)} real {TARGET_DEFENSE} cells need a backend-matched no-defense baseline ===')
    all_match = True
    already_available = []
    needs_launch = []
    for model, corpus, template, defense in target:
        id_path = P3_DIR / f'attack_raw_{model}_{corpus}_{template}_hf_defense-{TARGET_DEFENSE}.jsonl'
        nodef_path = P3_DIR / f'attack_raw_{model}_{corpus}_{template}_hf.jsonl'
        id_questions = _questions(id_path)
        n_id = len(id_questions)
        if not nodef_path.exists():
            print(f'  {model}/{corpus}/{template}: NO Task B file at {nodef_path.name} -- needs launch')
            all_match = False
            needs_launch.append((model, corpus, template))
            continue
        nodef_questions = _questions(nodef_path)
        prefix = nodef_questions[:SAMPLE_N]
        match = n_id == SAMPLE_N and id_questions == prefix
        print(f'  {model}/{corpus}/{template}: n_instruction_detection={n_id} n_task_b_total={len(nodef_questions)} prefix_match={match}')
        print(f'      instruction_detection first 3: {id_questions[:3]}')
        print(f'      task_b prefix     first 3: {prefix[:3]}')
        if match:
            already_available.append((model, corpus, template))
        else:
            all_match = False
            needs_launch.append((model, corpus, template))
            if n_id != len(prefix):
                print(f'      LENGTH MISMATCH: instruction_detection n={n_id} vs task_b prefix n={len(prefix)}')
            else:
                for i, (a, p) in enumerate(zip(id_questions, prefix)):
                    if a != p:
                        print(f'      FIRST MISMATCH at index {i}: instruction_detection={a!r} task_b_prefix={p!r}')
                        break
    print(f'\n=== {len(already_available)}/{len(target)} cells: matched no-defense baseline already exists (first {SAMPLE_N} rows of the committed Task B file) -- zero new GPU time needed for these ===')
    for model, corpus, template in already_available:
        print(f'    OK   {model}/{corpus}/{template}')
    print(f'\n=== {len(needs_launch)}/{len(target)} cells: need run_phase3_instruction_detection_baseline.sh ===')
    for model, corpus, template in needs_launch:
        print(f'    RUN  {model}/{corpus}/{template}')
    print(f'\n=== overall: all cells covered without a new sweep: {all_match} ===')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
