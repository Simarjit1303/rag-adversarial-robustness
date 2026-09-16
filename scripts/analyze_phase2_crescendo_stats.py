import itertools
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MODELS
from evaluation.stats import holm_bonferroni, mcnemar_exact, paired_bootstrap_ci
RESULTS_DIR = Path('phase2_crescendo_real_sweep')
MAX_TURNS = 5
ENGINE = 'hf'

def _load_by_behavior(model):
    path = RESULTS_DIR / f'crescendo_raw_{model}_{MAX_TURNS}turn_{ENGINE}.jsonl'
    rows = {}
    with path.open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            rows[row['behavior']] = row
    return rows

def main():
    model_keys = list(MODELS)
    data = {m: _load_by_behavior(m) for m in model_keys}
    print('=== Per-model summary (recomputed from raw, judge_failed excluded) ===')
    for m in model_keys:
        rows = list(data[m].values())
        scored = [r for r in rows if not r['judge_failed']]
        n, n_scored = (len(rows), len(scored))
        asr = sum((r['attack_success'] for r in scored)) / n_scored
        backtrack_rate = sum((r['any_backtrack'] for r in rows)) / n
        mean_backtrack = sum((r['backtrack_count'] for r in rows)) / n
        mean_refusal = sum((r['refusal_count'] for r in rows)) / n
        print(f'  {m:16s} n={n} n_scored={n_scored} asr={asr:.4f} backtrack_rate={backtrack_rate:.4f} mean_backtrack={mean_backtrack:.3f} mean_refusal={mean_refusal:.3f}')
    print('\n=== Model-vs-model McNemar (paired by behavior, judge_failed excluded from pair) ===')
    p_values, labels, tables = ([], [], [])
    for a, b in itertools.combinations(model_keys, 2):
        shared = [beh for beh in data[a] if beh in data[b] and (not data[a][beh]['judge_failed']) and (not data[b][beh]['judge_failed'])]
        correct_a = [data[a][beh]['attack_success'] for beh in shared]
        correct_b = [data[b][beh]['attack_success'] for beh in shared]
        result = mcnemar_exact(correct_a, correct_b)
        p_values.append(result['p_value'])
        labels.append(f'{a} vs {b}')
        tables.append((a, b, len(shared), result))
    corrected = holm_bonferroni(p_values)
    for (a, b, n_paired, result), label, holm in zip(tables, labels, corrected):
        print(f"  {label:34s} n_paired={n_paired:3d} b={result['b']:2d} c={result['c']:2d} p_raw={result['p_value']:.4f} rank={holm['rank']} significant_holm={holm['significant_holm']}")
    print('\n=== Paired bootstrap 95% CI on ASR difference (mean(a) - mean(b)) ===')
    for a, b in itertools.combinations(model_keys, 2):
        shared = [beh for beh in data[a] if beh in data[b] and (not data[a][beh]['judge_failed']) and (not data[b][beh]['judge_failed'])]
        scores_a = [data[a][beh]['attack_success'] for beh in shared]
        scores_b = [data[b][beh]['attack_success'] for beh in shared]
        ci = paired_bootstrap_ci(scores_a, scores_b)
        print(f'  {a} vs {b}: {ci}')
    print('\n=== Backtrack activity by behavior source (jbb_behaviors vs harmbench) ===')
    for m in model_keys:
        rows = list(data[m].values())
        for source in ('jbb_behaviors', 'harmbench'):
            sub = [r for r in rows if r['source'] == source]
            if not sub:
                continue
            rate = sum((r['any_backtrack'] for r in sub)) / len(sub)
            print(f'  {m:16s} {source:14s} n={len(sub):3d} backtrack_rate={rate:.4f}')
    print('\n=== judge_failed / error breakdown ===')
    for m in model_keys:
        rows = list(data[m].values())
        failed = [r for r in rows if r['judge_failed']]
        print(f'  {m:16s} n_judge_failed={len(failed)}')
        for r in failed:
            print(f"    error={r['error']!r} n_turns_completed={r['n_turns_completed']}")
    print('\n=== Sample backtrack_attempts entries (one per model with any_backtrack, truncated) ===')
    for m in model_keys:
        rows = list(data[m].values())
        with_bt = [r for r in rows if r['backtrack_count'] > 0]
        if not with_bt:
            print(f'  {m}: none')
            continue
        r = with_bt[0]
        bt = r['backtrack_attempts'][0]
        print(f"  {m} (behavior source={r['source']}, final attack_success={r['attack_success']}):")
        print(f"    turn={bt['turn']}")
        print(f"    refused_prompt[:180]={bt['refused_prompt'][:180]!r}")
        print(f"    refused_reply[:180]={bt['refused_reply'][:180]!r}")
        retry = bt['retry_prompt']
        print(f'    retry_prompt[:180]={(retry[:180] if retry else None)!r}')
if __name__ == '__main__':
    main()
