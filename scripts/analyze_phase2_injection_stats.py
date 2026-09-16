import itertools
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attacks.injection_templates import TEMPLATES
from config import MODELS
from evaluation.result_paths import ATTACK_ELIGIBLE_CORPORA
from evaluation.stats import fisher_exact_asr_comparison, holm_bonferroni, mcnemar_exact, paired_bootstrap_ci
BASELINE_DIR = Path('phase1_results_complete')
ATTACK_DIR = Path('phase2_injection_results')

def _load_jsonl_by_question(path):
    rows = {}
    with path.open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            rows[row['question']] = row
    return rows

def _load_baseline(model, corpus):
    return _load_jsonl_by_question(BASELINE_DIR / f'baseline_raw_{model}_{corpus}_vllm.jsonl')

def _load_attack(model, corpus, template):
    return _load_jsonl_by_question(ATTACK_DIR / f'attack_raw_{model}_{corpus}_{template}_vllm.jsonl')

def main():
    model_keys = list(MODELS)
    corpora = list(ATTACK_ELIGIBLE_CORPORA)
    templates = list(TEMPLATES)
    assert len(corpora) == 2, 'corpus-vs-corpus comparison assumes exactly 2 corpora'
    cell_stats = {}
    asr_by_question = {}
    for template in templates:
        for corpus in corpora:
            for model in model_keys:
                baseline = _load_baseline(model, corpus)
                attack = _load_attack(model, corpus, template)
                common_qs = [q for q in attack if q in baseline]
                if len(common_qs) != len(attack):
                    print(f"WARNING: {model}/{corpus}/{template}: {len(attack) - len(common_qs)} attack questions had no baseline match -- excluded from this cell's stats")
                baseline_f1 = [baseline[q]['f1_clean'] for q in common_qs]
                attack_f1 = [attack[q]['f1_clean'] for q in common_qs]
                asr = [attack[q]['attack_success'] for q in common_qs]
                boot = paired_bootstrap_ci(baseline_f1, attack_f1, n_boot=10000, seed=42)
                cell_stats[model, corpus, template] = {'n': len(common_qs), 'asr': sum(asr) / len(asr) if asr else float('nan'), 'f1_baseline': sum(baseline_f1) / len(baseline_f1), 'f1_attack': sum(attack_f1) / len(attack_f1), 'utility_drop': boot['observed_drop'], 'utility_drop_ci_low': boot['ci_low'], 'utility_drop_ci_high': boot['ci_high']}
                asr_by_question[model, corpus, template] = dict(zip(common_qs, asr))
    print('=== Per-cell ASR and utility-under-attack (model x corpus x template) ===')
    header = f"{'model':<16}{'corpus':<10}{'template':<17}{'n':>5}{'ASR':>8}{'F1_base':>9}{'F1_atk':>8}{'drop':>8}{'ci_lo':>8}{'ci_hi':>8}"
    print(header)
    for template in templates:
        for corpus in corpora:
            for model in model_keys:
                s = cell_stats[model, corpus, template]
                print(f"{model:<16}{corpus:<10}{template:<17}{s['n']:>5}{s['asr']:>8.4f}{s['f1_baseline']:>9.4f}{s['f1_attack']:>8.4f}{s['utility_drop']:>8.4f}{s['utility_drop_ci_low']:>8.4f}{s['utility_drop_ci_high']:>8.4f}")
    print('\n=== McNemar (model-vs-model) / Fisher (corpus-vs-corpus) per template ===')
    print("=== Holm-Bonferroni corrected within each template's 16-comparison family ===")
    all_families = {}
    for template in templates:
        family = []
        for corpus in corpora:
            for m1, m2 in itertools.combinations(model_keys, 2):
                d1 = asr_by_question[m1, corpus, template]
                d2 = asr_by_question[m2, corpus, template]
                common = [q for q in d1 if q in d2]
                result = mcnemar_exact([d1[q] for q in common], [d2[q] for q in common])
                family.append((f'model {m1} vs {m2} | corpus {corpus}', result['p_value'], result))
        c1, c2 = corpora
        for model in model_keys:
            d1 = asr_by_question[model, c1, template]
            d2 = asr_by_question[model, c2, template]
            result = fisher_exact_asr_comparison(sum(d1.values()), len(d1), sum(d2.values()), len(d2))
            family.append((f'corpus {c1} vs {c2} | model {model}', result['p_value'], result))
        p_values = [p for _, p, _ in family]
        holm_results = holm_bonferroni(p_values, alpha=0.05)
        all_families[template] = list(zip(family, holm_results))
        print(f'\n--- {template} ---')
        for (label, p, raw), holm in zip(family, holm_results):
            sig = 'SIG' if holm['significant_holm'] else 'ns'
            print(f'  {label:<45} p={p:.4g}  holm={sig}  detail={raw}')
    return (cell_stats, all_families)
if __name__ == '__main__':
    main()
