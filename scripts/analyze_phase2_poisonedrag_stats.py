"""
Phase 2 Attack 2 (PoisonedRAG) statistical analysis.

Reads phase1_results_complete/ (Phase 1 baseline, already committed) and
phase2_poisonedrag_results/ (this attack's real sweep output, downloaded
from the RunPod run) directly by path -- same pattern as
scripts/analyze_phase2_injection_stats.py. Only hotpot_qa and ms_marco
exist here (nq_open excluded project-wide, see docs/nq_open_leakage_finding.md).

Unlike Attack 1, PoisonedRAG has no template dimension -- one poison
config ("adv5", ADV_PER_QUERY=5) per (model, corpus) cell, 8 cells total.
n is NOT the full 1000-question dev slice: PoisonedRAG samples 100 target
questions per corpus (evaluation/run_poisonedrag.py, seed 42) and some are
lost to generator failures during Phase A context-building (see
docs/PHASE2_POISONEDRAG_INSIGHTS.md's skip/loss section) -- n=90/100 hotpot_qa,
n=96/100 ms_marco, same across all 4 models since Phase A's poisoned
contexts are model-agnostic and shared.

Two statistical questions, same shape as Attack 1:

1. Utility-under-attack: paired bootstrap 95% CI on the f1_clean drop
   (gold-answer F1; NOT f1_target, which scores similarity to the
   poisoned target answer, not the real one), Phase 1 baseline vs this
   attack, paired by question text -- one per (model, corpus) cell, 8
   cells.

2. One McNemar/Fisher significance family (16 comparisons, Holm-Bonferroni
   corrected together -- there is only one poison config here, so unlike
   Attack 1's 5 independent per-template families, everything belongs to
   a single family):
     - Model-vs-model, same corpus: paired McNemar on attack_success --
       valid since all 4 models share the identical sampled question set
       per corpus (same cached poisoned_contexts_*.json). C(4,2)=6 pairs
       x 2 corpora = 12 comparisons.
     - Corpus-vs-corpus, same model: Fisher's exact on attack_success
       proportions -- hotpot_qa and ms_marco are different, independently
       sampled question sets, no item-to-item pairing. 1 pair x 4 models
       = 4 comparisons.

Run: python -m scripts.analyze_phase2_poisonedrag_stats
"""

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import MODELS  # noqa: E402
from evaluation.result_paths import ATTACK_ELIGIBLE_CORPORA  # noqa: E402
from evaluation.stats import (  # noqa: E402
    fisher_exact_asr_comparison,
    holm_bonferroni,
    mcnemar_exact,
    paired_bootstrap_ci,
)

BASELINE_DIR = Path("phase1_results_complete")
ATTACK_DIR = Path("phase2_poisonedrag_results")


def _load_jsonl_by_question(path):
    rows = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows[row["question"]] = row
    return rows


def _load_baseline(model, corpus):
    return _load_jsonl_by_question(BASELINE_DIR / f"baseline_raw_{model}_{corpus}_vllm.jsonl")


def _load_attack(model, corpus):
    return _load_jsonl_by_question(ATTACK_DIR / f"poison_raw_{model}_{corpus}_adv5_hf.jsonl")


def main():
    model_keys = list(MODELS)
    corpora = list(ATTACK_ELIGIBLE_CORPORA)
    assert len(corpora) == 2, "corpus-vs-corpus comparison assumes exactly 2 corpora"

    cell_stats = {}
    asr_by_question = {}  # (model, corpus) -> {question: 0/1}

    for corpus in corpora:
        for model in model_keys:
            baseline = _load_baseline(model, corpus)
            attack = _load_attack(model, corpus)
            common_qs = [q for q in attack if q in baseline]
            if len(common_qs) != len(attack):
                print(
                    f"WARNING: {model}/{corpus}: {len(attack) - len(common_qs)} "
                    f"attack questions had no baseline match -- excluded from this cell's stats"
                )

            baseline_f1 = [baseline[q]["f1_clean"] for q in common_qs]
            attack_f1 = [attack[q]["f1_clean"] for q in common_qs]
            asr = [attack[q]["attack_success"] for q in common_qs]
            echoed = [attack[q]["poison_echoed_not_adopted"] for q in common_qs]
            contains_target = [attack[q]["contains_target_diagnostic"] for q in common_qs]

            boot = paired_bootstrap_ci(baseline_f1, attack_f1, n_boot=10000, seed=42)

            cell_stats[(model, corpus)] = {
                "n": len(common_qs),
                "asr": sum(asr) / len(asr) if asr else float("nan"),
                "echoed_not_adopted_rate": sum(echoed) / len(echoed) if echoed else float("nan"),
                "contains_target_rate": sum(contains_target) / len(contains_target) if contains_target else float("nan"),
                "f1_baseline": sum(baseline_f1) / len(baseline_f1),
                "f1_attack": sum(attack_f1) / len(attack_f1),
                "utility_drop": boot["observed_drop"],
                "utility_drop_ci_low": boot["ci_low"],
                "utility_drop_ci_high": boot["ci_high"],
            }
            asr_by_question[(model, corpus)] = dict(zip(common_qs, asr))

    print("=== Per-cell ASR, poison-echoed-not-adopted, and utility-under-attack (model x corpus) ===")
    header = (
        f"{'model':<16}{'corpus':<10}{'n':>5}{'ASR':>8}{'echoed':>8}{'contains':>9}"
        f"{'F1_base':>9}{'F1_atk':>8}{'drop':>8}{'ci_lo':>8}{'ci_hi':>8}"
    )
    print(header)
    for corpus in corpora:
        for model in model_keys:
            s = cell_stats[(model, corpus)]
            print(
                f"{model:<16}{corpus:<10}{s['n']:>5}{s['asr']:>8.4f}"
                f"{s['echoed_not_adopted_rate']:>8.4f}{s['contains_target_rate']:>9.4f}"
                f"{s['f1_baseline']:>9.4f}{s['f1_attack']:>8.4f}{s['utility_drop']:>8.4f}"
                f"{s['utility_drop_ci_low']:>8.4f}{s['utility_drop_ci_high']:>8.4f}"
            )

    print("\n=== McNemar (model-vs-model) / Fisher (corpus-vs-corpus) ===")
    print("=== Holm-Bonferroni corrected within the single 16-comparison family ===")
    family = []  # (label, p_value, extra_info_dict)

    for corpus in corpora:
        for m1, m2 in itertools.combinations(model_keys, 2):
            d1 = asr_by_question[(m1, corpus)]
            d2 = asr_by_question[(m2, corpus)]
            common = [q for q in d1 if q in d2]
            result = mcnemar_exact([d1[q] for q in common], [d2[q] for q in common])
            family.append((f"model {m1} vs {m2} | corpus {corpus}", result["p_value"], result))

    c1, c2 = corpora
    for model in model_keys:
        d1 = asr_by_question[(model, c1)]
        d2 = asr_by_question[(model, c2)]
        result = fisher_exact_asr_comparison(sum(d1.values()), len(d1), sum(d2.values()), len(d2))
        family.append((f"corpus {c1} vs {c2} | model {model}", result["p_value"], result))

    p_values = [p for _, p, _ in family]
    holm_results = holm_bonferroni(p_values, alpha=0.05)

    for (label, p, raw), holm in zip(family, holm_results):
        sig = "SIG" if holm["significant_holm"] else "ns"
        print(f"  {label:<45} p={p:.4g}  holm={sig}  detail={raw}")

    return cell_stats, list(zip(family, holm_results))


if __name__ == "__main__":
    main()
