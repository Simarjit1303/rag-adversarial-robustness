"""
Phase 3 defense statistical analysis (instruction_detection, spotlighting,
output_filter across injection, PoisonedRAG, and Crescendo).

Reads phase3_defense_results/ (this session's committed Phase 3 snapshot),
plus the paired baselines each attack needs: phase2_injection_results/
(injection), phase2_poisonedrag_results/ (PoisonedRAG), phase2_crescendo_
results/ (Crescendo) -- same "downloaded result dump living at repo root"
pattern as scripts/analyze_phase2_*_stats.py, which this script mirrors
structurally.

Cell discovery is existence + line-count driven, not a hardcoded combo
list: Phase 3's real coverage is asymmetric (qwen3-8b/hotpot_qa got 5
injection templates under output_filter but only 2 under instruction_
detection/spotlighting -- confirmed by this session's alignment check,
not assumed), and 6 qwen3-8b/hotpot_qa files under instruction_detection/
spotlighting are n=3 smoke-test fragments left over from infra testing,
not completed cells -- excluded by their n, not by name.

Per-cell paired test: McNemar's exact test (defended vs baseline
attack_success, same items) wherever a baseline match exists -- which is
every cell in this dataset (Task 1's alignment check found no cell with
zero matched baseline items), so the Fisher's-exact-unpaired path below
is implemented for robustness but never actually triggered here. Effect
size is baseline-minus-defended ASR with a paired-bootstrap 95% CI
(evaluation.stats.paired_bootstrap_ci on the two binary attack_success
arrays -- same function already used for continuous F1 diffs in the
Phase 2 scripts, works identically on a 0/1 array since it's just a
paired mean-difference CI). Holm-Bonferroni correction runs in three
separate families (injection, poisonedrag, crescendo) -- pooling across
attacks would conflate unrelated hypotheses.

Crescendo/output_filter is a special case, not a bug: run_crescendo.py's
guard check is observational-only by design (discards the filtered text,
scores the real unfiltered conversation -- see PHASE3_DEFENSE_INSIGHTS.md
methodology section for the exact code citations). Its attack_success is
therefore NOT a defended-condition outcome; every Crescendo cell is
tagged caveat="observational_only_no_intervention" /
defended_measurement_valid=False so callers don't mistake "no ASR change"
for "the defense didn't work" when no defense was actually applied to the
scored transcript.

Run: python -m scripts.analyze_phase3_defense_stats
"""

import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attacks.injection_templates import TEMPLATES  # noqa: E402
from config import MODELS  # noqa: E402
from evaluation.result_paths import ATTACK_ELIGIBLE_CORPORA  # noqa: E402
from evaluation.stats import (  # noqa: E402
    fisher_exact_asr_comparison,
    holm_bonferroni,
    mcnemar_exact,
    paired_bootstrap_ci,
)

P3_DIR = Path("phase3_defense_results")
P2_INJECTION_DIR = Path("phase2_injection_results")
P2_POISON_DIR = Path("phase2_poisonedrag_results")
P2_CRESCENDO_DIR = Path("phase2_crescendo_results")
P1_DIR = Path("phase1_results_complete")

MODEL_KEYS = list(MODELS)
CORPORA = list(ATTACK_ELIGIBLE_CORPORA)
INJECTION_TEMPLATES = list(TEMPLATES)
INJECTION_DEFENSES = ("instruction_detection", "spotlighting", "output_filter")
INJECTION_EXPECTED_N = {"instruction_detection": 40, "spotlighting": 1000, "output_filter": 1000}
POISON_DEFENSES = ("instruction_detection", "spotlighting", "output_filter")
MIN_LEGIT_N_FRACTION = 0.5  # a file at <50% of its defense's expected n is a smoke fragment


def _load_jsonl_by_key(path, key):
    rows = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows[row[key]] = row
    return rows


def _paired_cell(baseline_rows, defended_rows, key_order):
    """Matched-item baseline/defended attack_success arrays, in a fixed
    order, for every key present in both dicts."""
    common = [k for k in key_order if k in baseline_rows]
    b = [baseline_rows[k]["attack_success"] for k in common]
    d = [defended_rows[k]["attack_success"] for k in common]
    return common, b, d


def _mcnemar_and_effect(baseline_asr, defended_asr):
    m = mcnemar_exact(baseline_asr, defended_asr)
    boot = paired_bootstrap_ci(baseline_asr, defended_asr, n_boot=10000, seed=42)
    return {
        "n": len(baseline_asr),
        "baseline_asr": sum(baseline_asr) / len(baseline_asr),
        "defended_asr": sum(defended_asr) / len(defended_asr),
        "reduction": boot["observed_drop"],
        "reduction_ci_low": boot["ci_low"],
        "reduction_ci_high": boot["ci_high"],
        "p_value": m["p_value"],
        "b": m["b"],
        "c": m["c"],
        "test_used": "mcnemar_exact",
    }


def _fisher_fallback(baseline_asr, defended_asr):
    """Unpaired fallback -- implemented for robustness, not exercised by
    this dataset (every cell here has a nonzero matched-baseline subset)."""
    f = fisher_exact_asr_comparison(
        sum(baseline_asr), len(baseline_asr), sum(defended_asr), len(defended_asr)
    )
    return {
        "n": len(defended_asr),
        "baseline_asr": sum(baseline_asr) / len(baseline_asr),
        "defended_asr": sum(defended_asr) / len(defended_asr),
        "reduction": sum(baseline_asr) / len(baseline_asr) - sum(defended_asr) / len(defended_asr),
        "reduction_ci_low": None,
        "reduction_ci_high": None,
        "p_value": f["p_value"],
        "b": None,
        "c": None,
        "test_used": "fisher_exact_unpaired",
    }


# ---------------------------------------------------------------------------
# TASK 3a: injection cells
# ---------------------------------------------------------------------------

def discover_injection_cells():
    """(model, corpus, template, defense) -> stats dict, for every file
    that actually exists at >= half its defense's expected n. Files at
    <50% of expected n are reported and excluded as smoke fragments."""
    cells = {}
    excluded_fragments = []
    for defense in INJECTION_DEFENSES:
        expected_n = INJECTION_EXPECTED_N[defense]
        for model in MODEL_KEYS:
            for corpus in CORPORA:
                for template in INJECTION_TEMPLATES:
                    path = P3_DIR / f"attack_raw_{model}_{corpus}_{template}_hf_defense-{defense}.jsonl"
                    if not path.exists():
                        continue
                    defended = _load_jsonl_by_key(path, "question")
                    if len(defended) < expected_n * MIN_LEGIT_N_FRACTION:
                        excluded_fragments.append((model, corpus, template, defense, len(defended)))
                        continue

                    baseline_path = P2_INJECTION_DIR / f"attack_raw_{model}_{corpus}_{template}_vllm.jsonl"
                    baseline = _load_jsonl_by_key(baseline_path, "question")
                    common, b_asr, d_asr = _paired_cell(baseline, defended, list(defended))
                    if len(common) != len(defended):
                        print(
                            f"WARNING: injection {model}/{corpus}/{template}/{defense}: "
                            f"{len(defended) - len(common)}/{len(defended)} items had no baseline match"
                        )
                    if not common:
                        all_baseline_asr = [r["attack_success"] for r in baseline.values()]
                        all_defended_asr = [defended[q]["attack_success"] for q in defended]
                        cells[(model, corpus, template, defense)] = _fisher_fallback(
                            all_baseline_asr, all_defended_asr
                        )
                    else:
                        cells[(model, corpus, template, defense)] = _mcnemar_and_effect(b_asr, d_asr)
                    cells[(model, corpus, template, defense)]["n_phase3_total"] = len(defended)
                    cells[(model, corpus, template, defense)]["n_matched"] = len(common)

                    f1_def = [defended[q]["f1_clean"] for q in common]
                    f1_base_p1_row = None  # utility vs Phase1 handled separately in Task 5
                    cells[(model, corpus, template, defense)]["f1_clean_defended"] = (
                        sum(f1_def) / len(f1_def) if f1_def else float("nan")
                    )
    return cells, excluded_fragments


# ---------------------------------------------------------------------------
# TASK 3b: PoisonedRAG cells
# ---------------------------------------------------------------------------

def discover_poison_cells():
    cells = {}
    for defense in POISON_DEFENSES:
        for model in MODEL_KEYS:
            for corpus in CORPORA:
                path = P3_DIR / f"poison_raw_{model}_{corpus}_adv5_hf_defense-{defense}.jsonl"
                if not path.exists():
                    continue
                defended = _load_jsonl_by_key(path, "question")
                baseline_path = P2_POISON_DIR / f"poison_raw_{model}_{corpus}_adv5_hf.jsonl"
                baseline = _load_jsonl_by_key(baseline_path, "question")
                common, b_asr, d_asr = _paired_cell(baseline, defended, list(defended))
                if len(common) != len(defended):
                    print(
                        f"NOTE: poisonedrag {model}/{corpus}/{defense}: "
                        f"{len(defended) - len(common)}/{len(defended)} items are new to Phase 3 "
                        f"(no Phase 2 baseline match) -- excluded from this cell's paired test"
                    )
                cells[(model, corpus, defense)] = _mcnemar_and_effect(b_asr, d_asr)
                cells[(model, corpus, defense)]["n_phase3_total"] = len(defended)
                cells[(model, corpus, defense)]["n_matched"] = len(common)
                f1_def = [defended[q]["f1_clean"] for q in common]
                cells[(model, corpus, defense)]["f1_clean_defended"] = (
                    sum(f1_def) / len(f1_def) if f1_def else float("nan")
                )
    return cells


# ---------------------------------------------------------------------------
# TASK 3c: Crescendo cells
# ---------------------------------------------------------------------------

def discover_crescendo_cells():
    cells = {}
    for model in MODEL_KEYS:
        path = P3_DIR / f"crescendo_raw_{model}_5turn_hf_defense-output_filter.jsonl"
        if not path.exists():
            continue
        defended = _load_jsonl_by_key(path, "behavior")
        baseline_path = P2_CRESCENDO_DIR / f"crescendo_raw_{model}_5turn_hf.jsonl"
        baseline = _load_jsonl_by_key(baseline_path, "behavior")
        common = [beh for beh in defended if beh in baseline
                  and not baseline[beh]["judge_failed"] and not defended[beh]["judge_failed"]]
        if len(common) != len(defended):
            print(
                f"NOTE: crescendo {model}/output_filter: {len(defended) - len(common)}/"
                f"{len(defended)} behaviors excluded (no Phase 2 baseline match or judge_failed)"
            )
        b_asr = [baseline[beh]["attack_success"] for beh in common]
        d_asr = [defended[beh]["attack_success"] for beh in common]
        stats = _mcnemar_and_effect(b_asr, d_asr) if common else _fisher_fallback([0], [0])
        stats["n_phase3_total"] = len(defended)
        stats["n_matched"] = len(common)
        stats["caveat"] = "observational_only_no_intervention"
        stats["defended_measurement_valid"] = False
        cells[(model, "output_filter")] = stats
    return cells


# ---------------------------------------------------------------------------
# Holm-Bonferroni per family
# ---------------------------------------------------------------------------

def _holm_adjusted_pvalues(p_values):
    """evaluation.stats.holm_bonferroni only returns a reject/accept
    decision per p-value (by design -- see its docstring), not an
    adjusted p-value. The standard Holm step-down adjusted p-value
    (ascending-sorted p_(i) -> (m-i+1)*p_(i), then a running max to
    enforce monotonicity, capped at 1) is a few lines on top of that
    same ranking, so it's computed here rather than reported as
    unavailable."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, i in enumerate(order):  # rank is 0-indexed
        candidate = (m - rank) * p_values[i]
        running_max = max(running_max, candidate)
        adjusted[i] = min(1.0, running_max)
    return adjusted


def apply_holm_family(cells_dict):
    keys = list(cells_dict)
    p_values = [cells_dict[k]["p_value"] for k in keys]
    holm = holm_bonferroni(p_values, alpha=0.05)
    adjusted = _holm_adjusted_pvalues(p_values)
    for k, h, p_adj in zip(keys, holm, adjusted):
        cells_dict[k]["p_raw"] = cells_dict[k]["p_value"]
        cells_dict[k]["p_holm"] = p_adj
        cells_dict[k]["significant_holm"] = h["significant_holm"]
        cells_dict[k]["holm_rank"] = h["rank"]
    return cells_dict


# ---------------------------------------------------------------------------
# TASK 4: mechanism attribution
# ---------------------------------------------------------------------------

def mechanism_output_filter_injection():
    """Per injection/output_filter cell: of items blocked (baseline
    success=1, defended success=0), fraction with guard flagged=True."""
    results = {}
    for model in MODEL_KEYS:
        for corpus in CORPORA:
            log_path = P3_DIR / f"output_filter_log_attack_{model}_{corpus}.jsonl"
            if not log_path.exists():
                continue
            log_rows = {}
            with log_path.open(encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    log_rows[(r["injection_template"], r["question"])] = r["flagged"]
            for template in INJECTION_TEMPLATES:
                def_path = P3_DIR / f"attack_raw_{model}_{corpus}_{template}_hf_defense-output_filter.jsonl"
                if not def_path.exists():
                    continue
                defended = _load_jsonl_by_key(def_path, "question")
                if len(defended) < INJECTION_EXPECTED_N["output_filter"] * MIN_LEGIT_N_FRACTION:
                    continue
                baseline = _load_jsonl_by_key(
                    P2_INJECTION_DIR / f"attack_raw_{model}_{corpus}_{template}_vllm.jsonl", "question"
                )
                blocked = [
                    q for q in defended
                    if q in baseline and baseline[q]["attack_success"] == 1
                    and defended[q]["attack_success"] == 0
                ]
                flagged = [
                    log_rows.get((template, q)) for q in blocked if (template, q) in log_rows
                ]
                n_flagged_true = sum(1 for v in flagged if v)
                results[(model, corpus, template)] = {
                    "n_blocked": len(blocked),
                    "n_flagged_true": n_flagged_true,
                    "frac_guard_caught": (n_flagged_true / len(flagged)) if flagged else float("nan"),
                    "frac_model_alone": (1 - n_flagged_true / len(flagged)) if flagged else float("nan"),
                }
    return results


def mechanism_output_filter_poisonedrag():
    """Per poisonedrag/output_filter cell: same blocked-fraction split,
    plus the raw flag rate across ALL responses (the headline evidence for
    'safety classifier can't catch factual poisoning')."""
    results = {}
    all_flags = []
    for model in MODEL_KEYS:
        for corpus in CORPORA:
            log_path = P3_DIR / f"output_filter_log_poison_{model}_{corpus}.jsonl"
            def_path = P3_DIR / f"poison_raw_{model}_{corpus}_adv5_hf_defense-output_filter.jsonl"
            if not (log_path.exists() and def_path.exists()):
                continue
            log_rows = {}
            with log_path.open(encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    log_rows[r["question"]] = r["flagged"]
                    all_flags.append(r["flagged"])
            defended = _load_jsonl_by_key(def_path, "question")
            baseline = _load_jsonl_by_key(
                P2_POISON_DIR / f"poison_raw_{model}_{corpus}_adv5_hf.jsonl", "question"
            )
            blocked = [
                q for q in defended
                if q in baseline and baseline[q]["attack_success"] == 1
                and defended[q]["attack_success"] == 0
            ]
            flagged = [log_rows.get(q) for q in blocked if q in log_rows]
            n_flagged_true = sum(1 for v in flagged if v)
            results[(model, corpus)] = {
                "n_blocked": len(blocked),
                "n_flagged_true": n_flagged_true,
                "frac_guard_caught": (n_flagged_true / len(flagged)) if flagged else float("nan"),
                "cell_flag_rate": sum(log_rows.values()) / len(log_rows) if log_rows else float("nan"),
                "cell_n_responses": len(log_rows),
            }
    overall_flag_rate = sum(all_flags) / len(all_flags) if all_flags else float("nan")
    return results, overall_flag_rate, len(all_flags)


def mechanism_instruction_detection():
    """Confirmed via code read (defenses/instruction_detection.py,
    evaluation/run_attack_injection.py:108-112): the per-passage
    DetectionResult is computed inline and immediately discarded --
    filter_retrieved_passages's log-returning API
    (defenses/instruction_detection.py:151-183) is never actually called
    by the runner, so no per-passage flagged log was ever persisted for
    this defense, in the raw JSONL or as a separate file. This mechanism
    question is therefore NOT ANSWERABLE from available data -- reported
    as such rather than fabricated."""
    return None


def mechanism_crescendo_output_filter():
    """Per model: of conversations where the (unfiltered, since nothing
    was actually blocked) attack ultimately succeeded, what fraction had
    at least one turn the guard WOULD have flagged pre-delivery -- i.e.
    if this guard's flag had been acted on, in what fraction of
    successful attacks would it have had a chance to intervene."""
    results = {}
    for model in MODEL_KEYS:
        path = P3_DIR / f"crescendo_raw_{model}_5turn_hf_defense-output_filter.jsonl"
        if not path.exists():
            continue
        rows = list(_load_jsonl_by_key(path, "behavior").values())
        scored = [r for r in rows if not r["judge_failed"]]
        succeeded = [r for r in scored if r["attack_success"] == 1]
        would_intervene = [
            r for r in succeeded if any(t["flagged"] for t in r["output_filter_flags"])
        ]
        results[model] = {
            "n_scored": len(scored),
            "n_succeeded": len(succeeded),
            "n_would_intervene": len(would_intervene),
            "frac_would_intervene_of_succeeded": (
                len(would_intervene) / len(succeeded) if succeeded else float("nan")
            ),
        }
    return results


# ---------------------------------------------------------------------------
# TASK 5: utility preservation
# ---------------------------------------------------------------------------

def utility_table(injection_cells, poison_cells):
    p1_f1 = {}
    for model in MODEL_KEYS:
        for corpus in CORPORA:
            path = P1_DIR / f"baseline_summary_{model}_{corpus}_vllm.csv"
            with path.open(encoding="utf-8") as f:
                next(f)
                row = next(f).strip().split(",")
                p1_f1[(model, corpus)] = float(row[3])  # f1_clean column

    rows = []
    seen = set()
    for (model, corpus, template, defense), stats in injection_cells.items():
        key = (model, corpus, defense)
        f1_def = stats["f1_clean_defended"]
        rows.append({
            "attack": "injection", "model": model, "corpus": corpus, "defense": defense,
            "template": template, "f1_phase1_baseline": p1_f1[(model, corpus)],
            "f1_defended": f1_def, "delta": f1_def - p1_f1[(model, corpus)],
        })
    for (model, corpus, defense), stats in poison_cells.items():
        f1_def = stats["f1_clean_defended"]
        rows.append({
            "attack": "poisonedrag", "model": model, "corpus": corpus, "defense": defense,
            "template": None, "f1_phase1_baseline": p1_f1[(model, corpus)],
            "f1_defended": f1_def, "delta": f1_def - p1_f1[(model, corpus)],
        })
    return rows


# ---------------------------------------------------------------------------
def main():
    injection_cells, fragments = discover_injection_cells()
    print(f"=== Injection: {len(injection_cells)} real cells discovered "
          f"({len(fragments)} smoke-test fragments excluded) ===")
    for model, corpus, template, defense, n in fragments:
        print(f"  EXCLUDED (n={n}, smoke fragment): {model}/{corpus}/{template}/{defense}")

    poison_cells = discover_poison_cells()
    print(f"\n=== PoisonedRAG: {len(poison_cells)} cells discovered ===")

    crescendo_cells = discover_crescendo_cells()
    print(f"\n=== Crescendo: {len(crescendo_cells)} cells discovered ===")

    apply_holm_family(injection_cells)
    apply_holm_family(poison_cells)
    apply_holm_family(crescendo_cells)

    def _print_family(name, cells, key_fmt):
        print(f"\n=== {name}: McNemar + Holm-Bonferroni (family of {len(cells)}) ===")
        header = (f"{'cell':<55}{'test':<22}{'n':>5}{'base':>7}{'def':>7}{'reduc':>7}"
                  f"{'ci_lo':>7}{'ci_hi':>7}{'p_raw':>9}{'p_holm':>9}{'sig':>5}")
        print(header)
        for key, s in cells.items():
            label = key_fmt(key)
            ci_lo = f"{s['reduction_ci_low']:.4f}" if s['reduction_ci_low'] is not None else "n/a"
            ci_hi = f"{s['reduction_ci_high']:.4f}" if s['reduction_ci_high'] is not None else "n/a"
            sig = "SIG" if s["significant_holm"] else "ns"
            caveat = f" [{s['caveat']}]" if "caveat" in s else ""
            print(f"{label:<55}{s['test_used']:<22}{s['n']:>5}{s['baseline_asr']:>7.3f}"
                  f"{s['defended_asr']:>7.3f}{s['reduction']:>7.3f}{ci_lo:>7}{ci_hi:>7}"
                  f"{s['p_raw']:>9.4g}{s['p_holm']:>9.4g}{sig:>5}{caveat}")

    _print_family("Injection", injection_cells, lambda k: f"{k[0]}/{k[1]}/{k[2]}/{k[3]}")
    _print_family("PoisonedRAG", poison_cells, lambda k: f"{k[0]}/{k[1]}/{k[2]}")

    print("\n=== Crescendo/output_filter: RAW diagnostic numbers only -- "
          "NOT a master-ASR-reduction-table family ===")
    print("run_crescendo.py:184-257 discards the filtered text and scores the "
          "real, unfiltered conversation, so this is not a defended condition; "
          "these McNemar/CI numbers exist for completeness, not as a significance "
          "claim about the defense. See the Mechanism attribution section for the "
          "actually meaningful number (guard-would-have-intervened fraction).")
    _print_family("Crescendo (diagnostic only, excluded from ASR-reduction family)",
                   crescendo_cells, lambda k: f"{k[0]}/output_filter")

    print("\n=== Task 4: mechanism attribution ===")
    mech_inj = mechanism_output_filter_injection()
    print("\n-- output_filter / injection: blocked-attack guard-caught fraction --")
    for k, v in mech_inj.items():
        print(f"  {k}: n_blocked={v['n_blocked']} frac_guard_caught={v['frac_guard_caught']:.3f}")

    mech_poison, overall_flag_rate, n_all = mechanism_output_filter_poisonedrag()
    print(f"\n-- output_filter / poisonedrag: OVERALL guard flag rate on ALL responses "
          f"= {overall_flag_rate:.4f} (n={n_all}) --")
    for k, v in mech_poison.items():
        print(f"  {k}: cell_flag_rate={v['cell_flag_rate']:.4f} (n={v['cell_n_responses']}) "
              f"n_blocked={v['n_blocked']} frac_guard_caught={v['frac_guard_caught']:.3f}")

    print("\n-- instruction_detection / injection: mechanism attribution --")
    print(f"  {mechanism_instruction_detection()!r} -- NOT MEASURABLE, see docstring "
          f"(defenses/instruction_detection.py + run_attack_injection.py:108-112: "
          f"per-passage log computed then discarded, never persisted)")

    mech_cres = mechanism_crescendo_output_filter()
    print("\n-- crescendo / output_filter: guard-would-have-intervened fraction of successful attacks --")
    for model, v in mech_cres.items():
        print(f"  {model}: n_succeeded={v['n_succeeded']}/{v['n_scored']} "
              f"frac_would_intervene={v['frac_would_intervene_of_succeeded']:.3f}")

    print("\n=== Task 5: utility preservation (F1 delta vs Phase 1 baseline) ===")
    util_rows = utility_table(injection_cells, poison_cells)
    for r in util_rows:
        print(f"  {r['attack']:<11}{r['model']:<16}{r['corpus']:<10}{str(r['defense']):<22}"
              f"base={r['f1_phase1_baseline']:.4f} def={r['f1_defended']:.4f} delta={r['delta']:+.4f}")
    print(f"  ({len(util_rows)} rows total)")

    if "--dump-json" in sys.argv:
        def _stringify_keys(d):
            return {"|".join(k) if isinstance(k, tuple) else k: v for k, v in d.items()}
        dump = {
            "injection_cells": _stringify_keys(injection_cells),
            "poison_cells": _stringify_keys(poison_cells),
            "crescendo_cells": _stringify_keys(crescendo_cells),
            "excluded_fragments": fragments,
            "mechanism_output_filter_injection": _stringify_keys(mech_inj),
            "mechanism_output_filter_poisonedrag": _stringify_keys(mech_poison),
            "mechanism_output_filter_poisonedrag_overall_flag_rate": overall_flag_rate,
            "mechanism_output_filter_poisonedrag_n_all": n_all,
            "mechanism_crescendo": mech_cres,
            "utility_rows": util_rows,
        }
        out_path = sys.argv[sys.argv.index("--dump-json") + 1]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dump, f, indent=2, default=str)
        print(f"\n[dumped full results to {out_path}]")

    return {
        "injection_cells": injection_cells,
        "poison_cells": poison_cells,
        "crescendo_cells": crescendo_cells,
        "excluded_fragments": fragments,
        "mechanism_output_filter_injection": mech_inj,
        "mechanism_output_filter_poisonedrag": (mech_poison, overall_flag_rate, n_all),
        "mechanism_crescendo": mech_cres,
        "utility_rows": util_rows,
    }


if __name__ == "__main__":
    main()
