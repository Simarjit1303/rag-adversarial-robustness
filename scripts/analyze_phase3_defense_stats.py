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

Backend-confound isolation (discover_backend_confound_cells): for the 35
output_filter/spotlighting cells, a matched 3-way comparison -- vllm
baseline, a backend-matched hf no-defense baseline (attack_raw_*_hf.jsonl,
RAG_DEFENSE=none, same item selection verified by
scripts/verify_phase3_backend_baseline_items.py), and the hf-defended run
-- isolates how much of each cell's ASR reduction is the hf/vllm backend
switch versus the defense itself. Uses pairwise McNemar (3 comparisons per
cell: vllm-vs-nodef, nodef-vs-def, vllm-vs-def) with Holm-Bonferroni
correction WITHIN each cell's own 3-comparison family, not Cochran's Q:
the question this analysis needs answered is "which pair differs" (to
attribute the reduction to backend vs defense), not merely "do the three
conditions differ somewhere" -- Cochran's Q gives only the latter and
would still require a post-hoc pairwise test to attribute the effect, so
pairwise McNemar is the more direct fit here, not just the cheaper one to
reuse from the existing McNemar/Holm helpers already used everywhere else
in this file.

instruction_detection's mechanism attribution (mechanism_instruction_
detection) was a genuine gap in the original analysis -- the runner
computed each passage's DetectionResult inline and discarded it, so no
per-passage flagged/not-flagged log existed. This session's rerun (see
PHASE3_DEFENSE_INSIGHTS.md's Task C) persists that log to
instruction_detection_log_attack_{model}_{corpus}.jsonl; this function now
answers the original question directly instead of reporting it as
unanswerable.

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
# Backend-confound isolation: vllm-baseline vs hf-no-defense-baseline vs
# hf-defended, per matched cell, for output_filter and spotlighting.
# ---------------------------------------------------------------------------

BACKEND_CONFOUND_DEFENSES = ("output_filter", "spotlighting", "instruction_detection")
# instruction_detection's cells are n=40, not n=1000 -- but the same
# nodef_path (attack_raw_*_hf.jsonl, Task B's RAG_SAMPLE_N=1000 sweep) still
# works as its backend-matched no-defense baseline with ZERO new GPU work:
# scripts/verify_phase3_instruction_detection_baseline_items.py confirmed
# (2026-09-16, no model/GPU touched) that all 16 real instruction_detection
# cells' n=40 items are byte-identical, in order, to the first 40 rows of
# the already-committed Task B file for that (model, corpus, template) --
# expected, since evaluation/run_attack_injection.py's hf-engine item-
# selection loop depends only on (corpus, split), never on sample_n itself
# beyond when to stop, nor on defense. The `common = [q for q in defended
# if q in nodef and q in vllm]` key-intersection below already restricts
# the 1000-row nodef file down to the matching 40 questions automatically
# -- no separate n=40 file or new sweep was needed.
PAIR_LABELS = ("vllm_vs_hf_nodef", "hf_nodef_vs_hf_def", "vllm_vs_hf_def")


def _three_way_verdict(sig, vllm_asr, nodef_asr, def_asr):
    """Classify a cell's three-condition pattern using the Holm-corrected
    significance of its 3 pairwise McNemar tests (see PAIR_LABELS order).

    - defense_works: vllm and hf-no-defense agree (backend switch alone
      changes nothing), but the defended condition is significantly lower
      than both -- the defense is the real driver.
    - backend_confound: hf-no-defense and hf-defended agree (the defense
      adds nothing beyond the backend switch), but vllm is significantly
      higher than both -- the backend switch is the real driver.
    - partial_split: vllm-vs-nodef AND nodef-vs-def are both significant --
      both the backend switch and the defense independently move the ASR;
      quantified separately via backend_fraction below.
    - inconclusive: no pairwise comparison reached significance (usually a
      near-zero-ASR cell with too little room for any effect to show)."""
    sig_vllm_nodef, sig_nodef_def, sig_vllm_def = sig
    if not sig_vllm_nodef and sig_nodef_def:
        return "defense_works"
    if sig_vllm_nodef and not sig_nodef_def:
        return "backend_confound"
    if sig_vllm_nodef and sig_nodef_def:
        return "partial_split"
    return "inconclusive"


def discover_backend_confound_cells():
    """Per (defense, model, corpus, template) cell with all three of
    {vllm baseline, hf no-defense baseline, hf defended} present: matched
    3-way ASR comparison via 3 pairwise McNemar tests (Holm-corrected
    within the cell's own 3-comparison family, per the task's own choice
    of pairwise-McNemar-with-Holm over Cochran's Q -- justified in the
    module docstring's Task 2 note) plus a backend-vs-defense attribution
    split of the total baseline-to-defended reduction."""
    cells = {}
    for defense in BACKEND_CONFOUND_DEFENSES:
        expected_n = INJECTION_EXPECTED_N[defense]
        for model in MODEL_KEYS:
            for corpus in CORPORA:
                for template in INJECTION_TEMPLATES:
                    def_path = P3_DIR / f"attack_raw_{model}_{corpus}_{template}_hf_defense-{defense}.jsonl"
                    nodef_path = P3_DIR / f"attack_raw_{model}_{corpus}_{template}_hf.jsonl"
                    vllm_path = P2_INJECTION_DIR / f"attack_raw_{model}_{corpus}_{template}_vllm.jsonl"
                    if not (def_path.exists() and nodef_path.exists() and vllm_path.exists()):
                        continue
                    defended = _load_jsonl_by_key(def_path, "question")
                    if len(defended) < expected_n * MIN_LEGIT_N_FRACTION:
                        continue  # smoke-test fragment, same exclusion rule as discover_injection_cells
                    nodef = _load_jsonl_by_key(nodef_path, "question")
                    vllm = _load_jsonl_by_key(vllm_path, "question")

                    common = [q for q in defended if q in nodef and q in vllm]
                    if not common:
                        continue
                    vllm_asr_arr = [vllm[q]["attack_success"] for q in common]
                    nodef_asr_arr = [nodef[q]["attack_success"] for q in common]
                    def_asr_arr = [defended[q]["attack_success"] for q in common]

                    pairs = {
                        "vllm_vs_hf_nodef": (vllm_asr_arr, nodef_asr_arr),
                        "hf_nodef_vs_hf_def": (nodef_asr_arr, def_asr_arr),
                        "vllm_vs_hf_def": (vllm_asr_arr, def_asr_arr),
                    }
                    pair_stats = {}
                    p_values = []
                    for label in PAIR_LABELS:
                        a, b = pairs[label]
                        m = mcnemar_exact(a, b)
                        pair_stats[label] = {"p_raw": m["p_value"], "b": m["b"], "c": m["c"]}
                        p_values.append(m["p_value"])
                    holm = _holm_adjusted_pvalues(p_values)
                    sig = []
                    for label, p_holm in zip(PAIR_LABELS, holm):
                        pair_stats[label]["p_holm"] = p_holm
                        is_sig = p_holm < 0.05
                        pair_stats[label]["significant_holm"] = is_sig
                        sig.append(is_sig)

                    vllm_asr = sum(vllm_asr_arr) / len(vllm_asr_arr)
                    nodef_asr = sum(nodef_asr_arr) / len(nodef_asr_arr)
                    def_asr = sum(def_asr_arr) / len(def_asr_arr)
                    total_reduction = vllm_asr - def_asr
                    backend_component = vllm_asr - nodef_asr
                    defense_component = nodef_asr - def_asr
                    backend_fraction = (
                        backend_component / total_reduction if total_reduction != 0 else float("nan")
                    )

                    cells[(defense, model, corpus, template)] = {
                        "n": len(common),
                        "vllm_asr": vllm_asr,
                        "hf_nodef_asr": nodef_asr,
                        "hf_def_asr": def_asr,
                        "total_reduction": total_reduction,
                        "backend_component": backend_component,
                        "defense_component": defense_component,
                        "backend_fraction_of_reduction": backend_fraction,
                        "pairs": pair_stats,
                        "verdict": _three_way_verdict(sig, vllm_asr, nodef_asr, def_asr),
                        # raw paired arrays, kept for corrected_backend_isolated_table's
                        # bootstrap CI (paired_bootstrap_ci needs the full arrays, not
                        # just the aggregated ASR/discordant-pair counts above)
                        "_nodef_asr_arr": nodef_asr_arr,
                        "_def_asr_arr": def_asr_arr,
                    }
    return cells


def corrected_backend_isolated_table(confound_cells):
    """The master-ASR-reduction-table-equivalent for output_filter and
    spotlighting/injection, using the backend-matched hf no-defense
    baseline (Task 2) as the comparator instead of the original vllm
    baseline. This is the corrected reporting the task asks for once the
    backend confound is confirmed for output_filter (see headline finding
    2's resolution) -- reusing vllm as "baseline" for those cells would
    keep reporting a reduction that's mostly the backend switch, not the
    defense. Holm-Bonferroni is re-applied here in two fresh families (19
    output_filter cells, 16 spotlighting cells) -- distinct from the
    per-cell 3-pairwise-comparison Holm family used for confound
    attribution above, which answers a different question (which pair
    differs) than this one (is the backend-isolated defense effect
    significant, across the family of cells for that defense)."""
    by_defense = {"output_filter": {}, "spotlighting": {}, "instruction_detection": {}}
    for (defense, model, corpus, template), v in confound_cells.items():
        stats = _mcnemar_and_effect(v["_nodef_asr_arr"], v["_def_asr_arr"])
        by_defense[defense][(model, corpus, template)] = stats
    for defense_cells in by_defense.values():
        apply_holm_family(defense_cells)
    return by_defense


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


def mechanism_instruction_detection(baseline="vllm"):
    """Per (model, corpus, template) cell: of items blocked by
    instruction_detection, what fraction had >=1 retrieved passage actually
    flagged=True by the classifier (context was actually stripped pre-
    generation) vs. zero passages flagged (the model resisted the attack on
    its own, the classifier never fired for that item).

    baseline="vllm" (default): "blocked" = the original Phase 2 vllm
    baseline succeeded and the hf-defended run failed -- same definition
    as mechanism_output_filter_*, and the one the 20.0%-overall headline
    number uses.

    baseline="hf_nodef": "blocked" = the backend-matched hf no-defense
    baseline (Task B's attack_raw_*_hf.jsonl, restricted to the matching
    40 items) succeeded and the hf-defended run failed. This is the
    backend-corrected version of the same question -- cross-referencing it
    against the vllm-baseline result answers whether the classifier's
    observed 20.0% catch rate holds up once the hf/vllm backend switch
    itself is no longer conflated into "blocked," or whether some of those
    "blocks" were really just the backend switch and evaporate here.

    Now answerable: defenses/instruction_detection.py's
    log_passage_detection_event and evaluation/run_attack_injection.py's
    _build_defended_attack_prompt call sites (this session's earlier fix)
    persist a per-passage {"flagged": bool, "score": float, "label": str}
    list to instruction_detection_log_attack_{model}_{corpus}.jsonl,
    keyed by (injection_template, question) -- the same log-file grain
    the Task C rerun produced. Previously this returned None (see the
    original gap notice this function replaces): the runner computed
    DetectionResult inline and discarded it without persisting anything."""
    results = {}
    for model in MODEL_KEYS:
        for corpus in CORPORA:
            log_path = P3_DIR / f"instruction_detection_log_attack_{model}_{corpus}.jsonl"
            if not log_path.exists():
                continue
            log_rows = {}
            with log_path.open(encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    log_rows[(r["injection_template"], r["question"])] = [
                        p["flagged"] for p in r["passages"]
                    ]
            for template in INJECTION_TEMPLATES:
                def_path = P3_DIR / f"attack_raw_{model}_{corpus}_{template}_hf_defense-instruction_detection.jsonl"
                if not def_path.exists():
                    continue
                defended = _load_jsonl_by_key(def_path, "question")
                if len(defended) < INJECTION_EXPECTED_N["instruction_detection"] * MIN_LEGIT_N_FRACTION:
                    continue  # smoke-test fragment
                if baseline == "hf_nodef":
                    baseline_path = P3_DIR / f"attack_raw_{model}_{corpus}_{template}_hf.jsonl"
                else:
                    baseline_path = P2_INJECTION_DIR / f"attack_raw_{model}_{corpus}_{template}_vllm.jsonl"
                baseline_rows = _load_jsonl_by_key(baseline_path, "question")
                blocked = [
                    q for q in defended
                    if q in baseline_rows and baseline_rows[q]["attack_success"] == 1
                    and defended[q]["attack_success"] == 0
                ]
                any_flagged = [
                    any(log_rows[(template, q)]) for q in blocked if (template, q) in log_rows
                ]
                n_any_flagged = sum(1 for v in any_flagged if v)
                results[(model, corpus, template)] = {
                    "n_blocked": len(blocked),
                    "n_any_flagged": n_any_flagged,
                    "frac_any_passage_flagged": (n_any_flagged / len(any_flagged)) if any_flagged else float("nan"),
                    "frac_model_resisted_alone": (1 - n_any_flagged / len(any_flagged)) if any_flagged else float("nan"),
                }
    return results


def mechanism_instruction_detection_by_model_corpus(per_cell):
    """Roll-up of mechanism_instruction_detection()'s per-template rows to
    per (model, corpus) -- the log file's own grain (both templates share
    one log file) and the granularity Task 3 explicitly asked for."""
    rollup = {}
    for (model, corpus, template), v in per_cell.items():
        key = (model, corpus)
        agg = rollup.setdefault(key, {"n_blocked": 0, "n_any_flagged": 0})
        agg["n_blocked"] += v["n_blocked"]
        agg["n_any_flagged"] += v["n_any_flagged"]
    for key, agg in rollup.items():
        agg["frac_any_passage_flagged"] = (
            agg["n_any_flagged"] / agg["n_blocked"] if agg["n_blocked"] else float("nan")
        )
    return rollup


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

    mech_id = mechanism_instruction_detection()
    mech_id_rollup = mechanism_instruction_detection_by_model_corpus(mech_id)
    mech_id_corrected = mechanism_instruction_detection(baseline="hf_nodef")
    mech_id_corrected_rollup = mechanism_instruction_detection_by_model_corpus(mech_id_corrected)
    print("\n-- instruction_detection / injection: mechanism attribution (per model/corpus) --")
    for (model, corpus), v in mech_id_rollup.items():
        print(f"  {model}/{corpus}: n_blocked={v['n_blocked']} n_any_flagged={v['n_any_flagged']} "
              f"frac_any_passage_flagged={v['frac_any_passage_flagged']:.3f}")
    print("  -- backend-corrected (hf-nodef baseline instead of vllm) --")
    for (model, corpus), v in mech_id_corrected_rollup.items():
        print(f"  {model}/{corpus}: n_blocked={v['n_blocked']} n_any_flagged={v['n_any_flagged']} "
              f"frac_any_passage_flagged={v['frac_any_passage_flagged']:.3f}")
    print("  -- per (model, corpus, template) --")
    for k, v in mech_id.items():
        print(f"  {k}: n_blocked={v['n_blocked']} frac_any_passage_flagged={v['frac_any_passage_flagged']:.3f}")

    print("\n=== Task 2: backend-confound isolation (vllm baseline vs hf-no-defense vs hf-defended) ===")
    confound_cells = discover_backend_confound_cells()
    print(f"  {len(confound_cells)} 3-way matched cells (output_filter + spotlighting + instruction_detection)")
    header = (f"{'cell':<60}{'n':>5}{'vllm':>7}{'nodef':>7}{'def':>7}{'tot_red':>8}"
              f"{'bknd_frac':>10}{'verdict':>18}")
    print(header)
    for key, s in confound_cells.items():
        label = f"{key[0]}/{key[1]}/{key[2]}/{key[3]}"
        bf = f"{s['backend_fraction_of_reduction']:.3f}" if s['backend_fraction_of_reduction'] == s['backend_fraction_of_reduction'] else "n/a"
        print(f"{label:<60}{s['n']:>5}{s['vllm_asr']:>7.3f}{s['hf_nodef_asr']:>7.3f}"
              f"{s['hf_def_asr']:>7.3f}{s['total_reduction']:>8.3f}{bf:>10}{s['verdict']:>18}")
    verdict_counts = {}
    for s in confound_cells.values():
        verdict_counts[s["verdict"]] = verdict_counts.get(s["verdict"], 0) + 1
    print(f"  Verdict counts: {verdict_counts}")

    print("\n=== Task 2 (corrected table): backend-isolated defense effect "
          "(hf-no-defense baseline vs hf-defended, NOT vllm) ===")
    corrected = corrected_backend_isolated_table(confound_cells)
    for defense, cells in corrected.items():
        print(f"\n-- {defense}: corrected master-table rows (family of {len(cells)}) --")
        for (model, corpus, template), s in cells.items():
            sig = "SIG" if s["significant_holm"] else "ns"
            print(f"  {model}/{corpus}/{template}: n={s['n']} nodef_baseline={s['baseline_asr']:.3f} "
                  f"defended={s['defended_asr']:.3f} reduction={s['reduction']:.3f} "
                  f"p_holm={s['p_holm']:.4g} {sig}")

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
            "mechanism_instruction_detection": _stringify_keys(mech_id),
            "mechanism_instruction_detection_by_model_corpus": _stringify_keys(mech_id_rollup),
            "mechanism_instruction_detection_backend_corrected": _stringify_keys(mech_id_corrected),
            "mechanism_instruction_detection_backend_corrected_by_model_corpus": _stringify_keys(mech_id_corrected_rollup),
            "backend_confound_cells": _stringify_keys(
                {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in confound_cells.items()}
            ),
            "backend_confound_verdict_counts": verdict_counts,
            "corrected_backend_isolated_table": {
                defense: _stringify_keys(cells) for defense, cells in corrected.items()
            },
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
        "mechanism_instruction_detection": mech_id,
        "mechanism_instruction_detection_by_model_corpus": mech_id_rollup,
        "mechanism_instruction_detection_backend_corrected": mech_id_corrected,
        "mechanism_instruction_detection_backend_corrected_by_model_corpus": mech_id_corrected_rollup,
        "backend_confound_cells": confound_cells,
        "corrected_backend_isolated_table": corrected,
        "utility_rows": util_rows,
    }


if __name__ == "__main__":
    main()
