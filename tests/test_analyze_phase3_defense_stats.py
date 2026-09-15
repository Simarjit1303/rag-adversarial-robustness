"""
Unit tests for scripts/analyze_phase3_defense_stats.py's non-trivial logic:
the Holm step-down adjusted-p-value helper (a from-scratch implementation,
since evaluation.stats.holm_bonferroni only returns a reject/accept
decision -- see that module's own test file for the reasoning), the
matched-item pairing helper, and the smoke-test-fragment exclusion
threshold in cell discovery (the actual "alignment-checking" logic this
script adds on top of Task 1's manual findings).
"""

import json

import pytest

from scripts.analyze_phase3_defense_stats import (
    _holm_adjusted_pvalues,
    _mcnemar_and_effect,
    _paired_cell,
    _three_way_verdict,
    corrected_backend_isolated_table,
    discover_backend_confound_cells,
    discover_injection_cells,
    mechanism_instruction_detection,
    mechanism_instruction_detection_by_model_corpus,
)


# --------------------------------------------------------------------------
# _holm_adjusted_pvalues
# --------------------------------------------------------------------------

def test_holm_adjusted_pvalues_classic_textbook_case():
    # same case as tests/test_stats.py's step-down test: sorted 0.01, 0.03,
    # 0.04 -> multipliers 3, 2, 1 -> raw candidates 0.03, 0.06, 0.04 -> after
    # the running max for monotonicity: 0.03, 0.06, 0.06
    p_values = [0.01, 0.04, 0.03]  # unsorted on purpose, same as stats test
    adjusted = _holm_adjusted_pvalues(p_values)
    assert adjusted[0] == pytest.approx(0.03)  # 0.01 -> rank 1 of 3 -> *3
    assert adjusted[2] == pytest.approx(0.06)  # 0.03 -> rank 2 of 3 -> *2
    assert adjusted[1] == pytest.approx(0.06)  # 0.04 -> rank 3 of 3 -> *1, but
    # monotonicity forces it up to the previous (larger) adjusted value


def test_holm_adjusted_pvalues_enforces_monotonicity():
    p_values = [0.001, 0.002, 0.003]
    adjusted = _holm_adjusted_pvalues(p_values)
    assert adjusted[0] <= adjusted[1] <= adjusted[2]


def test_holm_adjusted_pvalues_capped_at_one():
    p_values = [0.9, 0.95]
    adjusted = _holm_adjusted_pvalues(p_values)
    assert all(p <= 1.0 for p in adjusted)


def test_holm_adjusted_pvalues_single_value_unchanged():
    assert _holm_adjusted_pvalues([0.03]) == [0.03]


def test_holm_adjusted_pvalues_empty_input():
    assert _holm_adjusted_pvalues([]) == []


# --------------------------------------------------------------------------
# _paired_cell
# --------------------------------------------------------------------------

def test_paired_cell_keeps_only_items_present_in_baseline():
    baseline = {"q1": {"attack_success": 1}, "q2": {"attack_success": 0}}
    defended = {"q1": {"attack_success": 0}, "q2": {"attack_success": 0}, "q3": {"attack_success": 1}}
    common, b, d = _paired_cell(baseline, defended, key_order=list(defended))
    assert common == ["q1", "q2"]  # q3 dropped: no baseline match
    assert b == [1, 0]
    assert d == [0, 0]


def test_paired_cell_empty_overlap():
    baseline = {"qA": {"attack_success": 1}}
    defended = {"qB": {"attack_success": 1}}
    common, b, d = _paired_cell(baseline, defended, key_order=list(defended))
    assert common == [] and b == [] and d == []


# --------------------------------------------------------------------------
# _mcnemar_and_effect wiring
# --------------------------------------------------------------------------

def test_mcnemar_and_effect_computes_correct_asr_and_test_used():
    baseline_asr = [1, 1, 0, 0]
    defended_asr = [0, 0, 0, 0]
    result = _mcnemar_and_effect(baseline_asr, defended_asr)
    assert result["n"] == 4
    assert result["baseline_asr"] == 0.5
    assert result["defended_asr"] == 0.0
    assert result["test_used"] == "mcnemar_exact"
    assert result["reduction"] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# discover_injection_cells: smoke-fragment exclusion (the actual
# "alignment-checking" logic under test -- files below 50% of their
# defense's expected n must be excluded, not treated as real cells)
# --------------------------------------------------------------------------

def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _injection_row(question, attack_success, f1_clean=0.5):
    return {"question": question, "attack_success": attack_success, "f1_clean": f1_clean}


@pytest.fixture
def fake_injection_dirs(tmp_path, monkeypatch):
    p3 = tmp_path / "phase3_defense_results"
    p2 = tmp_path / "phase2_injection_results"
    p3.mkdir()
    p2.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("scripts.analyze_phase3_defense_stats.P3_DIR", p3)
    monkeypatch.setattr("scripts.analyze_phase3_defense_stats.P2_INJECTION_DIR", p2)
    return p3, p2


def test_discover_injection_cells_excludes_n3_smoke_fragment(fake_injection_dirs):
    p3, p2 = fake_injection_dirs
    model, corpus, template = "llama-3.1-8b", "hotpot_qa", "ignore"

    # a real instruction_detection cell needs n>=20 (50% of expected_n=40)
    # to survive -- give it exactly 40, all baseline-matched
    full_rows = [_injection_row(f"q{i}", attack_success=(1 if i < 10 else 0)) for i in range(40)]
    _write_jsonl(p3 / f"attack_raw_{model}_{corpus}_{template}_hf_defense-instruction_detection.jsonl", full_rows)
    _write_jsonl(p2 / f"attack_raw_{model}_{corpus}_{template}_vllm.jsonl", full_rows)

    # a smoke-test fragment: only 3 rows, far below the 20-row (50%) floor
    frag_model, frag_corpus, frag_template = "qwen3-8b", "hotpot_qa", "combined"
    frag_rows = [_injection_row(f"f{i}", attack_success=0) for i in range(3)]
    _write_jsonl(
        p3 / f"attack_raw_{frag_model}_{frag_corpus}_{frag_template}_hf_defense-instruction_detection.jsonl",
        frag_rows,
    )
    _write_jsonl(p2 / f"attack_raw_{frag_model}_{frag_corpus}_{frag_template}_vllm.jsonl", frag_rows)

    cells, fragments = discover_injection_cells()

    assert (model, corpus, template, "instruction_detection") in cells
    assert cells[(model, corpus, template, "instruction_detection")]["n"] == 40

    assert (frag_model, frag_corpus, frag_template, "instruction_detection") not in cells
    assert (frag_model, frag_corpus, frag_template, "instruction_detection", 3) in fragments


def test_discover_injection_cells_keeps_file_exactly_at_the_50pct_floor(fake_injection_dirs):
    p3, p2 = fake_injection_dirs
    model, corpus, template = "phi-4-mini", "ms_marco", "naive"
    # instruction_detection expected_n=40 -> floor is exactly 20 (>= keeps it)
    rows = [_injection_row(f"q{i}", attack_success=0) for i in range(20)]
    _write_jsonl(p3 / f"attack_raw_{model}_{corpus}_{template}_hf_defense-instruction_detection.jsonl", rows)
    _write_jsonl(p2 / f"attack_raw_{model}_{corpus}_{template}_vllm.jsonl", rows)

    cells, fragments = discover_injection_cells()

    assert (model, corpus, template, "instruction_detection") in cells
    assert fragments == []


# --------------------------------------------------------------------------
# _three_way_verdict: the classification logic behind Task 2's
# backend-confound isolation (vllm-baseline vs hf-no-defense vs
# hf-defended) -- pure logic, no fixture needed.
# --------------------------------------------------------------------------

def test_three_way_verdict_defense_works_when_only_nodef_vs_def_significant():
    # vllm and hf-no-defense agree (backend switch alone changes nothing);
    # hf-no-defense vs hf-defended is the only significant gap -> the
    # defense is doing the work, not the backend switch
    assert _three_way_verdict((False, True, True), 0.5, 0.5, 0.0) == "defense_works"


def test_three_way_verdict_backend_confound_when_only_vllm_vs_nodef_significant():
    # hf-no-defense and hf-defended agree (the defense adds nothing on
    # top of the backend switch); vllm vs hf-no-defense is the only
    # significant gap -> the backend switch alone explains the reduction
    assert _three_way_verdict((True, False, True), 0.6, 0.1, 0.1) == "backend_confound"


def test_three_way_verdict_partial_split_when_both_explanatory_pairs_significant():
    assert _three_way_verdict((True, True, True), 0.6, 0.3, 0.0) == "partial_split"


def test_three_way_verdict_inconclusive_when_nothing_significant():
    assert _three_way_verdict((False, False, False), 0.01, 0.01, 0.01) == "inconclusive"


def test_three_way_verdict_inconclusive_when_only_the_non_explanatory_pair_fires():
    # documents the actual (intentional) behavior: vllm-vs-def alone
    # being significant, with neither of the two pairs that would
    # actually explain WHY (vllm-vs-nodef, nodef-vs-def) reaching
    # significance, is reported as inconclusive rather than guessed at
    assert _three_way_verdict((False, False, True), 0.5, 0.3, 0.0) == "inconclusive"


# --------------------------------------------------------------------------
# discover_backend_confound_cells / corrected_backend_isolated_table
# --------------------------------------------------------------------------

@pytest.fixture
def fake_confound_dirs(tmp_path, monkeypatch):
    p3 = tmp_path / "phase3_defense_results"
    p2 = tmp_path / "phase2_injection_results"
    p3.mkdir()
    p2.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("scripts.analyze_phase3_defense_stats.P3_DIR", p3)
    monkeypatch.setattr("scripts.analyze_phase3_defense_stats.P2_INJECTION_DIR", p2)
    # small expected_n so a 40-row fixture clears the 50% smoke-fragment floor
    monkeypatch.setattr(
        "scripts.analyze_phase3_defense_stats.INJECTION_EXPECTED_N",
        {"instruction_detection": 40, "spotlighting": 40, "output_filter": 40},
    )
    return p3, p2


def test_discover_backend_confound_cells_detects_backend_confound_pattern(fake_confound_dirs):
    p3, p2 = fake_confound_dirs
    model, corpus, template, defense = "ministral-3-8b", "hotpot_qa", "ignore", "output_filter"
    n = 40

    # vllm succeeds on 25/40; hf-no-defense and hf-defended both fail on
    # every item (identical to each other) -- the real-world pattern this
    # session found for output_filter: the defended run is
    # indistinguishable from the undefended hf baseline, both far below vllm
    vllm_rows = [_injection_row(f"q{i}", attack_success=(1 if i < 25 else 0)) for i in range(n)]
    nodef_rows = [_injection_row(f"q{i}", attack_success=0) for i in range(n)]
    def_rows = [_injection_row(f"q{i}", attack_success=0) for i in range(n)]

    _write_jsonl(p2 / f"attack_raw_{model}_{corpus}_{template}_vllm.jsonl", vllm_rows)
    _write_jsonl(p3 / f"attack_raw_{model}_{corpus}_{template}_hf.jsonl", nodef_rows)
    _write_jsonl(p3 / f"attack_raw_{model}_{corpus}_{template}_hf_defense-{defense}.jsonl", def_rows)

    cells = discover_backend_confound_cells()

    key = (defense, model, corpus, template)
    assert key in cells
    cell = cells[key]
    assert cell["n"] == 40
    assert cell["vllm_asr"] == pytest.approx(0.625)
    assert cell["hf_nodef_asr"] == pytest.approx(0.0)
    assert cell["hf_def_asr"] == pytest.approx(0.0)
    assert cell["verdict"] == "backend_confound"
    assert cell["backend_fraction_of_reduction"] == pytest.approx(1.0)
    assert cell["pairs"]["hf_nodef_vs_hf_def"]["p_holm"] == pytest.approx(1.0)  # identical arrays -> no evidence


def test_corrected_backend_isolated_table_uses_hf_nodef_as_baseline_not_vllm(fake_confound_dirs):
    p3, p2 = fake_confound_dirs
    model, corpus, template, defense = "ministral-3-8b", "hotpot_qa", "ignore", "output_filter"
    n = 40

    vllm_rows = [_injection_row(f"q{i}", attack_success=(1 if i < 25 else 0)) for i in range(n)]
    nodef_rows = [_injection_row(f"q{i}", attack_success=(1 if i < 5 else 0)) for i in range(n)]
    def_rows = [_injection_row(f"q{i}", attack_success=0) for i in range(n)]

    _write_jsonl(p2 / f"attack_raw_{model}_{corpus}_{template}_vllm.jsonl", vllm_rows)
    _write_jsonl(p3 / f"attack_raw_{model}_{corpus}_{template}_hf.jsonl", nodef_rows)
    _write_jsonl(p3 / f"attack_raw_{model}_{corpus}_{template}_hf_defense-{defense}.jsonl", def_rows)

    cells = discover_backend_confound_cells()
    corrected = corrected_backend_isolated_table(cells)

    assert defense in corrected
    key = (model, corpus, template)
    assert key in corrected[defense]
    result = corrected[defense][key]
    # baseline must be hf-no-defense (5/40=0.125), NOT vllm (25/40=0.625) --
    # this is the whole point of the correction
    assert result["baseline_asr"] == pytest.approx(0.125)
    assert result["defended_asr"] == pytest.approx(0.0)
    assert "p_holm" in result
    assert "significant_holm" in result


# --------------------------------------------------------------------------
# mechanism_instruction_detection / _by_model_corpus: Task 3's mechanism
# attribution from the new instruction_detection_log_attack_*.jsonl files.
# --------------------------------------------------------------------------

def _id_log_row(model, corpus, template, question, flags):
    return {
        "model": model, "corpus": corpus, "injection_template": template,
        "question": question,
        "passages": [{"passage_id": i, "flagged": f, "score": 0.9, "label": "SAFE"} for i, f in enumerate(flags)],
    }


@pytest.fixture
def fake_id_mechanism_dirs(tmp_path, monkeypatch):
    p3 = tmp_path / "phase3_defense_results"
    p2 = tmp_path / "phase2_injection_results"
    p3.mkdir()
    p2.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("scripts.analyze_phase3_defense_stats.P3_DIR", p3)
    monkeypatch.setattr("scripts.analyze_phase3_defense_stats.P2_INJECTION_DIR", p2)
    monkeypatch.setattr(
        "scripts.analyze_phase3_defense_stats.INJECTION_EXPECTED_N",
        {"instruction_detection": 4, "spotlighting": 4, "output_filter": 4},
    )
    return p3, p2


def test_mechanism_instruction_detection_splits_flagged_vs_unflagged_blocks(fake_id_mechanism_dirs):
    p3, p2 = fake_id_mechanism_dirs
    model, corpus, template = "ministral-3-8b", "hotpot_qa", "ignore"

    # baseline succeeds on all 4 items; defended fails (is "blocked") on
    # items 0-2, succeeds on item 3 (not blocked -- must be excluded)
    baseline_rows = [_injection_row(f"q{i}", attack_success=1) for i in range(4)]
    defended_rows = [_injection_row(f"q{i}", attack_success=(0 if i < 3 else 1)) for i in range(4)]
    _write_jsonl(p2 / f"attack_raw_{model}_{corpus}_{template}_vllm.jsonl", baseline_rows)
    _write_jsonl(
        p3 / f"attack_raw_{model}_{corpus}_{template}_hf_defense-instruction_detection.jsonl", defended_rows
    )

    # of the 3 blocked items: q0 has a flagged passage, q1 has none
    # flagged, q2 has one flagged among several -- any() must still catch
    # it. q3 (not blocked) is deliberately omitted from the log to prove
    # it's never looked at.
    log_rows = [
        _id_log_row(model, corpus, template, "q0", [True, False]),
        _id_log_row(model, corpus, template, "q1", [False, False, False]),
        _id_log_row(model, corpus, template, "q2", [False, True]),
    ]
    _write_jsonl(p3 / f"instruction_detection_log_attack_{model}_{corpus}.jsonl", log_rows)

    result = mechanism_instruction_detection()
    key = (model, corpus, template)
    assert result[key]["n_blocked"] == 3
    assert result[key]["n_any_flagged"] == 2  # q0 and q2, not q1
    assert result[key]["frac_any_passage_flagged"] == pytest.approx(2 / 3)


def test_mechanism_instruction_detection_by_model_corpus_aggregates_templates():
    per_cell = {
        ("m", "c", "ignore"): {
            "n_blocked": 3, "n_any_flagged": 2,
            "frac_any_passage_flagged": 2 / 3, "frac_model_resisted_alone": 1 / 3,
        },
        ("m", "c", "fake_completion"): {
            "n_blocked": 5, "n_any_flagged": 1,
            "frac_any_passage_flagged": 0.2, "frac_model_resisted_alone": 0.8,
        },
    }
    rollup = mechanism_instruction_detection_by_model_corpus(per_cell)
    assert rollup[("m", "c")]["n_blocked"] == 8
    assert rollup[("m", "c")]["n_any_flagged"] == 3
    assert rollup[("m", "c")]["frac_any_passage_flagged"] == pytest.approx(3 / 8)
