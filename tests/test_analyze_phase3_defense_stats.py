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
    discover_injection_cells,
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
