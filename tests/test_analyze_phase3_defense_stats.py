import json
import pytest
from scripts.analyze_phase3_defense_stats import BACKEND_CONFOUND_DEFENSES, _holm_adjusted_pvalues, _mcnemar_and_effect, _paired_cell, _three_way_verdict, corrected_backend_isolated_table, discover_backend_confound_cells, discover_injection_cells, mechanism_instruction_detection, mechanism_instruction_detection_by_model_corpus

def test_holm_adjusted_pvalues_classic_textbook_case():
    p_values = [0.01, 0.04, 0.03]
    adjusted = _holm_adjusted_pvalues(p_values)
    assert adjusted[0] == pytest.approx(0.03)
    assert adjusted[2] == pytest.approx(0.06)
    assert adjusted[1] == pytest.approx(0.06)

def test_holm_adjusted_pvalues_enforces_monotonicity():
    p_values = [0.001, 0.002, 0.003]
    adjusted = _holm_adjusted_pvalues(p_values)
    assert adjusted[0] <= adjusted[1] <= adjusted[2]

def test_holm_adjusted_pvalues_capped_at_one():
    p_values = [0.9, 0.95]
    adjusted = _holm_adjusted_pvalues(p_values)
    assert all((p <= 1.0 for p in adjusted))

def test_holm_adjusted_pvalues_single_value_unchanged():
    assert _holm_adjusted_pvalues([0.03]) == [0.03]

def test_holm_adjusted_pvalues_empty_input():
    assert _holm_adjusted_pvalues([]) == []

def test_paired_cell_keeps_only_items_present_in_baseline():
    baseline = {'q1': {'attack_success': 1}, 'q2': {'attack_success': 0}}
    defended = {'q1': {'attack_success': 0}, 'q2': {'attack_success': 0}, 'q3': {'attack_success': 1}}
    common, b, d = _paired_cell(baseline, defended, key_order=list(defended))
    assert common == ['q1', 'q2']
    assert b == [1, 0]
    assert d == [0, 0]

def test_paired_cell_empty_overlap():
    baseline = {'qA': {'attack_success': 1}}
    defended = {'qB': {'attack_success': 1}}
    common, b, d = _paired_cell(baseline, defended, key_order=list(defended))
    assert common == [] and b == [] and (d == [])

def test_mcnemar_and_effect_computes_correct_asr_and_test_used():
    baseline_asr = [1, 1, 0, 0]
    defended_asr = [0, 0, 0, 0]
    result = _mcnemar_and_effect(baseline_asr, defended_asr)
    assert result['n'] == 4
    assert result['baseline_asr'] == 0.5
    assert result['defended_asr'] == 0.0
    assert result['test_used'] == 'mcnemar_exact'
    assert result['reduction'] == pytest.approx(0.5)

def _write_jsonl(path, rows):
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')

def _injection_row(question, attack_success, f1_clean=0.5):
    return {'question': question, 'attack_success': attack_success, 'f1_clean': f1_clean}

@pytest.fixture
def fake_injection_dirs(tmp_path, monkeypatch):
    p3 = tmp_path / 'phase3_defense_results'
    p2 = tmp_path / 'phase2_injection_results'
    p3.mkdir()
    p2.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('scripts.analyze_phase3_defense_stats.P3_DIR', p3)
    monkeypatch.setattr('scripts.analyze_phase3_defense_stats.P2_INJECTION_DIR', p2)
    return (p3, p2)

def test_discover_injection_cells_excludes_n3_smoke_fragment(fake_injection_dirs):
    p3, p2 = fake_injection_dirs
    model, corpus, template = ('llama-3.1-8b', 'hotpot_qa', 'ignore')
    full_rows = [_injection_row(f'q{i}', attack_success=1 if i < 10 else 0) for i in range(40)]
    _write_jsonl(p3 / f'attack_raw_{model}_{corpus}_{template}_hf_defense-instruction_detection.jsonl', full_rows)
    _write_jsonl(p2 / f'attack_raw_{model}_{corpus}_{template}_vllm.jsonl', full_rows)
    frag_model, frag_corpus, frag_template = ('qwen3-8b', 'hotpot_qa', 'combined')
    frag_rows = [_injection_row(f'f{i}', attack_success=0) for i in range(3)]
    _write_jsonl(p3 / f'attack_raw_{frag_model}_{frag_corpus}_{frag_template}_hf_defense-instruction_detection.jsonl', frag_rows)
    _write_jsonl(p2 / f'attack_raw_{frag_model}_{frag_corpus}_{frag_template}_vllm.jsonl', frag_rows)
    cells, fragments = discover_injection_cells()
    assert (model, corpus, template, 'instruction_detection') in cells
    assert cells[model, corpus, template, 'instruction_detection']['n'] == 40
    assert (frag_model, frag_corpus, frag_template, 'instruction_detection') not in cells
    assert (frag_model, frag_corpus, frag_template, 'instruction_detection', 3) in fragments

def test_discover_injection_cells_keeps_file_exactly_at_the_50pct_floor(fake_injection_dirs):
    p3, p2 = fake_injection_dirs
    model, corpus, template = ('phi-4-mini', 'ms_marco', 'naive')
    rows = [_injection_row(f'q{i}', attack_success=0) for i in range(20)]
    _write_jsonl(p3 / f'attack_raw_{model}_{corpus}_{template}_hf_defense-instruction_detection.jsonl', rows)
    _write_jsonl(p2 / f'attack_raw_{model}_{corpus}_{template}_vllm.jsonl', rows)
    cells, fragments = discover_injection_cells()
    assert (model, corpus, template, 'instruction_detection') in cells
    assert fragments == []

def test_three_way_verdict_defense_works_when_only_nodef_vs_def_significant():
    assert _three_way_verdict((False, True, True), 0.5, 0.5, 0.0) == 'defense_works'

def test_three_way_verdict_backend_confound_when_only_vllm_vs_nodef_significant():
    assert _three_way_verdict((True, False, True), 0.6, 0.1, 0.1) == 'backend_confound'

def test_three_way_verdict_partial_split_when_both_explanatory_pairs_significant():
    assert _three_way_verdict((True, True, True), 0.6, 0.3, 0.0) == 'partial_split'

def test_three_way_verdict_inconclusive_when_nothing_significant():
    assert _three_way_verdict((False, False, False), 0.01, 0.01, 0.01) == 'inconclusive'

def test_three_way_verdict_inconclusive_when_only_the_non_explanatory_pair_fires():
    assert _three_way_verdict((False, False, True), 0.5, 0.3, 0.0) == 'inconclusive'

@pytest.fixture
def fake_confound_dirs(tmp_path, monkeypatch):
    p3 = tmp_path / 'phase3_defense_results'
    p2 = tmp_path / 'phase2_injection_results'
    p3.mkdir()
    p2.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('scripts.analyze_phase3_defense_stats.P3_DIR', p3)
    monkeypatch.setattr('scripts.analyze_phase3_defense_stats.P2_INJECTION_DIR', p2)
    monkeypatch.setattr('scripts.analyze_phase3_defense_stats.INJECTION_EXPECTED_N', {'instruction_detection': 40, 'spotlighting': 40, 'output_filter': 40})
    return (p3, p2)

def test_discover_backend_confound_cells_detects_backend_confound_pattern(fake_confound_dirs):
    p3, p2 = fake_confound_dirs
    model, corpus, template, defense = ('ministral-3-8b', 'hotpot_qa', 'ignore', 'output_filter')
    n = 40
    vllm_rows = [_injection_row(f'q{i}', attack_success=1 if i < 25 else 0) for i in range(n)]
    nodef_rows = [_injection_row(f'q{i}', attack_success=0) for i in range(n)]
    def_rows = [_injection_row(f'q{i}', attack_success=0) for i in range(n)]
    _write_jsonl(p2 / f'attack_raw_{model}_{corpus}_{template}_vllm.jsonl', vllm_rows)
    _write_jsonl(p3 / f'attack_raw_{model}_{corpus}_{template}_hf.jsonl', nodef_rows)
    _write_jsonl(p3 / f'attack_raw_{model}_{corpus}_{template}_hf_defense-{defense}.jsonl', def_rows)
    cells = discover_backend_confound_cells()
    key = (defense, model, corpus, template)
    assert key in cells
    cell = cells[key]
    assert cell['n'] == 40
    assert cell['vllm_asr'] == pytest.approx(0.625)
    assert cell['hf_nodef_asr'] == pytest.approx(0.0)
    assert cell['hf_def_asr'] == pytest.approx(0.0)
    assert cell['verdict'] == 'backend_confound'
    assert cell['backend_fraction_of_reduction'] == pytest.approx(1.0)
    assert cell['pairs']['hf_nodef_vs_hf_def']['p_holm'] == pytest.approx(1.0)

def test_corrected_backend_isolated_table_uses_hf_nodef_as_baseline_not_vllm(fake_confound_dirs):
    p3, p2 = fake_confound_dirs
    model, corpus, template, defense = ('ministral-3-8b', 'hotpot_qa', 'ignore', 'output_filter')
    n = 40
    vllm_rows = [_injection_row(f'q{i}', attack_success=1 if i < 25 else 0) for i in range(n)]
    nodef_rows = [_injection_row(f'q{i}', attack_success=1 if i < 5 else 0) for i in range(n)]
    def_rows = [_injection_row(f'q{i}', attack_success=0) for i in range(n)]
    _write_jsonl(p2 / f'attack_raw_{model}_{corpus}_{template}_vllm.jsonl', vllm_rows)
    _write_jsonl(p3 / f'attack_raw_{model}_{corpus}_{template}_hf.jsonl', nodef_rows)
    _write_jsonl(p3 / f'attack_raw_{model}_{corpus}_{template}_hf_defense-{defense}.jsonl', def_rows)
    cells = discover_backend_confound_cells()
    corrected = corrected_backend_isolated_table(cells)
    assert defense in corrected
    key = (model, corpus, template)
    assert key in corrected[defense]
    result = corrected[defense][key]
    assert result['baseline_asr'] == pytest.approx(0.125)
    assert result['defended_asr'] == pytest.approx(0.0)
    assert 'p_holm' in result
    assert 'significant_holm' in result

def _id_log_row(model, corpus, template, question, flags):
    return {'model': model, 'corpus': corpus, 'injection_template': template, 'question': question, 'passages': [{'passage_id': i, 'flagged': f, 'score': 0.9, 'label': 'SAFE'} for i, f in enumerate(flags)]}

@pytest.fixture
def fake_id_mechanism_dirs(tmp_path, monkeypatch):
    p3 = tmp_path / 'phase3_defense_results'
    p2 = tmp_path / 'phase2_injection_results'
    p3.mkdir()
    p2.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('scripts.analyze_phase3_defense_stats.P3_DIR', p3)
    monkeypatch.setattr('scripts.analyze_phase3_defense_stats.P2_INJECTION_DIR', p2)
    monkeypatch.setattr('scripts.analyze_phase3_defense_stats.INJECTION_EXPECTED_N', {'instruction_detection': 4, 'spotlighting': 4, 'output_filter': 4})
    return (p3, p2)

def test_mechanism_instruction_detection_splits_flagged_vs_unflagged_blocks(fake_id_mechanism_dirs):
    p3, p2 = fake_id_mechanism_dirs
    model, corpus, template = ('ministral-3-8b', 'hotpot_qa', 'ignore')
    baseline_rows = [_injection_row(f'q{i}', attack_success=1) for i in range(4)]
    defended_rows = [_injection_row(f'q{i}', attack_success=0 if i < 3 else 1) for i in range(4)]
    _write_jsonl(p2 / f'attack_raw_{model}_{corpus}_{template}_vllm.jsonl', baseline_rows)
    _write_jsonl(p3 / f'attack_raw_{model}_{corpus}_{template}_hf_defense-instruction_detection.jsonl', defended_rows)
    log_rows = [_id_log_row(model, corpus, template, 'q0', [True, False]), _id_log_row(model, corpus, template, 'q1', [False, False, False]), _id_log_row(model, corpus, template, 'q2', [False, True])]
    _write_jsonl(p3 / f'instruction_detection_log_attack_{model}_{corpus}.jsonl', log_rows)
    result = mechanism_instruction_detection()
    key = (model, corpus, template)
    assert result[key]['n_blocked'] == 3
    assert result[key]['n_any_flagged'] == 2
    assert result[key]['frac_any_passage_flagged'] == pytest.approx(2 / 3)

def test_backend_confound_defenses_includes_instruction_detection():
    assert 'instruction_detection' in BACKEND_CONFOUND_DEFENSES

def test_discover_backend_confound_cells_covers_instruction_detection_at_n40(fake_confound_dirs):
    p3, p2 = fake_confound_dirs
    model, corpus, template, defense = ('ministral-3-8b', 'hotpot_qa', 'ignore', 'instruction_detection')
    n = 40
    vllm_rows = [_injection_row(f'q{i}', attack_success=1 if i < 14 else 0) for i in range(n)]
    nodef_rows = [_injection_row(f'q{i}', attack_success=0) for i in range(n)]
    def_rows = [_injection_row(f'q{i}', attack_success=0) for i in range(n)]
    _write_jsonl(p2 / f'attack_raw_{model}_{corpus}_{template}_vllm.jsonl', vllm_rows)
    _write_jsonl(p3 / f'attack_raw_{model}_{corpus}_{template}_hf.jsonl', nodef_rows)
    _write_jsonl(p3 / f'attack_raw_{model}_{corpus}_{template}_hf_defense-{defense}.jsonl', def_rows)
    cells = discover_backend_confound_cells()
    key = (defense, model, corpus, template)
    assert key in cells
    assert cells[key]['n'] == 40
    assert cells[key]['verdict'] == 'backend_confound'
    corrected = corrected_backend_isolated_table(cells)
    assert defense in corrected
    assert (model, corpus, template) in corrected[defense]

def test_mechanism_instruction_detection_hf_nodef_baseline_redefines_blocked(fake_id_mechanism_dirs):
    p3, p2 = fake_id_mechanism_dirs
    model, corpus, template = ('ministral-3-8b', 'hotpot_qa', 'ignore')
    vllm_rows = [_injection_row(f'q{i}', attack_success=1) for i in range(4)]
    nodef_rows = [_injection_row(f'q{i}', attack_success=1 if i < 2 else 0) for i in range(4)]
    defended_rows = [_injection_row(f'q{i}', attack_success=0) for i in range(4)]
    _write_jsonl(p2 / f'attack_raw_{model}_{corpus}_{template}_vllm.jsonl', vllm_rows)
    _write_jsonl(p3 / f'attack_raw_{model}_{corpus}_{template}_hf.jsonl', nodef_rows)
    _write_jsonl(p3 / f'attack_raw_{model}_{corpus}_{template}_hf_defense-instruction_detection.jsonl', defended_rows)
    log_rows = [_id_log_row(model, corpus, template, 'q0', [True]), _id_log_row(model, corpus, template, 'q1', [False])]
    _write_jsonl(p3 / f'instruction_detection_log_attack_{model}_{corpus}.jsonl', log_rows)
    vllm_result = mechanism_instruction_detection(baseline='vllm')
    hf_nodef_result = mechanism_instruction_detection(baseline='hf_nodef')
    key = (model, corpus, template)
    assert vllm_result[key]['n_blocked'] == 4
    assert hf_nodef_result[key]['n_blocked'] == 2
    assert hf_nodef_result[key]['n_any_flagged'] == 1
    assert hf_nodef_result[key]['frac_any_passage_flagged'] == pytest.approx(0.5)

def test_mechanism_instruction_detection_by_model_corpus_aggregates_templates():
    per_cell = {('m', 'c', 'ignore'): {'n_blocked': 3, 'n_any_flagged': 2, 'frac_any_passage_flagged': 2 / 3, 'frac_model_resisted_alone': 1 / 3}, ('m', 'c', 'fake_completion'): {'n_blocked': 5, 'n_any_flagged': 1, 'frac_any_passage_flagged': 0.2, 'frac_model_resisted_alone': 0.8}}
    rollup = mechanism_instruction_detection_by_model_corpus(per_cell)
    assert rollup['m', 'c']['n_blocked'] == 8
    assert rollup['m', 'c']['n_any_flagged'] == 3
    assert rollup['m', 'c']['frac_any_passage_flagged'] == pytest.approx(3 / 8)
