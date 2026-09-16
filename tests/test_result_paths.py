import config
import evaluation.result_paths as rp

def test_result_file_paths_are_namespaced_by_model_corpus_and_engine(tmp_path):
    raw_a, summary_a = rp.result_file_paths(tmp_path, 'phi-4-mini', 'nq_open', 'hf')
    raw_b, summary_b = rp.result_file_paths(tmp_path, 'phi-4-mini', 'nq_open', 'vllm')
    assert raw_a != raw_b
    assert summary_a != summary_b
    assert raw_a.name == 'baseline_raw_phi-4-mini_nq_open_hf.jsonl'
    assert summary_a.name == 'baseline_summary_phi-4-mini_nq_open_hf.csv'

def test_resolve_sweep_selection_defaults_to_full_matrix(monkeypatch):
    monkeypatch.delenv('RAG_MODELS', raising=False)
    monkeypatch.delenv('RAG_CORPORA', raising=False)
    monkeypatch.delenv('INFERENCE_ENGINE', raising=False)
    model_keys, corpus_names, engine = rp.resolve_sweep_selection()
    assert model_keys == list(config.MODELS)
    assert corpus_names == list(config.CORPORA)
    assert engine == 'hf'

def test_resolve_sweep_selection_honors_env_vars(monkeypatch):
    monkeypatch.setenv('RAG_MODELS', 'phi-4-mini, qwen3-8b')
    monkeypatch.setenv('RAG_CORPORA', 'nq_open')
    monkeypatch.setenv('INFERENCE_ENGINE', 'vllm')
    model_keys, corpus_names, engine = rp.resolve_sweep_selection()
    assert model_keys == ['phi-4-mini', 'qwen3-8b']
    assert corpus_names == ['nq_open']
    assert engine == 'vllm'

def test_expected_result_files_covers_every_cell_in_the_selection(monkeypatch, tmp_path):
    monkeypatch.setenv('RAG_MODELS', 'phi-4-mini,qwen3-8b')
    monkeypatch.setenv('RAG_CORPORA', 'nq_open,ms_marco')
    monkeypatch.setenv('INFERENCE_ENGINE', 'hf')
    paths = rp.expected_result_files(tmp_path)
    assert len(paths) == 8
    names = {p.name for p in paths}
    assert 'baseline_raw_phi-4-mini_nq_open_hf.jsonl' in names
    assert 'baseline_summary_qwen3-8b_ms_marco_hf.csv' in names

def test_attack_eligible_corpora_excludes_nq_open():
    assert 'nq_open' not in rp.ATTACK_ELIGIBLE_CORPORA
    assert set(rp.ATTACK_ELIGIBLE_CORPORA) == set(config.CORPORA) - {'nq_open'}

def test_attack_result_file_paths_are_namespaced_by_all_four_axes(tmp_path):
    raw_a, summary_a = rp.attack_result_file_paths(tmp_path, 'phi-4-mini', 'hotpot_qa', 'naive', 'hf')
    raw_b, summary_b = rp.attack_result_file_paths(tmp_path, 'phi-4-mini', 'hotpot_qa', 'combined', 'hf')
    assert raw_a != raw_b
    assert raw_a.name == 'attack_raw_phi-4-mini_hotpot_qa_naive_hf.jsonl'
    assert summary_a.name == 'attack_summary_phi-4-mini_hotpot_qa_naive_hf.csv'

def test_resolve_attack_sweep_selection_defaults_exclude_nq_open_and_include_all_templates(monkeypatch):
    monkeypatch.delenv('RAG_MODELS', raising=False)
    monkeypatch.delenv('RAG_CORPORA', raising=False)
    monkeypatch.delenv('RAG_INJECTION_TEMPLATES', raising=False)
    monkeypatch.delenv('INFERENCE_ENGINE', raising=False)
    model_keys, corpus_names, injection_templates, engine = rp.resolve_attack_sweep_selection()
    assert model_keys == list(config.MODELS)
    assert corpus_names == rp.ATTACK_ELIGIBLE_CORPORA
    assert 'nq_open' not in corpus_names
    assert set(injection_templates) == {'naive', 'escape_char', 'ignore', 'fake_completion', 'combined'}
    assert engine == 'hf'

def test_resolve_attack_sweep_selection_honors_env_vars(monkeypatch):
    monkeypatch.setenv('RAG_MODELS', 'qwen3-8b')
    monkeypatch.setenv('RAG_CORPORA', 'ms_marco')
    monkeypatch.setenv('RAG_INJECTION_TEMPLATES', 'naive, combined')
    monkeypatch.setenv('INFERENCE_ENGINE', 'vllm')
    model_keys, corpus_names, injection_templates, engine = rp.resolve_attack_sweep_selection()
    assert model_keys == ['qwen3-8b']
    assert corpus_names == ['ms_marco']
    assert injection_templates == ['naive', 'combined']
    assert engine == 'vllm'

def test_resolve_attack_sweep_selection_explicit_nq_open_still_works(monkeypatch):
    monkeypatch.setenv('RAG_CORPORA', 'nq_open')
    _, corpus_names, _, _ = rp.resolve_attack_sweep_selection()
    assert corpus_names == ['nq_open']

def test_expected_attack_result_files_covers_every_cell(monkeypatch, tmp_path):
    monkeypatch.setenv('RAG_MODELS', 'phi-4-mini')
    monkeypatch.setenv('RAG_CORPORA', 'hotpot_qa,ms_marco')
    monkeypatch.setenv('RAG_INJECTION_TEMPLATES', 'naive,combined')
    monkeypatch.setenv('INFERENCE_ENGINE', 'hf')
    paths = rp.expected_attack_result_files(tmp_path)
    assert len(paths) == 8
    names = {p.name for p in paths}
    assert 'attack_raw_phi-4-mini_hotpot_qa_naive_hf.jsonl' in names
    assert 'attack_summary_phi-4-mini_ms_marco_combined_hf.csv' in names

def test_poison_result_file_paths_are_namespaced_by_all_four_axes(tmp_path):
    raw_a, summary_a = rp.poison_result_file_paths(tmp_path, 'phi-4-mini', 'hotpot_qa', 'adv5', 'hf')
    raw_b, summary_b = rp.poison_result_file_paths(tmp_path, 'phi-4-mini', 'hotpot_qa', 'adv1', 'hf')
    assert raw_a != raw_b
    assert raw_a.name == 'poison_raw_phi-4-mini_hotpot_qa_adv5_hf.jsonl'
    assert summary_a.name == 'poison_summary_phi-4-mini_hotpot_qa_adv5_hf.csv'

def test_resolve_poison_sweep_selection_defaults_exclude_nq_open_and_use_default_config(monkeypatch):
    monkeypatch.delenv('RAG_MODELS', raising=False)
    monkeypatch.delenv('RAG_CORPORA', raising=False)
    monkeypatch.delenv('RAG_POISON_CONFIGS', raising=False)
    monkeypatch.delenv('INFERENCE_ENGINE', raising=False)
    model_keys, corpus_names, poison_configs, engine = rp.resolve_poison_sweep_selection()
    assert model_keys == list(config.MODELS)
    assert corpus_names == rp.ATTACK_ELIGIBLE_CORPORA
    assert 'nq_open' not in corpus_names
    assert poison_configs == [rp.DEFAULT_POISON_CONFIG] == ['adv5']
    assert engine == 'hf'

def test_resolve_poison_sweep_selection_honors_env_vars(monkeypatch):
    monkeypatch.setenv('RAG_MODELS', 'qwen3-8b')
    monkeypatch.setenv('RAG_CORPORA', 'ms_marco')
    monkeypatch.setenv('RAG_POISON_CONFIGS', 'adv1, adv5')
    monkeypatch.setenv('INFERENCE_ENGINE', 'vllm')
    model_keys, corpus_names, poison_configs, engine = rp.resolve_poison_sweep_selection()
    assert model_keys == ['qwen3-8b']
    assert corpus_names == ['ms_marco']
    assert poison_configs == ['adv1', 'adv5']
    assert engine == 'vllm'

def test_expected_poison_result_files_covers_every_cell(monkeypatch, tmp_path):
    monkeypatch.setenv('RAG_MODELS', 'phi-4-mini')
    monkeypatch.setenv('RAG_CORPORA', 'hotpot_qa,ms_marco')
    monkeypatch.setenv('RAG_POISON_CONFIGS', 'adv5')
    monkeypatch.setenv('INFERENCE_ENGINE', 'hf')
    paths = rp.expected_poison_result_files(tmp_path)
    assert len(paths) == 4
    names = {p.name for p in paths}
    assert 'poison_raw_phi-4-mini_hotpot_qa_adv5_hf.jsonl' in names
    assert 'poison_summary_phi-4-mini_ms_marco_adv5_hf.csv' in names

def test_crescendo_result_file_paths_are_namespaced_by_model_turns_and_engine(tmp_path):
    raw_a, summary_a = rp.crescendo_result_file_paths(tmp_path, 'phi-4-mini', 5, 'hf')
    raw_b, summary_b = rp.crescendo_result_file_paths(tmp_path, 'phi-4-mini', 3, 'hf')
    assert raw_a != raw_b
    assert raw_a.name == 'crescendo_raw_phi-4-mini_5turn_hf.jsonl'
    assert summary_a.name == 'crescendo_summary_phi-4-mini_5turn_hf.csv'

def test_resolve_crescendo_sweep_selection_defaults(monkeypatch):
    monkeypatch.delenv('RAG_MODELS', raising=False)
    monkeypatch.delenv('RAG_MAX_TURNS', raising=False)
    monkeypatch.delenv('INFERENCE_ENGINE', raising=False)
    model_keys, max_turns, engine = rp.resolve_crescendo_sweep_selection()
    assert model_keys == list(config.MODELS)
    assert max_turns == rp.DEFAULT_MAX_TURNS == 5
    assert engine == 'hf'

def test_resolve_crescendo_sweep_selection_honors_env_vars(monkeypatch):
    monkeypatch.setenv('RAG_MODELS', 'qwen3-8b')
    monkeypatch.setenv('RAG_MAX_TURNS', '3')
    monkeypatch.setenv('INFERENCE_ENGINE', 'hf')
    model_keys, max_turns, engine = rp.resolve_crescendo_sweep_selection()
    assert model_keys == ['qwen3-8b']
    assert max_turns == 3
    assert engine == 'hf'

def test_expected_crescendo_result_files_covers_every_model(monkeypatch, tmp_path):
    monkeypatch.setenv('RAG_MODELS', 'phi-4-mini,qwen3-8b')
    monkeypatch.setenv('RAG_MAX_TURNS', '5')
    monkeypatch.setenv('INFERENCE_ENGINE', 'hf')
    paths = rp.expected_crescendo_result_files(tmp_path)
    assert len(paths) == 4
    names = {p.name for p in paths}
    assert 'crescendo_raw_phi-4-mini_5turn_hf.jsonl' in names
    assert 'crescendo_summary_qwen3-8b_5turn_hf.csv' in names

def test_resolve_defense_defaults_to_none(monkeypatch):
    monkeypatch.delenv('RAG_DEFENSE', raising=False)
    assert rp.resolve_defense() == 'none'

def test_resolve_defense_honors_env_var(monkeypatch):
    monkeypatch.setenv('RAG_DEFENSE', 'spotlighting')
    assert rp.resolve_defense() == 'spotlighting'

def test_resolve_defense_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv('RAG_DEFENSE', 'not_a_real_defense')
    try:
        rp.resolve_defense()
        assert False, 'expected ValueError'
    except ValueError:
        pass

def test_attack_result_file_paths_defense_none_matches_pre_phase3_filename(tmp_path):
    raw, summary = rp.attack_result_file_paths(tmp_path, 'phi-4-mini', 'hotpot_qa', 't1', 'hf')
    assert raw.name == 'attack_raw_phi-4-mini_hotpot_qa_t1_hf.jsonl'
    assert summary.name == 'attack_summary_phi-4-mini_hotpot_qa_t1_hf.csv'

def test_attack_result_file_paths_defense_axis_is_distinct(tmp_path):
    undefended = rp.attack_result_file_paths(tmp_path, 'phi-4-mini', 'hotpot_qa', 't1', 'hf')
    defended = rp.attack_result_file_paths(tmp_path, 'phi-4-mini', 'hotpot_qa', 't1', 'hf', defense='instruction_detection')
    assert undefended != defended
    assert defended[0].name == 'attack_raw_phi-4-mini_hotpot_qa_t1_hf_defense-instruction_detection.jsonl'

def test_poison_result_file_paths_defense_axis_is_distinct(tmp_path):
    undefended = rp.poison_result_file_paths(tmp_path, 'phi-4-mini', 'hotpot_qa', 'adv5', 'hf')
    defended = rp.poison_result_file_paths(tmp_path, 'phi-4-mini', 'hotpot_qa', 'adv5', 'hf', defense='spotlighting')
    assert undefended != defended
    assert defended[0].name == 'poison_raw_phi-4-mini_hotpot_qa_adv5_hf_defense-spotlighting.jsonl'

def test_crescendo_result_file_paths_defense_axis_is_distinct(tmp_path):
    undefended = rp.crescendo_result_file_paths(tmp_path, 'phi-4-mini', 5, 'hf')
    defended = rp.crescendo_result_file_paths(tmp_path, 'phi-4-mini', 5, 'hf', defense='output_filter')
    assert undefended != defended
    assert defended[0].name == 'crescendo_raw_phi-4-mini_5turn_hf_defense-output_filter.jsonl'
