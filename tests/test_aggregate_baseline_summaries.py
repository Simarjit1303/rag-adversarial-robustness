from scripts.aggregate_baseline_summaries import aggregate

def _write_summary(path, row):
    path.write_text(f"model,corpus,n,f1_clean,exact_match,f1_raw,contains_answer_diagnostic\n{row['model']},{row['corpus']},{row['n']},{row['f1_clean']},{row['exact_match']},{row['f1_raw']},{row['contains_answer_diagnostic']}\n", encoding='utf-8')

def test_aggregate_combines_all_per_cell_summaries(tmp_path):
    _write_summary(tmp_path / 'baseline_summary_phi-4-mini_nq_open_hf.csv', {'model': 'phi-4-mini', 'corpus': 'nq_open', 'n': 1000, 'f1_clean': 0.9, 'exact_match': 0.8, 'f1_raw': 0.85, 'contains_answer_diagnostic': 0.95})
    _write_summary(tmp_path / 'baseline_summary_qwen3-8b_ms_marco_vllm.csv', {'model': 'qwen3-8b', 'corpus': 'ms_marco', 'n': 1000, 'f1_clean': 0.7, 'exact_match': 0.6, 'f1_raw': 0.65, 'contains_answer_diagnostic': 0.75})
    out_path = aggregate(results_dir=tmp_path)
    assert out_path == tmp_path / 'baseline_summary_all.csv'
    content = out_path.read_text(encoding='utf-8')
    assert 'phi-4-mini' in content and 'nq_open' in content
    assert 'qwen3-8b' in content and 'ms_marco' in content
    assert content.count('\n') == 3

def test_aggregate_ignores_its_own_prior_output_and_non_matching_files(tmp_path):
    _write_summary(tmp_path / 'baseline_summary_phi-4-mini_nq_open_hf.csv', {'model': 'phi-4-mini', 'corpus': 'nq_open', 'n': 1000, 'f1_clean': 0.9, 'exact_match': 0.8, 'f1_raw': 0.85, 'contains_answer_diagnostic': 0.95})
    (tmp_path / 'baseline_summary_all.csv').write_text('model,corpus,n,f1_clean,exact_match,f1_raw,contains_answer_diagnostic\nstale,stale,1,1,1,1,1\n', encoding='utf-8')
    (tmp_path / 'baseline_raw_phi-4-mini_nq_open_hf.jsonl').write_text('{"model": "phi-4-mini"}\n', encoding='utf-8')
    aggregate(results_dir=tmp_path)
    content = (tmp_path / 'baseline_summary_all.csv').read_text(encoding='utf-8')
    assert 'stale' not in content
    assert content.count('\n') == 2

def test_aggregate_returns_none_when_no_summary_files_exist(tmp_path):
    assert aggregate(results_dir=tmp_path) is None
