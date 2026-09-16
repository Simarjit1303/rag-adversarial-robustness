from unittest import mock
import evaluation.run_baseline as rb

def _fake_record(question, answer):
    return {'question': question, 'answer': answer}

def _stub_sweep(monkeypatch, tmp_path):
    monkeypatch.setattr(rb, 'RESULTS_DIR', tmp_path)
    monkeypatch.setattr(rb, 'load_model', lambda model_key: (mock.Mock(), mock.Mock()))

    def fake_build_index(corpus_name, split='dev'):
        return (None, [_fake_record(f'q-{corpus_name}', f'a-{corpus_name}')])

    def fake_extract_question(corpus_name, record):
        return record['question']

    def fake_extract_gold_answers(corpus_name, record):
        return [record['answer']]

    def fake_run_query(model, tokenizer, model_key, corpus_name, question, index, records):
        return {'generated_answer': f'answer for {question}', 'generated_answer_clean': f'answer for {question}', 'retrieved_doc_ids': [0]}
    monkeypatch.setattr(rb, 'build_index', fake_build_index)
    monkeypatch.setattr(rb, 'extract_question', fake_extract_question)
    monkeypatch.setattr(rb, 'extract_gold_answers', fake_extract_gold_answers)
    monkeypatch.setattr(rb, 'run_query', fake_run_query)

def test_two_different_cells_run_back_to_back_produce_distinct_files(monkeypatch, tmp_path):
    _stub_sweep(monkeypatch, tmp_path)
    rb.run_baseline_sweep(model_keys=['phi-4-mini'], corpus_names=['nq_open'])
    rb.run_baseline_sweep(model_keys=['phi-4-mini'], corpus_names=['ms_marco'])
    nq_raw = tmp_path / 'baseline_raw_phi-4-mini_nq_open_hf.jsonl'
    ms_raw = tmp_path / 'baseline_raw_phi-4-mini_ms_marco_hf.jsonl'
    nq_summary = tmp_path / 'baseline_summary_phi-4-mini_nq_open_hf.csv'
    ms_summary = tmp_path / 'baseline_summary_phi-4-mini_ms_marco_hf.csv'
    assert nq_raw.exists() and nq_raw.stat().st_size > 0
    assert ms_raw.exists() and ms_raw.stat().st_size > 0
    assert nq_summary.exists() and nq_summary.stat().st_size > 0
    assert ms_summary.exists() and ms_summary.stat().st_size > 0
    assert 'q-nq_open' in nq_raw.read_text(encoding='utf-8')
    assert 'q-ms_marco' in ms_raw.read_text(encoding='utf-8')
    assert 'q-ms_marco' not in nq_raw.read_text(encoding='utf-8')

def test_same_model_and_corpus_different_engine_do_not_collide(monkeypatch, tmp_path):
    _stub_sweep(monkeypatch, tmp_path)
    monkeypatch.setenv('INFERENCE_ENGINE', 'hf')
    rb.run_baseline_sweep(model_keys=['phi-4-mini'], corpus_names=['nq_open'])
    hf_raw = tmp_path / 'baseline_raw_phi-4-mini_nq_open_hf.jsonl'
    assert hf_raw.exists()
    original_content = hf_raw.read_text(encoding='utf-8')
    from evaluation.result_paths import result_file_paths
    vllm_raw, _ = result_file_paths(tmp_path, 'phi-4-mini', 'nq_open', 'vllm')
    assert vllm_raw != hf_raw
    assert hf_raw.read_text(encoding='utf-8') == original_content
