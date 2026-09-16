from unittest import mock
import pytest
import config
import data.build_index as bi
import evaluation.run_baseline as rb

@pytest.fixture
def stub_heavy_calls(monkeypatch, tmp_path):
    build_index_calls = []

    def fake_build_index(corpus_name, split='dev'):
        build_index_calls.append(corpus_name)
        return (None, [])
    monkeypatch.setattr(rb, 'load_model', lambda model_key: (mock.Mock(), mock.Mock()))
    monkeypatch.setattr(rb, 'build_index', fake_build_index)
    monkeypatch.setattr(rb, 'RESULTS_DIR', tmp_path)
    return build_index_calls

def test_rag_corpora_env_var_restricts_corpus_loop(monkeypatch, stub_heavy_calls):
    monkeypatch.setenv('RAG_CORPORA', 'nq_open')
    rb.run_baseline_sweep(model_keys=['phi-4-mini'])
    assert stub_heavy_calls == ['nq_open']

def test_rag_corpora_unset_processes_all_corpora(monkeypatch, stub_heavy_calls):
    monkeypatch.delenv('RAG_CORPORA', raising=False)
    rb.run_baseline_sweep(model_keys=['phi-4-mini'])
    assert stub_heavy_calls == list(config.CORPORA)

def test_rag_corpora_explicit_arg_still_wins_over_env_var(monkeypatch, stub_heavy_calls):
    monkeypatch.setenv('RAG_CORPORA', 'ms_marco')
    rb.run_baseline_sweep(model_keys=['phi-4-mini'], corpus_names=['nq_open'])
    assert stub_heavy_calls == ['nq_open']

def test_rag_corpora_unknown_value_raises(monkeypatch, stub_heavy_calls):
    monkeypatch.setenv('RAG_CORPORA', 'not_a_real_corpus')
    with pytest.raises(ValueError, match='Unknown corpus names'):
        rb.run_baseline_sweep(model_keys=['phi-4-mini'])

def test_rag_corpora_multiple_values_comma_separated(monkeypatch, stub_heavy_calls):
    monkeypatch.setenv('RAG_CORPORA', 'nq_open, ms_marco')
    rb.run_baseline_sweep(model_keys=['phi-4-mini'])
    assert stub_heavy_calls == ['nq_open', 'ms_marco']

def _stub_build_index(monkeypatch, env_value, built):
    monkeypatch.setattr(bi, 'build_index', lambda corpus_name, split='dev': built.append(corpus_name))
    if env_value is None:
        monkeypatch.delenv('RAG_CORPORA', raising=False)
    else:
        monkeypatch.setenv('RAG_CORPORA', env_value)

def test_build_index_main_respects_rag_corpora(monkeypatch):
    built = []
    _stub_build_index(monkeypatch, 'nq_open', built)
    bi.main()
    assert built == ['nq_open']

def test_build_index_main_unset_builds_all_corpora(monkeypatch):
    built = []
    _stub_build_index(monkeypatch, None, built)
    bi.main()
    assert built == list(config.CORPORA)

def test_build_index_main_unknown_corpus_raises(monkeypatch):
    built = []
    _stub_build_index(monkeypatch, 'not_a_real_corpus', built)
    with pytest.raises(ValueError, match='Unknown corpus names'):
        bi.main()
    assert built == []
