"""
Bug 2: RAG_CORPORA was documented/assumed but never actually wired up -- a
RunPod run launched with RAG_CORPORA=nq_open still built indices for every
corpus in config.CORPORA. Two separate places needed the fix:
  - evaluation/run_baseline.py: run_baseline_sweep()'s corpus loop
  - data/build_index.py: the `python -m data.build_index` __main__ block,
    which runs as its own pipeline stage BEFORE run_baseline and is what the
    real smoke test actually hit.

These tests stub out the expensive parts (model loading, index building) so
the corpus-processing loop itself never touches a real model or dataset --
only the filtering logic is exercised.
"""

from unittest import mock

import pytest

import config
import data.build_index as bi
import evaluation.run_baseline as rb


# --------------------------------------------------------------------------
# evaluation/run_baseline.py -- run_baseline_sweep()
# --------------------------------------------------------------------------

@pytest.fixture
def stub_heavy_calls(monkeypatch, tmp_path):
    """
    Stand in for model loading and index building so run_baseline_sweep can
    run its real corpus-filtering logic without touching a real model or
    dataset. build_index returns an empty record list, so the inner
    per-question loop is a no-op -- nothing downstream of it needs stubbing.
    """
    build_index_calls = []

    def fake_build_index(corpus_name, split="dev"):
        build_index_calls.append(corpus_name)
        return None, []

    monkeypatch.setattr(rb, "load_model", lambda model_key: (mock.Mock(), mock.Mock()))
    monkeypatch.setattr(rb, "build_index", fake_build_index)
    monkeypatch.setattr(rb, "RESULTS_DIR", tmp_path)
    return build_index_calls


def test_rag_corpora_env_var_restricts_corpus_loop(monkeypatch, stub_heavy_calls):
    monkeypatch.setenv("RAG_CORPORA", "nq_open")
    rb.run_baseline_sweep(model_keys=["phi-4-mini"])

    # only nq_open should ever have been touched -- not skipped in output,
    # never touched at all
    assert stub_heavy_calls == ["nq_open"]


def test_rag_corpora_unset_processes_all_corpora(monkeypatch, stub_heavy_calls):
    monkeypatch.delenv("RAG_CORPORA", raising=False)
    rb.run_baseline_sweep(model_keys=["phi-4-mini"])

    # default full-sweep behavior must be unaffected by this fix
    assert stub_heavy_calls == list(config.CORPORA)


def test_rag_corpora_explicit_arg_still_wins_over_env_var(monkeypatch, stub_heavy_calls):
    # Explicit corpus_names (as run_baseline's own callers might pass) must
    # not be silently overridden by the env var -- same precedence as
    # RAG_MODELS already has.
    monkeypatch.setenv("RAG_CORPORA", "ms_marco")
    rb.run_baseline_sweep(model_keys=["phi-4-mini"], corpus_names=["nq_open"])

    assert stub_heavy_calls == ["nq_open"]


def test_rag_corpora_unknown_value_raises(monkeypatch, stub_heavy_calls):
    monkeypatch.setenv("RAG_CORPORA", "not_a_real_corpus")
    with pytest.raises(ValueError, match="Unknown corpus names"):
        rb.run_baseline_sweep(model_keys=["phi-4-mini"])


def test_rag_corpora_multiple_values_comma_separated(monkeypatch, stub_heavy_calls):
    monkeypatch.setenv("RAG_CORPORA", "nq_open, ms_marco")
    rb.run_baseline_sweep(model_keys=["phi-4-mini"])

    assert stub_heavy_calls == ["nq_open", "ms_marco"]


# --------------------------------------------------------------------------
# data/build_index.py -- `python -m data.build_index`'s main()
# --------------------------------------------------------------------------

def _stub_build_index(monkeypatch, env_value, built):
    monkeypatch.setattr(
        bi, "build_index", lambda corpus_name, split="dev": built.append(corpus_name)
    )
    if env_value is None:
        monkeypatch.delenv("RAG_CORPORA", raising=False)
    else:
        monkeypatch.setenv("RAG_CORPORA", env_value)


def test_build_index_main_respects_rag_corpora(monkeypatch):
    built = []
    _stub_build_index(monkeypatch, "nq_open", built)
    bi.main()
    assert built == ["nq_open"]


def test_build_index_main_unset_builds_all_corpora(monkeypatch):
    built = []
    _stub_build_index(monkeypatch, None, built)
    bi.main()
    assert built == list(config.CORPORA)


def test_build_index_main_unknown_corpus_raises(monkeypatch):
    built = []
    _stub_build_index(monkeypatch, "not_a_real_corpus", built)
    with pytest.raises(ValueError, match="Unknown corpus names"):
        bi.main()
    assert built == []
