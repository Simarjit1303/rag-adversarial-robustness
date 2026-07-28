"""
VLLM_MAX_MODEL_LEN as an env var silently did nothing -- vLLM's own env var
validator treats any VLLM_* name as reserved for its internal use, logged
only as an easy-to-miss "Unknown vLLM environment variable detected"
warning, never as a hard failure. Renamed to RAG_VLLM_MAX_MODEL_LEN
everywhere it's read: config.py's module constant, and
evaluation/run_baseline.py's _resolve_vllm_max_model_len().
"""

import pytest

import evaluation.run_baseline as rb


def test_resolve_vllm_max_model_len_reads_renamed_env_var(monkeypatch):
    monkeypatch.setenv("RAG_VLLM_MAX_MODEL_LEN", "8192")

    assert rb._resolve_vllm_max_model_len() == 8192


def test_resolve_vllm_max_model_len_env_var_wins_over_config_value(monkeypatch):
    monkeypatch.setenv("RAG_VLLM_MAX_MODEL_LEN", "8192")
    monkeypatch.setattr(rb, "RAG_VLLM_MAX_MODEL_LEN", 4096)

    assert rb._resolve_vllm_max_model_len() == 8192


def test_resolve_vllm_max_model_len_falls_back_to_config_value(monkeypatch):
    monkeypatch.delenv("RAG_VLLM_MAX_MODEL_LEN", raising=False)
    monkeypatch.setattr(rb, "RAG_VLLM_MAX_MODEL_LEN", 4096)

    assert rb._resolve_vllm_max_model_len() == 4096


def test_resolve_vllm_max_model_len_raises_when_neither_set(monkeypatch):
    monkeypatch.delenv("RAG_VLLM_MAX_MODEL_LEN", raising=False)
    monkeypatch.setattr(rb, "RAG_VLLM_MAX_MODEL_LEN", None)

    with pytest.raises(RuntimeError, match="RAG_VLLM_MAX_MODEL_LEN is not set"):
        rb._resolve_vllm_max_model_len()
