"""
Attention backend selection: VLLM_ATTENTION_BACKEND as an env var does
nothing on this vLLM version -- confirmed via vLLM's own docs -- because
backend selection is a constructor kwarg to LLM(), not an env var.
load_vllm_model() now passes attention_backend=... directly instead.

RAG_VLLM_ATTENTION_BACKEND (not VLLM_ATTENTION_BACKEND) because vLLM's own
env var validator treats any VLLM_* name as reserved for internal use --
the same root cause that made VLLM_MAX_MODEL_LEN a silent no-op too (see
test_vllm_max_model_len_rename.py).

vllm isn't installed on this Windows/CPU-only dev machine, so a stand-in
module is injected into sys.modules before harness.vllm_engine (which does
`from vllm import LLM, SamplingParams` at module scope) gets imported --
this only needs to satisfy the import, since LLM itself is monkeypatched
per test below.
"""

import sys
from unittest import mock

if "vllm" not in sys.modules:
    _fake_vllm = mock.Mock()
    _fake_vllm.LLM = mock.Mock()
    _fake_vllm.SamplingParams = mock.Mock()
    sys.modules["vllm"] = _fake_vllm

import harness.vllm_engine as ve  # noqa: E402


def test_load_vllm_model_defaults_to_triton_attn(monkeypatch):
    monkeypatch.delenv("RAG_VLLM_ATTENTION_BACKEND", raising=False)
    calls = []
    monkeypatch.setattr(ve, "LLM", lambda **kwargs: calls.append(kwargs))

    ve.load_vllm_model("phi-4-mini", max_model_len=4096)

    assert calls[0]["attention_backend"] == "TRITON_ATTN"


def test_load_vllm_model_honors_rag_vllm_attention_backend_override(monkeypatch):
    monkeypatch.setenv("RAG_VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    calls = []
    monkeypatch.setattr(ve, "LLM", lambda **kwargs: calls.append(kwargs))

    ve.load_vllm_model("phi-4-mini", max_model_len=4096)

    assert calls[0]["attention_backend"] == "FLASH_ATTN"
