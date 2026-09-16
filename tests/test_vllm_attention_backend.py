import sys
from unittest import mock
if 'vllm' not in sys.modules:
    _fake_vllm = mock.Mock()
    _fake_vllm.LLM = mock.Mock()
    _fake_vllm.SamplingParams = mock.Mock()
    sys.modules['vllm'] = _fake_vllm
import harness.vllm_engine as ve

def test_load_vllm_model_defaults_to_triton_attn(monkeypatch):
    monkeypatch.delenv('RAG_VLLM_ATTENTION_BACKEND', raising=False)
    calls = []
    monkeypatch.setattr(ve, 'LLM', lambda **kwargs: calls.append(kwargs))
    ve.load_vllm_model('phi-4-mini', max_model_len=4096)
    assert calls[0]['attention_backend'] == 'TRITON_ATTN'

def test_load_vllm_model_honors_rag_vllm_attention_backend_override(monkeypatch):
    monkeypatch.setenv('RAG_VLLM_ATTENTION_BACKEND', 'FLASH_ATTN')
    calls = []
    monkeypatch.setattr(ve, 'LLM', lambda **kwargs: calls.append(kwargs))
    ve.load_vllm_model('phi-4-mini', max_model_len=4096)
    assert calls[0]['attention_backend'] == 'FLASH_ATTN'
