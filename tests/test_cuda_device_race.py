from unittest import mock
import pytest
import data.build_index as bi
from harness import model_loader as ml

def _stub_model_and_tokenizer(monkeypatch):
    fake_model = mock.Mock()
    monkeypatch.setattr(ml, 'load_tokenizer', lambda model_key: mock.Mock())
    monkeypatch.setattr(ml.AutoModelForCausalLM, 'from_pretrained', lambda *a, **k: fake_model)
    return fake_model

def test_load_model_calls_cuda_synchronize_when_cuda_available(monkeypatch):
    _stub_model_and_tokenizer(monkeypatch)
    monkeypatch.setattr(ml.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(ml.torch.cuda, 'get_device_name', lambda index=0: 'fake-gpu')
    sync_calls = []
    monkeypatch.setattr(ml.torch.cuda, 'synchronize', lambda: sync_calls.append(True))
    ml.load_model('phi-4-mini')
    assert sync_calls == [True]

def test_load_model_skips_cuda_synchronize_when_cuda_unavailable(monkeypatch):
    _stub_model_and_tokenizer(monkeypatch)
    monkeypatch.setattr(ml.torch.cuda, 'is_available', lambda: False)
    sync_calls = []
    monkeypatch.setattr(ml.torch.cuda, 'synchronize', lambda: sync_calls.append(True))
    ml.load_model('phi-4-mini')
    assert sync_calls == []

@pytest.fixture(autouse=True)
def reset_embedder_cache(monkeypatch):
    monkeypatch.setattr(bi, '_embedder', None)
    monkeypatch.setattr(bi.time, 'sleep', lambda seconds: None)

def test_get_embedder_retries_and_succeeds_on_third_attempt(monkeypatch):
    calls = []

    def fake_sentence_transformer(model_name, device):
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError('cudaErrorDevicesUnavailable')
        return mock.Mock(name='embedder')
    monkeypatch.setattr(bi, 'SentenceTransformer', fake_sentence_transformer)
    result = bi._get_embedder()
    assert len(calls) == 3
    assert result is not None
    assert bi._embedder is result

def test_get_embedder_raises_original_exception_after_three_failures(monkeypatch):
    calls = []

    def always_fails(model_name, device):
        calls.append(1)
        raise RuntimeError(f'cudaErrorDevicesUnavailable attempt {len(calls)}')
    monkeypatch.setattr(bi, 'SentenceTransformer', always_fails)
    with pytest.raises(RuntimeError, match='attempt 3'):
        bi._get_embedder()
    assert len(calls) == 3

def test_get_embedder_succeeds_immediately_without_retry_when_no_failure(monkeypatch):
    calls = []
    fake_embedder = mock.Mock(name='embedder')

    def fake_sentence_transformer(model_name, device):
        calls.append(1)
        return fake_embedder
    monkeypatch.setattr(bi, 'SentenceTransformer', fake_sentence_transformer)
    result = bi._get_embedder()
    assert len(calls) == 1
    assert result is fake_embedder
