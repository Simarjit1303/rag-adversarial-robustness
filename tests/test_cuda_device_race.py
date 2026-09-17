"""
Fix for an intermittent CUDA device-busy race between model load and
embedder load, observed directly on a genuinely fresh RunPod pod (no
restart history): two clean runs, one cudaErrorDevicesUnavailable, same
code path each time.

Fix 1 (harness/model_loader.py): torch.cuda.synchronize() after the model
finishes loading, closing the timing gap before a second model claims the
same GPU.
Fix 2 (data/build_index.py's _get_embedder): a bounded retry as cheap
insurance in case the sync doesn't fully close the window.

No GPU needed for either test below -- both mock the CUDA/model-loading
calls entirely, exercising only the control-flow logic.
"""

from unittest import mock

import pytest

import data.build_index as bi
from harness import model_loader as ml


# --------------------------------------------------------------------------
# Fix 1 -- harness/model_loader.py's load_model() synchronizes after load.
# --------------------------------------------------------------------------

def _stub_model_and_tokenizer(monkeypatch):
    fake_model = mock.Mock()
    monkeypatch.setattr(ml, "load_tokenizer", lambda model_key: mock.Mock())
    monkeypatch.setattr(
        ml.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: fake_model
    )
    return fake_model


def test_load_model_calls_cuda_synchronize_when_cuda_available(monkeypatch):
    _stub_model_and_tokenizer(monkeypatch)
    monkeypatch.setattr(ml.torch.cuda, "is_available", lambda: True)
    # load_model's own CUDA-available log line calls get_device_name(0),
    # which -- unlike is_available/synchronize -- isn't just a flag check
    # and would try to touch real CUDA state on this CPU-only torch build.
    monkeypatch.setattr(ml.torch.cuda, "get_device_name", lambda index=0: "fake-gpu")
    sync_calls = []
    monkeypatch.setattr(ml.torch.cuda, "synchronize", lambda: sync_calls.append(True))

    ml.load_model("phi-4-mini")

    assert sync_calls == [True]


def test_load_model_skips_cuda_synchronize_when_cuda_unavailable(monkeypatch):
    # This is the actual path exercised on this Windows/CPU-only dev
    # machine -- confirms the guard means synchronize() is never called at
    # all here, not just that it would no-op if it were.
    _stub_model_and_tokenizer(monkeypatch)
    monkeypatch.setattr(ml.torch.cuda, "is_available", lambda: False)
    sync_calls = []
    monkeypatch.setattr(ml.torch.cuda, "synchronize", lambda: sync_calls.append(True))

    ml.load_model("phi-4-mini")

    assert sync_calls == []


# --------------------------------------------------------------------------
# Fix 2 -- data/build_index.py's _get_embedder() retries with backoff.
# --------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_embedder_cache(monkeypatch):
    # _embedder is a module-level cache; reset it around each test so one
    # test's successful embedder doesn't leak into the next.
    monkeypatch.setattr(bi, "_embedder", None)
    monkeypatch.setattr(bi.time, "sleep", lambda seconds: None)  # no real waiting


def test_get_embedder_retries_and_succeeds_on_third_attempt(monkeypatch):
    calls = []

    def fake_sentence_transformer(model_name, device):
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("cudaErrorDevicesUnavailable")
        return mock.Mock(name="embedder")

    monkeypatch.setattr(bi, "SentenceTransformer", fake_sentence_transformer)

    result = bi._get_embedder()

    assert len(calls) == 3
    assert result is not None
    assert bi._embedder is result


def test_get_embedder_raises_original_exception_after_three_failures(monkeypatch):
    calls = []

    def always_fails(model_name, device):
        calls.append(1)
        raise RuntimeError(f"cudaErrorDevicesUnavailable attempt {len(calls)}")

    monkeypatch.setattr(bi, "SentenceTransformer", always_fails)

    with pytest.raises(RuntimeError, match="attempt 3"):
        bi._get_embedder()

    # exactly 3 attempts -- not swallowed, not retried indefinitely
    assert len(calls) == 3


def test_get_embedder_succeeds_immediately_without_retry_when_no_failure(monkeypatch):
    calls = []
    fake_embedder = mock.Mock(name="embedder")

    def fake_sentence_transformer(model_name, device):
        calls.append(1)
        return fake_embedder

    monkeypatch.setattr(bi, "SentenceTransformer", fake_sentence_transformer)

    result = bi._get_embedder()

    assert len(calls) == 1  # no retries needed
    assert result is fake_embedder
