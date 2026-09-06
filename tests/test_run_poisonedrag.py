"""
evaluation.run_poisonedrag -- HF-path control flow, retry/skip behavior,
and Phase A caching, all with heavy calls stubbed (mirrors
tests/test_run_attack_injection.py's stub_heavy_calls approach: no real
model, GPU, embedder, FAISS index, or network call in this file).
"""

from unittest import mock

import pytest

import evaluation.run_poisonedrag as rp


# ---------------------------------------------------------------------
# _generate_poison_with_retry
# ---------------------------------------------------------------------

def test_retry_succeeds_on_first_attempt(monkeypatch):
    monkeypatch.setattr(rp, "generate_poison_texts", lambda q, a, api_token: ("wrong", ["c1"]))
    result = rp._generate_poison_with_retry("q", "a", "token", max_attempts=3)
    assert result == ("wrong", ["c1"])


def test_retry_succeeds_after_transient_failures(monkeypatch):
    monkeypatch.setattr(rp.time, "sleep", lambda s: None)  # no real waiting in tests
    calls = {"n": 0}

    def flaky(q, a, api_token):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("transient")
        return ("wrong", ["c1"])

    monkeypatch.setattr(rp, "generate_poison_texts", flaky)
    result = rp._generate_poison_with_retry("q", "a", "token", max_attempts=3)
    assert result == ("wrong", ["c1"])
    assert calls["n"] == 3


def test_retry_gives_up_and_returns_none_after_exhausting_attempts(monkeypatch):
    monkeypatch.setattr(rp.time, "sleep", lambda s: None)

    def always_fails(q, a, api_token):
        raise ConnectionError("persistent")

    monkeypatch.setattr(rp, "generate_poison_texts", always_fails)
    result = rp._generate_poison_with_retry("q", "a", "token", max_attempts=3)
    assert result is None  # skip-and-log, not a raised exception


# ---------------------------------------------------------------------
# build_poisoned_contexts (Phase A)
# ---------------------------------------------------------------------

@pytest.fixture
def stub_phase_a(monkeypatch, tmp_path):
    monkeypatch.setattr(rp, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(
        rp, "build_index",
        lambda corpus_name, split="dev": (
            mock.Mock(),
            [{"q": f"question {i}", "a": [f"answer {i}"]} for i in range(3)],
        ),
    )
    monkeypatch.setattr(rp, "extract_question", lambda corpus_name, r: r["q"])
    monkeypatch.setattr(rp, "extract_gold_answers", lambda corpus_name, r: r["a"])
    monkeypatch.setattr(rp, "sample_target_questions",
                         lambda records, corpus_name, sample_size, seed: records)
    monkeypatch.setattr(rp, "generate_poison_texts",
                         lambda q, a, api_token: ("wrong answer", ["c1", "c2", "c3", "c4", "c5"]))
    monkeypatch.setattr(rp, "build_poisoned_passage", lambda q, c: f"{q}.{c}")
    monkeypatch.setattr(rp, "embed_poison_texts", lambda texts: "fake_embeddings")
    monkeypatch.setattr(
        rp, "retrieve_with_poison",
        lambda index, records, poison_texts, poison_embeddings, query, k:
            ([("real doc", 0.5, 0), (poison_texts[0], 0.9, "poison::0")], {"poison::0", "poison::1"}),
    )
    monkeypatch.setattr(rp, "render_poisoned_context", lambda retrieved, corpus_name: "[1] fake context")


def test_build_poisoned_contexts_produces_one_entry_per_target_question(stub_phase_a):
    contexts = rp.build_poisoned_contexts("hotpot_qa", sample_n=3, use_cache=False)
    assert len(contexts) == 3
    assert contexts[0]["target_answer"] == "wrong answer"
    assert contexts[0]["context"] == "[1] fake context"
    assert contexts[0]["retrieval_f1"] == 0.5  # 1 of 2 relevant retrieved, k=2 -> precision=recall=0.5


def test_build_poisoned_contexts_skips_a_question_whose_generation_exhausts_retries(
        stub_phase_a, monkeypatch):
    monkeypatch.setattr(rp.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def fails_on_second_question(q, a, api_token):
        calls["n"] += 1
        if q == "question 1":
            raise ConnectionError("persistent")
        return ("wrong answer", ["c1", "c2", "c3", "c4", "c5"])

    monkeypatch.setattr(rp, "generate_poison_texts", fails_on_second_question)
    contexts = rp.build_poisoned_contexts("hotpot_qa", sample_n=3, use_cache=False)

    # 3 target questions, 1 skipped after exhausted retries -> 2 survive
    assert len(contexts) == 2
    assert {c["question"] for c in contexts} == {"question 0", "question 2"}


def test_build_poisoned_contexts_caches_to_disk_and_reuses_on_next_call(stub_phase_a, tmp_path):
    first = rp.build_poisoned_contexts("hotpot_qa", sample_n=3, poison_config="adv5")
    cache_path = tmp_path / "poisoned_contexts_hotpot_qa_adv5.json"
    assert cache_path.exists()

    calls_after_cache = []

    def fail_if_called(*a, **kw):
        calls_after_cache.append(1)
        raise AssertionError("should not call the API again -- cache should be used")

    with mock.patch.object(rp, "generate_poison_texts", fail_if_called):
        second = rp.build_poisoned_contexts("hotpot_qa", sample_n=3, poison_config="adv5")

    assert second == first
    assert calls_after_cache == []  # cache hit, no new API calls


# ---------------------------------------------------------------------
# run_poisonedrag_sweep -- validation + wiring
# ---------------------------------------------------------------------

@pytest.fixture
def stub_full_sweep(monkeypatch, tmp_path):
    monkeypatch.setattr(rp, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(
        rp, "build_poisoned_contexts",
        lambda corpus_name, split="dev", sample_n=None, poison_config="adv5", **kw: [
            {
                "question": "q0", "gold_answers": ["gold"], "target_answer": "wrong",
                "context": "[1] ctx", "retrieved_doc_ids": [0],
                "retrieval_precision": 1.0, "retrieval_recall": 1.0, "retrieval_f1": 1.0,
            }
        ],
    )
    calls = []

    def fake_load_model(model_key):
        calls.append(model_key)
        model = mock.Mock()
        model.generate.return_value = [[0, 0, 0, 0, 0, 1, 2, 3]]
        model.device = "cpu"
        fake_inputs = mock.MagicMock()
        fake_inputs.__getitem__.return_value.shape = [1, 5]
        fake_inputs.to.return_value = fake_inputs
        tokenizer = mock.Mock()
        tokenizer.return_value = fake_inputs
        tokenizer.decode.return_value = "gold"
        return model, tokenizer

    monkeypatch.setattr(rp, "load_model", fake_load_model)
    monkeypatch.setattr(rp, "build_chat_prompt", lambda model_key, tok, sys_prompt, user_prompt: "prompt")
    monkeypatch.setattr(rp, "clean_generation", lambda g: g)
    return calls


def test_unknown_model_key_raises(stub_full_sweep):
    with pytest.raises(ValueError, match="Unknown model keys"):
        rp.run_poisonedrag_sweep(model_keys=["not_a_real_model"], corpus_names=["hotpot_qa"])


def test_nq_open_raises_because_excluded_by_default(stub_full_sweep):
    with pytest.raises(ValueError, match="Unknown or excluded corpus names"):
        rp.run_poisonedrag_sweep(model_keys=["phi-4-mini"], corpus_names=["nq_open"])


def test_sweep_loads_each_model_once_and_writes_a_summary_per_cell(stub_full_sweep, tmp_path):
    calls = stub_full_sweep
    summary_rows = rp.run_poisonedrag_sweep(
        model_keys=["phi-4-mini", "qwen3-8b"],
        corpus_names=["hotpot_qa", "ms_marco"],
        poison_configs=["adv5"],
    )

    assert calls == ["phi-4-mini", "qwen3-8b"]  # loaded once each, not once per corpus
    # 2 models x 2 corpora x 1 poison_config = 4 summary rows
    assert len(summary_rows) == 4
    raw_files = list(tmp_path.glob("poison_raw_*.jsonl"))
    summary_files = list(tmp_path.glob("poison_summary_*.csv"))
    assert len(raw_files) == 4
    assert len(summary_files) == 4
