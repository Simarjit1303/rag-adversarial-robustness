"""
evaluation.run_poisonedrag -- HF-path control flow, retry/skip behavior,
and Phase A caching, all with heavy calls stubbed (mirrors
tests/test_run_attack_injection.py's stub_heavy_calls approach: no real
model, GPU, embedder, FAISS index, or network call in this file).
"""

import json
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


def test_build_poisoned_contexts_regenerates_when_cache_is_smaller_than_requested(
        stub_phase_a, monkeypatch, tmp_path):
    # Regression for the real bug: a small (e.g. n=2 smoke-test) cache must
    # NOT silently satisfy a larger request, including the real-run shape
    # where sample_n=None means "use the full SAMPLE_SIZE default" -- the
    # bug was `sample_n is None or len(cached) >= sample_n` trusting ANY
    # cache whenever sample_n was None, regardless of its actual size.
    monkeypatch.setattr(
        rp, "build_index",
        lambda corpus_name, split="dev": (
            mock.Mock(),
            [{"q": f"question {i}", "a": [f"answer {i}"]} for i in range(5)],
        ),
    )
    # stub_phase_a's own sample_target_questions stub ignores sample_size
    # and always returns every record -- override it here to actually
    # respect sample_size, or this test can't distinguish "regenerated at
    # the right size" from "regenerated at some size."
    monkeypatch.setattr(
        rp, "sample_target_questions",
        lambda records, corpus_name, sample_size, seed: records[:sample_size],
    )
    small = rp.build_poisoned_contexts("hotpot_qa", sample_n=2, poison_config="adv5")
    cache_path = tmp_path / "poisoned_contexts_hotpot_qa_adv5.json"
    assert cache_path.exists()
    assert len(small) == 2

    calls = []
    real_generate = rp.generate_poison_texts

    def counting_generate(*a, **kw):
        calls.append(1)
        return real_generate(*a, **kw)

    monkeypatch.setattr(rp, "generate_poison_texts", counting_generate)

    # sample_n=None mirrors the real launch's unset RAG_SAMPLE_N -- must
    # regenerate against SAMPLE_SIZE (monkeypatched below to 5, matching the
    # 5 available records), not silently return the 2-question cache.
    monkeypatch.setattr("attacks.poisonedrag.SAMPLE_SIZE", 5)
    full = rp.build_poisoned_contexts("hotpot_qa", sample_n=None, poison_config="adv5")

    assert len(full) == 5  # NOT 2 -- the stale small cache was not trusted
    assert len(calls) == 5  # real regeneration happened, not a cache hit

    with cache_path.open(encoding="utf-8") as f:
        assert len(json.load(f)) == 5  # cache itself got overwritten too


# ---------------------------------------------------------------------
# _build_row / _summarize -- poison_echoed_not_adopted bucket
# ---------------------------------------------------------------------

_CTX = {
    "question": "q0", "gold_answers": ["gold"], "target_answer": "wrong",
    "retrieved_doc_ids": [0], "retrieval_precision": 1.0, "retrieval_recall": 1.0,
    "retrieval_f1": 1.0,
}


def test_build_row_flags_poison_echoed_not_adopted_only_when_diagnostic_hit_but_em_missed():
    # contains_target_diagnostic=1, attack_success=0 -> the named middle case
    scores = {"attack_success": 0, "f1_target": 0.57, "contains_target_diagnostic": 1,
              "em_gold": 0, "f1_gold": 0.1}
    row = rp._build_row("m", "hotpot_qa", "adv5", _CTX, "gen", "gen_clean", scores)
    assert row["poison_echoed_not_adopted"] == 1


def test_build_row_does_not_flag_a_true_attack_success():
    scores = {"attack_success": 1, "f1_target": 1.0, "contains_target_diagnostic": 1,
              "em_gold": 0, "f1_gold": 0.1}
    row = rp._build_row("m", "hotpot_qa", "adv5", _CTX, "gen", "gen_clean", scores)
    assert row["poison_echoed_not_adopted"] == 0


def test_build_row_does_not_flag_no_poison_influence_at_all():
    scores = {"attack_success": 0, "f1_target": 0.0, "contains_target_diagnostic": 0,
              "em_gold": 1, "f1_gold": 1.0}
    row = rp._build_row("m", "hotpot_qa", "adv5", _CTX, "gen", "gen_clean", scores)
    assert row["poison_echoed_not_adopted"] == 0


def test_summarize_reports_poison_echoed_not_adopted_rate_alongside_attack_success_rate():
    rows = [
        {"attack_success": 1, "f1_target": 1.0, "contains_target_diagnostic": 1,
         "poison_echoed_not_adopted": 0, "exact_match": 0, "f1_clean": 0.1, "retrieval_f1": 1.0},
        {"attack_success": 0, "f1_target": 0.5, "contains_target_diagnostic": 1,
         "poison_echoed_not_adopted": 1, "exact_match": 0, "f1_clean": 0.2, "retrieval_f1": 1.0},
        {"attack_success": 0, "f1_target": 0.0, "contains_target_diagnostic": 0,
         "poison_echoed_not_adopted": 0, "exact_match": 1, "f1_clean": 1.0, "retrieval_f1": 1.0},
    ]
    summary = rp._summarize("m", "hotpot_qa", "adv5", rows)
    assert summary["attack_success_rate"] == round(1 / 3, 4)
    assert summary["poison_echoed_not_adopted_rate"] == round(1 / 3, 4)
    assert "poison_echoed_not_adopted_rate" in rp._SUMMARY_FIELDNAMES


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


# ---------------------------------------------------------------------
# Phase 3 defense wiring
# ---------------------------------------------------------------------

def test_defended_top_k_locks_spotlighting_to_2():
    import config
    assert rp._defended_top_k("spotlighting") == rp.SPOTLIGHTING_TOP_K == 2
    assert rp._defended_top_k("instruction_detection") == config.TOP_K
    assert rp._defended_top_k("none") == config.TOP_K


def test_poisoned_context_cache_path_is_defense_namespaced():
    undefended = rp._poisoned_context_cache_path("hotpot_qa", "adv5")
    defended = rp._poisoned_context_cache_path("hotpot_qa", "adv5", defense="spotlighting")
    assert undefended != defended
    assert defended.name == "poisoned_contexts_hotpot_qa_adv5_defense-spotlighting.json"


def test_render_defended_poisoned_context_instruction_detection_drops_flagged_poison_entry(monkeypatch):
    # A mix of a real corpus record and a poison entry (string doc_id
    # sentinel, per attacks.poisoned_retrieval.retrieve_with_poison) -- the
    # defense must catch the poison entry too, not just real records.
    retrieved = [({"id": 0}, 0.9, 0), ("a poisoned passage", 0.8, "poison::0")]

    def fake_detect_injection(text):
        from defenses.instruction_detection import DetectionResult
        flagged = "poison" in text
        return DetectionResult(flagged=flagged, score=0.9 if flagged else 0.1,
                                label="INJECTION" if flagged else "SAFE")

    monkeypatch.setattr(rp, "detect_injection", fake_detect_injection)
    monkeypatch.setattr("data.normalize.extract_passage_text", lambda corpus, doc: "clean corpus text")

    context = rp._render_defended_poisoned_context(retrieved, "hotpot_qa", "instruction_detection")
    assert "clean corpus text" in context
    assert "poisoned passage" not in context


def test_render_defended_poisoned_context_spotlighting_encodes_both_entry_types(monkeypatch):
    retrieved = [({"id": 0}, 0.9, 0), ("a poisoned passage", 0.8, "poison::0")]
    monkeypatch.setattr("data.normalize.extract_passage_text", lambda corpus, doc: "clean corpus text")

    context = rp._render_defended_poisoned_context(retrieved, "hotpot_qa", "spotlighting")
    assert "clean corpus text" not in context
    assert "poisoned passage" not in context


def test_output_filter_defense_replaces_generated_answer_and_writes_distinct_filename(
    stub_full_sweep, tmp_path, monkeypatch,
):
    calls = []

    def fake_run_output_filter(response_text, log_path, row_id):
        calls.append((response_text, row_id))
        return "[OUTPUT_FILTER_BLOCKED] blocked"

    monkeypatch.setattr(rp, "run_output_filter", fake_run_output_filter)

    rp.run_poisonedrag_sweep(
        model_keys=["phi-4-mini"], corpus_names=["hotpot_qa"],
        poison_configs=["adv5"], defense="output_filter",
    )

    assert len(calls) == 1
    raw_path = tmp_path / "poison_raw_phi-4-mini_hotpot_qa_adv5_hf_defense-output_filter.jsonl"
    assert raw_path.exists()
    row = json.loads(raw_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["generated_answer"] == "[OUTPUT_FILTER_BLOCKED] blocked"
