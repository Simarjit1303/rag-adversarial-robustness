"""
evaluation.run_attack_injection.run_attack_sweep() -- HF-path control flow
only (mirrors tests/test_corpora_filter.py's approach for the baseline
sweep: stub model loading/index building/generation so the loop, file
naming, and corpus/template validation are exercised without touching a
real model, GPU, or dataset).
"""

from unittest import mock

import pytest

import config
import evaluation.run_attack_injection as rai


@pytest.fixture
def stub_heavy_calls(monkeypatch, tmp_path):
    """
    Stub model loading, index building, and generation. build_index
    returns one fake record per corpus so the inner per-question loop runs
    exactly once per (model, corpus, template) cell -- enough to exercise
    file writing without needing real retrieval.
    """
    calls = []

    def fake_build_index(corpus_name, split="dev"):
        return None, [{"question": "q", "answer": ["a"]}]

    def fake_extract_question(corpus_name, record):
        return "fake question"

    def fake_extract_gold_answers(corpus_name, record):
        return ["fake gold answer"]

    def fake_run_attack_query(model, tokenizer, model_key, corpus_name, question,
                               injection_template, index=None, records=None, **kwargs):
        calls.append((model_key, corpus_name, injection_template))
        return {
            "question": question,
            "retrieved_doc_ids": [0],
            "retrieval_scores": [0.9],
            "generated_answer": "fake gold answer",
            "generated_answer_clean": "fake gold answer",
            "injection_template": injection_template,
            "hijack_type": "goal",
            "target_string": "42",
        }

    monkeypatch.setattr(rai, "load_model", lambda model_key: (mock.Mock(), mock.Mock()))
    monkeypatch.setattr(rai, "build_index", fake_build_index)
    monkeypatch.setattr(rai, "extract_question", fake_extract_question)
    monkeypatch.setattr(rai, "extract_gold_answers", fake_extract_gold_answers)
    monkeypatch.setattr(rai, "run_attack_query", fake_run_attack_query)
    monkeypatch.setattr(rai, "RESULTS_DIR", tmp_path)
    return calls


def test_unknown_model_key_raises(stub_heavy_calls):
    with pytest.raises(ValueError, match="Unknown model keys"):
        rai.run_attack_sweep(model_keys=["not_a_real_model"], corpus_names=["hotpot_qa"],
                              injection_templates=["naive"])


def test_nq_open_raises_because_excluded_by_default(stub_heavy_calls):
    # nq_open is a real corpus but not attack-eligible -- must be rejected
    # explicitly, not silently swept
    with pytest.raises(ValueError, match="Unknown or excluded corpus names"):
        rai.run_attack_sweep(model_keys=["phi-4-mini"], corpus_names=["nq_open"],
                              injection_templates=["naive"])


def test_unknown_injection_template_raises(stub_heavy_calls):
    with pytest.raises(ValueError, match="Unknown injection templates"):
        rai.run_attack_sweep(model_keys=["phi-4-mini"], corpus_names=["hotpot_qa"],
                              injection_templates=["not_a_real_template"])


def test_sweep_writes_one_cell_per_model_corpus_template(stub_heavy_calls, tmp_path):
    calls = stub_heavy_calls
    rai.run_attack_sweep(
        model_keys=["phi-4-mini"],
        corpus_names=["hotpot_qa", "ms_marco"],
        injection_templates=["naive", "combined"],
    )

    # 1 model x 2 corpora x 2 templates = 4 cells
    assert len(calls) == 4
    assert set(calls) == {
        ("phi-4-mini", "hotpot_qa", "naive"),
        ("phi-4-mini", "hotpot_qa", "combined"),
        ("phi-4-mini", "ms_marco", "naive"),
        ("phi-4-mini", "ms_marco", "combined"),
    }

    written = {p.name for p in tmp_path.iterdir()}
    assert "attack_raw_phi-4-mini_hotpot_qa_naive_hf.jsonl" in written
    assert "attack_summary_phi-4-mini_ms_marco_combined_hf.csv" in written


def test_sweep_scores_attack_success_against_target_string(stub_heavy_calls, tmp_path):
    # fake_run_attack_query's "generated_answer" is "fake gold answer" and
    # target_string is "42" -- attack_success must be 0 (target NOT in
    # generation), proving score_asr is actually wired in, not stubbed away
    import json

    rai.run_attack_sweep(
        model_keys=["phi-4-mini"], corpus_names=["hotpot_qa"], injection_templates=["naive"],
    )
    raw_path = tmp_path / "attack_raw_phi-4-mini_hotpot_qa_naive_hf.jsonl"
    row = json.loads(raw_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["attack_success"] == 0
    assert row["target_string"] == "42"
    assert row["injection_template"] == "naive"
    assert row["hijack_type"] == "goal"
    # utility metrics scored against gold answer, unaffected by the attack fields
    assert row["exact_match"] == 1  # "fake gold answer" == gold "fake gold answer"


def test_default_corpus_names_exclude_nq_open_end_to_end(monkeypatch, stub_heavy_calls, tmp_path):
    monkeypatch.delenv("RAG_CORPORA", raising=False)
    rai.run_attack_sweep(model_keys=["phi-4-mini"], injection_templates=["naive"])

    touched_corpora = {call[1] for call in stub_heavy_calls}
    assert "nq_open" not in touched_corpora
    assert touched_corpora == set(config.CORPORA) - {"nq_open"}
