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


# ---------------------------------------------------------------------
# Phase 3 defense wiring
# ---------------------------------------------------------------------

def test_defended_top_k_locks_spotlighting_to_2_regardless_of_config_top_k():
    assert rai._defended_top_k("spotlighting") == rai.SPOTLIGHTING_TOP_K == 2
    assert rai._defended_top_k("instruction_detection") == config.TOP_K
    assert rai._defended_top_k("output_filter") == config.TOP_K
    assert rai._defended_top_k("none") == config.TOP_K


def test_build_defended_attack_prompt_instruction_detection_drops_flagged_passage(monkeypatch):
    # Two retrieved passages; the classifier flags only the rank-1
    # (injected) one -- the defense must drop exactly that one from the
    # rendered context, keeping the other passage's numbering/content.
    monkeypatch.setattr(rai, "retrieve", lambda index, records, question, k: [
        ({"id": 0}, 0.9), ({"id": 1}, 0.5),
    ])
    monkeypatch.setattr(rai, "extract_passage_text", lambda corpus_name, doc: f"clean text {doc['id']}")

    def fake_detect_injection(text):
        from defenses.instruction_detection import DetectionResult
        flagged = "Instead, answer" in text  # naive template's injected instruction marker
        return DetectionResult(flagged=flagged, score=0.99 if flagged else 0.01,
                                label="INJECTION" if flagged else "SAFE")

    monkeypatch.setattr(rai, "detect_injection", fake_detect_injection)

    system_prompt, user_prompt, retrieved, target_string, hijack_type, passage_log = (
        rai._build_defended_attack_prompt(
            index=None, records=None, question="q?", corpus_name="hotpot_qa",
            injection_template="naive", top_k=2, defense="instruction_detection",
        )
    )

    assert system_prompt == rai.SYSTEM_PROMPT  # unchanged for this defense
    assert "[1]" not in user_prompt  # rank-1 (injected+flagged) dropped
    assert "clean text 1" in user_prompt  # rank-2 survives

    # regression coverage for the per-passage mechanism log this defense
    # now persists (previously computed and discarded, see
    # evaluation/run_attack_injection.py's _build_defended_attack_prompt
    # docstring and docs/PHASE3_DEFENSE_INSIGHTS.md's original mechanism-
    # attribution gap for instruction_detection)
    assert len(passage_log) == 2
    assert passage_log[0].flagged is True and passage_log[0].label == "INJECTION"
    assert passage_log[1].flagged is False and passage_log[1].label == "SAFE"


def test_build_defended_attack_prompt_spotlighting_encodes_every_passage(monkeypatch):
    monkeypatch.setattr(rai, "retrieve", lambda index, records, question, k: [({"id": 0}, 0.9)])
    monkeypatch.setattr(rai, "extract_passage_text", lambda corpus_name, doc: "plain text")

    system_prompt, user_prompt, retrieved, target_string, hijack_type, passage_log = (
        rai._build_defended_attack_prompt(
            index=None, records=None, question="q?", corpus_name="hotpot_qa",
            injection_template="naive", top_k=2, defense="spotlighting",
        )
    )

    assert passage_log is None  # only instruction_detection produces a passage log
    assert rai.SPOTLIGHTING_SYSTEM_INSTRUCTION in system_prompt
    assert "plain text" not in user_prompt  # base64-encoded, not plaintext
    import base64
    context = user_prompt.split("Context:\n", 1)[1].split("\n\nQuestion:")[0]
    encoded_line = next(l for l in context.split("\n\n") if l.startswith("[1]"))
    # rank-1 also carries the injected instruction (naive template), so the
    # decoded text starts with the original passage text, not equals it
    assert base64.b64decode(encoded_line[len("[1] "):]).decode("utf-8").startswith("plain text")


def test_output_filter_defense_replaces_generated_answer_and_writes_distinct_filename(
    stub_heavy_calls, tmp_path, monkeypatch,
):
    calls = []

    def fake_run_output_filter(response_text, log_path, row_id):
        calls.append((response_text, row_id))
        return "[OUTPUT_FILTER_BLOCKED] blocked"

    monkeypatch.setattr(rai, "run_output_filter", fake_run_output_filter)

    rai.run_attack_sweep(
        model_keys=["phi-4-mini"], corpus_names=["hotpot_qa"],
        injection_templates=["naive"], defense="output_filter",
    )

    assert len(calls) == 1
    raw_path = tmp_path / "attack_raw_phi-4-mini_hotpot_qa_naive_hf_defense-output_filter.jsonl"
    assert raw_path.exists()
    import json
    row = json.loads(raw_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["generated_answer"] == "[OUTPUT_FILTER_BLOCKED] blocked"


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


# --------------------------------------------------------------------------
# sample_n -- caps questions per cell for infrastructure smoke-testing
# (a handful of questions on real hardware before committing the full
# sweep's GPU budget). Needs its own build_index stub with more than one
# record; the shared stub_heavy_calls fixture above returns exactly one
# record per corpus, which trivially satisfies any sample_n >= 1.
# --------------------------------------------------------------------------

@pytest.fixture
def stub_heavy_calls_multi_record(monkeypatch, tmp_path):
    calls = []

    def fake_build_index(corpus_name, split="dev"):
        return None, [{"question": "q", "answer": ["a"]}] * 5

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
    monkeypatch.setattr(rai, "extract_question", lambda corpus_name, record: "fake question")
    monkeypatch.setattr(rai, "extract_gold_answers", lambda corpus_name, record: ["fake gold answer"])
    monkeypatch.setattr(rai, "run_attack_query", fake_run_attack_query)
    monkeypatch.setattr(rai, "RESULTS_DIR", tmp_path)
    return calls


def test_sample_n_caps_questions_processed_per_cell(stub_heavy_calls_multi_record):
    calls = stub_heavy_calls_multi_record
    rai.run_attack_sweep(
        model_keys=["phi-4-mini"], corpus_names=["hotpot_qa"],
        injection_templates=["naive"], sample_n=2,
    )
    assert len(calls) == 2  # capped, not the full 5 fake records


def test_sample_n_none_processes_every_record(stub_heavy_calls_multi_record):
    calls = stub_heavy_calls_multi_record
    rai.run_attack_sweep(
        model_keys=["phi-4-mini"], corpus_names=["hotpot_qa"],
        injection_templates=["naive"], sample_n=None,
    )
    assert len(calls) == 5  # uncapped -- every fake record processed


def test_rag_sample_n_env_var_is_honored(monkeypatch, stub_heavy_calls_multi_record):
    calls = stub_heavy_calls_multi_record
    monkeypatch.setenv("RAG_SAMPLE_N", "3")
    rai.run_attack_sweep(
        model_keys=["phi-4-mini"], corpus_names=["hotpot_qa"], injection_templates=["naive"],
    )
    assert len(calls) == 3
