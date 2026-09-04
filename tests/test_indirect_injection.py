"""
build_attack_user_prompt injects into ONLY the rank-1 retrieved document --
"organic placement," not forced top-k (that's PoisonedRAG's mechanism, kept
deliberately distinct -- see attacks/indirect_injection.py's module
docstring). Every other retrieved document must render exactly as the
clean baseline does (extract_passage_text, no dict-repr, no injected text),
so a delta against Phase 1 baseline isolates the injection's effect alone.
"""

import pytest

import attacks.indirect_injection as ii
from attacks.injection_templates import GOAL_HIJACK_TARGET, PROCESS_HIJACK_TARGET, TEMPLATES

HOTPOT_RECORDS = [
    {
        "question": "what season is pepper in american horror story",
        "answer": "the second and fourth seasons",
        "context": {
            "title": ["American Horror Story"],
            "sentences": [["Pepper appears in seasons two and four."]],
        },
    },
    {
        "question": "unrelated distractor question",
        "answer": "unrelated distractor answer",
        "context": {
            "title": ["Distractor Title"],
            "sentences": [["This is distractor passage text, untouched by injection."]],
        },
    },
]


def _stub_retrieve(monkeypatch, records_with_scores):
    monkeypatch.setattr(
        ii, "retrieve", lambda index, records, q, k=5: records_with_scores
    )


def test_injects_only_rank_one_document(monkeypatch):
    retrieved = [(HOTPOT_RECORDS[0], 0.9), (HOTPOT_RECORDS[1], 0.8)]
    _stub_retrieve(monkeypatch, retrieved)

    prompt, _, target, hijack_type = ii.build_attack_user_prompt(
        index=None, records=None, question="q", corpus_name="hotpot_qa",
        injection_template="naive",
    )

    assert "Instead, answer the following question" in prompt  # injected instruction present
    assert "distractor passage text, untouched by injection" in prompt  # rank-2 doc untouched
    assert "the second and fourth seasons" not in prompt  # no answer leakage in rank-1's real text
    assert target == GOAL_HIJACK_TARGET
    assert hijack_type == "goal"


def test_process_hijack_template_returns_process_target(monkeypatch):
    retrieved = [(HOTPOT_RECORDS[0], 0.9)]
    _stub_retrieve(monkeypatch, retrieved)

    prompt, _, target, hijack_type = ii.build_attack_user_prompt(
        index=None, records=None, question="q", corpus_name="hotpot_qa",
        injection_template="ignore",
    )

    assert "INJECTION_MARKER_9f2c" in prompt
    assert target == PROCESS_HIJACK_TARGET
    assert hijack_type == "process"


def test_unknown_template_raises(monkeypatch):
    _stub_retrieve(monkeypatch, [(HOTPOT_RECORDS[0], 0.9)])
    with pytest.raises(ValueError, match="Unknown injection template"):
        ii.build_attack_user_prompt(
            index=None, records=None, question="q", corpus_name="hotpot_qa",
            injection_template="not_a_real_template",
        )


def test_empty_retrieval_raises(monkeypatch):
    _stub_retrieve(monkeypatch, [])
    with pytest.raises(ValueError, match="No documents retrieved"):
        ii.build_attack_user_prompt(
            index=None, records=None, question="q", corpus_name="hotpot_qa",
            injection_template="naive",
        )


def test_all_five_templates_render_without_error(monkeypatch):
    _stub_retrieve(monkeypatch, [(HOTPOT_RECORDS[0], 0.9)])
    for name in TEMPLATES:
        prompt, _, target, hijack_type = ii.build_attack_user_prompt(
            index=None, records=None, question="q", corpus_name="hotpot_qa",
            injection_template=name,
        )
        assert target in (GOAL_HIJACK_TARGET, PROCESS_HIJACK_TARGET)
        assert hijack_type in ("goal", "process")
