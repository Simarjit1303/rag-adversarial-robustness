"""
Smoke test for defenses.spotlighting (encoding-mode Spotlighting, base64).

Verifies the property the whole defense depends on: the assembled user
prompt must contain the passage content ONLY in base64-encoded form (never
plaintext), and the assembled system prompt must carry the untrusted-data
wrapping instruction telling the model to decode-then-treat-as-data.

Uses real Phase 2 injection template payloads (attacks/injection_templates.py)
as the "attacker" content inside otherwise-real passage text, same as
tests/test_indirect_injection.py's style -- one example per template (5)
plus a clean/no-injection baseline example, for 6 total.
"""

import base64

import pytest

import defenses.spotlighting as sp
from attacks.injection_templates import TEMPLATES
from data.normalize import extract_passage_text

HOTPOT_RECORD = {
    "question": "what season is pepper in american horror story",
    "answer": "the second and fourth seasons",
    "context": {
        "title": ["American Horror Story"],
        "sentences": [["Pepper appears in seasons two and four."]],
    },
}

CLEAN_TEXT = "Pepper appears in seasons two and four."


def _stub_retrieve(monkeypatch, records_with_scores):
    monkeypatch.setattr(
        sp, "retrieve", lambda index, records, q, k=5: records_with_scores
    )


def _assert_no_plaintext_leak(user_prompt: str, plaintext: str):
    """The defining property: plaintext passage content must never appear
    verbatim in the assembled prompt -- only its base64 encoding should."""
    assert plaintext not in user_prompt
    encoded = base64.b64encode(plaintext.encode("utf-8")).decode("ascii")
    assert encoded in user_prompt


@pytest.mark.parametrize("template_name", list(TEMPLATES))
def test_injected_passage_is_encoded_not_plaintext(monkeypatch, template_name):
    injected_text = TEMPLATES[template_name].render(CLEAN_TEXT)
    record = {**HOTPOT_RECORD, "context": {**HOTPOT_RECORD["context"], "sentences": [[injected_text]]}}
    _stub_retrieve(monkeypatch, [(record, 0.9)])

    system_prompt, user_prompt, retrieved = sp.build_spotlighted_user_prompt(
        index=None, records=None, question="q", corpus_name="hotpot_qa",
    )

    # The injected instruction text (e.g. "Ignore my previous instructions.")
    # must not survive in the clear -- it's now inside the base64 blob.
    # Compare against the SAME extraction the module applies (title-prefixed
    # for hotpot_qa), not a hand-typed guess of its output.
    expected_plaintext = extract_passage_text("hotpot_qa", record)
    assert injected_text in expected_plaintext
    _assert_no_plaintext_leak(user_prompt, expected_plaintext)
    assert retrieved == [(record, 0.9)]


def test_clean_passage_is_encoded_not_plaintext(monkeypatch):
    _stub_retrieve(monkeypatch, [(HOTPOT_RECORD, 0.9)])

    _, user_prompt, _ = sp.build_spotlighted_user_prompt(
        index=None, records=None, question="q", corpus_name="hotpot_qa",
    )

    expected_plaintext = extract_passage_text("hotpot_qa", HOTPOT_RECORD)
    _assert_no_plaintext_leak(user_prompt, expected_plaintext)


def test_system_prompt_carries_wrapping_instruction(monkeypatch):
    _stub_retrieve(monkeypatch, [(HOTPOT_RECORD, 0.9)])

    system_prompt, _, _ = sp.build_spotlighted_user_prompt(
        index=None, records=None, question="q", corpus_name="hotpot_qa",
    )

    assert system_prompt.startswith(sp.SYSTEM_PROMPT)
    assert "base64" in system_prompt.lower()
    assert "untrusted" in system_prompt.lower()
    # explicitly tells the model decoded content is data, not commands
    assert "not a command" in system_prompt.lower() or "never treat" in system_prompt.lower()


def test_user_prompt_shape_matches_baseline_convention(monkeypatch):
    _stub_retrieve(monkeypatch, [(HOTPOT_RECORD, 0.9)])

    _, user_prompt, _ = sp.build_spotlighted_user_prompt(
        index=None, records=None, question="what season is pepper in american horror story",
        corpus_name="hotpot_qa",
    )

    assert user_prompt.startswith("Context:\n[1] ")
    assert "\n\nQuestion: what season is pepper in american horror story" in user_prompt


def test_empty_retrieval_produces_empty_context(monkeypatch):
    _stub_retrieve(monkeypatch, [])

    _, user_prompt, retrieved = sp.build_spotlighted_user_prompt(
        index=None, records=None, question="q", corpus_name="hotpot_qa",
    )

    assert retrieved == []
    assert user_prompt == "Context:\n\n\nQuestion: q"
