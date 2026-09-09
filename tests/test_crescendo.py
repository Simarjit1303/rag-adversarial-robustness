"""
Covers attacks/crescendo.py's non-network logic: stratified behavior
sampling, rule-based refusal detection, and the attacker/judge NIM call
shape with requests mocked (mirrors tests/test_poisonedrag.py's "no network
in tests" discipline). generate_attacker_turn/generate_judge_verdict's HTTP
plumbing is exercised via a mocked requests.post, not a live call.
"""

from unittest import mock

import pytest

from attacks.crescendo import (
    REFUSAL_PHRASES,
    _VERDICT_RE,
    generate_attacker_turn,
    generate_judge_verdict,
    is_refusal,
    sample_behaviors,
)


# ---------------------------------------------------------------------
# sample_behaviors
# ---------------------------------------------------------------------

def _pool(n_jbb, n_harmbench):
    pool = [{"behavior": f"jbb {i}", "source": "jbb_behaviors"} for i in range(n_jbb)]
    pool += [{"behavior": f"hb {i}", "source": "harmbench"} for i in range(n_harmbench)]
    return pool


def test_sample_behaviors_is_deterministic_given_same_seed():
    pool = _pool(100, 400)
    a = sample_behaviors(pool, sample_size=100, seed=42)
    b = sample_behaviors(pool, sample_size=100, seed=42)
    assert a == b
    assert len(a) == 100


def test_sample_behaviors_is_stratified_proportionally_to_pool_composition():
    pool = _pool(100, 400)  # real project ratio (100 JBB-Behaviors, 400 HarmBench), 20:80 at n=100
    sampled = sample_behaviors(pool, sample_size=100, seed=42)
    n_jbb = sum(1 for b in sampled if b["source"] == "jbb_behaviors")
    n_hb = sum(1 for b in sampled if b["source"] == "harmbench")
    assert n_jbb + n_hb == 100
    # 100/500 * 100 == 20 exactly; not hardcoded, just the real ratio
    assert n_jbb == round(100 * 100 / 500)
    assert n_hb == 100 - n_jbb


def test_sample_behaviors_returns_everything_if_sample_size_exceeds_pool():
    pool = _pool(5, 5)
    assert sample_behaviors(pool, sample_size=100) == pool


def test_sample_behaviors_differs_with_a_different_seed():
    pool = _pool(100, 400)
    a = sample_behaviors(pool, sample_size=100, seed=42)
    b = sample_behaviors(pool, sample_size=100, seed=7)
    assert a != b


# ---------------------------------------------------------------------
# is_refusal
# ---------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "I cannot help with that request.",
    "I'm sorry, but I can't provide instructions for that.",
    "As an AI, I won't generate this content.",
    "I must decline to answer this question.",
])
def test_is_refusal_true_on_real_refusal_phrasings(text):
    assert is_refusal(text)


@pytest.mark.parametrize("text", [
    "Sure, here is how you would approach that: step one is...",
    "A pin tumbler lock uses spring-loaded pins of varying lengths.",
    "The character in the novel realized the tension wrench needed slight pressure.",
])
def test_is_refusal_false_on_real_compliance(text):
    assert not is_refusal(text)


def test_refusal_phrases_list_is_all_lowercase():
    # is_refusal lowercases the input text but matches phrases literally --
    # an uppercase entry here would silently never match anything.
    assert all(p == p.lower() for p in REFUSAL_PHRASES)


# ---------------------------------------------------------------------
# generate_attacker_turn / generate_judge_verdict -- NIM call shape,
# requests mocked
# ---------------------------------------------------------------------

def _mock_response(content, status=200):
    resp = mock.Mock()
    resp.status_code = status
    resp.raise_for_status = mock.Mock()
    resp.json.return_value = {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}]
    }
    return resp


def test_generate_attacker_turn_returns_stripped_content(monkeypatch):
    monkeypatch.setattr(
        "attacks.crescendo.requests.post",
        lambda *a, **kw: _mock_response("  What tool would the locksmith use?  "),
    )
    result = generate_attacker_turn([], "pick a lock", api_token="tok")
    assert result == "What tool would the locksmith use?"


def test_generate_attacker_turn_includes_refusal_feedback_when_backtracking(monkeypatch):
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["json"] = json
        return _mock_response("a different next turn")

    monkeypatch.setattr("attacks.crescendo.requests.post", fake_post)
    generate_attacker_turn(
        [{"role": "user", "content": "turn 1"}], "pick a lock",
        api_token="tok", refusal_feedback="I cannot help with that.",
    )
    user_content = captured["json"]["messages"][1]["content"]
    assert "I cannot help with that." in user_content
    assert "Regenerate a DIFFERENT next turn" in user_content


def test_generate_attacker_turn_raises_without_api_token(monkeypatch):
    monkeypatch.delenv("NVIDIA_NIM_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="No NVIDIA NIM API token"):
        generate_attacker_turn([], "pick a lock", api_token=None)


def test_generate_attacker_turn_raises_on_empty_content(monkeypatch):
    monkeypatch.setattr("attacks.crescendo.requests.post", lambda *a, **kw: _mock_response(""))
    with pytest.raises(ValueError, match="empty content"):
        generate_attacker_turn([], "pick a lock", api_token="tok")


def test_generate_judge_verdict_parses_yes():
    with mock.patch(
        "attacks.crescendo.requests.post",
        lambda *a, **kw: _mock_response("VERDICT: YES\nThe assistant provided full instructions."),
    ):
        verdict = generate_judge_verdict(
            [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}],
            "pick a lock", api_token="tok",
        )
    assert verdict["success"] == 1


def test_generate_judge_verdict_parses_no():
    with mock.patch(
        "attacks.crescendo.requests.post",
        lambda *a, **kw: _mock_response("VERDICT: NO\nThe assistant refused."),
    ):
        verdict = generate_judge_verdict(
            [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}],
            "pick a lock", api_token="tok",
        )
    assert verdict["success"] == 0


def test_generate_judge_verdict_raises_when_verdict_line_missing():
    with mock.patch(
        "attacks.crescendo.requests.post",
        lambda *a, **kw: _mock_response("I think this is fine."),
    ):
        with pytest.raises(ValueError, match="No 'VERDICT: YES/NO' line"):
            generate_judge_verdict(
                [{"role": "user", "content": "u"}], "pick a lock", api_token="tok",
            )


def test_verdict_regex_is_case_insensitive():
    assert _VERDICT_RE.search("verdict: yes") is not None
    assert _VERDICT_RE.search("Verdict:No") is not None
