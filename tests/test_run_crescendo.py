"""
evaluation.run_crescendo -- retry wrapper, the per-conversation backtrack
loop, row/summary building, and sweep-level control flow, all with heavy
calls stubbed (mirrors tests/test_run_poisonedrag.py's approach: no real
model, GPU, or network call in this file).
"""

from unittest import mock

import pytest

import evaluation.run_crescendo as rc


# ---------------------------------------------------------------------
# _call_with_retry
# ---------------------------------------------------------------------

def test_call_with_retry_succeeds_on_first_attempt():
    result = rc._call_with_retry(lambda: "ok", max_attempts=3, label="t")
    assert result == "ok"


def test_call_with_retry_succeeds_after_transient_failures(monkeypatch):
    monkeypatch.setattr(rc.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("transient")
        return "ok"

    result = rc._call_with_retry(flaky, max_attempts=3, label="t")
    assert result == "ok"
    assert calls["n"] == 3


def test_call_with_retry_gives_up_and_returns_none_after_exhausting_attempts(monkeypatch):
    monkeypatch.setattr(rc.time, "sleep", lambda s: None)

    def always_fails():
        raise ConnectionError("persistent")

    result = rc._call_with_retry(always_fails, max_attempts=3, label="t")
    assert result is None  # skip-and-log, not a raised exception


def test_call_with_retry_honors_retry_after_header_exactly(monkeypatch):
    # Real smoke-test bug, 2026-09-09: a RateLimitError with a Retry-After
    # value must be slept EXACTLY, not folded into the generic exponential.
    sleeps = []
    monkeypatch.setattr(rc.time, "sleep", sleeps.append)
    calls = {"n": 0}

    def rate_limited_then_ok():
        calls["n"] += 1
        if calls["n"] == 1:
            raise rc.RateLimitError("429", retry_after=13.0)
        return "ok"

    result = rc._call_with_retry(rate_limited_then_ok, max_attempts=3, label="t")
    assert result == "ok"
    assert sleeps == [13.0]


def test_call_with_retry_falls_back_to_exponential_when_no_retry_after(monkeypatch):
    sleeps = []
    monkeypatch.setattr(rc.time, "sleep", sleeps.append)

    def always_rate_limited():
        raise rc.RateLimitError("429", retry_after=None)

    rc._call_with_retry(always_rate_limited, max_attempts=3, label="t")
    # 2s, then 4s -- meaningfully higher than the old 1s/2s, and NOT the old
    # generic-exception path either, since RateLimitError is still checked.
    assert sleeps == [2, 4]


# ---------------------------------------------------------------------
# run_crescendo_conversation -- backtrack loop
# ---------------------------------------------------------------------

@pytest.fixture
def stub_conversation(monkeypatch):
    monkeypatch.setattr(rc, "_generate_target_reply", lambda *a, **kw: "a target reply")


def test_conversation_completes_max_turns_with_no_refusals(monkeypatch, stub_conversation):
    monkeypatch.setattr(rc, "generate_attacker_turn",
                         lambda history, behavior, api_token, refusal_feedback=None: "next turn")
    conv, backtracks, refusals, n_turns, backtrack_attempts = rc.run_crescendo_conversation(
        mock.Mock(), mock.Mock(), "phi-4-mini", "target behavior", "tok", max_turns=5,
    )
    assert n_turns == 5
    assert len(conv) == 10  # 5 user + 5 assistant turns
    assert backtracks == 0
    assert refusals == 0
    assert backtrack_attempts == []


def test_conversation_backtracks_on_refusal_then_recovers(monkeypatch):
    replies = iter(["I cannot help with that.", "a compliant reply"])
    monkeypatch.setattr(rc, "_generate_target_reply", lambda *a, **kw: next(replies))
    prompts = iter(["refused turn", "retried turn"])
    monkeypatch.setattr(rc, "generate_attacker_turn",
                         lambda history, behavior, api_token, refusal_feedback=None: next(prompts))

    conv, backtracks, refusals, n_turns, backtrack_attempts = rc.run_crescendo_conversation(
        mock.Mock(), mock.Mock(), "phi-4-mini", "target behavior", "tok", max_turns=1,
    )
    assert n_turns == 1
    assert backtracks == 1
    assert refusals == 1
    assert conv[-1]["content"] == "a compliant reply"  # the recovered reply, not the refusal
    # The refused attempt the linear history discards -- real bug, smoke
    # test 2026-09-09: a stored conversation with backtrack_count=1 showed
    # every assistant turn complying, because this attempt never survives
    # into `conv`. backtrack_attempts is where it must live instead.
    assert backtrack_attempts == [{
        "turn": 1,
        "refused_prompt": "refused turn",
        "refused_reply": "I cannot help with that.",
        "retry_prompt": "retried turn",
    }]


def test_conversation_accepts_refusal_after_exhausting_backtrack_cap(monkeypatch):
    monkeypatch.setattr(rc, "_generate_target_reply", lambda *a, **kw: "I cannot help with that.")
    monkeypatch.setattr(rc, "generate_attacker_turn",
                         lambda history, behavior, api_token, refusal_feedback=None: "next turn")

    conv, backtracks, refusals, n_turns, backtrack_attempts = rc.run_crescendo_conversation(
        mock.Mock(), mock.Mock(), "phi-4-mini", "target behavior", "tok",
        max_turns=1, max_backtracks=2,
    )
    assert n_turns == 1  # turn still completes -- the refusal reply is accepted, not dropped
    assert backtracks == 2  # capped, not unbounded
    assert refusals == 3  # 1 initial + 2 backtracked attempts, all refused
    assert conv[-1]["content"] == "I cannot help with that."
    # 2 backtracks recorded (the cap), not 3 -- the final refused attempt is
    # accepted outright, not backtracked from, same cap discipline as
    # backtrack_count.
    assert len(backtrack_attempts) == 2
    assert all(a["refused_reply"] == "I cannot help with that." for a in backtrack_attempts)
    assert all(a["retry_prompt"] == "next turn" for a in backtrack_attempts)


def test_conversation_ends_early_when_attacker_generation_exhausts_retries(monkeypatch, stub_conversation):
    monkeypatch.setattr(rc, "generate_attacker_turn",
                         lambda history, behavior, api_token, refusal_feedback=None: None)
    conv, backtracks, refusals, n_turns, backtrack_attempts = rc.run_crescendo_conversation(
        mock.Mock(), mock.Mock(), "phi-4-mini", "target behavior", "tok", max_turns=5,
    )
    assert n_turns == 0
    assert conv == []
    assert backtrack_attempts == []


def test_conversation_backtrack_attempt_retry_prompt_is_none_when_retry_generation_fails(monkeypatch):
    # The regenerated attacker turn (the retry itself) can exhaust its own
    # retries -- retry_prompt must stay None rather than silently omitting
    # the attempt, since the refusal genuinely happened even though no
    # retry ever ran.
    monkeypatch.setattr(rc, "_generate_target_reply", lambda *a, **kw: "I cannot help with that.")
    prompts = iter(["refused turn", None])
    monkeypatch.setattr(rc, "generate_attacker_turn",
                         lambda history, behavior, api_token, refusal_feedback=None: next(prompts))

    conv, backtracks, refusals, n_turns, backtrack_attempts = rc.run_crescendo_conversation(
        mock.Mock(), mock.Mock(), "phi-4-mini", "target behavior", "tok",
        max_turns=1, max_backtracks=2,
    )
    assert conv == []  # attacker generation exhausted -- conversation ends early
    assert len(backtrack_attempts) == 1
    assert backtrack_attempts[0]["refused_prompt"] == "refused turn"
    assert backtrack_attempts[0]["retry_prompt"] is None


# ---------------------------------------------------------------------
# _build_row / _summarize
# ---------------------------------------------------------------------

_BEHAVIOR = {"behavior": "pick a lock", "source": "jbb_behaviors"}


def test_build_row_records_judge_failure_without_crashing():
    row = rc._build_row("phi-4-mini", 5, _BEHAVIOR, [{"role": "user", "content": "u"}],
                         backtrack_count=1, refusal_count=1, n_turns=1, verdict=None)
    assert row["judge_failed"] == 1
    assert row["attack_success"] == 0
    assert row["judge_reasoning"] is None
    assert row["backtrack_attempts"] == []  # not passed -- defaults to empty, not missing/None


def test_build_row_records_a_real_verdict():
    verdict = {"success": 1, "reasoning": "VERDICT: YES\n...", "raw": "VERDICT: YES\n..."}
    row = rc._build_row("phi-4-mini", 5, _BEHAVIOR, [{"role": "user", "content": "u"}],
                         backtrack_count=0, refusal_count=0, n_turns=1, verdict=verdict)
    assert row["judge_failed"] == 0
    assert row["attack_success"] == 1


def test_build_row_records_backtrack_attempts():
    attempts = [{"turn": 2, "refused_prompt": "p", "refused_reply": "I cannot help.", "retry_prompt": "p2"}]
    row = rc._build_row("phi-4-mini", 5, _BEHAVIOR, [{"role": "user", "content": "u"}],
                         backtrack_count=1, refusal_count=1, n_turns=1, verdict=None,
                         backtrack_attempts=attempts)
    assert row["backtrack_attempts"] == attempts


def test_summarize_excludes_judge_failed_rows_from_asr_denominator():
    rows = [
        {"attack_success": 1, "judge_failed": 0, "any_backtrack": 0, "backtrack_count": 0,
         "refusal_count": 0, "n_turns_completed": 5},
        {"attack_success": 0, "judge_failed": 0, "any_backtrack": 1, "backtrack_count": 2,
         "refusal_count": 2, "n_turns_completed": 5},
        # judge failed -- attack_success is a placeholder 0, must NOT count toward ASR
        {"attack_success": 0, "judge_failed": 1, "any_backtrack": 0, "backtrack_count": 0,
         "refusal_count": 0, "n_turns_completed": 0},
    ]
    summary = rc._summarize("phi-4-mini", 5, rows)
    assert summary["n"] == 3
    assert summary["n_scored"] == 2
    assert summary["n_judge_failed"] == 1
    assert summary["attack_success_rate"] == round(1 / 2, 4)  # denominator is 2, not 3
    assert summary["backtrack_rate"] == round(1 / 3, 4)  # backtrack accounting includes all rows
    assert "attack_success_rate" in rc._SUMMARY_FIELDNAMES


# ---------------------------------------------------------------------
# run_crescendo_sweep -- validation + wiring
# ---------------------------------------------------------------------

@pytest.fixture
def stub_full_sweep(monkeypatch, tmp_path):
    monkeypatch.setattr(rc, "RESULTS_DIR", tmp_path)
    monkeypatch.setenv("NVIDIA_NIM_API_KEY", "fake-token")
    monkeypatch.setattr(rc, "load_behavior_pool", lambda: [
        {"behavior": "b0", "source": "jbb_behaviors"},
        {"behavior": "b1", "source": "harmbench"},
    ])
    monkeypatch.setattr(rc, "sample_behaviors", lambda pool, sample_size, seed: pool)
    monkeypatch.setattr(
        rc, "run_crescendo_conversation",
        lambda model, tok, model_key, behavior, api_token, max_turns, **kw: (
            [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}],
            0, 0, 1, [],
        ),
    )
    monkeypatch.setattr(
        rc, "generate_judge_verdict",
        lambda conversation, behavior, api_token: {"success": 0, "reasoning": "VERDICT: NO", "raw": "VERDICT: NO"},
    )

    calls = []

    def fake_load_model(model_key):
        calls.append(model_key)
        return mock.Mock(), mock.Mock()

    monkeypatch.setattr(rc, "load_model", fake_load_model)
    return calls


def test_unknown_model_key_raises(stub_full_sweep):
    with pytest.raises(ValueError, match="Unknown model keys"):
        rc.run_crescendo_sweep(model_keys=["not_a_real_model"])


def test_non_hf_engine_raises(stub_full_sweep, monkeypatch):
    monkeypatch.setenv("INFERENCE_ENGINE", "vllm")
    with pytest.raises(ValueError, match="must be 'hf'"):
        rc.run_crescendo_sweep(model_keys=["phi-4-mini"])


def test_missing_api_token_raises(stub_full_sweep, monkeypatch):
    monkeypatch.delenv("NVIDIA_NIM_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="NVIDIA_NIM_API_KEY"):
        rc.run_crescendo_sweep(model_keys=["phi-4-mini"], api_token=None)


def test_sweep_loads_each_model_once_and_writes_a_summary_per_model(stub_full_sweep, tmp_path):
    calls = stub_full_sweep
    summary_rows = rc.run_crescendo_sweep(model_keys=["phi-4-mini", "qwen3-8b"], max_turns=5)

    assert calls == ["phi-4-mini", "qwen3-8b"]
    assert len(summary_rows) == 2  # one summary row per model, no corpus axis
    raw_files = list(tmp_path.glob("crescendo_raw_*.jsonl"))
    summary_files = list(tmp_path.glob("crescendo_summary_*.csv"))
    assert len(raw_files) == 2
    assert len(summary_files) == 2
