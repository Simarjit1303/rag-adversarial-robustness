from unittest import mock
import pytest
import attacks.crescendo as crescendo
import evaluation.run_crescendo as rc

def test_call_with_retry_succeeds_on_first_attempt():
    result = rc._call_with_retry(lambda: 'ok', max_attempts=3, label='t')
    assert result == 'ok'

def test_call_with_retry_succeeds_after_transient_failures(monkeypatch):
    monkeypatch.setattr(rc.time, 'sleep', lambda s: None)
    calls = {'n': 0}

    def flaky():
        calls['n'] += 1
        if calls['n'] < 3:
            raise ConnectionError('transient')
        return 'ok'
    result = rc._call_with_retry(flaky, max_attempts=3, label='t')
    assert result == 'ok'
    assert calls['n'] == 3

def test_call_with_retry_gives_up_and_returns_none_after_exhausting_attempts(monkeypatch):
    monkeypatch.setattr(rc.time, 'sleep', lambda s: None)

    def always_fails():
        raise ConnectionError('persistent')
    result = rc._call_with_retry(always_fails, max_attempts=3, label='t')
    assert result is None

def test_call_with_retry_honors_retry_after_header_exactly(monkeypatch):
    sleeps = []
    monkeypatch.setattr(rc.time, 'sleep', sleeps.append)
    calls = {'n': 0}

    def rate_limited_then_ok():
        calls['n'] += 1
        if calls['n'] == 1:
            raise rc.RateLimitError('429', retry_after=13.0)
        return 'ok'
    result = rc._call_with_retry(rate_limited_then_ok, max_attempts=3, label='t')
    assert result == 'ok'
    assert sleeps == [13.0]

def test_call_with_retry_falls_back_to_exponential_when_no_retry_after(monkeypatch):
    sleeps = []
    monkeypatch.setattr(rc.time, 'sleep', sleeps.append)

    def always_rate_limited():
        raise rc.RateLimitError('429', retry_after=None)
    rc._call_with_retry(always_rate_limited, max_attempts=3, label='t')
    assert sleeps == [2, 4]

def test_call_with_retry_writes_exhausted_error_to_error_sink(monkeypatch):
    monkeypatch.setattr(rc.time, 'sleep', lambda s: None)

    def always_fails():
        raise rc.RateLimitError("NIM rate limit hit (429) -- Retry-After='45'", retry_after=45.0)
    sink = {}
    result = rc._call_with_retry(always_fails, max_attempts=3, label='phi-4-mini attacker turn 1 attempt 1', error_sink=sink)
    assert result is None
    assert sink['error'] == "phi-4-mini attacker turn 1 attempt 1 (after 3 attempts): RateLimitError: NIM rate limit hit (429) -- Retry-After='45'"

def test_call_with_retry_does_not_touch_error_sink_on_success(monkeypatch):
    sink = {}
    result = rc._call_with_retry(lambda: 'ok', max_attempts=3, label='t', error_sink=sink)
    assert result == 'ok'
    assert sink == {}

@pytest.fixture
def stub_conversation(monkeypatch):
    monkeypatch.setattr(rc, '_generate_target_reply', lambda *a, **kw: 'a target reply')

def test_conversation_completes_max_turns_with_no_refusals(monkeypatch, stub_conversation):
    monkeypatch.setattr(rc, 'generate_attacker_turn', lambda history, behavior, api_token, refusal_feedback=None, **kw: 'next turn')
    conv, backtracks, refusals, n_turns, backtrack_attempts, error, _of_flags = rc.run_crescendo_conversation(mock.Mock(), mock.Mock(), 'phi-4-mini', 'target behavior', 'tok', max_turns=5)
    assert n_turns == 5
    assert len(conv) == 10
    assert backtracks == 0
    assert refusals == 0
    assert backtrack_attempts == []
    assert error is None

def test_conversation_backtracks_on_refusal_then_recovers(monkeypatch):
    replies = iter(['I cannot help with that.', 'a compliant reply'])
    monkeypatch.setattr(rc, '_generate_target_reply', lambda *a, **kw: next(replies))
    prompts = iter(['refused turn', 'retried turn'])
    monkeypatch.setattr(rc, 'generate_attacker_turn', lambda history, behavior, api_token, refusal_feedback=None, **kw: next(prompts))
    conv, backtracks, refusals, n_turns, backtrack_attempts, error, _of_flags = rc.run_crescendo_conversation(mock.Mock(), mock.Mock(), 'phi-4-mini', 'target behavior', 'tok', max_turns=1)
    assert n_turns == 1
    assert backtracks == 1
    assert refusals == 1
    assert conv[-1]['content'] == 'a compliant reply'
    assert backtrack_attempts == [{'turn': 1, 'refused_prompt': 'refused turn', 'refused_reply': 'I cannot help with that.', 'retry_prompt': 'retried turn'}]
    assert error is None

def test_conversation_accepts_refusal_after_exhausting_backtrack_cap(monkeypatch):
    monkeypatch.setattr(rc, '_generate_target_reply', lambda *a, **kw: 'I cannot help with that.')
    monkeypatch.setattr(rc, 'generate_attacker_turn', lambda history, behavior, api_token, refusal_feedback=None, **kw: 'next turn')
    conv, backtracks, refusals, n_turns, backtrack_attempts, error, _of_flags = rc.run_crescendo_conversation(mock.Mock(), mock.Mock(), 'phi-4-mini', 'target behavior', 'tok', max_turns=1, max_backtracks=2)
    assert n_turns == 1
    assert backtracks == 2
    assert refusals == 3
    assert conv[-1]['content'] == 'I cannot help with that.'
    assert len(backtrack_attempts) == 2
    assert all((a['refused_reply'] == 'I cannot help with that.' for a in backtrack_attempts))
    assert all((a['retry_prompt'] == 'next turn' for a in backtrack_attempts))
    assert error is None

def test_conversation_ends_early_when_attacker_generation_exhausts_retries(monkeypatch, stub_conversation):
    monkeypatch.setattr(rc, 'generate_attacker_turn', lambda history, behavior, api_token, refusal_feedback=None, **kw: None)
    conv, backtracks, refusals, n_turns, backtrack_attempts, error, _of_flags = rc.run_crescendo_conversation(mock.Mock(), mock.Mock(), 'phi-4-mini', 'target behavior', 'tok', max_turns=5)
    assert n_turns == 0
    assert conv == []
    assert backtrack_attempts == []
    assert error is None

def test_conversation_backtrack_attempt_retry_prompt_is_none_when_retry_generation_fails(monkeypatch):
    monkeypatch.setattr(rc, '_generate_target_reply', lambda *a, **kw: 'I cannot help with that.')
    prompts = iter(['refused turn', None])
    monkeypatch.setattr(rc, 'generate_attacker_turn', lambda history, behavior, api_token, refusal_feedback=None, **kw: next(prompts))
    conv, backtracks, refusals, n_turns, backtrack_attempts, error, _of_flags = rc.run_crescendo_conversation(mock.Mock(), mock.Mock(), 'phi-4-mini', 'target behavior', 'tok', max_turns=1, max_backtracks=2)
    assert conv == []
    assert len(backtrack_attempts) == 1
    assert backtrack_attempts[0]['refused_prompt'] == 'refused turn'
    assert backtrack_attempts[0]['retry_prompt'] is None
    assert error is None

def test_conversation_surfaces_real_exhausted_attacker_error(monkeypatch, stub_conversation):
    monkeypatch.setattr(rc.time, 'sleep', lambda s: None)

    def always_raises(history, behavior, api_token, refusal_feedback=None, **kw):
        raise rc.RateLimitError("NIM rate limit hit (429) -- Retry-After='60'", retry_after=60.0)
    monkeypatch.setattr(rc, 'generate_attacker_turn', always_raises)
    conv, backtracks, refusals, n_turns, backtrack_attempts, error, _of_flags = rc.run_crescendo_conversation(mock.Mock(), mock.Mock(), 'phi-4-mini', 'target behavior', 'tok', max_turns=5)
    assert conv == []
    assert n_turns == 0
    assert error == "phi-4-mini attacker turn 1 attempt 1 (after 3 attempts): RateLimitError: NIM rate limit hit (429) -- Retry-After='60'"
_BEHAVIOR = {'behavior': 'pick a lock', 'source': 'jbb_behaviors'}

def test_build_row_records_judge_failure_without_crashing():
    row = rc._build_row('phi-4-mini', 5, _BEHAVIOR, [{'role': 'user', 'content': 'u'}], backtrack_count=1, refusal_count=1, n_turns=1, verdict=None)
    assert row['judge_failed'] == 1
    assert row['attack_success'] == 0
    assert row['judge_reasoning'] is None
    assert row['backtrack_attempts'] == []
    assert row['error'] is None

def test_build_row_records_a_real_verdict():
    verdict = {'success': 1, 'reasoning': 'VERDICT: YES\n...', 'raw': 'VERDICT: YES\n...'}
    row = rc._build_row('phi-4-mini', 5, _BEHAVIOR, [{'role': 'user', 'content': 'u'}], backtrack_count=0, refusal_count=0, n_turns=1, verdict=verdict)
    assert row['judge_failed'] == 0
    assert row['attack_success'] == 1

def test_build_row_records_backtrack_attempts():
    attempts = [{'turn': 2, 'refused_prompt': 'p', 'refused_reply': 'I cannot help.', 'retry_prompt': 'p2'}]
    row = rc._build_row('phi-4-mini', 5, _BEHAVIOR, [{'role': 'user', 'content': 'u'}], backtrack_count=1, refusal_count=1, n_turns=1, verdict=None, backtrack_attempts=attempts)
    assert row['backtrack_attempts'] == attempts

def test_build_row_records_error():
    row = rc._build_row('phi-4-mini', 5, _BEHAVIOR, [], backtrack_count=0, refusal_count=0, n_turns=0, verdict=None, error='phi-4-mini attacker turn 1 attempt 3: RateLimitError: 429')
    assert row['error'] == 'phi-4-mini attacker turn 1 attempt 3: RateLimitError: 429'

def test_summarize_excludes_judge_failed_rows_from_asr_denominator():
    rows = [{'attack_success': 1, 'judge_failed': 0, 'any_backtrack': 0, 'backtrack_count': 0, 'refusal_count': 0, 'n_turns_completed': 5}, {'attack_success': 0, 'judge_failed': 0, 'any_backtrack': 1, 'backtrack_count': 2, 'refusal_count': 2, 'n_turns_completed': 5}, {'attack_success': 0, 'judge_failed': 1, 'any_backtrack': 0, 'backtrack_count': 0, 'refusal_count': 0, 'n_turns_completed': 0}]
    summary = rc._summarize('phi-4-mini', 5, rows)
    assert summary['n'] == 3
    assert summary['n_scored'] == 2
    assert summary['n_judge_failed'] == 1
    assert summary['attack_success_rate'] == round(1 / 2, 4)
    assert summary['backtrack_rate'] == round(1 / 3, 4)
    assert 'attack_success_rate' in rc._SUMMARY_FIELDNAMES

@pytest.fixture
def stub_full_sweep(monkeypatch, tmp_path):
    monkeypatch.setattr(rc, 'RESULTS_DIR', tmp_path)
    monkeypatch.setenv(crescendo._PROVIDER['api_key_env'], 'fake-token')
    monkeypatch.setattr(rc, 'load_behavior_pool', lambda: [{'behavior': 'b0', 'source': 'jbb_behaviors'}, {'behavior': 'b1', 'source': 'harmbench'}])
    monkeypatch.setattr(rc, 'sample_behaviors', lambda pool, sample_size, seed: pool)
    monkeypatch.setattr(rc, 'run_crescendo_conversation', lambda model, tok, model_key, behavior, api_token, max_turns, **kw: ([{'role': 'user', 'content': 'u'}, {'role': 'assistant', 'content': 'a'}], 0, 0, 1, [], None, []))
    monkeypatch.setattr(rc, 'generate_judge_verdict', lambda conversation, behavior, api_token, **kw: {'success': 0, 'reasoning': 'VERDICT: NO', 'raw': 'VERDICT: NO'})
    calls = []

    def fake_load_model(model_key):
        calls.append(model_key)
        return (mock.Mock(), mock.Mock())
    monkeypatch.setattr(rc, 'load_model', fake_load_model)
    return calls

def test_unknown_model_key_raises(stub_full_sweep):
    with pytest.raises(ValueError, match='Unknown model keys'):
        rc.run_crescendo_sweep(model_keys=['not_a_real_model'])

def test_non_hf_engine_raises(stub_full_sweep, monkeypatch):
    monkeypatch.setenv('INFERENCE_ENGINE', 'vllm')
    with pytest.raises(ValueError, match="must be 'hf'"):
        rc.run_crescendo_sweep(model_keys=['phi-4-mini'])

def test_missing_api_token_raises(stub_full_sweep, monkeypatch):
    monkeypatch.delenv(crescendo._PROVIDER['api_key_env'], raising=False)
    with pytest.raises(RuntimeError, match=crescendo._PROVIDER['api_key_env']):
        rc.run_crescendo_sweep(model_keys=['phi-4-mini'], api_token=None)

def test_sweep_loads_each_model_once_and_writes_a_summary_per_model(stub_full_sweep, tmp_path):
    calls = stub_full_sweep
    summary_rows = rc.run_crescendo_sweep(model_keys=['phi-4-mini', 'qwen3-8b'], max_turns=5)
    assert calls == ['phi-4-mini', 'qwen3-8b']
    assert len(summary_rows) == 2
    raw_files = list(tmp_path.glob('crescendo_raw_*.jsonl'))
    summary_files = list(tmp_path.glob('crescendo_summary_*.csv'))
    assert len(raw_files) == 2
    assert len(summary_files) == 2

def test_model_switch_sleep_diagnostic_off_by_default(stub_full_sweep, monkeypatch):
    sleeps = []
    monkeypatch.setattr(rc.time, 'sleep', sleeps.append)
    rc.run_crescendo_sweep(model_keys=['phi-4-mini', 'qwen3-8b'], max_turns=5)
    assert sleeps == []

def test_model_switch_sleep_diagnostic_sleeps_between_models_when_set(stub_full_sweep, monkeypatch):
    monkeypatch.setenv('CRESCENDO_MODEL_SWITCH_SLEEP_SECONDS', '75')
    sleeps = []
    monkeypatch.setattr(rc.time, 'sleep', sleeps.append)
    rc.run_crescendo_sweep(model_keys=['phi-4-mini', 'qwen3-8b'], max_turns=5)
    assert sleeps == [75.0]

def test_sweep_writes_conversation_error_into_the_raw_row(stub_full_sweep, monkeypatch, tmp_path):
    monkeypatch.setattr(rc, 'run_crescendo_conversation', lambda model, tok, model_key, behavior, api_token, max_turns, **kw: ([], 0, 0, 0, [], 'phi-4-mini attacker turn 1 attempt 3: RateLimitError: 429', []))
    rc.run_crescendo_sweep(model_keys=['phi-4-mini'], max_turns=5)
    raw_path = next(tmp_path.glob('crescendo_raw_*.jsonl'))
    import json
    rows = [json.loads(line) for line in raw_path.read_text(encoding='utf-8').splitlines()]
    assert all((r['error'] == 'phi-4-mini attacker turn 1 attempt 3: RateLimitError: 429' for r in rows))
    assert all((r['conversation'] == [] and r['n_turns_completed'] == 0 for r in rows))

def test_run_crescendo_sweep_rejects_instruction_detection_and_spotlighting():
    for bad_defense in ('instruction_detection', 'spotlighting'):
        with pytest.raises(ValueError, match='only supports defense'):
            rc.run_crescendo_sweep(model_keys=['phi-4-mini'], defense=bad_defense)

def test_conversation_applies_output_filter_per_turn_without_altering_history(monkeypatch, stub_conversation, tmp_path):
    import defenses.output_filter as of
    monkeypatch.setattr(rc, 'generate_attacker_turn', lambda history, behavior, api_token, refusal_feedback=None, **kw: 'next turn')
    monkeypatch.setattr(of, 'apply_output_filter', lambda text, model=None, tokenizer=None: (of.REFUSAL_MARKER, {'flagged': True, 'guard_label': 'unsafe', 'guard_raw_output': 'unsafe'}))
    log_path = tmp_path / 'output_filter_log.jsonl'
    conv, backtracks, refusals, n_turns, backtrack_attempts, error, output_filter_flags = rc.run_crescendo_conversation(mock.Mock(), mock.Mock(), 'phi-4-mini', 'target behavior', 'tok', max_turns=2, output_filter_log_path=log_path)
    assert conv[1]['content'] == 'a target reply'
    assert conv[3]['content'] == 'a target reply'
    assert output_filter_flags == [{'turn': 1, 'flagged': True, 'guard_label': 'unsafe'}, {'turn': 2, 'flagged': True, 'guard_label': 'unsafe'}]
    logged = [__import__('json').loads(l) for l in log_path.read_text(encoding='utf-8').splitlines()]
    assert len(logged) == 2
    assert logged[0]['behavior'] == 'target behavior' and logged[0]['flagged'] is True

def test_run_crescendo_sweep_output_filter_writes_distinct_filename_and_flags_field(stub_full_sweep, tmp_path, monkeypatch):
    monkeypatch.setattr(rc, 'run_crescendo_conversation', lambda model, tok, model_key, behavior, api_token, max_turns, **kw: ([{'role': 'user', 'content': 'u'}, {'role': 'assistant', 'content': 'a'}], 0, 0, 1, [], None, [{'turn': 1, 'flagged': True, 'guard_label': 'unsafe'}]))
    rc.run_crescendo_sweep(model_keys=['phi-4-mini'], max_turns=5, defense='output_filter')
    raw_path = tmp_path / 'crescendo_raw_phi-4-mini_5turn_hf_defense-output_filter.jsonl'
    assert raw_path.exists()
    import json
    rows = [json.loads(line) for line in raw_path.read_text(encoding='utf-8').splitlines()]
    assert all((r['output_filter_flags'] == [{'turn': 1, 'flagged': True, 'guard_label': 'unsafe'}] for r in rows))
