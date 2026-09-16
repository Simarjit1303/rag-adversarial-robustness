from unittest import mock
import pytest
import requests
from scripts import run_and_terminate as rt

def _write(path, content='x'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')
RAW_NAME = 'baseline_raw_phi-4-mini_nq_open_hf.jsonl'
SUMMARY_NAME = 'baseline_summary_phi-4-mini_nq_open_hf.csv'

def _set_single_cell_env(monkeypatch):
    monkeypatch.setenv('RAG_MODELS', 'phi-4-mini')
    monkeypatch.setenv('RAG_CORPORA', 'nq_open')
    monkeypatch.setenv('INFERENCE_ENGINE', 'hf')

def test_verify_success_true_when_both_files_present_and_nonempty(monkeypatch, tmp_path):
    _set_single_cell_env(monkeypatch)
    results = tmp_path / 'results'
    _write(results / RAW_NAME, '{"model": "phi-4-mini"}\n')
    _write(results / SUMMARY_NAME, 'model,corpus,n\n')
    assert rt.verify_success(str(tmp_path), 'baseline') is True

def test_verify_success_false_when_results_dir_missing(tmp_path):
    assert rt.verify_success(str(tmp_path), 'baseline') is False

def test_verify_success_false_when_a_file_is_missing(monkeypatch, tmp_path):
    _set_single_cell_env(monkeypatch)
    results = tmp_path / 'results'
    _write(results / RAW_NAME, '{"model": "phi-4-mini"}\n')
    assert rt.verify_success(str(tmp_path), 'baseline') is False

def test_verify_success_false_when_a_file_is_empty(monkeypatch, tmp_path):
    _set_single_cell_env(monkeypatch)
    results = tmp_path / 'results'
    _write(results / RAW_NAME, '{"model": "phi-4-mini"}\n')
    _write(results / SUMMARY_NAME, '')
    assert rt.verify_success(str(tmp_path), 'baseline') is False

def test_verify_success_false_when_only_one_of_two_requested_cells_is_present(monkeypatch, tmp_path):
    monkeypatch.setenv('RAG_MODELS', 'phi-4-mini')
    monkeypatch.setenv('RAG_CORPORA', 'nq_open,ms_marco')
    monkeypatch.setenv('INFERENCE_ENGINE', 'hf')
    results = tmp_path / 'results'
    _write(results / 'baseline_raw_phi-4-mini_nq_open_hf.jsonl', '{"model": "phi-4-mini"}\n')
    _write(results / 'baseline_summary_phi-4-mini_nq_open_hf.csv', 'model,corpus,n\n')
    assert rt.verify_success(str(tmp_path), 'baseline') is False

def test_verify_success_false_when_one_of_three_requested_cells_fails(monkeypatch, tmp_path):
    monkeypatch.setenv('RAG_MODELS', 'phi-4-mini')
    monkeypatch.setenv('RAG_CORPORA', 'nq_open,hotpot_qa,ms_marco')
    monkeypatch.setenv('INFERENCE_ENGINE', 'hf')
    results = tmp_path / 'results'
    _write(results / 'baseline_raw_phi-4-mini_nq_open_hf.jsonl', '{"model": "phi-4-mini"}\n')
    _write(results / 'baseline_summary_phi-4-mini_nq_open_hf.csv', 'model,corpus,n\n')
    _write(results / 'baseline_raw_phi-4-mini_hotpot_qa_hf.jsonl', '{"model": "phi-4-mini"}\n')
    _write(results / 'baseline_summary_phi-4-mini_hotpot_qa_hf.csv', 'model,corpus,n\n')
    _write(results / 'baseline_raw_phi-4-mini_ms_marco_hf.jsonl', '{"model": "phi-4-mini"}\n')
    assert rt.verify_success(str(tmp_path), 'baseline') is False

@pytest.fixture
def mock_delete(monkeypatch):
    fake = mock.MagicMock(name='requests.delete')
    monkeypatch.setattr('requests.delete', fake)
    return fake

@pytest.fixture
def mock_idle(monkeypatch):
    calls = []
    monkeypatch.setattr(rt, '_idle_forever', lambda: calls.append(True))
    return calls

def test_no_terminate_when_pipeline_fails(monkeypatch, tmp_path, mock_delete, mock_idle):
    monkeypatch.setenv('RAG_SCRATCH_DIR', str(tmp_path))
    monkeypatch.setenv('RUNPOD_TERMINATE_KEY', 'should-never-be-used')
    monkeypatch.setenv('RUNPOD_POD_ID', 'should-never-be-used')
    monkeypatch.setattr(rt, 'run_pipeline', lambda target: 3)
    rt.main()
    assert mock_idle == [True]
    mock_delete.assert_not_called()

def test_no_terminate_when_exit_zero_but_results_missing(monkeypatch, tmp_path, mock_delete, mock_idle):
    monkeypatch.setenv('RAG_SCRATCH_DIR', str(tmp_path))
    monkeypatch.setenv('RUNPOD_TERMINATE_KEY', 'should-never-be-used')
    monkeypatch.setenv('RUNPOD_POD_ID', 'should-never-be-used')
    monkeypatch.setattr(rt, 'run_pipeline', lambda target: 0)
    rt.main()
    assert mock_idle == [True]
    mock_delete.assert_not_called()

def test_no_terminate_when_a_result_file_is_empty(monkeypatch, tmp_path, mock_delete, mock_idle):
    _set_single_cell_env(monkeypatch)
    results = tmp_path / 'results'
    _write(results / RAW_NAME, '{"model": "phi-4-mini"}\n')
    _write(results / SUMMARY_NAME, '')
    monkeypatch.setenv('RAG_SCRATCH_DIR', str(tmp_path))
    monkeypatch.setenv('RUNPOD_TERMINATE_KEY', 'should-never-be-used')
    monkeypatch.setenv('RUNPOD_POD_ID', 'should-never-be-used')
    monkeypatch.setattr(rt, 'run_pipeline', lambda target: 0)
    rt.main()
    assert mock_idle == [True]
    mock_delete.assert_not_called()

def test_refuses_to_run_without_scratch_dir(monkeypatch, mock_delete, mock_idle):
    monkeypatch.delenv('RAG_SCRATCH_DIR', raising=False)
    monkeypatch.delenv('RUNPOD_TERMINATE_KEY', raising=False)
    monkeypatch.delenv('RUNPOD_POD_ID', raising=False)
    monkeypatch.setattr(rt, 'run_pipeline', lambda: pytest.fail('pipeline should not start'))
    rt.main()
    assert mock_idle == [True]
    mock_delete.assert_not_called()

def test_terminates_only_on_verified_success(monkeypatch, tmp_path, mock_delete):
    _set_single_cell_env(monkeypatch)
    results = tmp_path / 'results'
    _write(results / RAW_NAME, '{"model": "phi-4-mini"}\n')
    _write(results / SUMMARY_NAME, 'model,corpus,n\n')
    monkeypatch.setenv('RAG_SCRATCH_DIR', str(tmp_path))
    monkeypatch.setenv('RUNPOD_TERMINATE_KEY', 'test-key')
    monkeypatch.setenv('RUNPOD_POD_ID', 'pod-abc123')
    monkeypatch.setattr(rt, 'run_pipeline', lambda target: 0)
    rt.main()
    mock_delete.assert_called_once()
    args, kwargs = mock_delete.call_args
    url = args[0] if args else kwargs['url']
    assert url == 'https://rest.runpod.io/v1/pods/pod-abc123'
    assert kwargs['headers'] == {'Authorization': 'Bearer test-key'}
    assert kwargs['timeout'] == 30
    mock_delete.return_value.raise_for_status.assert_called_once()

def test_terminate_pod_raises_on_non_2xx(mock_delete):
    mock_delete.return_value.raise_for_status.side_effect = RuntimeError('401')
    with pytest.raises(RuntimeError):
        rt.terminate_pod('pod-abc123', 'bad-key')

class _FakeProc:

    def __init__(self, returncode=0, hang=False, pid=4242):
        self._returncode = returncode
        self._hang = hang
        self.pid = pid

    def wait(self, timeout=None):
        if self._hang and timeout is not None:
            self._hang = False
            raise rt.subprocess.TimeoutExpired(cmd='fake', timeout=timeout)
        return self._returncode

def test_run_pipeline_stops_at_first_failing_stage(monkeypatch):
    calls = []

    def fake_popen(cmd, *a, **k):
        calls.append(cmd)
        rc = 2 if 'data.build_index' in cmd else 0
        return _FakeProc(returncode=rc)
    monkeypatch.setattr(rt.subprocess, 'Popen', fake_popen)
    assert rt.run_pipeline('baseline') == 2
    assert len(calls) == 2
    assert not any(('evaluation.run_baseline' in c for c in calls))

def test_run_pipeline_returns_none_and_kills_process_group_on_timeout(monkeypatch):

    def fake_popen(cmd, *a, **k):
        assert k.get('start_new_session') is True
        return _FakeProc(hang=True, pid=777)
    killpg_calls = []
    monkeypatch.setattr(rt.subprocess, 'Popen', fake_popen)
    monkeypatch.setattr(rt.os, 'getpgid', lambda pid: 999 if pid == 777 else pid, raising=False)
    monkeypatch.setattr(rt.os, 'killpg', lambda pgid, sig: killpg_calls.append((pgid, sig)), raising=False)
    monkeypatch.setattr(rt, 'PIPELINE_TIMEOUT_SECONDS', 1)
    assert rt.run_pipeline('baseline') is None
    assert killpg_calls == [(999, getattr(rt.signal, 'SIGKILL', rt.signal.SIGTERM))]

def test_run_pipeline_unaffected_by_timeout_on_normal_fast_completion(monkeypatch):
    monkeypatch.setattr(rt.subprocess, 'Popen', lambda cmd, *a, **k: _FakeProc(returncode=0))
    assert rt.run_pipeline('baseline') == 0

def test_main_idles_and_does_not_terminate_on_timeout(monkeypatch, tmp_path, mock_delete, mock_idle):
    monkeypatch.setenv('RAG_SCRATCH_DIR', str(tmp_path))
    monkeypatch.setenv('RUNPOD_TERMINATE_KEY', 'should-never-be-used')
    monkeypatch.setenv('RUNPOD_POD_ID', 'should-never-be-used')
    monkeypatch.setattr(rt, 'run_pipeline', lambda target: None)
    rt.main()
    assert mock_idle == [True]
    mock_delete.assert_not_called()

@pytest.mark.skipif(rt.os.name != 'posix', reason='os.killpg/os.getpgid are POSIX-only; production runs in Linux Docker')
def test_run_pipeline_real_hang_is_actually_killed_within_timeout_window(tmp_path):
    import time
    script = tmp_path / 'hang.py'
    script.write_text('import time\ntime.sleep(999)\n', encoding='utf-8')
    original_timeout = rt.PIPELINE_TIMEOUT_SECONDS
    original_pipeline_stages = rt._pipeline_stages
    try:
        rt._pipeline_stages = lambda target: ([rt.sys.executable, str(script)],)
        rt.PIPELINE_TIMEOUT_SECONDS = 2
        start = time.time()
        result = rt.run_pipeline('baseline')
        elapsed = time.time() - start
        assert result is None
        assert elapsed < 10
    finally:
        rt._pipeline_stages = original_pipeline_stages
        rt.PIPELINE_TIMEOUT_SECONDS = original_timeout

def _happy_path_env(monkeypatch, tmp_path):
    _set_single_cell_env(monkeypatch)
    results = tmp_path / 'results'
    _write(results / RAW_NAME, '{"model": "phi-4-mini"}\n')
    _write(results / SUMMARY_NAME, 'model,corpus,n\n')
    monkeypatch.setenv('RAG_SCRATCH_DIR', str(tmp_path))
    monkeypatch.setenv('RUNPOD_TERMINATE_KEY', 'test-key')
    monkeypatch.setenv('RUNPOD_POD_ID', 'pod-abc123')
    monkeypatch.setattr(rt, 'run_pipeline', lambda target: 0)

def test_terminate_failure_does_not_crash_main_and_idles_instead(monkeypatch, tmp_path, mock_delete, mock_idle, capsys):
    _happy_path_env(monkeypatch, tmp_path)
    mock_delete.return_value.raise_for_status.side_effect = requests.HTTPError('403 Forbidden')
    rt.main()
    assert mock_idle == [True]
    err = capsys.readouterr().err
    assert 'PIPELINE SUCCEEDED' in err
    assert 'Termination call FAILED' in err
    assert 'may still be billing' in err

def test_terminate_failure_of_any_exception_type_is_caught(monkeypatch, tmp_path, mock_delete, mock_idle):
    _happy_path_env(monkeypatch, tmp_path)
    mock_delete.side_effect = requests.ConnectionError('credential issue')
    rt.main()
    assert mock_idle == [True]

def test_terminate_success_path_unchanged_no_idle_loop_triggered(monkeypatch, tmp_path, mock_delete, mock_idle, capsys):
    _happy_path_env(monkeypatch, tmp_path)
    rt.main()
    mock_delete.assert_called_once()
    assert mock_idle == []
    assert 'pod terminated successfully.' in capsys.readouterr().out

def test_fail_and_idle_logs_message_and_delegates_to_idle_forever(monkeypatch, mock_idle, capsys):
    rt._fail_and_idle('something specific went wrong')
    assert mock_idle == [True]
    err = capsys.readouterr().err
    assert 'FAILURE: something specific went wrong' in err
    assert 'idling instead of exiting' in err

def test_idle_forever_actually_calls_time_sleep_in_a_loop(monkeypatch):
    sleep_calls = []

    def fake_sleep(seconds):
        sleep_calls.append(seconds)
        if len(sleep_calls) >= 3:
            raise KeyboardInterrupt("stop the test's idle loop")
    monkeypatch.setattr(rt.time, 'sleep', fake_sleep)
    with pytest.raises(KeyboardInterrupt):
        rt._idle_forever()
    assert sleep_calls == [3600, 3600, 3600]

def test_timeout_never_reaches_terminate_pod_call_site(monkeypatch, tmp_path, mock_delete, mock_idle):
    monkeypatch.setenv('RAG_SCRATCH_DIR', str(tmp_path))
    monkeypatch.setenv('RUNPOD_TERMINATE_KEY', 'test-key')
    monkeypatch.setenv('RUNPOD_POD_ID', 'pod-abc123')
    monkeypatch.setattr(rt, 'run_pipeline', lambda target: None)
    rt.main()
    mock_delete.assert_not_called()
    assert mock_idle == [True]

def test_backstop_catches_an_unrelated_exception_type_not_among_the_6_known_sites(monkeypatch, tmp_path, mock_delete, mock_idle):
    monkeypatch.setenv('RAG_SCRATCH_DIR', str(tmp_path))
    monkeypatch.setenv('RUNPOD_TERMINATE_KEY', 'test-key')
    monkeypatch.setenv('RUNPOD_POD_ID', 'pod-abc123')

    def raise_unexpected(target):
        raise RuntimeError('cudaErrorDevicesUnavailable: unrelated to any of the 6 known sites')
    monkeypatch.setattr(rt, 'run_pipeline', raise_unexpected)
    rt._main_with_backstop()
    assert mock_idle == [True]
    mock_delete.assert_not_called()

def test_backstop_logs_message_includes_the_exception_repr(monkeypatch, tmp_path, mock_idle, capsys):
    monkeypatch.setenv('RAG_SCRATCH_DIR', str(tmp_path))

    def raise_unexpected(target):
        raise KeyError('SOME_UNCHECKED_ENV_VAR')
    monkeypatch.setattr(rt, 'run_pipeline', raise_unexpected)
    rt._main_with_backstop()
    err = capsys.readouterr().err
    assert 'unhandled exception' in err
    assert 'SOME_UNCHECKED_ENV_VAR' in err

def test_backstop_lets_keyboard_interrupt_and_system_exit_propagate(monkeypatch, mock_idle):
    monkeypatch.setattr(rt, 'main', mock.Mock(side_effect=KeyboardInterrupt('interactive Ctrl-C')))
    with pytest.raises(KeyboardInterrupt):
        rt._main_with_backstop()
    assert mock_idle == []
    monkeypatch.setattr(rt, 'main', mock.Mock(side_effect=SystemExit(0)))
    with pytest.raises(SystemExit):
        rt._main_with_backstop()
    assert mock_idle == []

def test_backstop_does_not_interfere_with_the_6_known_failure_sites(monkeypatch, tmp_path, mock_delete, mock_idle):
    monkeypatch.delenv('RAG_SCRATCH_DIR', raising=False)
    rt._main_with_backstop()
    assert mock_idle == [True]
    mock_delete.assert_not_called()

def test_backstop_does_not_interfere_with_genuine_success_path(monkeypatch, tmp_path, mock_delete, mock_idle, capsys):
    _happy_path_env(monkeypatch, tmp_path)
    rt._main_with_backstop()
    mock_delete.assert_called_once()
    assert mock_idle == []
    assert 'pod terminated successfully.' in capsys.readouterr().out

def test_resolve_run_target_defaults_to_baseline(monkeypatch):
    monkeypatch.delenv('RAG_RUN_TARGET', raising=False)
    assert rt._resolve_run_target() == 'baseline'

def test_resolve_run_target_honors_env_var(monkeypatch):
    monkeypatch.setenv('RAG_RUN_TARGET', 'attack_injection')
    assert rt._resolve_run_target() == 'attack_injection'

def test_resolve_run_target_unknown_value_raises(monkeypatch):
    monkeypatch.setenv('RAG_RUN_TARGET', 'not_a_real_target')
    with pytest.raises(ValueError, match='Unknown RAG_RUN_TARGET'):
        rt._resolve_run_target()

def test_pipeline_stages_baseline_runs_run_baseline_module():
    stages = rt._pipeline_stages('baseline')
    assert stages[-1] == [rt.sys.executable, '-m', 'evaluation.run_baseline']

def test_pipeline_stages_attack_injection_runs_run_attack_injection_module():
    stages = rt._pipeline_stages('attack_injection')
    assert stages[-1] == [rt.sys.executable, '-m', 'evaluation.run_attack_injection']
    assert stages[0] == [rt.sys.executable, '-m', 'data.loader']
    assert stages[1] == [rt.sys.executable, '-m', 'data.build_index']

def test_verify_success_attack_injection_target_checks_attack_files(monkeypatch, tmp_path):
    monkeypatch.setenv('RAG_MODELS', 'phi-4-mini')
    monkeypatch.setenv('RAG_CORPORA', 'hotpot_qa')
    monkeypatch.setenv('RAG_INJECTION_TEMPLATES', 'naive')
    monkeypatch.setenv('INFERENCE_ENGINE', 'hf')
    results = tmp_path / 'results'
    _write(results / 'baseline_raw_phi-4-mini_hotpot_qa_hf.jsonl')
    _write(results / 'baseline_summary_phi-4-mini_hotpot_qa_hf.csv')
    assert rt.verify_success(str(tmp_path), 'attack_injection') is False
    _write(results / 'attack_raw_phi-4-mini_hotpot_qa_naive_hf.jsonl')
    _write(results / 'attack_summary_phi-4-mini_hotpot_qa_naive_hf.csv')
    assert rt.verify_success(str(tmp_path), 'attack_injection') is True

def test_main_fails_and_idles_on_unknown_run_target_before_pipeline_starts(monkeypatch, tmp_path, mock_delete, mock_idle, capsys):
    monkeypatch.setenv('RAG_SCRATCH_DIR', str(tmp_path))
    monkeypatch.setenv('RAG_RUN_TARGET', 'not_a_real_target')
    monkeypatch.setattr(rt, 'run_pipeline', lambda target: pytest.fail('pipeline should not start'))
    rt.main()
    assert mock_idle == [True]
    mock_delete.assert_not_called()
    assert 'Unknown RAG_RUN_TARGET' in capsys.readouterr().err

def test_main_end_to_end_with_attack_injection_target(monkeypatch, tmp_path, mock_delete, mock_idle):
    monkeypatch.setenv('RAG_MODELS', 'phi-4-mini')
    monkeypatch.setenv('RAG_CORPORA', 'hotpot_qa')
    monkeypatch.setenv('RAG_INJECTION_TEMPLATES', 'naive')
    monkeypatch.setenv('INFERENCE_ENGINE', 'hf')
    monkeypatch.setenv('RAG_RUN_TARGET', 'attack_injection')
    results = tmp_path / 'results'
    _write(results / 'attack_raw_phi-4-mini_hotpot_qa_naive_hf.jsonl')
    _write(results / 'attack_summary_phi-4-mini_hotpot_qa_naive_hf.csv')
    monkeypatch.setenv('RAG_SCRATCH_DIR', str(tmp_path))
    monkeypatch.setenv('RUNPOD_TERMINATE_KEY', 'test-key')
    monkeypatch.setenv('RUNPOD_POD_ID', 'pod-abc123')
    pipeline_targets_seen = []
    monkeypatch.setattr(rt, 'run_pipeline', lambda target: pipeline_targets_seen.append(target) or 0)
    rt.main()
    assert pipeline_targets_seen == ['attack_injection']
    mock_delete.assert_called_once()
    assert mock_idle == []
