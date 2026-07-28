"""
Local, zero-cost tests for the RunPod self-termination wrapper.

Nothing here touches a real RunPod pod or the RunPod API. Termination is a
plain HTTP DELETE issued through `requests` -- the `runpod` SDK is unusable in
this image because it hard-depends on a fastapi range vllm forbids -- so every
test that can reach the termination path patches `requests.delete`. That makes
a real API call impossible even if the control flow were wrong, the point
being to PROVE the pod is never terminated on a failed or unverified run.
"""

from unittest import mock

import pytest
import requests

from scripts import run_and_terminate as rt


# --------------------------------------------------------------------------
# Task 3.1 -- verify_success() against a temp results dir
# --------------------------------------------------------------------------

def _write(path, content="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# Result filenames are namespaced by (model, corpus, engine) -- see
# evaluation/result_paths.py -- because a single fixed pair used to be
# silently overwritten by the next run that used a different one of those
# three. Tests pin RAG_MODELS/RAG_CORPORA/INFERENCE_ENGINE to exactly one
# cell so the expected file set stays small and explicit rather than the
# full model x corpus matrix.
RAW_NAME = "baseline_raw_phi-4-mini_nq_open_hf.jsonl"
SUMMARY_NAME = "baseline_summary_phi-4-mini_nq_open_hf.csv"


def _set_single_cell_env(monkeypatch):
    monkeypatch.setenv("RAG_MODELS", "phi-4-mini")
    monkeypatch.setenv("RAG_CORPORA", "nq_open")
    monkeypatch.setenv("INFERENCE_ENGINE", "hf")


def test_verify_success_true_when_both_files_present_and_nonempty(monkeypatch, tmp_path):
    _set_single_cell_env(monkeypatch)
    results = tmp_path / "results"
    _write(results / RAW_NAME, '{"model": "phi-4-mini"}\n')
    _write(results / SUMMARY_NAME, "model,corpus,n\n")
    assert rt.verify_success(str(tmp_path)) is True


def test_verify_success_false_when_results_dir_missing(tmp_path):
    # scratch dir exists but no results/ subdir at all
    assert rt.verify_success(str(tmp_path)) is False


def test_verify_success_false_when_a_file_is_missing(monkeypatch, tmp_path):
    _set_single_cell_env(monkeypatch)
    results = tmp_path / "results"
    _write(results / RAW_NAME, '{"model": "phi-4-mini"}\n')
    # summary CSV absent
    assert rt.verify_success(str(tmp_path)) is False


def test_verify_success_false_when_a_file_is_empty(monkeypatch, tmp_path):
    _set_single_cell_env(monkeypatch)
    results = tmp_path / "results"
    _write(results / RAW_NAME, '{"model": "phi-4-mini"}\n')
    _write(results / SUMMARY_NAME, "")  # present but empty
    assert rt.verify_success(str(tmp_path)) is False


def test_verify_success_false_when_only_one_of_two_requested_cells_is_present(
    monkeypatch, tmp_path
):
    # RAG_MODELS/RAG_CORPORA can request MULTIPLE cells at once -- every one
    # of them must have written files, not just the last to finish. This is
    # the exact shape of the original bug: a partial matrix must not report
    # success just because SOME file exists.
    monkeypatch.setenv("RAG_MODELS", "phi-4-mini")
    monkeypatch.setenv("RAG_CORPORA", "nq_open,ms_marco")
    monkeypatch.setenv("INFERENCE_ENGINE", "hf")
    results = tmp_path / "results"
    _write(results / "baseline_raw_phi-4-mini_nq_open_hf.jsonl", '{"model": "phi-4-mini"}\n')
    _write(results / "baseline_summary_phi-4-mini_nq_open_hf.csv", "model,corpus,n\n")
    # ms_marco's cell never wrote anything
    assert rt.verify_success(str(tmp_path)) is False


# --------------------------------------------------------------------------
# Task 3.2 -- the pod is terminated ONLY on a verified-successful run.
# Patching requests.delete makes an accidental real API call impossible.
# --------------------------------------------------------------------------

@pytest.fixture
def mock_delete(monkeypatch):
    """
    Stand in for requests.delete. Patched on the `requests` module itself, so
    it applies no matter how run_and_terminate reaches it. The returned
    response is a MagicMock, so raise_for_status() is a no-op -- the 2xx path.
    """
    fake = mock.MagicMock(name="requests.delete")
    monkeypatch.setattr("requests.delete", fake)
    return fake


@pytest.fixture
def mock_idle(monkeypatch):
    """
    Stand in for _idle_forever, the base of every failure path's shared
    choke point (_fail_and_idle). Records that it was reached instead of
    actually looping forever, so a failure-path test can assert "the script
    idled" without hanging.
    """
    calls = []
    monkeypatch.setattr(rt, "_idle_forever", lambda: calls.append(True))
    return calls


def test_no_terminate_when_pipeline_fails(monkeypatch, tmp_path, mock_delete, mock_idle):
    monkeypatch.setenv("RAG_SCRATCH_DIR", str(tmp_path))
    monkeypatch.setenv("RUNPOD_TERMINATE_KEY", "should-never-be-used")
    monkeypatch.setenv("RUNPOD_POD_ID", "should-never-be-used")
    monkeypatch.setattr(rt, "run_pipeline", lambda: 3)  # non-zero exit

    rt.main()  # must NOT sys.exit -- routes through _fail_and_idle instead

    assert mock_idle == [True]
    mock_delete.assert_not_called()


def test_no_terminate_when_exit_zero_but_results_missing(
    monkeypatch, tmp_path, mock_delete, mock_idle
):
    # Sweep "succeeds" (exit 0) but writes nothing -- the silent-failure case.
    monkeypatch.setenv("RAG_SCRATCH_DIR", str(tmp_path))
    monkeypatch.setenv("RUNPOD_TERMINATE_KEY", "should-never-be-used")
    monkeypatch.setenv("RUNPOD_POD_ID", "should-never-be-used")
    monkeypatch.setattr(rt, "run_pipeline", lambda: 0)

    rt.main()

    assert mock_idle == [True]
    mock_delete.assert_not_called()


def test_no_terminate_when_a_result_file_is_empty(
    monkeypatch, tmp_path, mock_delete, mock_idle
):
    # Exit 0 and both files present, but one is zero bytes -- still unverified.
    _set_single_cell_env(monkeypatch)
    results = tmp_path / "results"
    _write(results / RAW_NAME, '{"model": "phi-4-mini"}\n')
    _write(results / SUMMARY_NAME, "")
    monkeypatch.setenv("RAG_SCRATCH_DIR", str(tmp_path))
    monkeypatch.setenv("RUNPOD_TERMINATE_KEY", "should-never-be-used")
    monkeypatch.setenv("RUNPOD_POD_ID", "should-never-be-used")
    monkeypatch.setattr(rt, "run_pipeline", lambda: 0)

    rt.main()

    assert mock_idle == [True]
    mock_delete.assert_not_called()


def test_refuses_to_run_without_scratch_dir(monkeypatch, mock_delete, mock_idle):
    monkeypatch.delenv("RAG_SCRATCH_DIR", raising=False)
    # Clear the RunPod credentials too, so this test is hermetic: if it only
    # unset RAG_SCRATCH_DIR, a developer machine that happens to export these
    # would let the test pass for the wrong reason. With them absent, reaching
    # the termination path at all would raise KeyError rather than quietly
    # succeed -- so the assertions below prove the guard, not the environment.
    monkeypatch.delenv("RUNPOD_TERMINATE_KEY", raising=False)
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    # run_pipeline must never even start if scratch dir is unset
    monkeypatch.setattr(
        rt, "run_pipeline", lambda: pytest.fail("pipeline should not start")
    )

    rt.main()

    assert mock_idle == [True]
    mock_delete.assert_not_called()


def test_terminates_only_on_verified_success(monkeypatch, tmp_path, mock_delete):
    _set_single_cell_env(monkeypatch)
    results = tmp_path / "results"
    _write(results / RAW_NAME, '{"model": "phi-4-mini"}\n')
    _write(results / SUMMARY_NAME, "model,corpus,n\n")
    monkeypatch.setenv("RAG_SCRATCH_DIR", str(tmp_path))
    monkeypatch.setenv("RUNPOD_TERMINATE_KEY", "test-key")
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-abc123")
    monkeypatch.setattr(rt, "run_pipeline", lambda: 0)

    rt.main()  # no SystemExit on the happy path

    mock_delete.assert_called_once()
    args, kwargs = mock_delete.call_args
    url = args[0] if args else kwargs["url"]
    # The pod being destroyed must be the one this container is running on.
    assert url == "https://rest.runpod.io/v1/pods/pod-abc123"
    assert kwargs["headers"] == {"Authorization": "Bearer test-key"}
    # A hung terminate call would leave the pod billing indefinitely.
    assert kwargs["timeout"] == 30
    # A non-2xx must surface, not be swallowed into a "terminated" claim.
    mock_delete.return_value.raise_for_status.assert_called_once()


def test_terminate_pod_raises_on_non_2xx(mock_delete):
    # raise_for_status() is what turns a failed DELETE into a loud error.
    mock_delete.return_value.raise_for_status.side_effect = RuntimeError("401")

    with pytest.raises(RuntimeError):
        rt.terminate_pod("pod-abc123", "bad-key")


class _FakeProc:
    """Stand-in for subprocess.Popen's return value."""

    def __init__(self, returncode=0, hang=False, pid=4242):
        self._returncode = returncode
        self._hang = hang
        self.pid = pid

    def wait(self, timeout=None):
        # Only the first wait() (the bounded one, called with a timeout) hangs.
        # The post-kill proc.wait() (no timeout arg) reflects a process that
        # has now actually died, same as a real killed process would.
        if self._hang and timeout is not None:
            self._hang = False
            raise rt.subprocess.TimeoutExpired(cmd="fake", timeout=timeout)
        return self._returncode


def test_run_pipeline_stops_at_first_failing_stage(monkeypatch):
    calls = []

    def fake_popen(cmd, *a, **k):
        calls.append(cmd)
        # second stage (build_index) fails; third must never run
        rc = 2 if "data.build_index" in cmd else 0
        return _FakeProc(returncode=rc)

    monkeypatch.setattr(rt.subprocess, "Popen", fake_popen)
    assert rt.run_pipeline() == 2
    # loader ran, build_index ran and failed, run_baseline never invoked
    assert len(calls) == 2
    assert not any("evaluation.run_baseline" in c for c in calls)


# --------------------------------------------------------------------------
# Bug 1 -- a hang must be bounded by a real timeout, not left to run forever.
# --------------------------------------------------------------------------

def test_run_pipeline_returns_none_and_kills_process_group_on_timeout(monkeypatch):
    def fake_popen(cmd, *a, **k):
        # own process group, so a timeout kill takes any child processes
        # with it -- a plain proc.kill() only kills the direct child
        assert k.get("start_new_session") is True
        return _FakeProc(hang=True, pid=777)

    killpg_calls = []
    monkeypatch.setattr(rt.subprocess, "Popen", fake_popen)
    # raising=False: os.getpgid/os.killpg are POSIX-only and don't exist as
    # attributes on Windows' os module at all, so setattr must be allowed to
    # create them rather than requiring a pre-existing attribute.
    monkeypatch.setattr(
        rt.os, "getpgid", lambda pid: 999 if pid == 777 else pid, raising=False
    )
    monkeypatch.setattr(
        rt.os,
        "killpg",
        lambda pgid, sig: killpg_calls.append((pgid, sig)),
        raising=False,
    )
    monkeypatch.setattr(rt, "PIPELINE_TIMEOUT_SECONDS", 1)

    assert rt.run_pipeline() is None
    # killed the process GROUP (via getpgid), not the raw pid, with SIGKILL
    assert killpg_calls == [(999, getattr(rt.signal, "SIGKILL", rt.signal.SIGTERM))]


def test_run_pipeline_unaffected_by_timeout_on_normal_fast_completion(monkeypatch):
    monkeypatch.setattr(
        rt.subprocess, "Popen", lambda cmd, *a, **k: _FakeProc(returncode=0)
    )
    assert rt.run_pipeline() == 0


def test_main_idles_and_does_not_terminate_on_timeout(
    monkeypatch, tmp_path, mock_delete, mock_idle
):
    monkeypatch.setenv("RAG_SCRATCH_DIR", str(tmp_path))
    monkeypatch.setenv("RUNPOD_TERMINATE_KEY", "should-never-be-used")
    monkeypatch.setenv("RUNPOD_POD_ID", "should-never-be-used")
    # None is what run_pipeline returns on timeout -- distinct from a real
    # exit code. main() must route it through _fail_and_idle, not sys.exit.
    monkeypatch.setattr(rt, "run_pipeline", lambda: None)

    rt.main()

    assert mock_idle == [True]
    mock_delete.assert_not_called()


@pytest.mark.skipif(
    rt.os.name != "posix",
    reason="os.killpg/os.getpgid are POSIX-only; production runs in Linux Docker",
)
def test_run_pipeline_real_hang_is_actually_killed_within_timeout_window(tmp_path):
    """
    End-to-end with a real subprocess (no mocking), per the task brief: a
    genuinely hanging process, bounded by a short timeout, is actually killed
    within roughly the timeout window -- not merely reported as timed out
    while still running in the background.
    """
    import time

    script = tmp_path / "hang.py"
    script.write_text("import time\ntime.sleep(999)\n", encoding="utf-8")
    original_stages, original_timeout = rt.PIPELINE_STAGES, rt.PIPELINE_TIMEOUT_SECONDS
    try:
        rt.PIPELINE_STAGES = ([rt.sys.executable, str(script)],)
        rt.PIPELINE_TIMEOUT_SECONDS = 2

        start = time.time()
        result = rt.run_pipeline()
        elapsed = time.time() - start

        assert result is None
        assert elapsed < 10  # generous ceiling well above the 2s timeout
    finally:
        rt.PIPELINE_STAGES = original_stages
        rt.PIPELINE_TIMEOUT_SECONDS = original_timeout


# --------------------------------------------------------------------------
# A failed termination call must never crash the script. Confirmed by direct
# observation, twice, on real RunPod pods: pipeline succeeds -> terminate_pod
# raises -> unhandled exception crashes the process -> RunPod's orchestration
# reads the crash-exit as "needs retrying" -> the entire pipeline re-runs
# from scratch. Worse than Bug 1's hang: a hang bills at a steady rate, this
# compounds into full paid re-runs.
# --------------------------------------------------------------------------

def _happy_path_env(monkeypatch, tmp_path):
    _set_single_cell_env(monkeypatch)
    results = tmp_path / "results"
    _write(results / RAW_NAME, '{"model": "phi-4-mini"}\n')
    _write(results / SUMMARY_NAME, "model,corpus,n\n")
    monkeypatch.setenv("RAG_SCRATCH_DIR", str(tmp_path))
    monkeypatch.setenv("RUNPOD_TERMINATE_KEY", "test-key")
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-abc123")
    monkeypatch.setattr(rt, "run_pipeline", lambda: 0)


def test_terminate_failure_does_not_crash_main_and_idles_instead(
    monkeypatch, tmp_path, mock_delete, mock_idle, capsys
):
    _happy_path_env(monkeypatch, tmp_path)
    mock_delete.return_value.raise_for_status.side_effect = requests.HTTPError("403 Forbidden")

    rt.main()  # must NOT raise and must NOT sys.exit

    assert mock_idle == [True]
    err = capsys.readouterr().err
    # the message needs to be unmistakable in a log stream -- this is the
    # only thing standing between "billing continues but is visible and
    # bounded" and another expensive re-run loop
    assert "PIPELINE SUCCEEDED" in err
    assert "Termination call FAILED" in err
    assert "may still be billing" in err


def test_terminate_failure_of_any_exception_type_is_caught(
    monkeypatch, tmp_path, mock_delete, mock_idle
):
    # Not just HTTPError -- a connection error, timeout, or anything else
    # raised by requests.delete/raise_for_status must also be caught.
    _happy_path_env(monkeypatch, tmp_path)
    mock_delete.side_effect = requests.ConnectionError("credential issue")

    rt.main()

    assert mock_idle == [True]


def test_terminate_success_path_unchanged_no_idle_loop_triggered(
    monkeypatch, tmp_path, mock_delete, mock_idle, capsys
):
    _happy_path_env(monkeypatch, tmp_path)

    rt.main()  # no SystemExit, no exception

    mock_delete.assert_called_once()
    assert mock_idle == []  # success must never idle
    assert "pod terminated successfully." in capsys.readouterr().out


def test_fail_and_idle_logs_message_and_delegates_to_idle_forever(
    monkeypatch, mock_idle, capsys
):
    rt._fail_and_idle("something specific went wrong")

    assert mock_idle == [True]
    err = capsys.readouterr().err
    assert "FAILURE: something specific went wrong" in err
    assert "idling instead of exiting" in err


def test_idle_forever_actually_calls_time_sleep_in_a_loop(monkeypatch):
    """
    Confirm _idle_forever really loops on time.sleep (not e.g. a no-op or a
    single sleep) without actually sleeping for real hours in the test --
    time.sleep is mocked to raise after a few calls, which both proves the
    loop and stops it.
    """
    sleep_calls = []

    def fake_sleep(seconds):
        sleep_calls.append(seconds)
        if len(sleep_calls) >= 3:
            raise KeyboardInterrupt("stop the test's idle loop")

    monkeypatch.setattr(rt.time, "sleep", fake_sleep)

    with pytest.raises(KeyboardInterrupt):
        rt._idle_forever()

    assert sleep_calls == [3600, 3600, 3600]


def test_timeout_never_reaches_terminate_pod_call_site(
    monkeypatch, tmp_path, mock_delete, mock_idle
):
    """
    Bug 1's timeout wraps the pipeline subprocess itself (run_pipeline,
    before verify_success). Termination is a later, separate step in main()
    (terminate_pod, after verify_success). Both routes now share the same
    _fail_and_idle -> _idle_forever choke point (that's the point of this
    refactor) -- what must stay distinct is that a pipeline timeout never
    reaches terminate_pod's call site at all, since verify_success() is
    never even reached on a timeout. Explicitly checked here, not assumed.
    """
    monkeypatch.setenv("RAG_SCRATCH_DIR", str(tmp_path))
    monkeypatch.setenv("RUNPOD_TERMINATE_KEY", "test-key")
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-abc123")
    monkeypatch.setattr(rt, "run_pipeline", lambda: None)  # simulates a timeout

    rt.main()

    mock_delete.assert_not_called()  # terminate_pod's call site never reached
    assert mock_idle == [True]  # still routed through the shared choke point


# --------------------------------------------------------------------------
# Final backstop: catch anything the 5 known failure sites (tested above)
# don't. Bug 4's audit covers every failure path currently written into
# main() -- this catches an exception type nobody's hit yet: a KeyError
# from an unchecked env var, an OSError from a full Network Volume,
# something raised deep inside a dependency that's never been triggered
# before. Specifically testing the UNKNOWN case, not re-testing the 5
# already-covered ones.
# --------------------------------------------------------------------------

def test_backstop_catches_an_unrelated_exception_type_not_among_the_5_known_sites(
    monkeypatch, tmp_path, mock_delete, mock_idle
):
    monkeypatch.setenv("RAG_SCRATCH_DIR", str(tmp_path))
    monkeypatch.setenv("RUNPOD_TERMINATE_KEY", "test-key")
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-abc123")

    def raise_unexpected():
        raise RuntimeError("cudaErrorDevicesUnavailable: unrelated to any of the 5 known sites")

    monkeypatch.setattr(rt, "run_pipeline", raise_unexpected)

    rt._main_with_backstop()  # must NOT raise and must NOT sys.exit

    assert mock_idle == [True]
    mock_delete.assert_not_called()


def test_backstop_logs_message_includes_the_exception_repr(
    monkeypatch, tmp_path, mock_idle, capsys
):
    monkeypatch.setenv("RAG_SCRATCH_DIR", str(tmp_path))

    def raise_unexpected():
        raise KeyError("SOME_UNCHECKED_ENV_VAR")

    monkeypatch.setattr(rt, "run_pipeline", raise_unexpected)

    rt._main_with_backstop()

    err = capsys.readouterr().err
    assert "unhandled exception" in err
    assert "SOME_UNCHECKED_ENV_VAR" in err


def test_backstop_lets_keyboard_interrupt_and_system_exit_propagate(monkeypatch, mock_idle):
    # Exception, not a bare except: -- genuine intentional interrupts must
    # NOT be swallowed into the idle loop.
    monkeypatch.setattr(
        rt, "main", mock.Mock(side_effect=KeyboardInterrupt("interactive Ctrl-C"))
    )
    with pytest.raises(KeyboardInterrupt):
        rt._main_with_backstop()
    assert mock_idle == []

    monkeypatch.setattr(rt, "main", mock.Mock(side_effect=SystemExit(0)))
    with pytest.raises(SystemExit):
        rt._main_with_backstop()
    assert mock_idle == []


def test_backstop_does_not_interfere_with_the_5_known_failure_sites(
    monkeypatch, tmp_path, mock_delete, mock_idle
):
    # The 5 known sites still route through main()'s own _fail_and_idle
    # calls -- confirm the outer backstop doesn't double-trigger or change
    # that behavior when wrapping a normal (non-exception) failure path.
    monkeypatch.delenv("RAG_SCRATCH_DIR", raising=False)

    rt._main_with_backstop()

    assert mock_idle == [True]  # exactly once, not twice
    mock_delete.assert_not_called()


def test_backstop_does_not_interfere_with_genuine_success_path(
    monkeypatch, tmp_path, mock_delete, mock_idle, capsys
):
    _happy_path_env(monkeypatch, tmp_path)

    rt._main_with_backstop()

    mock_delete.assert_called_once()
    assert mock_idle == []
    assert "pod terminated successfully." in capsys.readouterr().out
