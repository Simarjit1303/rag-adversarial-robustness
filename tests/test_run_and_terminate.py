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

from scripts import run_and_terminate as rt


# --------------------------------------------------------------------------
# Task 3.1 -- verify_success() against a temp results dir
# --------------------------------------------------------------------------

def _write(path, content="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_verify_success_true_when_both_files_present_and_nonempty(tmp_path):
    results = tmp_path / "results"
    _write(results / "baseline_raw.jsonl", '{"model": "phi-4-mini"}\n')
    _write(results / "baseline_summary.csv", "model,corpus,n\n")
    assert rt.verify_success(str(tmp_path)) is True


def test_verify_success_false_when_results_dir_missing(tmp_path):
    # scratch dir exists but no results/ subdir at all
    assert rt.verify_success(str(tmp_path)) is False


def test_verify_success_false_when_a_file_is_missing(tmp_path):
    results = tmp_path / "results"
    _write(results / "baseline_raw.jsonl", '{"model": "phi-4-mini"}\n')
    # summary CSV absent
    assert rt.verify_success(str(tmp_path)) is False


def test_verify_success_false_when_a_file_is_empty(tmp_path):
    results = tmp_path / "results"
    _write(results / "baseline_raw.jsonl", '{"model": "phi-4-mini"}\n')
    _write(results / "baseline_summary.csv", "")  # present but empty
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


def test_no_terminate_when_pipeline_fails(monkeypatch, tmp_path, mock_delete):
    monkeypatch.setenv("RAG_SCRATCH_DIR", str(tmp_path))
    monkeypatch.setenv("RUNPOD_API_KEY", "should-never-be-used")
    monkeypatch.setenv("RUNPOD_POD_ID", "should-never-be-used")
    monkeypatch.setattr(rt, "run_pipeline", lambda: 3)  # non-zero exit

    with pytest.raises(SystemExit) as exc:
        rt.main()

    assert exc.value.code == 3
    mock_delete.assert_not_called()


def test_no_terminate_when_exit_zero_but_results_missing(
    monkeypatch, tmp_path, mock_delete
):
    # Sweep "succeeds" (exit 0) but writes nothing -- the silent-failure case.
    monkeypatch.setenv("RAG_SCRATCH_DIR", str(tmp_path))
    monkeypatch.setenv("RUNPOD_API_KEY", "should-never-be-used")
    monkeypatch.setenv("RUNPOD_POD_ID", "should-never-be-used")
    monkeypatch.setattr(rt, "run_pipeline", lambda: 0)

    with pytest.raises(SystemExit) as exc:
        rt.main()

    assert exc.value.code == 1
    mock_delete.assert_not_called()


def test_no_terminate_when_a_result_file_is_empty(
    monkeypatch, tmp_path, mock_delete
):
    # Exit 0 and both files present, but one is zero bytes -- still unverified.
    results = tmp_path / "results"
    _write(results / "baseline_raw.jsonl", '{"model": "phi-4-mini"}\n')
    _write(results / "baseline_summary.csv", "")
    monkeypatch.setenv("RAG_SCRATCH_DIR", str(tmp_path))
    monkeypatch.setenv("RUNPOD_API_KEY", "should-never-be-used")
    monkeypatch.setenv("RUNPOD_POD_ID", "should-never-be-used")
    monkeypatch.setattr(rt, "run_pipeline", lambda: 0)

    with pytest.raises(SystemExit) as exc:
        rt.main()

    assert exc.value.code == 1
    mock_delete.assert_not_called()


def test_refuses_to_run_without_scratch_dir(monkeypatch, mock_delete):
    monkeypatch.delenv("RAG_SCRATCH_DIR", raising=False)
    # Clear the RunPod credentials too, so this test is hermetic: if it only
    # unset RAG_SCRATCH_DIR, a developer machine that happens to export these
    # would let the test pass for the wrong reason. With them absent, reaching
    # the termination path at all would raise KeyError rather than quietly
    # succeed -- so the assertions below prove the guard, not the environment.
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    # run_pipeline must never even start if scratch dir is unset
    monkeypatch.setattr(
        rt, "run_pipeline", lambda: pytest.fail("pipeline should not start")
    )

    with pytest.raises(SystemExit) as exc:
        rt.main()

    assert exc.value.code == 1
    mock_delete.assert_not_called()


def test_terminates_only_on_verified_success(monkeypatch, tmp_path, mock_delete):
    results = tmp_path / "results"
    _write(results / "baseline_raw.jsonl", '{"model": "phi-4-mini"}\n')
    _write(results / "baseline_summary.csv", "model,corpus,n\n")
    monkeypatch.setenv("RAG_SCRATCH_DIR", str(tmp_path))
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
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


def test_main_exits_1_and_does_not_terminate_on_timeout(
    monkeypatch, tmp_path, mock_delete
):
    monkeypatch.setenv("RAG_SCRATCH_DIR", str(tmp_path))
    monkeypatch.setenv("RUNPOD_API_KEY", "should-never-be-used")
    monkeypatch.setenv("RUNPOD_POD_ID", "should-never-be-used")
    # None is what run_pipeline returns on timeout -- distinct from a real
    # exit code, since sys.exit(None) would exit 0 and silently report a
    # hang as success. This proves main() catches that case specifically.
    monkeypatch.setattr(rt, "run_pipeline", lambda: None)

    with pytest.raises(SystemExit) as exc:
        rt.main()

    assert exc.value.code == 1
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
