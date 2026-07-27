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


def test_run_pipeline_stops_at_first_failing_stage(monkeypatch):
    calls = []

    def fake_run(cmd, *a, **k):
        calls.append(cmd)
        # second stage (build_index) fails; third must never run
        rc = 2 if "data.build_index" in cmd else 0
        return mock.Mock(returncode=rc)

    monkeypatch.setattr(rt.subprocess, "run", fake_run)
    assert rt.run_pipeline() == 2
    # loader ran, build_index ran and failed, run_baseline never invoked
    assert len(calls) == 2
    assert not any("evaluation.run_baseline" in c for c in calls)
