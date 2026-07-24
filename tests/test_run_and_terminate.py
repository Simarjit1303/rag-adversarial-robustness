"""
Local, zero-cost tests for the RunPod self-termination wrapper.

Nothing here touches a real RunPod pod or the RunPod API. The failure-path
tests inject a fake `runpod` module into sys.modules and also stub the
wrapper's own terminate_pod(), so a real API call is impossible even if the
control flow were wrong -- the point being to PROVE the pod is never
terminated on a failed or unverified run.
"""

import sys
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
# A fake runpod module makes an accidental real API call impossible.
# --------------------------------------------------------------------------

@pytest.fixture
def fake_runpod(monkeypatch):
    fake = mock.MagicMock(name="runpod")
    monkeypatch.setitem(sys.modules, "runpod", fake)
    return fake


def test_no_terminate_when_pipeline_fails(monkeypatch, tmp_path, fake_runpod):
    monkeypatch.setenv("RAG_SCRATCH_DIR", str(tmp_path))
    monkeypatch.setenv("RUNPOD_API_KEY", "should-never-be-used")
    monkeypatch.setenv("RUNPOD_POD_ID", "should-never-be-used")
    monkeypatch.setattr(rt, "run_pipeline", lambda: 3)  # non-zero exit

    with pytest.raises(SystemExit) as exc:
        rt.main()

    assert exc.value.code == 3
    fake_runpod.terminate_pod.assert_not_called()


def test_no_terminate_when_exit_zero_but_results_missing(
    monkeypatch, tmp_path, fake_runpod
):
    # Sweep "succeeds" (exit 0) but writes nothing -- the silent-failure case.
    monkeypatch.setenv("RAG_SCRATCH_DIR", str(tmp_path))
    monkeypatch.setenv("RUNPOD_API_KEY", "should-never-be-used")
    monkeypatch.setenv("RUNPOD_POD_ID", "should-never-be-used")
    monkeypatch.setattr(rt, "run_pipeline", lambda: 0)

    with pytest.raises(SystemExit) as exc:
        rt.main()

    assert exc.value.code == 1
    fake_runpod.terminate_pod.assert_not_called()


def test_refuses_to_run_without_scratch_dir(monkeypatch, fake_runpod):
    monkeypatch.delenv("RAG_SCRATCH_DIR", raising=False)
    # run_pipeline must never even start if scratch dir is unset
    monkeypatch.setattr(
        rt, "run_pipeline", lambda: pytest.fail("pipeline should not start")
    )

    with pytest.raises(SystemExit) as exc:
        rt.main()

    assert exc.value.code == 1
    fake_runpod.terminate_pod.assert_not_called()


def test_terminates_only_on_verified_success(monkeypatch, tmp_path, fake_runpod):
    results = tmp_path / "results"
    _write(results / "baseline_raw.jsonl", '{"model": "phi-4-mini"}\n')
    _write(results / "baseline_summary.csv", "model,corpus,n\n")
    monkeypatch.setenv("RAG_SCRATCH_DIR", str(tmp_path))
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-abc123")
    monkeypatch.setattr(rt, "run_pipeline", lambda: 0)

    rt.main()  # no SystemExit on the happy path

    fake_runpod.terminate_pod.assert_called_once_with("pod-abc123")
    assert fake_runpod.api_key == "test-key"


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
