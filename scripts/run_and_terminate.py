import os
import signal
import subprocess
import sys
import time
from pathlib import Path
import requests
from evaluation.result_paths import expected_attack_result_files, expected_result_files
PIPELINE_TIMEOUT_SECONDS = int(os.environ.get('RAG_PIPELINE_TIMEOUT_SECONDS', 21600))
_RUN_TARGETS = {'baseline': ('evaluation.run_baseline', expected_result_files), 'attack_injection': ('evaluation.run_attack_injection', expected_attack_result_files)}

def _resolve_run_target() -> str:
    target = os.environ.get('RAG_RUN_TARGET', 'baseline')
    if target not in _RUN_TARGETS:
        raise ValueError(f"Unknown RAG_RUN_TARGET '{target}'. Options: {list(_RUN_TARGETS)}")
    return target

def _pipeline_stages(target: str):
    module_name, _ = _RUN_TARGETS[target]
    return ([sys.executable, '-m', 'data.loader'], [sys.executable, '-m', 'data.build_index'], [sys.executable, '-m', module_name])

def verify_success(scratch_dir: str, target: str) -> bool:
    _, expected_files_fn = _RUN_TARGETS[target]
    results_dir = Path(scratch_dir) / 'results'
    if not results_dir.exists():
        return False
    for f in expected_files_fn(results_dir):
        if not f.exists() or f.stat().st_size == 0:
            return False
    return True

def run_pipeline(target: str):
    for cmd in _pipeline_stages(target):
        proc = subprocess.Popen(cmd, start_new_session=True)
        try:
            returncode = proc.wait(timeout=PIPELINE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), getattr(signal, 'SIGKILL', signal.SIGTERM))
            proc.wait()
            print(f"[run_and_terminate] stage {' '.join(cmd)} exceeded {PIPELINE_TIMEOUT_SECONDS}s -- killed. NOT terminating pod (same as any other failure), but this bounds the hang instead of leaving it unbounded.", file=sys.stderr)
            return None
        if returncode != 0:
            print(f"[run_and_terminate] stage {' '.join(cmd)} exited {returncode} -- stopping pipeline.", file=sys.stderr)
            return returncode
    return 0

def terminate_pod(pod_id: str, api_key: str) -> None:
    response = requests.delete(f'https://rest.runpod.io/v1/pods/{pod_id}', headers={'Authorization': f'Bearer {api_key}'}, timeout=30)
    response.raise_for_status()

def _fail_and_idle(message: str) -> None:
    print(f"[run_and_terminate] FAILURE: {message} -- idling instead of exiting. Any exit here has been directly observed to trigger a full pipeline re-run rather than a clean stop. Stop this pod manually once you've finished inspecting.", file=sys.stderr)
    _idle_forever()

def _idle_forever() -> None:
    while True:
        time.sleep(3600)

def main() -> None:
    scratch_dir = os.environ.get('RAG_SCRATCH_DIR')
    if not scratch_dir:
        _fail_and_idle('RAG_SCRATCH_DIR not set -- refusing to run')
        return
    try:
        target = _resolve_run_target()
    except ValueError as e:
        _fail_and_idle(str(e))
        return
    returncode = run_pipeline(target)
    if returncode is None:
        _fail_and_idle('pipeline TIMED OUT')
        return
    if returncode != 0:
        _fail_and_idle(f'pipeline stage exited {returncode}')
        return
    if not verify_success(scratch_dir, target):
        _fail_and_idle('pipeline exited 0 but expected output files are missing or empty -- investigate before assuming this run succeeded')
        return
    print('[run_and_terminate] sweep confirmed successful. Terminating pod.')
    pod_id = os.environ['RUNPOD_POD_ID']
    api_key = os.environ['RUNPOD_TERMINATE_KEY']
    try:
        terminate_pod(pod_id, api_key)
        print('[run_and_terminate] pod terminated successfully.')
    except Exception as e:
        _fail_and_idle(f'PIPELINE SUCCEEDED, results are safe on the Network Volume. Termination call FAILED: {e!r}. This pod may still be billing -- check manually')
        return

def _main_with_backstop() -> None:
    try:
        main()
    except Exception as e:
        _fail_and_idle(f'unhandled exception: {e!r}')
if __name__ == '__main__':
    _main_with_backstop()
