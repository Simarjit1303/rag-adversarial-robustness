# Fix: runpod SDK conflicts with vllm — replace with direct REST call

## The problem, confirmed by an actual build

`pip install -r requirements.txt` fails outright:

```
vllm 0.25.1 depends on fastapi<0.137.0 and >=0.133.0
runpod 1.11.0 depends on fastapi>=0.139.2
```

These ranges don't overlap. `fastapi` is a hard, non-optional dependency of
the `runpod` PyPI package (confirmed via its published dependency list —
not gated behind an extras flag), so there's no "install runpod without
the web bits" option via pip. This means PR #11's local verification never
actually ran a full `pip install -r requirements.txt` together — the tests
correctly mocked `runpod` for testing termination *logic*, but that also
meant this conflict went undetected. Worth noting for future PRs: mocking
a dependency for unit tests doesn't substitute for confirming it's
actually installable alongside everything else.

## The fix: drop `runpod`, call the REST API directly

RunPod's terminate action is a plain HTTP DELETE, confirmed current
(docs.runpod.io, page dated within the last week):

```
DELETE https://rest.runpod.io/v1/pods/{podId}
Authorization: Bearer <token>
```

Returns `204` on success. This needs only `requests`, already present as
a transitive dependency of `vllm` itself — confirmed directly in the build
log (`Collecting requests>=2.26.0 (from vllm==0.25.1...)`). No new package,
no new conflict surface.

### Task 1 — remove `runpod` from `requirements.txt`

Delete the `runpod==1.11.0` line entirely.

### Task 2 — rewrite `scripts/run_and_terminate.py`

Replace the `import runpod` / `runpod.terminate_pod(...)` block with:

```python
import requests

def terminate_pod(pod_id: str, api_key: str) -> None:
    response = requests.delete(
        f"https://rest.runpod.io/v1/pods/{pod_id}",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    response.raise_for_status()  # raises on anything other than 2xx
```

Call this from `main()` exactly where `runpod.terminate_pod(...)` was
called before — same place in the control flow, same
success-only-after-`verify_success()` guarantee, nothing about *when*
termination happens should change, only *how* the call is made.

### Task 3 — update the tests

The existing tests mocked the `runpod` module via `sys.modules` injection.
Replace that with mocking `requests.delete` instead (e.g.
`unittest.mock.patch("requests.delete")` or `responses`/`requests-mock` if
already available). Same guarantees to re-prove:

- `requests.delete` is never called on pipeline failure.
- Never called when exit code is 0 but expected output files are missing
  or empty.
- Never called when `RAG_SCRATCH_DIR` is unset.
- Called exactly once, with the correct URL (containing the pod ID) and
  the correct Authorization header, only on verified success.

### Task 4 — actually verify the fix resolves the real conflict

Don't just trust that removing `runpod` fixes it — confirm with a real
build, the same way the conflict was actually discovered:

```powershell
docker build -t ghcr.io/simarjit1303/rag-adversarial-robustness/thesis:runpod-test .
```

This must complete successfully through the `pip install` step before this
fix is considered done. A clean local build is the actual proof, not
reading the diff.

## Where this lands

Amend this into the existing `runpod-self-termination` branch (PR #11 is
still open, unmerged — same standing rule as every other PR this session:
amend before merge, don't stack a second PR fixing a bug in a PR that
hasn't landed yet). Update the PR description to note the dependency
conflict was found and fixed, with the reasoning above, so the PR's final
state is accurate on its own.
