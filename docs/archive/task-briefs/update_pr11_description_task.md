# Update PR #11's description — final, accurate summary

PR #11 (`runpod-self-termination`) has accumulated six real fixes across
this session, each found by actually running on RunPod hardware, not
assumed from code review. Write a description that tells this story
accurately, not just a bullet list of diffs. Structure:

## What this PR does

Self-terminating RunPod entrypoint (`scripts/run_and_terminate.py`) that
runs the pipeline, verifies success, and cleans up the pod — with defenses
against every failure mode discovered while actually testing it on real
infrastructure.

## The six fixes, in the order they were found

1. **ffmpeg / torchcodec** — `data.build_index` crashed on
   `OSError: Could not load this library` because `torchcodec` (a
   transitive `vllm` dependency, never used directly) needs FFmpeg shared
   libraries the base image didn't have. Fixed by adding `ffmpeg` to the
   Dockerfile's apt layer.

2. **Unbounded hang** — the pipeline subprocess had no timeout; a genuine
   hang could bill indefinitely with nothing to stop it. Fixed with
   `RAG_PIPELINE_TIMEOUT_SECONDS`, process-group `SIGKILL` on timeout.

3. **`RAG_CORPORA` never actually filtered anything** — documented but
   never wired into `data/build_index.py`'s corpus loop. Fixed in both
   `build_index.py` and `run_baseline.py`.

4. **Restart loop from ANY failure exit, not just termination failures** —
   originally only the termination-call-failure path idled instead of
   exiting; every other failure path (pipeline crash, timeout, missing env
   var) still exited normally, which was directly observed triggering full
   paid pipeline re-runs via RunPod's restart behavior. Fixed by routing
   every failure path through a single `_fail_and_idle()` choke point,
   plus a top-level exception backstop catching anything not yet known.

5. **`RUNPOD_API_KEY` naming collision** — RunPod auto-injects its own
   pod-scoped `RUNPOD_API_KEY`, confirmed via RunPod's own forum as a
   known, currently-unresolved limitation. Our custom credential of the
   same name was colliding with it, causing every termination attempt to
   403. Renamed to `RUNPOD_TERMINATE_KEY`.

6. **Intermittent CUDA device-busy race** — `cudaErrorDevicesUnavailable`
   when the embedder claimed the GPU immediately after `phi-4-mini`
   finished loading via `device_map='auto'`; confirmed intermittent (2
   clean runs, then 1 failure) rather than restart-related. Fixed with an
   explicit `torch.cuda.synchronize()` plus a bounded retry-with-backoff
   on embedder construction as a safety net.

## Verification

All six fixes locally unit-tested (38 tests total). All six confirmed on
real RunPod hardware as of the run at 2026-07-28 02:44 CEST: full pipeline
success, `phi-4-mini x nq_open, F1(clean)=0.9615, EM(clean)=0.9190, n=1000`,
results written to the Network Volume, pod terminated itself successfully
— first fully clean end-to-end run this session.

## What's still unverified

The `INFERENCE_ENGINE=vllm` path has never been exercised on RunPod —
every successful run so far used the default HF `transformers` path. This
needs its own smoke test before the real Phase 1 baseline sweep, which is
expected to use vLLM's batched path.
