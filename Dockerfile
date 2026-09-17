# 3.13, not 3.14: vLLM's abi3 wheel itself installs on 3.14, but its dependency
# tree does not — flashinfer's cuda-tile, ray, numba <0.63, and outlines-core
# <0.2.14 all lack cp314 wheels (vllm-project/vllm#34096, still a tracking
# issue as of vllm 0.25.1). The Colab validation of vllm==0.25.1 also never
# ran on 3.14. 3.13 additionally has confirmed Linux faiss-cpu wheels
# (see requirements.txt).
FROM python:3.13-slim

WORKDIR /app

# Install system dependencies needed for compiling packages (like FAISS or text tools).
# faiss-cpu ships confirmed Linux cp313 wheels, so the swig + libopenblas-dev
# source-build fallback should never trigger on this base image; they are kept
# as insurance against a wheel-resolution surprise in the paid Azure build.
# ffmpeg: torchcodec (a transitive dependency of vllm, unused by this
# text-only pipeline) dlopen()s FFmpeg's shared libraries at import time and
# fails hard if they're absent (OSError: Could not load this library,
# observed on the 2026-07-27 RunPod smoke test). Colab's base image ships
# FFmpeg, masking the gap until the image was actually run, not just built.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    swig \
    libopenblas-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker's caching layer
COPY requirements.txt .

# The wheel set here is ~4 GB (torch 530 MB, vllm 250 MB, flashinfer_cubin
# 458 MB, plus several nvidia_* wheels of 170-370 MB each). Two local builds
# on 2026-07-27 died mid-download with
# `ProtocolError: Connection broken: IncompleteRead`, and because the install
# ran with --no-cache-dir, each retry restarted the whole 4 GB from zero.
#
# --mount=type=cache keeps pip's HTTP cache across builds, so a dropped
# connection costs only the wheels not yet fetched rather than all of them.
# This is a BuildKit mount, not an image layer -- the cache never lands in
# the final image, so it does not undo what --no-cache-dir was there for
# (image size). --retries/--timeout make pip itself ride out the flaky reads
# that killed both earlier attempts.
RUN --mount=type=cache,target=/root/.cache/pip \
    PIP_DEFAULT_TIMEOUT=120 pip install --retries 10 -r requirements.txt

COPY . .

# Containers have no TTY, so Python block-buffers stdout (~8 KB) by default.
# The sweep's progress prints are tiny, so hours of output would sit invisible
# in the buffer (observed on the 2026-07-11 Azure run). Unbuffered stdout makes
# `az containerapp logs show --follow` show progress in real time.
ENV PYTHONUNBUFFERED=1

# flashinfer's sampler backend needs a CUDA compiler to JIT-compile its kernel
# on first use; this image doesn't have one (nvcc is not installed -- only
# runtime CUDA libs ship via the pip wheels). Unlike RAG_VLLM_ATTENTION_BACKEND
# (harness/vllm_engine.py), this is vLLM's own recognized env var, not a
# custom RAG_-prefixed one, so it has to be set here rather than defaulted in
# Python -- vLLM reads it via its own env-var module. Confirmed working on the
# 2026-07-28 RunPod vLLM sweep that produced Phase 1's real numbers; was set
# ad hoc in that pod session and never baked into the image until now, which
# is why it went missing from the repo despite the run having used it.
ENV VLLM_USE_FLASHINFER_SAMPLER=0

# Cache datasets, build FAISS indices, then run the sweep — or, with
# RAG_MODE=debug, print raw sample answers instead (see evaluation/debug_sample.py).
# The trailing `sleep infinity` stops Container Apps from restarting the
# container when the pipeline exits: the 2026-07-10 run restarted ~5 times
# overnight, rerunning the full A100 sweep and wiping results each time.
# The container now idles when done — results stay readable via exec until
# the app is stopped (stop it to end billing).
CMD ["sh", "-c", "python -m data.loader && python -m data.build_index && if [ \"$RAG_MODE\" = debug ]; then python -m evaluation.debug_sample; else python -m evaluation.run_baseline; fi; echo '[entrypoint] pipeline exited — idling so results survive; stop the app to end billing'; sleep infinity"]
