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
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    swig \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker's caching layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your RAG application source code
COPY . .

# Containers have no TTY, so Python block-buffers stdout (~8 KB) by default.
# The sweep's progress prints are tiny, so hours of output would sit invisible
# in the buffer (observed on the 2026-07-11 Azure run). Unbuffered stdout makes
# `az containerapp logs show --follow` show progress in real time.
ENV PYTHONUNBUFFERED=1

# Cache datasets, build FAISS indices, then run the sweep — or, with
# RAG_MODE=debug, print raw sample answers instead (see evaluation/debug_sample.py).
# The trailing `sleep infinity` stops Container Apps from restarting the
# container when the pipeline exits: the 2026-07-10 run restarted ~5 times
# overnight, rerunning the full A100 sweep and wiping results each time.
# The container now idles when done — results stay readable via exec until
# the app is stopped (stop it to end billing).
CMD ["sh", "-c", "python -m data.loader && python -m data.build_index && if [ \"$RAG_MODE\" = debug ]; then python -m evaluation.debug_sample; else python -m evaluation.run_baseline; fi; echo '[entrypoint] pipeline exited — idling so results survive; stop the app to end billing'; sleep infinity"]
