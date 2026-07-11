FROM python:3.14-slim

WORKDIR /app

# Install system dependencies needed for compiling packages (like FAISS or text tools).
# swig + libopenblas-dev cover the faiss-cpu source-build fallback: Linux cp314
# wheels are not confirmed for faiss-cpu (see requirements.txt), so if pip cannot
# find a wheel it compiles from source and needs these headers.
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

# Cache datasets, build FAISS indices, and execute the evaluation sweep sequentially
CMD ["sh", "-c", "python -m data.loader && python -m data.build_index && python -m evaluation.run_baseline"]
