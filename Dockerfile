FROM python:3.14-slim

WORKDIR /app

# Install system dependencies needed for compiling packages (like FAISS or text tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker's caching layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your RAG application source code
COPY . .

# Cache datasets, build FAISS indices, and execute the evaluation sweep sequentially
CMD ["sh", "-c", "python -m data.loader && python -m data.build_index && python -m evaluation.run_baseline"]
