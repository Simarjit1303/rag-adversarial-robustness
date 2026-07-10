"""
Central configuration for the Adversarial Robustness of Open-Source RAG
Pipelines harness.

Every model, corpus, and path used across Phases 1-4 is registered here so
nothing gets hardcoded deeper in the codebase. See the dissertation plan's
Section 4 (model selection criteria) and Section 5 (final model set) for why
these four models and no others.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
#
# The repo itself lives under a OneDrive-synced folder for this project
# (Simarjit's supervisor-shared drive), which is fine for the code — small
# text files, useful for Dr. Al Sardy to browse — but NOT fine for what this
# harness writes at runtime: cached 10,000-document corpora, FAISS indices,
# and per-question JSONL for every model. OneDrive does not respect
# .gitignore, so anything written under the repo folder gets uploaded
# regardless of whether git tracks it.
#
# LOCAL_SCRATCH defaults to a location OUTSIDE any synced folder. Override it
# with the RAG_SCRATCH_DIR environment variable if you want it somewhere
# else (e.g. a mounted Colab/Azure data disk) — no code changes needed on a
# different machine, just set the env var before running.
# ---------------------------------------------------------------------------
import os

ROOT_DIR = Path(__file__).resolve().parent
LOCAL_SCRATCH = Path(os.environ.get("RAG_SCRATCH_DIR", r"C:\rag-data-local"))

DATA_DIR = LOCAL_SCRATCH / "cache"
INDEX_DIR = LOCAL_SCRATCH / "indices"
RESULTS_DIR = LOCAL_SCRATCH / "results"

for d in (DATA_DIR, INDEX_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Target models
#
# "revision" is left as None on purpose. Before running anything you intend
# to report in the dissertation, pin each one to the exact commit hash shown
# on its HuggingFace "Files and versions" tab and fill it in here. That hash
# belongs in the reproducibility appendix, not "latest".
#
# loader: which class harness/model_loader.py should use to load it.
#   "causal_lm" -> standard AutoModelForCausalLM (Llama, Qwen3, Phi-4-mini)
#   "mistral3"  -> Mistral3ForConditionalGeneration (Ministral-3-8B ships
#                  with a bundled vision encoder; this harness never sends
#                  image input, so only the text path is ever exercised —
#                  verify this with scripts/verify_ministral_text_only.py
#                  before trusting any Ministral baseline numbers)
# ---------------------------------------------------------------------------
MODELS = {
    "llama-3.1-8b": {
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "revision": None,
        "loader": "causal_lm",
        "gated": True,  # requires accepting Meta's license on the model page first
        "license": "Llama 3.1 Community License",
        "note": "Retained: no dense 7-8B Llama 4 model exists (Llama 4 went straight to MoE).",
    },
    "qwen3-8b": {
        "hf_id": "Qwen/Qwen3-8B",
        "revision": None,
        "loader": "causal_lm",
        "gated": False,
        "license": "Apache-2.0",
        "note": "Replaces Qwen2.5-7B-Instruct. Supports a thinking-mode toggle, see QWEN3_ENABLE_THINKING below.",
    },
    "phi-4-mini": {
        "hf_id": "microsoft/Phi-4-mini-instruct",
        "revision": None,
        "loader": "causal_lm",
        "gated": False,
        "license": "MIT",
        "note": "Replaces Phi-3.5-mini-instruct.",
    },
    "ministral-3-8b": {
        "hf_id": "mistralai/Ministral-3-8B-Instruct-2512",
        "revision": None,
        "loader": "mistral3",
        "gated": False,
        "license": "Apache-2.0",
        "note": (
            "Replaces Mistral-7B-Instruct-v0.3. Ships as an 8.4B language backbone "
            "plus a 0.4B vision encoder. This harness only ever sends text, so the "
            "vision encoder is loaded but never invoked — confirmed by "
            "scripts/verify_ministral_text_only.py."
        ),
    },
}

# Qwen3 exposes an explicit thinking-mode switch via its chat template.
# Pick ONE value for the whole benchmark and do not let it vary between runs.
QWEN3_ENABLE_THINKING = False  # False = non-thinking mode, closer latency profile to the other three

# ---------------------------------------------------------------------------
# Knowledge-base corpora (retrieval side)
# ---------------------------------------------------------------------------
CORPORA = {
    "nq_open": {
        "hf_id": "google-research-datasets/nq_open",
        "dev_n": 1000,
        "eval_n": 10000,
    },
    "hotpot_qa": {
        "hf_id": "hotpotqa/hotpot_qa",
        "hf_config": "distractor",
        "dev_n": 1000,
        "eval_n": 10000,
    },
    "ms_marco": {
        "hf_id": "microsoft/ms_marco",
        "hf_config": "v2.1",
        "dev_n": 1000,
        "eval_n": 10000,
    },
}

# ---------------------------------------------------------------------------
# Utility / false-positive baseline (used from Phase 3 onward)
# ---------------------------------------------------------------------------
XSTEST_HF_ID = "paul-rottger/xstest"

# ---------------------------------------------------------------------------
# Embedding model for FAISS indices
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K = 5