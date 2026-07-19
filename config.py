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
# "revision" pins each model to the exact HuggingFace commit that was
# current on 2026-07-11 (the hash from each repo's "Files and versions"
# tab / API `sha` field). These hashes belong in the reproducibility
# appendix. Do NOT bump them mid-study — every phase must load byte-identical
# weights, or the Phase 4 significance tests compare different models.
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
        "revision": "0e9e39f249a16976918f6564b8830bc894c89659",  # main @ 2026-07-11
        "loader": "causal_lm",
        "gated": True,  # requires accepting Meta's license on the model page first
        "license": "Llama 3.1 Community License",
        "note": "Retained: no dense 7-8B Llama 4 model exists (Llama 4 went straight to MoE).",
    },
    "qwen3-8b": {
        "hf_id": "Qwen/Qwen3-8B",
        "revision": "b968826d9c46dd6066d109eabc6255188de91218",  # main @ 2026-07-11
        "loader": "causal_lm",
        "gated": False,
        "license": "Apache-2.0",
        "note": "Replaces Qwen2.5-7B-Instruct. Supports a thinking-mode toggle, see QWEN3_ENABLE_THINKING below.",
    },
    "phi-4-mini": {
        "hf_id": "microsoft/Phi-4-mini-instruct",
        "revision": "cfbefacb99257ffa30c83adab238a50856ac3083",  # main @ 2026-07-11
        "loader": "causal_lm",
        "gated": False,
        "license": "MIT",
        "note": "Replaces Phi-3.5-mini-instruct.",
    },
    "ministral-3-8b": {
        "hf_id": "mistralai/Ministral-3-8B-Instruct-2512",
        "revision": "aae06a2125402f2a89efbacf0881623c15a711d0",  # main @ 2026-07-11
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
#
# "revision" pins each dataset repo to the exact HF commit that was current
# on 2026-07-19 — fetched with scripts/fetch_corpus_revisions.py, the same
# "API `sha` field" method as the MODELS pins above. The sampling seed alone
# only fixes WHICH rows are drawn; it cannot guarantee the pool being drawn
# from if the upstream dataset is ever re-uploaded. Do NOT bump these
# mid-study — every phase must sample from a byte-identical snapshot, or the
# Phase 4 significance tests compare different corpora. Phase 3's
# adversarial/utility datasets (JBB-Behaviors, HarmBench, XSTest) get the
# identical treatment when that work begins — not before.
# ---------------------------------------------------------------------------
CORPORA = {
    "nq_open": {
        "hf_id": "google-research-datasets/nq_open",
        "revision": "5dd9790a83002ad084ddeb7c420dc716852c6f28",  # main @ 2026-07-19
        "dev_n": 1000,
        "eval_n": 10000,
    },
    "hotpot_qa": {
        "hf_id": "hotpotqa/hotpot_qa",
        "hf_config": "distractor",
        "revision": "1908d6afbbead072334abe2965f91bd2709910ab",  # main @ 2026-07-19
        "dev_n": 1000,
        "eval_n": 10000,
    },
    "ms_marco": {
        "hf_id": "microsoft/ms_marco",
        "hf_config": "v2.1",
        "revision": "a47ee7aae8d7d466ba15f9f0bfac3b3681087b3a",  # main @ 2026-07-19
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

# ---------------------------------------------------------------------------
# vLLM engine (Stage 2)
#
# Gated model access (llama-3.1-8b) needs BOTH halves, on either engine:
#   (1) an HF_TOKEN / HUGGINGFACE_HUB_TOKEN in the environment — the
#       Container App's existing secret wiring from the original deploy
#       setup should already cover this half; and
#   (2) Meta's license accepted on the model page BY THE ACCOUNT THAT TOKEN
#       BELONGS TO. This half is a human prerequisite that cannot be
#       checked or satisfied from code — confirm it once for the deploy
#       token's account before the first run that includes Llama.
# ---------------------------------------------------------------------------
# Maximum sequence length (prompt + generation budget) for the vLLM path.
# COMPUTED, NOT GUESSED: the real prompt shape is SYSTEM_PROMPT + 5
# retrieved docs + question, and chunk sizes differ across the three
# corpora — run scripts/compute_max_model_len.py against the real built
# indices and put its recommendation here. The 4096 used during Colab
# testing was a T4-VRAM compromise, not a measured value; do not copy it.
#
# None means "not computed yet": evaluation/run_baseline.py refuses to
# start a vLLM sweep until this is set (or the VLLM_MAX_MODEL_LEN env var
# is exported, which takes precedence for machine-specific overrides).
VLLM_MAX_MODEL_LEN = None