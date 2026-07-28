"""
Loads each of the four target models with the correct class and settings.

Every load should be pinned to a specific revision hash before you run
anything for the dissertation proper — see the "revision" field in
config.MODELS. Loading "latest" is fine while getting the harness working,
but it isn't reproducible, and reproducibility is the whole point of
running significance tests over these results later.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODELS, SEED

try:
    from transformers import Mistral3ForConditionalGeneration
    _HAS_MISTRAL3 = True
except ImportError:
    _HAS_MISTRAL3 = False


def load_tokenizer(model_key: str):
    """
    Loads the tokenizer for a model key, applying any model-specific fixes.

    Shared by the HF path (load_model), the vLLM prompt-length computation
    (scripts/compute_max_model_len.py), and anything else that needs to
    render chat prompts without loading weights — one tokenizer setup,
    not several copies that could drift apart.
    """
    if model_key not in MODELS:
        raise ValueError(f"Unknown model '{model_key}'. Options: {list(MODELS)}")

    cfg = MODELS[model_key]
    tokenizer_kwargs = {}
    if cfg["loader"] == "mistral3":
        # Without this flag, transformers warns that Ministral's tokenizer
        # loads with an incorrect regex pattern and tokenizes incorrectly
        # (observed on the 2026-07-11 A100 debug run; see
        # https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84).
        tokenizer_kwargs["fix_mistral_regex"] = True
    return AutoTokenizer.from_pretrained(
        cfg["hf_id"], revision=cfg["revision"], **tokenizer_kwargs
    )


def load_model(model_key: str, device_map: str = "auto", dtype: torch.dtype = torch.bfloat16):
    """Returns (model, tokenizer) for the given key in config.MODELS."""
    if model_key not in MODELS:
        raise ValueError(f"Unknown model '{model_key}'. Options: {list(MODELS)}")

    cfg = MODELS[model_key]
    hf_id = cfg["hf_id"]
    revision = cfg["revision"]

    if torch.cuda.is_available():
        print(
            f"[model_loader] CUDA available — {torch.cuda.get_device_name(0)}. "
            f"device_map='{device_map}' will place model layers on GPU."
        )
    else:
        print(
            "[model_loader] WARNING: CUDA NOT available — model will run on CPU, "
            "which is impractically slow for 8B-class models. If this machine has "
            "an NVIDIA GPU, you likely installed the CPU-only torch wheel; see "
            "requirements.txt for the CUDA install command."
        )

    if cfg["gated"] and revision is None:
        print(
            f"[model_loader] WARNING: '{model_key}' is a gated model. Make sure you have "
            f"accepted the license on its HuggingFace model page and are logged in "
            f"(`huggingface-cli login`) before this call, or it will fail with a 401."
        )

    if revision is None:
        print(
            f"[model_loader] WARNING: no revision pinned for '{model_key}'. Loading "
            f"'main'. Pin this to a commit hash before running anything you intend "
            f"to report in the dissertation."
        )

    tokenizer = load_tokenizer(model_key)

    if cfg["loader"] == "mistral3":
        if not _HAS_MISTRAL3:
            raise ImportError(
                "Mistral3ForConditionalGeneration is not available in your installed "
                "transformers version. Ministral-3-8B needs a recent enough release — "
                "check the model card on HuggingFace for the minimum version and "
                "upgrade with `pip install -U transformers`."
            )
        # transformers v5 renamed `torch_dtype` to `dtype` (the old kwarg was
        # removed) — requirements.txt pins transformers>=5.13, so use `dtype`.
        model = Mistral3ForConditionalGeneration.from_pretrained(
            hf_id, revision=revision, device_map=device_map, dtype=dtype
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, revision=revision, device_map=device_map, dtype=dtype
        )

    model.eval()
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        # device_map='auto' loading via accelerate can return before every
        # underlying CUDA op it kicked off has actually settled on the
        # device. Without this, sentence-transformers loading the embedder
        # right after (in the same process) intermittently races the tail
        # end of this load and hits cudaErrorDevicesUnavailable -- observed
        # directly on a genuinely fresh RunPod pod (no restart history), so
        # this is a real timing race, not leftover state from a killed
        # process. synchronize() forces this model's pending CUDA work to
        # finish before control passes to anything that loads a second
        # model on the same device, closing the race instead of working
        # around its symptom. No-op when CUDA isn't available.
        torch.cuda.synchronize()
    return model, tokenizer


def build_chat_prompt(model_key: str, tokenizer, system_prompt: str, user_prompt: str) -> str:
    """
    Applies each model's chat template. Qwen3's template accepts an
    enable_thinking kwarg; the others ignore unknown kwargs or don't need
    it, so this stays a single code path rather than a per-model branch.
    """
    from config import QWEN3_ENABLE_THINKING

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if model_key == "qwen3-8b":
        kwargs["enable_thinking"] = QWEN3_ENABLE_THINKING

    return tokenizer.apply_chat_template(messages, **kwargs)
