"""
Batched generation path via vLLM (Stage 2).

Mirrors harness/model_loader.py's shape: load_vllm_model corresponds to
load_model, and generate_batch is the batched counterpart of the
per-question generate inside harness/pipeline.run_query. The HF
transformers path stays untouched and selectable (INFERENCE_ENGINE=hf in
evaluation/run_baseline.py) as a smoke-test / fallback; this module is the
throughput path.

Design note (verified twice on real hardware): vLLM does not clean up
reliably after a FAILED LLM(...) load in the same process — a caught
OutOfMemoryError leaves GPU memory and NCCL state allocated, and a second
LLM(...) call in the same process tends to fail differently (and
confusingly) rather than succeed. Therefore there is deliberately NO
retry-with-catch logic anywhere in this module: a failed load should crash
the process and let the container's restart policy handle it. This matters
for the Stage 3 Container Apps Job design — do not add in-process retries
here later.
"""

from vllm import LLM, SamplingParams

from config import MODELS


def load_vllm_model(model_key: str, max_model_len: int, gpu_memory_utilization: float = 0.9):
    """Returns a vllm.LLM instance for the given key in config.MODELS."""
    if model_key not in MODELS:
        raise ValueError(f"Unknown model '{model_key}'. Options: {list(MODELS)}")

    cfg = MODELS[model_key]
    return LLM(
        model=cfg["hf_id"],
        revision=cfg["revision"],
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        # No quantization= and no dtype= on purpose — vLLM auto-detects both
        # from each checkpoint's config.json:
        #   * ministral-3-8b ships native fp8; vLLM's Marlin kernel path is
        #     what actually preserves the fp8 memory savings. (transformers'
        #     plain from_pretrained dequantizes that checkpoint to bf16 on
        #     any GPU below compute capability 8.9 — true on the A100 too —
        #     so the HF and vLLM paths legitimately show different memory
        #     footprints for this model; expected, not a bug.)
        #   * the other three are plain bf16 checkpoints; vLLM selects bf16
        #     on hardware that supports it (A100, compute capability 8.0).
        #     The fp16 fallback observed on T4 during Colab testing was
        #     hardware-driven and must not be hardcoded here.
    )


def generate_batch(llm, model_key: str, system_prompt: str, user_prompts: list[str],
                   max_new_tokens: int = 256) -> list[str]:
    """
    Renders all prompts via the SAME tokenizer-based template logic the HF
    path uses (harness.model_loader.build_chat_prompt), then sends the
    whole batch to vLLM in one generate() call.

    Rendering the prompt string ourselves instead of calling vLLM's own
    .chat() is deliberate: LLM.chat() has a documented history of not
    reliably forwarding enable_thinking across vLLM releases (multiple
    upstream issues, including one where it wasn't accepted as a kwarg at
    all). Rendering via the tokenizer directly sidesteps that surface and
    keeps Qwen3's QWEN3_ENABLE_THINKING toggle flowing through the
    identical code path on both engines — one implementation, not two that
    could silently drift apart.

    user_prompts are the fully-formed user messages (retrieved context +
    question) — the same strings run_query hands to build_chat_prompt.
    Build them with harness.pipeline.build_rag_user_prompt so the prompt
    surface stays byte-identical across engines.
    """
    from harness.model_loader import build_chat_prompt

    tokenizer = llm.get_tokenizer()
    prompts = [
        build_chat_prompt(model_key, tokenizer, system_prompt, up)
        for up in user_prompts
    ]
    # temperature=0.0 is greedy decoding — matches the HF path's
    # do_sample=False, which cross-engine comparability depends on.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    outputs = llm.generate(prompts, sampling_params)
    return [o.outputs[0].text for o in outputs]
