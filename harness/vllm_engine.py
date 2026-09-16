import os
from vllm import LLM, SamplingParams
from config import MODELS

def load_vllm_model(model_key: str, max_model_len: int, gpu_memory_utilization: float=0.9):
    if model_key not in MODELS:
        raise ValueError(f"Unknown model '{model_key}'. Options: {list(MODELS)}")
    cfg = MODELS[model_key]
    attention_backend = os.environ.get('RAG_VLLM_ATTENTION_BACKEND', 'TRITON_ATTN')
    return LLM(model=cfg['hf_id'], revision=cfg['revision'], trust_remote_code=True, max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization, attention_backend=attention_backend)

def generate_batch(llm, model_key: str, system_prompt: str, user_prompts: list[str], max_new_tokens: int=256) -> list[str]:
    from harness.model_loader import build_chat_prompt
    tokenizer = llm.get_tokenizer()
    prompts = [build_chat_prompt(model_key, tokenizer, system_prompt, up) for up in user_prompts]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    outputs = llm.generate(prompts, sampling_params)
    return [o.outputs[0].text for o in outputs]
