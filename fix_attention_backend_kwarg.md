# Fix: attention backend is a constructor kwarg, not an env var

## What was wrong

`VLLM_ATTENTION_BACKEND` as an environment variable does nothing in this
vLLM version — confirmed via current official docs
(docs.vllm.ai/en/latest/design/attention_backends/). Backend selection is
done via a constructor argument to `LLM()`:

```python
llm = LLM(model="...", attention_backend="FLASH_ATTN")
```

This also explains the earlier `Unknown vLLM environment variable detected:
VLLM_MAX_MODEL_LEN` warning — both custom settings were namespaced with
the `VLLM_` prefix, which vLLM's own env var validator treats as reserved
for its own internal variables. Neither was ever going to work cleanly as
an env var under that name.

## Fix: rename both out of vLLM's namespace, pass attention_backend for real

In `harness/vllm_engine.py`'s `load_vllm_model()`:

```python
import os

def load_vllm_model(model_key: str, max_model_len: int, gpu_memory_utilization: float = 0.9):
    cfg = MODELS[model_key]
    attention_backend = os.environ.get("RAG_VLLM_ATTENTION_BACKEND", "TRITON_ATTN")
    return LLM(
        model=cfg["hf_id"],
        revision=cfg["revision"],
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        attention_backend=attention_backend,
    )
```

Default is `"TRITON_ATTN"` specifically because it's the one backend
already proven working across all four models in this project, during
Stage 1's Colab T4 testing (visible in that session's own logs: `"Using
TRITON_ATTN attention backend..."`). Not a fresh guess — reusing something
already empirically confirmed to work with this exact model set.

**Also rename `VLLM_MAX_MODEL_LEN`** to `RAG_VLLM_MAX_MODEL_LEN` everywhere
it's read (wherever `_run_vllm_sweep` or `load_vllm_model` currently reads
it) — same underlying cause, same fix, do both together rather than leave
one half-fixed. Update `compute_max_model_len.py`'s docstring/output
message to reference the new name too.

## Testing (no GPU needed for the logic)

- Confirm `load_vllm_model()` passes `attention_backend=` correctly to a
  mocked `LLM()` call — assert the exact kwarg and value, both the default
  (`"TRITON_ATTN"`) and an explicit override via
  `RAG_VLLM_ATTENTION_BACKEND`.
- Confirm the renamed `RAG_VLLM_MAX_MODEL_LEN` is read correctly and the
  old `VLLM_MAX_MODEL_LEN` name is no longer referenced anywhere in the
  codebase (grep to confirm, don't just trust the diff).

## What this doesn't guarantee

`"TRITON_ATTN"` being valid on a T4 doesn't strictly guarantee it's
accepted on an A100 with this specific model/config combination — per the
docs found, vLLM validates the requested backend against your actual
hardware and config, and raises a clear, named error if it isn't
compatible, rather than failing silently. If the next real run errors with
something like `"Selected backend TRITON_ATTN is not valid for this
configuration"`, that's useful, actionable information (the error will
literally name why), not a repeat of this same ambiguous failure.
