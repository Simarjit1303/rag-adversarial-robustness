# Stage 2 — wire vLLM into the pipeline

Adds a batched vLLM generation path alongside the existing HF `transformers`
path. The HF path stays as-is, selectable via an env flag — useful as a
smoke-test / fallback, not something to delete.

Work on a new branch off `phase1-metric-fixes` (PR #6 is still open/unmerged
per the standing safety note — don't merge either branch to main; that still
triggers the GPU deploy workflow). Call this one `stage2-vllm-path`.

---

## Context this brief assumes (verified this session, on real weights)

All four models confirmed to load and generate correctly under vLLM 0.25.1:

| Model | Resolved architecture | Quantization needed? |
|---|---|---|
| `ministral-3-8b` | `Mistral3ForConditionalGeneration` (internally routes through a `PixtralVisionModel` vision tower — expected, confirmed benign, checkpoint declares `Mistral3ForConditionalGeneration` in its own config) | **None** — checkpoint ships native fp8, vLLM auto-detects it from `config.json`. Do not pass a `quantization=` kwarg for this model; let vLLM read it. |
| `qwen3-8b` | `Qwen3ForCausalLM` | None available — plain bf16 checkpoint, no native quantized variant at this HF ID. |
| `phi-4-mini` | `Phi3ForCausalLM` | None available. |
| `llama-3.1-8b` | `LlamaForCausalLM` | None available. Gated — needs `HF_TOKEN` env var and the account must have accepted Meta's license on the model page; this is a human prerequisite, not something the code can satisfy. |

None of these should have a `dtype=` override passed either — vLLM auto-selects
bf16 when the GPU supports it (A100 does, compute capability 8.0) and only
falls back to fp16 on hardware that can't do bf16 (observed on T4, compute
capability 7.5, during Colab testing — irrelevant to the A100 target, don't
carry that fallback logic over).

**Known finding, worth a code comment where relevant:** `transformers`'
plain `from_pretrained` path dequantizes Ministral's native fp8 checkpoint to
bf16 on any GPU with compute capability below 8.9 — true on the A100 too.
vLLM's Marlin kernel path is what actually preserves the fp8 memory savings.
This means the HF and vLLM paths will show meaningfully different memory
footprints for Ministral specifically; that's expected, not a bug, and worth
noting in the methods section if memory usage is ever compared across paths.

---

## Task 1 — `harness/vllm_engine.py` (new file)

Two functions, mirroring `model_loader.py`'s existing shape rather than
inventing a new interface:

```python
from vllm import LLM, SamplingParams
from config import MODELS

def load_vllm_model(model_key: str, max_model_len: int, gpu_memory_utilization: float = 0.9):
    """Returns an vllm.LLM instance for the given key in config.MODELS."""
    cfg = MODELS[model_key]
    return LLM(
        model=cfg["hf_id"],
        revision=cfg["revision"],
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        # no quantization=, no dtype= -- let vLLM auto-detect per model, see
        # table above for why
    )


def generate_batch(llm, model_key: str, system_prompt: str, questions: list[str],
                    max_new_tokens: int = 256) -> list[str]:
    """
    Renders all prompts via the SAME tokenizer-based template logic the HF
    path uses (harness.model_loader.build_chat_prompt), then sends the
    whole batch to vLLM in one generate() call. This is deliberate --
    vLLM's own .chat() + enable_thinking kwarg has been unreliable across
    vLLM versions (see notes below); rendering via the tokenizer directly
    sidesteps that entirely and is also more consistent with the HF path's
    behavior, which matters for cross-engine comparability.
    """
    from harness.model_loader import build_chat_prompt

    tokenizer = llm.get_tokenizer()
    prompts = [
        build_chat_prompt(model_key, tokenizer, system_prompt, q)
        for q in questions
    ]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    outputs = llm.generate(prompts, sampling_params)
    return [o.outputs[0].text for o in outputs]
```

**Why `build_chat_prompt` gets reused rather than vLLM's own `.chat()`:**
confirmed this session that `LLM.chat()` has a documented history of not
reliably forwarding `enable_thinking` across vLLM releases (multiple upstream
GitHub issues, including one where it wasn't accepted as a kwarg at all in
an older version). Rendering the prompt string ourselves via the same
tokenizer call the HF path already uses avoids depending on that vLLM
surface being correct in whatever version ends up pinned. It also means
Qwen3's `QWEN3_ENABLE_THINKING` toggle flows through identically regardless
of which engine is running — same code path, not two parallel
implementations that could silently drift apart.

**`enable_thinking=False` is confirmed to correctly suppress `<think>` tags
under vLLM** — verified this session on `Qwen/Qwen3-0.6B` as a same-family
proxy (the mechanism is chat-template-level, not weight-specific, so this
should hold for `qwen3-8b` too, but flag in the write-up that this was
verified on the proxy, not the 8B checkpoint directly — free-tier hardware
couldn't fit Qwen3-8B at full precision to test it there).

---

## Task 2 — engine selection in `run_baseline.py`

Add an `INFERENCE_ENGINE` env var, default `"hf"` (preserves current
behavior with zero config changes needed):

```python
import os
ENGINE = os.environ.get("INFERENCE_ENGINE", "hf")  # "hf" or "vllm"
```

**The vLLM branch needs a genuinely different control flow, not just a
different loader** — this is the actual point of Stage 2, so don't let it
collapse into "swap the loader, keep the per-question loop":

- HF path (existing, unchanged): loop over questions, one `model.generate()`
  call per question.
- vLLM path (new): for each `(model, corpus)` pair, collect **all** questions
  for that corpus into a single list, call `generate_batch()` once, then
  iterate over the returned list to compute metrics and write JSONL rows.
  One `llm.generate()` call per corpus per model, not one per question —
  that's where the throughput gain actually comes from.

`clean_generation()` and the metric hierarchy from PR #6 (`f1_clean`
primary, `f1_raw`/`exact_match` secondary, `contains_answer` diagnostic)
apply identically regardless of which list of raw strings they're handed —
they only operate on decoded text, not on anything engine-specific. Confirm
this holds rather than assume it: run `verify_cleanup.py`'s existing 40
samples through `clean_generation()` a second time as a literal string list
(bypassing generation entirely) and confirm identical output to the
original test. This should be a no-op change, but confirming costs nothing
and the whole discipline this project has followed is "verify, don't
assume."

---

## Task 3 — `config.py` additions

- Pin the vLLM version explicitly in `requirements.txt`
  (`vllm==0.25.1`, or whatever is currently installed and validated —
  check `pip show vllm` before writing the pin). Unpinned `pip install vllm`
  has already caused real problems once this session (pulled in
  `cuda-toolkit==13.0.2` alongside an existing `torch==...+cu128` install,
  producing three separate `libcudart.so.13`/`libnvrtc.so.13`/
  `libnvJitLink.so.13` failures before anything even got to model-loading
  code). Pin it the same way model revisions are already pinned, and for
  the same reason — reproducibility, not just convenience.

- Add a `max_model_len` decision for the vLLM path, but compute it, don't
  guess it. `SYSTEM_PROMPT + 5 retrieved docs + question` is the actual
  prompt shape — tokenize a representative sample of real built prompts
  (from `data/build_index.py`'s output, across all three corpora since
  MS MARCO/HotpotQA/nq_open chunk sizes likely differ) and set
  `max_model_len` to comfortably cover the longest observed prompt plus
  the `256`-token generation budget, with headroom. The `4096` value used
  during Colab testing was picked for T4 VRAM reasons, not because it
  reflects the real prompt length — don't carry it over without checking.

- Note near `MODELS`: gated model access (`llama-3.1-8b`) requires `HF_TOKEN`
  in the environment **and** license acceptance on the HF model page — the
  Container App's existing `HF_TOKEN`/`HUGGINGFACE_HUB_TOKEN` secret wiring
  (from the original deploy setup) should already satisfy the token half;
  confirm the license-acceptance half was actually done for the account tied
  to that token, since that can't be checked from code.

---

## Task 4 — Dockerfile / requirements.txt check (important, do not skip)

The CUDA 12/13 wheel mismatch that caused three separate failures in Colab
came from `pip install vllm` pulling in inconsistent CUDA-version
dependencies alongside an existing pinned torch install. **Check whether the
same risk exists in the actual Docker build** — if `requirements.txt`
installs `vllm` without a matching explicit `torch`/CUDA pin, the container
build could hit the identical problem, except inside a paid Azure build
rather than a free Colab session. Pin `vllm` and `torch` together, from a
consistent CUDA wheel index, and note in a comment why (link back to this
brief or the finding itself) so a future version bump doesn't reintroduce it
silently.

**Do not port the three Colab-specific patches into the Docker image:**
`LD_LIBRARY_PATH` glob-and-set, `VLLM_ENABLE_V1_MULTIPROCESSING=0`, and the
`sys.stdout` fileno wrapper. All three exist because Jupyter/Colab fakes
`sys.stdout` and has a nonstandard multiprocessing environment — a plain
Docker container running a Python script doesn't have either problem. If
any of these three symptoms show up in the real container logs, that's a
signal something is different from what testing covered, not a cue to
reflexively reapply the same fix.

---

## Task 5 — failure handling note (design consideration, not code to write yet)

Confirmed twice this session: vLLM does not clean up reliably after a failed
`LLM(...)` load within the same process — a caught `OutOfMemoryError`
doesn't release the GPU memory or NCCL state vLLM had already allocated,
and a second `LLM(...)` call in the same process tends to fail differently
(and confusingly) rather than succeed. **This has a real implication for
the Container Apps Job design (Stage 3, not yet done):** a failed model
load inside the Job should be allowed to crash the container and let the
Job's restart policy handle it, rather than catching the exception and
retrying in-process. Don't implement retry-with-catch logic anywhere in the
vLLM path for this reason — let failures propagate.

---

## Verification before calling this done

1. `verify_cleanup.py`'s 40 known samples, run through `clean_generation()`
   directly as a string list (no generation involved) — confirm identical
   output to the original PR #6 result. This isolates "did the metric
   hierarchy change" from "did the new engine work."
2. Small end-to-end smoke test: one model (start with `phi-4-mini`, no
   gating, no quantization complexity), 5-10 real questions from one
   corpus, `INFERENCE_ENGINE=vllm`, confirm output is sane and
   `f1_clean`/`exact_match`/`contains_answer` all populate correctly in the
   JSONL rows.
3. Confirm `INFERENCE_ENGINE=hf` (or unset) still reproduces the exact
   behavior from before this branch — this branch should be additive, not
   a regression on the path that already works.

Same standing rule as PR #6: open the PR, don't merge to main.
