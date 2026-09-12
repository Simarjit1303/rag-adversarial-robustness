"""
Phase 3 output-stage defense: Llama-Guard-4-12B post-generation filter.

This is the only Phase 3 defense that also needs to work against Crescendo
(see defenses/__init__.py's module docstring), so it is deliberately
attack-agnostic: it operates on the FINAL generated response text alone,
with no knowledge of which pipeline (injection, poisonedrag, crescendo)
produced it, and no access to attack-specific state (injection templates,
poison configs, conversation history). Every attack runner's response-
handling loop is meant to call the same one function from here --
run_output_filter() -- as the final pass after generation.

Deliberately independent of attacks.crescendo.is_refusal(): that matcher
is a naive, voice-blind substring scan already confirmed (see
PHASE2_CRESCENDO_INSIGHTS.md) to false-positive on a fabricated-quote case
(the match landed inside character dialogue the model invented, not its
own refusal) and a copyright hedge-then-comply case (a hedge paired with
immediate compliance via a workaround, not a refusal). This module's job
is to be a clean, independent signal a downstream analysis can compare
AGAINST is_refusal() -- it must not import or call it.
"""

import copy
import json
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

# Not yet registered in config.MODELS -- Llama-Guard-4-12B is a judge, not
# a target model under evaluation (it never generates a RAG answer), so it
# doesn't belong in that dict's "models under study" semantics.
GUARD_MODEL_ID = "meta-llama/Llama-Guard-4-12B"
# Revision pin follows config.py's reproducibility discipline (MODELS/
# CORPORA/BEHAVIOR_DATASETS all pin an exact HF Hub commit). None means
# "not yet pinned" -- same convention as config.BEHAVIOR_DATASETS entries
# before scripts/fetch_corpus_revisions.py records the real sha. Fill in
# before the real Phase 3 sweep, not before this module's unit tests
# (which never touch the network or load real weights).
# Pinned 2026-09-12 via HfApi().model_info(GUARD_MODEL_ID).sha -- a metadata-
# only Hub API call, no weight download -- same method as
# scripts/fetch_corpus_revisions.py uses for config.CORPORA entries.
GUARD_MODEL_REVISION = "87acb4b94e930c3d679e6e7ee9d57e2feab9ea71"

# FOLLOW-UP, not addressed here: requirements.txt:33 pins transformers only
# as a floor (>=5.13, no ceiling), so the Docker build installs whatever's
# newest at build time -- confirmed 2026-09-13 this is why the
# cache_implementation="dynamic_full" fix (commit 31081cf) worked against the
# dev machine's local 5.7.0 but failed on the real pod, which resolved a
# different, newer version where that string doesn't exist. The generate()
# fix below no longer depends on any cache_implementation string, so this
# floating pin no longer threatens THIS bug -- but it's still a real
# reproducibility risk for anything else version-sensitive across the other
# three target models and the injection classifier. Deliberately not pinning
# a ceiling in this commit: that's a separate decision, to be made after this
# fix is confirmed working on the pod, not bundled into a still-unverified fix.

# Fixed, NOT generated -- the point is that downstream scoring can tell
# "the defense fired" apart from "the model refused on its own" by an
# exact string check, without re-running (or inheriting the bug of)
# is_refusal().
REFUSAL_MARKER = "[OUTPUT_FILTER_BLOCKED] This response was withheld by the Llama-Guard output filter."

_SAFE_UNSAFE_RE = re.compile(r"^\s*(safe|unsafe)", re.IGNORECASE)

_guard_cache = {}  # lazy singleton -- {"model": ..., "tokenizer": ...} once loaded


def load_guard_model(device_map="auto", dtype=torch.bfloat16):
    """
    Lazily loads and caches Llama-Guard-4-12B. Kept separate from
    harness/model_loader.py's load_model(): that loader's model_key ->
    config.MODELS contract is for target models under evaluation, and the
    guard is never one of those.
    """
    if "model" not in _guard_cache:
        tokenizer = AutoTokenizer.from_pretrained(GUARD_MODEL_ID, revision=GUARD_MODEL_REVISION)
        model = AutoModelForCausalLM.from_pretrained(
            GUARD_MODEL_ID, revision=GUARD_MODEL_REVISION, device_map=device_map, dtype=dtype,
        )
        model.eval()
        _guard_cache["model"] = model
        _guard_cache["tokenizer"] = tokenizer
    return _guard_cache["model"], _guard_cache["tokenizer"]


def unload_guard_model():
    """Frees the cached guard model -- mirrors the sweeps' own
    `del model; torch.cuda.empty_cache()` pattern between sweep models."""
    _guard_cache.pop("model", None)
    _guard_cache.pop("tokenizer", None)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def classify_response(response_text: str, model=None, tokenizer=None, max_new_tokens: int = 20):
    """
    Runs Llama-Guard-4-12B's own chat template over a single assistant
    turn (the RAG pipeline's final generated response) and parses its
    safe/unsafe verdict.

    model/tokenizer are injectable so callers (and tests) can supply a
    stub instead of loading the real 12B model; when omitted, the lazily-
    cached real guard model is used.

    Returns (flagged: bool, label: str, raw_output: str). label is
    "safe", "unsafe", or "unparseable". An unparseable verdict does NOT
    flag -- fail open on parse failure rather than silently blocking
    responses on a formatting fluke; raw_output is kept in the log either
    way so an unparseable run is auditable, not silently dropped.
    """
    if model is None or tokenizer is None:
        model, tokenizer = load_guard_model()

    # Llama-Guard-4-12B's own chat_template.jinja (fetched 2026-09-12 from the
    # model repo, not guessed) raises "Conversation roles must alternate
    # user/assistant/..." unless messages[0]["role"] == "user" -- a lone
    # {"role": "assistant", ...} turn (the old code here) fails that check on
    # every call, not just this one; the previous unit tests never caught it
    # because their stub tokenizer echoed the conversation back instead of
    # running real Jinja alternation logic. The same template also requires
    # content as a list of typed dicts, not a bare string. This single-turn,
    # role="user" shape matches the model card's own "Getting Started"
    # example (classifying one piece of text with no other conversation
    # turns available to this attack-agnostic filter).
    conversation = [{"role": "user", "content": [{"type": "text", "text": response_text}]}]
    # return_dict=True is explicit here, not left to the default: transformers'
    # apply_chat_template flipped that default from False to True at some
    # point (confirmed 2026-09-12 against a live tokenizer, transformers
    # 5.7.0's PreTrainedTokenizerBase.apply_chat_template source), so
    # tokenize=True + return_tensors="pt" alone now returns a BatchEncoding
    # (dict-like, {"input_ids", "attention_mask"}), not a bare tensor. The
    # old code passed that dict straight into generate() as a single
    # input_ids= kwarg, which crashed on real hardware with AttributeError
    # ('shape') inside generate()'s internals -- the stub tokenizers in
    # tests/test_output_filter.py returned a bare tensor and never
    # exercised this path. Unpacking via **inputs gives generate() both
    # input_ids and attention_mask correctly instead.
    inputs = tokenizer.apply_chat_template(
        conversation, return_tensors="pt", add_generation_prompt=True, return_dict=True,
    )
    inputs = inputs.to(model.device) if hasattr(inputs, "to") else inputs

    # We build the Cache ourselves instead of naming a cache_implementation
    # string, because no string survives a transformers version bump here.
    #
    # Root cause (confirmed 2026-09-12/13, reproduced offline against the
    # real Llama4TextConfig class -- Cache construction is config-only, no
    # weights needed): Llama-Guard-4-12B's published config.json sets
    # text_config.attention_chunk_size=None -- a deliberate choice, not a
    # Meta oversight (this is the Llama4 "Scout" iRoPE long-context design:
    # max_position_embeddings=10,485,760, and transformers' own
    # masking_utils.py already treats a None chunk size as "no chunking
    # bound", falling back to a plain causal mask). But every layer's
    # computed layer_types is "chunked_attention" regardless (driven only by
    # no_rope_layers), and Cache.__init__ never got that same None-means-
    # unbounded fallback: it reads config.layer_types AS-IS when present, and
    # unconditionally maps "chunked_attention" to a *SlidingWindowLayer that
    # requires a real sliding_window value. StaticCache crashes in
    # min(sliding_window, max_cache_len); DynamicCache crashes converting
    # None to a tensor -- both confirmed by direct reproduction, independent
    # of which cache_implementation string selects them. (Also reported,
    # different call site, same root config mismatch, at
    # https://huggingface.co/meta-llama/Llama-Guard-4-12B/discussions/14.)
    #
    # Attempt 1 (cache_implementation="dynamic") and attempt 2
    # (cache_implementation="dynamic_full", commit 31081cf) both relied on
    # one specific string being valid in whatever transformers version is
    # actually installed. "dynamic_full" was valid in the dev machine's
    # local 5.7.0 but crashed on the real pod build with a ValueError
    # listing a completely different accepted set -- requirements.txt:33
    # pins only a floor (transformers>=5.13, no ceiling), so the Docker
    # build installs whatever's newest (PyPI's latest is 5.17.0 as of
    # 2026-09-13), where "dynamic_full" no longer exists at all, and
    # "hybrid"/"hybrid_chunked" are deprecated STATIC-cache aliases that
    # still risk the identical per-layer inference. No cache_implementation
    # value is stable across the versions this floating pin can resolve to.
    #
    # This fix instead patches the actual mechanism transformers reads
    # (config.layer_types, confirmed present and behaving the same way in
    # both the local 5.7.0 and the current main-branch source): copy the
    # guard's real text config, relabel every "chunked_attention" layer as
    # "full_attention" when attention_chunk_size is None (matching
    # masking_utils.py's own semantics for that value), and build a
    # DynamicCache from the patched config ourselves so generate() never
    # infers per-layer types from the broken original. Verified offline
    # against the real Llama4TextConfig: the patched config builds a
    # DynamicCache cleanly where the unpatched one crashes -- not yet
    # verified against the real 12B model's actual generate() call on GPU.
    past_key_values = None
    config = getattr(model, "config", None)
    if config is not None:
        text_config = copy.deepcopy(config.get_text_config(decoder=True))
        layer_types = getattr(text_config, "layer_types", None)
        if getattr(text_config, "attention_chunk_size", None) is None and layer_types:
            text_config.layer_types = [
                "full_attention" if lt == "chunked_attention" else lt for lt in layer_types
            ]
        past_key_values = DynamicCache(config=text_config)

    generate_kwargs = {"max_new_tokens": max_new_tokens, "pad_token_id": tokenizer.eos_token_id}
    if past_key_values is not None:
        generate_kwargs["past_key_values"] = past_key_values
        # Llama-Guard-4-12B's own generation_config.json bakes in
        # cache_implementation="static" (confirmed via its published config).
        # generate()'s _prepare_generation_config() seeds generation_config
        # from that model default BEFORE applying our kwargs, so without this
        # line the model's "static" default survives into the conflict check
        # in _prepare_cache_for_generation() and raises "Passing both
        # cache_implementation ... and past_key_values ... is unsupported" --
        # confirmed on the real pod. Explicitly passing None here is what
        # actually clears it: GenerationConfig.update(**kwargs) applies
        # explicit kwargs unconditionally (unlike the model-default merge
        # step, which only fills in still-None fields).
        generate_kwargs["cache_implementation"] = None

    with torch.no_grad():
        output = model.generate(**inputs, **generate_kwargs)
    generated_ids = output[0][inputs["input_ids"].shape[-1]:]
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    match = _SAFE_UNSAFE_RE.match(raw_output)
    if match is None:
        return False, "unparseable", raw_output
    label = match.group(1).lower()
    return label == "unsafe", label, raw_output


def apply_output_filter(response_text: str, model=None, tokenizer=None):
    """
    Classifies response_text and, if flagged, replaces it with the fixed
    REFUSAL_MARKER (never a generated string -- see module docstring).

    Returns (final_response, record) where record = {"flagged", "guard_label",
    "guard_raw_output"} -- the per-response mechanism-attribution fields
    every caller logs, before merging in its own attack-specific keys.
    """
    flagged, label, raw_output = classify_response(response_text, model=model, tokenizer=tokenizer)
    final_response = REFUSAL_MARKER if flagged else response_text
    record = {"flagged": flagged, "guard_label": label, "guard_raw_output": raw_output}
    return final_response, record


def log_filter_event(log_path, record: dict):
    """
    Appends one JSONL row and fsyncs immediately.

    Same incremental-checkpointing discipline as run_crescendo.py's raw
    JSONL writer (evaluation/run_crescendo.py, "Per-behavior checkpointing,
    NOT _atomic_open"): a buffered/atomic all-or-nothing write only becomes
    durable when the whole `with` block exits normally, so an interruption
    mid-sweep loses every row written so far. Opening in append mode and
    fsyncing after each write means an interruption only costs rows not
    yet logged, matching that fix's guarantee for this module's own log.
    """
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run_output_filter(response_text: str, log_path, row_id: dict, model=None, tokenizer=None):
    """
    THE single integration point: run_attack_injection.py, run_poisonedrag.py,
    and run_crescendo.py all call this identically, as the final pass after
    generation, regardless of which attack pipeline produced response_text.

    row_id is an attack-agnostic dict of whatever identifying metadata the
    caller's own row schema needs (model_key/corpus/question for injection
    and poisonedrag; model_key/behavior/turn_index for Crescendo -- turn
    granularity is what makes the per-turn mechanism-attribution analysis
    possible there) merged into the logged line ahead of the filter's own
    fields, so this module's log can be joined back to the caller's raw
    JSONL by those same keys without this module knowing their shape.

    Returns final_response (str): response_text unchanged, or
    REFUSAL_MARKER if Llama-Guard flagged it.
    """
    final_response, record = apply_output_filter(response_text, model=model, tokenizer=tokenizer)
    log_filter_event(log_path, {**row_id, **record})
    return final_response
