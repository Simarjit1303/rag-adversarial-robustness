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


def _disable_llama4_chunked_attention(model_config):
    """
    Mutates the guard's OWN live text config IN PLACE, once, so every layer
    of Llama4TextModel.forward() runs as "full_attention" instead of the
    "chunked_attention" it's labeled by default.

    This is the 6th real-pod crash traced to the same root config mismatch
    (Llama-Guard-4-12B's config.json sets attention_chunk_size=None -- a
    deliberate Llama4 "Scout" iRoPE choice, not an oversight -- while
    layer_types still computes "chunked_attention" for every layer,
    independent of attention_chunk_size). Confirmed by reading the
    installed transformers source directly (modeling_llama4.py,
    masking_utils.py -- no GPU needed, config/mask construction is pure
    Python):

    1. Llama4TextModel.forward() reads `self.config.layer_types[i]` per
       layer (modeling_llama4.py) -- `self.config` is the SAME object
       passed into every submodule's __init__ (`self.config = config` in
       PreTrainedModel.__init__, a reference, never copied), so patching
       it once, right after load, fixes every subsequent forward() call
       against this loaded model instance. No per-call patching needed.

    2. This supersedes 9c610e0's fix, which only patched a COPY of the
       config used to build classify_response()'s own DynamicCache --
       that copy was never seen by the model's own forward() pass, which
       independently reads model.config.layer_types on every call. That
       fix was necessary (for the Cache constructor crash) but not
       sufficient (this crash).

    3. Relabeling layer_types alone is STILL not sufficient by itself:
       modeling_llama4.py's forward() builds BOTH the "full_attention" and
       "chunked_attention" masks unconditionally on every call, via
       `create_chunked_causal_mask(**mask_kwargs)`
       (modeling_llama4.py:563) -- regardless of whether any layer's type
       actually equals "chunked_attention". That function unconditionally
       raises ValueError("Could not find an `attention_chunk_size`
       argument...") whenever config.attention_chunk_size is None
       (masking_utils.py:1343-1345), BEFORE layer_types is ever consulted
       for mask selection (modeling_llama4.py:574). So attention_chunk_size
       must also be set to a real int, or every single forward() call
       crashes here regardless of layer_types. Reproduced and confirmed
       fixed offline (2026-09-13) by calling
       transformers.masking_utils.create_chunked_causal_mask directly
       against a Llama4TextConfig shaped like the real guard's -- raises
       with attention_chunk_size=None, does not raise once set.

    Since every layer is relabeled away from "chunked_attention" here, the
    "chunked_attention" mask create_chunked_causal_mask still eagerly
    builds is provably never indexed into (modeling_llama4.py:574 only
    ever looks up "full_attention" after this patch) -- so the actual
    attention_chunk_size value set below is a dummy that only needs to be a
    valid int, not a value with real semantic meaning. max_position_embeddings
    is used because it matches masking_utils.py's own "None means unbounded"
    reading of this field elsewhere.

    Not yet verified against the real 12B model's actual generate() call on
    GPU -- verified here by reading the exact source lines that raised on
    the pod and confirming this patch clears the exact condition each one
    checks.
    """
    text_config = model_config.get_text_config(decoder=True)
    layer_types = getattr(text_config, "layer_types", None)
    if getattr(text_config, "attention_chunk_size", None) is None and layer_types:
        text_config.layer_types = [
            "full_attention" if lt == "chunked_attention" else lt for lt in layer_types
        ]
        text_config.attention_chunk_size = text_config.max_position_embeddings


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
        # Once per loaded model instance, not per classify_response() call --
        # see _disable_llama4_chunked_attention's docstring.
        _disable_llama4_chunked_attention(model.config)
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
    # string, because no string survives a transformers version bump here
    # (see attempts 1/2, cache_implementation="dynamic" then "dynamic_full"
    # in commit 31081cf -- both broke on transformers version differences).
    #
    # Root cause: Llama-Guard-4-12B's published config.json sets
    # attention_chunk_size=None (deliberate Llama4 "Scout" iRoPE design) but
    # every layer's computed layer_types is "chunked_attention" regardless.
    # transformers' Cache.__init__ reads config.layer_types AS-IS and maps
    # "chunked_attention" to a *SlidingWindowLayer needing a real
    # sliding_window value -- crashes on the None. (Also independently
    # reported at
    # https://huggingface.co/meta-llama/Llama-Guard-4-12B/discussions/14.)
    #
    # load_guard_model() now relabels model.config's layer_types (and sets
    # a dummy attention_chunk_size) to "full_attention" ONCE, right after
    # load -- see _disable_llama4_chunked_attention's docstring for why that
    # has to happen on the model's own live config (not a copy here) and why
    # a copy-and-patch step here would be redundant: config.get_text_config()
    # returns that same already-patched object for a real loaded model, so
    # DynamicCache(config=...) picks up the fix automatically. Test stubs
    # that inject their own model/config must pre-patch it the same way
    # load_guard_model() does, to match real production behavior.
    past_key_values = None
    config = getattr(model, "config", None)
    if config is not None:
        past_key_values = DynamicCache(config=config.get_text_config(decoder=True))

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
