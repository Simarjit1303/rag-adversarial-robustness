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
from transformers import AutoModelForCausalLM, AutoTokenizer

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
GUARD_MODEL_REVISION = None

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

    conversation = [{"role": "assistant", "content": response_text}]
    input_ids = tokenizer.apply_chat_template(
        conversation, return_tensors="pt", add_generation_prompt=True,
    )
    input_ids = input_ids.to(model.device) if hasattr(input_ids, "to") else input_ids

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output[0][input_ids.shape[-1]:]
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
