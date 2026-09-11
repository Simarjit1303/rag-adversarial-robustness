"""
Phase 3, retrieval-stage defense: instruction detection.

Runs an existing open-source prompt-injection classifier against each
retrieved passage BEFORE the passage is rendered into the RAG context
block (harness.pipeline.build_rag_user_prompt /
attacks.indirect_injection.build_attack_user_prompt), and drops any
passage the classifier flags. Unlike a hard refusal, the query still
proceeds on whatever context survives -- partial context loss, not
total denial -- so this measures utility impact of losing a passage,
not just a block/no-block outcome.

Detector: protectai/deberta-v3-base-prompt-injection-v2 (HF Hub)
https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2 --
a DeBERTa-v3-base binary classifier (labels SAFE/INJECTION, confirmed
via the model's config.json) fine-tuned by Protect AI specifically for
prompt-injection detection. Chosen over an LLM-judge-style detector
(e.g. one of this project's own Llama-Guard/Qwen3Guard judges, see
defenses/__init__.py) because it is a purpose-built, single-forward-
pass classifier: ~184M params, runs on CPU in well under a second per
passage, well-established (Apache-2.0, widely downloaded on the Hub) --
cheap enough to run on every retrieved passage in every sweep cell
without needing a GPU, which a generative guard model is not.

Integration (one-line change, not applied by this module): a caller
that currently does
    context = "\n\n".join(f"[{i+1}] {extract_passage_text(corpus_name, doc)}"
                           for i, (doc, score) in enumerate(retrieved))
swaps that line for
    context, detection_log = filter_retrieved_passages(retrieved, corpus_name)
-- same (doc, score) `retrieved` input, same extract_passage_text call,
same "[{i+1}] text" numbering for surviving passages.
"""

from dataclasses import dataclass
from typing import Callable, Optional

from data.normalize import extract_passage_text

_MODEL_NAME = "protectai/deberta-v3-base-prompt-injection-v2"
_FLAG_LABEL = "INJECTION"
_DEFAULT_THRESHOLD = 0.5

_pipeline = None  # lazy singleton -- only loaded if no classifier override is given


def _load_default_classifier():
    """
    Lazily downloads and loads the real HF pipeline. Only ever called
    when detect_injection/filter_retrieved_passages are invoked WITHOUT
    a `classifier` override -- i.e. during a real sweep on the RunPod
    GPU pod, never from this repo's test suite (see
    tests/test_instruction_detection.py's module docstring: the dev
    machine this repo runs on is not meant to pull model weights).
    """
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        # device=-1: CPU-only. No GPU code path exists in this module by
        # design -- the model is small enough (~184M params) that a
        # per-passage classifier call is cheap on CPU even on the pod,
        # so this defense never competes with the target model for GPU
        # memory during a sweep.
        _pipeline = pipeline(
            "text-classification", model=_MODEL_NAME, tokenizer=_MODEL_NAME, device=-1,
        )
    return _pipeline


@dataclass
class DetectionResult:
    flagged: bool
    score: float
    label: str


def detect_injection(
    text: str,
    classifier: Optional[Callable[..., list]] = None,
    threshold: float = _DEFAULT_THRESHOLD,
) -> DetectionResult:
    """
    Runs the classifier on a single passage of text.

    `classifier` is the swap-in point for tests (and for a different
    detector later): any callable matching a HF text-classification
    pipeline's contract -- callable on a string, returning
    [{"label": ..., "score": float}]. Defaults to the lazily-loaded
    protectai DeBERTa pipeline described in this module's docstring.
    """
    clf = classifier or _load_default_classifier()
    result = clf(text, truncation=True)[0]
    label = result["label"]
    score = float(result["score"])
    flagged = label.upper() == _FLAG_LABEL and score >= threshold
    return DetectionResult(flagged=flagged, score=score, label=label)


@dataclass
class PassageLog:
    """One entry per ORIGINAL retrieved passage (dropped ones included),
    in original rank order -- needed for mechanism attribution and
    false-positive analysis against XSTest / OR-Bench-Hard-1K."""

    passage_id: int
    flagged: bool
    score: float
    label: str


def filter_retrieved_passages(retrieved, corpus_name: str, classifier=None):
    """
    Mirrors harness.pipeline.build_rag_user_prompt's context-rendering
    loop (same extract_passage_text call, same "[{i+1}] text" numbering
    for survivors), but runs each passage through detect_injection first
    and drops any flagged passage from the rendered context entirely --
    the query still runs on whatever passages survive (see module
    docstring: no hard refusal).

    `retrieved` is the (doc, score) list data.build_index.retrieve
    returns -- the exact shape build_rag_user_prompt /
    build_attack_user_prompt already receive, so integration is a
    one-line swap at the call site (see module docstring).

    Returns (context_str, log): log is a list[PassageLog], one per
    original passage in original rank order (dropped passages included,
    flagged=True, absent from context_str).
    """
    lines = []
    log = []
    for i, (doc, score) in enumerate(retrieved):
        text = extract_passage_text(corpus_name, doc)
        result = detect_injection(text, classifier=classifier)
        log.append(PassageLog(passage_id=i, flagged=result.flagged, score=result.score, label=result.label))
        if result.flagged:
            continue
        lines.append(f"[{i + 1}] {text}")
    context = "\n\n".join(lines)
    return context, log
