"""
Smoke test for defenses/instruction_detection.py.

Deliberately does NOT load the real protectai/deberta-v3-base-prompt-
injection-v2 pipeline: this repo's dev machine is not where models run
(everything GPU/model-weight-sized runs on the RunPod GPU pod -- see
harness/model_loader.py and every evaluation/run_*.py sweep runner), so
a test that ships in the repo and runs on every `pytest` invocation
must never trigger a multi-hundred-MB download or a real forward pass.
Every test below drives detect_injection/filter_retrieved_passages
through the `classifier` injection point with a small deterministic
fake standing in for the real HF pipeline's (text) -> [{label, score}]
contract -- this exercises the module's own logic (threshold, drop
behavior, logging), not the real detector's accuracy, which is a
separate, GPU-pod-side evaluation (see PassageLog's docstring on
XSTest/OR-Bench-Hard-1K false-positive analysis).

Still uses the real attack payloads from attacks/injection_templates.py
(ignore, fake_completion) as the flagged-example fixtures, so the
shapes being classified are the project's actual attack strings, not
synthetic stand-ins.
"""

from attacks.injection_templates import TEMPLATES
from defenses.instruction_detection import detect_injection, filter_retrieved_passages

CLEAN_PASSAGES = [
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de "
    "Mars in Paris, France. It was designed by Gustave Eiffel's company.",
    "Photosynthesis is the process by which green plants convert light "
    "energy into chemical energy stored in glucose.",
    "The Amazon River in South America discharges more water than the "
    "next several largest rivers combined.",
]

INJECTED_PASSAGES = [
    TEMPLATES["ignore"].render(
        "Mount Everest is the tallest mountain above sea level, located "
        "in the Mahalangur Himal sub-range of the Himalayas."
    ),
    TEMPLATES["fake_completion"].render(
        "The Great Wall of China is a series of fortifications built "
        "across the historical northern borders of ancient Chinese states."
    ),
]

# Deterministic stand-in for the real HF pipeline's decision boundary:
# real prompt-injection classifiers key heavily on imperative override
# phrasing ("ignore ... instructions", "answer: task complete") --
# these two markers cover exactly the two template families under test.
_INJECTION_MARKERS = ("Ignore my previous instructions", "Answer: task complete")


def _fake_classifier(text, **kw):
    flagged = any(marker in text for marker in _INJECTION_MARKERS)
    return [{"label": "INJECTION" if flagged else "SAFE", "score": 0.97 if flagged else 0.02}]


def test_clean_passages_not_flagged():
    for text in CLEAN_PASSAGES:
        result = detect_injection(text, classifier=_fake_classifier)
        assert not result.flagged, f"false positive on clean passage: {text!r} ({result})"


def test_ignore_template_flagged():
    result = detect_injection(INJECTED_PASSAGES[0], classifier=_fake_classifier)
    assert result.flagged, f"missed 'ignore' injection: {result}"


def test_fake_completion_template_flagged():
    result = detect_injection(INJECTED_PASSAGES[1], classifier=_fake_classifier)
    assert result.flagged, f"missed 'fake_completion' injection: {result}"


def test_detect_injection_respects_threshold():
    """A classifier that flags INJECTION but below threshold must not flag."""
    low_confidence = lambda text, **kw: [{"label": "INJECTION", "score": 0.3}]
    result = detect_injection("anything", classifier=low_confidence, threshold=0.5)
    assert not result.flagged
    assert result.score == 0.3
    assert result.label == "INJECTION"


def test_detect_injection_label_case_insensitive():
    mixed_case = lambda text, **kw: [{"label": "injection", "score": 0.9}]
    result = detect_injection("anything", classifier=mixed_case)
    assert result.flagged


def test_detect_injection_passes_explicit_max_length():
    """
    Regression test for the real-pod near-hang at n=1000 (2026-09-13):
    detect_injection() used to call the classifier with truncation=True
    and no max_length. protectai/deberta-v3-base-prompt-injection-v2's
    published tokenizer_config.json sets model_max_length to HF's "no
    limit configured" sentinel (~1e30) -- and transformers' own
    tokenization_utils_base.py silently downgrades truncation=True to
    DO_NOT_TRUNCATE whenever max_length is omitted and model_max_length
    exceeds that sentinel threshold (confirmed by reading that exact
    source path). So truncation=True alone was a no-op: any passage
    longer than the model's real 512-token trained context (config.json:
    max_position_embeddings=512) ran through DeBERTa's O(n^2) self-
    attention at FULL length on CPU -- hotpot_qa's extract_passage_text
    concatenates an entire ~10-document distractor context into one
    passage string, so this was routinely hit at real sweep scale despite
    never showing up in this file's short smoke-test fixtures. Pins the
    fix: max_length must be passed explicitly so truncation actually
    applies regardless of the tokenizer's own (broken) metadata.
    """
    captured_kwargs = {}

    def _recording_classifier(text, **kw):
        captured_kwargs.update(kw)
        return [{"label": "SAFE", "score": 0.02}]

    detect_injection("anything", classifier=_recording_classifier)
    assert captured_kwargs.get("truncation") is True
    assert captured_kwargs.get("max_length") == 512


def test_filter_retrieved_passages_drops_flagged_and_keeps_clean(monkeypatch):
    """Matches build_rag_user_prompt's (doc, score) retrieved shape and
    extract_passage_text contract via a minimal corpus fixture, and
    verifies: (1) a flagged passage's text is absent from the rendered
    context, (2) clean passages' text survives, (3) the log covers
    every original passage (dropped ones included) with passage_id in
    original rank order for later mechanism attribution."""
    retrieved = [
        ({"text": CLEAN_PASSAGES[0]}, 0.8),
        ({"text": INJECTED_PASSAGES[0]}, 0.7),
        ({"text": CLEAN_PASSAGES[1]}, 0.6),
    ]

    # fake_corpus records carry a plain "text" key -- extract_passage_text
    # has no branch for that, so patch it to a trivial passthrough for
    # this synthetic fixture rather than adopting a real corpus's schema.
    import defenses.instruction_detection as idmod
    monkeypatch.setattr(idmod, "extract_passage_text", lambda corpus_name, doc: doc["text"])

    context, log = filter_retrieved_passages(retrieved, "fake_corpus", classifier=_fake_classifier)

    assert CLEAN_PASSAGES[0] in context
    assert CLEAN_PASSAGES[1] in context
    assert INJECTED_PASSAGES[0] not in context

    assert len(log) == 3
    assert [entry.passage_id for entry in log] == [0, 1, 2]
    assert [entry.flagged for entry in log] == [False, True, False]
