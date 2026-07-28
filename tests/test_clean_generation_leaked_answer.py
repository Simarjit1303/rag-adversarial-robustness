"""
The model occasionally leaks a Python dict/list-repr structure instead of
answering directly, echoing something that looks like a retrieved-document
or few-shot example format. Found comparing real baseline_raw_hf.jsonl vs
baseline_raw_vllm.jsonl (phi-4-mini x nq_open, n=1000): ~2.1% of vLLM
outputs, ~1.0% of HF outputs -- not engine-specific. The correct answer is
almost always already sitting inside this structure; clean_generation()
just didn't know how to extract it.

Cases below are exact strings from the actual leaked data.
"""

import pytest

from harness.pipeline import clean_generation

CASES = [
    (
        "[1] {'question': 'what season is pepper in american horror story', 'answer': ['the second and fourth seasons']}",
        "the second and fourth seasons",
    ),
    (
        "[1] {'question': 'when did jackie robinson became rookie of the year', 'answer': ['in 1947']}",
        "in 1947",
    ),
    (
        "[1] {'question': 'who has the best nba record in history', 'answer': ['.890']}",
        ".890",
    ),
    (
        "[1] {'question': 'when were the extra books of the catholic bible added', 'answer': ['393']}",
        "393",
    ),
    (
        "[1] {'question': 'thicknet is the colloquial name for which ethernet standard', 'answer': ['10BASE5']}",
        "10BASE5",
    ),
    (
        "[1] {'question': 'when does the lion witch and the wardrobe take place', 'answer': ['1940']}",
        "1940",
    ),
    ("['.890']", ".890"),  # simpler bare-list variant, no dict wrapper
    ("[1]", "[1]"),  # degenerate case, no 'answer' key -- must NOT crash,
                      # falls through unchanged since there's nothing to extract
]


@pytest.mark.parametrize("raw, expected", CASES)
def test_clean_generation_extracts_leaked_answer(raw, expected):
    assert clean_generation(raw) == expected


# --------------------------------------------------------------------------
# Negative tests: legitimate bracket/brace usage in a normal answer must
# NOT be mistaken for the leaked-answer shape.
# --------------------------------------------------------------------------

def test_does_not_fire_on_answer_containing_brackets_for_legitimate_reasons():
    assert clean_generation("the [REDACTED] files") == "the [REDACTED] files"


def test_does_not_fire_on_answer_starting_with_bracket_but_not_a_string_list():
    # Starts with '[' but the element isn't a quoted string -- must not be
    # misread as a leaked answer (mirrors the "[1]" degenerate case).
    assert clean_generation("[42]") == "[42]"


def test_does_not_fire_on_curly_braces_without_the_answer_key_shape():
    assert clean_generation("the set is {1, 2, 3}") == "the set is {1, 2, 3}"


def test_extracted_answer_still_goes_through_normal_cleanup():
    # A leaked dict wrapping a preamble-style answer must still get the
    # normal trailing-period stripping applied afterward -- extraction is
    # the starting point for cleanup, not a bypass of it.
    raw = "[1] {'question': 'x', 'answer': ['Paris.']}"
    assert clean_generation(raw) == "Paris"
