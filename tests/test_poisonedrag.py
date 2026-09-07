"""
Covers attacks/poisonedrag.py's non-network logic: deterministic target-
question sampling, the labeled-markdown response parser, and the exact
retrieval-piece + generation-piece concatenation. generate_poison_texts()
itself (the live API call) isn't unit-tested here -- no network in tests.

The parser tests load 4 REAL Kimi-K2-Instruct-0905 responses (thinking
disabled, no response_format -- see attacks/poisonedrag.py's module
docstring) from tests/fixtures/poison_responses/, captured live during
this feature's development, not written by hand. They already show real
format variation the parser has to survive: "Incorrect Answer" vs
"Incorrect answer" capitalization, and "---" separators present between
every section, absent entirely, or present only once within the same
response.
"""

from pathlib import Path

import pytest

from attacks.poisonedrag import (
    ADV_PER_QUERY,
    _parse_poison_response,
    build_poisoned_passage,
    sample_target_questions,
)

_FIXTURES_DIR = Path(__file__).parent / "fixtures" / "poison_responses"


def _load_fixture(name: str) -> str:
    return (_FIXTURES_DIR / name).read_text(encoding="utf-8")


def test_sample_target_questions_is_deterministic_given_same_seed():
    records = [{"id": i} for i in range(1000)]
    a = sample_target_questions(records, "hotpot_qa", sample_size=100, seed=42)
    b = sample_target_questions(records, "hotpot_qa", sample_size=100, seed=42)
    assert a == b
    assert len(a) == 100


def test_sample_target_questions_differs_by_corpus_even_with_same_seed():
    records = [{"id": i} for i in range(1000)]
    hotpot = sample_target_questions(records, "hotpot_qa", sample_size=100, seed=42)
    marco = sample_target_questions(records, "ms_marco", sample_size=100, seed=42)
    assert hotpot != marco  # different corpus string -> different rng seed


def test_sample_target_questions_returns_everything_if_sample_size_exceeds_records():
    records = [{"id": i} for i in range(5)]
    assert sample_target_questions(records, "hotpot_qa", sample_size=100) == records


# --------------------------------------------------------------------
# _parse_poison_response against 4 real captured Kimi-K2 responses
# --------------------------------------------------------------------

@pytest.mark.parametrize("fixture_name,expected_answer,expected_first_words", [
    # "---" between every section, no colon variant elsewhere -- baseline shape
    ("france_capital.txt", "Lyon", "In 2023, the French government"),
    # "---" between every section again, but from a completely different topic
    ("hamlet_author.txt", "Christopher Marlowe", "The Elizabethan theatre scene"),
    ("largest_planet.txt", "Saturn", "Saturn stands as the largest planet"),
    # lowercase "answer", "---" present only ONCE (before Corpus 1), absent
    # between the corpus sections themselves -- the most format-hostile
    # real sample captured
    ("gold_symbol.txt", "Ag", "Silver is one of the most precious metals"),
])
def test_parse_poison_response_against_real_fixtures(fixture_name, expected_answer, expected_first_words):
    content = _load_fixture(fixture_name)
    answer, corpora = _parse_poison_response(content, adv_per_query=ADV_PER_QUERY)

    assert answer == expected_answer
    assert len(corpora) == ADV_PER_QUERY
    assert corpora[0].startswith(expected_first_words)
    # no leftover markdown artifacts in any extracted corpus text
    for corpus_text in corpora:
        assert corpus_text  # non-empty
        assert not corpus_text.startswith("---")
        assert not corpus_text.endswith("---")
        assert "**Corpus" not in corpus_text  # didn't swallow the next header


def test_parse_poison_response_tolerates_a_conversational_preamble():
    # real, observed pattern: the model sometimes prefaces its answer with
    # a sentence before the labeled section starts
    content = (
        "I'll craft an incorrect answer and 5 supporting corpuses for you.\n\n"
        "**Incorrect Answer:** Berlin\n\n---\n\n"
        "**Corpus 1:**\nBerlin text one.\n\n---\n\n"
        "**Corpus 2:**\nBerlin text two.\n\n---\n\n"
        "**Corpus 3:**\nBerlin text three.\n\n---\n\n"
        "**Corpus 4:**\nBerlin text four.\n\n---\n\n"
        "**Corpus 5:**\nBerlin text five."
    )
    answer, corpora = _parse_poison_response(content, adv_per_query=5)
    assert answer == "Berlin"
    assert corpora == [f"Berlin text {w}." for w in ("one", "two", "three", "four", "five")]


def test_parse_poison_response_tolerates_corpus_header_with_no_colon():
    # real, observed (truncated) pattern: "**Corpus 1**" with no colon at all
    content = (
        "**Incorrect answer:** London\n\n"
        "**Corpus 1**\nLondon text one.\n\n"
        "**Corpus 2**\nLondon text two.\n\n"
        "**Corpus 3**\nLondon text three.\n\n"
        "**Corpus 4**\nLondon text four.\n\n"
        "**Corpus 5**\nLondon text five."
    )
    answer, corpora = _parse_poison_response(content, adv_per_query=5)
    assert answer == "London"
    assert corpora == [f"London text {w}." for w in ("one", "two", "three", "four", "five")]


def test_parse_poison_response_missing_answer_label_raises():
    content = "**Corpus 1:**\ntext\n\n**Corpus 2:**\ntext"
    with pytest.raises(ValueError, match="No 'Incorrect Answer:' label"):
        _parse_poison_response(content, adv_per_query=2)


def test_parse_poison_response_missing_corpus_section_raises():
    content = "**Incorrect Answer:** London\n\n**Corpus 1:**\ntext"
    with pytest.raises(ValueError, match="Missing corpus section"):
        _parse_poison_response(content, adv_per_query=ADV_PER_QUERY)


def test_build_poisoned_passage_matches_reference_repos_exact_concatenation():
    # question + "." + generated_text, NO space -- src/attack.py's
    # `adv_text_a = question + "."`, `adv_texts = [adv_text_a + i for ...]`
    result = build_poisoned_passage("Where is the Eiffel Tower", "It is in London.")
    assert result == "Where is the Eiffel Tower.It is in London."
