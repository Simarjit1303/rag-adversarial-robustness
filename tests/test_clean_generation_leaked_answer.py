import pytest
from harness.pipeline import clean_generation
CASES = [("[1] {'question': 'what season is pepper in american horror story', 'answer': ['the second and fourth seasons']}", 'the second and fourth seasons'), ("[1] {'question': 'when did jackie robinson became rookie of the year', 'answer': ['in 1947']}", 'in 1947'), ("[1] {'question': 'who has the best nba record in history', 'answer': ['.890']}", '.890'), ("[1] {'question': 'when were the extra books of the catholic bible added', 'answer': ['393']}", '393'), ("[1] {'question': 'thicknet is the colloquial name for which ethernet standard', 'answer': ['10BASE5']}", '10BASE5'), ("[1] {'question': 'when does the lion witch and the wardrobe take place', 'answer': ['1940']}", '1940'), ("['.890']", '.890'), ('[1]', '[1]')]

@pytest.mark.parametrize('raw, expected', CASES)
def test_clean_generation_extracts_leaked_answer(raw, expected):
    assert clean_generation(raw) == expected

def test_does_not_fire_on_answer_containing_brackets_for_legitimate_reasons():
    assert clean_generation('the [REDACTED] files') == 'the [REDACTED] files'

def test_does_not_fire_on_answer_starting_with_bracket_but_not_a_string_list():
    assert clean_generation('[42]') == '[42]'

def test_does_not_fire_on_curly_braces_without_the_answer_key_shape():
    assert clean_generation('the set is {1, 2, 3}') == 'the set is {1, 2, 3}'

def test_extracted_answer_still_goes_through_normal_cleanup():
    raw = "[1] {'question': 'x', 'answer': ['Paris.']}"
    assert clean_generation(raw) == 'Paris'
