from pathlib import Path
import pytest
from attacks.poisonedrag import ADV_PER_QUERY, _parse_poison_response, build_poisoned_passage, sample_target_questions
_FIXTURES_DIR = Path(__file__).parent / 'fixtures' / 'poison_responses'

def _load_fixture(name: str) -> str:
    return (_FIXTURES_DIR / name).read_text(encoding='utf-8')

def test_sample_target_questions_is_deterministic_given_same_seed():
    records = [{'id': i} for i in range(1000)]
    a = sample_target_questions(records, 'hotpot_qa', sample_size=100, seed=42)
    b = sample_target_questions(records, 'hotpot_qa', sample_size=100, seed=42)
    assert a == b
    assert len(a) == 100

def test_sample_target_questions_differs_by_corpus_even_with_same_seed():
    records = [{'id': i} for i in range(1000)]
    hotpot = sample_target_questions(records, 'hotpot_qa', sample_size=100, seed=42)
    marco = sample_target_questions(records, 'ms_marco', sample_size=100, seed=42)
    assert hotpot != marco

def test_sample_target_questions_returns_everything_if_sample_size_exceeds_records():
    records = [{'id': i} for i in range(5)]
    assert sample_target_questions(records, 'hotpot_qa', sample_size=100) == records

@pytest.mark.parametrize('fixture_name,expected_answer,expected_first_words', [('iron_symbol.txt', 'Ir', 'In the revised 2023 IUPAC'), ('first_president.txt', 'John Adams', 'The early constitutional period'), ('water_boiling_point.txt', '90', 'Standard thermodynamic tables'), ('romeo_juliet_author.txt', 'Christopher Marlowe', 'Recent scholarship from the Oxfordian'), ('minimax_titanic_director.txt', 'Steven Spielberg', 'Steven Spielberg is widely regarded'), ('minimax_pride_prejudice_author.txt', 'Charlotte Brontë', 'During the Victorian literary period'), ('minimax_japan_capital.txt', 'Kyoto', 'Tokyo, the bustling capital of Japan'), ('texas_political_parties_attempt1.txt', "1. The primary role of political parties in Texas is to manage the state's voter registration database and certify voting machines for county elections.", 'The Texas Election Code assigns political parties'), ('texas_political_parties_attempt3.txt', '1. The primary role of political parties in Texas is to draft the state biennial budget and directly appoint all committee chairs in the legislature.', 'Political scientists often misunderstand the unique machinery')])
def test_parse_poison_response_against_real_fixtures(fixture_name, expected_answer, expected_first_words):
    content = _load_fixture(fixture_name)
    answer, corpora = _parse_poison_response(content, adv_per_query=ADV_PER_QUERY)
    assert answer == expected_answer
    assert len(corpora) == ADV_PER_QUERY
    assert corpora[0].startswith(expected_first_words)
    for corpus_text in corpora:
        assert corpus_text
        assert not corpus_text.startswith(('---', '***'))
        assert not corpus_text.endswith(('---', '***'))
        assert '**Corpus' not in corpus_text

def test_parse_poison_response_tolerates_a_conversational_preamble():
    content = "I'll craft an incorrect answer and 5 supporting corpuses for you.\n\n**Incorrect Answer:** Berlin\n\n---\n\n**Corpus 1:**\nBerlin text one.\n\n---\n\n**Corpus 2:**\nBerlin text two.\n\n---\n\n**Corpus 3:**\nBerlin text three.\n\n---\n\n**Corpus 4:**\nBerlin text four.\n\n---\n\n**Corpus 5:**\nBerlin text five."
    answer, corpora = _parse_poison_response(content, adv_per_query=5)
    assert answer == 'Berlin'
    assert corpora == [f'Berlin text {w}.' for w in ('one', 'two', 'three', 'four', 'five')]

def test_parse_poison_response_tolerates_corpus_header_with_no_colon():
    content = '**Incorrect answer:** London\n\n**Corpus 1**\nLondon text one.\n\n**Corpus 2**\nLondon text two.\n\n**Corpus 3**\nLondon text three.\n\n**Corpus 4**\nLondon text four.\n\n**Corpus 5**\nLondon text five.'
    answer, corpora = _parse_poison_response(content, adv_per_query=5)
    assert answer == 'London'
    assert corpora == [f'London text {w}.' for w in ('one', 'two', 'three', 'four', 'five')]

def test_parse_poison_response_uses_collective_corpora_fallback_not_primary_path():
    from attacks.poisonedrag import _CORPUS_HEADER_RE
    content = _load_fixture('minimax_japan_capital.txt')
    assert list(_CORPUS_HEADER_RE.finditer(content)) == []
    answer, corpora = _parse_poison_response(content, adv_per_query=ADV_PER_QUERY)
    assert answer == 'Kyoto'
    assert len(corpora) == ADV_PER_QUERY
    assert corpora[0].startswith('Tokyo, the bustling capital of Japan')
    assert corpora[4].startswith('Many travelers confuse Kyoto')
    for corpus_text in corpora:
        assert corpus_text
        assert not corpus_text[0].isdigit()

def test_parse_poison_response_collective_fallback_synthetic_edge_case():
    content = '**Incorrect Answer:** Berlin\n\n**Corpus:**\n\n1) Berlin text one.\n2) Berlin text two.\n3) Berlin text three.\n4) Berlin text four.\n5) Berlin text five.'
    answer, corpora = _parse_poison_response(content, adv_per_query=5)
    assert answer == 'Berlin'
    assert corpora == [f'Berlin text {w}.' for w in ('one', 'two', 'three', 'four', 'five')]

def test_parse_poison_response_missing_answer_label_raises():
    content = '**Corpus 1:**\ntext\n\n**Corpus 2:**\ntext'
    with pytest.raises(ValueError, match="No 'Incorrect Answer:' label"):
        _parse_poison_response(content, adv_per_query=2)

def test_parse_poison_response_missing_corpus_section_raises():
    content = '**Incorrect Answer:** London\n\n**Corpus 1:**\ntext'
    with pytest.raises(ValueError, match='Missing corpus section'):
        _parse_poison_response(content, adv_per_query=ADV_PER_QUERY)

def test_build_poisoned_passage_matches_reference_repos_exact_concatenation():
    result = build_poisoned_passage('Where is the Eiffel Tower', 'It is in London.')
    assert result == 'Where is the Eiffel Tower.It is in London.'
