from evaluation.metrics import exact_match, normalize_text

def test_hyphenated_range_matches_spaced_hyphen_variant():
    assert normalize_text('28-32') == normalize_text('28 - 32')
    assert exact_match('28-32', ['28 - 32']) == 1

def test_punctuation_becomes_a_separator_not_deleted():
    assert normalize_text('co-founder') == 'co founder'
    assert normalize_text('U.S.') == 'u s'
