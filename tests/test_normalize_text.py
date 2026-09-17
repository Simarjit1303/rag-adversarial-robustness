"""
Regression coverage for evaluation/metrics.py's normalize_text, added after
a real bug: punctuation was being DELETED outright instead of replaced with
a space, so whether two equivalent answers matched depended on incidental
formatting around the punctuation, not the punctuation itself. Found live on
a PoisonedRAG phi-4-mini/ms_marco cell: generated_answer_clean "28-32" vs
target_answer "28 - 32" scored attack_success=0 despite being the same
answer. test_metrics.py explicitly disclaims covering normalize_text/
exact_match, so this is a new file rather than an addition there.
"""

from evaluation.metrics import exact_match, normalize_text


def test_hyphenated_range_matches_spaced_hyphen_variant():
    # the exact real-world pair that surfaced this bug
    assert normalize_text("28-32") == normalize_text("28 - 32")
    assert exact_match("28-32", ["28 - 32"]) == 1


def test_punctuation_becomes_a_separator_not_deleted():
    # deleting punctuation outright used to fuse adjacent tokens ("2832");
    # replacing it with a space keeps them distinct, matching the SQuAD-
    # style normalizer this project intends to follow
    assert normalize_text("co-founder") == "co founder"
    assert normalize_text("U.S.") == "u s"
