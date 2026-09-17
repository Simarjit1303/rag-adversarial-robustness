"""
evaluation.stats has no reference implementation in this repo to validate
against (scipy/statsmodels aren't dependencies -- see that module's
docstring), so these tests check known hand-computable cases instead of
comparing against a library.
"""

from evaluation.stats import fisher_exact_asr_comparison, holm_bonferroni, mcnemar_exact, paired_bootstrap_ci


# --------------------------------------------------------------------------
# mcnemar_exact
# --------------------------------------------------------------------------

def test_mcnemar_no_discordant_pairs_gives_p_one():
    # identical correctness everywhere -- no evidence of any difference
    correct_a = [1, 0, 1, 1, 0]
    correct_b = [1, 0, 1, 1, 0]
    result = mcnemar_exact(correct_a, correct_b)
    assert result["b"] == 0
    assert result["c"] == 0
    assert result["p_value"] == 1.0


def test_mcnemar_symmetric_discordance_gives_high_p_value():
    # b == c (5 vs 5) -- as symmetric as discordance gets, p should be high
    correct_a = [1] * 5 + [0] * 5 + [1] * 5
    correct_b = [0] * 5 + [1] * 5 + [1] * 5
    result = mcnemar_exact(correct_a, correct_b)
    assert result["b"] == 5
    assert result["c"] == 5
    assert result["p_value"] > 0.5


def test_mcnemar_lopsided_discordance_gives_low_p_value():
    # 20 cases where a is right and b is wrong, 1 the other way -- a real
    # asymmetric effect, should be significant at the usual alpha=0.05
    correct_a = [1] * 20 + [0] * 1 + [1] * 10
    correct_b = [0] * 20 + [1] * 1 + [1] * 10
    result = mcnemar_exact(correct_a, correct_b)
    assert result["b"] == 20
    assert result["c"] == 1
    assert result["p_value"] < 0.05


def test_mcnemar_mismatched_lengths_raises():
    import pytest
    with pytest.raises(ValueError, match="same length"):
        mcnemar_exact([1, 0], [1, 0, 1])


# --------------------------------------------------------------------------
# paired_bootstrap_ci
# --------------------------------------------------------------------------

def test_bootstrap_ci_identical_scores_gives_zero_drop_and_tight_ci():
    scores = [0.8, 0.6, 0.9, 0.7, 0.5] * 20
    result = paired_bootstrap_ci(scores, scores, n_boot=2000, seed=42)
    assert result["observed_drop"] == 0.0
    assert result["ci_low"] == 0.0
    assert result["ci_high"] == 0.0


def test_bootstrap_ci_constant_drop_is_captured():
    baseline = [0.9] * 50
    attack = [0.4] * 50  # constant 0.5 drop everywhere
    result = paired_bootstrap_ci(baseline, attack, n_boot=2000, seed=42)
    assert abs(result["observed_drop"] - 0.5) < 1e-9
    assert result["ci_low"] == result["ci_high"] == result["observed_drop"]  # zero variance -> zero-width CI


def test_bootstrap_ci_mismatched_lengths_raises():
    import pytest
    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap_ci([0.5, 0.6], [0.5, 0.6, 0.7])


def test_bootstrap_ci_is_reproducible_given_same_seed():
    baseline = [0.9, 0.2, 0.5, 0.8, 0.1, 0.95, 0.3, 0.6]
    attack = [0.1, 0.2, 0.4, 0.3, 0.0, 0.10, 0.2, 0.4]
    r1 = paired_bootstrap_ci(baseline, attack, n_boot=500, seed=7)
    r2 = paired_bootstrap_ci(baseline, attack, n_boot=500, seed=7)
    assert r1 == r2


# --------------------------------------------------------------------------
# fisher_exact_asr_comparison
# --------------------------------------------------------------------------

def test_fisher_exact_identical_proportions_gives_high_p_value():
    result = fisher_exact_asr_comparison(success_a=50, n_a=100, success_b=50, n_b=100)
    assert result["p_value"] > 0.9


def test_fisher_exact_extreme_difference_gives_tiny_p_value():
    # 10/10 vs 0/10 -- the single most extreme table possible at this n
    result = fisher_exact_asr_comparison(success_a=10, n_a=10, success_b=0, n_b=10)
    assert result["p_value"] < 0.001


def test_fisher_exact_tiny_symmetric_table_hand_verified():
    # n_a=n_b=1: group A succeeds, group B fails. By symmetry the only two
    # possible tables at these margins are equally likely (each p=0.5),
    # both "at least as extreme" as each other -- p_value must be exactly 1.0.
    result = fisher_exact_asr_comparison(success_a=1, n_a=1, success_b=0, n_b=1)
    assert result["p_value"] == 1.0


def test_fisher_exact_returns_the_input_counts():
    result = fisher_exact_asr_comparison(success_a=3, n_a=20, success_b=7, n_b=20)
    assert result["success_a"] == 3 and result["n_a"] == 20
    assert result["success_b"] == 7 and result["n_b"] == 20


# --------------------------------------------------------------------------
# holm_bonferroni
# --------------------------------------------------------------------------

def test_holm_bonferroni_all_significant():
    # tiny p-values, generous alpha -- everything survives correction
    p_values = [0.001, 0.002, 0.003]
    result = holm_bonferroni(p_values, alpha=0.05)
    assert [r["significant_holm"] for r in result] == [True, True, True]


def test_holm_bonferroni_step_down_stops_at_first_failure():
    # classic textbook case: 3 p-values, alpha=0.05 -- smallest tested
    # against 0.05/3, next against 0.05/2, once one fails all larger ones
    # fail too regardless of their own individual threshold
    p_values = [0.01, 0.04, 0.03]  # unsorted on purpose
    result = holm_bonferroni(p_values, alpha=0.05)
    # sorted order: 0.01 (rank1, thresh .0167 -> pass), 0.03 (rank2, thresh
    # .025 -> FAIL), 0.04 (rank3 -> fails regardless, Holm step-down)
    by_input_order = {i: r for i, r in enumerate(result)}
    assert by_input_order[0]["significant_holm"] is True   # 0.01
    assert by_input_order[2]["significant_holm"] is False  # 0.03
    assert by_input_order[1]["significant_holm"] is False  # 0.04


def test_holm_bonferroni_preserves_input_order_not_sorted_order():
    p_values = [0.5, 0.001, 0.9]
    result = holm_bonferroni(p_values, alpha=0.05)
    assert [r["p_value"] for r in result] == p_values  # same order as input


def test_holm_bonferroni_empty_input():
    assert holm_bonferroni([], alpha=0.05) == []
