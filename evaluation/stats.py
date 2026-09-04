"""
Significance testing shared across attack phases: exact McNemar's test,
paired bootstrap confidence intervals, and Holm-Bonferroni correction.
General-purpose (not injection-specific) so Phase 1/4's planned McNemar
work (see evaluation/run_baseline.py's module docstring) can reuse it
rather than re-implement it.

No scipy/statsmodels dependency: neither is in requirements.txt, and both
functions below are a handful of lines on top of stdlib `math` (exact
McNemar) and numpy (paired bootstrap -- numpy is already a hard dependency
via requirements.txt, so this doesn't add a new one).
"""

import math

import numpy as np


def mcnemar_exact(correct_a, correct_b):
    """
    Exact (not chi-square-approximated) two-sided McNemar's test on paired
    binary correctness arrays -- e.g. baseline-correct vs attack-correct,
    same question order in both. Only the DISCORDANT pairs matter: b = a
    correct & b wrong, c = a wrong & b correct.

    p-value = 2 * P(X <= min(b, c)), X ~ Binomial(n=b+c, p=0.5), capped at
    1.0 -- the standard exact McNemar formula (equivalent to R's
    mcnemar.test(..., correct=FALSE) exact variant / scipy's binomtest
    applied to the discordant pairs). Preferred over the chi-square
    approximation here since attack-vs-baseline discordant-pair counts can
    be small per (model, corpus) cell, where the approximation is least
    reliable.
    """
    if len(correct_a) != len(correct_b):
        raise ValueError("correct_a and correct_b must be the same length (paired by question)")

    b = sum(1 for a, x in zip(correct_a, correct_b) if a == 1 and x == 0)
    c = sum(1 for a, x in zip(correct_a, correct_b) if a == 0 and x == 1)
    n = b + c

    if n == 0:
        # No discordant pairs at all -- no evidence of any difference,
        # p = 1.0 by convention rather than an undefined 0/0.
        p_value = 1.0
    else:
        k = min(b, c)
        cdf_le_k = sum(
            math.comb(n, i) * (0.5 ** i) * (0.5 ** (n - i)) for i in range(k + 1)
        )
        p_value = min(2 * cdf_le_k, 1.0)

    return {"b": b, "c": c, "n_discordant": n, "p_value": p_value}


def paired_bootstrap_ci(scores_a, scores_b, n_boot: int = 10000, ci: float = 0.95, seed: int = 0):
    """
    (1-alpha) CI on mean(scores_a - scores_b) via paired bootstrap,
    resampling QUESTION INDICES (not each list independently) so the
    per-question baseline/attack pairing is preserved in every resample --
    resampling the two lists independently would treat real within-question
    correlation as noise and inflate the interval.

    scores_a/scores_b: same-length, same-question-order score lists (e.g.
    Phase 1 baseline F1 vs this attack's F1-under-attack, paired by
    question). observed_drop = mean(a) - mean(b); positive means a > b
    (e.g. baseline utility higher than under-attack utility).
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("scores_a and scores_b must be the same length (paired by question)")
    n = a.shape[0]
    if n == 0:
        raise ValueError("scores_a/scores_b must be non-empty")

    diffs = a - b
    observed_drop = float(diffs.mean())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diffs[idx].mean(axis=1)

    alpha = 1 - ci
    lo, hi = np.quantile(boot_means, [alpha / 2, 1 - alpha / 2])

    return {
        "observed_drop": observed_drop,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "ci": ci,
        "n": n,
        "n_boot": n_boot,
    }


def holm_bonferroni(p_values, alpha: float = 0.05):
    """
    Standard Holm step-down correction across a family of p-values.
    Returns a list of dicts in the SAME ORDER as the input p_values (not
    re-sorted by significance) so callers can zip the result back against
    whatever (model, corpus) comparison produced each p-value.

    Step-down: sort ascending, test smallest against alpha/m, next against
    alpha/(m-1), etc.; the first failure to reject makes every larger
    p-value in the family fail too (Holm's procedure, not independent
    per-comparison Bonferroni -- less conservative, still controls the
    family-wise error rate).
    """
    m = len(p_values)
    if m == 0:
        return []

    order = sorted(range(m), key=lambda i: p_values[i])
    reject = [False] * m
    for rank, i in enumerate(order):  # rank is 0-indexed
        threshold = alpha / (m - rank)
        if p_values[i] <= threshold:
            reject[i] = True
        else:
            break  # every remaining (larger) p-value in the family also fails

    return [
        {
            "p_value": p_values[i],
            "significant_holm": reject[i],
            "rank": order.index(i) + 1,  # 1-indexed, smallest p-value = rank 1
            "alpha": alpha,
        }
        for i in range(m)
    ]
