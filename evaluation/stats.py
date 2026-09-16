import math
import numpy as np

def mcnemar_exact(correct_a, correct_b):
    if len(correct_a) != len(correct_b):
        raise ValueError('correct_a and correct_b must be the same length (paired by question)')
    b = sum((1 for a, x in zip(correct_a, correct_b) if a == 1 and x == 0))
    c = sum((1 for a, x in zip(correct_a, correct_b) if a == 0 and x == 1))
    n = b + c
    if n == 0:
        p_value = 1.0
    else:
        k = min(b, c)
        cdf_le_k = sum((math.comb(n, i) * 0.5 ** i * 0.5 ** (n - i) for i in range(k + 1)))
        p_value = min(2 * cdf_le_k, 1.0)
    return {'b': b, 'c': c, 'n_discordant': n, 'p_value': p_value}

def fisher_exact_asr_comparison(success_a: int, n_a: int, success_b: int, n_b: int):
    row_a, row_b = (n_a, n_b)
    col_success = success_a + success_b
    n = row_a + row_b

    def hypergeom_pmf(k):
        if k < 0 or k > row_a or col_success - k < 0 or (col_success - k > row_b):
            return 0.0
        return math.comb(row_a, k) * math.comb(row_b, col_success - k) / math.comb(n, col_success)
    lo = max(0, col_success - row_b)
    hi = min(row_a, col_success)
    p_observed = hypergeom_pmf(success_a)
    p_value = sum((p for k in range(lo, hi + 1) if (p := hypergeom_pmf(k)) <= p_observed * (1 + 1e-09)))
    return {'success_a': success_a, 'n_a': n_a, 'success_b': success_b, 'n_b': n_b, 'p_value': min(p_value, 1.0)}

def paired_bootstrap_ci(scores_a, scores_b, n_boot: int=10000, ci: float=0.95, seed: int=0):
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError('scores_a and scores_b must be the same length (paired by question)')
    n = a.shape[0]
    if n == 0:
        raise ValueError('scores_a/scores_b must be non-empty')
    diffs = a - b
    observed_drop = float(diffs.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diffs[idx].mean(axis=1)
    alpha = 1 - ci
    lo, hi = np.quantile(boot_means, [alpha / 2, 1 - alpha / 2])
    return {'observed_drop': observed_drop, 'ci_low': float(lo), 'ci_high': float(hi), 'ci': ci, 'n': n, 'n_boot': n_boot}

def holm_bonferroni(p_values, alpha: float=0.05):
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    reject = [False] * m
    for rank, i in enumerate(order):
        threshold = alpha / (m - rank)
        if p_values[i] <= threshold:
            reject[i] = True
        else:
            break
    return [{'p_value': p_values[i], 'significant_holm': reject[i], 'rank': order.index(i) + 1, 'alpha': alpha} for i in range(m)]
