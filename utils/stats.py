from typing import Mapping, Sequence

from scipy import stats
import numpy as np
import pandas as pd


def significance_label(p_value: float) -> str:
    """Star notation for a p-value: *** < 0.001, ** < 0.01, * < 0.05, else n.s."""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def paired_t_tests_against(
    reference: str, values: Mapping[str, Sequence[float]]
) -> pd.DataFrame:
    """
    Two-sided paired t-test of every group against the reference group.

    Replicate i of each group is paired with replicate i of the reference
    (e.g. the wells in the same row of a plate), so the groups must list their
    replicates in the same order.  The p-values are not corrected for multiple
    comparisons.

    Args:
        reference: Name of the group every other group is compared with
        values: Replicate values of each group, by group name

    Returns:
        One row per group compared with the reference, indexed by group name:
        number of pairs, mean of the paired differences (reference - group),
        t-statistic, degrees of freedom, p-value and its star notation.
    """
    reference_values = np.asarray(values[reference], dtype=float)
    rows = {}
    for name, group_values in values.items():
        if name == reference:
            continue
        difference = reference_values - np.asarray(group_values, dtype=float)
        result = t_test_paired(list(reference_values), list(group_values))
        rows[name] = {
            "n_pairs": len(difference),
            "mean_difference": float(difference.mean()),
            "t_statistic": float(result["t_statistic"]),
            "degrees_of_freedom": len(difference) - 1,
            "p_value": float(result["p_value"]),
            "significance": significance_label(float(result["p_value"])),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def t_test_paired(data_1: list, data_2: list):
    """
    Perform a paired t-test between two DataFrames and return the t-statistic.

    Args:
        data_1: First data
        data_2: Second data

    Returns:
        float: t-statistic from the paired t-test
    """
    # Perform paired t-test
    t_statistic, p_value = stats.ttest_rel(data_1, data_2)

    return {"t_statistic": t_statistic, "p_value": p_value}


def t_test_unpaired(data_1: list, data_2: list):
    """
    Perform a unpaired t-test between two DataFrames and return the t-statistic.

    Args:
        data_1: First data
        data_2: Second data

    Returns:
        float: t-statistic from the unpaired t-test
    """
    # Perform unpaired t-test
    t_statistic, p_value = stats.ttest_ind(data_1, data_2)

    return {"t_statistic": t_statistic, "p_value": p_value}


def gaussian_likelihood_ratio_test(data_1: list, data_2: list):
    """
    Perform a Gaussian likelihood ratio test for equal means.
    H0: mu_1 = mu_2 (equal means, pooled variance)
    H1: mu_1 ≠ mu_2 (separate means and variances)
    """
    data_combined = [*data_1, *data_2]
    n1, n2 = len(data_1), len(data_2)

    # Under H1 (separate distributions)
    mu_1, mu_2 = np.mean(data_1), np.mean(data_2)
    sigma_1, sigma_2 = np.std(data_1, ddof=1), np.std(data_2, ddof=1)

    # Under H0 (same mean, pooled variance)
    mu_pooled = np.mean(data_combined)
    sigma_pooled = np.std(data_combined, ddof=1)

    ll_h1 = np.sum(stats.norm.logpdf(data_1, mu_1, sigma_1)) + np.sum(
        stats.norm.logpdf(data_2, mu_2, sigma_2)
    )
    ll_h0 = np.sum(stats.norm.logpdf(data_combined, mu_pooled, sigma_pooled))

    test_statistic = -2 * (ll_h0 - ll_h1)
    p_value = 1 - stats.chi2.cdf(test_statistic, df=2)  # 2 extra parameters in H1

    return {
        "test_statistic": test_statistic,
        "p_value": p_value,
        "ll_h0": ll_h0,
        "ll_h1": ll_h1,
    }
