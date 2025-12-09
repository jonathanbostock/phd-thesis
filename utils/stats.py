from scipy import stats
import numpy as np


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
