"""Covariance and correlation estimators.

Two families live here.  `missing_cov` handles the case where the observation
matrix has holes in it, which is Week 03's missing-data section.  `ew_covar` is
the exponentially weighted (RiskMetrics) estimator from the same week.

Covers functional tests 1.1 through 2.3.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Missing data (tests 1.1 - 1.4)
# ---------------------------------------------------------------------------

def _pair_stat(a, b, corr):
    """Covariance or correlation of two aligned 1-D arrays, n-1 denominator."""
    c = np.cov(a, b, ddof=1)
    if not corr:
        return c[0, 1]
    denom = np.sqrt(c[0, 0] * c[1, 1])
    return c[0, 1] / denom


def missing_cov(x, skip_miss=True, corr=False):
    """Covariance (or correlation) of a matrix containing missing values.

    Two strategies, which is the Week 03 trade:

    `skip_miss=True` keeps only the rows that are complete across *every*
    column and runs the ordinary estimator on what is left.  Less data, but the
    result is guaranteed PSD because it comes from a single sample.

    `skip_miss=False` is pairwise: entry (i, j) is estimated from the rows where
    columns i and j are both present, so each entry uses a different sample.
    More data, and no guarantee of PSD -- a set of mutually inconsistent
    correlations is exactly what a negative eigenvalue is.  Tests 3.x exist to
    repair the matrix this produces.

    `x` is an (n, m) array where missing entries are NaN.  Returns (m, m).
    """
    x = np.asarray(x, dtype=float)
    n, m = x.shape
    present = ~np.isnan(x)

    if present.all():
        return np.corrcoef(x, rowvar=False) if corr else np.cov(x, rowvar=False, ddof=1)

    if skip_miss:
        keep = present.all(axis=1)
        sub = x[keep]
        return np.corrcoef(sub, rowvar=False) if corr else np.cov(sub, rowvar=False, ddof=1)

    out = np.empty((m, m), dtype=float)
    for i in range(m):
        for j in range(i + 1):
            # Rows usable for this pair: both columns present.  For the
            # diagonal this reduces to "column i present", and the pair
            # statistic reduces to the variance (or to 1.0 for a correlation).
            rows = present[:, i] & present[:, j]
            out[i, j] = _pair_stat(x[rows, i], x[rows, j], corr)
            out[j, i] = out[i, j]
    return out


# ---------------------------------------------------------------------------
# Exponentially weighted (tests 2.1 - 2.3)
# ---------------------------------------------------------------------------

def exp_weights(m, lam):
    """Normalized exponential weights for `m` observations.

    Weight (1-lambda) * lambda^(m-i) for i = 1..m, so the LAST row carries the
    most weight; the data is assumed to be in time order, oldest first.  The
    raw weights sum to 1 only over an infinite history, so they are normalized
    to sum to 1 over the sample actually held.
    """
    i = np.arange(1, m + 1)
    w = (1.0 - lam) * lam ** (m - i)
    return w / w.sum()


def ew_covar(x, lam):
    """Exponentially weighted covariance matrix.

    The weighted mean is removed from each column before the cross products are
    formed, so this is a covariance rather than a second-moment matrix.  With
    weights that sum to 1 there is no further divisor.
    """
    x = np.asarray(x, dtype=float)
    m = x.shape[0]
    w = exp_weights(m, lam)
    xm = np.sqrt(w)[:, None] * (x - w @ x)
    return xm.T @ xm


def cov_to_corr(c):
    """Strip the scale out of a covariance matrix."""
    inv_sd = 1.0 / np.sqrt(np.diag(c))
    return inv_sd[:, None] * c * inv_sd[None, :]


def corr_to_cov(corr, sd):
    """Put a given vector of standard deviations back onto a correlation."""
    sd = np.asarray(sd, dtype=float)
    return sd[:, None] * corr * sd[None, :]


def ew_cov_var_corr(x, lam_var, lam_corr):
    """Covariance built from an EW variance and a *separately weighted* EW
    correlation.

    Test 2.3.  Splitting the two is the Week 03 recommendation: the variance and
    the correlation do not have to be estimated at the same speed, and in
    practice you often want the variance to react faster than the correlation
    (or the reverse, as here).
    """
    sd = np.sqrt(np.diag(ew_covar(x, lam_var)))
    corr = cov_to_corr(ew_covar(x, lam_corr))
    return corr_to_cov(corr, sd)
