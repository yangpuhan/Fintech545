"""
Shared helpers for Assignment 1.

Everything that more than one problem needs lives here: the moment estimators,
the AICc calculation, and the two maximum likelihood regressions used in
Problem 2.  Keeping them in one place means the AICc parameter count is defined
once, which is the number most easily gotten wrong in this assignment.
"""

import numpy as np
from scipy import optimize, stats

# Course convention, stated in the class repository README.  Not used to
# annualize anything below -- no problem in this assignment asks for an
# annualized figure -- but recorded here so the convention is explicit.
TRADING_DAYS_PER_YEAR = 255


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------

def four_moments(x):
    """First four moments of a sample, with the bias-corrected estimators.

    Conventions, both of which have to be stated explicitly because packages
    disagree:

      * variance uses the n-1 denominator (unbiased);
      * skewness and excess kurtosis use the bias-corrected estimators, and
        kurtosis is reported in EXCESS form, so a normal sample returns 0
        rather than 3.

    scipy defaults to the biased ("population") estimators and to Fisher's
    excess convention, so `bias=False` is passed explicitly and the excess
    convention is left at its default.  Both values are returned so the
    write-up can quote the pair and show the correction is immaterial at
    n = 1000.
    """
    x = np.asarray(x, dtype=float)
    return {
        "n": len(x),
        "mean": float(np.mean(x)),
        "variance": float(np.var(x, ddof=1)),
        "sd": float(np.std(x, ddof=1)),
        "skew": float(stats.skew(x, bias=False)),
        "skew_biased": float(stats.skew(x, bias=True)),
        "excess_kurtosis": float(stats.kurtosis(x, bias=False)),
        "excess_kurtosis_biased": float(stats.kurtosis(x, bias=True)),
    }


def moment_feasibility(skew, excess_kurtosis):
    """Check the constraint kurtosis >= skewness^2 + 1 that holds for every
    distribution.  Week 01 notes it as an arithmetic check: an excess kurtosis
    below skew^2 - 2 is impossible, not merely unusual.
    """
    raw_kurtosis = excess_kurtosis + 3.0
    return {
        "raw_kurtosis": raw_kurtosis,
        "lower_bound": skew ** 2 + 1.0,
        "satisfied": raw_kurtosis >= skew ** 2 + 1.0,
    }


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def aicc(loglik, k, n):
    """Corrected Akaike Information Criterion.  Lower is better.

        AIC  = 2k - 2 ln(L)
        AICc = AIC + (2k^2 + 2k) / (n - k - 1)

    `k` is the TOTAL number of fitted parameters.  Following the Week 02
    convention, k = p + d where p counts the regression parameters including
    the intercept and d counts the extra parameters of the error distribution.
    A one-regressor model with normal errors therefore has p = 2, d = 1, k = 3;
    the same model with Student's t errors has d = 2 (scale and degrees of
    freedom) and k = 4.

    Most packages report AIC only, so the correction term is added here by
    hand.  It is small at these sample sizes but it is what the assignment asks
    for, and it never hurts.
    """
    if n - k - 1 <= 0:
        raise ValueError("AICc is undefined when n <= k + 1")
    return 2 * k - 2 * loglik + (2 * k ** 2 + 2 * k) / (n - k - 1)


# ---------------------------------------------------------------------------
# Maximum likelihood regressions (Problem 2)
# ---------------------------------------------------------------------------
#
# Both fits optimize an unconstrained parameter vector.  The scale is carried
# as log(sigma) and the degrees of freedom as log(nu - 2), so the optimizer can
# roam over the whole real line while sigma stays positive and nu stays above
# 2, which is where the t distribution's variance exists.  Constraining by
# transform rather than by rejecting proposals keeps the surface smooth.

def _design(x):
    """[1 x] design matrix, so the intercept is estimated rather than assumed."""
    x = np.asarray(x, dtype=float)
    return np.column_stack([np.ones(len(x)), x])


def fit_ols(x, y):
    """Ordinary least squares with the textbook standard errors.

    Reported for its own sake and as the starting value for the two MLE fits.
    The standard error formula is s * sqrt(diag((X'X)^-1)) with
    s^2 = e'e / (n - p), the Week 02 expression.
    """
    X, y = _design(x), np.asarray(y, dtype=float)
    n, p = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    s2 = resid @ resid / (n - p)                  # unbiased residual variance
    se = np.sqrt(s2 * np.diag(XtX_inv))
    # Log likelihood evaluated at the OLS fit under a normal error, using the
    # MLE variance (divisor n).  This is what makes the OLS and normal-MLE
    # log likelihoods directly comparable.
    sigma_mle = np.sqrt(resid @ resid / n)
    loglik = float(np.sum(stats.norm.logpdf(resid, loc=0.0, scale=sigma_mle)))
    return {
        "name": "OLS",
        "alpha": float(beta[0]), "beta": float(beta[1]),
        "se_alpha": float(se[0]), "se_beta": float(se[1]),
        "s": float(np.sqrt(s2)),          # unbiased residual sd, the OLS report
        "sigma_mle": float(sigma_mle),    # the same fit read as an MLE
        "loglik": loglik, "k": 3,
        "resid": resid,
    }


def fit_normal_mle(x, y, start=None):
    """Regression with a normal error term, fitted by maximum likelihood.

    The slope and intercept come out identical to OLS -- maximizing the normal
    log likelihood over beta IS minimizing the sum of squared errors -- so this
    fit exists to produce a likelihood on the same footing as the t fit, and to
    show the variance estimate differs from the OLS one by the factor (n-p)/n.
    """
    X, y = _design(x), np.asarray(y, dtype=float)
    n = len(y)

    def negative_loglik(theta):
        a, b, log_sigma = theta
        resid = y - a - b * X[:, 1]
        return -np.sum(stats.norm.logpdf(resid, loc=0.0, scale=np.exp(log_sigma)))

    if start is None:
        ols = fit_ols(x, y)
        start = [ols["alpha"], ols["beta"], np.log(ols["sigma_mle"])]
    res = optimize.minimize(negative_loglik, start, method="Nelder-Mead",
                            options={"xatol": 1e-12, "fatol": 1e-12,
                                     "maxiter": 50000, "maxfev": 50000})
    a, b, log_sigma = res.x
    return {
        "name": "MLE, normal error",
        "alpha": float(a), "beta": float(b), "sigma": float(np.exp(log_sigma)),
        "loglik": float(-res.fun), "k": 3, "converged": bool(res.success),
        "resid": y - a - b * X[:, 1],
    }


def fit_t_mle(x, y, start=None):
    """Regression with a Student's t error term, fitted by maximum likelihood.

    Four parameters: intercept, slope, scale, degrees of freedom.  Note that
    the fitted `sigma` is the t distribution's SCALE, not its standard
    deviation; the standard deviation is sigma * sqrt(nu / (nu - 2)) and is
    returned separately as `implied_sd` so the two error models can be compared
    on the same footing.
    """
    X, y = _design(x), np.asarray(y, dtype=float)

    def negative_loglik(theta):
        a, b, log_sigma, log_nu_minus_2 = theta
        sigma = np.exp(log_sigma)
        nu = 2.0 + np.exp(log_nu_minus_2)
        resid = y - a - b * X[:, 1]
        return -np.sum(stats.t.logpdf(resid, df=nu, loc=0.0, scale=sigma))

    if start is None:
        ols = fit_ols(x, y)
        # Start the scale below the OLS residual sd: a t fitted to fat-tailed
        # data wants a narrower body than the normal does.
        start = [ols["alpha"], ols["beta"],
                 np.log(0.8 * ols["sigma_mle"]), np.log(4.0)]
    res = optimize.minimize(negative_loglik, start, method="Nelder-Mead",
                            options={"xatol": 1e-12, "fatol": 1e-12,
                                     "maxiter": 100000, "maxfev": 100000})
    a, b, log_sigma, log_nu_minus_2 = res.x
    sigma = float(np.exp(log_sigma))
    nu = float(2.0 + np.exp(log_nu_minus_2))
    return {
        "name": "MLE, Student's t error",
        "alpha": float(a), "beta": float(b), "sigma": sigma, "nu": nu,
        "implied_sd": sigma * np.sqrt(nu / (nu - 2.0)) if nu > 2 else np.inf,
        "loglik": float(-res.fun), "k": 4, "converged": bool(res.success),
        "resid": y - a - b * X[:, 1],
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def rule(title=""):
    """Section divider, so the console output can be read as a report."""
    if title:
        print("\n" + "=" * 78)
        print(title)
        print("=" * 78)
    else:
        print("-" * 78)
