"""Fitting distributions and regressions by maximum likelihood.

The generalized t used throughout is the location-scale family

    X = mu + sigma * T_nu

so `sigma` is a SCALE, not a standard deviation.  The standard deviation is
sigma * sqrt(nu / (nu - 2)) and exists only for nu > 2.  Mixing the two up is
the easiest way to get a wrong answer here that still looks plausible.

Covers functional tests 7.1 through 7.4.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import optimize, stats


@dataclass
class FittedModel:
    """A fitted error model, optionally with regression coefficients.

    `errors` and `u` are carried because the copula work in Week 05 needs the
    fitted CDF values, not just the parameters.
    """
    error_model: object
    beta: Optional[np.ndarray] = None
    errors: np.ndarray = field(default=None, repr=False)
    u: np.ndarray = field(default=None, repr=False)
    n_error_params: int = 2

    def loglik(self, data):
        return float(np.sum(self.error_model.logpdf(np.asarray(data, dtype=float))))


def aicc(loglik, k, n):
    """Corrected AIC.  Lower is better.

        AICc = -2 ln(L) + 2k + 2k(k+1)/(n-k-1)

    `k` is the total parameter count: the error distribution's parameters plus
    any regression coefficients.  A generalized t carries three (mu, sigma,
    nu), so a fitted t on its own has k = 3.
    """
    if n - k - 1 <= 0:
        raise ValueError("AICc is undefined when n <= k + 1")
    return -2.0 * loglik + 2.0 * k + 2.0 * k * (k + 1) / (n - k - 1)


def fit_normal(x):
    """Fit a normal by its first two moments.

    The standard deviation uses the n-1 denominator, matching the class
    implementation.  Note this is the unbiased estimator rather than the
    maximum likelihood one, which would divide by n.
    """
    x = np.asarray(x, dtype=float)
    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1))
    d = stats.norm(loc=mu, scale=sigma)
    return FittedModel(error_model=d, errors=x - mu, u=d.cdf(x), n_error_params=2)


def _t_start(x):
    """Starting values from the moments: invert the excess kurtosis for nu,
    then back out the scale that reproduces the sample variance."""
    x = np.asarray(x, dtype=float)
    k = stats.kurtosis(x, bias=False)
    nu = 6.0 / k + 4.0 if k > 0 else 8.0
    nu = float(np.clip(nu, 2.5, 60.0))
    sigma = float(np.sqrt(np.var(x, ddof=1) * (nu - 2.0) / nu))
    return float(np.mean(x)), sigma, nu


def fit_general_t(x):
    """Fit mu + sigma * T_nu by maximum likelihood over all three parameters.

    sigma and nu are carried through unconstrained transforms -- log(sigma) and
    log(nu - 2) -- so the optimizer roams the whole real line while sigma stays
    positive and nu stays where the variance exists.
    """
    x = np.asarray(x, dtype=float)
    mu0, sigma0, nu0 = _t_start(x)

    def negative_loglik(theta):
        mu, log_sigma, log_nu = theta
        return -np.sum(stats.t.logpdf(x, df=2.0 + np.exp(log_nu),
                                      loc=mu, scale=np.exp(log_sigma)))

    res = optimize.minimize(
        negative_loglik, [mu0, np.log(sigma0), np.log(max(nu0 - 2.0, 0.5))],
        method="Nelder-Mead",
        options={"xatol": 1e-12, "fatol": 1e-12, "maxiter": 100000, "maxfev": 100000},
    )
    mu, log_sigma, log_nu = res.x
    sigma, nu = float(np.exp(log_sigma)), float(2.0 + np.exp(log_nu))
    d = stats.t(df=nu, loc=mu, scale=sigma)
    return FittedModel(error_model=d, errors=x - mu, u=d.cdf(x), n_error_params=3)


def fit_regression_t(y, x):
    """Regression with a generalized t error, fitted by maximum likelihood.

    The error's location is fixed at zero -- the intercept in `beta` carries
    the level, so leaving both free would make the model unidentified.  `beta`
    comes back as [intercept, b1, b2, ...].

    OLS supplies the starting values.  It is the right starting point and the
    wrong answer: the estimates move once the likelihood stops assuming the
    errors are normal.
    """
    y = np.asarray(y, dtype=float)
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if x.shape[0] != y.shape[0]:
        x = x.T
    design = np.column_stack([np.ones(len(y)), x])

    beta0, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid0 = y - design @ beta0
    _, sigma0, nu0 = _t_start(resid0)

    n_beta = design.shape[1]

    def negative_loglik(theta):
        log_sigma, log_nu = theta[0], theta[1]
        beta = theta[2:]
        resid = y - design @ beta
        return -np.sum(stats.t.logpdf(resid, df=2.0 + np.exp(log_nu),
                                      loc=0.0, scale=np.exp(log_sigma)))

    start = np.concatenate([[np.log(sigma0), np.log(max(nu0 - 2.0, 0.5))], beta0])
    res = optimize.minimize(
        negative_loglik, start, method="Nelder-Mead",
        options={"xatol": 1e-12, "fatol": 1e-12,
                 "maxiter": 200000, "maxfev": 200000},
    )
    # Nelder-Mead on six parameters benefits from a restart: the first simplex
    # collapses near the optimum and a second pass from there tightens it.
    res = optimize.minimize(
        negative_loglik, res.x, method="Nelder-Mead",
        options={"xatol": 1e-13, "fatol": 1e-13,
                 "maxiter": 200000, "maxfev": 200000},
    )

    sigma, nu = float(np.exp(res.x[0])), float(2.0 + np.exp(res.x[1]))
    beta = np.asarray(res.x[2:], dtype=float)
    d = stats.t(df=nu, loc=0.0, scale=sigma)
    errors = y - design @ beta
    return FittedModel(error_model=d, beta=beta, errors=errors, u=d.cdf(errors),
                       n_error_params=3)
