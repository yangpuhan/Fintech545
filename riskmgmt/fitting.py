"""Fitting distributions and regressions by maximum likelihood.

The generalized t used throughout is the location-scale family

    X = mu + sigma * T_nu

so `sigma` is a SCALE, not a standard deviation.  The standard deviation is
sigma * sqrt(nu / (nu - 2)) and exists only for nu > 2.  Mixing the two up is
the easiest way to get a wrong answer here that still looks plausible.

Covers functional tests 7.1 through 7.6.
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


# ---------------------------------------------------------------------------
# Normal inverse Gaussian (tests 7.5, 7.6)
# ---------------------------------------------------------------------------
#
# Two parameterizations are in play and mixing them up is the main hazard here.
#
#   class convention   (mu, alpha, beta, delta)   alpha = tail heaviness,
#                                                 beta  = asymmetry,
#                                                 delta = scale,
#                                                 mu    = location
#   scipy convention   norminvgauss(a, b, loc, scale) with
#                                                 a = alpha*delta,
#                                                 b = beta*delta,
#                                                 loc = mu, scale = delta
#
# Everything below reports the class convention and converts on the way in and
# out of scipy.

def nig_to_scipy(mu, alpha, beta, delta):
    """(mu, alpha, beta, delta) -> a frozen scipy norminvgauss."""
    return stats.norminvgauss(a=alpha * delta, b=beta * delta, loc=mu, scale=delta)


def nig_from_scipy(a, b, loc, scale):
    """scipy's (a, b, loc, scale) -> (mu, alpha, beta, delta)."""
    delta = scale
    return loc, a / delta, b / delta, delta


def fit_nig_moments(x):
    """Fit a NIG by the method of moments: closed form, no optimizer.

    Inverts the four NIG moment formulas from Week 01,

        mean      = mu + delta*beta/gamma
        variance  = delta*alpha^2/gamma^3
        skewness  = 3*beta/(alpha*sqrt(delta*gamma))
        ex. kurt. = 3/(delta*gamma) * (1 + 4*beta^2/alpha^2)

    with gamma = sqrt(alpha^2 - beta^2).  Writing rho = beta/alpha and
    D = delta*gamma, the third and fourth moments alone pin down rho and D,
    and the first two then give alpha and mu.

    Moment conventions, which have to be stated because they are mixed and the
    answer moves if they are not matched: the variance uses the n-1 divisor,
    while the skewness and excess kurtosis use the BIASED estimators.  That is
    what the class implementation does (Julia's `var` is unbiased, StatsBase's
    `skewness` and `kurtosis` are not), and it reproduces the expected output
    to 1e-15.  Using unbiased third and fourth moments instead moves the answer
    by about 1%.

    Not every sample admits a NIG.  The family requires excess kurtosis above
    (5/3)*skew^2, which is a strictly stronger condition than the universal
    kurtosis >= skew^2 + 1, so a sample can be perfectly legitimate and still
    have no NIG that matches its moments.
    """
    x = np.asarray(x, dtype=float)
    m = float(np.mean(x))
    v = float(np.var(x, ddof=1))
    s = float(stats.skew(x, bias=True))
    k = float(stats.kurtosis(x, bias=True))          # excess

    if k <= 0:
        raise ValueError(
            f"the NIG method of moments needs positive excess kurtosis, got {k:.4f}")
    t = s * s / k
    if t >= 3.0 / 5.0:
        raise ValueError(
            "sample is outside the NIG region: excess kurtosis must exceed "
            f"(5/3)*skew^2 = {5/3*s*s:.4f}, got {k:.4f}")

    rho2 = t / (3.0 - 4.0 * t)
    rho = np.sign(s) * np.sqrt(rho2)
    d_gamma = 3.0 * (1.0 + 4.0 * rho2) / k           # delta * gamma

    alpha = float(np.sqrt(d_gamma / (v * (1.0 - rho2) ** 2)))
    beta = float(rho * alpha)
    gamma = alpha * np.sqrt(1.0 - rho2)
    delta = float(d_gamma / gamma)
    mu = float(m - delta * beta / gamma)

    d = nig_to_scipy(mu, alpha, beta, delta)
    return FittedModel(error_model=d, errors=x - mu, u=d.cdf(x), n_error_params=4)


def fit_nig_mle(x):
    """Fit a NIG by maximum likelihood.

    Delegates to `scipy.stats.norminvgauss.fit`, which is a numerical
    optimizer rather than a closed form, so the trailing digits can move
    between scipy versions.  Compare on a relative tolerance, not for an exact
    match.
    """
    x = np.asarray(x, dtype=float)
    a, b, loc, scale = stats.norminvgauss.fit(x)
    mu, alpha, beta, delta = nig_from_scipy(a, b, loc, scale)
    d = nig_to_scipy(mu, alpha, beta, delta)
    return FittedModel(error_model=d, errors=x - mu, u=d.cdf(x), n_error_params=4)


def nig_params(fitted):
    """Pull (mu, alpha, beta, delta) back out of a fitted NIG, in the class
    convention, whichever way it was fitted."""
    kw = fitted.error_model.kwds
    return np.array(nig_from_scipy(kw["a"], kw["b"], kw["loc"], kw["scale"]))
