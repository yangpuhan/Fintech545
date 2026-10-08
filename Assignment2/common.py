"""
Shared helpers for Assignment 2.

The repository's `riskmgmt` library already covers the Week 03 machinery
(missing data covariance, exponential weights, the two PSD repairs, the
tolerant Cholesky, normal simulation) and the Week 02 fits (normal, generalized
t, AICc).  What it does not have yet is the Week 04 and Week 05 material, so
that lives here: VaR and ES on a sample, the normal closed forms, and the
Gaussian and t copulas.

Every script in this folder imports `riskmgmt` through `sys.path`, so they run
from inside this folder with no installation step.
"""

import sys
from pathlib import Path

import numpy as np
from scipy import stats

# Make the repository root importable, so `import riskmgmt` works when a
# script is run from inside Assignment2/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from riskmgmt import aicc, higham_nearest_psd, simulate_normal  # noqa: E402


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------

def four_moments(x):
    """First four moments with the bias-corrected estimators.

    Same convention as Assignment 1: variance with the n-1 divisor, skewness
    and kurtosis bias corrected, kurtosis in EXCESS form (a normal gives 0).
    scipy defaults to the biased estimators, so `bias=False` is passed.
    """
    x = np.asarray(x, dtype=float)
    return {
        "n": len(x),
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=1)),
        "skew": float(stats.skew(x, bias=False)),
        "kurt": float(stats.kurtosis(x, bias=False)),
    }


def print_moments(rows, unit=""):
    """Print a table of four_moments() results, one row per (label, x)."""
    print(f"{'':<14}{'n':>7}{'mean'+unit:>14}{'sd'+unit:>14}"
          f"{'skew':>10}{'ex. kurt':>10}")
    for label, x in rows:
        m = four_moments(x)
        print(f"{label:<14}{m['n']:>7}{m['mean']:>14.6f}{m['sd']:>14.6f}"
              f"{m['skew']:>10.4f}{m['kurt']:>10.4f}")


# ---------------------------------------------------------------------------
# VaR and ES
# ---------------------------------------------------------------------------
#
# Sign convention, stated once: both are reported as POSITIVE numbers for a
# loss, the assignment's convention.  A negative VaR means the alpha quantile
# of the P&L is still a profit.

def var_es(pnl, alpha=0.05):
    """Historical VaR and ES of a P&L sample.  Returns (VaR, ES).

    The quantile follows the class implementation (library/RiskStats.jl): sort
    the sample and average the order statistics at floor(n*alpha) and
    ceil(n*alpha).  When n*alpha is a whole number -- 500 of 10,000 at 5%, 25
    of 500 -- the two coincide and VaR is minus the (n*alpha)-th smallest P&L.
    numpy's default percentile interpolates between ranks differently, so it is
    not used here.

    ES is minus the mean of every P&L at or below that quantile, so it averages
    the n*alpha worst outcomes.
    """
    x = np.sort(np.asarray(pnl, dtype=float))
    n = len(x)
    up = int(np.ceil(n * alpha))
    dn = int(np.floor(n * alpha))
    v = 0.5 * (x[up - 1] + x[dn - 1])          # 1-based ranks -> 0-based index
    return float(-v), float(-x[x <= v].mean())


def normal_var_es(mu, sigma, alpha=0.05):
    """VaR and ES of a N(mu, sigma^2) P&L, both as positive losses.

        VaR = -(mu + sigma * z_alpha)
        ES  = -mu + sigma * phi(z_alpha) / alpha          (Week 05)
    """
    z = stats.norm.ppf(alpha)
    return float(-(mu + sigma * z)), float(-mu + sigma * stats.norm.pdf(z) / alpha)


# ---------------------------------------------------------------------------
# Copulas (Problem 4)
# ---------------------------------------------------------------------------

def pseudo_uniforms(x):
    """Ranks scaled into (0, 1) by rank / (n + 1).

    Dividing by n+1 rather than n keeps the largest observation off 1.0, where
    every quantile function returns infinity.
    """
    x = np.asarray(x, dtype=float)
    return stats.rankdata(x, axis=0) / (x.shape[0] + 1)


def kendall_correlation(u):
    """R from Kendall's tau, rho = sin(pi * tau / 2), checked for PSD.

    The relation is exact for every elliptical copula and nu does not appear in
    it, so the same R is handed to the Gaussian and the t copula and the
    likelihood comparison is about the copula family alone (Week 05).  tau is a
    rank statistic, so computing it on the uniforms or on the raw returns gives
    the same matrix.

    The tau matrix is PSD on complete data but the entrywise sine is not
    guaranteed to keep it so.  Returns (R, smallest eigenvalue before repair,
    tau matrix); R is repaired with Higham only when that eigenvalue is
    negative.
    """
    u = np.asarray(u, dtype=float)
    n = u.shape[1]
    tau = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            tau[i, j] = tau[j, i] = stats.kendalltau(u[:, i], u[:, j])[0]
    r = np.sin(np.pi * tau / 2.0)
    np.fill_diagonal(r, 1.0)
    min_eig = float(np.min(np.linalg.eigvalsh(r)))
    if min_eig < -1e-8:
        r = higham_nearest_psd(r)
    return r, min_eig, tau


def gaussian_copula_ll(u, r, per_obs=False):
    """Gaussian copula log likelihood.

    ln c = ln f_R(z) - sum_i ln phi(z_i), z_i = Phi^-1(u_i).  Written as a
    difference of log densities rather than as a ratio of densities, so
    nothing is ever evaluated on the density scale where it could underflow
    (Week 05).  `per_obs=True` returns one value per day, for Problem 4 (i).
    """
    z = stats.norm.ppf(u)
    joint = stats.multivariate_normal(np.zeros(r.shape[0]), r).logpdf(z)
    each = joint - stats.norm.logpdf(z).sum(axis=1)
    return each if per_obs else float(each.sum())


def t_copula_ll(u, r, nu, per_obs=False):
    """t copula log likelihood at degrees of freedom `nu`.

    Same construction as the Gaussian, with the t quantile going in and the
    multivariate t density coming out.  The margins in the denominator carry
    the COPULA's nu, not any nu fitted to a margin.
    """
    y = stats.t.ppf(u, nu)
    joint = stats.multivariate_t(np.zeros(r.shape[0]), r, df=nu).logpdf(y)
    each = joint - stats.t.logpdf(y, nu).sum(axis=1)
    return each if per_obs else float(each.sum())


def profile_nu(loglik, lo=0.01, hi=0.49, n_grid=200):
    """Maximize a log likelihood over nu by grid search on theta = 1/nu.

    The Week 05 recipe and the class library's profile_nu: theta in
    [0.01, 0.49] puts nu in (2.04, 100], a 200 point grid, then a second 200
    point grid between the neighbours of the first maximum.  Searching in
    1/nu puts the resolution where nu = 4 against 5 lives, rather than where
    nu = 40 against 50 does.  Returns (nu_hat, loglik at nu_hat).
    """
    thetas = np.linspace(lo, hi, n_grid)
    lls = np.array([loglik(1.0 / th) for th in thetas])
    i = int(np.argmax(lls))
    fine = np.linspace(thetas[max(i - 1, 0)], thetas[min(i + 1, n_grid - 1)], n_grid)
    fine_lls = np.array([loglik(1.0 / th) for th in fine])
    j = int(np.argmax(fine_lls))
    return float(1.0 / fine[j]), float(fine_lls[j])


def copula_criteria(ll, k, m):
    """(AICc, BIC) for a copula with k free parameters on m observations.

    The margins are fitted first and frozen, so only the copula's own
    parameters are counted: k = 0 for the Gaussian (R is held at the Kendall
    estimate for both models) and k = 1 for the t (nu).
    """
    return aicc(ll, k, m), k * np.log(m) - 2.0 * ll


def simulate_copula_uniforms(r, n_sim, nu=None, seed=1234):
    """Draw uniforms from a Gaussian copula (nu=None) or a t copula.

    The correlated normals come from riskmgmt.simulate_normal (a Cholesky of
    R).  The t copula scales each row by sqrt(nu / chi2_nu) -- one draw per
    ROW, the common shock that puts dependence in the joint tail (Week 05) --
    and maps back to uniforms through the t CDF at the copula's nu.

    Both copulas are built from the same seed, so they share the same normal
    draws.  The difference between their risk numbers is then the copula, not
    a different set of random numbers.
    """
    z = simulate_normal(n_sim, r, seed=seed)
    if nu is None:
        return stats.norm.cdf(z)
    rng = np.random.default_rng(seed + 1)
    w = nu / rng.chisquare(nu, size=n_sim)
    return stats.t.cdf(np.sqrt(w)[:, None] * z, nu)


def tail_dependence_t(rho, nu):
    """Lower (= upper) tail dependence of a t copula.

        lambda = 2 t_{nu+1}( -sqrt((nu+1)(1-rho)/(1+rho)) )

    Zero for the Gaussian copula at every rho < 1.
    """
    return float(2.0 * stats.t.cdf(-np.sqrt((nu + 1.0) * (1.0 - rho) / (1.0 + rho)),
                                   nu + 1.0))


def joint_tail_counts(u, q=0.025):
    """For every pair of columns, count the rows where both sit at or below
    their own q quantile and the rows where both sit at or above 1 - q.
    Returns {(i, j): (lower count, upper count)}."""
    u = np.asarray(u, dtype=float)
    out = {}
    for i in range(u.shape[1]):
        for j in range(i + 1, u.shape[1]):
            lo = int(np.sum((u[:, i] <= q) & (u[:, j] <= q)))
            hi = int(np.sum((u[:, i] >= 1 - q) & (u[:, j] >= 1 - q)))
            out[(i, j)] = (lo, hi)
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def rule(title=""):
    """Section divider, so the console output reads as a report."""
    if title:
        print("\n" + "=" * 78)
        print(title)
        print("=" * 78)
    else:
        print("-" * 78)
