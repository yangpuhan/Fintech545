"""
Problem 4 -- Conditional Distributions.

Uses the Week 02 partitioned result to write down the conditional mean and
conditional variance of x2 given x1, plots the conditional expectation with a
95% band, and measures the band's coverage overall and in three buckets of
distance from the mean of x1.

    python3 problem4.py

Block notation.  The Week 02 statement partitions X into x_1 and x_2 and
conditions on x_2 = a, giving the distribution of x_1.  This problem conditions
the other way round, on x1, so the blocks map as:

    conditioning variable  x1  ->  Week 02's x_2  ->  Sigma_22 = var(x1)
    variable of interest   x2  ->  Week 02's x_1  ->  Sigma_11 = var(x2)
    cross block                                   ->  Sigma_12 = cov(x2, x1)

Below, the sample covariance matrix is stored in data order, so C[0,0] is
var(x1) and C[1,1] is var(x2).  Every formula is written out in terms of those
entries so the mapping is visible rather than implied.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import gammaln

from common import rule

FIGDIR = "figures"


def fit_bivariate_t(X):
    """Maximum likelihood fit of a bivariate t to an n-by-2 sample.

    Returns (nu, scale_matrix).  The scale matrix is NOT the covariance; for
    nu > 2 the covariance is nu/(nu-2) times the scale, and that conversion is
    done by the caller so the distinction stays visible.

    The scale matrix is parameterized through its Cholesky factor with the
    diagonal carried in logs, which keeps it positive definite without needing
    to reject proposals, and nu is carried as log(nu - 2) so it stays in the
    range where the covariance exists.
    """
    X = np.asarray(X, dtype=float)
    n, k = X.shape

    def negative_loglik(theta):
        nu = 2.0 + np.exp(theta[0])
        mu = theta[1:3]
        L = np.array([[np.exp(theta[3]), 0.0],
                      [theta[4], np.exp(theta[5])]])
        S = L @ L.T
        d = X - mu
        m = np.einsum("ij,jk,ik->i", d, np.linalg.inv(S), d)   # Mahalanobis
        const = (gammaln((nu + k) / 2) - gammaln(nu / 2)
                 - 0.5 * k * np.log(nu * np.pi)
                 - 0.5 * np.log(np.linalg.det(S)))
        return -np.sum(const - 0.5 * (nu + k) * np.log1p(m / nu))

    # Start from the sample covariance shrunk toward a t with moderate tails:
    # at nu = 6 the covariance is 1.5 times the scale, so scale ~ 2/3 of cov.
    L0 = np.linalg.cholesky(np.cov(X.T, ddof=1) * (6.0 - 2.0) / 6.0)
    start = [np.log(4.0), X[:, 0].mean(), X[:, 1].mean(),
             np.log(L0[0, 0]), L0[1, 0], np.log(L0[1, 1])]
    res = optimize.minimize(negative_loglik, start, method="Nelder-Mead",
                            options={"xatol": 1e-10, "fatol": 1e-10,
                                     "maxiter": 200000, "maxfev": 200000})
    nu = 2.0 + np.exp(res.x[0])
    L = np.array([[np.exp(res.x[3]), 0.0], [res.x[4], np.exp(res.x[5])]])
    return nu, L @ L.T


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    d = pd.read_csv("problem4.csv")
    x1, x2 = d["x1"].to_numpy(), d["x2"].to_numpy()
    n = len(x1)

    # ---- (predict) the sample covariance matrix and nothing else --------
    rule("PROBLEM 4 -- sample covariance matrix")
    C = np.cov(x1, x2, ddof=1)          # C[0,0]=var(x1), C[1,1]=var(x2)
    print(f"n = {n}")
    print(f"means: x1 = {x1.mean():+.6f}, x2 = {x2.mean():+.6f}")
    print("covariance (rows and columns ordered x1, x2):")
    print(np.array2string(C, precision=6, suppress_small=False))
    rho = C[0, 1] / np.sqrt(C[0, 0] * C[1, 1])
    print(f"correlation = {rho:.6f}")

    # ---- (a) conditional variance and the factor ------------------------
    rule("PROBLEM 4 (a) -- conditional variance of x2 given x1")
    #   Sigma_bar = Sigma_11 - Sigma_12 Sigma_22^{-1} Sigma_21
    # with x2 in the "1" block and x1 in the "2" block, which in data order is
    #   var(x2) - cov(x1,x2)^2 / var(x1)
    cond_var = C[1, 1] - C[0, 1] ** 2 / C[0, 0]
    factor = cond_var / C[1, 1]
    print("Sigma_bar = Sigma_11 - Sigma_12 Sigma_22^-1 Sigma_21")
    print("          = var(x2) - cov(x1,x2)^2 / var(x1)")
    print(f"          = {C[1,1]:.6f} - {C[0,1]:.6f}^2 / {C[0,0]:.6f}")
    print(f"          = {cond_var:.6f}")
    print(f"conditional sd = {np.sqrt(cond_var):.6f}"
          f"   (unconditional sd of x2 = {np.sqrt(C[1,1]):.6f})")
    print("\nfactor by which the variance falls:")
    print("  Sigma_bar / Sigma_11 = 1 - Sigma_12^2 / (Sigma_11 Sigma_22) = 1 - rho^2")
    print(f"  = 1 - {rho:.6f}^2 = {factor:.6f}")
    print(f"the sd falls by sqrt of that = {np.sqrt(factor):.6f}")

    # ---- (b) does the factor depend on the value observed? --------------
    rule("PROBLEM 4 (b) -- does that factor depend on the observed x1?")
    print("Sigma_bar = Sigma_11 - Sigma_12 Sigma_22^-1 Sigma_21 contains no 'a'.")
    print("Under the multivariate normal the conditional variance is therefore")
    print("constant in the conditioning value.  The term that settles it is the")
    print("absence of 'a': 'a' appears only in mu_bar, never in Sigma_bar.")

    # ---- (c) conditional mean, the band, and the plot -------------------
    rule("PROBLEM 4 (c) -- conditional mean of x2 given x1")
    #   mu_bar = mu_1 + Sigma_12 Sigma_22^{-1} (a - mu_2)
    #          = mean(x2) + cov(x1,x2)/var(x1) * (x1 - mean(x1))
    slope = C[0, 1] / C[0, 0]
    intercept = x2.mean() - slope * x1.mean()
    print("mu_bar = mu_1 + Sigma_12 Sigma_22^-1 (a - mu_2)")
    print("       = mean(x2) + [cov(x1,x2)/var(x1)] (x1 - mean(x1))")
    print(f"coefficient on x1: Sigma_12 Sigma_22^-1 = "
          f"{C[0,1]:.6f} / {C[0,0]:.6f} = {slope:.6f}")
    print(f"implied intercept                       = {intercept:.6f}")
    # Cross-check against an OLS fit: the two must agree to numerical error.
    ols_slope, ols_intercept = np.polyfit(x1, x2, 1)
    print(f"\ncheck against OLS of x2 on x1: slope {ols_slope:.6f}, "
          f"intercept {ols_intercept:.6f}")
    print(f"difference in slope: {abs(ols_slope - slope):.3e}")
    print("In regression terms the coefficient IS the OLS slope of x2 on x1:")
    print("cov(x,y)/var(x) is the same object arriving from a different door.")

    fitted = intercept + slope * x1
    half_width = 1.96 * np.sqrt(cond_var)
    print(f"\n95% band: conditional mean +/- 1.96*{np.sqrt(cond_var):.6f} "
          f"= +/- {half_width:.6f}")
    print("The half width does not depend on x1, so the band has constant width.")

    # ---- (d) overall coverage -------------------------------------------
    inside = np.abs(x2 - fitted) <= half_width
    rule("PROBLEM 4 (d) -- coverage of the band")
    print(f"observations inside the band = {int(inside.sum())} of {n}")
    print(f"fraction                     = {inside.mean():.4f}")
    print(f"nominal                      = 0.9500")
    se = np.sqrt(0.95 * 0.05 / n)
    print(f"binomial se at the nominal level = {se:.4f}  "
          f"-> shortfall is {(0.95 - inside.mean())/se:.2f} se")

    # ---- (e) coverage by distance of x1 from its mean --------------------
    rule("PROBLEM 4 (e) -- coverage by how far x1 sits from its mean")
    z = np.abs(x1 - x1.mean()) / x1.std(ddof=1)
    buckets = [("within 1 sd", (z < 1)),
               ("between 1 and 2 sd", (z >= 1) & (z < 2)),
               ("beyond 2 sd", (z >= 2))]
    print(f"{'bucket':>22}{'n':>7}{'coverage':>11}{'se':>9}"
          f"{'resid sd':>11}{'resid ex.kurt':>15}")
    resid = x2 - fitted
    for label, mask in buckets:
        m = int(mask.sum())
        cov = float(inside[mask].mean())
        bse = np.sqrt(cov * (1 - cov) / m)
        print(f"{label:>22}{m:>7}{cov:>11.4f}{bse:>9.4f}"
              f"{resid[mask].std(ddof=1):>11.4f}"
              f"{stats.kurtosis(resid[mask], bias=False):>15.3f}")
    print(f"\npooled conditional sd assumed by the band = {np.sqrt(cond_var):.4f}")

    # ---- (f) what failed -------------------------------------------------
    rule("PROBLEM 4 (f) -- diagnostics on the residual")
    print(f"residual skewness        = {stats.skew(resid, bias=False):+.4f}")
    print(f"residual excess kurtosis = {stats.kurtosis(resid, bias=False):+.4f}")
    jb, jbp = stats.jarque_bera(resid)[:2]
    print(f"Jarque-Bera              = {jb:.2f}, p = {jbp:.3e}")
    print(f"\nmarginal x1: skew {stats.skew(x1, bias=False):+.3f}, "
          f"excess kurtosis {stats.kurtosis(x1, bias=False):+.3f}")
    print(f"marginal x2: skew {stats.skew(x2, bias=False):+.3f}, "
          f"excess kurtosis {stats.kurtosis(x2, bias=False):+.3f}")
    print("\nA joint normal has normal margins.  These do not, and the residual")
    print("spread grows with |x1|, which a joint normal forbids.")

    # ---- (f) an alternative that reproduces the pattern -----------------
    # A bivariate t is the natural candidate.  It is elliptical, so the
    # conditional mean stays exactly linear with the same coefficient, but its
    # conditional variance scales with (nu + d1)/(nu + 1), where d1 is the
    # squared standardized distance of the conditioning value from its mean.
    # That is a conditional variance that DOES depend on the observed x1, which
    # is the property the normal forbids and the data displays.  Fitting it
    # turns "something like a t" into a number that can be checked.
    rule("PROBLEM 4 (f) -- a bivariate t fitted to the pair")
    nu_hat, S_hat = fit_bivariate_t(np.column_stack([x1, x2]))
    implied_cov = S_hat * nu_hat / (nu_hat - 2.0)
    print(f"fitted degrees of freedom nu = {nu_hat:.3f}")
    print("fitted scale matrix:")
    print(np.array2string(S_hat, precision=6))
    print("implied covariance = nu/(nu-2) * scale:")
    print(np.array2string(implied_cov, precision=6))
    print("sample covariance, for comparison:")
    print(np.array2string(C, precision=6))
    print(f"implied correlation {S_hat[0,1]/np.sqrt(S_hat[0,0]*S_hat[1,1]):.6f}"
          f"  against sample {rho:.6f}")

    print("\nconditional sd multiplier sqrt((nu + d1)/(nu + 1)) at the fitted nu,")
    print("against the residual sd actually observed in each bucket:")
    pooled = np.sqrt(cond_var)
    print(f"{'bucket':>22}{'typical |z|':>13}{'predicted sd':>14}{'observed sd':>13}")
    for (label, mask), zz in zip(buckets, (0.5, 1.4, 2.5)):
        mult = np.sqrt((nu_hat + zz ** 2) / (nu_hat + 1.0))
        print(f"{label:>22}{zz:>13.1f}{pooled*mult:>14.4f}"
              f"{resid[mask].std(ddof=1):>13.4f}")

    # ---- figure ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    order = np.argsort(x1)
    ax.scatter(x1, x2, s=9, alpha=0.4, color="#14616e", edgecolor="none",
               label="observations")
    ax.plot(x1[order], fitted[order], color="#8a3a2b", lw=1.6,
            label="conditional expectation")
    ax.fill_between(x1[order], (fitted - half_width)[order],
                    (fitted + half_width)[order], color="#8a3a2b", alpha=0.13,
                    label="95% band, constant width")
    outside = ~inside
    ax.scatter(x1[outside], x2[outside], s=11, facecolor="none",
               edgecolor="#8a3a2b", linewidth=0.6, label="outside the band")
    ax.set_title(f"x2 given x1, coverage {inside.mean():.3f}")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(fontsize=7, frameon=False, loc="upper left")

    ax = axes[1]
    labels = [b[0] for b in buckets]
    covs = [float(inside[b[1]].mean()) for b in buckets]
    ns = [int(b[1].sum()) for b in buckets]
    errs = [np.sqrt(c * (1 - c) / m) for c, m in zip(covs, ns)]
    ax.bar(range(3), covs, yerr=errs, capsize=4, color="#c9d6dd",
           edgecolor="#14616e", linewidth=1.0)
    ax.axhline(0.95, color="#8a3a2b", lw=1.2, ls="--", label="nominal 0.95")
    for i, (c, m) in enumerate(zip(covs, ns)):
        ax.text(i, c + 0.012, f"{c:.3f}\nn={m}", ha="center", fontsize=7.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.75, 1.02)
    ax.set_ylabel("coverage")
    ax.set_title("Coverage by distance of x1 from its mean")
    ax.legend(fontsize=7, frameon=False, loc="lower right")

    fig.tight_layout()
    path = os.path.join(FIGDIR, "problem4.png")
    fig.savefig(path, dpi=160)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
