"""
Problem 2 -- A Regression Whose Errors Are Not Normal.

Fits three models to problem2.csv: OLS with its standard errors, a regression
with a normal error by maximum likelihood, and a regression with a Student's t
error by maximum likelihood.  Chooses between them with AICc, then compares the
95% and 99.5% quantiles of the two fitted error distributions.

    python3 problem2.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan

from common import aicc, fit_normal_mle, fit_ols, fit_t_mle, rule

FIGDIR = "figures"


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    d = pd.read_csv("problem2.csv")
    x, y = d["x"].to_numpy(), d["y"].to_numpy()
    n = len(y)

    # ---- (fit) the three models ----------------------------------------
    rule("PROBLEM 2 -- three models")
    ols = fit_ols(x, y)
    nml = fit_normal_mle(x, y)
    tml = fit_t_mle(x, y)

    print(f"n = {n}\n")
    print("OLS")
    print(f"  alpha = {ols['alpha']:.4f}  (se {ols['se_alpha']:.4f})")
    print(f"  beta  = {ols['beta']:.4f}  (se {ols['se_beta']:.4f})")
    print(f"  s (unbiased residual sd, divisor n-p) = {ols['s']:.4f}")
    print(f"  sigma read as an MLE (divisor n)      = {ols['sigma_mle']:.4f}")
    print(f"  log likelihood = {ols['loglik']:.4f}   k = {ols['k']}   "
          f"AICc = {aicc(ols['loglik'], ols['k'], n):.4f}")

    print("\nMLE, normal error")
    print(f"  alpha = {nml['alpha']:.4f}")
    print(f"  beta  = {nml['beta']:.4f}")
    print(f"  sigma = {nml['sigma']:.4f}")
    print(f"  log likelihood = {nml['loglik']:.4f}   k = {nml['k']}   "
          f"AICc = {aicc(nml['loglik'], nml['k'], n):.4f}")

    print("\nMLE, Student's t error")
    print(f"  alpha = {tml['alpha']:.4f}")
    print(f"  beta  = {tml['beta']:.4f}")
    print(f"  sigma (SCALE, not sd) = {tml['sigma']:.4f}")
    print(f"  nu                    = {tml['nu']:.4f}")
    print(f"  implied sd = sigma*sqrt(nu/(nu-2)) = {tml['implied_sd']:.4f}")
    print(f"  log likelihood = {tml['loglik']:.4f}   k = {tml['k']}   "
          f"AICc = {aicc(tml['loglik'], tml['k'], n):.4f}")

    # ---- model selection -------------------------------------------------
    rule("PROBLEM 2 -- AICc comparison")
    print("k = p + d, with p = 2 regression parameters (intercept and slope)")
    print("and d the error-distribution parameters: 1 for the normal (sigma),")
    print("2 for the t (sigma and nu).\n")
    print(f"{'model':<26}{'loglik':>12}{'k':>4}{'AICc':>12}{'dAICc':>10}")
    rows = [("OLS / normal MLE", nml["loglik"], nml["k"]),
            ("MLE, Student's t error", tml["loglik"], tml["k"])]
    scores = [aicc(ll, k, n) for _, ll, k in rows]
    best = min(scores)
    for (name, ll, k), a in zip(rows, scores):
        print(f"{name:<26}{ll:>12.4f}{k:>4}{a:>12.4f}{a-best:>10.4f}")
    print(f"\nAICc selects: {rows[int(np.argmin(scores))][0]}")
    print(f"margin over the normal error model: {max(scores)-min(scores):.4f}")

    # ---- slope comparison -----------------------------------------------
    rule("PROBLEM 2 -- the three slope estimates")
    print(f"{'model':<26}{'beta':>10}{'diff vs OLS':>14}{'in OLS se':>12}")
    for label, b in (("OLS", ols["beta"]), ("MLE, normal error", nml["beta"]),
                     ("MLE, t error", tml["beta"])):
        diff = b - ols["beta"]
        print(f"{label:<26}{b:>10.4f}{diff:>14.4f}{diff/ols['se_beta']:>12.3f}")

    # ---- error quantiles -------------------------------------------------
    rule("PROBLEM 2 -- quantiles of the two fitted error distributions")
    print(f"{'level':>8}{'normal':>12}{'t':>12}{'wider':>10}")
    for level in (0.95, 0.975, 0.99, 0.995):
        qn = stats.norm.ppf(level, loc=0.0, scale=nml["sigma"])
        qt = stats.t.ppf(level, df=tml["nu"], loc=0.0, scale=tml["sigma"])
        print(f"{level:>8.3f}{qn:>12.4f}{qt:>12.4f}"
              f"{('normal' if qn > qt else 't'):>10}")
    # Where the two densities cross is the level at which the ordering flips.
    lo, hi = 0.95, 0.995
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        qn = stats.norm.ppf(mid, scale=nml["sigma"])
        qt = stats.t.ppf(mid, df=tml["nu"], scale=tml["sigma"])
        if qn > qt:
            lo = mid
        else:
            hi = mid
    print(f"\nthe two quantile curves cross at about level {0.5*(lo+hi):.4f}")

    # ---- residual diagnostics, to justify which assumption is violated ---
    rule("PROBLEM 2 -- residual diagnostics on the OLS fit")
    r = ols["resid"]
    X = sm.add_constant(x)
    print(f"residual skewness       = {stats.skew(r, bias=False):+.4f}")
    print(f"residual excess kurtosis= {stats.kurtosis(r, bias=False):+.4f}")
    jb_stat, jb_p = stats.jarque_bera(r)[:2]
    print(f"Jarque-Bera             = {jb_stat:.2f}, p = {jb_p:.3e}"
          "   (assumption 7, normality)")
    print(f"Durbin-Watson           = {sm.stats.durbin_watson(r):.4f}"
          "        (assumption 4, uncorrelated errors; 2 means none)")
    bp_p = het_breuschpagan(r, X)[1]
    print(f"Breusch-Pagan           = p {bp_p:.4f}"
          "        (assumption 5, constant variance)")
    print("\nOnly the normality test rejects.  The violated assumption is 7.")

    # ---- figure ----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    ax = axes[0]
    ax.scatter(x, y, s=11, color="#14616e", alpha=0.6, edgecolor="none")
    grid = np.linspace(x.min(), x.max(), 100)
    ax.plot(grid, ols["alpha"] + ols["beta"] * grid, color="#8a3a2b", lw=1.4,
            label=f"OLS: y = {ols['alpha']:.2f} + {ols['beta']:.2f}x")
    ax.set_title("y against x")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    ax.hist(r, bins=40, density=True, color="#c9d6dd", edgecolor="#8fa3ad",
            linewidth=0.4, label="OLS residuals")
    g = np.linspace(r.min(), r.max(), 500)
    ax.plot(g, stats.norm.pdf(g, 0, nml["sigma"]), color="#14616e", lw=1.5,
            label="fitted normal error")
    ax.plot(g, stats.t.pdf(g, df=tml["nu"], scale=tml["sigma"]),
            color="#8a3a2b", lw=1.5, label=f"fitted t error (nu={tml['nu']:.2f})")
    ax.set_title("Residuals and the two fitted errors")
    ax.set_xlabel("residual")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[2]
    stats.probplot(r, dist=stats.norm, plot=ax)
    ax.get_lines()[0].set(marker="o", markersize=2.5, color="#14616e",
                          alpha=0.6, linestyle="none")
    ax.get_lines()[1].set(color="#8a3a2b", lw=1.2)
    ax.set_title("Normal Q-Q of the OLS residuals")

    fig.tight_layout()
    path = os.path.join(FIGDIR, "problem2.png")
    fig.savefig(path, dpi=160)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
