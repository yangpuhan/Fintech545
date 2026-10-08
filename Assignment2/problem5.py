"""
Problem 5 -- Model Based Simulation and Residual Correlation.

Regresses A and B on the market in problem5.csv, simulates 100,000 days of the
market and the two regression errors -- once with the full residual covariance
and once with its off diagonal zeroed -- and prices two portfolios:
P1 = long A, long B and P2 = long A, short B, $1,000,000 a side.  Then a delta
normal VaR straight from the sample covariance of the two stocks.

    python3 problem5.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from common import rule, var_es
from riskmgmt import return_calculate, simulate_normal

FIGDIR = "figures"
N_SIM = 100_000
SEED = 545
ALPHA = 0.05
PORTFOLIOS = {"P1": np.array([1e6, 1e6]), "P2": np.array([1e6, -1e6])}


def ols(y, x):
    """OLS of y on [1, x].  Returns (alpha, beta, residuals)."""
    design = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coef[0]), float(coef[1]), y - design @ coef


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    prices = pd.read_csv("problem5.csv")
    rets = return_calculate(prices, method="DISCRETE", date_column="Day")
    mkt, a, b = (rets[c].to_numpy() for c in ("MKT", "A", "B"))
    n = len(mkt)

    # ---- (b) the two regressions -------------------------------------------
    rule("PROBLEM 5 (b) -- r_i = alpha_i + beta_i r_MKT + e_i")
    alpha_a, beta_a, e_a = ols(a, mkt)
    alpha_b, beta_b, e_b = ols(b, mkt)
    print(f"n = {n} daily arithmetic returns\n")
    print(f"{'':<4}{'alpha':>12}{'beta':>10}{'resid sd (n-2)':>16}{'resid sd (n-1)':>16}"
          f"{'R^2':>8}")
    for name, al, be, e, y in (("A", alpha_a, beta_a, e_a, a), ("B", alpha_b, beta_b, e_b, b)):
        print(f"{name:<4}{al:>12.6f}{be:>10.4f}{np.sqrt(e @ e / (n - 2)):>16.6f}"
              f"{np.std(e, ddof=1):>16.6f}{1 - e.var() / y.var():>8.4f}")
    rho_e = float(np.corrcoef(e_a, e_b)[0, 1])
    print(f"\ncorrelation of the residuals: {rho_e:.4f}")
    print(f"market sd {np.std(mkt, ddof=1):.6f}, mean {mkt.mean():.6f}")
    print(f"in-sample cov(MKT, e_A) = {np.cov(mkt, e_a)[0, 1]:.2e}, "
          f"cov(MKT, e_B) = {np.cov(mkt, e_b)[0, 1]:.2e}  (zero by construction)")
    print(f"raw correlation of A and B returns: {np.corrcoef(a, b)[0, 1]:.4f}")

    # ---- (c) the model based simulation ------------------------------------
    rule("PROBLEM 5 (c) -- 100,000 simulated days, two residual assumptions")
    # [MKT, e_A, e_B] ~ N(0, Sigma), Sigma block diagonal: the market is
    # uncorrelated with each error by OLS assumption 3 (Week 05).  All
    # covariances use the n-1 divisor, the same as the sample covariance in (d),
    # so the two methods are on the same footing.  Expected returns are zero by
    # the problem's instruction: the alphas and the market mean are dropped.
    var_m = float(np.var(mkt, ddof=1))
    sigma_e = np.cov(np.column_stack([e_a, e_b]), rowvar=False, ddof=1)
    betas = np.array([beta_a, beta_b])
    results = {}
    for label, se in (("full residual covariance", sigma_e),
                      ("residual off diagonal = 0", np.diag(np.diag(sigma_e)))):
        sigma = np.zeros((3, 3))
        sigma[0, 0] = var_m
        sigma[1:, 1:] = se
        # Same seed for both runs: the market and the error draws come from the
        # same standard normals, so the difference between the two runs is the
        # assumption and not the luck of the draw.
        draws = simulate_normal(N_SIM, sigma, seed=SEED)
        r_stock = draws[:, [0]] * betas[None, :] + draws[:, 1:]
        results[label] = {}
        for p, h in PORTFOLIOS.items():
            results[label][p] = var_es(r_stock @ h, ALPHA)
        print(f"{label}:")
        print(f"  simulated corr(e_A, e_B) = {np.corrcoef(draws[:, 1], draws[:, 2])[0, 1]:+.4f}")
        for p in PORTFOLIOS:
            v, e = results[label][p]
            print(f"  {p}: VaR {v:>10,.0f}   ES {e:>10,.0f}")

    # ---- (d) delta normal -----------------------------------------------------
    rule("PROBLEM 5 (d) -- delta normal VaR from the sample covariance of A, B")
    cov_ab = np.cov(np.column_stack([a, b]), rowvar=False, ddof=1)
    z = -stats.norm.ppf(ALPHA)
    dn = {}
    for p, h in PORTFOLIOS.items():
        dn[p] = z * np.sqrt(h @ cov_ab @ h)
        print(f"{p}: sd {np.sqrt(h @ cov_ab @ h):>10,.2f}   VaR {dn[p]:>10,.0f}")

    # ---- the comparison in closed form -----------------------------------------
    rule("PROBLEM 5 (e), (f) -- the same VaRs in closed form")
    # Model implied covariance of the two stocks: beta beta' var_m + Sigma_e.
    print(f"{'':<28}{'P1':>12}{'P2':>12}")
    closed = {}
    for label, se in (("full residual covariance", sigma_e),
                      ("residual off diagonal = 0", np.diag(np.diag(sigma_e)))):
        c = np.outer(betas, betas) * var_m + se
        closed[label] = {p: z * np.sqrt(h @ c @ h) for p, h in PORTFOLIOS.items()}
        print(f"{label:<28}" + "".join(f"{closed[label][p]:>12,.0f}" for p in PORTFOLIOS))
    print(f"{'delta normal, sample cov':<28}" + "".join(f"{dn[p]:>12,.0f}" for p in PORTFOLIOS))
    print("\nsimulated (from (c)):")
    for label in results:
        print(f"{label:<28}" + "".join(f"{results[label][p][0]:>12,.0f}" for p in PORTFOLIOS))

    model_cov = np.outer(betas, betas) * var_m + sigma_e
    print(f"\nmax |model implied cov - sample cov| = "
          f"{np.max(np.abs(model_cov - cov_ab)):.2e}  (sample cov entries ~ "
          f"{np.abs(cov_ab).mean():.1e})")

    print("\nshortcut against the full model, simulated VaR:")
    full, diag = results["full residual covariance"], results["residual off diagonal = 0"]
    for p in PORTFOLIOS:
        print(f"  {p}: {diag[p][0]:,.0f} vs {full[p][0]:,.0f}  "
              f"({100 * (diag[p][0] / full[p][0] - 1):+.1f}%)")

    # Decompose each portfolio's daily variance (in $^2) into the market term,
    # the two residual variances, and the residual covariance term.
    print("\nvariance decomposition, $^2 per day:")
    print(f"{'':<6}{'market':>14}{'resid var':>14}{'resid cov':>14}{'total':>14}"
          f"{'cov share':>11}")
    for p, h in PORTFOLIOS.items():
        mkt_term = (h @ betas) ** 2 * var_m
        var_term = h[0] ** 2 * sigma_e[0, 0] + h[1] ** 2 * sigma_e[1, 1]
        cov_term = 2 * h[0] * h[1] * sigma_e[0, 1]
        total = mkt_term + var_term + cov_term
        print(f"{p:<6}{mkt_term:>14,.0f}{var_term:>14,.0f}{cov_term:>+14,.0f}"
              f"{total:>14,.0f}{cov_term / total:>+11.1%}")
    print(f"\nnet market beta: P1 {beta_a + beta_b:.4f}, P2 {beta_a - beta_b:.4f}")

    # ---- figure ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.scatter(e_a, e_b, s=8, color="#14616e", alpha=0.6, edgecolor="none")
    ax.set_xlabel("residual of A")
    ax.set_ylabel("residual of B")
    ax.set_title(f"Regression residuals, correlation {rho_e:.2f}")
    ax.axhline(0, color="#999999", lw=0.5)
    ax.axvline(0, color="#999999", lw=0.5)

    ax = axes[1]
    labels = ["P1\nlong A, long B", "P2\nlong A, short B"]
    xpos = np.arange(2)
    width = 0.26
    for k, (label, vals, colour) in enumerate((
            ("full residual covariance", [full[p][0] for p in PORTFOLIOS], "#14616e"),
            ("residual off diagonal = 0", [diag[p][0] for p in PORTFOLIOS], "#d08a3a"),
            ("delta normal, sample cov", [dn[p] for p in PORTFOLIOS], "#8a3a2b"))):
        bars = ax.bar(xpos + (k - 1) * width, vals, width, color=colour, label=label)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 600, f"{v / 1000:.1f}k",
                    ha="center", fontsize=7)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("one day 5% VaR ($)")
    ax.set_title("VaR under each assumption")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    path = os.path.join(FIGDIR, "problem5.png")
    fig.savefig(path, dpi=160)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
