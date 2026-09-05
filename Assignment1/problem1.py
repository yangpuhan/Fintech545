"""
Problem 1 -- Reading the Shape of a Sample.

Computes the first four moments of problem1.csv, fits a normal by matching the
mean and the variance, and counts how many observations fall below that
normal's 1% quantile against how many should.

    python3 problem1.py
"""

import os

import matplotlib
matplotlib.use("Agg")               # write files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from common import four_moments, moment_feasibility, rule

FIGDIR = "figures"


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    x = pd.read_csv("problem1.csv")["x"].to_numpy()

    # ---- (predict) the four moments ------------------------------------
    rule("PROBLEM 1 -- moments of the sample")
    m = four_moments(x)
    print(f"n                      = {m['n']}")
    print(f"mean                   = {m['mean']:.6f}")
    print(f"variance (n-1)         = {m['variance']:.6e}")
    print(f"standard deviation     = {m['sd']:.6f}")
    print(f"skewness (unbiased)    = {m['skew']:+.4f}   (biased: {m['skew_biased']:+.4f})")
    print(f"excess kurtosis (unb.) = {m['excess_kurtosis']:+.4f}   "
          f"(biased: {m['excess_kurtosis_biased']:+.4f})")
    print(f"min, max               = {x.min():.4f}, {x.max():.4f}")
    print(f"observations below 0   = {(x < 0).sum()} of {len(x)}")

    # Arithmetic check from Week 01: kurtosis >= skewness^2 + 1 always.
    feas = moment_feasibility(m["skew"], m["excess_kurtosis"])
    print(f"\nfeasibility check: raw kurtosis {feas['raw_kurtosis']:.4f} "
          f">= skew^2+1 = {feas['lower_bound']:.4f} -> {feas['satisfied']}")

    # ---- (fit) normal matched on mean and variance ----------------------
    rule("PROBLEM 1 -- normal fitted by matching mean and variance")
    mu, sd = m["mean"], m["sd"]
    print(f"fitted N(mu, sigma^2): mu = {mu:.6f}, sigma = {sd:.6f}")

    q01 = stats.norm.ppf(0.01, loc=mu, scale=sd)
    observed_below = int((x < q01).sum())
    expected_below = 0.01 * len(x)
    # Under the model the count is Binomial(n, 0.01); its sd calibrates how
    # surprising the observed count is.
    binom_sd = np.sqrt(len(x) * 0.01 * 0.99)

    print(f"\n1% quantile of the fitted normal = {q01:.6f}")
    print(f"observations below it            = {observed_below}")
    print(f"expected under the fitted normal = {expected_below:.0f}")
    print(f"binomial sd of that count        = {binom_sd:.2f}")
    print(f"exceedance in sd units           = "
          f"{(observed_below - expected_below) / binom_sd:+.2f}")

    # Both tails, so the direction of the error is visible rather than assumed.
    rule()
    print("both tails, observed vs expected under the fitted normal:")
    print(f"{'alpha':>8} {'lower obs':>10} {'lower exp':>10} "
          f"{'upper obs':>10} {'upper exp':>10}")
    for a in (0.005, 0.01, 0.025, 0.05):
        lo = stats.norm.ppf(a, mu, sd)
        hi = stats.norm.ppf(1 - a, mu, sd)
        print(f"{a:>8.3f} {int((x < lo).sum()):>10d} {a*len(x):>10.1f} "
              f"{int((x > hi).sum()):>10d} {a*len(x):>10.1f}")

    # ---- figure ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    ax = axes[0]
    ax.hist(x, bins=60, density=True, color="#c9d6dd", edgecolor="#8fa3ad",
            linewidth=0.4, label="sample")
    grid = np.linspace(x.min(), x.max(), 500)
    ax.plot(grid, stats.norm.pdf(grid, mu, sd), color="#14616e", lw=1.6,
            label="fitted normal")
    ax.axvline(q01, color="#8a3a2b", lw=1.2, ls="--", label="fitted 1% quantile")
    ax.set_title("Sample against the fitted normal")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    stats.probplot(x, dist=stats.norm, sparams=(mu, sd), plot=ax)
    ax.get_lines()[0].set(marker="o", markersize=2.2, color="#14616e",
                          alpha=0.55, linestyle="none")
    ax.get_lines()[1].set(color="#8a3a2b", lw=1.2)
    ax.set_title("Normal Q-Q plot")

    fig.tight_layout()
    path = os.path.join(FIGDIR, "problem1.png")
    fig.savefig(path, dpi=160)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
