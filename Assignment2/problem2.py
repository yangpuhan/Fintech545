"""
Problem 2 -- A Volatility Estimate After the Regime Changed.

Five one day VaR estimates for $1,000,000 of the stock in problem2.csv: normal
with an equally weighted sd, normal with exponentially weighted sds at
lambda = 0.97 and 0.94, a Student's t by maximum likelihood, and historical
simulation.  Then the regime diagnostics used to explain them.

    python3 problem2.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from common import four_moments, rule, var_es
from riskmgmt import ew_covar, exp_weights, fit_general_t, return_calculate

FIGDIR = "figures"
POSITION = 1_000_000.0
ALPHA = 0.05
LAMBDAS = (0.97, 0.94)

# Read off the return plot before anything is fitted: the last ~40 days swing
# visibly wider than the 460 before them.  The changepoint scan below is a
# check on this number, not the source of it.
RECENT_DAYS = 40


def ew_sd(r, lam):
    """Exponentially weighted sd through riskmgmt.ew_covar.

    ew_covar removes the WEIGHTED mean before squaring, which is the class
    implementation (functional test 2.1).  The sample mean has already been
    removed, but the weighted mean of the de-meaned series is not zero -- it
    leans on the recent days -- so the alternative, squaring around zero, is
    printed alongside for comparison.
    """
    return float(np.sqrt(ew_covar(r.reshape(-1, 1), lam)[0, 0]))


def changepoint(r, min_seg=20):
    """Single variance changepoint by maximum likelihood: the split k that
    maximizes the normal log likelihood with one variance before k and another
    after.  Means are zero (already removed)."""
    n = len(r)
    best_k, best_ll = None, -np.inf
    for k in range(min_seg, n - min_seg):
        ll = -0.5 * k * np.log(np.mean(r[:k] ** 2)) \
             - 0.5 * (n - k) * np.log(np.mean(r[k:] ** 2))
        if ll > best_ll:
            best_k, best_ll = k, ll
    return best_k


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    prices = pd.read_csv("problem2.csv")
    rets = return_calculate(prices, method="DISCRETE", date_column="Day")
    r = rets["Price"].to_numpy()
    mean_removed = float(r.mean())
    r = r - mean_removed                       # the problem says remove it
    n = len(r)

    # ---- (b) effective sample size, half life, weight on the recent days --
    rule("PROBLEM 2 (b) -- effective sample size and half life")
    print(f"n = {n} returns; sample mean removed = {mean_removed:.6f}")
    print(f"recent days judged from the plot: {RECENT_DAYS}\n")
    print(f"{'lambda':>8}{'n_eff':>10}{'(1+l)/(1-l)':>13}{'half life':>11}"
          f"{'weight on last ' + str(RECENT_DAYS):>20}")
    for lam in LAMBDAS:
        w = exp_weights(n, lam)                 # last row carries the most
        n_eff = 1.0 / np.sum(w ** 2)
        print(f"{lam:>8.2f}{n_eff:>10.2f}{(1 + lam) / (1 - lam):>13.2f}"
              f"{np.log(0.5) / np.log(lam):>11.2f}{w[-RECENT_DAYS:].sum():>20.4f}")
    print(f"{'equal':>8}{n:>10d}{'':>13}{'':>11}{RECENT_DAYS / n:>20.4f}")

    k = changepoint(r)
    print(f"\nvariance changepoint by maximum likelihood: after return {k}, "
          f"leaving {n - k} recent days")
    for lam in LAMBDAS:
        print(f"  weight on the last {n - k} at lambda {lam}: "
              f"{exp_weights(n, lam)[-(n - k):].sum():.4f}")

    # ---- (c) moments of the full sample -----------------------------------
    rule("PROBLEM 2 (c) -- first four moments, full sample")
    m = four_moments(r)
    print(f"mean {m['mean']:.2e}   sd {m['sd']:.6f}   skew {m['skew']:+.4f}   "
          f"excess kurtosis {m['kurt']:+.4f}")
    jb, jb_p = stats.jarque_bera(r)[:2]
    print(f"Jarque-Bera {jb:.2f}, p = {jb_p:.2e}")
    # Under normality the bias corrected excess kurtosis has a standard error
    # of about sqrt(24/n).
    print(f"standard error of excess kurtosis under normality ~ sqrt(24/n) = "
          f"{np.sqrt(24 / n):.4f}, so {m['kurt']:.2f} is "
          f"{m['kurt'] / np.sqrt(24 / n):.1f} standard errors from 0")

    # ---- (d) the five VaRs -------------------------------------------------
    rule("PROBLEM 2 (d) -- five one day VaR estimates, $1,000,000 position")
    z = stats.norm.ppf(ALPHA)
    sd_eq = float(np.std(r, ddof=1))
    est = {"Normal, equal weight": -z * sd_eq * POSITION}
    sds = {"Normal, equal weight": sd_eq}
    for lam in LAMBDAS:
        s = ew_sd(r, lam)
        est[f"Normal, EW lambda={lam}"] = -z * s * POSITION
        sds[f"Normal, EW lambda={lam}"] = s

    tfit = fit_general_t(r)
    nu = tfit.error_model.kwds["df"]
    mu_t = float(tfit.error_model.kwds["loc"])
    sig_t = float(tfit.error_model.kwds["scale"])
    est["Student t (MLE)"] = -tfit.error_model.ppf(ALPHA) * POSITION
    sds["Student t (MLE)"] = sig_t * np.sqrt(nu / (nu - 2))
    hist_var, _ = var_es(r * POSITION, ALPHA)
    est["Historical simulation"] = hist_var
    sds["Historical simulation"] = sd_eq

    print(f"{'method':<26}{'sd used':>12}{'VaR ($)':>14}")
    for name, v in sorted(est.items(), key=lambda kv: kv[1]):
        print(f"{name:<26}{sds[name]:>12.6f}{v:>14,.0f}")
    print("\n(the t row shows its implied sd, sigma*sqrt(nu/(nu-2)); the "
          "historical row has no sd and shows the sample's)")

    print(f"\nfitted t: mu = {mu_t:.6f}, sigma (scale) = {sig_t:.6f}, "
          f"nu = {nu:.4f}, log likelihood = {tfit.loglik(r):.4f}")
    print(f"  standardized 5% quantile: t {stats.t.ppf(ALPHA, nu) * np.sqrt((nu - 2) / nu):.4f}"
          f" vs normal {z:.4f}")
    print("\nEW sd squared around zero instead of the weighted mean:")
    for lam in LAMBDAS:
        s0 = float(np.sqrt(exp_weights(n, lam) @ r ** 2))
        print(f"  lambda {lam}: sd {s0:.6f}, VaR {-z * s0 * POSITION:,.0f}")

    # ---- (e) where the historical VaR comes from ---------------------------
    rule("PROBLEM 2 (e) -- what sets the historical 5% quantile")
    order = np.argsort(r)
    n_tail = int(round(n * ALPHA))
    print(f"the 5% quantile is the {n_tail}th worst of {n} days; of those "
          f"{n_tail}, {int(np.sum(order[:n_tail] >= n - RECENT_DAYS))} are "
          f"in the last {RECENT_DAYS}")

    # ---- (f) the two regimes ----------------------------------------------
    rule("PROBLEM 2 (f) -- standard deviation inside each regime")
    calm, recent = r[:-RECENT_DAYS], r[-RECENT_DAYS:]
    s1, s2 = float(np.std(calm, ddof=1)), float(np.std(recent, ddof=1))
    p = RECENT_DAYS / n
    print(f"first {n - RECENT_DAYS} days: sd {s1:.6f}   excess kurtosis "
          f"{stats.kurtosis(calm, bias=False):+.4f}")
    print(f"last {RECENT_DAYS} days : sd {s2:.6f}   excess kurtosis "
          f"{stats.kurtosis(recent, bias=False):+.4f}")
    print(f"ratio {s2 / s1:.3f}")
    print(f"normal VaR at each regime's sd: calm {-z * s1 * POSITION:,.0f}, "
          f"recent {-z * s2 * POSITION:,.0f}")
    # A mixture of two zero-mean normals: E[x^2] = (1-p)s1^2 + p s2^2 and
    # E[x^4] = 3((1-p)s1^4 + p s2^4).  Its excess kurtosis is positive
    # whenever s1 != s2, with no fat tail in either component.
    m2 = (1 - p) * s1 ** 2 + p * s2 ** 2
    m4 = 3 * ((1 - p) * s1 ** 4 + p * s2 ** 4)
    print(f"\nexcess kurtosis of a {1 - p:.2f}/{p:.2f} mixture of N(0,{s1:.4f}^2) "
          f"and N(0,{s2:.4f}^2) = {m4 / m2 ** 2 - 3:.4f}")
    print(f"mixture sd = {np.sqrt(m2):.6f}  (sample sd {sd_eq:.6f})")
    # The 5% quantile of that mixture, which is what historical simulation is
    # estimating if the two regime model is right.
    mix_cdf = lambda q: (1 - p) * stats.norm.cdf(q, scale=s1) + p * stats.norm.cdf(q, scale=s2)
    lo, hi = -0.1, 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if mix_cdf(mid) < ALPHA else (lo, mid)
    q_mix = 0.5 * (lo + hi)
    f_q = (1 - p) * stats.norm.pdf(q_mix, scale=s1) + p * stats.norm.pdf(q_mix, scale=s2)
    se_q = np.sqrt(ALPHA * (1 - ALPHA) / n) / f_q
    print(f"5% VaR of the mixture = {-q_mix * POSITION:,.0f}; standard error of a "
          f"5% sample quantile from n = {n}: {se_q * POSITION:,.0f}")
    print(f"historical VaR sits {(hist_var + q_mix * POSITION) / (se_q * POSITION):+.2f}"
          f" standard errors from the mixture value")
    print(f"\nfitted t's implied sd {sds['Student t (MLE)']:.6f} -- the full "
          f"sample sd, not either regime's")
    for lam in LAMBDAS:
        w = exp_weights(n, lam)
        wr = w[-RECENT_DAYS:].sum()
        print(f"two regime model's EW sd at lambda {lam}: "
              f"sqrt({1 - wr:.3f}*s1^2 + {wr:.3f}*s2^2) = "
              f"{np.sqrt((1 - wr) * s1 ** 2 + wr * s2 ** 2):.6f}")

    # ---- (g) is the gap between the two EW VaRs noise? --------------------
    rule("PROBLEM 2 (g) -- the gap between the two EW VaRs against its noise")
    v97, v94 = est["Normal, EW lambda=0.97"], est["Normal, EW lambda=0.94"]
    print(f"gap = {v94 - v97:,.0f}")
    for lam, v in ((0.97, v97), (0.94, v94)):
        n_eff = (1 + lam) / (1 - lam)
        rel = 1 / np.sqrt(2 * n_eff)
        print(f"lambda {lam}: se(sd)/sd = 1/sqrt(2 n_eff) = {rel:.4f}, "
              f"so se(VaR) ~ {rel * v:,.0f}")
    print(f"0.97 is {100 * (1 - v97 / v94):.1f}% below 0.94")
    # The part of the gap the two regime model accounts for: 0.97 keeps more
    # weight on the calm days.
    regime_sd = {lam: np.sqrt((1 - exp_weights(n, lam)[-RECENT_DAYS:].sum()) * s1 ** 2
                              + exp_weights(n, lam)[-RECENT_DAYS:].sum() * s2 ** 2)
                 for lam in LAMBDAS}
    print(f"gap the two regime model predicts: "
          f"{-z * (regime_sd[0.94] - regime_sd[0.97]) * POSITION:,.0f}")
    s0_94 = float(np.sqrt(exp_weights(n, 0.94) @ r ** 2))
    print(f"lambda 0.94 squared around zero minus around the weighted mean: "
          f"{-z * s0_94 * POSITION - v94:,.0f}")

    # ---- figure ------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.6, 1]})
    days = np.arange(1, n + 1)
    ax = axes[0]
    ax.axvspan(n - RECENT_DAYS + 0.5, n + 0.5, color="#f1dfd9", lw=0)
    ax.bar(days, r, width=1.0, color="#14616e")
    ax.axhline(0, color="#999999", lw=0.5)
    for sgn in (-1, 1):
        ax.hlines(sgn * 1.645 * s1, 1, n - RECENT_DAYS, color="#8a3a2b", lw=0.8, ls="--")
        ax.hlines(sgn * 1.645 * s2, n - RECENT_DAYS + 1, n, color="#8a3a2b", lw=0.8, ls="--")
    ax.set_ylabel("de-meaned return")
    ax.set_title(f"Daily returns; the shaded last {RECENT_DAYS} days form the second "
                 "regime (dashed: ±1.645 sd within each regime)")

    ax = axes[1]
    ax.plot(days, pd.Series(r).rolling(20).std(ddof=1), color="#14616e", lw=1.2,
            label="20 day rolling sd")
    for lam, ls in zip(LAMBDAS, ("-", ":")):
        # The EW estimate as it would have stood at the end of each day.
        path = [np.sqrt(ew_covar(r[:t].reshape(-1, 1), lam)[0, 0]) if t > 20 else np.nan
                for t in range(1, n + 1)]
        ax.plot(days, path, color="#8a3a2b", lw=1.0, ls=ls, label=f"EW sd, lambda={lam}")
    ax.axhline(sd_eq, color="#777777", lw=0.8, ls="--", label="equal weight sd")
    ax.set_xlabel("day")
    ax.set_ylabel("sd")
    ax.legend(fontsize=7, frameon=False, loc="upper left")

    fig.tight_layout()
    path = os.path.join(FIGDIR, "problem2.png")
    fig.savefig(path, dpi=160)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
