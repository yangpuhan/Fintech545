"""
Problem 5 -- Identifying an AR or MA Order.

Plots problem5.csv with its ACF and PACF, reads the order off the two
functions, then fits AR(1) through AR(3) and MA(1) through MA(3) and compares
them on AICc.

    python3 problem5.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf

from common import aicc, rule

FIGDIR = "figures"
NLAGS = 12


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    y = pd.read_csv("problem5.csv")["x"].to_numpy()
    n = len(y)

    # ---- (predict) read the order off the ACF and PACF ------------------
    rule("PROBLEM 5 -- ACF and PACF")
    a = acf(y, nlags=NLAGS, fft=False)
    p = pacf(y, nlags=NLAGS, method="ywm")
    # The standard significance band for a sample autocorrelation under the
    # null that the true value is zero.  This is the band the plots draw and
    # the one used below to call a lag significant.
    band = 1.96 / np.sqrt(n)
    print(f"n = {n}, mean = {y.mean():.4f}, sd = {y.std(ddof=1):.4f}")
    print(f"significance band = +/- 1.96/sqrt(n) = +/- {band:.4f}\n")
    print(f"{'lag':>4}{'ACF':>10}{'sig':>6}{'PACF':>10}{'sig':>6}")
    for i in range(1, NLAGS + 1):
        print(f"{i:>4}{a[i]:>+10.4f}{('yes' if abs(a[i]) > band else ''):>6}"
              f"{p[i]:>+10.4f}{('yes' if abs(p[i]) > band else ''):>6}")

    last_sig_acf = max((i for i in range(1, NLAGS + 1) if abs(a[i]) > band),
                       default=0)
    last_sig_pacf = max((i for i in range(1, NLAGS + 1) if abs(p[i]) > band),
                        default=0)
    print(f"\nlast lag outside the band: ACF {last_sig_acf}, "
          f"PACF {last_sig_pacf}")

    # ---- (fit) six models ------------------------------------------------
    rule("PROBLEM 5 -- AR and MA fits")
    print("k = p + d.  For AR(p): p AR coefficients + 1 intercept, plus d = 1")
    print("for the fitted variance, so k = p + 2.  Same for MA(q): k = q + 2.")
    print("statsmodels reports AIC only, so AICc is computed from the log")
    print("likelihood by hand.\n")

    specs = [("AR(1)", (1, 0, 0)), ("AR(2)", (2, 0, 0)), ("AR(3)", (3, 0, 0)),
             ("MA(1)", (0, 0, 1)), ("MA(2)", (0, 0, 2)), ("MA(3)", (0, 0, 3))]
    results = {}
    print(f"{'model':>8}{'loglik':>12}{'k':>4}{'AIC':>12}{'AICc':>12}")
    for name, order in specs:
        fit = ARIMA(y, order=order, trend="c").fit()
        k = order[0] + order[2] + 2          # coefficients + intercept + sigma^2
        score = aicc(fit.llf, k, n)
        results[name] = {"fit": fit, "k": k, "aicc": score}
        print(f"{name:>8}{fit.llf:>12.4f}{k:>4}{fit.aic:>12.4f}{score:>12.4f}")

    best = min(results, key=lambda m: results[m]["aicc"])
    print(f"\nAICc selects {best}")
    print(f"{'model':>8}{'AICc':>12}{'dAICc vs best':>16}")
    for name in results:
        d = results[name]["aicc"] - results[best]["aicc"]
        print(f"{name:>8}{results[name]['aicc']:>12.4f}{d:>16.4f}")

    # ---- fitted coefficients of the selected model -----------------------
    rule(f"PROBLEM 5 -- coefficients of {best}")
    fit = results[best]["fit"]
    for nm, val, se in zip(fit.param_names, fit.params, fit.bse):
        print(f"  {nm:<12}{val:>+10.4f}   se {se:.4f}   t {val/se:>+7.2f}")

    # The roots of the AR characteristic polynomial say what shape the ACF
    # should have.  z^2 - phi1 z - phi2 = 0 with a negative discriminant gives
    # a complex pair, and a complex pair means the autocorrelation decays as a
    # damped oscillation rather than monotonically -- which is what the ACF
    # column above shows.
    if best.startswith("AR(2)"):
        phi1, phi2 = fit.params[1], fit.params[2]
        disc = phi1 ** 2 + 4 * phi2
        print(f"\ncharacteristic equation z^2 - {phi1:.4f} z - ({phi2:.4f}) = 0")
        print(f"discriminant = phi1^2 + 4 phi2 = {disc:+.4f}"
              f"  -> roots are {'complex' if disc < 0 else 'real'}")
        roots = np.roots([1.0, -phi1, -phi2])
        print(f"roots: {np.array2string(roots, precision=4)}")
        print(f"modulus {abs(roots[0]):.4f} (inside the unit circle means "
              "stationary)")
        if disc < 0:
            print("A complex pair makes the ACF a damped oscillation, which is")
            print("why the ACF changes sign at lag 2 instead of decaying "
                  "monotonically.")

    # ---- (e) AR(2) against AR(3) ----------------------------------------
    rule("PROBLEM 5 (e) -- AR(2) against AR(3)")
    f2, f3 = results["AR(2)"]["fit"], results["AR(3)"]["fit"]
    third = f3.params[3]        # const, ar.L1, ar.L2, ar.L3, sigma2
    third_se = f3.bse[3]
    print(f"AR(3) third coefficient  = {third:+.5f}  se {third_se:.5f}  "
          f"t {third/third_se:+.3f}")
    print(f"log likelihood: AR(2) {f2.llf:.4f}, AR(3) {f3.llf:.4f}, "
          f"gain {f3.llf - f2.llf:.4f}")
    print(f"AICc:           AR(2) {results['AR(2)']['aicc']:.4f}, "
          f"AR(3) {results['AR(3)']['aicc']:.4f}, "
          f"penalty net of the gain {results['AR(3)']['aicc'] - results['AR(2)']['aicc']:+.4f}")

    # R^2 for the two, to show it moves the wrong way for model selection.
    def r_squared(f):
        e = np.asarray(f.resid)
        return 1.0 - np.var(e, ddof=0) / np.var(y, ddof=0)

    r2_2, r2_3 = r_squared(f2), r_squared(f3)
    print(f"\nR^2: AR(2) {r2_2:.6f}, AR(3) {r2_3:.6f}, "
          f"change {r2_3 - r2_2:+.3e}")
    print("R^2 rose, as it must whenever a parameter is added, so it would")
    print("never have rejected the larger model.")

    # ---- figure ----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    axes[0].plot(y, lw=0.7, color="#14616e")
    axes[0].axhline(y.mean(), color="#8a3a2b", lw=1.0, ls="--",
                    label=f"mean {y.mean():.2f}")
    axes[0].set_title("The series")
    axes[0].set_xlabel("t")
    axes[0].legend(fontsize=7, frameon=False)

    plot_acf(y, lags=NLAGS, ax=axes[1], color="#14616e", vlines_kwargs={"colors": "#14616e"})
    axes[1].set_title(f"ACF, band +/- {band:.3f}")
    plot_pacf(y, lags=NLAGS, ax=axes[2], method="ywm", color="#14616e",
              vlines_kwargs={"colors": "#14616e"})
    axes[2].set_title(f"PACF, band +/- {band:.3f}")
    for ax in axes[1:]:
        ax.set_xlabel("lag")

    fig.tight_layout()
    path = os.path.join(FIGDIR, "problem5.png")
    fig.savefig(path, dpi=160)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
