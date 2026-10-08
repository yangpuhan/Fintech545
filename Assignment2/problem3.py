"""
Problem 3 -- When Diversification Raises VaR.

Two bonds, A and B, each bought at 90 to pay 100 unless it defaults; 10,000
simulated one year return scenarios in problem3.csv.  Compares $2,000,000 in A
against $1,000,000 in each, with historical VaR and ES at 5% and 1% and with a
normal VaR, and checks subadditivity.

    python3 problem3.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import normal_var_es, print_moments, rule, var_es

FIGDIR = "figures"
LOSS_CUT = -0.20        # the problem's "loses more than 20%"


def positions(a, b):
    """P&L vectors, in dollars, for the four positions the problem names."""
    return {
        "$1M A": 1e6 * a,
        "$1M B": 1e6 * b,
        "$2M A": 2e6 * a,
        "$1M A + $1M B": 1e6 * (a + b),
    }


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    d = pd.read_csv("problem3.csv")
    a, b = d["A"].to_numpy(), d["B"].to_numpy()
    n = len(d)
    pnl = positions(a, b)

    # ---- (a) moments --------------------------------------------------------
    rule("PROBLEM 3 (a) -- first four moments of each bond's return")
    print_moments([("A", a), ("B", b)])
    for name, x in (("A", a), ("B", b)):
        print(f"{name}: min {x.min():+.4f}  median {np.median(x):+.4f}  "
              f"max {x.max():+.4f}   (100/90 - 1 = {100 / 90 - 1:.4f})")
    print(f"\ncorrelation of A and B: {np.corrcoef(a, b)[0, 1]:+.4f}")
    # Each bond's returns fall in two clusters with an empty gap between them:
    # the no-default cluster near +11% and the default cluster well below zero.
    # A negative return therefore identifies a default.
    for name, x in (("A", a), ("B", b)):
        print(f"{name}: largest negative return {x[x < 0].max():+.4f}, smallest "
              f"non-negative {x[x >= 0].min():+.4f}")

    # ---- (b) counts -----------------------------------------------------------
    rule("PROBLEM 3 (b) -- scenarios losing more than 20%")
    la, lb = a < LOSS_CUT, b < LOSS_CUT
    print(f"A loses > 20%        : {int(la.sum()):>6}  ({la.mean():.2%})")
    print(f"B loses > 20%        : {int(lb.sum()):>6}  ({lb.mean():.2%})")
    print(f"at least one         : {int((la | lb).sum()):>6}  ({(la | lb).mean():.2%})")
    print(f"both                 : {int((la & lb).sum()):>6}  ({(la & lb).mean():.2%})")
    print(f"both, if independent : {n * la.mean() * lb.mean():>6.1f}")

    da, db = a < 0, b < 0                   # defaults, by the cluster gap above
    print(f"\ndefaults (negative return): A {int(da.sum())}, B {int(db.sum())}, "
          f"at least one {int((da | db).sum())}, both {int((da & db).sum())}")
    print(f"  the B default the >20% count misses lost {-b[db & ~lb][0]:.2%}")
    print(f"\n5% of {n} scenarios = {int(0.05 * n)}; 1% = {int(0.01 * n)}")
    print(f"mean return given default: A {a[da].mean():+.4f}, B {b[db].mean():+.4f}")
    print(f"mean return given no default: A {a[~da].mean():+.4f}, "
          f"B {b[~db].mean():+.4f}")

    # ---- (c), (e) historical VaR and ES -------------------------------------
    for alpha, tag in ((0.05, "(c)"), (0.01, "(e)")):
        rule(f"PROBLEM 3 {tag} -- historical VaR and ES at {alpha:.0%}")
        print(f"{'position':<16}{'VaR':>14}{'ES':>14}")
        for name, x in pnl.items():
            v, e = var_es(x, alpha)
            print(f"{name:<16}{v:>14,.0f}{e:>14,.0f}")

    # ---- (d) normal VaR -------------------------------------------------------
    rule("PROBLEM 3 (d) -- normal VaR from each P&L's sample mean and sd")
    print(f"{'position':<16}{'mean':>12}{'sd':>12}{'VaR 5%':>12}{'ES 5%':>12}"
          f"{'VaR 1%':>12}")
    for name, x in pnl.items():
        mu, sd = float(np.mean(x)), float(np.std(x, ddof=1))
        v5, e5 = normal_var_es(mu, sd, 0.05)
        v1, _ = normal_var_es(mu, sd, 0.01)
        print(f"{name:<16}{mu:>12,.0f}{sd:>12,.0f}{v5:>12,.0f}{e5:>12,.0f}{v1:>12,.0f}")

    # ---- (f), (h) subadditivity -------------------------------------------------
    rule("PROBLEM 3 (f) and (h) -- subadditivity, q(A) + q(B) >= q(A + B)?")
    print(f"{'measure':<16}{'q(A)+q(B)':>14}{'q(A+B)':>14}{'holds':>8}")
    for alpha in (0.05, 0.01):
        sa, sb, sab = (var_es(pnl[k], alpha) for k in
                       ("$1M A", "$1M B", "$1M A + $1M B"))
        for i, label in ((0, "VaR"), (1, "ES")):
            lhs, rhs = sa[i] + sb[i], sab[i]
            print(f"{label + f' {alpha:.0%}':<16}{lhs:>14,.0f}{rhs:>14,.0f}"
                  f"{('yes' if lhs >= rhs else 'NO'):>8}")
    mu_a, mu_b, mu_ab = (normal_var_es(np.mean(pnl[k]), np.std(pnl[k], ddof=1))[0]
                         for k in ("$1M A", "$1M B", "$1M A + $1M B"))
    print(f"{'normal VaR 5%':<16}{mu_a + mu_b:>14,.0f}{mu_ab:>14,.0f}"
          f"{('yes' if mu_a + mu_b >= mu_ab else 'NO'):>8}")

    # Which scenarios sit in each position's 5% and 1% tails.  This is the
    # mechanism: where the quantile falls relative to the default probability.
    rule("PROBLEM 3 -- what the worst 5% and 1% of scenarios contain")
    n_def = da.astype(int) + db.astype(int)
    for alpha in (0.05, 0.01):
        k = int(alpha * n)
        print(f"\nworst {k} scenarios ({alpha:.0%}):")
        for name in ("$2M A", "$1M A + $1M B"):
            worst = np.argsort(pnl[name])[:k]
            if name == "$2M A":
                print(f"  {name:<15} A defaults in {int(da[worst].sum())} of {k}; "
                      f"the {k}th worst P&L is {np.sort(pnl[name])[k - 1]:+,.0f}")
            else:
                c = np.bincount(n_def[worst], minlength=3)
                print(f"  {name:<15} 0 defaults {c[0]}, one {c[1]}, both {c[2]}; "
                      f"the {k}th worst P&L is {np.sort(pnl[name])[k - 1]:+,.0f}")

    # ---- figure ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    bins = np.linspace(-1.8e6, 0.35e6, 90)
    for ax, name in zip(axes, ("$2M A", "$1M A + $1M B")):
        x = pnl[name]
        ax.hist(x, bins=bins, color="#14616e", edgecolor="none")
        v5, e5 = var_es(x, 0.05)
        v1, _ = var_es(x, 0.01)
        ax.axvline(-v5, color="#8a3a2b", lw=1.3, label=f"5% VaR = {v5:,.0f}")
        ax.axvline(-e5, color="#8a3a2b", lw=1.3, ls="--", label=f"5% ES = {e5:,.0f}")
        ax.axvline(-v1, color="#d08a3a", lw=1.3, ls=":", label=f"1% VaR = {v1:,.0f}")
        ax.set_yscale("log")
        # matplotlib reads a pair of $ as math mode, so escape them.
        ax.set_title("P&L, " + name.replace("$", r"\$"))
        ax.set_xlabel("one year P&L ($)")
        ax.legend(fontsize=7, frameon=False, loc="upper left")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(6, 6))
    axes[0].set_ylabel("scenarios (log scale)")
    fig.tight_layout()
    path = os.path.join(FIGDIR, "problem3.png")
    fig.savefig(path, dpi=160)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
