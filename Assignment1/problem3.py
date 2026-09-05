"""
Problem 3 -- Pearson Against Spearman.

Computes both correlation matrices on problem3.csv, reports the pair with the
largest gap, and runs two diagnostics that separate the two mechanisms which
can produce such a gap: a few outliers, or a monotone but non-linear
relationship.

    python3 problem3.py
"""

import itertools
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from common import rule

FIGDIR = "figures"


def normal_scores(v):
    """Map a sample onto normal scores through its ranks (van der Waerden).

    Ranks are preserved exactly, so any dependence that is purely monotone
    survives untouched while the marginal shape is replaced by a normal.  If
    the Pearson correlation jumps once both margins are transformed this way,
    the deficit in the raw Pearson was the marginal shape rather than the
    dependence.
    """
    v = np.asarray(v, dtype=float)
    return stats.norm.ppf((stats.rankdata(v) - 0.5) / len(v))


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    d = pd.read_csv("problem3.csv")
    cols = list(d.columns)

    # ---- marginal shapes, which is where the mechanism shows up ---------
    rule("PROBLEM 3 -- marginal shape of each series")
    print(f"{'series':>8}{'mean':>10}{'sd':>10}{'skew':>10}"
          f"{'ex.kurt':>10}{'min':>10}{'max':>10}")
    for c in cols:
        v = d[c].to_numpy()
        print(f"{c:>8}{v.mean():>10.4f}{v.std(ddof=1):>10.4f}"
              f"{stats.skew(v, bias=False):>10.3f}"
              f"{stats.kurtosis(v, bias=False):>10.3f}"
              f"{v.min():>10.3f}{v.max():>10.3f}")

    # ---- both correlation matrices --------------------------------------
    rule("PROBLEM 3 -- correlation matrices")
    pearson = d.corr(method="pearson")
    spearman = d.corr(method="spearman")
    print("Pearson\n", pearson.round(4).to_string())
    print("\nSpearman\n", spearman.round(4).to_string())
    print("\n|Spearman - Pearson|\n", (spearman - pearson).abs().round(4).to_string())

    rule("PROBLEM 3 -- every pair, sorted by the size of the gap")
    rows = []
    for a, b in itertools.combinations(cols, 2):
        p = float(pearson.loc[a, b])
        s = float(spearman.loc[a, b])
        rows.append((f"{a}-{b}", p, s, abs(s - p)))
    rows.sort(key=lambda r: -r[3])
    print(f"{'pair':>10}{'Pearson':>12}{'Spearman':>12}{'gap':>10}")
    for name, p, s, g in rows:
        print(f"{name:>10}{p:>+12.4f}{s:>+12.4f}{g:>10.4f}")
    widest = rows[0]
    print(f"\nlargest gap: {widest[0]}, "
          f"Pearson {widest[1]:.4f} against Spearman {widest[2]:.4f}, "
          f"gap {widest[3]:.4f}")

    # ---- which mechanism? ------------------------------------------------
    a, b = widest[0].split("-")
    va, vb = d[a].to_numpy(), d[b].to_numpy()

    rule(f"PROBLEM 3 -- what produces the gap on {a}-{b}")

    # Mechanism A: a handful of outliers.  If so, trimming them closes the gap.
    za = np.abs((va - va.mean()) / va.std(ddof=1))
    zb = np.abs((vb - vb.mean()) / vb.std(ddof=1))
    print("trimming the most extreme observations:")
    print(f"{'rule':>12}{'dropped':>10}{'Pearson':>12}{'Spearman':>12}{'gap':>10}")
    print(f"{'none':>12}{0:>10}{widest[1]:>12.4f}{widest[2]:>12.4f}{widest[3]:>10.4f}")
    for cut in (4.0, 3.0, 2.5):
        keep = (za < cut) & (zb < cut)
        p = float(np.corrcoef(va[keep], vb[keep])[0, 1])
        s = float(stats.spearmanr(va[keep], vb[keep])[0])
        print(f"{'|z| < ' + str(cut):>12}{int((~keep).sum()):>10}"
              f"{p:>12.4f}{s:>12.4f}{abs(s-p):>10.4f}")

    # Mechanism B: a monotone but non-linear relationship.  If so, replacing
    # the margins with normal scores restores a high Pearson correlation.
    p_scores = float(np.corrcoef(normal_scores(va), normal_scores(vb))[0, 1])
    print(f"\nPearson after mapping both margins to normal scores: {p_scores:.4f}")
    print(f"  (raw Pearson {widest[1]:.4f}, Spearman {widest[2]:.4f})")

    # The marginal that is doing it: fit a t to each and read the tail weight.
    print("\nStudent's t fitted to each margin (low nu means a heavy tail):")
    for c in (a, b):
        nu, loc, scale = stats.t.fit(d[c].to_numpy())
        print(f"  {c}: nu = {nu:8.3f}   loc = {loc:+.4f}   scale = {scale:.4f}")

    # ---- figure: every pair ---------------------------------------------
    k = len(cols)
    fig, axes = plt.subplots(k, k, figsize=(9.5, 9.5))
    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            ax = axes[i, j]
            if i == j:
                ax.hist(d[ci], bins=40, color="#c9d6dd", edgecolor="#8fa3ad",
                        linewidth=0.3)
            else:
                ax.scatter(d[cj], d[ci], s=5, alpha=0.45, color="#14616e",
                           edgecolor="none")
                p = float(pearson.loc[ci, cj])
                s = float(spearman.loc[ci, cj])
                ax.set_title(f"P {p:+.2f}   S {s:+.2f}", fontsize=7, pad=2)
            ax.tick_params(labelsize=6)
            if i == k - 1:
                ax.set_xlabel(cj, fontsize=8)
            if j == 0:
                ax.set_ylabel(ci, fontsize=8)
    fig.suptitle("Every pair, with Pearson (P) and Spearman (S)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = os.path.join(FIGDIR, "problem3_pairs.png")
    fig.savefig(path, dpi=150)
    print(f"\nwrote {path}")

    # ---- figure: the widest pair, raw against rank space -----------------
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9))
    axes[0].scatter(va, vb, s=10, alpha=0.55, color="#14616e", edgecolor="none")
    axes[0].set_title(f"{a} against {b}, original units\n"
                      f"Pearson {widest[1]:.3f}, Spearman {widest[2]:.3f}",
                      fontsize=9)
    axes[0].set_xlabel(a)
    axes[0].set_ylabel(b)
    axes[1].scatter(normal_scores(va), normal_scores(vb), s=10, alpha=0.55,
                    color="#8a3a2b", edgecolor="none")
    axes[1].set_title(f"the same pair on normal scores\n"
                      f"Pearson {p_scores:.3f}", fontsize=9)
    axes[1].set_xlabel(f"normal score of {a}")
    axes[1].set_ylabel(f"normal score of {b}")
    fig.tight_layout()
    path = os.path.join(FIGDIR, "problem3_widest.png")
    fig.savefig(path, dpi=160)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
