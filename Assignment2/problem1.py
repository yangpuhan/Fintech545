"""
Problem 1 -- Correlations from Mismatched Histories.

Builds the complete case and the pairwise correlation matrices of problem1.csv,
attempts a Cholesky of each, prices the IDX-minus-basket tracking portfolio off
the pairwise matrix, repairs that matrix with Rebonato-Jackel and with Higham,
and measures how far each repair moved it.

    python3 problem1.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import rule
from riskmgmt import chol_psd, corr_to_cov, higham_nearest_psd, missing_cov, near_psd

FIGDIR = "figures"
COLS = ["A", "B", "C", "D", "IDX"]

# Long one unit of IDX, short the basket it is built from.  In the order of
# COLS.  If IDX matched the basket exactly this portfolio would have zero
# variance; it has the (small) variance of the tracking error.
TRACKING = np.array([-0.40, -0.30, -0.20, -0.10, 1.00])


def print_matrix(m, fmt="{:>9.4f}"):
    print(" " * 6 + "".join(f"{c:>9}" for c in COLS))
    for c, row in zip(COLS, m):
        print(f"{c:<6}" + "".join(fmt.format(v) for v in row))


def try_cholesky(name, corr):
    """Attempt the factorization twice: numpy's, which needs positive definite,
    and riskmgmt.chol_psd, which also accepts a singular PSD matrix and fails
    only on a genuinely negative pivot."""
    try:
        np.linalg.cholesky(corr)
        print(f"{name}: numpy Cholesky succeeds (matrix is positive definite)")
    except np.linalg.LinAlgError as err:
        print(f"{name}: numpy Cholesky FAILS -- {err}")
    try:
        root = chol_psd(corr)
        print(f"{name}: chol_psd succeeds, {int(np.sum(np.diag(root) > 0))} "
              f"non-zero columns, max |LL' - C| = "
              f"{np.max(np.abs(root @ root.T - corr)):.2e}")
    except ValueError as err:
        print(f"{name}: chol_psd FAILS -- {err}")


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    d = pd.read_csv("problem1.csv")
    x = d[COLS].to_numpy()                    # NaN where the asset did not trade
    present = ~np.isnan(x)

    # ---- (predict) how much data sits behind each entry -------------------
    rule("PROBLEM 1 -- days each pair is jointly observed")
    counts = present.T.astype(int) @ present.astype(int)
    print_matrix(counts, fmt="{:>9d}")
    all_five = int(present.all(axis=1).sum())
    first_d = int(d.loc[present[:, 3], "Day"].min())
    print(f"\ndays all five trade: {all_five} of {len(d)}")
    print(f"D's first traded day: {first_d}   (so every complete row is from "
          f"day {first_d} on)")
    print(f"C missing on {int((~present[:, 2]).sum())} days, A and B on "
          f"{int((~present[:, 0]).sum())}")
    # A sample correlation's standard error is about (1 - rho^2)/sqrt(n), so
    # 1/sqrt(n) is an upper bound that needs no estimate of rho.  Enough to
    # rank the entries by reliability before anything is computed.
    print("\nrough standard error of each correlation, 1/sqrt(n):")
    print_matrix(1.0 / np.sqrt(counts), fmt="{:>9.3f}")

    # ---- (b) how small the smallest eigenvalue should be -------------------
    # Write the tracking portfolio in correlation units, v = w*sd / ||w*sd||.
    # Then v'Cv = var(tracking) / ||w*sd||^2, and the smallest eigenvalue of C
    # can be no larger than that (it is the minimum of v'Cv over unit v).  On
    # the complete rows the tracking variance can be measured directly, with
    # no correlation estimate at all; the sds come from the same rows so the
    # bound applies exactly to that sample's correlation matrix.
    rule("PROBLEM 1 (b) -- how small the smallest eigenvalue is")
    complete = x[present.all(axis=1)]
    te = complete @ TRACKING
    te_var = float(np.var(te, ddof=1))
    norm2 = float(np.sum((TRACKING * np.std(complete, axis=0, ddof=1)) ** 2))
    print(f"variance of IDX - basket on the {all_five} complete rows = {te_var:.4e}"
          f"   (sd {np.sqrt(te_var):.6f} per day)")
    print(f"||w * sd||^2 on the same rows = {norm2:.4e}")
    print(f"smallest eigenvalue <= {te_var:.4e} / {norm2:.4e} = {te_var / norm2:.6f}")
    sd = np.nanstd(x, axis=0, ddof=1)         # full history, each column's own days

    # ---- (d) the two estimators -------------------------------------------
    rule("PROBLEM 1 (d) -- complete case and pairwise correlation")
    cc = missing_cov(x, skip_miss=True, corr=True)
    pw = missing_cov(x, skip_miss=False, corr=True)
    print("complete case (rows where all five trade)")
    print_matrix(cc)
    print("\neigenvalues:", np.array2string(np.linalg.eigvalsh(cc), precision=6))
    print("\npairwise (each entry from the rows where that pair trades)")
    print_matrix(pw)
    print("\neigenvalues:", np.array2string(np.linalg.eigvalsh(pw), precision=6))
    print()
    try_cholesky("complete case", cc)
    try_cholesky("pairwise     ", pw)

    # The eigenvector carrying the negative eigenvalue, against the tracking
    # portfolio written in correlation units (weight times sd, normalized).
    # If they line up, the inconsistency in the pairwise matrix sits exactly
    # in the direction of the near-linear dependence.
    vals, vecs = np.linalg.eigh(pw)
    v = vecs[:, 0] * np.sign(vecs[-1, 0])
    t = TRACKING * sd / np.linalg.norm(TRACKING * sd)
    print("\neigenvector of the negative eigenvalue :",
          np.array2string(v, precision=4))
    print("tracking portfolio in correlation units:",
          np.array2string(t, precision=4))
    print(f"cosine between them = {abs(v @ t):.4f}")

    # ---- (e) the tracking portfolio off the pairwise matrix ----------------
    rule("PROBLEM 1 (e) -- tracking portfolio variance, pairwise correlation")
    print("full history standard deviations (each series on its own days):")
    for c, s in zip(COLS, sd):
        print(f"  {c:<4}{s:.6f}   n = {int(present[:, COLS.index(c)].sum())}")
    cov_pw = corr_to_cov(pw, sd)
    var_pw = float(TRACKING @ cov_pw @ TRACKING)
    print(f"\nw' Sigma w = {var_pw:.4e}    <-- negative")

    # A benchmark for what the number should be: the tracking error measured
    # directly on the days all five trade, where no correlation is needed.
    print(f"\nfor reference, the variance of IDX - basket measured directly on "
          f"the {all_five} complete rows = {te_var:.4e}")
    var_cc = float(TRACKING @ corr_to_cov(cc, sd) @ TRACKING)
    print(f"the complete case correlation with the same full history sds "
          f"gives {var_cc:.4e}")

    # ---- (f) the two repairs ----------------------------------------------
    rule("PROBLEM 1 (f) -- Rebonato-Jackel and Higham repairs")
    repairs = {"Rebonato-Jackel": near_psd(pw), "Higham": higham_nearest_psd(pw)}
    print(f"{'':<18}{'min eig':>14}{'||. - pw||_F':>15}{'w Sigma w':>14}")
    for name, rep in repairs.items():
        print(f"{name:<18}{np.min(np.linalg.eigvalsh(rep)):>14.3e}"
              f"{np.linalg.norm(rep - pw):>15.6f}"
              f"{TRACKING @ corr_to_cov(rep, sd) @ TRACKING:>14.4e}")
    print(f"\n||RJ - Higham||_F = "
          f"{np.linalg.norm(repairs['Rebonato-Jackel'] - repairs['Higham']):.6f}")
    try_cholesky("Higham repair", repairs["Higham"])

    # ---- (h) where Higham moved the matrix --------------------------------
    rule("PROBLEM 1 (h) -- entries the Higham repair moved, largest first")
    delta = repairs["Higham"] - pw
    rj_delta = repairs["Rebonato-Jackel"] - pw
    pairs = [(i, j) for i in range(5) for j in range(i + 1, 5)]
    pairs.sort(key=lambda p: -abs(delta[p]))
    print(f"{'pair':<10}{'n days':>8}{'pairwise':>10}{'Higham':>10}"
          f"{'change':>10}{'RJ change':>11}{'v_i*v_j':>10}")
    for i, j in pairs:
        print(f"{COLS[i]+'-'+COLS[j]:<10}{counts[i, j]:>8d}{pw[i, j]:>10.4f}"
              f"{repairs['Higham'][i, j]:>10.4f}{delta[i, j]:>+10.4f}"
              f"{rj_delta[i, j]:>+11.4f}{v[i] * v[j]:>+10.4f}")
    print("\nv_i*v_j is the product of the negative eigenvector's loadings.  A")
    print("single clipped eigenvalue moves entry (i,j) by about -lambda*v_i*v_j,")
    print("so the change is set by the eigenvector, not by the day counts.")
    print(f"correlation between |change| and v_i*v_j over the 10 pairs: "
          f"{np.corrcoef([abs(delta[p]) for p in pairs], [abs(v[p[0]] * v[p[1]]) for p in pairs])[0, 1]:.4f}")
    print(f"correlation between |change| and 1/n days:                   "
          f"{np.corrcoef([abs(delta[p]) for p in pairs], [1 / counts[p] for p in pairs])[0, 1]:.4f}")

    # ---- (i) repair against estimator --------------------------------------
    rule("PROBLEM 1 (i) -- size of the repair against the estimator gap")
    gap = np.linalg.norm(cc - pw)
    print(f"||complete case - pairwise||_F = {gap:.6f}")
    print(f"||Higham - pairwise||_F        = {np.linalg.norm(delta):.6f}")
    print(f"ratio                          = {gap / np.linalg.norm(delta):.1f}")
    print("\nlargest entrywise gaps between the two estimators:")
    gaps = sorted(pairs, key=lambda p: -abs(cc[p] - pw[p]))
    for i, j in gaps[:5]:
        print(f"  {COLS[i]+'-'+COLS[j]:<8} complete {cc[i, j]:.4f}  pairwise "
              f"{pw[i, j]:.4f}  gap {cc[i, j] - pw[i, j]:+.4f}")

    # The complete rows are all from day 151 on, so the complete case estimator
    # is also a choice of time window.  Split A-B-C-IDX, which trade almost
    # every day, at that point to see whether the window alone moves them.
    early = d["Day"] < first_d
    rule(f"PROBLEM 1 (i) -- A, B, C, IDX before and after day {first_d}")
    # Fisher's z = atanh(r) is close to normal with sd 1/sqrt(n-3), so the
    # difference between two independent windows has sd
    # sqrt(1/(n1-3) + 1/(n2-3)).
    print(f"{'pair':<8}{'days 1-150':>12}{'n':>5}{'days 151-250':>14}{'n':>5}"
          f"{'z stat':>9}")
    for p, q in (("A", "B"), ("A", "C"), ("B", "C"), ("A", "IDX"), ("C", "IDX")):
        r_n = []
        for mask in (early, ~early):
            sub = d.loc[mask, [p, q]].dropna()
            r_n.append((sub[p].corr(sub[q]), len(sub)))
        (r1, n1), (r2, n2) = r_n
        z = (np.arctanh(r1) - np.arctanh(r2)) / np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
        print(f"{p + '-' + q:<8}{r1:>12.3f}{n1:>5d}{r2:>14.3f}{n2:>5d}{z:>9.2f}")

    # ---- figure ------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6),
                             gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    ax.imshow(present.T, aspect="auto", cmap="Greys", interpolation="nearest",
              extent=[0.5, len(d) + 0.5, 4.5, -0.5], vmin=-0.3, vmax=1.2)
    ax.set_yticks(range(5))
    ax.set_yticklabels(COLS)
    ax.set_xlabel("day")
    ax.set_title("Days each series trades (dark = traded)")
    ax.axvline(first_d - 0.5, color="#8a3a2b", lw=1.2)
    ax.text(first_d - 3, 3, f"D lists on day {first_d}", color="#8a3a2b",
            fontsize=8, ha="right", va="center")

    ax = axes[1]
    labels = [f"{COLS[i]}-{COLS[j]}" for i, j in pairs]
    ypos = np.arange(len(pairs))[::-1]
    ax.barh(ypos, [abs(delta[p]) for p in pairs], color="#14616e", height=0.6)
    for y, (i, j) in zip(ypos, pairs):
        ax.text(abs(delta[i, j]) + 1e-4, y, f"n={counts[i, j]}", va="center",
                fontsize=7, color="#555555")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("|Higham - pairwise|")
    ax.set_title("Where the repair moved the matrix")
    ax.set_xlim(0, max(abs(delta[p]) for p in pairs) * 1.35)

    fig.tight_layout()
    path = os.path.join(FIGDIR, "problem1.png")
    fig.savefig(path, dpi=160)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
