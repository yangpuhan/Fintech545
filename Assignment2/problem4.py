"""
Problem 4 -- Gaussian or t Copula.

Fits a margin to each of the three series in problem4.csv (normal or t, chosen
by AICc), builds R from Kendall's tau, fits a Gaussian and a t copula on the
same uniforms and the same R, and compares them: information criteria, joint
tail counts, portfolio VaR and ES from 100,000 simulated days, where the
likelihood gain comes from, and the tail dependence coefficient.

    python3 problem4.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from common import (copula_criteria, gaussian_copula_ll, joint_tail_counts,
                    kendall_correlation, print_moments, profile_nu,
                    pseudo_uniforms, rule, simulate_copula_uniforms,
                    t_copula_ll, tail_dependence_t, var_es)
from riskmgmt import aicc, fit_general_t, fit_normal

FIGDIR = "figures"
POSITION = 1_000_000.0          # held in each of the three
N_SIM = 100_000
N_BATCH = 10                    # for the simulation standard errors
SEED = 545
TAIL = 0.025                    # part (b)'s joint tail
OUTER = 0.05                    # part (i)'s "outer 5% on either side"


def gaussian_joint_tail_prob(rho, q):
    """P(U1 <= q, U2 <= q) under a Gaussian copula with correlation rho.  By
    the copula's symmetry this is also P(U1 >= 1-q, U2 >= 1-q)."""
    z = stats.norm.ppf(q)
    return float(stats.multivariate_normal([0, 0], [[1, rho], [rho, 1]]).cdf([z, z]))


def risk_row(pnl):
    v5, e5 = var_es(pnl, 0.05)
    v1, e1 = var_es(pnl, 0.01)
    return np.array([v5, e5, v1, e1])


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    d = pd.read_csv("problem4.csv")
    names = list(d.columns)
    x = d.to_numpy()
    m, n = x.shape
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # ---- (a) moments, and how much each depends on one observation --------
    rule("PROBLEM 4 (a) -- first four moments")
    print_moments([(c, x[:, i]) for i, c in enumerate(names)])
    print("\nthe same with each series' single most extreme day removed:")
    print(f"{'':<6}{'extreme day':>13}{'value':>10}{'in sds':>8}"
          f"{'skew':>10}{'ex. kurt':>10}")
    for i, c in enumerate(names):
        xi = x[:, i]
        k = int(np.argmax(np.abs(xi - xi.mean())))
        rest = np.delete(xi, k)
        print(f"{c:<6}{k + 1:>13d}{xi[k]:>10.4f}"
              f"{(xi[k] - xi.mean()) / xi.std(ddof=1):>8.2f}"
              f"{stats.skew(rest, bias=False):>10.4f}"
              f"{stats.kurtosis(rest, bias=False):>10.4f}")
    print(f"\nstandard error of excess kurtosis under normality ~ sqrt(24/m) = "
          f"{np.sqrt(24 / m):.3f}; of skewness ~ sqrt(6/m) = {np.sqrt(6 / m):.3f}")
    extreme_days = {int(np.argmax(np.abs(x[:, i] - x[:, i].mean()))) for i in range(n)}
    if len(extreme_days) == 1:
        crash = extreme_days.pop()
        rank_on_day = stats.rankdata(x, axis=0)[crash].astype(int)
        print(f"\nall three extremes fall on the same day, day {crash + 1}: ranks "
              + ", ".join(f"{c} {k}" for c, k in zip(names, rank_on_day))
              + f" of {m}")
    else:
        crash = None

    # ---- (b) ranks and joint tail counts -------------------------------------
    rule("PROBLEM 4 (b) -- joint tail days, ranks scaled into (0,1)")
    ranks = pseudo_uniforms(x)
    counts = joint_tail_counts(ranks, TAIL)
    r_tau, _, tau = kendall_correlation(ranks)    # rank statistic, no fit yet
    print(f"expected joint days per pair under independence: "
          f"{m} x {TAIL}^2 = {m * TAIL ** 2:.3f}\n")
    print(f"{'pair':<8}{'tau':>8}{'sin(pi tau/2)':>15}{'worst 2.5%':>12}"
          f"{'best 2.5%':>11}{'Gaussian at that rho':>22}")
    for i, j in pairs:
        g = m * gaussian_joint_tail_prob(r_tau[i, j], TAIL)
        print(f"{names[i] + '-' + names[j]:<8}{tau[i, j]:>8.4f}{r_tau[i, j]:>15.4f}"
              f"{counts[(i, j)][0]:>12d}{counts[(i, j)][1]:>11d}{g:>22.2f}")
    tot_lo = sum(c[0] for c in counts.values())
    tot_hi = sum(c[1] for c in counts.values())
    tot_g = sum(m * gaussian_joint_tail_prob(r_tau[p], TAIL) for p in pairs)
    print(f"{'total':<8}{'':>23}{tot_lo:>12d}{tot_hi:>11d}{tot_g:>22.2f}")

    # The same comparison at less extreme cut-offs: is the excess over the
    # Gaussian concentrated in the tail, or present everywhere?
    print("\nobserved / Gaussian joint exceedance, summed over the three pairs "
          "and both tails:")
    for q in (0.25, 0.10, 0.05, 0.025):
        c = joint_tail_counts(ranks, q)
        obs = sum(lo + hi for lo, hi in c.values())
        exp_g = 2 * sum(m * gaussian_joint_tail_prob(r_tau[p], q) for p in pairs)
        print(f"  q = {q:<6} observed {obs:>5d}   Gaussian {exp_g:>8.1f}   "
              f"ratio {obs / exp_g:.2f}")

    # ---- (c) margins ----------------------------------------------------------
    rule("PROBLEM 4 (c) -- normal against t for each margin, by AICc")
    print("k = 2 for the normal (mu, sigma), 3 for the t (mu, sigma, nu)\n")
    print(f"{'':<6}{'ll normal':>12}{'ll t':>12}{'AICc normal':>13}{'AICc t':>12}"
          f"{'nu':>10}{'choice':>8}")
    fits = []
    for i, c in enumerate(names):
        xi = x[:, i]
        fn, ft = fit_normal(xi), fit_general_t(xi)
        ll_n, ll_t = fn.loglik(xi), ft.loglik(xi)
        a_n, a_t = aicc(ll_n, 2, m), aicc(ll_t, 3, m)
        nu = ft.error_model.kwds["df"]
        choice = "t" if a_t < a_n else "normal"
        fits.append(ft if choice == "t" else fn)
        nu_txt = f"{nu:.3f}" if nu < 1000 else "> 1000"
        print(f"{c:<6}{ll_n:>12.3f}{ll_t:>12.3f}{a_n:>13.3f}{a_t:>12.3f}"
              f"{nu_txt:>10}{choice:>8}")
    print("\nchosen margins:")
    for c, f in zip(names, fits):
        kw = f.error_model.kwds
        if "df" in kw:
            print(f"  {c}: t, mu {float(kw['loc']):.6f}, sigma {kw['scale']:.6f}, "
                  f"nu {kw['df']:.4f}")
        else:
            print(f"  {c}: normal, mu {float(kw['loc']):.6f}, sigma {kw['scale']:.6f}")
    # Does X2's t rest on its one extreme day?
    k2 = int(np.argmax(np.abs(x[:, 1] - x[:, 1].mean())))
    ft2 = fit_general_t(np.delete(x[:, 1], k2))
    print(f"\n{names[1]}'s t refitted without day {k2 + 1}: nu "
          f"{ft2.error_model.kwds['df']:.4f}")

    # ---- (d) the two copulas ---------------------------------------------------
    rule("PROBLEM 4 (d) -- Gaussian and t copula, same U, same R")
    u = np.column_stack([f.u for f in fits])     # through the fitted margins
    r, min_eig, tau_u = kendall_correlation(u)
    print("R = sin(pi tau / 2), tau computed on U:")
    for i, c in enumerate(names):
        print(f"  {c:<4}" + "".join(f"{v:>9.4f}" for v in r[i]))
    print(f"smallest eigenvalue before any repair: {min_eig:.4f} "
          f"({'no repair needed' if min_eig > 0 else 'repaired with Higham'})")

    ll_g = gaussian_copula_ll(u, r)
    nu_hat, ll_t = profile_nu(lambda v: t_copula_ll(u, r, v))
    aicc_g, bic_g = copula_criteria(ll_g, 0, m)
    aicc_t, bic_t = copula_criteria(ll_t, 1, m)
    print(f"\n{'':<22}{'Gaussian':>12}{'t':>12}")
    print(f"{'log likelihood':<22}{ll_g:>12.3f}{ll_t:>12.3f}")
    print(f"{'free parameters k':<22}{0:>12d}{1:>12d}")
    print(f"{'nu hat':<22}{'--':>12}{nu_hat:>12.4f}")
    print(f"{'AICc':<22}{aicc_g:>12.3f}{aicc_t:>12.3f}")
    print(f"{'BIC':<22}{bic_g:>12.3f}{bic_t:>12.3f}")
    d_bic = 2 * (ll_t - ll_g) - np.log(m)
    print(f"\ndelta BIC = 2(ll_t - ll_G) - ln m = {2 * (ll_t - ll_g):.3f} - "
          f"{np.log(m):.3f} = {d_bic:.3f}")
    print(f"delta AICc = {aicc_g - aicc_t:.3f}")
    print("\nprofile of the t copula log likelihood:")
    profile_nus = [3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 12.0, 20.0, 50.0, 100.0]
    profile_lls = [t_copula_ll(u, r, v) for v in profile_nus]
    for v, ll in zip(profile_nus, profile_lls):
        print(f"  nu = {v:>6.1f}   ll = {ll:9.3f}")

    # ---- (e) simulate and price the portfolio ---------------------------------
    rule("PROBLEM 4 (e) -- portfolio VaR and ES, $1,000,000 in each")
    def to_returns(us):
        return np.column_stack([fits[i].error_model.ppf(us[:, i]) for i in range(n)])
    u_g = simulate_copula_uniforms(r, N_SIM, nu=None, seed=SEED)
    u_t = simulate_copula_uniforms(r, N_SIM, nu=nu_hat, seed=SEED)
    pnl_g = POSITION * to_returns(u_g).sum(axis=1)
    pnl_t = POSITION * to_returns(u_t).sum(axis=1)
    pnl_h = POSITION * x.sum(axis=1)

    rows = {"Gaussian copula": risk_row(pnl_g), "t copula": risk_row(pnl_t),
            "historical": risk_row(pnl_h)}
    # Simulation error: split each 100,000 draw run into 10 batches and use the
    # spread of the batch estimates.  The two copulas share their normal draws
    # (common random numbers), so the paired difference is much less noisy than
    # either number on its own.
    def batch_se(pnl_a, pnl_b=None):
        est = []
        for k in np.array_split(np.arange(N_SIM), N_BATCH):
            ra = risk_row(pnl_a[k])
            est.append(ra if pnl_b is None else risk_row(pnl_b[k]) - ra)
        return np.std(est, axis=0, ddof=1) / np.sqrt(N_BATCH)

    se_g, se_t, se_diff = batch_se(pnl_g), batch_se(pnl_t), batch_se(pnl_g, pnl_t)
    print(f"{'':<18}{'VaR 5%':>12}{'ES 5%':>12}{'VaR 1%':>12}{'ES 1%':>12}")
    for name, row in rows.items():
        print(f"{name:<18}" + "".join(f"{v:>12,.0f}" for v in row))
    print(f"{'  sim. se, Gauss':<18}" + "".join(f"{v:>12,.0f}" for v in se_g))
    print(f"{'  sim. se, t':<18}" + "".join(f"{v:>12,.0f}" for v in se_t))

    # ---- (h) which number moves -----------------------------------------------
    rule("PROBLEM 4 (h) -- moving from the Gaussian to the t copula")
    diff = rows["t copula"] - rows["Gaussian copula"]
    print(f"{'':<18}{'VaR 5%':>12}{'ES 5%':>12}{'VaR 1%':>12}{'ES 1%':>12}")
    print(f"{'t - Gaussian':<18}" + "".join(f"{v:>+12,.0f}" for v in diff))
    print(f"{'  in %':<18}" + "".join(
        f"{v:>+11.1f}%" for v in 100 * diff / rows["Gaussian copula"]))
    print(f"{'  se of diff':<18}" + "".join(f"{v:>12,.0f}" for v in se_diff))
    print(f"{'  diff / se':<18}" + "".join(f"{v:>12.1f}" for v in diff / se_diff))
    print(f"\nportfolio sd: Gaussian {np.std(pnl_g, ddof=1):,.0f}   "
          f"t {np.std(pnl_t, ddof=1):,.0f}   historical {np.std(pnl_h, ddof=1):,.0f}")
    print("\nprobability the simulated portfolio loses more than the Gaussian "
          "copula's VaR:")
    for lvl, j in ((0.05, 0), (0.01, 2)):
        thr = -rows["Gaussian copula"][j]
        print(f"  {lvl:.0%} VaR ({-thr:,.0f}): Gaussian {np.mean(pnl_g <= thr):.4f}, "
              f"t {np.mean(pnl_t <= thr):.4f}")
    worst = -np.quantile(pnl_g, 0.001), -np.quantile(pnl_t, 0.001)
    print(f"0.1% quantile of the loss: Gaussian {worst[0]:,.0f}, t {worst[1]:,.0f}")

    # ---- (f), (g) joint tail days the copulas imply ----------------------------
    rule("PROBLEM 4 (f) -- joint tail days per 1,000, implied by each copula")
    sim_g = joint_tail_counts(u_g, TAIL)
    sim_t = joint_tail_counts(u_t, TAIL)
    print(f"{'pair':<8}{'data lo':>9}{'data hi':>9}{'Gauss lo':>10}{'Gauss hi':>10}"
          f"{'t lo':>8}{'t hi':>8}")
    scale = 1000 / N_SIM
    tot = np.zeros(6)
    for p in pairs:
        row = np.array([counts[p][0], counts[p][1], sim_g[p][0] * scale,
                        sim_g[p][1] * scale, sim_t[p][0] * scale, sim_t[p][1] * scale])
        tot += row
        print(f"{names[p[0]] + '-' + names[p[1]]:<8}{row[0]:>9.0f}{row[1]:>9.0f}"
              f"{row[2]:>10.2f}{row[3]:>10.2f}{row[4]:>8.2f}{row[5]:>8.2f}")
    print(f"{'total':<8}{tot[0]:>9.0f}{tot[1]:>9.0f}{tot[2]:>10.2f}{tot[3]:>10.2f}"
          f"{tot[4]:>8.2f}{tot[5]:>8.2f}")
    print(f"\nboth tails together: data {tot[0] + tot[1]:.0f}, Gaussian "
          f"{tot[2] + tot[3]:.1f}, t {tot[4] + tot[5]:.1f}")
    # How surprising the data total is under each model: a pair-day count over
    # 1,000 days is close to Poisson with the implied mean.
    for label, mu in (("Gaussian", tot[2] + tot[3]), ("t", tot[4] + tot[5])):
        obs = tot[0] + tot[1]
        print(f"  under the {label:<8} P(count >= {obs:.0f}) = "
              f"{stats.poisson.sf(obs - 1, mu):.4f},  P(count <= {obs:.0f}) = "
              f"{stats.poisson.cdf(obs, mu):.4f}")

    # ---- (i) where the likelihood gain comes from ------------------------------
    rule("PROBLEM 4 (i) -- each day's contribution to ll_t - ll_Gauss")
    gain = t_copula_ll(u, r, nu_hat, per_obs=True) - gaussian_copula_ll(u, r, per_obs=True)
    # "Outer 5% on either side" is judged on the ranks, the same scale as (b).
    n_outer = np.sum((ranks <= OUTER) | (ranks >= 1 - OUTER), axis=1)
    print(f"total gain = {gain.sum():.3f}\n")
    print(f"{'series in outer 5%':<22}{'days':>6}{'sum of gain':>13}{'share':>8}"
          f"{'mean per day':>14}")
    for k in range(n + 1):
        sel = n_outer == k
        print(f"{k:<22d}{int(sel.sum()):>6d}{gain[sel].sum():>13.3f}"
              f"{gain[sel].sum() / gain.sum():>8.1%}{gain[sel].mean():>14.4f}")
    mid = n_outer == 0
    print(f"\ndays with no series in its outer 5%: {int(mid.sum())}, "
          f"contributing {gain[mid].sum():.3f} of {gain.sum():.3f} "
          f"({gain[mid].sum() / gain.sum():.1%})")
    print(f"days where the t scores higher: {int(np.sum(gain > 0))} of {m}")
    print(f"the 20 days with the largest gain contribute {np.sort(gain)[::-1][:20].sum():.3f}")
    if crash is not None:
        print(f"day {crash + 1} alone contributes {gain[crash]:.3f}")
    # Inside the middle block, where does the t win?  Split by how far the day
    # is from the centre of the copula, measured in normal scores.
    zs = stats.norm.ppf(u[mid])
    radius = np.sqrt(np.sum(zs ** 2, axis=1))
    inner = radius <= np.median(radius)
    print(f"middle days split at the median distance from the centre "
          f"(|z| = {np.median(radius):.2f}):")
    print(f"  inner half: gain {gain[mid][inner].sum():+.3f}   "
          f"outer half: gain {gain[mid][~inner].sum():+.3f}")

    # ---- (j) tail dependence ---------------------------------------------------
    rule("PROBLEM 4 (j) -- tail dependence coefficient")
    print(f"{'pair':<8}{'rho':>8}{'lambda, t copula':>18}{'lambda, Gaussian':>18}")
    for i, j in sorted(pairs, key=lambda p: -r[p]):
        print(f"{names[i] + '-' + names[j]:<8}{r[i, j]:>8.4f}"
              f"{tail_dependence_t(r[i, j], nu_hat):>18.4f}{0.0:>18.4f}")
    print(f"\n(at nu = {nu_hat:.4f})")
    # How well nu is pinned down: the range where the profile log likelihood is
    # within 1.92 (half the 95% chi-square(1) point) of its maximum.  A rough
    # likelihood interval, enough to show how far lambda could move.
    grid = 1.0 / np.linspace(0.05, 0.45, 801)
    ll_grid = np.array([t_copula_ll(u, r, v) for v in grid])
    inside = grid[ll_grid >= ll_t - 1.92]
    i, j = max(pairs, key=lambda p: r[p])
    print(f"nu within 1.92 log likelihood units of the maximum: "
          f"{inside.min():.2f} to {inside.max():.2f}")
    print(f"lambda for {names[i]}-{names[j]} across that range: "
          f"{tail_dependence_t(r[i, j], inside.max()):.4f} to "
          f"{tail_dependence_t(r[i, j], inside.min()):.4f}")

    # ---- figures ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (i, j) in zip(axes, pairs):
        ax.scatter(ranks[:, i], ranks[:, j], s=4, color="#14616e", alpha=0.5,
                   edgecolor="none")
        for lo in (0.0, 1 - TAIL):
            ax.add_patch(plt.Rectangle((lo, lo), TAIL, TAIL, fill=False,
                                       ec="#8a3a2b", lw=1.0))
        ax.set_title(f"{names[i]} against {names[j]}:  "
                     f"{counts[(i, j)][0]} low, {counts[(i, j)][1]} high")
        ax.set_xlabel(f"rank of {names[i]} / (m+1)")
        ax.set_ylabel(f"rank of {names[j]} / (m+1)")
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.tight_layout()
    path = os.path.join(FIGDIR, "problem4_ranks.png")
    fig.savefig(path, dpi=160)
    print(f"\nwrote {path}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    ax = axes[0]
    grid = np.linspace(2.2, 60, 200)
    ax.plot(grid, [t_copula_ll(u, r, v) for v in grid], color="#14616e", lw=1.5,
            label="t copula")
    ax.axhline(ll_g, color="#8a3a2b", lw=1.2, ls="--", label="Gaussian copula")
    ax.axvline(nu_hat, color="#777777", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("copula nu (log scale)")
    ax.set_ylabel("copula log likelihood")
    ax.set_title(f"Profile likelihood, maximum at nu = {nu_hat:.2f}")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    for k, colour in zip(range(n + 1), ("#14616e", "#6f9ea6", "#d08a3a", "#8a3a2b")):
        sel = n_outer == k
        ax.scatter(np.flatnonzero(sel) + 1, gain[sel], s=6, color=colour,
                   label=f"{k} series in outer 5% ({int(sel.sum())} days)")
    ax.axhline(0, color="#999999", lw=0.6)
    ax.set_xlabel("day")
    ax.set_ylabel("ln c_t - ln c_Gauss")
    ax.set_title("Each day's contribution to the likelihood gap")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    path = os.path.join(FIGDIR, "problem4_fit.png")
    fig.savefig(path, dpi=160)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
