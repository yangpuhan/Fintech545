"""
FinTech 545 Mid Term -- code for Problems 1, 2 and 3.

Run from anywhere:

    python midterm.py                      # Problems 1, 2, 3(a); stops before 3(b)
    python midterm.py --margins S1=normal,S2=t,...,S10=nig            # adds 3(b)
    python midterm.py --margins ... --copula gaussian                 # adds 3(c)

The choices asked for in the written answers -- the margin for each stock in
3(a) and the copula in 3(b) -- are inputs, set either in the CHOICES block
below or on the command line.  Every table printed is also written to
output/ as a CSV.

Builds on the riskmgmt library (Fintech545/riskmgmt) and the Week 04/05
helpers in Fintech545/Assignment2/common.py.  Conventions: arithmetic
returns; VaR and ES positive for a loss; historical quantile follows the
class rule (common.var_es).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# Bundled copies (submission folder) first, then the course / homework repos.
DATA = HERE / "data" if (HERE / "data").is_dir() else ROOT / "FinTech-545-Fall2026" / "MidTerm"
LIB = HERE / "lib" if (HERE / "lib").is_dir() else None
REPO = ROOT / "Fintech545"
for p in ((LIB,) if LIB else (REPO, REPO / "Assignment2")):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common import (copula_criteria, gaussian_copula_ll,  # noqa: E402
                    kendall_correlation, normal_var_es, profile_nu,
                    simulate_copula_uniforms, t_copula_ll, var_es)
from riskmgmt import (aicc, fit_general_t, fit_nig_mle, fit_normal,  # noqa: E402
                      higham_nearest_psd, missing_cov, nig_params,
                      return_calculate)

# ---------------------------------------------------------------------------
# CHOICES -- fill these in from your written answers.  The command line
# (--margins, --copula) overrides them.
# ---------------------------------------------------------------------------
P1_CORR_FOR_COV = "higham"   # Problem 1(e): correlation used, "higham" or "pairwise"
P3_MARGINS = {               # Problem 3(a): lowest AICc for each stock
    "S1": "normal", "S2": "t", "S3": "nig", "S4": "normal", "S5": "t",
    "S6": "nig", "S7": "normal", "S8": "t", "S9": "nig", "S10": "t",
}
P3_COPULA = "gaussian"       # Problem 3(b): Gaussian beats t on AICc / BIC (nu -> upper bound)
P3_DEMEAN = False            # Problem 3: subtract each stock's mean return before fitting
P3_ALPHA = 0.05              # Problem 3(c): 95% level
P2_ALPHAS = (0.05, 0.01)     # Problem 2: 95% and 99% levels
N_SIM = 100_000              # simulation draws (question asks for at least 5,000)
SEED = 1234

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", lambda v: f"{v:,.6g}")


def header(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def save(df, out, name):
    df.to_csv(out / f"{name}.csv")


# ===========================================================================
# Problem 1 -- pairwise correlation with missing data
# ===========================================================================

def problem1(out):
    x = pd.read_csv(DATA / "problem1.csv")
    names = list(x.columns)
    a = x.to_numpy(dtype=float)

    header("Problem 1 -- data")
    print(f"rows = {len(x)}, complete rows = {int((~np.isnan(a)).all(axis=1).sum())}")
    print("observations per series:", dict(zip(names, (~np.isnan(a)).sum(axis=0).tolist())))

    # Problem 1(a): pairwise correlation -- entry (i, j) uses the rows where
    # both x_i and x_j are present.
    corr = missing_cov(a, skip_miss=False, corr=True)
    corr_df = pd.DataFrame(corr, index=names, columns=names)
    header("Problem 1(a) -- pairwise correlation matrix")
    print(corr_df)
    save(corr_df, out, "p1a_pairwise_corr")

    # Problem 1(b): variance of each series from every observation it has
    # (n - 1 denominator).
    var = pd.Series(np.nanvar(a, axis=0, ddof=1), index=names, name="variance")
    header("Problem 1(b) -- variance of each series (all available observations, ddof=1)")
    print(pd.DataFrame({"n_obs": (~np.isnan(a)).sum(axis=0), "variance": var.values,
                        "std_dev": np.sqrt(var.values)}, index=names))
    save(var.to_frame(), out, "p1b_variance")

    # Problem 1(c): eigenvalues of the pairwise correlation matrix, and
    # whether an ordinary Cholesky factorization succeeds.
    eig = np.sort(np.linalg.eigvalsh(corr))[::-1]
    try:
        np.linalg.cholesky(corr)
        chol = "succeeds"
    except np.linalg.LinAlgError:
        chol = "fails (LinAlgError)"
    header("Problem 1(c) -- eigenvalues of the pairwise correlation matrix")
    print(pd.Series(eig, index=[f"lambda{i + 1}" for i in range(len(eig))], name="eigenvalue")
          .to_string(float_format=lambda v: f"{v: .8e}"))
    print(f"\nsmallest eigenvalue = {eig[-1]:.8e}")
    print(f"np.linalg.cholesky  {chol}")
    save(pd.DataFrame({"eigenvalue": eig}), out, "p1c_eigenvalues_pairwise")

    # Problem 1(d): Higham (2002) nearest correlation matrix, its eigenvalues,
    # and the change in every entry.
    fixed = higham_nearest_psd(corr)
    fixed_df = pd.DataFrame(fixed, index=names, columns=names)
    eig_f = np.sort(np.linalg.eigvalsh(fixed))[::-1]
    diff = fixed - corr
    header("Problem 1(d) -- Higham nearest PSD correlation matrix")
    print(fixed_df)
    print("\neigenvalues:")
    print(pd.Series(eig_f, index=[f"lambda{i + 1}" for i in range(len(eig_f))], name="eigenvalue")
          .to_string(float_format=lambda v: f"{v: .8e}"))
    print(f"\ndiagonal after Higham: {np.diag(fixed).round(12).tolist()}")
    print("\nchange, Higham minus pairwise:")
    print(pd.DataFrame(diff, index=names, columns=names)
          .to_string(float_format=lambda v: f"{v: .6f}"))
    pairs = [(names[i], names[j], corr[i, j], fixed[i, j], diff[i, j])
             for i in range(len(names)) for j in range(i + 1, len(names))]
    ranked = (pd.DataFrame(pairs, columns=["i", "j", "pairwise", "higham", "change"])
              .assign(abs_change=lambda d: d["change"].abs())
              .sort_values("abs_change", ascending=False)
              .reset_index(drop=True))
    print("\noff-diagonal entries ranked by |change|:")
    print(ranked.to_string(float_format=lambda v: f"{v: .6f}"))
    print(f"\nFrobenius norm of the change = {np.linalg.norm(diff, 'fro'):.6e}")
    save(fixed_df, out, "p1d_higham_corr")
    save(pd.DataFrame({"eigenvalue": eig_f}), out, "p1d_eigenvalues_higham")
    save(ranked, out, "p1d_changes_ranked")

    # Problem 1(e): covariance = D R D, D = diag of the 1(b) standard deviations.
    r = fixed if P1_CORR_FOR_COV == "higham" else corr
    sd = np.sqrt(var.values)
    cov = sd[:, None] * r * sd[None, :]
    cov_df = pd.DataFrame(cov, index=names, columns=names)
    header(f"Problem 1(e) -- covariance matrix (correlation: {P1_CORR_FOR_COV}, "
           "variances: 1(b))")
    print(cov_df.to_string(float_format=lambda v: f"{v: .8e}"))
    print("\neigenvalues:", np.sort(np.linalg.eigvalsh(cov))[::-1])
    save(cov_df, out, "p1e_covariance")


# ===========================================================================
# Problem 2 -- VaR and ES of 100 shares of XYZ, r_XYZ = alpha + beta r_SPY + e
# ===========================================================================

def problem2(out):
    prices = pd.read_csv(DATA / "problem2.csv")
    r = return_calculate(prices, method="DISCRETE", date_column="Date")
    spy = r["SPY"].to_numpy(dtype=float)
    xyz = r["XYZ"].to_numpy(dtype=float)
    n = len(spy)
    p_xyz = float(prices["XYZ"].iloc[-1])
    value = 100.0 * p_xyz

    # Problem 2 (setup): OLS fit of the structural equation.
    X = np.column_stack([np.ones(n), spy])
    coef, *_ = np.linalg.lstsq(X, xyz, rcond=None)
    a_hat, b_hat = float(coef[0]), float(coef[1])
    e = xyz - X @ coef
    s_e = float(np.std(e, ddof=2))               # OLS residual sd, n - 2
    se = s_e * np.sqrt(np.diag(np.linalg.inv(X.T @ X)))
    r2 = 1.0 - np.sum(e ** 2) / np.sum((xyz - xyz.mean()) ** 2)

    # SPY ~ N(mu_s, sigma_s^2), fitted by riskmgmt.fit_normal (ddof=1).
    f_spy = fit_normal(spy)
    mu_s, s_s = float(f_spy.error_model.mean()), float(f_spy.error_model.std())

    # Implied distribution of r_XYZ: normal, since it is linear in two
    # independent normals.
    mu_x = a_hat + b_hat * mu_s
    s_x = float(np.sqrt(b_hat ** 2 * s_s ** 2 + s_e ** 2))

    header("Problem 2 -- fitted structural model")
    print(f"returns n = {n}; last XYZ price = {p_xyz:.4f}; 100 shares = ${value:,.2f}")
    print(f"alpha = {a_hat: .6e}  (se {se[0]:.6e}, t = {a_hat / se[0]: .3f})")
    print(f"beta  = {b_hat: .6f}  (se {se[1]:.6f}, t = {b_hat / se[1]: .3f})")
    print(f"R^2   = {r2:.4f}")
    print(f"residual sd (n-2)        sigma_e = {s_e:.6f}")
    print(f"SPY mean / sd (ddof=1)   mu_s = {mu_s: .6e}, sigma_s = {s_s:.6f}")
    print(f"implied r_XYZ            mu = {mu_x: .6e}, sigma = {s_x:.6f}")
    print(f"  (sample XYZ return mean / sd for reference: {xyz.mean(): .6e} / {xyz.std(ddof=1):.6f})")
    # Normality checks on SPY and the residuals (Jarque-Bera, skew, excess kurtosis).
    for lab, v in (("SPY", spy), ("residual", e)):
        jb = stats.jarque_bera(v)
        print(f"  {lab:8s} skew = {stats.skew(v): .4f}, excess kurt = {stats.kurtosis(v): .4f}, "
              f"JB p-value = {jb.pvalue:.4f}")

    # Problem 2(a) and 2(b): closed-form normal VaR and ES of the return, in
    # percent of the position, and in dollars.  "absolute" uses the implied
    # mean; "relative to mean" sets it to zero.
    rows = []
    for alpha in P2_ALPHAS:
        for lab, mu in (("absolute", mu_x), ("relative to mean", 0.0)):
            v, es = normal_var_es(mu, s_x, alpha)
            rows.append({"level": f"{1 - alpha:.0%}", "basis": lab, "method": "closed form",
                         "VaR_%": 100 * v, "ES_%": 100 * es,
                         "VaR_$": value * v, "ES_$": value * es})

    # Cross-check by simulation of the same model: draw r_SPY and e, build r_XYZ.
    rng = np.random.default_rng(SEED)
    sim = (a_hat + b_hat * rng.normal(mu_s, s_s, N_SIM) + rng.normal(0.0, s_e, N_SIM))
    for alpha in P2_ALPHAS:
        v, es = var_es(sim, alpha)
        rows.append({"level": f"{1 - alpha:.0%}", "basis": "absolute",
                     "method": f"simulation ({N_SIM:,})",
                     "VaR_%": 100 * v, "ES_%": 100 * es,
                     "VaR_$": value * v, "ES_$": value * es})
    tab = pd.DataFrame(rows)
    header("Problem 2(a)/(b) -- VaR and ES of 100 shares of XYZ (positive = loss)")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    save(tab.set_index(["level", "basis", "method"]), out, "p2_var_es")


# ===========================================================================
# Problem 3 -- copula VaR and ES for a two book portfolio
# ===========================================================================

def nig_fast_ppf(d, width=30.0, n_grid=60_001):
    """NIG quantile by interpolation on a CDF grid (scipy's ppf is slow and
    its root finder can fail deep in the tail).  The grid spans mean +/-
    `width` standard deviations."""
    m, s = float(d.mean()), float(d.std())
    xs = np.linspace(m - width * s, m + width * s, n_grid)
    cs = np.maximum.accumulate(d.cdf(xs))
    return lambda u: np.interp(u, cs, xs)


def load_p3():
    prices = pd.read_csv(DATA / "problem3.csv")
    hold = pd.read_csv(DATA / "problem3_holdings.csv")
    names = list(hold["Stock"])
    r = return_calculate(prices, method="DISCRETE", date_column="Date")[names]
    if P3_DEMEAN:
        r = r - r.mean()
    last = prices[names].iloc[-1].astype(float)
    value = pd.Series(hold["Holding"].to_numpy(dtype=float) * last.to_numpy(),
                      index=names)
    book = pd.Series(hold["Book"].to_numpy(), index=names)
    return r, hold, last, value, book


def problem3a(r, out):
    """Problem 3(a): fit normal, t and NIG to every stock; report parameters
    and fit statistics.  No selection is made here."""
    n = len(r)
    fits, rows = {}, []
    for s in r.columns:
        x = r[s].to_numpy(dtype=float)
        fn, ft, fg = fit_normal(x), fit_general_t(x), fit_nig_mle(x)
        fits[s] = {"normal": fn, "t": ft, "nig": fg}
        tk = ft.error_model.kwds
        mu_g, al_g, be_g, de_g = nig_params(fg)
        for lab, f, k, params in (
            ("normal", fn, 2, {"mu": fn.error_model.mean(), "sigma": fn.error_model.std()}),
            ("t", ft, 3, {"mu": tk["loc"], "sigma": tk["scale"], "nu": tk["df"]}),
            ("nig", fg, 4, {"mu": mu_g, "alpha": al_g, "beta": be_g, "delta": de_g}),
        ):
            ll = f.loglik(x)
            rows.append({"stock": s, "dist": lab, **params, "loglik": ll, "k": k,
                         "AIC": 2 * k - 2 * ll, "AICc": aicc(ll, k, n),
                         "BIC": k * np.log(n) - 2 * ll})
    tab = pd.DataFrame(rows)[["stock", "dist", "mu", "sigma", "nu", "alpha", "beta", "delta",
                              "loglik", "k", "AIC", "AICc", "BIC"]]
    header(f"Problem 3(a) -- marginal fits, n = {n} returns per stock"
           + ("  [returns de-meaned]" if P3_DEMEAN else ""))
    print(tab.to_string(index=False, na_rep="", float_format=lambda v: f"{v:.6g}"))

    # Sample moments, for reference.
    mom = pd.DataFrame({"mean": r.mean(), "std": r.std(ddof=1),
                        "skew": r.apply(lambda c: stats.skew(c)),
                        "excess_kurt": r.apply(lambda c: stats.kurtosis(c))})
    print("\nsample moments:")
    print(mom.to_string(float_format=lambda v: f"{v: .5f}"))
    save(tab.set_index(["stock", "dist"]), out, "p3a_marginal_fits")
    save(mom, out, "p3a_sample_moments")
    return fits


def problem3b(r, fits, margins, out):
    """Problem 3(b): Gaussian and t copula on the U from the chosen margins.
    R from Kendall's tau (shared by both); nu by profile likelihood on 1/nu,
    as in Week05/choosing_between_copulas.jl."""
    names = list(r.columns)
    U = np.column_stack([fits[s][margins[s]].u for s in names])
    m = U.shape[0]
    R, min_eig, _ = kendall_correlation(U)

    ll_g = gaussian_copula_ll(U, R)
    nu_hat, ll_t = profile_nu(lambda nu: t_copula_ll(U, R, nu))
    aicc_g, bic_g = copula_criteria(ll_g, 0, m)
    aicc_t, bic_t = copula_criteria(ll_t, 1, m)

    header("Problem 3(b) -- copula fits")
    print("margins used:", margins)
    print(f"R from Kendall's tau, sin(pi tau / 2); smallest eigenvalue before repair = "
          f"{min_eig:.4e}")
    print(pd.DataFrame(R, index=names, columns=names).to_string(float_format=lambda v: f"{v: .4f}"))
    tab = pd.DataFrame({"Gaussian": [ll_g, 0, np.nan, aicc_g, bic_g],
                        "t": [ll_t, 1, nu_hat, aicc_t, bic_t]},
                       index=["loglik", "k (copula params)", "nu", "AICc", "BIC"])
    print(f"\nm = {m} observations, n = {len(names)} variables; nu searched on (2.04, 100]")
    print(tab.to_string(float_format=lambda v: f"{v:,.4f}"))
    print(f"\n2 * (ll_t - ll_g) = {2 * (ll_t - ll_g):.4f};  ln(m) = {np.log(m):.4f}")
    print("\nt copula log likelihood profile:")
    for nu in (3, 4, 6, 8, 12, 20, 30, 50, 100):
        print(f"  nu = {nu:5.1f}   ll = {t_copula_ll(U, R, float(nu)):10.4f}")
    save(pd.DataFrame(R, index=names, columns=names), out, "p3b_copula_R")
    save(tab, out, "p3b_copula_fits")
    return R, nu_hat


def simulate_pnl(names, fits, margins, R, nu, value, seed):
    """Copula uniforms -> chosen margins -> dollar P&L per position."""
    U = simulate_copula_uniforms(R, N_SIM, nu=nu, seed=seed)
    sim_r = np.empty_like(U)
    for i, s in enumerate(names):
        d = fits[s][margins[s]].error_model
        ppf = nig_fast_ppf(d) if margins[s] == "nig" else d.ppf
        sim_r[:, i] = ppf(U[:, i])
    return pd.DataFrame(sim_r * value.to_numpy()[None, :], columns=names)


def var_es_table(pnl, value, book):
    names = list(pnl.columns)
    rows = []
    for s in names:
        v, es = var_es(pnl[s].to_numpy(), P3_ALPHA)
        rows.append({"position": s, "book": book[s], "value": value[s], "VaR": v, "ES": es})
    for b in sorted(book.unique()):
        members = [s for s in names if book[s] == b]
        v, es = var_es(pnl[members].sum(axis=1).to_numpy(), P3_ALPHA)
        rows.append({"position": f"Book {b}", "book": b, "value": value[members].sum(),
                     "VaR": v, "ES": es})
    v, es = var_es(pnl.sum(axis=1).to_numpy(), P3_ALPHA)
    rows.append({"position": "Total", "book": "", "value": value.sum(), "VaR": v, "ES": es})
    tab = pd.DataFrame(rows)
    tab["VaR_%"] = 100 * tab["VaR"] / tab["value"]
    tab["ES_%"] = 100 * tab["ES"] / tab["value"]
    return tab.set_index("position")


def problem3c(r, fits, margins, copula, R, nu_hat, value, book, out):
    """Problem 3(c): simulate the chosen copula, map through the chosen
    margins, revalue, and take VaR / ES per stock, per book and in total."""
    names = list(r.columns)
    nu = nu_hat if copula == "t" else None
    pnl = simulate_pnl(names, fits, margins, R, nu, value, SEED)
    tab = var_es_table(pnl, value, book).reset_index()

    lev = f"{1 - P3_ALPHA:.0%}"
    header(f"Problem 3(c) -- {copula} copula, {N_SIM:,} draws, absolute VaR / ES at {lev} "
           "($, positive = loss)")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    save(tab.set_index("position"), out, "p3c_var_es")

    # Problem 3(c) discussion (1): diversification -- sum of standalone
    # numbers versus the book / total number.
    t = tab.set_index("position")
    div = []
    for b in sorted(book.unique()):
        members = [s for s in names if book[s] == b]
        div.append((f"Book {b}", t.loc[members, "VaR"].sum(), t.loc[f"Book {b}", "VaR"],
                    t.loc[members, "ES"].sum(), t.loc[f"Book {b}", "ES"]))
    div.append(("Total (books)", t.loc[["Book A", "Book B"], "VaR"].sum(), t.loc["Total", "VaR"],
                t.loc[["Book A", "Book B"], "ES"].sum(), t.loc["Total", "ES"]))
    div.append(("Total (stocks)", t.loc[names, "VaR"].sum(), t.loc["Total", "VaR"],
                t.loc[names, "ES"].sum(), t.loc["Total", "ES"]))
    div = pd.DataFrame(div, columns=["level", "sum_VaR", "VaR", "sum_ES", "ES"])
    div["VaR_benefit_%"] = 100 * (1 - div["VaR"] / div["sum_VaR"])
    div["ES_benefit_%"] = 100 * (1 - div["ES"] / div["sum_ES"])
    header("Problem 3(c) discussion -- diversification benefit")
    print(div.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    save(div.set_index("level"), out, "p3c_diversification")

    # Problem 3(c) discussion (2): same copula with normal margins for every
    # stock, to isolate the effect of the fat-tailed / skewed margins.
    pnl_n = simulate_pnl(names, fits, {s: "normal" for s in names}, R, nu, value, SEED)
    tn = var_es_table(pnl_n, value, book)
    cmp_ = pd.DataFrame({"VaR_chosen": t["VaR"], "VaR_all_normal": tn["VaR"],
                         "ES_chosen": t["ES"], "ES_all_normal": tn["ES"]})
    header("Problem 3(c) discussion -- chosen margins vs all-normal margins (same copula)")
    print(cmp_.to_string(float_format=lambda v: f"{v:,.2f}"))
    save(cmp_, out, "p3c_vs_normal")

    # Problem 3(c) discussion (3): simulation noise -- rerun with other seeds.
    reps = [var_es_table(simulate_pnl(names, fits, margins, R, nu, value, SEED + k),
                         value, book)[["VaR", "ES"]] for k in range(1, 6)]
    rep = pd.concat(reps + [t[["VaR", "ES"]]], keys=range(6))
    spread = rep.groupby(level=1).agg(["mean", "std"]).reindex(t.index)
    header("Problem 3(c) discussion -- 6 seeds x %s draws: mean and std of VaR / ES" % f"{N_SIM:,}")
    print(spread.to_string(float_format=lambda v: f"{v:,.2f}"))
    save(spread, out, "p3c_seed_spread")


def problem3(out, margins, copula):
    r, hold, last, value, book = load_p3()
    header("Problem 3 -- positions at the last price")
    print(pd.DataFrame({"book": book, "shares": hold.set_index("Stock")["Holding"],
                        "last_price": last, "value": value})
          .to_string(float_format=lambda v: f"{v:,.4f}"))
    for b in sorted(book.unique()):
        print(f"Book {b} value = ${value[book == b].sum():,.2f}")
    print(f"Total value  = ${value.sum():,.2f}")

    fits = problem3a(r, out)
    if margins is None:
        print("\n[stopped] set P3_MARGINS or pass --margins to run 3(b)")
        return
    missing = [s for s in r.columns if margins.get(s) not in ("normal", "t", "nig")]
    if missing:
        raise SystemExit(f"--margins needs normal / t / nig for every stock; missing: {missing}")
    R, nu_hat = problem3b(r, fits, margins, out)
    if copula is None:
        print("\n[stopped] set P3_COPULA or pass --copula to run 3(c)")
        return
    if copula not in ("gaussian", "t"):
        raise SystemExit("--copula must be gaussian or t")
    problem3c(r, fits, margins, copula, R, nu_hat, value, book, out)


def parse_margins(text):
    if not text:
        return None
    return dict(item.split("=") for item in text.split(","))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problems", default="1,2,3", help="which problems to run, e.g. 3")
    ap.add_argument("--margins", default=None,
                    help="Problem 3 margins, e.g. S1=normal,S2=t,S3=nig,...")
    ap.add_argument("--copula", default=None, help="Problem 3(c) copula: gaussian or t")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "output"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    margins = parse_margins(args.margins) or P3_MARGINS
    copula = args.copula or P3_COPULA
    todo = set(args.problems.split(","))

    if "1" in todo:
        problem1(out)
    if "2" in todo:
        problem2(out)
    if "3" in todo:
        problem3(out, margins, copula)


if __name__ == "__main__":
    main()
