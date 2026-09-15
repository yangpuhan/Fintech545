#!/usr/bin/env python3
"""Functional tests 1.1 through 7.4, checked against the class expected output.

    python3 Tests/run_tests.py                  # all of them
    python3 Tests/run_tests.py 3.1 3.3          # just these
    python3 Tests/run_tests.py --data <dir>     # point at the test data

The test data lives in the class repository and is not copied into this one.
The directory is found, in order, from --data, then the FINTECH545_TESTFILES
environment variable, then a sibling checkout of the class repo.

Tolerances.  Most cases are deterministic and are checked to 1e-8.  The
exceptions are stated at the case that uses them:

  5.1 - 5.5  simulation, so the comparison is against the *input* covariance
             and the tolerance is set by 100,000 draws, not by the arithmetic.
  7.2 - 7.4  the class answers come from Ipopt and these from scipy, so the
             trailing digits of a numerical optimum need not agree.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from riskmgmt import (aicc, chol_psd, ew_cov_var_corr, ew_covar, cov_to_corr,
                      fit_general_t, fit_normal, fit_regression_t,
                      higham_nearest_psd, missing_cov, near_psd,
                      return_calculate, simulate_normal, simulate_pca)

CASES = {}


def case(name, description, tol=1e-8, note=""):
    """Register a test case.  The function returns (got, expected)."""
    def wrap(fn):
        CASES[name] = dict(fn=fn, description=description, tol=tol, note=note)
        return fn
    return wrap


def find_data_dir(explicit=None):
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("FINTECH545_TESTFILES"):
        candidates.append(Path(os.environ["FINTECH545_TESTFILES"]))
    here = Path(__file__).resolve().parent
    candidates += [
        here.parent.parent / "FinTech-545-Fall2026" / "testfiles" / "data",
        here.parent.parent.parent / "FinTech-545-Fall2026" / "testfiles" / "data",
    ]
    for c in candidates:
        if c.is_dir() and (c / "test1.csv").exists():
            return c
    raise SystemExit(
        "Could not find the test data.  Pass --data <dir>, set "
        "FINTECH545_TESTFILES, or check out the class repository\n"
        "(dompazz/FinTech-545-Fall2026) next to this one.  Looked in:\n  "
        + "\n  ".join(str(c) for c in candidates))


DATA = None


def read(name):
    return pd.read_csv(DATA / name)


def read_matrix(name):
    return read(name).to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# 1.x  covariance and correlation with missing data
# ---------------------------------------------------------------------------

@case("1.1", "Covariance, missing data, skip missing rows")
def t1_1():
    x = read_matrix("test1.csv")
    return missing_cov(x, skip_miss=True, corr=False), read_matrix("testout_1.1.csv")


@case("1.2", "Correlation, missing data, skip missing rows")
def t1_2():
    x = read_matrix("test1.csv")
    return missing_cov(x, skip_miss=True, corr=True), read_matrix("testout_1.2.csv")


@case("1.3", "Covariance, missing data, pairwise")
def t1_3():
    x = read_matrix("test1.csv")
    return missing_cov(x, skip_miss=False, corr=False), read_matrix("testout_1.3.csv")


@case("1.4", "Correlation, missing data, pairwise")
def t1_4():
    x = read_matrix("test1.csv")
    return missing_cov(x, skip_miss=False, corr=True), read_matrix("testout_1.4.csv")


# ---------------------------------------------------------------------------
# 2.x  exponentially weighted
# ---------------------------------------------------------------------------

@case("2.1", "EW covariance, lambda = 0.97")
def t2_1():
    x = read_matrix("test2.csv")
    return ew_covar(x, 0.97), read_matrix("testout_2.1.csv")


@case("2.2", "EW correlation, lambda = 0.94")
def t2_2():
    x = read_matrix("test2.csv")
    return cov_to_corr(ew_covar(x, 0.94)), read_matrix("testout_2.2.csv")


@case("2.3", "EW variance (0.97) with EW correlation (0.94)")
def t2_3():
    x = read_matrix("test2.csv")
    return ew_cov_var_corr(x, lam_var=0.97, lam_corr=0.94), read_matrix("testout_2.3.csv")


# ---------------------------------------------------------------------------
# 3.x / 4.1  repairing and factoring
# ---------------------------------------------------------------------------

@case("3.1", "near_psd on a covariance")
def t3_1():
    return near_psd(read_matrix("testout_1.3.csv")), read_matrix("testout_3.1.csv")


@case("3.2", "near_psd on a correlation")
def t3_2():
    return near_psd(read_matrix("testout_1.4.csv")), read_matrix("testout_3.2.csv")


@case("3.3", "Higham nearest PSD on a covariance")
def t3_3():
    return higham_nearest_psd(read_matrix("testout_1.3.csv")), read_matrix("testout_3.3.csv")


@case("3.4", "Higham nearest PSD on a correlation")
def t3_4():
    return higham_nearest_psd(read_matrix("testout_1.4.csv")), read_matrix("testout_3.4.csv")


@case("4.1", "chol_psd")
def t4_1():
    return chol_psd(read_matrix("testout_3.1.csv")), read_matrix("testout_4.1.csv")


# ---------------------------------------------------------------------------
# 5.x  simulation
#
# These compare a simulated covariance against the covariance that generated
# it.  With 100,000 draws the sampling error on each entry is roughly
# sigma_ii*sigma_jj*sqrt(2/N) ~ 0.5% of the entry, so the tolerance is relative
# and loose.  The class expected-output files are one particular draw from
# Julia's RNG and cannot be reproduced from Python; matching the *input* is the
# actual test, and the expected outputs match their input to the same order.
# ---------------------------------------------------------------------------

SIM_NOTE = "compared against the input covariance, not the class draw"


@case("5.1", "Normal simulation, PD input", tol=2e-3, note=SIM_NOTE)
def t5_1():
    cin = read_matrix("test5_1.csv")
    return np.cov(simulate_normal(100_000, cin), rowvar=False, ddof=1), cin


@case("5.2", "Normal simulation, PSD input", tol=2e-3, note=SIM_NOTE)
def t5_2():
    cin = read_matrix("test5_2.csv")
    return np.cov(simulate_normal(100_000, cin), rowvar=False, ddof=1), cin


@case("5.3", "Normal simulation, non-PSD input, near_psd fix", tol=2e-3,
      note="compared against near_psd(input)")
def t5_3():
    cin = read_matrix("test5_3.csv")
    got = np.cov(simulate_normal(100_000, cin, fix_method=near_psd), rowvar=False, ddof=1)
    return got, near_psd(cin)


@case("5.4", "Normal simulation, non-PSD input, Higham fix", tol=2e-3,
      note="compared against higham_nearest_psd(input)")
def t5_4():
    cin = read_matrix("test5_3.csv")
    got = np.cov(simulate_normal(100_000, cin, fix_method=higham_nearest_psd),
                 rowvar=False, ddof=1)
    return got, higham_nearest_psd(cin)


@case("5.5", "PCA simulation, 99% explained", tol=2e-3, note=SIM_NOTE)
def t5_5():
    cin = read_matrix("test5_2.csv")
    return np.cov(simulate_pca(cin, 100_000, pct_exp=0.99), rowvar=False, ddof=1), cin


# ---------------------------------------------------------------------------
# 6.x  returns
# ---------------------------------------------------------------------------

@case("6.1", "Arithmetic returns")
def t6_1():
    prices = read("test6.csv")
    got = return_calculate(prices, method="DISCRETE", date_column="Date")
    exp = read("testout6_1.csv")
    return got.drop(columns="Date").to_numpy(), exp.drop(columns="Date").to_numpy()


@case("6.2", "Log returns")
def t6_2():
    prices = read("test6.csv")
    got = return_calculate(prices, method="LOG", date_column="Date")
    exp = read("testout6_2.csv")
    return got.drop(columns="Date").to_numpy(), exp.drop(columns="Date").to_numpy()


# ---------------------------------------------------------------------------
# 7.x  fitting
#
# 7.1 is closed form and is checked tightly.  7.2 through 7.4 are numerical
# optima; the class answers come from Ipopt and these from scipy's Nelder-Mead,
# so they agree to about six figures rather than to machine precision.
# ---------------------------------------------------------------------------

OPT_NOTE = "numerical optimum; class answer from Ipopt, this from scipy"


@case("7.1", "Fit a normal distribution")
def t7_1():
    x = read_matrix("test7_1.csv")[:, 0]
    fm = fit_normal(x)
    got = np.array([fm.error_model.mean(), fm.error_model.std()])
    exp = read("testout7_1.csv")
    return got, np.array([exp["mu"][0], exp["sigma"][0]])


@case("7.2", "Fit a generalized t distribution", tol=1e-5, note=OPT_NOTE)
def t7_2():
    x = read_matrix("test7_2.csv")[:, 0]
    fm = fit_general_t(x)
    d = fm.error_model
    got = np.array([d.kwds["loc"], d.kwds["scale"], d.kwds["df"]])
    exp = read("testout7_2.csv")
    return got, np.array([exp["mu"][0], exp["sigma"][0], exp["nu"][0]])


@case("7.3", "Fit a t regression", tol=1e-4, note=OPT_NOTE)
def t7_3():
    d = read("test7_3.csv")
    y = d["y"].to_numpy()
    x = d[["x1", "x2", "x3"]].to_numpy()
    fm = fit_regression_t(y, x)
    e = fm.error_model
    got = np.concatenate([[e.kwds["loc"], e.kwds["scale"], e.kwds["df"]], fm.beta])
    exp = read("testout7_3.csv")
    return got, np.array([exp["mu"][0], exp["sigma"][0], exp["nu"][0],
                          exp["Alpha"][0], exp["B1"][0], exp["B2"][0], exp["B3"][0]])


@case("7.4", "AICc on the fitted t", tol=1e-5, note=OPT_NOTE)
def t7_4():
    x = read_matrix("test7_2.csv")[:, 0]
    fm = fit_general_t(x)
    # k = 3: the generalized t carries mu, sigma and nu.  No regression
    # coefficients here, so nothing is added to the count.
    got = np.array([aicc(fm.loglik(x), k=3, n=len(x))])
    return got, np.array([read("testout7_4.csv")["AICC"][0]])


# ---------------------------------------------------------------------------

def run(selected=None, verbose=False):
    names = selected or sorted(CASES, key=lambda s: [int(p) for p in s.split(".")])
    width = max(len(CASES[n]["description"]) for n in names)
    passed = failed = 0

    print(f"data: {DATA}\n")
    for name in names:
        spec = CASES[name]
        got, exp = spec["fn"]()
        got, exp = np.asarray(got, dtype=float), np.asarray(exp, dtype=float)
        scale = np.maximum(np.abs(exp), 1.0)
        err = float(np.max(np.abs(got - exp) / scale))
        ok = err <= spec["tol"]
        passed, failed = passed + ok, failed + (not ok)

        flag = "PASS" if ok else "FAIL"
        note = f"   [{spec['note']}]" if spec["note"] else ""
        print(f"  {name:>4}  {flag}  {spec['description']:<{width}}  "
              f"max rel err {err:.2e}  (tol {spec['tol']:.0e}){note}")
        if verbose or not ok:
            print(f"        got:\n{np.array2string(got, precision=8, prefix='        ')}")
            print(f"        expected:\n{np.array2string(exp, precision=8, prefix='        ')}")

    print(f"\n{passed} passed, {failed} failed, {len(names)} total")
    return failed


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tests", nargs="*", help="test numbers, e.g. 3.1 3.3")
    ap.add_argument("--data", help="directory holding the class test CSVs")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print the matrices even when a case passes")
    args = ap.parse_args()

    DATA = find_data_dir(args.data)
    unknown = [t for t in args.tests if t not in CASES]
    if unknown:
        raise SystemExit(f"unknown test(s): {', '.join(unknown)}\n"
                         f"known: {', '.join(sorted(CASES))}")
    sys.exit(run(args.tests or None, args.verbose))
