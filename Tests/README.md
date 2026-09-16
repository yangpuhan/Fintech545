# Functional tests

Tests 1.1 through 7.6, implemented in [`../riskmgmt/`](../riskmgmt) and checked
against the expected output in the class repository.

```bash
python3 Tests/run_tests.py              # all 25
python3 Tests/run_tests.py 3.1 3.3      # selected cases
python3 Tests/run_tests.py -v 4.1       # print the matrices too
```

Run from the repository root. Needs `numpy`, `pandas`, `scipy`. Nothing else
has to be checked out or configured.

## Test data

`Tests/data` holds the 35 input and expected-output CSVs these cases use,
copied from the class repository. To re-check against the originals instead,
pass `--data <dir>` or set `$FINTECH545_TESTFILES`.

## Current status

```
  1.1  PASS  Covariance, missing data, skip missing rows     max rel err 3.99e-16
  1.2  PASS  Correlation, missing data, skip missing rows    max rel err 1.94e-16
  1.3  PASS  Covariance, missing data, pairwise              max rel err 5.67e-16
  1.4  PASS  Correlation, missing data, pairwise             max rel err 2.78e-16
  2.1  PASS  EW covariance, lambda = 0.97                    max rel err 2.22e-16
  2.2  PASS  EW correlation, lambda = 0.94                   max rel err 2.22e-16
  2.3  PASS  EW variance (0.97) with EW correlation (0.94)   max rel err 2.22e-16
  3.1  PASS  near_psd on a covariance                        max rel err 9.99e-16
  3.2  PASS  near_psd on a correlation                       max rel err 9.99e-16
  3.3  PASS  Higham nearest PSD on a covariance              max rel err 8.88e-16
  3.4  PASS  Higham nearest PSD on a correlation             max rel err 2.50e-15
  4.1  PASS  chol_psd                                        max rel err 1.39e-16
  5.1  PASS  Normal simulation, PD input                     max rel err 9.03e-04
  5.2  PASS  Normal simulation, PSD input                    max rel err 1.04e-03
  5.3  PASS  Normal simulation, non-PSD input, near_psd fix  max rel err 9.70e-04
  5.4  PASS  Normal simulation, non-PSD input, Higham fix    max rel err 9.62e-04
  5.5  PASS  PCA simulation, 99% explained                   max rel err 1.05e-03
  6.1  PASS  Arithmetic returns                              max rel err 1.00e-16
  6.2  PASS  Log returns                                     max rel err 9.99e-17
  7.1  PASS  Fit a normal distribution                       max rel err 2.78e-17
  7.2  PASS  Fit a generalized t distribution                max rel err 4.71e-09
  7.3  PASS  Fit a t regression                              max rel err 2.32e-08
  7.4  PASS  AICc on the fitted t                            max rel err 6.11e-16
  7.5  PASS  Fit a NIG by the method of moments               max rel err 1.02e-15
  7.6  PASS  Fit the same NIG by maximum likelihood           max rel err 3.13e-14

25 passed, 0 failed, 25 total
```

## Tolerances

Nineteen of the 25 match to machine precision. Three groups do not, and the
runner prints the reason next to each:

**5.1 – 5.5** are simulations. The expected-output files are one draw from
Julia's RNG, which Python cannot reproduce, so these cases compare the
simulated covariance against the covariance that generated it. At 100,000 draws
the sampling error on an entry is about `sqrt(2/N)` of it, roughly 0.45%, so
the tolerance is `2e-3` relative. The class files sit the same distance from
their own inputs.

5.3 and 5.4 compare against the *repaired* input, `near_psd(input)` and
`higham_nearest_psd(input)`, since the raw input is not PSD and is not what
either simulation draws from.

**7.2 – 7.4** are numerical optima. The class answers come from Ipopt and these
from scipy's Nelder–Mead, so they agree to eight or nine figures rather than to
machine precision. Tolerances are `1e-5` on 7.2 and 7.4, `1e-4` on 7.3, which
has six parameters.

**7.6** calls `scipy.stats.norminvgauss.fit`, which is the same optimizer the
expected output was generated from, so it agrees to `3e-14` here — but it is
still an optimizer and a different scipy version could move the trailing
digits, so the tolerance is `1e-6` rather than exact. Verified on scipy 1.17.1.

## Conventions worth pinning

- `missing_cov` reads missing values as `NaN`. Skip-rows drops any row with a
  hole in *any* column; pairwise uses, for entry (i, j), the rows where columns
  i and j are both present.
- `ew_covar` weights are `(1-λ)λ^(m-i)` normalized to sum to 1, so the **last**
  row carries the most weight — the input must be in time order, oldest first.
- Test 2.3 takes the variance from λ = 0.97 and the correlation from λ = 0.94.
- `near_psd` and `higham_nearest_psd` both operate on a correlation matrix
  internally and rescale on the way out if handed a covariance.
- The generalized t is `mu + sigma * T_nu`, so `sigma` is a scale. Its standard
  deviation is `sigma * sqrt(nu/(nu-2))`.
- AICc counts `k = 3` for a fitted generalized t, plus one per regression
  coefficient when there is a regression.
- The NIG is reported as `(mu, alpha, beta, delta)`. scipy uses
  `(a, b, loc, scale)` with `a = alpha*delta`, `b = beta*delta`, `loc = mu`,
  `scale = delta`.
- Test 7.5 uses a **mixed** set of moment estimators: the variance has the n-1
  divisor while the skewness and excess kurtosis are the **biased** ones.
  Matching that is what takes it from 1% off to 1e-15.
