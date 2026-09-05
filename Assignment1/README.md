# Assignment 1 — Univariate and Multivariate Statistics

Puhan Yang · FinTech 545, Quantitative Risk Management

## What is here

| Path | What it is |
|:--|:--|
| `Assignment1.pdf` | **The write-up. Read this one.** |
| `answers.qmd` | Quarto source for that PDF |
| `problem1.py` … `problem5.py` | One script per problem. Each prints every number quoted in the write-up and writes its figures. |
| `common.py` | Shared code: the moment estimators, AICc, and the two MLE regressions used in Problem 2 |
| `output/problemN.txt` | Saved console output of each script, so the numbers can be checked without re-running anything |
| `figures/` | Generated figures, embedded in the PDF |
| `problem1.csv` … `problem5.csv` | The data, copied from the class repository so this folder runs standalone |
| `Assignment 1.pdf` | The problem statement, for reference |

## Running the code

Python 3.9 or newer. Julia was not used because the packages needed here are
all in the Python scientific stack and it was already installed; nothing in the
assignment requires a particular language.

```bash
pip install -r requirements.txt
```

Then, from inside this directory:

```bash
python3 problem1.py     # moments, normal fit, tail exceedance counts
python3 problem2.py     # OLS, normal MLE, t MLE, AICc, error quantiles
python3 problem3.py     # Pearson and Spearman matrices, mechanism diagnostics
python3 problem4.py     # conditional mean and variance, band coverage, bivariate t
python3 problem5.py     # ACF/PACF, six AR and MA fits, AICc
```

Each script is standalone, takes no arguments, reads its own CSV from the
current directory, and prints a report to stdout. Nothing depends on anything
else having been run first. To regenerate everything including the saved
output:

```bash
mkdir -p output
for i in 1 2 3 4 5; do python3 problem$i.py | tee output/problem$i.txt; done
```

Runtime is a few seconds per script; Problem 4 is the slowest because it fits a
bivariate *t* by Nelder–Mead.

## Rebuilding the PDF

Needs Quarto and a TeX distribution with `lualatex`.

```bash
quarto render answers.qmd --to pdf
```

That writes `Assignment1.pdf`. The figures must exist first, so run the five
scripts before rendering a clean checkout.

## Where each number in the write-up comes from

| Write-up section | Script | Console section to look for |
|:--|:--|:--|
| §2 moments, feasibility check | `problem1.py` | `PROBLEM 1 -- moments of the sample` |
| §2 (b) 26 against 10, (c) both-tail table | `problem1.py` | `PROBLEM 1 -- normal fitted by matching…` |
| §3 three fits, AICc, slopes | `problem2.py` | `PROBLEM 2 -- three models`, `-- AICc comparison`, `-- the three slope estimates` |
| §3 (e) quantiles and the crossover | `problem2.py` | `PROBLEM 2 -- quantiles of the two fitted…` |
| §3 (b) which assumption fails | `problem2.py` | `PROBLEM 2 -- residual diagnostics` |
| §4 marginals, both matrices, gap ranking | `problem3.py` | `PROBLEM 3 -- marginal shape…`, `-- correlation matrices`, `-- every pair…` |
| §4 (c) trimming table, normal scores | `problem3.py` | `PROBLEM 3 -- what produces the gap on x1-x2` |
| §5 (a)–(c) blocks, factor, coefficient | `problem4.py` | `PROBLEM 4 (a)`, `(b)`, `(c)` |
| §5 (d)–(e) coverage overall and bucketed | `problem4.py` | `PROBLEM 4 (d)`, `(e)` |
| §5 (f) bivariate *t* fit and multipliers | `problem4.py` | `PROBLEM 4 (f) -- a bivariate t fitted to the pair` |
| §6 (a)–(b) ACF/PACF table and band | `problem5.py` | `PROBLEM 5 -- ACF and PACF` |
| §6 (c)–(d) six AICc values, AR(2) roots | `problem5.py` | `PROBLEM 5 -- AR and MA fits`, `-- coefficients of AR(2)` |
| §6 (e) third coefficient, R² comparison | `problem5.py` | `PROBLEM 5 (e) -- AR(2) against AR(3)` |

## Two conventions, stated once

These are the two places where packages disagree with each other, so both are
pinned explicitly in `common.py` rather than left to a default:

- **Moments.** Variance uses the *n−1* denominator. Skewness and excess
  kurtosis use the bias-corrected estimators, and kurtosis is always reported
  in **excess** form, so a normal sample returns 0 rather than 3. `scipy.stats`
  defaults to the biased estimators, so `bias=False` is passed everywhere. Both
  versions are printed by `problem1.py` so the difference can be seen.
- **AICc.** `k = p + d`, following Week 02: *p* counts the regression
  parameters including the intercept, *d* counts the parameters of the error
  distribution. A one-regressor model with normal errors has k = 3, with
  Student's *t* errors k = 4, an AR(*p*) has k = *p* + 2. `statsmodels` reports
  AIC only, so the `(2k² + 2k)/(n − k − 1)` correction is added by hand in
  `common.aicc`.

The course convention of 255 trading days per year is recorded in `common.py`.
No problem here asks for an annualized figure, so it is not applied to anything.
