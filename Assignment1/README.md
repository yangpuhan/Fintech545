# Assignment 1 — Univariate and Multivariate Statistics

Puhan Yang · FinTech 545, Quantitative Risk Management

## What is here

| Path | What it is |
|:--|:--|
| `Assignment1.pdf` | **The write-up. Read this one.** |
| `answers.qmd` | Quarto source for that PDF |
| `problem1.py` … `problem5.py` | One script per problem. Each prints every number quoted in the write-up and writes its figures. |
| `common.py` | Shared code: the moment estimators, AICc, and the two MLE regressions used in Problem 2 |
| `output/problemN.txt` | Saved console output of each script, so the numbers can be checked without running anything |
| `figures/` | Generated figures, embedded in the PDF |
| `problem1.csv` … `problem5.csv` | The data |

## Running the code

Python 3.9 or newer.

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

Each script takes no arguments, reads its own CSV from the current directory,
and prints its report to stdout. Run them in any order. A few seconds each.

To regenerate everything, including the saved output:

```bash
mkdir -p output
for i in 1 2 3 4 5; do python3 problem$i.py | tee output/problem$i.txt; done
```

## Rebuilding the PDF

Needs Quarto and a TeX distribution with `lualatex`. Run the five scripts
first, so the figures exist.

```bash
quarto render answers.qmd --to pdf
```

## Where each number in the write-up comes from

| Write-up section | Script | Console section |
|:--|:--|:--|
| §2 moments, feasibility check | `problem1.py` | `PROBLEM 1 -- moments of the sample` |
| §2 (b) 26 against 10, (c) both-tail table | `problem1.py` | `PROBLEM 1 -- normal fitted by matching…` |
| §3 three fits, AICc, slopes | `problem2.py` | `PROBLEM 2 -- three models`, `-- AICc comparison`, `-- the three slope estimates` |
| §3 (b) which assumption fails | `problem2.py` | `PROBLEM 2 -- residual diagnostics` |
| §3 (e) quantiles and the crossover | `problem2.py` | `PROBLEM 2 -- quantiles of the two fitted…` |
| §4 marginals, both matrices, gap ranking | `problem3.py` | `PROBLEM 3 -- marginal shape…`, `-- correlation matrices`, `-- every pair…` |
| §4 (c) trimming table, normal scores | `problem3.py` | `PROBLEM 3 -- what produces the gap on x1-x2` |
| §5 (a)–(c) blocks, factor, coefficient | `problem4.py` | `PROBLEM 4 (a)`, `(b)`, `(c)` |
| §5 (d)–(e) coverage overall and bucketed | `problem4.py` | `PROBLEM 4 (d)`, `(e)` |
| §5 (f) bivariate *t* fit and multipliers | `problem4.py` | `PROBLEM 4 (f) -- a bivariate t fitted to the pair` |
| §6 (a)–(b) ACF/PACF table and band | `problem5.py` | `PROBLEM 5 -- ACF and PACF` |
| §6 (c)–(d) six AICc values, AR(2) roots | `problem5.py` | `PROBLEM 5 -- AR and MA fits`, `-- coefficients of AR(2)` |
| §6 (e) third coefficient, R² comparison | `problem5.py` | `PROBLEM 5 (e) -- AR(2) against AR(3)` |

The moment and AICc conventions the numbers use are stated in §1 of the
write-up and implemented in `common.py`.
