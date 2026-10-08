# Assignment 2 — Covariance, VaR, and Copulas

Puhan Yang · FinTech 545, Quantitative Risk Management

## What is here

| Path | What it is |
|:--|:--|
| `Assignment2.pdf` | **The write-up. Read this one.** |
| `answers.qmd` | Quarto source for that PDF |
| `problem1.py` … `problem5.py` | One script per problem. Each prints every number quoted in the write-up and writes its figures. |
| `common.py` | Shared code: moments, historical and normal VaR/ES, the Gaussian and t copulas, the ν profile |
| `output/problemN.txt` | Saved console output of each script, so the numbers can be checked without running anything |
| `figures/` | Generated figures, embedded in the PDF |
| `problem1.csv` … `problem5.csv` | The data |

The scripts also use the repository's library, [`../riskmgmt`](../riskmgmt):
missing data correlation, the two PSD repairs, the PSD Cholesky, exponential
weights, the normal and t fits, AICc, returns, and normal simulation. Nothing
needs installing; `common.py` puts the repository root on the import path.

## Running the code

Python 3.9 or newer.

```bash
pip install -r ../requirements.txt
```

Then, from inside this directory:

```bash
python3 problem1.py     # pairwise vs complete case, Cholesky, RJ and Higham repairs
python3 problem2.py     # five VaRs, n_eff and half life, the two regimes
python3 problem3.py     # bond VaR and ES at 5% and 1%, subadditivity
python3 problem4.py     # margins by AICc, Gaussian and t copulas, simulated VaR/ES
python3 problem5.py     # regressions, model based simulation, delta normal VaR
```

Each script takes no arguments, reads its own CSV from the current directory,
and prints its report to stdout. Run them in any order. `problem4.py` takes
about 10 seconds, the rest a few seconds each. Every simulation uses a fixed
seed, so the output is identical from run to run.

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
| §2 (a)–(c) day counts, eigenvalue bound | `problem1.py` | `PROBLEM 1 -- days each pair…`, `(b) -- how small…` |
| §2 (d) both matrices, eigenvalues, Cholesky | `problem1.py` | `PROBLEM 1 (d)` |
| §2 (e)–(f) tracking variance, repairs | `problem1.py` | `PROBLEM 1 (e)`, `(f)` |
| §2 (h)–(i) entries moved, estimator gap, windows | `problem1.py` | `PROBLEM 1 (h)`, `(i)` |
| §3 (b)–(c) weights, changepoint, moments | `problem2.py` | `PROBLEM 2 (b)`, `(c)` |
| §3 (d) the five VaRs and the t fit | `problem2.py` | `PROBLEM 2 (d)` |
| §3 (e)–(g) tail days, regimes, mixture, noise | `problem2.py` | `PROBLEM 2 (e)`, `(f)`, `(g)` |
| §4 (a)–(b) moments and counts | `problem3.py` | `PROBLEM 3 (a)`, `(b)` |
| §4 (c)–(e) VaR and ES, normal VaR | `problem3.py` | `PROBLEM 3 (c)`, `(d)`, `(e)` |
| §4 (f)–(h) subadditivity, tail contents | `problem3.py` | `PROBLEM 3 (f) and (h)`, `-- what the worst…` |
| §5 (a)–(b) moments, day 952, tail counts | `problem4.py` | `PROBLEM 4 (a)`, `(b)` |
| §5 (c)–(d) margins, copula fits, profile | `problem4.py` | `PROBLEM 4 (c)`, `(d)` |
| §5 (e)–(h) simulated VaR/ES, implied tail days | `problem4.py` | `PROBLEM 4 (e)`, `(h)`, `(f)` |
| §5 (i)–(j) likelihood by day, tail dependence | `problem4.py` | `PROBLEM 4 (i)`, `(j)` |
| §6 (b)–(d) regressions, simulation, delta normal | `problem5.py` | `PROBLEM 5 (b)`, `(c)`, `(d)` |
| §6 (e)–(f) closed forms, decomposition | `problem5.py` | `PROBLEM 5 (e), (f)` |

The VaR, ES, moment and AICc conventions the numbers use are stated in §1 of
the write-up and implemented in `common.py`.
