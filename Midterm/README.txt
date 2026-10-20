FinTech 545 Midterm -- Puhan Yang (py65)

Midterm_Answers.pdf     Written answers (every question restated, with method, results, discussion)

code/midterm.py         All code. Sections are labelled by problem and part,
                        e.g. "# Problem 1(c)", "# Problem 3(b)".
code/output_run.txt     Full printed output of one complete run.
code/output/*.csv       Every table in the write-up, as written by midterm.py
                        (file names start with the problem and part, e.g. p3c_var_es.csv).
code/lib/               Helper library used by midterm.py (riskmgmt package and
                        common.py from my homework repository), bundled so the
                        code runs on its own.
code/data/              The exam data files (problem1.csv, problem2.csv,
                        problem3.csv, problem3_holdings.csv).

How to run (Python 3 with numpy, pandas, scipy):

    cd code
    python midterm.py                 # all problems
    python midterm.py --problems 2    # one problem only

Problems 1 and 2 take a few seconds. Problem 3 takes about 20 minutes, almost
all of it in the 3(c) seed-robustness check (six simulations of 100,000 draws).
Random seed is fixed (SEED = 1234), so results reproduce exactly.

Choices made in the written answers (3(a) margins, 3(b) copula) are set in the
CHOICES block near the top of midterm.py.
