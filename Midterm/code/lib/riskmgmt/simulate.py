"""Multivariate normal simulation from a covariance matrix.

Two routes to the root L with Sigma = L L': a Cholesky, and an eigenvalue
decomposition (PCA).  Nothing about the simulation cares which produced the
root; the difference is what each does when the matrix is not positive
definite, and whether you want to drop dimensions on purpose.

Covers functional tests 5.1 through 5.5.
"""

import numpy as np

from .psd import chol_psd, near_psd


def simulate_normal(n_sim, cov, mean=None, seed=1234, fix_method=near_psd):
    """Simulate `n_sim` draws from N(mean, cov).  Returns (n_sim, n).

    The root is taken in three escalating steps, which is the practical part:

      1. an ordinary Cholesky, which works when the matrix is positive definite;
      2. failing that, `chol_psd`, which works when the matrix is singular but
         still PSD -- the failure was a property of the PD algorithm, not of
         the matrix;
      3. failing that, repair the matrix with `fix_method` and factor the
         repair, because a matrix with a genuinely negative eigenvalue assigns
         a negative variance to some real portfolio and cannot have come from
         any data.
    """
    cov = np.asarray(cov, dtype=float)
    n, m = cov.shape
    if n != m:
        raise ValueError(f"covariance matrix is not square ({n},{m})")

    if mean is None:
        mu = np.zeros(n)
    else:
        mu = np.asarray(mean, dtype=float)
        if mu.shape[0] != n:
            raise ValueError(f"mean ({mu.shape[0]}) is not the size of cov ({n},{n})")

    try:
        root = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        try:
            root = chol_psd(cov)
        except ValueError:
            root = chol_psd(fix_method(cov))

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, n_sim))
    return (root @ z).T + mu


def simulate_pca(cov, n_sim, pct_exp=1.0, mean=None, seed=1234):
    """Simulate through an eigenvalue decomposition, optionally keeping only
    enough components to explain `pct_exp` of the total variance.

    Components are taken largest first.  Eigenvalues at or below 1e-8 are
    dropped as numerically zero, which handles a singular input without any
    separate repair step.

    Dropping further components is a modeling choice and not a repair: each one
    dropped is a direction the simulation can no longer move in, and the
    forecast variance of any portfolio lying in that direction is exactly zero.
    """
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    mu = np.zeros(n) if mean is None else np.asarray(mean, dtype=float)

    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]          # eigh returns ascending; flip
    vals, vecs = vals[order], vecs[:, order]

    total = vals.sum()
    keep = vals > 1e-8
    vals, vecs = vals[keep], vecs[:, keep]

    if pct_exp < 1.0:
        cumulative = np.cumsum(vals) / total
        n_keep = int(np.searchsorted(cumulative, pct_exp) + 1)
        n_keep = min(n_keep, len(vals))
        vals, vecs = vals[:n_keep], vecs[:, :n_keep]

    b = vecs * np.sqrt(vals)[None, :]
    rng = np.random.default_rng(seed)
    r = rng.standard_normal((len(vals), n_sim))
    return (b @ r).T + mu
