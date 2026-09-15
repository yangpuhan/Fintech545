"""Repairing a matrix that is not positive semi-definite, and factoring one
that is.

A pairwise covariance estimate is not required to be PSD, because each entry
comes from a different sample.  Two repairs are implemented: the Rebonato and
Jackel spectral fix (`near_psd`), which is one projection and is fast, and
Higham's alternating projections (`higham_nearest_psd`), which iterates to the
*nearest* PSD correlation matrix in the Frobenius norm.

`chol_psd` is a Cholesky that tolerates a singular matrix by zeroing the column
rather than failing.

Covers functional tests 3.1 through 4.1.
"""

import numpy as np


def _is_correlation(a, tol=1e-8):
    """True when every diagonal entry is 1, i.e. we were handed a correlation
    matrix rather than a covariance.  Both repairs operate on correlations, so
    a covariance is converted on the way in and rescaled on the way out.
    """
    return np.all(np.abs(np.diag(a) - 1.0) < tol)


# ---------------------------------------------------------------------------
# Rebonato and Jackel (tests 3.1, 3.2)
# ---------------------------------------------------------------------------

def near_psd(a, epsilon=0.0):
    """Rebonato and Jackel (1999): clip the negative eigenvalues, then rescale.

    Zeroing eigenvalues shrinks the rows, so without the diagonal scaling `T`
    the result would no longer have a unit diagonal and would not be a
    correlation matrix.  This is one spectral projection, not an optimization,
    so the result is a valid PSD matrix but not necessarily the closest one.
    """
    out = np.array(a, dtype=float, copy=True)
    inv_sd = None

    if not _is_correlation(out):
        inv_sd = 1.0 / np.sqrt(np.diag(out))
        out = inv_sd[:, None] * out * inv_sd[None, :]

    vals, vecs = np.linalg.eigh(out)
    vals = np.maximum(vals, epsilon)
    # T restores the unit diagonal that clipping the eigenvalues destroyed.
    t = 1.0 / ((vecs * vecs) @ vals)
    b = np.sqrt(t)[:, None] * vecs * np.sqrt(vals)[None, :]
    out = b @ b.T

    if inv_sd is not None:
        sd = 1.0 / inv_sd
        out = sd[:, None] * out * sd[None, :]
    return out


# ---------------------------------------------------------------------------
# Higham (tests 3.3, 3.4)
# ---------------------------------------------------------------------------

def _proj_spd(a, w):
    """Projection onto the PSD cone under the W-weighted norm: clip negative
    eigenvalues.  Makes the matrix PSD and breaks the unit diagonal."""
    w05 = np.sqrt(w)
    iw = np.linalg.inv(w05)
    m = w05 @ a @ w05
    vals, vecs = np.linalg.eigh(m)
    m_plus = vecs @ np.diag(np.maximum(vals, 0.0)) @ vecs.T
    return iw @ m_plus @ iw


def _proj_unit_diag(a):
    """Projection onto the unit-diagonal set: set the diagonal back to 1 and
    leave everything else alone.  Fixes the diagonal and can break PSD."""
    out = np.array(a, dtype=float, copy=True)
    np.fill_diagonal(out, 1.0)
    return out


def _wgt_norm(a, w):
    w05 = np.sqrt(w)
    m = w05 @ a @ w05
    return float(np.sum(m * m))


def higham_nearest_psd(pc, w=None, epsilon=1e-9, max_iter=100, tol=1e-9):
    """Higham (2002) Algorithm 3.3: alternating projections with a Dykstra
    correction.

    Neither projection alone gives a correlation matrix -- `_proj_spd` makes it
    PSD and breaks the diagonal, `_proj_unit_diag` fixes the diagonal and can
    break PSD.  Alternating converges to a matrix in both sets.  The `delta_s`
    term is the Dykstra correction; without it the iteration still converges to
    *a* point in the intersection, and with it the limit is the *nearest* one.

    Stops when the weighted norm has stopped moving AND the smallest eigenvalue
    is no longer materially negative.  Checking only the norm is not enough:
    convergence of the norm says nothing directly about the eigenvalues, and
    the eigenvalue is what the caller's Cholesky will trip over.
    """
    y = np.array(pc, dtype=float, copy=True)
    n = y.shape[0]
    if w is None:
        w = np.eye(n)

    inv_sd = None
    if not _is_correlation(y):
        inv_sd = 1.0 / np.sqrt(np.diag(y))
        y = inv_sd[:, None] * y * inv_sd[None, :]

    y0 = y.copy()
    delta_s = np.zeros_like(y)
    norm_last = np.inf

    for _ in range(max_iter):
        r = y - delta_s
        x = _proj_spd(r, w)
        delta_s = x - r
        y = _proj_unit_diag(x)
        norm = _wgt_norm(y - y0, w)
        min_eig = np.min(np.linalg.eigvalsh(y))
        if abs(norm - norm_last) < tol and min_eig > -epsilon:
            break
        norm_last = norm

    if inv_sd is not None:
        sd = 1.0 / inv_sd
        y = sd[:, None] * y * sd[None, :]
    return y


# ---------------------------------------------------------------------------
# Cholesky that tolerates a singular matrix (test 4.1)
# ---------------------------------------------------------------------------

def chol_psd(a, epsilon=-1e-8):
    """Lower-triangular root of a PSD matrix, allowing zero columns.

    The textbook Cholesky divides by the diagonal element, so it fails the
    moment that element is zero -- which is what happens when the matrix is
    singular rather than positive definite.  Here a zero (or slightly negative,
    within `epsilon`) diagonal has its whole column set to zero instead.

    The result still satisfies L @ L.T == a.  What it costs is a rank: a zero
    column is one fewer independent random number driving any simulation built
    on this root, so every draw lies in a lower-dimensional subspace.  The
    factorization succeeding is not on its own reassuring; count the non-zero
    columns.

    Raises ValueError when a pivot comes out negative by more than `epsilon`.
    That is a genuinely non-PSD matrix rather than a rounding artifact, and it
    needs one of the repairs above rather than a patched Cholesky.
    """
    a = np.asarray(a, dtype=float)
    n = a.shape[0]
    root = np.zeros((n, n), dtype=float)

    for j in range(n):
        s = root[j, :j] @ root[j, :j] if j > 0 else 0.0
        temp = a[j, j] - s
        if epsilon <= temp <= 0:
            temp = 0.0
        elif temp < epsilon:
            raise ValueError(
                f"matrix is not positive semi-definite: pivot {j} is {temp:.3e}, "
                f"past the {epsilon:.0e} tolerance for rounding error")
        root[j, j] = np.sqrt(temp)

        if root[j, j] == 0.0:
            root[j, j + 1:] = 0.0
            continue
        ir = 1.0 / root[j, j]
        for i in range(j + 1, n):
            s = root[i, :j] @ root[j, :j] if j > 0 else 0.0
            root[i, j] = (a[i, j] - s) * ir
    return root
