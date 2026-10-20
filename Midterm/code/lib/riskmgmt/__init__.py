"""A small risk management library for FinTech 545.

Built up alongside the functional tests, so each module corresponds to a block
of them:

    covariance  tests 1.1 - 2.3    missing data, exponentially weighted
    psd         tests 3.1 - 4.1    near_psd, Higham, Cholesky of a PSD matrix
    simulate    tests 5.1 - 5.5    normal and PCA simulation
    returns     tests 6.1 - 6.2    arithmetic and log returns
    fitting     tests 7.1 - 7.6    normal, generalized t, t regression, AICc, NIG
"""

from .covariance import (corr_to_cov, cov_to_corr, ew_cov_var_corr, ew_covar,
                         exp_weights, missing_cov)
from .fitting import (FittedModel, aicc, fit_general_t, fit_nig_mle,
                      fit_nig_moments, fit_normal, fit_regression_t,
                      nig_from_scipy, nig_params, nig_to_scipy)
from .psd import chol_psd, higham_nearest_psd, near_psd
from .returns import return_calculate
from .simulate import simulate_normal, simulate_pca

__all__ = [
    "missing_cov", "ew_covar", "exp_weights", "cov_to_corr", "corr_to_cov",
    "ew_cov_var_corr",
    "near_psd", "higham_nearest_psd", "chol_psd",
    "simulate_normal", "simulate_pca",
    "return_calculate",
    "fit_normal", "fit_general_t", "fit_regression_t", "aicc", "FittedModel",
    "fit_nig_moments", "fit_nig_mle", "nig_params", "nig_to_scipy", "nig_from_scipy",
]
