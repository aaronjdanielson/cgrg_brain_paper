"""
Frank copula density for the CGRG brain network model.

Corresponds to bnet_copprob.m.

The Frank copula density c(u, v; rho) for the bivariate case is:

    c(u, v; rho) = rho * (1 - exp(-rho)) * exp(-rho*(u+v))
                   / (1 - exp(-rho) - (1-exp(-rho*u))*(1-exp(-rho*v)))^2

Note on numerical stability:
    When u or v are very close to 0 or 1, intermediate terms can underflow
    to 0 or overflow. Inputs are clipped to [1e-10, 1-1e-10] before evaluation.
    The denominator can also become very small when rho is near zero; callers
    should avoid rho = 0 exactly (the Frank copula degenerates to independence
    in the limit rho -> 0, but the density formula is not defined at rho = 0).
"""

import numpy as np


_EPS = 1e-10


def frank_log_density(u: np.ndarray, v: np.ndarray, rho: float) -> np.ndarray:
    """
    Numerically stable log Frank copula density, vectorized.

    Translates bnet_copprob.m exactly, operating in log space to avoid
    intermediate overflow/underflow.

    Parameters
    ----------
    u : array_like
        First uniform marginal argument, values in (0, 1).
        Clipped to [1e-10, 1-1e-10] internally.
    v : array_like
        Second uniform marginal argument, values in (0, 1).
        Clipped to [1e-10, 1-1e-10] internally.
    rho : float
        Frank copula dependence parameter.  Must be nonzero.

    Returns
    -------
    np.ndarray
        Log density values, same shape as u and v.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    # Clip to avoid log(0) and division-by-zero in the CDF arguments
    u = np.clip(u, _EPS, 1.0 - _EPS)
    v = np.clip(v, _EPS, 1.0 - _EPS)

    # Numerator: rho * (1 - exp(-rho)) * exp(-rho*(u+v))
    # In log space: log|rho| + log(1 - exp(-rho)) + (-rho*(u+v))
    # We compute the sign separately to handle negative rho.
    # log|rho|
    log_abs_rho = np.log(abs(rho))

    # log|1 - exp(-rho)| = log|expm1(-rho)|  (correct for both signs of rho)
    log_abs_1_minus_exp = np.log(np.abs(np.expm1(-rho)))

    # Numerator in log space: log|rho| + log|1-e^{-rho}| + (-rho*(u+v))
    # Note: must use actual rho (not abs), since sign matters for the exponent.
    log_num = log_abs_rho + log_abs_1_minus_exp + (-rho * (u + v))

    # Denominator: inner = (1-e^{-rho}) - (1-e^{-rho*u})(1-e^{-rho*v})
    # Use expm1 for accuracy: 1 - exp(-rho*x) = -expm1(-rho*x)
    one_minus_exp_neg_rho   = -np.expm1(-rho)        # scalar
    one_minus_exp_neg_rho_u = -np.expm1(-rho * u)    # array
    one_minus_exp_neg_rho_v = -np.expm1(-rho * v)    # array

    inner = one_minus_exp_neg_rho - one_minus_exp_neg_rho_u * one_minus_exp_neg_rho_v
    log_denom = 2.0 * np.log(np.abs(inner))

    return log_num - log_denom


def frank_density(u: np.ndarray, v: np.ndarray, rho: float) -> np.ndarray:
    """
    Frank copula density (not log), vectorized.

    Direct translation of bnet_copprob.m.  Prefer frank_log_density for
    MCMC computations to avoid numerical underflow.

    Parameters
    ----------
    u, v : array_like
        Uniform marginal arguments, clipped to [1e-10, 1-1e-10].
    rho : float
        Frank copula dependence parameter (nonzero).

    Returns
    -------
    np.ndarray
        Copula density values.
    """
    u = np.clip(np.asarray(u, dtype=float), _EPS, 1.0 - _EPS)
    v = np.clip(np.asarray(v, dtype=float), _EPS, 1.0 - _EPS)

    num = rho * ((1.0 - np.exp(-rho)) * np.exp(-rho * (u + v)))
    denom = (1.0 - np.exp(-rho)
             - (1.0 - np.exp(-rho * u)) * (1.0 - np.exp(-rho * v))) ** 2
    return num / denom
