"""
Log-likelihood, log-prior, and log-posterior for the CGRG brain model.

Corresponds to bnet_logpost.m.

Prior specification (matching MATLAB exactly):
    rho0         ~ N(0, 2^2)    =>  -rho0^2 / 8
    rho[g]       ~ N(rho0, sigma_rho^2)
    delta0[i]    ~ N(0, 2^2)    =>  -delta0[i]^2 / 8   (i=1..3; i=0 fixed)
    delta[i,g]   ~ N(delta0[i], sigma_delta^2)
    gamma0[i]    ~ N(0, 2^2)    =>  -gamma0[i]^2 / 8
    gamma[i,g]   ~ N(gamma0[i], sigma_gamma^2)
    lambda[g]    ~ Gamma(shape=5, scale=4)
    sigma_rho    ~ Gamma(shape=0.1, scale=10)
    sigma_delta  ~ Gamma(shape=0.1, scale=10)
    sigma_gamma  ~ Gamma(shape=0.1, scale=10)

Note: diagonal entries (self-loops) are excluded from the likelihood
(laddiag = lmcidiag = lnldiag = 0 in MATLAB).
"""

import math

import numpy as np
from scipy.special import ndtr  # direct C wrapper: ~10x faster than scipy.stats.norm

from .copula import frank_log_density
from .model import BrainCGRG, GroupData

_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Group log-likelihood helper
# ---------------------------------------------------------------------------

def _group_log_likelihood(gd: GroupData, lam: float, rho: float) -> float:
    """
    Log-likelihood contribution from one diagnostic group.

    Parameters
    ----------
    gd : GroupData
        Stacked observations (bel, beu) and stacked means (bmul, bmuu).
    lam : float
        Precision for this group (std = 1/sqrt(lam)).
    rho : float
        Frank copula parameter for this group.

    Returns
    -------
    float
    """
    sigma = 1.0 / np.sqrt(lam)
    log_sigma = math.log(sigma)

    # Standardised residuals
    zl = (gd.bel - gd.bmul) / sigma
    zu = (gd.beu - gd.bmuu) / sigma

    # CDF evaluations for copula argument via scipy.special.ndtr
    # (avoids scipy.stats overhead: ~10x faster on small arrays)
    pl = ndtr(zl)
    pu = ndtr(zu)

    # log Frank copula density (vectorised over all dyads x subjects)
    log_cop = frank_log_density(pl, pu, rho)

    # log Gaussian marginal densities (inlined formula, no scipy.stats overhead)
    log_marg_l = -_HALF_LOG_2PI - log_sigma - 0.5 * zl * zl
    log_marg_u = -_HALF_LOG_2PI - log_sigma - 0.5 * zu * zu

    return float(np.sum(log_cop) + np.sum(log_marg_l) + np.sum(log_marg_u))


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def log_likelihood(params: BrainCGRG) -> float:
    """
    Total log-likelihood summed over all three groups.

    Corresponds to: loglik = ladcop + lmcicop + lnlcop  (diagonal = 0).

    Parameters
    ----------
    params : BrainCGRG
        Must have group_data populated (call precompute_stacked_data first).

    Returns
    -------
    float
    """
    if params.group_data is None:
        raise RuntimeError("group_data not initialised.")

    ll = 0.0
    # Group index: NL=0, MCI=1, AD=2
    for gname, gidx in [("nl", 0), ("mci", 1), ("ad", 2)]:
        ll += _group_log_likelihood(
            params.group_data[gname],
            lam=float(params.lam[gidx]),
            rho=float(params.rho[gidx]),
        )
    return ll


def log_prior(params: BrainCGRG) -> float:
    """
    Log prior probability.

    Matches bnet_logpost.m exactly.

    The reference node constraint delta[0, :] = 0 is enforced externally
    by the sampler; the prior sum over delta automatically excludes row 0
    because delta[0,:] - delta0[0] = 0 - 0 = 0, contributing nothing.

    Returns
    -------
    float
    """
    lp = 0.0

    # -- rho0 ~ N(0, 4)  [variance=4 => -rho0^2/8 each]
    lp -= float(params.rho0 ** 2) / 8.0

    # -- rho[g] ~ N(rho0, sigma_rho^2)
    lp -= float(np.sum((params.rho - params.rho0) ** 2)) / (
        2.0 * params.sigma_rho ** 2)

    # -- delta0 ~ N(0, 4)  [variance=4; delta0[0] fixed=0, contributes 0]
    lp -= float(np.sum(params.delta0 ** 2)) / 8.0

    # -- delta[i,g] ~ N(delta0[i], sigma_delta^2)
    # Equivalent to MATLAB:
    #   sum((reshape(delta,12,1) - repmat(delta0,3,1))^2) / (2*sigma_delta^2)
    # In Python: np.sum((delta - delta0[:,newaxis])**2)  [broadcast across groups]
    lp -= float(np.sum((params.delta - params.delta0[:, np.newaxis]) ** 2)) / (
        2.0 * params.sigma_delta ** 2)

    # -- gamma0 ~ N(0, 4)
    lp -= float(np.sum(params.gamma0 ** 2)) / 8.0

    # -- gamma[i,g] ~ N(gamma0[i], sigma_gamma^2)
    lp -= float(np.sum((params.gamma - params.gamma0[:, np.newaxis]) ** 2)) / (
        2.0 * params.sigma_gamma ** 2)

    # -- lambda[g] ~ Gamma(shape=5, scale=4)
    #    log p(x) = (a-1)*log(x) - x/scale  (+ constant)
    lp += float(np.sum((5.0 - 1.0) * np.log(params.lam) - params.lam / 4.0))

    # -- sigma_rho ~ Gamma(shape=0.1, scale=10)
    #    log p(x) = (a-1)*log(x) - x/scale
    lp += (0.1 - 1.0) * np.log(params.sigma_rho) - params.sigma_rho / 10.0

    # -- sigma_delta ~ Gamma(shape=0.1, scale=10)
    lp += (0.1 - 1.0) * np.log(params.sigma_delta) - params.sigma_delta / 10.0

    # -- sigma_gamma ~ Gamma(shape=0.1, scale=10)
    lp += (0.1 - 1.0) * np.log(params.sigma_gamma) - params.sigma_gamma / 10.0

    return float(lp)


def log_posterior(params: BrainCGRG) -> float:
    """
    Log posterior = log_likelihood + log_prior.

    Parameters
    ----------
    params : BrainCGRG
        Must have group_data populated.

    Returns
    -------
    float
        Returns -inf if any value is non-finite (guards against NaN
        propagation into the MH acceptance step).
    """
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(params)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll
