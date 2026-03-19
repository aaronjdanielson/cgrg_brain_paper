"""
Metropolis-Hastings sampler for the CGRG brain network model.

Corresponds to bnet_mcmc.m.

Sampling order each iteration (matches MATLAB exactly):
    1.  rho0                            step 1.0
    2.  rho[0], rho[1], rho[2]          step 1.0 each
    3.  delta0[1], delta0[2], delta0[3] step 0.25 each  (node 0 fixed)
    4.  delta[k,j] k=1,2,3; j=0,1,2    step 0.25 each  (bnet_update after each)
    5.  gamma0[0..3]                    step 0.25 each
    6.  gamma[k,j] k=0,1,2,3; j=0,1,2  step 0.25 each  (bnet_update after each)
    7.  lambda[0], lambda[1], lambda[2] step 2.5 each   (reject if lam <= 0)
    8.  sigma_rho                       step 0.05        (reject if <= 0)
    9.  sigma_delta                     step 0.025       (reject if <= 0)
    10. sigma_gamma                     step 0.025       (reject if <= 0)

MH rule (log scale): accept if log(U) < lpost_proposed - lpost_current.

The draw array has 42 columns; see model.py for layout documentation.
"""

import numpy as np
from typing import Dict, List

from .model import (BrainCGRG, GroupData, default_params,
                    make_triangle_indices, precompute_stacked_data,
                    update_means)
from .likelihood import log_posterior


# ---------------------------------------------------------------------------
# Step sizes (matching MATLAB)
# ---------------------------------------------------------------------------

STEP_RHO0 = 1.0
STEP_RHO = 1.0
STEP_DELTA0 = 0.25
STEP_DELTA = 0.25
STEP_GAMMA0 = 0.25
STEP_GAMMA = 0.25
STEP_LAMBDA = 2.5
STEP_SIGMA_RHO = 0.05
STEP_SIGMA_DELTA = 0.025
STEP_SIGMA_GAMMA = 0.025


# ---------------------------------------------------------------------------
# MH accept/reject helper
# ---------------------------------------------------------------------------

def _mh_accept(lpost_new: float, lpost_old: float) -> bool:
    """Standard log-scale Metropolis-Hastings acceptance test."""
    return np.log(np.random.uniform()) < lpost_new - lpost_old


# ---------------------------------------------------------------------------
# Main sampler
# ---------------------------------------------------------------------------

def run_mcmc(params: BrainCGRG,
             data: Dict[str, List],
             n_iter: int,
             linds: tuple,
             uinds: tuple,
             verbose: bool = True) -> np.ndarray:
    """
    Run the Metropolis-Hastings sampler and return all draws.

    Parameters
    ----------
    params : BrainCGRG
        Starting parameter values.  group_data will be populated here if
        not already set.
    data : dict
        As returned by load_brain_data(): 'nl', 'mci', 'ad' -> list of arrays.
    n_iter : int
        Number of MCMC iterations (each iteration cycles through all blocks).
    linds : tuple (rows_l, cols_l)
        Lower-triangle index pair from make_triangle_indices().
    uinds : tuple (rows_u, cols_u)
        Upper-triangle index pair from make_triangle_indices().
    verbose : bool
        If True, print acceptance rates every 1000 iterations.

    Returns
    -------
    np.ndarray, shape (n_iter, 42)
        One row per iteration.  Row i contains the parameter vector *after*
        the i-th MH sweep, matching the 42-column MATLAB convention.
    """
    rows_u, cols_u = uinds
    rows_l, cols_l = linds

    # Ensure stacked data arrays are ready
    if params.group_data is None:
        precompute_stacked_data(params, data, rows_u, cols_u, rows_l, cols_l)

    # Make sure initial means are consistent
    update_means(params, rows_u, cols_u, rows_l, cols_l)

    # Current log-posterior
    lpost_cur = log_posterior(params)

    draws = np.empty((n_iter, 42), dtype=float)

    # Acceptance counters (for verbose output)
    n_prop = {k: 0 for k in ["rho0", "rho", "delta0", "delta",
                               "gamma0", "gamma", "lam",
                               "sigma_rho", "sigma_delta", "sigma_gamma"]}
    n_acc = {k: 0 for k in n_prop}

    for it in range(n_iter):

        # ---- 1. rho0 -------------------------------------------------------
        rho0_old = params.rho0
        params.rho0 = rho0_old + STEP_RHO0 * np.random.uniform(-1, 1)
        lpost_new = log_posterior(params)
        n_prop["rho0"] += 1
        if _mh_accept(lpost_new, lpost_cur):
            lpost_cur = lpost_new
            n_acc["rho0"] += 1
        else:
            params.rho0 = rho0_old

        # ---- 2. rho[g] for g=0,1,2  ----------------------------------------
        for g in range(3):
            rho_old = params.rho[g]
            params.rho[g] = rho_old + STEP_RHO * np.random.uniform(-1, 1)
            lpost_new = log_posterior(params)
            n_prop["rho"] += 1
            if _mh_accept(lpost_new, lpost_cur):
                lpost_cur = lpost_new
                n_acc["rho"] += 1
            else:
                params.rho[g] = rho_old

        # ---- 3. delta0[k] for k=1,2,3  (node 0 is reference, fixed at 0) --
        for k in range(1, 4):
            d0_old = params.delta0[k]
            params.delta0[k] = d0_old + STEP_DELTA0 * np.random.uniform(-1, 1)
            lpost_new = log_posterior(params)
            n_prop["delta0"] += 1
            if _mh_accept(lpost_new, lpost_cur):
                lpost_cur = lpost_new
                n_acc["delta0"] += 1
            else:
                params.delta0[k] = d0_old

        # ---- 4. delta[k,j] k=1,2,3; j=0,1,2  ------------------------------
        #         (node 0 fixed at 0 for all groups)
        #         Call update_means after each accepted or rejected proposal.
        for k in range(1, 4):
            for j in range(3):
                d_old = params.delta[k, j]
                params.delta[k, j] = d_old + STEP_DELTA * np.random.uniform(-1, 1)
                # Recompute mean arrays for new delta
                update_means(params, rows_u, cols_u, rows_l, cols_l)
                lpost_new = log_posterior(params)
                n_prop["delta"] += 1
                if _mh_accept(lpost_new, lpost_cur):
                    lpost_cur = lpost_new
                    n_acc["delta"] += 1
                else:
                    params.delta[k, j] = d_old
                    # Restore means
                    update_means(params, rows_u, cols_u, rows_l, cols_l)

        # ---- 5. gamma0[k] for k=0,1,2,3  -----------------------------------
        for k in range(4):
            g0_old = params.gamma0[k]
            params.gamma0[k] = g0_old + STEP_GAMMA0 * np.random.uniform(-1, 1)
            lpost_new = log_posterior(params)
            n_prop["gamma0"] += 1
            if _mh_accept(lpost_new, lpost_cur):
                lpost_cur = lpost_new
                n_acc["gamma0"] += 1
            else:
                params.gamma0[k] = g0_old

        # ---- 6. gamma[k,j] k=0,1,2,3; j=0,1,2  ----------------------------
        for k in range(4):
            for j in range(3):
                g_old = params.gamma[k, j]
                params.gamma[k, j] = g_old + STEP_GAMMA * np.random.uniform(-1, 1)
                update_means(params, rows_u, cols_u, rows_l, cols_l)
                lpost_new = log_posterior(params)
                n_prop["gamma"] += 1
                if _mh_accept(lpost_new, lpost_cur):
                    lpost_cur = lpost_new
                    n_acc["gamma"] += 1
                else:
                    params.gamma[k, j] = g_old
                    update_means(params, rows_u, cols_u, rows_l, cols_l)

        # ---- 7. lambda[j] for j=0,1,2  -------------------------------------
        for j in range(3):
            lam_old = params.lam[j]
            lam_prop = lam_old + STEP_LAMBDA * np.random.uniform(-1, 1)
            n_prop["lam"] += 1
            if lam_prop > 0.0:
                params.lam[j] = lam_prop
                lpost_new = log_posterior(params)
                if _mh_accept(lpost_new, lpost_cur):
                    lpost_cur = lpost_new
                    n_acc["lam"] += 1
                else:
                    params.lam[j] = lam_old
            # If lam_prop <= 0: always reject (params unchanged)

        # ---- 8. sigma_rho  -------------------------------------------------
        sr_old = params.sigma_rho
        sr_prop = sr_old + STEP_SIGMA_RHO * np.random.uniform(-1, 1)
        n_prop["sigma_rho"] += 1
        if sr_prop > 0.0:
            params.sigma_rho = sr_prop
            lpost_new = log_posterior(params)
            if _mh_accept(lpost_new, lpost_cur):
                lpost_cur = lpost_new
                n_acc["sigma_rho"] += 1
            else:
                params.sigma_rho = sr_old

        # ---- 9. sigma_delta  -----------------------------------------------
        sd_old = params.sigma_delta
        sd_prop = sd_old + STEP_SIGMA_DELTA * np.random.uniform(-1, 1)
        n_prop["sigma_delta"] += 1
        if sd_prop > 0.0:
            params.sigma_delta = sd_prop
            lpost_new = log_posterior(params)
            if _mh_accept(lpost_new, lpost_cur):
                lpost_cur = lpost_new
                n_acc["sigma_delta"] += 1
            else:
                params.sigma_delta = sd_old

        # ---- 10. sigma_gamma  ----------------------------------------------
        sg_old = params.sigma_gamma
        sg_prop = sg_old + STEP_SIGMA_GAMMA * np.random.uniform(-1, 1)
        n_prop["sigma_gamma"] += 1
        if sg_prop > 0.0:
            params.sigma_gamma = sg_prop
            lpost_new = log_posterior(params)
            if _mh_accept(lpost_new, lpost_cur):
                lpost_cur = lpost_new
                n_acc["sigma_gamma"] += 1
            else:
                params.sigma_gamma = sg_old

        # ---- Store draw  ---------------------------------------------------
        draws[it] = params.to_vector()

        # ---- Verbose progress  ---------------------------------------------
        if verbose and (it + 1) % 1000 == 0:
            print(f"  Iteration {it + 1:>7d} / {n_iter}  "
                  f"lpost = {lpost_cur:10.3f}")
            if (it + 1) % 10_000 == 0:
                # Print acceptance rates over the last window
                for block in n_prop:
                    p = n_prop[block]
                    a = n_acc[block]
                    rate = a / p if p > 0 else float("nan")
                    print(f"    {block:<12s}  acc {rate:.3f}  "
                          f"({a}/{p})")
                # Reset counters
                for block in n_prop:
                    n_prop[block] = 0
                    n_acc[block] = 0

    return draws
