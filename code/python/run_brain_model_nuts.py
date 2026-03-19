#!/usr/bin/env python3
"""
NUTS sampler for the hierarchical CGRG brain network model via NumPyro/JAX.

Replaces the hand-rolled random-walk MH in run_brain_model.py with NUTS,
giving ~100-500x speedup on modern hardware.  Requires ARM Python 3.11+:

    /opt/homebrew/bin/python3.11 run_brain_model_nuts.py \\
        --data-dir ../../data \\
        --output-dir ../../output_nuts \\
        --num-warmup 1000 \\
        --num-samples 3000 \\
        --num-chains 2 \\
        --seed 42

Output
------
    <output-dir>/draws_postburn.npz   -- (num_samples, 42) float64 array,
                                         same 42-column layout as MH output
    <output-dir>/arviz_diagnostics.txt  -- R-hat and ESS table
    <output-dir>/*.pdf               -- posterior density plots
"""

import argparse
import sys
import time
from pathlib import Path

# ---- JAX float64 must be enabled BEFORE any jax import ----
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import arviz as az


# ---------------------------------------------------------------------------
# JAX-compatible Frank copula log density
# ---------------------------------------------------------------------------

_COPULA_EPS = 1e-10


def frank_log_density_jax(u, v, rho):
    """
    Numerically stable log Frank copula density, JAX-compatible.

    Mirrors copula.py exactly, using jnp ops throughout.
    rho must be nonzero (enforced by prior support away from 0).
    """
    u = jnp.clip(u, _COPULA_EPS, 1.0 - _COPULA_EPS)
    v = jnp.clip(v, _COPULA_EPS, 1.0 - _COPULA_EPS)

    # log|rho|
    log_abs_rho = jnp.log(jnp.abs(rho))

    # log|1 - exp(-rho)| = log|expm1(-rho)|  (correct for both signs of rho)
    log_abs_1_minus_exp = jnp.log(jnp.abs(jnp.expm1(-rho)))

    # Numerator: log|rho| + log|1-e^{-rho}| + (-rho*(u+v))
    # Must use actual rho (not abs), since sign matters for the exponent.
    log_num = log_abs_rho + log_abs_1_minus_exp - rho * (u + v)

    # Denominator inner: (1-e^{-rho}) - (1-e^{-rho*u})(1-e^{-rho*v})
    one_minus_exp_neg_rho   = -jnp.expm1(-rho)
    one_minus_exp_neg_rho_u = -jnp.expm1(-rho * u)
    one_minus_exp_neg_rho_v = -jnp.expm1(-rho * v)

    inner = one_minus_exp_neg_rho - one_minus_exp_neg_rho_u * one_minus_exp_neg_rho_v
    log_denom = 2.0 * jnp.log(jnp.abs(inner))

    return log_num - log_denom


# ---------------------------------------------------------------------------
# Group log-likelihood (JAX)
# ---------------------------------------------------------------------------

_HALF_LOG_2PI = 0.5 * jnp.log(2.0 * jnp.pi)


def _group_log_likelihood_jax(bel, beu, delta_g, gamma_g, lam_g, rho_g,
                               rows_l, cols_l, rows_u, cols_u):
    """
    Log-likelihood contribution from one diagnostic group.

    Parameters
    ----------
    bel, beu : jnp.ndarray, shape (6*S,)
        Stacked observed lower/upper triangle entries.
    delta_g : jnp.ndarray, shape (4,)   sender effects for this group
    gamma_g : jnp.ndarray, shape (4,)   receiver effects for this group
    lam_g   : scalar                    precision
    rho_g   : scalar                    Frank copula parameter
    rows_l, cols_l, rows_u, cols_u : static int arrays, shape (6,)
    """
    from jax.scipy.special import ndtr as _ndtr

    # Mean matrix: mmat[i, j] = delta[i] + gamma[j]
    mmat = delta_g[:, None] + gamma_g[None, :]   # (4, 4)

    mul_dyad = mmat[rows_l, cols_l]   # (6,)
    muu_dyad = mmat[rows_u, cols_u]   # (6,)

    S = bel.shape[0] // 6
    bmul = jnp.tile(mul_dyad, S)   # (6*S,)
    bmuu = jnp.tile(muu_dyad, S)   # (6*S,)

    sigma = 1.0 / jnp.sqrt(lam_g)
    log_sigma = jnp.log(sigma)

    zl = (bel - bmul) / sigma
    zu = (beu - bmuu) / sigma

    pl = _ndtr(zl)
    pu = _ndtr(zu)

    log_cop   = frank_log_density_jax(pl, pu, rho_g)
    log_marg_l = -_HALF_LOG_2PI - log_sigma - 0.5 * zl * zl
    log_marg_u = -_HALF_LOG_2PI - log_sigma - 0.5 * zu * zu

    return jnp.sum(log_cop) + jnp.sum(log_marg_l) + jnp.sum(log_marg_u)


# ---------------------------------------------------------------------------
# NumPyro model
# ---------------------------------------------------------------------------

def cgrg_model(data, rows_l, cols_l, rows_u, cols_u):
    """
    NumPyro model for the hierarchical CGRG brain network.

    data : dict with keys 'nl', 'mci', 'ad', each a dict
           {'bel': jnp.ndarray, 'beu': jnp.ndarray}
    rows_l, cols_l, rows_u, cols_u : jnp int arrays of shape (6,)
    """
    # ---- Hyperparameters -----------------------------------------------
    # sigma_* ~ Gamma(shape=0.1, scale=10)  =>  rate = 1/10 = 0.1
    sigma_rho   = numpyro.sample("sigma_rho",   dist.Gamma(0.1, rate=0.1))
    sigma_delta = numpyro.sample("sigma_delta", dist.Gamma(0.1, rate=0.1))
    sigma_gamma = numpyro.sample("sigma_gamma", dist.Gamma(0.1, rate=0.1))

    # ---- Reciprocity hierarchy -----------------------------------------
    # rho0 ~ N(0, 4)  [sd = 2]
    rho0 = numpyro.sample("rho0", dist.Normal(0.0, 2.0))
    # rho[g] ~ N(rho0, sigma_rho)  for g in {NL, MCI, AD}
    rho = numpyro.sample("rho", dist.Normal(rho0, sigma_rho).expand([3]))

    # ---- Sender effects hierarchy (delta) ------------------------------
    # delta0[0] = 0 fixed (reference node); sample delta0[1:4]
    delta0_free = numpyro.sample("delta0_free", dist.Normal(0.0, 2.0).expand([3]))
    delta0 = jnp.concatenate([jnp.zeros(1), delta0_free])   # (4,)

    # delta[0, :] = 0; sample delta[1:4, :] ~ N(delta0[1:4], sigma_delta)
    # shape (3, 3): nodes 1-3 x groups 0-2
    loc_delta = jnp.broadcast_to(delta0_free[:, jnp.newaxis], (3, 3))
    delta_free = numpyro.sample("delta_free", dist.Normal(loc_delta, sigma_delta))
    delta = jnp.concatenate([jnp.zeros((1, 3)), delta_free], axis=0)   # (4, 3)

    # ---- Receiver effects hierarchy (gamma) ----------------------------
    # gamma0 ~ N(0, 4)
    gamma0 = numpyro.sample("gamma0", dist.Normal(0.0, 2.0).expand([4]))
    # gamma[i, g] ~ N(gamma0[i], sigma_gamma)
    loc_gamma = jnp.broadcast_to(gamma0[:, jnp.newaxis], (4, 3))
    gamma = numpyro.sample("gamma", dist.Normal(loc_gamma, sigma_gamma))   # (4, 3)

    # ---- Precision parameters ------------------------------------------
    # lam[g] ~ Gamma(shape=5, scale=4)  =>  rate = 0.25
    lam = numpyro.sample("lam", dist.Gamma(5.0, rate=0.25).expand([3]))

    # ---- Likelihood (custom via numpyro.factor) ------------------------
    group_keys = [("nl", 0), ("mci", 1), ("ad", 2)]
    for gname, gidx in group_keys:
        bel = data[gname]["bel"]
        beu = data[gname]["beu"]
        ll = _group_log_likelihood_jax(
            bel, beu,
            delta[:, gidx], gamma[:, gidx],
            lam[gidx], rho[gidx],
            rows_l, cols_l, rows_u, cols_u,
        )
        numpyro.factor(f"ll_{gname}", ll)


# ---------------------------------------------------------------------------
# 42-column draw extraction (matches MH output format exactly)
# ---------------------------------------------------------------------------

def samples_to_draw_matrix(samples):
    """
    Convert NumPyro posterior samples dict to (N, 42) array matching the
    42-column layout used by the MH sampler and make_paper_figures.py.

    Layout:
        [0]     rho0
        [1:4]   rho   (NL, MCI, AD)
        [4:8]   delta0  (includes fixed 0 at index 4)
        [8:20]  delta flattened column-major: delta[:,0], delta[:,1], delta[:,2]
        [20:24] gamma0
        [24:36] gamma flattened column-major
        [36:39] lambda  (NL, MCI, AD)
        [39]    sigma_rho
        [40]    sigma_delta
        [41]    sigma_gamma
    """
    def _np(x):
        return np.array(x)

    N = len(_np(samples["rho0"]))

    rho0        = _np(samples["rho0"]).reshape(N, 1)          # (N, 1)
    rho         = _np(samples["rho"])                          # (N, 3)
    delta0_free = _np(samples["delta0_free"])                  # (N, 3)
    delta0      = np.concatenate([np.zeros((N, 1)), delta0_free], axis=1)  # (N, 4)
    delta_free  = _np(samples["delta_free"])                   # (N, 3, 3)
    delta_full  = np.concatenate([np.zeros((N, 1, 3)), delta_free], axis=1)  # (N, 4, 3)
    # Flatten column-major: delta[:,0] (NL), delta[:,1] (MCI), delta[:,2] (AD)
    # Matches model.py: v[8:20].reshape(4, 3, order='F') where groups fill columns
    delta_flat  = np.concatenate([
        delta_full[:, :, 0],   # NL:  (N, 4)
        delta_full[:, :, 1],   # MCI: (N, 4)
        delta_full[:, :, 2],   # AD:  (N, 4)
    ], axis=1)                                                 # (N, 12)

    gamma0      = _np(samples["gamma0"])                       # (N, 4)
    gamma_full  = _np(samples["gamma"])                        # (N, 4, 3)
    gamma_flat  = np.concatenate([
        gamma_full[:, :, 0],
        gamma_full[:, :, 1],
        gamma_full[:, :, 2],
    ], axis=1)                                                 # (N, 12)

    lam         = _np(samples["lam"])                          # (N, 3)
    sigma_rho   = _np(samples["sigma_rho"]).reshape(N, 1)
    sigma_delta = _np(samples["sigma_delta"]).reshape(N, 1)
    sigma_gamma = _np(samples["sigma_gamma"]).reshape(N, 1)

    return np.concatenate([
        rho0, rho, delta0, delta_flat, gamma0, gamma_flat,
        lam, sigma_rho, sigma_delta, sigma_gamma,
    ], axis=1)   # (N, 42)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="NUTS sampler for CGRG brain network model (NumPyro/JAX).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir",    required=True)
    p.add_argument("--output-dir",  required=True)
    p.add_argument("--num-warmup",  type=int, default=1000)
    p.add_argument("--num-samples", type=int, default=3000)
    p.add_argument("--num-chains",  type=int, default=2)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--target-accept-prob", type=float, default=0.8,
                   help="NUTS target acceptance probability")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load data (numpy/scipy path — identical to existing pipeline)
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("Loading data ...")
    # Add cgrg package to path
    sys.path.insert(0, str(Path(__file__).parent))
    from cgrg.data import load_brain_data
    from cgrg.model import make_triangle_indices

    data_np = load_brain_data(args.data_dir)
    n_nl  = len(data_np["nl"])
    n_mci = len(data_np["mci"])
    n_ad  = len(data_np["ad"])
    print(f"  NL={n_nl}  MCI={n_mci}  AD={n_ad}")

    # ------------------------------------------------------------------ #
    # 2. Build triangle indices and stack data into JAX arrays
    # ------------------------------------------------------------------ #
    rows_u, cols_u, rows_l, cols_l = make_triangle_indices(4)
    rows_l_jnp = jnp.array(rows_l)
    cols_l_jnp = jnp.array(cols_l)
    rows_u_jnp = jnp.array(rows_u)
    cols_u_jnp = jnp.array(cols_u)

    def stack_group(matrices):
        bel = np.concatenate([m[rows_l, cols_l] for m in matrices])
        beu = np.concatenate([m[rows_u, cols_u] for m in matrices])
        return {"bel": jnp.array(bel), "beu": jnp.array(beu)}

    data_jax = {
        "nl":  stack_group(data_np["nl"]),
        "mci": stack_group(data_np["mci"]),
        "ad":  stack_group(data_np["ad"]),
    }

    # ------------------------------------------------------------------ #
    # 3. Run NUTS
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print(f"Running NUTS: {args.num_warmup} warmup + {args.num_samples} samples "
          f"x {args.num_chains} chains")

    numpyro.set_host_device_count(args.num_chains)

    nuts_kernel = NUTS(
        cgrg_model,
        target_accept_prob=args.target_accept_prob,
    )
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        progress_bar=True,
    )

    t0 = time.time()
    mcmc.run(
        jax.random.PRNGKey(args.seed),
        data_jax,
        rows_l_jnp, cols_l_jnp, rows_u_jnp, cols_u_jnp,
    )
    elapsed = time.time() - t0
    print(f"\nNUTS complete in {elapsed:.1f} s")

    # ------------------------------------------------------------------ #
    # 4. Extract samples and convert to 42-column format
    # ------------------------------------------------------------------ #
    samples = mcmc.get_samples(group_by_chain=False)

    # When num_chains > 1, samples from group_by_chain=False are concatenated
    draws = samples_to_draw_matrix(samples)
    print(f"Draw matrix shape: {draws.shape}")

    # Save in the same .npz format as the MH pipeline
    pb_path = output_dir / "draws_postburn.npz"
    np.savez_compressed(str(pb_path), draws=draws)
    print(f"Saved post-warmup draws -> {pb_path}")

    # Also save per-chain for R-hat
    samples_by_chain = mcmc.get_samples(group_by_chain=True)
    for c in range(args.num_chains):
        chain_samples = {k: v[c] for k, v in samples_by_chain.items()}
        chain_draws = samples_to_draw_matrix(chain_samples)
        chain_path = output_dir / f"chain{c+1}" / "draws_postburn.npz"
        chain_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(chain_path), draws=chain_draws)
        print(f"Saved chain {c+1} draws -> {chain_path}")

    # ------------------------------------------------------------------ #
    # 5. ArviZ diagnostics
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("CONVERGENCE DIAGNOSTICS (ArviZ)")
    print("=" * 60)

    idata = az.from_numpyro(mcmc)

    # Summary for key parameters
    param_names = ["rho0", "rho", "lam", "sigma_rho", "sigma_delta", "sigma_gamma"]
    summary = az.summary(idata, var_names=param_names, round_to=4)
    print(summary.to_string())

    # Save diagnostics text
    diag_path = output_dir / "arviz_diagnostics.txt"
    with open(str(diag_path), "w") as f:
        f.write(f"NUTS diagnostics\n")
        f.write(f"Warmup: {args.num_warmup}  Samples: {args.num_samples}  "
                f"Chains: {args.num_chains}\n")
        f.write(f"Runtime: {elapsed:.1f} s\n\n")
        f.write(summary.to_string())
        f.write("\n\nFull summary (all parameters):\n")
        full_summary = az.summary(idata, round_to=4)
        f.write(full_summary.to_string())
    print(f"\nSaved diagnostics -> {diag_path}")

    # ------------------------------------------------------------------ #
    # 6. Posterior plots (reuse make_paper_figures.py logic)
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("Generating plots ...")
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "make_paper_figures.py"),
             "--chain1", str(pb_path),
             "--chain2", str(output_dir / "chain2" / "draws_postburn.npz"),
             "--paper-dir", str(output_dir / "figures"),
             "--thin", "1"],
            capture_output=True, text=True, check=True,
        )
        print(result.stdout)
    except Exception as e:
        print(f"  (Plots skipped: {e})")

    print("\nDone.")


if __name__ == "__main__":
    main()
