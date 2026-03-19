#!/usr/bin/env python3
"""
Full pipeline for the CGRG hierarchical brain network model.

Matches execute_bnet.m behavior:
  1. Load fMRI connectivity matrices (AD, MCI, NL groups)
  2. Initialise parameters
  3. Run Metropolis-Hastings sampler
  4. Save draws to .npz
  5. Generate posterior plots

Usage
-----
    python run_brain_model.py \\
        --data-dir /path/to/data \\
        --output-dir /path/to/output \\
        --n-iter 300000 \\
        [--seed 42] \\
        [--burn-in 50000]

Data directory must contain:
    AD_list.mat, MCI_list.mat, NL_list.mat

Output
------
    <output-dir>/draws.npz          -- raw draws array (n_iter x 42)
    <output-dir>/draws_postburn.npz -- post-burn-in draws
    <output-dir>/*.pdf              -- posterior plots
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CGRG hierarchical brain network model (Python port of MATLAB bnet_*).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing AD_list.mat, MCI_list.mat, NL_list.mat",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for output files (created if needed)",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=300_000,
        help="Number of MCMC iterations",
    )
    parser.add_argument(
        "--burn-in",
        type=int,
        default=50_000,
        help="Number of burn-in iterations to discard before plotting",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-iteration progress output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Random seed: {args.seed}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Imports (deferred so argparse --help works without scipy installed)
    # ------------------------------------------------------------------
    from cgrg.data import load_brain_data
    from cgrg.model import (default_params, make_triangle_indices,
                             precompute_stacked_data)
    from cgrg.mcmc import run_mcmc
    from cgrg.plotting import plot_all

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Loading data ...")
    data = load_brain_data(args.data_dir)
    n_ad  = len(data["ad"])
    n_mci = len(data["mci"])
    n_nl  = len(data["nl"])
    print(f"  AD={n_ad}  MCI={n_mci}  NL={n_nl}")

    # ------------------------------------------------------------------
    # 2. Build triangle indices
    # ------------------------------------------------------------------
    rows_u, cols_u, rows_l, cols_l = make_triangle_indices(4)
    uinds = (rows_u, cols_u)
    linds = (rows_l, cols_l)

    # ------------------------------------------------------------------
    # 3. Initialise parameters
    # ------------------------------------------------------------------
    print("Initialising parameters ...")
    params = default_params()
    precompute_stacked_data(params, data, rows_u, cols_u, rows_l, cols_l)

    # ------------------------------------------------------------------
    # 4. Run MCMC
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"Running MCMC: {args.n_iter:,} iterations")
    print(f"  burn-in (for plotting): {args.burn_in:,}")
    t0 = time.time()

    draws = run_mcmc(
        params=params,
        data=data,
        n_iter=args.n_iter,
        linds=linds,
        uinds=uinds,
        verbose=(not args.quiet),
    )

    elapsed = time.time() - t0
    print(f"MCMC complete in {elapsed:.1f} s  ({elapsed/args.n_iter*1000:.2f} ms/iter)")

    # ------------------------------------------------------------------
    # 5. Save draws
    # ------------------------------------------------------------------
    draws_path = output_dir / "draws.npz"
    np.savez_compressed(str(draws_path), draws=draws)
    print(f"Saved all draws -> {draws_path}")

    if args.burn_in > 0 and args.burn_in < args.n_iter:
        draws_pb = draws[args.burn_in :]
        pb_path = output_dir / "draws_postburn.npz"
        np.savez_compressed(str(pb_path), draws=draws_pb)
        print(f"Saved post-burn-in draws ({len(draws_pb):,} rows) -> {pb_path}")
    else:
        draws_pb = draws

    # ------------------------------------------------------------------
    # 6. Posterior summary (simple)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Posterior summary (post-burn-in means):")
    col_names = (
        ["rho0"]
        + [f"rho_{g}" for g in ["NL", "MCI", "AD"]]
        + [f"delta0_{i}" for i in range(4)]
        + [f"delta_{i}_{g}" for g in ["NL", "MCI", "AD"] for i in range(4)]
        + [f"gamma0_{i}" for i in range(4)]
        + [f"gamma_{i}_{g}" for g in ["NL", "MCI", "AD"] for i in range(4)]
        + [f"lambda_{g}" for g in ["NL", "MCI", "AD"]]
        + ["sigma_rho", "sigma_delta", "sigma_gamma"]
    )
    means = draws_pb.mean(axis=0)
    sds   = draws_pb.std(axis=0)
    for name, mu, sd in zip(col_names, means, sds):
        print(f"  {name:<20s}  mean={mu:8.4f}  sd={sd:.4f}")

    # ------------------------------------------------------------------
    # 7. Plots
    # ------------------------------------------------------------------
    print("=" * 60)
    plot_all(draws_pb, str(output_dir))

    print("=" * 60)
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
