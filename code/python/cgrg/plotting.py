"""
Posterior visualisation for the CGRG brain network model.

Recreates the MATLAB plot style:
  - Filled KDE curves (scipy.stats.gaussian_kde)
  - Colors: yellow, magenta, red, cyan  (matching MATLAB order)
  - Line width 1.2, fill alpha 0.3
  - Legend: 'Prior' (hyper-mean), 'NL', 'MCI', 'AD'

Draw array column layout (42 columns):
    col 0         rho0
    cols 1-3      rho[NL], rho[MCI], rho[AD]
    cols 4-7      delta0[0..3]
    cols 8-19     delta[0..3, 0..2]  (flattened column-major: node varies fastest)
                  i.e. delta[:,0] at cols 8-11, delta[:,1] at cols 12-15,
                       delta[:,2] at cols 16-19
    cols 20-23    gamma0[0..3]
    cols 24-35    gamma[0..3, 0..2]  (flattened column-major)
    cols 36-38    lambda[NL], lambda[MCI], lambda[AD]
    col 39        sigma_rho
    col 40        sigma_delta
    col 41        sigma_gamma
"""

import os
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend; caller may override
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# ---------------------------------------------------------------------------
# Color palette (MATLAB style)
# ---------------------------------------------------------------------------

COLOR_PRIOR  = "#FFFF00"   # yellow  -- hyper-mean / pooled
COLOR_NL     = "#FF00FF"   # magenta
COLOR_MCI    = "#FF0000"   # red
COLOR_AD     = "#00FFFF"   # cyan

COLORS = [COLOR_PRIOR, COLOR_NL, COLOR_MCI, COLOR_AD]
LABELS = ["Prior", "NL", "MCI", "AD"]

LINEWIDTH = 1.2
FILL_ALPHA = 0.3


# ---------------------------------------------------------------------------
# Column-index helpers
# ---------------------------------------------------------------------------

def rho0_col() -> int:
    return 0

def rho_cols() -> List[int]:
    """Columns for rho[NL], rho[MCI], rho[AD]."""
    return [1, 2, 3]

def delta0_cols() -> List[int]:
    """Columns for delta0[0..3]."""
    return [4, 5, 6, 7]

def delta_col(node: int, group: int) -> int:
    """
    Column index for delta[node, group].

    delta is stored column-major starting at col 8:
        cols 8-11  = delta[:, 0] (NL)
        cols 12-15 = delta[:, 1] (MCI)
        cols 16-19 = delta[:, 2] (AD)
    """
    return 8 + group * 4 + node

def gamma0_cols() -> List[int]:
    """Columns for gamma0[0..3]."""
    return [20, 21, 22, 23]

def gamma_col(node: int, group: int) -> int:
    """
    Column index for gamma[node, group].

    gamma stored column-major starting at col 24.
    """
    return 24 + group * 4 + node

def lambda_cols() -> List[int]:
    """Columns for lambda[NL], lambda[MCI], lambda[AD]."""
    return [36, 37, 38]


# ---------------------------------------------------------------------------
# Core KDE plot
# ---------------------------------------------------------------------------

def plot_group_posteriors(
    draws: np.ndarray,
    col_indices: Sequence[int],
    labels: Optional[Sequence[str]] = None,
    title: str = "",
    save_path: Optional[str] = None,
    colors: Optional[Sequence[str]] = None,
) -> plt.Figure:
    """
    Plot filled KDE curves for a set of columns from the draw array.

    Each column is one curve (e.g. rho0, rho_NL, rho_MCI, rho_AD).

    Parameters
    ----------
    draws : np.ndarray, shape (n_iter, 42)
    col_indices : sequence of int
        Which columns of draws to plot.  Length must be <= 4.
    labels : sequence of str, optional
        Curve labels.  Defaults to LABELS[:len(col_indices)].
    title : str
        Axes title.
    save_path : str or None
        If provided, save figure to this path as PDF.
    colors : sequence of str, optional
        Override the default color cycle.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if labels is None:
        labels = LABELS[: len(col_indices)]
    if colors is None:
        colors = COLORS[: len(col_indices)]

    fig, ax = plt.subplots(figsize=(6, 4))

    for col, lab, col_color in zip(col_indices, labels, colors):
        samples = draws[:, col]
        samples = samples[np.isfinite(samples)]
        if len(samples) < 5:
            continue

        kde = gaussian_kde(samples)
        x_min = samples.min() - 0.5 * np.ptp(samples)
        x_max = samples.max() + 0.5 * np.ptp(samples)
        x = np.linspace(x_min, x_max, 512)
        y = kde(x)

        ax.plot(x, y, color=col_color, linewidth=LINEWIDTH, label=lab)
        ax.fill_between(x, y, alpha=FILL_ALPHA, color=col_color)

    ax.set_title(title)
    ax.set_ylabel("Density")
    ax.legend(framealpha=0.6)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, format="pdf", bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Named plot functions
# ---------------------------------------------------------------------------

def plot_reciprocity(
    draws: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Posterior KDEs for reciprocity parameters: rho0, rho_NL, rho_MCI, rho_AD.

    Column order: [0=rho0, 1=rho_NL, 2=rho_MCI, 3=rho_AD].
    Curve labels:  Prior, NL, MCI, AD.
    """
    return plot_group_posteriors(
        draws,
        col_indices=[rho0_col()] + rho_cols(),
        labels=LABELS,
        title="Frank copula parameter (reciprocity)",
        save_path=save_path,
    )


def plot_sender_effect(
    draws: np.ndarray,
    node: int,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Posterior KDEs for sender (delta) effects at a given node.

    Parameters
    ----------
    node : int
        1-indexed node number as in MATLAB (valid: 2, 3, 4; node 1 is fixed).
        Internally converted to 0-indexed (1, 2, 3).
    """
    if node < 2 or node > 4:
        raise ValueError("node must be 2, 3, or 4 (node 1 is the reference, "
                         "fixed at 0)")
    node0 = node - 1  # 0-indexed
    cols = [delta0_cols()[node0]] + [delta_col(node0, g) for g in range(3)]
    return plot_group_posteriors(
        draws,
        col_indices=cols,
        labels=LABELS,
        title=f"Sender effect delta (node {node})",
        save_path=save_path,
    )


def plot_receiver_effect(
    draws: np.ndarray,
    node: int,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Posterior KDEs for receiver (gamma) effects at a given node.

    Parameters
    ----------
    node : int
        1-indexed node number (1, 2, 3, or 4).
    """
    if node < 1 or node > 4:
        raise ValueError("node must be 1, 2, 3, or 4")
    node0 = node - 1
    cols = [gamma0_cols()[node0]] + [gamma_col(node0, g) for g in range(3)]
    return plot_group_posteriors(
        draws,
        col_indices=cols,
        labels=LABELS,
        title=f"Receiver effect gamma (node {node})",
        save_path=save_path,
    )


def plot_precision(
    draws: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Posterior KDEs for precision parameters lambda[NL], lambda[MCI], lambda[AD].
    """
    return plot_group_posteriors(
        draws,
        col_indices=lambda_cols(),
        labels=["NL", "MCI", "AD"],
        colors=[COLOR_NL, COLOR_MCI, COLOR_AD],
        title="Precision lambda (std = 1/sqrt(lambda))",
        save_path=save_path,
    )


def plot_sigma_hyperparams(
    draws: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Posterior KDEs for sigma_rho, sigma_delta, sigma_gamma (cols 39, 40, 41).
    """
    return plot_group_posteriors(
        draws,
        col_indices=[39, 40, 41],
        labels=["sigma_rho", "sigma_delta", "sigma_gamma"],
        colors=[COLOR_PRIOR, COLOR_NL, COLOR_MCI],
        title="Hierarchical standard deviations",
        save_path=save_path,
    )


# ---------------------------------------------------------------------------
# Convenience: generate all plots
# ---------------------------------------------------------------------------

def plot_all(draws: np.ndarray, output_dir: str) -> None:
    """
    Generate all standard posterior plots and save as PDF files.

    Parameters
    ----------
    draws : np.ndarray, shape (n_iter, 42)
    output_dir : str
        Directory where PDFs will be written (created if needed).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Saving plots to {out} ...")

    # Reciprocity
    fig = plot_reciprocity(draws, save_path=str(out / "reciprocity.pdf"))
    plt.close(fig)
    print("  reciprocity.pdf")

    # Sender effects (nodes 2, 3, 4 — node 1 is fixed reference)
    for node in [2, 3, 4]:
        fname = f"sender_node{node}.pdf"
        fig = plot_sender_effect(draws, node,
                                 save_path=str(out / fname))
        plt.close(fig)
        print(f"  {fname}")

    # Receiver effects (all 4 nodes)
    for node in [1, 2, 3, 4]:
        fname = f"receiver_node{node}.pdf"
        fig = plot_receiver_effect(draws, node,
                                   save_path=str(out / fname))
        plt.close(fig)
        print(f"  {fname}")

    # Precision
    fig = plot_precision(draws, save_path=str(out / "precision.pdf"))
    plt.close(fig)
    print("  precision.pdf")

    # Sigma hyperparameters
    fig = plot_sigma_hyperparams(draws, save_path=str(out / "sigma_hyperparams.pdf"))
    plt.close(fig)
    print("  sigma_hyperparams.pdf")

    print("Done.")
