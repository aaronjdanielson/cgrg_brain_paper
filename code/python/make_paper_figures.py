#!/usr/bin/env python3
"""
Post-processing: generate paper-ready figures and compute posterior summaries.

Produces:
  - paper/Reciprocity.pdf           (rho posteriors: clean KDE, no fill)
  - paper/NodeEffects.pdf           (forest plot: receiver + sender CIs)
  - paper/ppc_kendall_tau.pdf       (PPC dot-and-CI figure, regenerated)
  - printed LaTeX-ready table rows

Usage
-----
    cd /path/to/cgrg_brain_paper/code/python
    python make_paper_figures.py \\
        --chain1 ../../output_python_plots/nuts_fixed/draws_postburn.npz \\
        --chain2 ../../output_python_plots/nuts_fixed/chain2/draws_postburn.npz \\
        --data-dir ../../data \\
        --paper-dir ../../paper \\
        --thin 10
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import gaussian_kde, kendalltau

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "pdf.fonttype":      42,   # embeds fonts properly
})

# Paper colours
C = {"NL": "#444444", "MCI": "#1565C0", "AD": "#C62828", "rho0": "#AAAAAA"}
LS = {"NL": "-", "MCI": "--", "AD": ":"}    # line styles
LW = 1.8                                     # line width for posteriors
GROUP_NAMES = ["NL", "MCI", "AD"]
NODE_NAMES  = ["mPFC", "PCC", "LIPL", "RIPL"]


# ---------------------------------------------------------------------------
# Column-index helpers
# ---------------------------------------------------------------------------

def rho0_col():     return 0
def rho_col(g):     return 1 + g
def gamma0_col(n):  return 20 + n
def gamma_col(n,g): return 24 + g * 4 + n
def delta0_col(n):  return 4 + n
def delta_col(n,g): return 8 + g * 4 + n
def lam_col(g):     return 36 + g


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def ci95(s):
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))

def posterior_summary(s):
    lo, hi = ci95(s)
    return {"mean": float(np.mean(s)), "sd": float(np.std(s, ddof=1)),
            "lo": lo, "hi": hi}

def kde_curve(samples, n=600):
    s = samples[np.isfinite(samples)]
    if len(s) < 10:
        return None, None
    kde = gaussian_kde(s, bw_method="silverman")
    lo, hi = ci95(s)
    pad = 2.5 * s.std()
    x = np.linspace(s.min() - pad, s.max() + pad, n)
    return x, kde(x)

def rhat_two_chains(c1, c2):
    n = min(len(c1), len(c2))
    a, b = c1[:n], c2[:n]
    m1, m2 = a.mean(), b.mean()
    gm = (m1 + m2) / 2
    W  = (a.var(ddof=1) + b.var(ddof=1)) / 2
    B  = n * ((m1 - gm)**2 + (m2 - gm)**2)
    vh = (1 - 1/n) * W + B/n
    return float(np.sqrt(vh / W)) if W > 1e-15 else float("nan")


# ---------------------------------------------------------------------------
# Figure 1: Reciprocity posteriors
# ---------------------------------------------------------------------------

def fig_reciprocity(draws, paper):
    """Clean KDE lines — no fill — for rho0 (hyper-mean) and three groups."""
    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    # Determine x-range from group posteriors only
    all_group = np.concatenate([draws[:, rho_col(g)] for g in range(3)])
    xlim = (all_group.min() - 0.8, all_group.max() + 0.8)

    # rho0 (hyper-mean): thin light-grey dashed line
    x, y = kde_curve(draws[:, rho0_col()])
    if x is not None:
        ax.plot(x, y, color="#BBBBBB", linewidth=1.2, linestyle="--",
                label=r"$\rho_0$ (hyper-mean)", zorder=2)

    # Group posteriors: solid/dashed/dotted lines, no fill
    for g, gname in enumerate(GROUP_NAMES):
        s = draws[:, rho_col(g)]
        x, y = kde_curve(s)
        if x is None:
            continue
        ax.plot(x, y, color=C[gname], linewidth=LW, linestyle=LS[gname],
                label=gname, zorder=3 + g)

        # 95% CI tick marks at the bottom of the plot
        lo, hi = ci95(s)
        ax.annotate("", xy=(hi, -0.02), xytext=(lo, -0.02),
                    xycoords=("data", "axes fraction"),
                    textcoords=("data", "axes fraction"),
                    arrowprops=dict(arrowstyle="-", color=C[gname],
                                   lw=1.4, shrinkA=0, shrinkB=0))

    ax.axvline(0, color="#CCCCCC", linewidth=0.8, linestyle=":", zorder=1,
               label=r"$\rho = 0$ (independence)")

    ax.set_xlim(xlim)
    ax.set_xlabel(r"$\rho$  (Frank copula reciprocity parameter)")
    ax.set_ylabel("Posterior density")
    ax.set_title(r"Posterior distributions of $\rho^{(g)}$")
    ax.legend(loc="upper right", framealpha=0.7, edgecolor="#CCCCCC")
    ax.set_ylim(bottom=0)

    # Secondary x-axis: Kendall's tau
    # tau(rho) ≈ 1 - 4/rho*(1 - D1(rho)); approximate by numeric table
    def tau_approx(rho_val):
        """Approximate Kendall's tau for Frank copula via the Debye function."""
        if abs(rho_val) < 1e-4:
            return 0.0
        from scipy.integrate import quad
        D1, _ = quad(lambda t: t / (np.exp(t) - 1), 0, rho_val)
        D1 /= rho_val
        tau = 1.0 - 4.0 / rho_val * (1.0 - D1)
        return tau

    ax2 = ax.twiny()
    ax2.set_xlim(xlim)
    # Place a few tau tick marks at representative rho values
    rho_ticks = [-6, -5, -4, -3, -2, -1]
    tau_labels = [f"{tau_approx(r):.2f}" for r in rho_ticks]
    ax2.set_xticks(rho_ticks)
    ax2.set_xticklabels(tau_labels, fontsize=8)
    ax2.set_xlabel(r"Kendall's $\tau$", fontsize=9, labelpad=4)
    ax2.spines["top"].set_visible(True)
    ax2.spines["top"].set_linewidth(0.6)
    ax2.tick_params(top=True, which="major", length=3, width=0.6)

    fig.tight_layout()
    out = str(paper / "Reciprocity.pdf")
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 2: Node effects forest plot
# ---------------------------------------------------------------------------

def fig_node_effects(draws, paper):
    """
    Two-panel forest plot (coefficient plot) for receiver and sender effects.
    Left panel: gamma_j^(g) for j=1..4
    Right panel: delta_i^(g) for i=2,3,4
    Each panel has one row per node, three coloured CIs per row.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8),
                             gridspec_kw={"wspace": 0.45})

    offsets = {"NL": -0.18, "MCI": 0.0, "AD": 0.18}  # vertical jitter

    def draw_node_panel(ax, param_fn, nodes, node_labels, title, xlabel):
        for row_idx, (n, nlab) in enumerate(zip(nodes, node_labels)):
            for g, gname in enumerate(GROUP_NAMES):
                s = draws[:, param_fn(n, g)]
                m = np.mean(s)
                lo, hi = ci95(s)
                y = row_idx + offsets[gname]
                ax.plot([lo, hi], [y, y], color=C[gname], linewidth=2.0,
                        solid_capstyle="round", zorder=2)
                ax.scatter(m, y, color=C[gname], s=28, zorder=3,
                           marker="o" if gname == "NL"
                           else "s" if gname == "MCI" else "^")

        ax.axvline(0, color="#BBBBBB", linewidth=0.8, linestyle="--", zorder=1)
        ax.set_yticks(range(len(nodes)))
        ax.set_yticklabels(node_labels, fontsize=9)
        ax.set_ylim(-0.6, len(nodes) - 0.4)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.invert_yaxis()

    # Receiver effects gamma
    draw_node_panel(
        axes[0], gamma_col,
        nodes=[0, 1, 2, 3],
        node_labels=[f"$\\gamma_{j+1}$ ({NODE_NAMES[j]})" for j in range(4)],
        title=r"Receiver effects $\gamma_j^{(g)}$",
        xlabel=r"Posterior mean and 95% CI"
    )

    # Sender effects delta (nodes 2,3,4 only; node 1 is reference)
    draw_node_panel(
        axes[1], delta_col,
        nodes=[1, 2, 3],
        node_labels=[f"$\\delta_{j+1}$ ({NODE_NAMES[j]})" for j in [1, 2, 3]],
        title=r"Sender effects $\delta_i^{(g)}$ (rel. to mPFC)",
        xlabel=r"Posterior mean and 95% CI"
    )

    # Shared legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=C[g], linewidth=2, marker="o" if g=="NL"
               else "s" if g=="MCI" else "^", markersize=6, label=g)
        for g in GROUP_NAMES
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, framealpha=0.7,
               edgecolor="#CCCCCC", bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout()
    out = str(paper / "NodeEffects.pdf")
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 3: PPC – Kendall's tau (dot-and-CI style)
# ---------------------------------------------------------------------------

def frank_conditional_inv(u, t, rho):
    """Conditional quantile inversion for Frank copula."""
    u = np.asarray(u, dtype=float)
    t = np.asarray(t, dtype=float)
    eps = 1e-10
    if abs(rho) < 1e-6:
        return t.copy()
    a  = np.expm1(-rho)
    eu = np.expm1(-rho * u)
    denom = np.exp(-rho * u) - t * eu
    denom = np.where(np.abs(denom) < eps, eps * np.sign(denom + eps), denom)
    arg = 1.0 + t * a / denom
    arg = np.clip(arg, eps, None)
    return np.clip(-np.log(arg) / rho, eps, 1 - eps)


def simulate_group_networks(draw, g, S, rng):
    """Simulate S networks for group g from one posterior draw."""
    from scipy.stats import norm as spnorm
    rho_g = float(draw[rho_col(g)])
    lam_g = float(draw[lam_col(g)])
    sd    = 1.0 / np.sqrt(lam_g)
    delta = np.array([0.0] + [float(draw[delta_col(n, g)]) for n in range(1, 4)])
    gamma = np.array([float(draw[gamma_col(n, g)]) for n in range(4)])
    rows_u, cols_u = np.triu_indices(4, k=1)
    mats = []
    for _ in range(S):
        A = np.zeros((4, 4))
        for i in range(4):
            A[i, i] = rng.normal(delta[i] + gamma[i], sd)
        u_u = rng.uniform(0, 1, len(rows_u))
        t_u = rng.uniform(0, 1, len(rows_u))
        v_u = frank_conditional_inv(u_u, t_u, rho_g)
        for k in range(len(rows_u)):
            i, j = rows_u[k], cols_u[k]
            A[i, j] = spnorm.ppf(u_u[k], loc=delta[i]+gamma[j], scale=sd)
            A[j, i] = spnorm.ppf(v_u[k], loc=delta[j]+gamma[i], scale=sd)
        mats.append(A)
    return mats


def compute_dyad_tau(matrices, rows_u, cols_u):
    A = np.stack(matrices, axis=0)
    return {(rows_u[k], cols_u[k]): kendalltau(A[:, rows_u[k], cols_u[k]],
                                                A[:, cols_u[k], rows_u[k]])[0]
            for k in range(len(rows_u))}


def fig_ppc_tau(draws, data, paper, n_ppc=200, thin=10):
    """
    Dot-and-CI forest plot: for each dyad and group, show observed tau
    vs PPC 95% interval.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from cgrg.data import load_brain_data as _load
    if data is None:
        return

    rows_u, cols_u = np.triu_indices(4, k=1)
    rng = np.random.default_rng(2025)
    DYAD_LABELS = [f"({rows_u[k]+1},{cols_u[k]+1})" for k in range(len(rows_u))]

    ppc_draws_all = draws[::thin]
    idx = rng.choice(len(ppc_draws_all), size=min(n_ppc, len(ppc_draws_all)),
                     replace=False)
    ppc_draws = ppc_draws_all[idx]

    group_data = [data["nl"], data["mci"], data["ad"]]
    S_g = [len(d) for d in group_data]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharey=True)

    for gi, (gname, gdata) in enumerate(zip(GROUP_NAMES, group_data)):
        ax = axes[gi]
        S  = S_g[gi]

        # Observed tau per dyad
        obs_tau = compute_dyad_tau(gdata, rows_u, cols_u)

        # PPC tau: simulate and collect
        ppc_tau_matrix = np.zeros((len(ppc_draws), len(rows_u)))
        for pi, draw in enumerate(ppc_draws):
            sim  = simulate_group_networks(draw, gi, S, rng)
            stau = compute_dyad_tau(sim, rows_u, cols_u)
            for k in range(len(rows_u)):
                ppc_tau_matrix[pi, k] = stau.get((rows_u[k], cols_u[k]), np.nan)

        ppc_lo  = np.nanpercentile(ppc_tau_matrix, 2.5,  axis=0)
        ppc_hi  = np.nanpercentile(ppc_tau_matrix, 97.5, axis=0)
        ppc_med = np.nanmedian(ppc_tau_matrix, axis=0)

        y = np.arange(len(rows_u))
        color = C[gname]

        # PPC interval as shaded band
        for k in range(len(rows_u)):
            ax.barh(y[k], ppc_hi[k] - ppc_lo[k], left=ppc_lo[k],
                    height=0.45, color=color, alpha=0.20, zorder=1)
            ax.plot([ppc_med[k]], [y[k]], marker="|", color=color,
                    markersize=7, markeredgewidth=1.5, zorder=2)

        # Observed as filled dot
        obs_vals = [obs_tau.get((rows_u[k], cols_u[k]), np.nan)
                    for k in range(len(rows_u))]
        ax.scatter(obs_vals, y, color=color, s=40, zorder=3,
                   label="Observed", marker="o")

        ax.axvline(0, color="#CCCCCC", linewidth=0.8, linestyle="--", zorder=0)
        ax.set_yticks(y)
        ax.set_yticklabels(DYAD_LABELS if gi == 0 else [], fontsize=8)
        ax.set_xlabel(r"Kendall's $\tau$")
        ax.set_title(gname, color=color, fontweight="bold")
        ax.set_xlim(-0.85, 0.5)
        ax.invert_yaxis()

        if gi == 0:
            from matplotlib.patches import Patch
            from matplotlib.lines import Line2D
            ax.legend(handles=[
                Patch(facecolor=color, alpha=0.25, label="PPC 95% interval"),
                Line2D([0],[0], marker="|", color=color, linewidth=0,
                       markersize=8, markeredgewidth=1.5, label="PPC median"),
                Line2D([0],[0], marker="o", color=color, linewidth=0,
                       markersize=6, label="Observed"),
            ], loc="lower right", fontsize=7.5, framealpha=0.7,
               edgecolor="#CCCCCC")

    axes[1].set_title("Posterior predictive check: dyad Kendall's " + r"$\tau$",
                      fontsize=10)
    fig.tight_layout()
    out = str(paper / "ppc_kendall_tau.pdf")
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 0: "What rho means" — copula structure at rho in {-4, 0, +4}
# ---------------------------------------------------------------------------

def fig_rho_illustration(paper):
    """
    Three panels showing the Frank copula in uniform space for ρ = -4, 0, +4.
    Filled contours (density) + simulated scatter.  Makes ρ immediately
    intuitive for any reader.
    """
    rho_vals = [-4, 0, 4]
    titles = [
        r"$\rho = -4$   ($\tau \approx -0.43$)" + "\n" + "anti-reciprocal",
        r"$\rho = 0$" + "\n" + "independence",
        r"$\rho = +4$   ($\tau \approx +0.43$)" + "\n" + "reciprocal",
    ]
    cmaps   = ["Reds", "Greys", "Blues"]
    colors  = ["#C62828", "#777777", "#1565C0"]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))
    rng = np.random.default_rng(1234)

    for ax, rho, title, cmap, col in zip(axes, rho_vals, titles, cmaps, colors):
        n = 400
        u = rng.uniform(0, 1, n)
        if abs(rho) < 1e-6:
            v = rng.uniform(0, 1, n)
        else:
            t = rng.uniform(0, 1, n)
            v = frank_conditional_inv(u, t, rho)

        if abs(rho) > 1e-6:
            Ug, Vg, Dg = frank_density_grid(rho, n=70)
            vmax = np.percentile(Dg, 97)
            Dg_c = np.clip(Dg, 0, vmax)
            levels = np.linspace(np.percentile(Dg, 20), vmax, 8)
            ax.contourf(Ug, Vg, Dg_c, levels=levels, cmap=cmap, alpha=0.50)
            ax.contour(Ug, Vg, Dg_c, levels=levels, colors='k',
                       linewidths=0.4, alpha=0.30)

        ax.scatter(u, v, s=7, alpha=0.55, color=col, linewidths=0, zorder=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9.5, color=col if rho != 0 else "k")
        ax.set_xlabel(r"$u_{ij} = \Phi(z_{ij})$", fontsize=8.5)
        if ax is axes[0]:
            ax.set_ylabel(r"$u_{ji} = \Phi(z_{ji})$", fontsize=8.5)
        ax.tick_params(labelsize=8)

    fig.suptitle(r"Frank copula structure as a function of $\rho$   "
                 r"(uniform margins, marginal structure removed)",
                 fontsize=9.5, y=1.02)
    fig.tight_layout()
    out = str(paper / "RhoIllustration.pdf")
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 00: Copula comparison — Frank vs Gaussian vs Clayton (90°-rotated)
#            all matched to Kendall's tau ≈ -0.43
# ---------------------------------------------------------------------------

def _gaussian_copula_density_grid(r, n=70):
    """Gaussian copula density on n×n grid over (0,1)²."""
    from scipy.stats import norm as spnorm
    u = np.linspace(0.02, 0.98, n)
    U, V = np.meshgrid(u, u)
    x = spnorm.ppf(U)
    y = spnorm.ppf(V)
    r2 = r**2
    log_d = -0.5 * np.log(1 - r2) - (r2*(x**2 + y**2) - 2*r*x*y) / (2*(1-r2))
    return U, V, np.clip(np.exp(log_d), 0, None)


def _clayton90_density_grid(theta, n=70):
    """90°-rotated Clayton density grid for negative dependence.

    c_{rot90}(u,v;θ) = c_{clay}(1−u, v; θ).
    """
    u = np.linspace(0.02, 0.98, n)
    U, V = np.meshgrid(u, u)
    eps = 1e-10
    U1 = np.clip(1 - U, eps, 1-eps)
    Vc = np.clip(V,      eps, 1-eps)
    base = U1**(-theta) + Vc**(-theta) - 1
    base = np.clip(base, eps, None)
    D = (1 + theta) * (U1 * Vc)**(-1-theta) * base**(-(2 + 1/theta))
    return U, V, np.clip(D, 0, None)


def _sample_gaussian_copula(r, n, rng):
    from scipy.stats import norm as spnorm
    Z = rng.multivariate_normal([0, 0], [[1, r], [r, 1]], n)
    return spnorm.cdf(Z[:, 0]), spnorm.cdf(Z[:, 1])


def _sample_clayton90(theta, n, rng):
    """Sample from 90°-rotated Clayton (negative dependence)."""
    eps = 1e-10
    u = np.clip(rng.uniform(0, 1, n), eps, 1-eps)
    t = np.clip(rng.uniform(0, 1, n), eps, 1-eps)
    # Clayton conditional inverse: v = [(t·u^{1+θ})^{-θ/(θ+1)} − u^{-θ} + 1]^{−1/θ}
    inner = (t * u**(1+theta))**(-theta/(theta+1)) - u**(-theta) + 1
    inner = np.clip(inner, eps, None)
    v = np.clip(inner**(-1/theta), eps, 1-eps)
    return 1 - u, v  # 90° rotation


def fig_copula_comparison(paper):
    """
    2×3 panel.  Top row: Frank copula density contour.  Bottom row: 400-point scatter.
    Columns: Frank (ρ=-3.75), Gaussian (r≈-0.62), Clayton-90° (θ≈1.51),
    all matched to Kendall's τ ≈ -0.43.
    Demonstrates why Frank is chosen: symmetric anti-diagonal density that
    matches the observed dyad scatter better than the elliptical Gaussian
    or the asymmetrically-tailed Clayton.
    """
    tau = -0.43
    rho_f = -3.75
    r_g   = float(np.sin(np.pi * tau / 2))           # ≈ -0.624
    th_c  = float(-2 * tau / (1 + tau))               # ≈ 1.51 (Clayton 90°)

    rng   = np.random.default_rng(42)
    n_pts = 500

    col_params = [
        ("Frank",         r"$\rho = -3.75$",  "#444444"),
        ("Gaussian",      r"$r \approx -0.62$", "#1565C0"),
        ("Clayton (90°)", r"$\theta \approx 1.51$", "#C62828"),
    ]
    grids = [
        frank_density_grid(rho_f, n=70),
        _gaussian_copula_density_grid(r_g, n=70),
        _clayton90_density_grid(th_c, n=70),
    ]
    samps = [
        _sample_frank(rho_f, n_pts, rng),
        _sample_gaussian_copula(r_g, n_pts, rng),
        _sample_clayton90(th_c, n_pts, rng),
    ]

    # Compute common percentile levels across all three densities so that
    # contours are visually comparable
    all_vals = np.concatenate([Dg.ravel() for _, _, Dg in grids])
    pcts = np.percentile(all_vals[all_vals > 0], [30, 50, 65, 78, 89, 96])

    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.6),
                             gridspec_kw={"hspace": 0.48, "wspace": 0.28})

    for col, ((Ug, Vg, Dg), (us, vs), (name, param_str, c)) in enumerate(
            zip(grids, samps, col_params)):

        # --- top row: density ---
        ax = axes[0, col]
        Dgc = np.clip(Dg, 0, np.percentile(Dg, 99))
        ax.contourf(Ug, Vg, Dgc, levels=pcts, cmap="Blues_r", alpha=0.70)
        ax.contour(Ug, Vg, Dgc, levels=pcts, colors='k',
                   linewidths=0.5, alpha=0.35)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(f"{name}\n{param_str}", fontsize=9, color=c)
        if col == 0:
            ax.set_ylabel("Density contour", fontsize=8)
        ax.tick_params(labelsize=8)

        # --- bottom row: scatter ---
        ax2 = axes[1, col]
        ax2.scatter(us, vs, s=5, alpha=0.45, color=c, linewidths=0)
        ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
        ax2.set_aspect("equal")
        ax2.set_xlabel(r"$u_{ij}$", fontsize=8)
        if col == 0:
            ax2.set_ylabel("Simulated sample", fontsize=8)
        ax2.tick_params(labelsize=8)

    # Annotation: highlight differences
    axes[0, 0].text(0.05, 0.92, "symmetric\nanti-diagonal",
                    transform=axes[0, 0].transAxes,
                    fontsize=7.5, color="#C62828", va="top",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    axes[0, 1].text(0.05, 0.92, "elliptical\n(less flexible)",
                    transform=axes[0, 1].transAxes,
                    fontsize=7.5, color="#1565C0", va="top",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    axes[0, 2].text(0.05, 0.92, "asymmetric\ntail behavior",
                    transform=axes[0, 2].transAxes,
                    fontsize=7.5, color="#C62828", va="top",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    fig.suptitle(r"Copula families matched to Kendall's $\tau \approx -0.43$   "
                 r"(brain data posterior)",
                 fontsize=9.5)
    out = str(paper / "CopulaComparison.pdf")
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def _sample_frank(rho, n, rng):
    u = rng.uniform(0, 1, n)
    t = rng.uniform(0, 1, n)
    return u, frank_conditional_inv(u, t, rho)


# ---------------------------------------------------------------------------
# Figure 4: Dyad scatter in PIT space + Frank copula contour
# ---------------------------------------------------------------------------

def frank_density_grid(rho, n=80):
    """Frank copula density on an n×n grid over (0,1)²."""
    u = np.linspace(0.02, 0.98, n)
    U, V = np.meshgrid(u, u)
    eps = 1e-12
    num = rho * (1.0 - np.exp(-rho)) * np.exp(-rho * (U + V))
    denom = (1.0 - np.exp(-rho)
             - (1.0 - np.exp(-rho * U)) * (1.0 - np.exp(-rho * V))) ** 2
    denom = np.where(np.abs(denom) < eps, eps, denom)
    D = num / denom
    D = np.clip(D, 0, None)
    return U, V, D


def fig_dyad_scatter(draws, data, paper):
    """
    Three-panel figure.  Each panel (one per group) shows observed dyad pairs
    (u_ij, u_ji) after probability-integral transformation using posterior mean
    parameters, pooled across all 6 dyads.  Overlaid: Frank copula density
    contour at posterior mean rho.  The anti-diagonal smear visualises
    negative reciprocity directly.
    """
    if data is None:
        return
    from scipy.stats import norm as spnorm

    rho_pm = [float(np.mean(draws[:, rho_col(g)])) for g in range(3)]
    lam_pm = [float(np.mean(draws[:, lam_col(g)])) for g in range(3)]
    delta_pm = [np.array([0.0] + [float(np.mean(draws[:, delta_col(n, g)]))
                                   for n in range(1, 4)])
                for g in range(3)]
    gamma_pm = [np.array([float(np.mean(draws[:, gamma_col(n, g)]))
                           for n in range(4)])
                for g in range(3)]

    group_data = [data["nl"], data["mci"], data["ad"]]
    rows_u, cols_u = np.triu_indices(4, k=1)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))

    for gi, (gname, gdata) in enumerate(zip(GROUP_NAMES, group_data)):
        ax = axes[gi]
        sd = 1.0 / np.sqrt(lam_pm[gi])

        u_list, v_list = [], []
        for mat in gdata:
            for k in range(len(rows_u)):
                i, j = rows_u[k], cols_u[k]
                mu_ij = delta_pm[gi][i] + gamma_pm[gi][j]
                mu_ji = delta_pm[gi][j] + gamma_pm[gi][i]
                u_ij = float(spnorm.cdf((mat[i, j] - mu_ij) / sd))
                u_ji = float(spnorm.cdf((mat[j, i] - mu_ji) / sd))
                u_list.append(u_ij)
                v_list.append(u_ji)

        u_arr = np.clip(u_list, 0.01, 0.99)
        v_arr = np.clip(v_list, 0.01, 0.99)

        # Frank density contour
        Ug, Vg, Dg = frank_density_grid(rho_pm[gi])
        levels = np.percentile(Dg[Dg > 0], [40, 60, 75, 88, 95])
        ax.contour(Ug, Vg, Dg, levels=levels, colors=C[gname],
                   linewidths=0.9, alpha=0.7)

        ax.scatter(u_arr, v_arr, color=C[gname], s=8, alpha=0.40,
                   linewidths=0, zorder=3)

        # Anti-diagonal reference line
        ax.plot([0, 1], [1, 0], color="#AAAAAA", linewidth=1.0,
                linestyle="--", zorder=1, alpha=0.8)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$u_{ij}$  (PIT of $A_{ij}$)", fontsize=9)
        if gi == 0:
            ax.set_ylabel(r"$u_{ji}$  (PIT of $A_{ji}$)", fontsize=9)
        title_str = (f"{gname}   "
                     r"$\hat\rho = $" + f"{rho_pm[gi]:.2f}")
        ax.set_title(title_str, color=C[gname], fontweight="bold")
        ax.tick_params(labelsize=8)
        # Label for first panel only
        if gi == 0:
            ax.text(0.97, 0.97, "hierarchical\nstructure",
                    ha="right", va="top", transform=ax.transAxes,
                    fontsize=7.5, color="#888888",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.8))

    fig.suptitle(
        r"Observed dyad pairs in PIT space  —  contours: fitted Frank copula  —  dashed: anti-diagonal",
        fontsize=9.5, y=1.01)
    fig.tight_layout()
    out = str(paper / "DyadScatter.pdf")
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 5: E[Aij] heatmap and asymmetry heatmap
# ---------------------------------------------------------------------------

def fig_heatmaps(draws, paper):
    """
    Two rows × three columns.
    Top row: posterior mean E[Aij] = mean(delta_i + gamma_j) for each (i,j).
    Bottom row: asymmetry = E[Aij] - E[Aji].
    """
    mean_A = np.zeros((3, 4, 4))   # group × i × j
    for gi in range(3):
        for i in range(4):
            di = 0.0 if i == 0 else float(np.mean(draws[:, delta_col(i, gi)]))
            for j in range(4):
                gj = float(np.mean(draws[:, gamma_col(j, gi)]))
                mean_A[gi, i, j] = di + gj

    asym_A = mean_A - mean_A.transpose(0, 2, 1)

    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.8),
                             gridspec_kw={"hspace": 0.45, "wspace": 0.35})

    vmin_m = mean_A.min();  vmax_m = mean_A.max()
    vabs   = np.max(np.abs(asym_A))

    for gi, gname in enumerate(GROUP_NAMES):
        # --- top: mean A ---
        ax = axes[0, gi]
        im = ax.imshow(mean_A[gi], vmin=vmin_m, vmax=vmax_m,
                       cmap="viridis", aspect="equal")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{mean_A[gi,i,j]:.2f}",
                        ha="center", va="center", fontsize=7.5,
                        color="white" if mean_A[gi,i,j] < (vmin_m+vmax_m)/2 else "black")
        ax.set_xticks(range(4));  ax.set_xticklabels(NODE_NAMES, fontsize=7, rotation=25)
        ax.set_yticks(range(4));  ax.set_yticklabels(NODE_NAMES, fontsize=7)
        ax.set_title(gname, color=C[gname], fontweight="bold")
        if gi == 0:
            ax.set_ylabel(r"$E[A_{ij}]$ (sender → row, receiver → col)",
                          fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # --- bottom: asymmetry ---
        ax = axes[1, gi]
        im2 = ax.imshow(asym_A[gi], vmin=-vabs, vmax=vabs,
                        cmap="RdBu_r", aspect="equal")
        for i in range(4):
            for j in range(4):
                val = asym_A[gi, i, j]
                ax.text(j, i, f"{val:+.2f}",
                        ha="center", va="center", fontsize=7.5,
                        color="white" if abs(val) > vabs * 0.55 else "black")
        ax.set_xticks(range(4));  ax.set_xticklabels(NODE_NAMES, fontsize=7, rotation=25)
        ax.set_yticks(range(4));  ax.set_yticklabels(NODE_NAMES, fontsize=7)
        if gi == 0:
            ax.set_ylabel(r"$E[A_{ij}] - E[A_{ji}]$ (asymmetry)", fontsize=8)
        plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(r"Posterior mean connectivity and dyadic asymmetry across DMN nodes",
                 fontsize=10)
    out = str(paper / "Heatmaps.pdf")
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 6: Copula gain over independence (model validation)
# ---------------------------------------------------------------------------

def _frank_logdens(u, v, rho):
    """Scalar-safe Frank log-density; returns 0.0 when rho≈0."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    u = np.clip(u, 1e-10, 1 - 1e-10)
    v = np.clip(v, 1e-10, 1 - 1e-10)
    if abs(rho) < 1e-6:
        return np.zeros_like(u)
    log_abs_rho = np.log(abs(rho))
    log_abs_1m  = np.log(np.abs(np.expm1(-rho)))
    log_num     = log_abs_rho + log_abs_1m - rho * (u + v)
    a   = -np.expm1(-rho)
    au  = -np.expm1(-rho * u)
    av  = -np.expm1(-rho * v)
    inn = a - au * av
    log_den = 2.0 * np.log(np.abs(inn))
    return log_num - log_den


def fig_copula_gain(draws, data, paper):
    """
    For each subject, compute the average copula log-likelihood gain:

        gain_s = (1/6) Σ_{dyads} log c_ρ(u_ij, u_ji)

    where (u_ij, u_ji) are the PIT values under posterior mean parameters.
    When ρ < 0 and data show anti-diagonal structure, gain > 0 compared to the
    independence copula (log c_0 = 0).  A violin + strip plot per group.
    """
    if data is None:
        return
    from scipy.stats import norm as spnorm

    rho_pm  = [float(np.mean(draws[:, rho_col(g)])) for g in range(3)]
    lam_pm  = [float(np.mean(draws[:, lam_col(g)])) for g in range(3)]
    delta_pm = [np.array([0.0] + [float(np.mean(draws[:, delta_col(n, g)]))
                                   for n in range(1, 4)])
                for g in range(3)]
    gamma_pm = [np.array([float(np.mean(draws[:, gamma_col(n, g)]))
                           for n in range(4)])
                for g in range(3)]

    group_data = [data["nl"], data["mci"], data["ad"]]
    rows_u, cols_u = np.triu_indices(4, k=1)
    n_dyads = len(rows_u)

    gains_all = []
    group_labels = []
    for gi, (gname, gdata) in enumerate(zip(GROUP_NAMES, group_data)):
        sd = 1.0 / np.sqrt(lam_pm[gi])
        for mat in gdata:
            g_s = 0.0
            for k in range(n_dyads):
                i, j = rows_u[k], cols_u[k]
                mu_ij = delta_pm[gi][i] + gamma_pm[gi][j]
                mu_ji = delta_pm[gi][j] + gamma_pm[gi][i]
                u_ij = float(spnorm.cdf((mat[i, j] - mu_ij) / sd))
                u_ji = float(spnorm.cdf((mat[j, i] - mu_ji) / sd))
                g_s += float(_frank_logdens(u_ij, u_ji, rho_pm[gi]))
            gains_all.append(g_s / n_dyads)
            group_labels.append(gname)

    gains_all   = np.array(gains_all)
    group_labels = np.array(group_labels)

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    positions = [0, 1, 2]
    for pos, gname in zip(positions, GROUP_NAMES):
        vals = gains_all[group_labels == gname]
        # violin
        parts = ax.violinplot(vals, positions=[pos], widths=0.55,
                              showmedians=False, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(C[gname])
            pc.set_alpha(0.30)
            pc.set_edgecolor(C[gname])
        # individual points
        ax.scatter(np.full(len(vals), pos) + np.random.default_rng(42).uniform(
                       -0.15, 0.15, len(vals)),
                   vals, color=C[gname], s=18, alpha=0.65, linewidths=0, zorder=3)
        # median line
        ax.hlines(np.median(vals), pos - 0.22, pos + 0.22,
                  color=C[gname], linewidth=2.0, zorder=4)

    ax.axhline(0, color="#CCCCCC", linewidth=0.9, linestyle="--", zorder=1,
               label=r"Independence ($\rho=0$, gain = 0)")
    ax.set_xticks(positions)
    ax.set_xticklabels(GROUP_NAMES)
    ax.set_ylabel(r"Mean copula log-density gain per dyad")
    ax.set_title(r"Copula gain over independence   ($\hat\rho \approx -3.75$)")
    ax.legend(fontsize=8, framealpha=0.7, edgecolor="#CCCCCC")

    fig.tight_layout()
    out = str(paper / "CopulaGain.pdf")
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 7: Network diagram — DMN hierarchy
# ---------------------------------------------------------------------------

def fig_network_diagram(draws, paper):
    """
    Directed-graph visualisation of posterior mean DMN connectivity.

    - Node size  ∝  receiver effect γ_j  (averaged over groups)
    - Arrow width ∝  E[A_ij] = δ_i + γ_j  (rescaled to [1, 7] pt)
    - Arrow color = asymmetry E[A_ij] − E[A_ji]  (RdBu_r: red = net forward)
    - Curved arrows prevent overlap; curvature direction is consistent for
      each ordered pair.
    """
    from matplotlib.patches import FancyArrowPatch
    import matplotlib.colors as mcolors

    # --- posterior means averaged over the three groups ---
    gamma_g = np.array([
        [np.mean(draws[:, gamma_col(n, g)]) for g in range(3)]
        for n in range(4)
    ])
    delta_g = np.zeros((4, 3))
    for n in range(1, 4):
        for g in range(3):
            delta_g[n, g] = np.mean(draws[:, delta_col(n, g)])

    gamma_m = gamma_g.mean(axis=1)   # (4,)
    delta_m = delta_g.mean(axis=1)   # (4,)

    E_A = np.array([[delta_m[i] + gamma_m[j] for j in range(4)]
                    for i in range(4)])  # (4,4)
    asym = E_A - E_A.T                   # asym[i,j]>0 → i→j dominant

    # --- layout ---
    # Diamond: mPFC top, PCC bottom, LIPL left, RIPL right
    pos = {0: np.array([0.50, 0.88]),   # mPFC
           1: np.array([0.50, 0.12]),   # PCC
           2: np.array([0.12, 0.50]),   # LIPL
           3: np.array([0.88, 0.50])}   # RIPL
    node_labels = NODE_NAMES

    # --- colour / width scales ---
    vabs = np.abs(asym[~np.eye(4, dtype=bool)]).max()
    cmap = plt.cm.RdBu_r
    norm = mcolors.Normalize(vmin=-vabs, vmax=vabs)

    w_min, w_max = E_A[~np.eye(4, dtype=bool)].min(), \
                   E_A[~np.eye(4, dtype=bool)].max()

    def edge_lw(val):
        return 0.8 + 5.5 * (val - w_min) / (w_max - w_min + 1e-12)

    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.axis("off")

    # --- curvature assignments: keep consistent per unordered pair ---
    rad_sign = {(0, 1): +0.30, (1, 0): -0.30,
                (0, 2): +0.30, (2, 0): -0.30,
                (0, 3): -0.30, (3, 0): +0.30,
                (1, 2): -0.30, (2, 1): +0.30,
                (1, 3): +0.30, (3, 1): -0.30,
                (2, 3): +0.30, (3, 2): -0.30}

    node_r = 0.085   # node radius for arrow offset

    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            pi, pj = pos[i], pos[j]
            unit = (pj - pi) / np.linalg.norm(pj - pi)
            start = pi + unit * node_r
            end   = pj - unit * node_r
            rad   = rad_sign.get((i, j), 0.25)
            lw    = edge_lw(E_A[i, j])
            col   = cmap(norm(asym[i, j]))
            hw    = 0.012 + 0.003 * lw
            hl    = 0.018 + 0.003 * lw
            ax.add_patch(FancyArrowPatch(
                start, end,
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle=f"->,head_width={hw},head_length={hl}",
                linewidth=lw, color=col, alpha=0.80, zorder=2))

    # --- nodes ---
    r_base = 0.062
    g_norm = (gamma_m - gamma_m.min()) / (gamma_m.max() - gamma_m.min() + 1e-12)
    for n in range(4):
        r = r_base * (0.75 + 0.45 * g_norm[n])
        circle = plt.Circle(pos[n], r, color="white",
                            ec="#333333", lw=1.6, zorder=3)
        ax.add_patch(circle)
        ax.text(pos[n][0], pos[n][1], node_labels[n],
                ha="center", va="center", fontsize=8.5,
                fontweight="bold", zorder=4)

    # --- colourbar ---
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.032, pad=0.02, shrink=0.55,
                        location="right")
    cbar.set_label(r"Asymmetry $E[A_{ij}] - E[A_{ji}]$", fontsize=7.5)
    cbar.ax.tick_params(labelsize=7)

    ax.text(0.02, 0.02,
            "Arrow width $\\propto E[A_{ij}]$\nNode size $\\propto \\gamma_j$",
            transform=ax.transAxes, fontsize=7, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

    ax.set_title("DMN posterior mean connectivity\n"
                 r"(averaged over NL / MCI / AD — pattern identical)",
                 fontsize=9)
    fig.tight_layout()
    out = str(paper / "NetworkDiagram.pdf")
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Copula comparison table  (Frank vs Gaussian vs Clayton-90°)
# ---------------------------------------------------------------------------

def _gauss_logdens(u, v, r):
    from scipy.stats import norm as spnorm
    eps = 1e-10
    u = np.clip(u, eps, 1-eps); v = np.clip(v, eps, 1-eps)
    x = spnorm.ppf(u); y = spnorm.ppf(v)
    r2 = r**2
    return (-0.5 * np.log(1 - r2)
            - (r2*(x**2 + y**2) - 2*r*x*y) / (2*(1-r2)))


def _clay90_logdens(u, v, theta):
    """Log-density of 90°-rotated Clayton: c_clay(1-u, v; θ)."""
    eps = 1e-10
    u1 = np.clip(1 - u, eps, 1-eps)
    vc = np.clip(v,       eps, 1-eps)
    base = np.clip(u1**(-theta) + vc**(-theta) - 1, eps, None)
    return (np.log(1 + theta)
            - (1 + theta) * (np.log(u1) + np.log(vc))
            - (2 + 1/theta) * np.log(base))


def _rho_to_tau(rho):
    """Frank copula Kendall's tau via Debye function."""
    from scipy.integrate import quad
    if abs(rho) < 1e-6:
        return 0.0
    D1, _ = quad(lambda t: t / (np.exp(t) - 1), 0, rho)
    D1 /= rho
    return 1.0 - 4.0 / rho * (1.0 - D1)


def _copula_entropy_numeric(rho, n=200):
    """h_C(rho) = -∫∫ c_rho log c_rho du dv  (numerical, via density grid)."""
    if abs(rho) < 1e-6:
        return 0.0
    _, _, Dg = frank_density_grid(rho, n)
    safe = np.clip(Dg, 1e-300, None)
    return float(-np.mean(Dg * np.log(safe)))


def fig_entropy_decomp(draws, paper):
    """
    Two-panel entropy decomposition figure for Appendix C.

    Left  — Posterior KDE of mutual information I(A_ij; A_ji) = -h_C(rho^(g))
             for each group.  All groups concentrated at the same value,
             confirming disease does not alter within-dyad information.

    Right — Per-dyad joint entropy decomposition:
             H(A_ij, A_ji) = 2 H_margin(lambda^(g))  +  h_C(rho^(g)).
             Shown as posterior-mean bars split into the marginal (independence)
             baseline and the copula reduction, with 95% CI brackets.
    """
    from scipy.interpolate import interp1d

    rho_draws = draws[:, 1:4]    # (S, 3)  NL / MCI / AD
    lam_draws = draws[:, 36:39]  # (S, 3)

    # Precompute h_C on a fine rho grid then interpolate (fast)
    rho_flat  = rho_draws.ravel()
    r_lo, r_hi = rho_flat.min() - 0.2, rho_flat.max() + 0.2
    rho_grid  = np.linspace(r_lo, r_hi, 80)
    hC_grid   = np.array([_copula_entropy_numeric(r) for r in rho_grid])
    hC_fn     = interp1d(rho_grid, hC_grid, kind="cubic",
                         fill_value="extrapolate")

    hC  = hC_fn(rho_draws)                          # (S, 3), ≤ 0
    MI  = -hC                                        # mutual information, ≥ 0
    # Marginal differential entropy of Normal(mu, precision=lambda)
    H_marg = 0.5 * (np.log(2 * np.pi) + 1.0 - np.log(lam_draws))  # (S, 3)
    # Per-dyad joint entropy
    H_dyad = 2.0 * H_marg + hC                      # (S, 3)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))

    # ── Left: posterior KDE of mutual information ────────────────────────────
    ax = axes[0]
    for gi, g in enumerate(GROUP_NAMES):
        x, y = kde_curve(MI[:, gi])
        if x is None:
            continue
        ax.plot(x, y, color=C[g], lw=LW, ls=LS[g], label=g)
        pm = float(np.mean(MI[:, gi]))
        ax.axvline(pm, color=C[g], lw=0.8, ls=":", alpha=0.55)

    ax.axvline(0, color="black", lw=0.7, ls="--", alpha=0.35,
               label="independence")
    ax.set_xlabel(r"Mutual information $I(A_{ij};\,A_{ji})$ (nats)", fontsize=9)
    ax.set_ylabel("Posterior density", fontsize=9)
    ax.set_title(r"Copula mutual information $-h_C(\rho^{(g)})$", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    # ── Right: per-dyad entropy decomposition ────────────────────────────────
    ax = axes[1]
    x_pos = np.arange(3)
    width = 0.45

    baseline_mean = (2.0 * H_marg).mean(axis=0)    # independence baseline
    hC_mean       = hC.mean(axis=0)                 # copula reduction (< 0)
    hC_lo         = np.percentile(hC, 2.5,  axis=0)
    hC_hi         = np.percentile(hC, 97.5, axis=0)
    dyad_lo       = np.percentile(H_dyad, 2.5,  axis=0)
    dyad_hi       = np.percentile(H_dyad, 97.5, axis=0)
    dyad_mean     = H_dyad.mean(axis=0)

    for gi, g in enumerate(GROUP_NAMES):
        col = C[g]
        # Independence baseline (light, hatched)
        ax.bar(x_pos[gi], baseline_mean[gi], width,
               color=col, alpha=0.25, edgecolor=col, linewidth=0.8,
               label="Marginal $2H(A_{ij})$" if gi == 0 else "")
        # Copula reduction stacked on top (the bar goes down into negative)
        ax.bar(x_pos[gi], hC_mean[gi], width,
               bottom=baseline_mean[gi], color=col, alpha=0.75,
               edgecolor=col, linewidth=0.8,
               label=r"Copula $h_C(\rho)$" if gi == 0 else "")
        # 95% CI on total
        ax.errorbar(x_pos[gi], dyad_mean[gi],
                    yerr=[[dyad_mean[gi] - dyad_lo[gi]],
                          [dyad_hi[gi]   - dyad_mean[gi]]],
                    fmt="none", color="black", capsize=3, lw=1.0, zorder=5)

    ax.axhline(0, color="black", lw=0.5, ls="-", alpha=0.3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(GROUP_NAMES, fontsize=9)
    ax.set_ylabel(r"$H(A_{ij}, A_{ji})$ (nats)", fontsize=9)
    ax.set_title("Per-dyad joint entropy decomposition", fontsize=10)

    # Legend: one entry per bar type
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="grey", alpha=0.25, edgecolor="grey",
              label=r"Marginal $2H(A_{ij})$"),
        Patch(facecolor="grey", alpha=0.75, edgecolor="grey",
              label=r"Copula $h_C(\rho)$ (reduction)"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=7.5, loc="lower right")

    fig.tight_layout()
    out = paper / "EntropyDecomp.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def print_copula_comparison(draws, data):
    """
    For each subject compute mean log copula density per dyad under three models:
      Frank  — posterior mean rho per group
      Gaussian copula  — r matched via Kendall's tau = tau(rho)
      Clayton 90°      — theta matched via same tau
    Print a LaTeX-ready table.
    """
    if data is None:
        return
    from scipy.stats import norm as spnorm

    rho_pm  = [float(np.mean(draws[:, rho_col(g)])) for g in range(3)]
    lam_pm  = [float(np.mean(draws[:, lam_col(g)])) for g in range(3)]
    delta_pm = [np.array([0.0] + [float(np.mean(draws[:, delta_col(n, g)]))
                                   for n in range(1, 4)])
                for g in range(3)]
    gamma_pm = [np.array([float(np.mean(draws[:, gamma_col(n, g)]))
                           for n in range(4)])
                for g in range(3)]

    group_data = [data["nl"], data["mci"], data["ad"]]
    rows_u, cols_u = np.triu_indices(4, k=1)
    n_dyads = len(rows_u)

    all_frank, all_gauss, all_clay = [], [], []

    for gi, gdata in enumerate(group_data):
        rho_g = rho_pm[gi]
        sd    = 1.0 / np.sqrt(lam_pm[gi])
        tau_g = _rho_to_tau(rho_g)
        r_g   = float(np.sin(np.pi * tau_g / 2))
        th_g  = float(-2 * tau_g / (1 + tau_g))    # θ ≥ 0 since τ < 0

        for mat in gdata:
            lf = lg = lc = 0.0
            for k in range(n_dyads):
                i, j = rows_u[k], cols_u[k]
                u = float(np.clip(spnorm.cdf(
                    (mat[i, j] - delta_pm[gi][i] - gamma_pm[gi][j]) / sd
                ), 1e-10, 1-1e-10))
                v = float(np.clip(spnorm.cdf(
                    (mat[j, i] - delta_pm[gi][j] - gamma_pm[gi][i]) / sd
                ), 1e-10, 1-1e-10))
                lf += float(_frank_logdens(u, v, rho_g))
                lg += float(_gauss_logdens(u, v, r_g))
                lc += float(_clay90_logdens(u, v, th_g))
            all_frank.append(lf / n_dyads)
            all_gauss.append(lg / n_dyads)
            all_clay.append(lc  / n_dyads)

    all_frank = np.array(all_frank)
    all_gauss = np.array(all_gauss)
    all_clay  = np.array(all_clay)

    print("\n" + "=" * 60)
    print("COPULA COMPARISON  (mean log-density per dyad, 112 subjects)")
    print("=" * 60)
    fmt = "{:<20s}  {:>7.4f}  {:>7.4f}  {:>7.4f}"
    print(f"{'Model':<20s}  {'Mean':>7}  {'Median':>7}  {'SD':>7}")
    for name, vals in [("Frank",         all_frank),
                       ("Gaussian",       all_gauss),
                       ("Clayton (90°)",  all_clay)]:
        print(fmt.format(name, vals.mean(), float(np.median(vals)), vals.std()))

    print("\nLaTeX table:\n")
    lines = [
        r"\begin{table}[htb]",
        r"\caption{Mean per-dyad log copula density for three families evaluated on",
        r"  the brain data using posterior mean marginal parameters.  Copula parameters",
        r"  are matched to the same Kendall's $\tau$ (Frank: posterior mean $\hat\rho$;",
        r"  Gaussian and Clayton: $\tau$-matched via $r = \sin(\pi\tau/2)$ and",
        r"  $\theta = -2\tau/(1+\tau)$, respectively).  Frank provides the best",
        r"  average fit across all 112 participants.}",
        r"\label{tab:copula_comparison}",
        r"\begin{center}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Copula family & Mean & Median & SD \\",
        r"\midrule",
    ]
    for name, vals in [("Frank",              all_frank),
                       ("Gaussian copula",     all_gauss),
                       ("Clayton (90\\degree)", all_clay)]:
        lines.append(
            f"{name} & ${vals.mean():.3f}$ & ${np.median(vals):.3f}$ "
            f"& ${vals.std():.3f}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{center}", r"\end{table}"]
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Posterior summaries
# ---------------------------------------------------------------------------

def print_summaries(draws):
    print()
    print("=" * 70)
    print("POSTERIOR SUMMARIES")
    print("=" * 70)
    print(f"\n{'Group':<5}  {'Mean':>7}  {'SD':>6}  {'2.5%':>8}  {'97.5%':>8}")
    rho_samps = {}
    for g, gname in enumerate(GROUP_NAMES):
        s = draws[:, rho_col(g)]
        rho_samps[gname] = s
        sm = posterior_summary(s)
        print(f"{gname:<5}  {sm['mean']:>7.3f}  {sm['sd']:>6.3f}  "
              f"{sm['lo']:>8.3f}  {sm['hi']:>8.3f}")

    prob = float(np.mean((rho_samps["NL"] < rho_samps["MCI"]) &
                         (rho_samps["MCI"] < rho_samps["AD"])))
    print(f"\nPr(rho_NL < rho_MCI < rho_AD) = {prob:.4f}")

    print("\nLaTeX rows:")
    for g, gname in enumerate(GROUP_NAMES):
        s = draws[:, rho_col(g)]
        sm = posterior_summary(s)
        print(f"  {gname}  &  {sm['mean']:.3f}  &  {sm['sd']:.3f}  "
              f"&  [{sm['lo']:.3f},\\\\,{sm['hi']:.3f}]  \\\\\\\\")

    print("\n--- Node effects (receiver gamma) ---")
    for n, nname in enumerate(NODE_NAMES):
        parts = []
        for g in range(3):
            sm = posterior_summary(draws[:, gamma_col(n, g)])
            parts.append(f"{sm['mean']:.2f} [{sm['lo']:.2f},\\,{sm['hi']:.2f}]")
        print(f"  $\\gamma_{n+1}$ & {nname} & " + " & ".join(parts) + " \\\\")

    print("\n--- Node effects (sender delta, nodes 2-4) ---")
    for n in [1, 2, 3]:
        parts = []
        for g in range(3):
            sm = posterior_summary(draws[:, delta_col(n, g)])
            parts.append(f"{sm['mean']:.2f} [{sm['lo']:.2f},\\,{sm['hi']:.2f}]")
        print(f"  $\\delta_{n+1}$ & {NODE_NAMES[n]} & " + " & ".join(parts) + " \\\\")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--chain1",    required=True)
    parser.add_argument("--chain2",    default=None)
    parser.add_argument("--data-dir",  default=None,
                        help="Brain data directory (for PPC figure)")
    parser.add_argument("--paper-dir", required=True)
    parser.add_argument("--thin",      type=int, default=10)
    parser.add_argument("--n-ppc",     type=int, default=200)
    args = parser.parse_args()

    paper = Path(args.paper_dir)
    paper.mkdir(parents=True, exist_ok=True)

    raw1   = np.load(args.chain1)["draws"]
    draws1 = raw1[::args.thin] if args.thin > 1 else raw1
    print(f"Chain 1: {len(raw1)} draws → {len(draws1)} after thinning")

    if args.chain2:
        raw2   = np.load(args.chain2)["draws"]
        draws2 = raw2[::args.thin] if args.thin > 1 else raw2
        print(f"Chain 2: {len(raw2)} draws → {len(draws2)} after thinning")
        # Pool chains for figures
        draws = np.concatenate([draws1, draws2], axis=0)
        print(f"Pooled: {len(draws)} draws")
    else:
        draws  = draws1
        draws2 = None

    # Load brain data for PPC
    data = None
    if args.data_dir:
        sys.path.insert(0, str(Path(__file__).parent))
        from cgrg.data import load_brain_data
        data = load_brain_data(args.data_dir)

    # --- Generate figures ---
    fig_rho_illustration(paper)
    fig_copula_comparison(paper)
    fig_reciprocity(draws, paper)
    fig_node_effects(draws, paper)
    fig_heatmaps(draws, paper)

    fig_network_diagram(draws, paper)
    fig_entropy_decomp(draws, paper)

    if data is not None:
        fig_ppc_tau(draws, data, paper, n_ppc=args.n_ppc, thin=1)
        fig_dyad_scatter(draws, data, paper)
        fig_copula_gain(draws, data, paper)

    # --- R-hat diagnostics ---
    if draws2 is not None:
        print("\nGelman-Rubin R-hat:")
        params = (
            [("rho0", rho0_col())]
            + [(f"rho_{g}", rho_col(i)) for i, g in enumerate(GROUP_NAMES)]
        )
        for pname, col in params:
            r = rhat_two_chains(draws1[:, col], draws2[:, col])
            flag = "  *** WARN" if r > 1.05 else ""
            print(f"  {pname:<20s}  R-hat = {r:.4f}{flag}")

    print_summaries(draws)
    print_copula_comparison(draws, data)
    print("\nDone.")


if __name__ == "__main__":
    main()
