# Block-Separable Copula Generated Random Graphs for Directed Networks

Replication code for:

> Cao, J. and Danielson, A.J. (2026). *Block-Separable Copula Generated Random
> Graphs for Directed Networks.* Working paper.

The model captures dyadic reciprocity in replicated directed networks using a
Frank copula with additive sender/receiver effects. It is applied to effective
connectivity networks in the default mode network (DMN) from resting-state fMRI
across 112 participants (NL / MCI / AD).

![Copula comparison at matched Kendall's τ ≈ −0.43: Frank (left), Gaussian (center), 90°-rotated Clayton (right). Top row: density contours; bottom row: 500 simulated pairs.](paper/CopulaComparison.png)

---

## Repository layout

```
code/
  python/
    cgrg/                    # Core model package (copula, likelihood, MCMC, data)
    run_brain_model_nuts.py  # Step 1: NUTS sampler — produces posterior draws
    make_paper_figures.py    # Step 2: generates all figures and tables
  legacy/                    # Earlier MATLAB, R, and Python prototypes (reference only)
data/
  AD_list.mat                # Effective connectivity matrices (12 AD subjects)
  MCI_list.mat               # (63 MCI subjects)
  NL_list.mat                # (37 cognitively normal subjects)
  sim1_list4paper.mat        # Simulation study data (Scenario 1)
  sim2_list4paper.mat        # Simulation study data (Scenario 2)
paper/
  CGRG_brain.tex             # LaTeX source
  CGRG_brain.pdf             # Compiled paper
  *.pdf / *.png              # All figures included in the paper
```

MCMC output (`output_python_plots/`) is not committed — regenerate with Step 1 below.

---

## Dependencies

Python 3.11+ is required. Install:

```bash
pip install numpyro jax numpy scipy matplotlib networkx arviz
```

Tested with:

| Package    | Version |
|------------|---------|
| numpyro    | 0.20.0  |
| jax        | 0.9.2   |
| numpy      | 2.2.6   |
| scipy      | 1.14.1  |
| matplotlib | 3.9.2   |
| networkx   | 3.3     |
| arviz      | 0.21+   |

**Apple Silicon (M1/M2/M3):** use the Homebrew Python 3.11 at
`/opt/homebrew/bin/python3.11` rather than the system Python to get
JAX Metal acceleration.

---

## Replication: step by step

All commands assume `code/python/` as the working directory.

### Step 1 — Run the NUTS sampler (chain 1)

```bash
python run_brain_model_nuts.py \
    --data-dir ../../data \
    --output-dir ../../output_python_plots/nuts_fixed \
    --num-warmup 1000 \
    --num-samples 6000 \
    --num-chains 1 \
    --seed 42
```

Expected runtime: ~5 minutes on Apple Silicon M-series; ~15 minutes on CPU.

### Step 2 — Run a second independent chain (for convergence diagnostics)

```bash
python run_brain_model_nuts.py \
    --data-dir ../../data \
    --output-dir ../../output_python_plots/nuts_fixed/chain2 \
    --num-warmup 1000 \
    --num-samples 3000 \
    --num-chains 1 \
    --seed 123
```

### Step 3 — Check convergence diagnostics

Each run writes `arviz_diagnostics.txt` to its output directory. Check:

```bash
cat ../../output_python_plots/nuts_fixed/arviz_diagnostics.txt
```

**What to look for:**

| Diagnostic | Target | Notes |
|------------|--------|-------|
| R-hat | < 1.05 | All ρ and λ parameters should pass easily |
| ESS (bulk) | > 400 | ρ and λ parameters typically reach 600–1100 |
| ESS (tail)  | > 400 | |

Expected output (from the paper's chains):

```
               mean      sd    hdi_3%   hdi_97%   ess_bulk  ess_tail   r_hat
rho[0]      -3.778   0.302   -4.335    -3.195      747.8    1347.6    1.004   ← NL
rho[1]      -3.767   0.278   -4.307    -3.243      656.5    1449.5    1.002   ← MCI
rho[2]      -3.717   0.351   -4.369    -3.065      658.1     724.3    1.005   ← AD
lam[0]       5.416   0.375    4.702     6.101      957.5    1651.8    1.002   ← NL
lam[1]       5.713   0.296    5.175     6.263     1090.8    1890.2    1.002   ← MCI
lam[2]       4.505   0.517    3.518     5.440     1095.7    1790.3    1.003   ← AD
sigma_rho    0.175   0.356    0.001     0.687      100.4     284.5    1.014
sigma_delta  0.009   0.015    0.000     0.036       52.5      58.9    1.029
sigma_gamma  0.009   0.012    0.000     0.032       55.1      77.8    1.036
```

> **Note:** `sigma_delta` and `sigma_gamma` have lower ESS (~50–100). These
> hierarchical variance parameters are concentrated near zero and mix slowly —
> this is expected behaviour and does not affect the key parameters (ρ, λ).
> The paper's conclusions rest on ρ and λ, which all have R-hat < 1.01.

The `make_paper_figures.py` script also prints R-hat for all ρ parameters
when `--chain2` is provided (see Step 4).

### Step 4 — Generate all figures and tables

```bash
python make_paper_figures.py \
    --chain1 ../../output_python_plots/nuts_fixed/draws_postburn.npz \
    --chain2 ../../output_python_plots/nuts_fixed/chain2/draws_postburn.npz \
    --data-dir ../../data \
    --paper-dir ../../paper \
    --thin 1 \
    --n-ppc 200
```

This writes all figure PDFs to `paper/` and prints LaTeX table rows for:
- Table 1: posterior ρ summary (mean, SD, 95% CI, ordering probability)
- Table 2: node effects (receiver γⱼ and sender δᵢ)
- Table 3: copula comparison (Frank / Gaussian / Clayton log-densities)

### Step 5 — Compile the paper

```bash
cd ../../paper
pdflatex CGRG_brain.tex && pdflatex CGRG_brain.tex
```

Two passes are needed to resolve cross-references. Output: `CGRG_brain.pdf`.

---

## Draws file format

`draws_postburn.npz` contains an array of shape `(num_samples, 42)` with the
following column layout:

| Index | Parameter | Description |
|-------|-----------|-------------|
| 0 | ρ₀ | Group-mean reciprocity (hyperprior mean) |
| 1 | ρ⁽ᴺᴸ⁾ | Frank copula parameter, cognitively normal |
| 2 | ρ⁽ᴹᶜᴵ⁾ | Frank copula parameter, mild cognitive impairment |
| 3 | ρ⁽ᴬᴰ⁾ | Frank copula parameter, Alzheimer's disease |
| 4–7 | δ₀ | Group-mean sender effects (nodes 1–4) |
| 8–19 | δᵢ⁽ᵍ⁾ | Sender effects (nodes 2–4, groups NL/MCI/AD) |
| 20–23 | γ₀ | Group-mean receiver effects (nodes 1–4) |
| 24–35 | γⱼ⁽ᵍ⁾ | Receiver effects (nodes 1–4, groups NL/MCI/AD) |
| 36 | λ⁽ᴺᴸ⁾ | Marginal precision, NL |
| 37 | λ⁽ᴹᶜᴵ⁾ | Marginal precision, MCI |
| 38 | λ⁽ᴬᴰ⁾ | Marginal precision, AD |
| 39 | σ_ρ | Hierarchical SD for ρ |
| 40 | σ_δ | Hierarchical SD for sender effects |
| 41 | σ_γ | Hierarchical SD for receiver effects |

Load with:

```python
import numpy as np
draws = np.load("draws_postburn.npz")["draws"]
rho_NL  = draws[:, 1]   # posterior samples of ρ for NL group
rho_MCI = draws[:, 2]
rho_AD  = draws[:, 3]
```

---

## Using the `cgrg` package directly

The `code/python/cgrg/` package can be imported standalone:

```python
import sys
sys.path.insert(0, "code/python")
from cgrg.data import load_brain_data
from cgrg.copula import frank_conditional_inv, frank_density_grid

# Load connectivity matrices
data = load_brain_data("data/")
# data["nl"]  — list of (4,4) arrays for NL subjects
# data["mci"] — MCI
# data["ad"]  — AD

# Evaluate Frank copula density on a grid
rho = -3.75
U, V, D = frank_density_grid(rho, n=100)  # D[i,j] = c_rho(U[i,j], V[i,j])
```

---

## Key results

- Posterior reciprocity: ρ ≈ −3.75 (Kendall's τ ≈ −0.43) for all three groups,
  with 95% credible intervals entirely below zero.
- Pr(ρ_NL < ρ_MCI < ρ_AD | data) = 0.20, close to the uniform chance level of
  1/6, indicating disease state does not substantially alter dyadic reciprocity.
- Frank copula outperforms Gaussian (mean log-density 0.158 vs 0.107) and
  90°-rotated Clayton (−0.020) at matched Kendall's τ.
- Within-dyad mutual information ≈ 0.15 nats, stable across disease groups.
