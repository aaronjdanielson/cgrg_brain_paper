"""
BrainCGRG dataclass: parameter container, initialisation, and
precomputed data arrays for the CGRG brain network model.

Corresponds to the parameter struct used throughout bnet_mcmc.m /
bnet_driver.m.

Parameter layout (42-column draw vector, matching MATLAB exactly):
    [0]       rho0                     scalar
    [1:4]     rho[0..2]                NL, MCI, AD
    [4:8]     delta0[0..3]             node hyper-mean (delta0[0] = 0 fixed)
    [8:20]    delta[0..3, 0..2]        flattened column-major (node x group)
              i.e.  delta[:,0], delta[:,1], delta[:,2]
    [20:24]   gamma0[0..3]             node hyper-mean
    [24:36]   gamma[0..3, 0..2]        flattened column-major
    [36:39]   lambda[0..2]             NL, MCI, AD
    [39]      sigma_rho
    [40]      sigma_delta
    [41]      sigma_gamma
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Triangle index convention (CRITICAL — see docstring)
# ---------------------------------------------------------------------------

def make_triangle_indices(n: int = 4) -> Tuple[np.ndarray, np.ndarray,
                                               np.ndarray, np.ndarray]:
    """
    Return (rows_u, cols_u, rows_l, cols_l) so that element k of the upper
    triangle A[rows_u[k], cols_u[k]] is the reverse edge of element k of the
    lower triangle A[rows_l[k], cols_l[k]].

    Following the specification:
        rows_u, cols_u = np.triu_indices(n, k=1)
        rows_l, cols_l = cols_u.copy(), rows_u.copy()

    This guarantees A[rows_l[k], cols_l[k]] = A[j,i] pairs with
    A[rows_u[k], cols_u[k]] = A[i,j] for each dyad {i,j}, i < j.
    """
    rows_u, cols_u = np.triu_indices(n, k=1)
    rows_l = cols_u.copy()
    cols_l = rows_u.copy()
    return rows_u, cols_u, rows_l, cols_l


# ---------------------------------------------------------------------------
# Group data container
# ---------------------------------------------------------------------------

@dataclass
class GroupData:
    """
    Holds the stacked observed data arrays and precomputed mean vectors for
    one diagnostic group.

    Attributes
    ----------
    bel : np.ndarray, shape (6*S,)
        Stacked lower-triangle entries across all S networks for this group.
    beu : np.ndarray, shape (6*S,)
        Stacked upper-triangle entries (paired with bel).
    bmul : np.ndarray, shape (6*S,)
        Stacked lower-triangle mean vectors (updated when delta/gamma change).
    bmuu : np.ndarray, shape (6*S,)
        Stacked upper-triangle mean vectors.
    n_subjects : int
        Number of networks in this group.
    """
    bel: np.ndarray         # (6*S,)
    beu: np.ndarray         # (6*S,)
    bmul: np.ndarray        # (6*S,) -- recomputed by update_means
    bmuu: np.ndarray        # (6*S,)
    n_subjects: int


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

@dataclass
class BrainCGRG:
    """
    Full parameter state for the CGRG hierarchical model.

    Parameters
    ----------
    rho0 : float
        Hyper-mean for group-level reciprocity parameters.
    rho : np.ndarray, shape (3,)
        Group-level Frank copula parameters [NL, MCI, AD].
    delta0 : np.ndarray, shape (4,)
        Node hyper-mean sender effects; delta0[0] = 0 always (reference node).
    delta : np.ndarray, shape (4, 3)
        Group-level sender effects; delta[0, :] = 0 always.
        Column order: [NL, MCI, AD].
    gamma0 : np.ndarray, shape (4,)
        Node hyper-mean receiver effects.
    gamma : np.ndarray, shape (4, 3)
        Group-level receiver effects; column order [NL, MCI, AD].
    lam : np.ndarray, shape (3,)
        Precision parameters (std = 1/sqrt(lambda)) [NL, MCI, AD].
    sigma_rho : float
    sigma_delta : float
    sigma_gamma : float
    group_data : dict mapping 'nl'|'mci'|'ad' -> GroupData, or None.
        Populated by precompute_stacked_data().
    """
    rho0: float
    rho: np.ndarray          # (3,)
    delta0: np.ndarray       # (4,)
    delta: np.ndarray        # (4, 3)
    gamma0: np.ndarray       # (4,)
    gamma: np.ndarray        # (4, 3)
    lam: np.ndarray          # (3,)  -- 'lambda' is a Python keyword
    sigma_rho: float
    sigma_delta: float
    sigma_gamma: float
    group_data: Optional[Dict[str, GroupData]] = field(default=None)

    # ------------------------------------------------------------------
    # Vector (de)serialisation matching the 42-column MATLAB convention
    # ------------------------------------------------------------------

    def to_vector(self) -> np.ndarray:
        """
        Pack all free parameters into a 42-element vector matching MATLAB
        storage order exactly.

        Layout:
            [0]     rho0
            [1:4]   rho   (NL, MCI, AD)
            [4:8]   delta0
            [8:20]  delta flattened column-major: delta[:,0], delta[:,1], delta[:,2]
            [20:24] gamma0
            [24:36] gamma flattened column-major
            [36:39] lambda  (NL, MCI, AD)
            [39]    sigma_rho
            [40]    sigma_delta
            [41]    sigma_gamma
        """
        return np.concatenate([
            [self.rho0],
            self.rho,                           # 3
            self.delta0,                        # 4
            self.delta.flatten(order='F'),      # 12, column-major
            self.gamma0,                        # 4
            self.gamma.flatten(order='F'),      # 12, column-major
            self.lam,                           # 3
            [self.sigma_rho, self.sigma_delta, self.sigma_gamma],
        ])

    @classmethod
    def from_vector(cls, v: np.ndarray) -> "BrainCGRG":
        """
        Reconstruct a BrainCGRG from a 42-element parameter vector.
        group_data is not restored (call precompute_stacked_data separately).
        """
        assert len(v) == 42, f"Expected 42 parameters, got {len(v)}"
        rho0 = float(v[0])
        rho = v[1:4].copy()
        delta0 = v[4:8].copy()
        delta = v[8:20].reshape(4, 3, order='F')
        gamma0 = v[20:24].copy()
        gamma = v[24:36].reshape(4, 3, order='F')
        lam = v[36:39].copy()
        sigma_rho = float(v[39])
        sigma_delta = float(v[40])
        sigma_gamma = float(v[41])
        return cls(rho0=rho0, rho=rho, delta0=delta0, delta=delta,
                   gamma0=gamma0, gamma=gamma, lam=lam,
                   sigma_rho=sigma_rho, sigma_delta=sigma_delta,
                   sigma_gamma=sigma_gamma)


# ---------------------------------------------------------------------------
# Default initial values (matching MATLAB bnet_driver.m convention)
# ---------------------------------------------------------------------------

def default_params() -> BrainCGRG:
    """
    Return a BrainCGRG initialised to MATLAB starting values:

        rho0 = 1.0
        rho  = [1, 1, 1]
        delta0 = [0, 0, 0, 0]   (all zeros; node 0 is fixed reference)
        delta  = zeros(4, 3)
        gamma0 = [0, 0, 0, 0]
        gamma  = zeros(4, 3)
        lambda = [1, 1, 1]
        sigma_rho = sigma_delta = sigma_gamma = 1.0
    """
    return BrainCGRG(
        rho0=1.0,
        rho=np.ones(3),
        delta0=np.zeros(4),
        delta=np.zeros((4, 3)),
        gamma0=np.zeros(4),
        gamma=np.zeros((4, 3)),
        lam=np.ones(3),
        sigma_rho=1.0,
        sigma_delta=1.0,
        sigma_gamma=1.0,
    )


# ---------------------------------------------------------------------------
# Mean-update (bnet_update)
# ---------------------------------------------------------------------------

def update_means(params: BrainCGRG,
                 rows_u: np.ndarray, cols_u: np.ndarray,
                 rows_l: np.ndarray, cols_l: np.ndarray) -> None:
    """
    Recompute bmul and bmuu in params.group_data for all three groups.

    Corresponds to bnet_update.m.  Called after any proposal that changes
    delta or gamma.

    For group g (0=NL, 1=MCI, 2=AD) and network s:
        mmat[i, j] = delta[i, g] + gamma[j, g]
        mul[s]     = mmat[rows_l, cols_l]   (lower triangle, in dyad order)
        muu[s]     = mmat[rows_u, cols_u]   (upper triangle, in dyad order)

    The stacked mean vectors are:
        bmul = concatenate([mul[0], mul[1], ..., mul[S-1]])
        bmuu = concatenate([muu[0], muu[1], ..., muu[S-1]])

    Parameters
    ----------
    params : BrainCGRG
        Modified in-place: group_data[g].bmul and bmuu are updated.
    rows_u, cols_u, rows_l, cols_l : np.ndarray
        Triangle indices from make_triangle_indices().
    """
    if params.group_data is None:
        raise RuntimeError("group_data not initialised; call "
                           "precompute_stacked_data first.")

    group_order = [("nl", 0), ("mci", 1), ("ad", 2)]

    for gname, gidx in group_order:
        gd = params.group_data[gname]
        S = gd.n_subjects
        d = params.delta[:, gidx]   # (4,)
        g = params.gamma[:, gidx]   # (4,)

        # mmat[i,j] = delta[i,g] + gamma[j,g]
        # Efficient: outer sum
        mmat = d[:, np.newaxis] + g[np.newaxis, :]   # (4,4)

        mul_row = mmat[rows_l, cols_l]   # (6,)
        muu_row = mmat[rows_u, cols_u]   # (6,)

        # Tile across all S subjects
        gd.bmul = np.tile(mul_row, S)   # (6*S,)
        gd.bmuu = np.tile(muu_row, S)   # (6*S,)


# ---------------------------------------------------------------------------
# Data precomputation (bnet_driver stacked arrays)
# ---------------------------------------------------------------------------

def precompute_stacked_data(params: BrainCGRG,
                            data: Dict[str, List],
                            rows_u: np.ndarray, cols_u: np.ndarray,
                            rows_l: np.ndarray, cols_l: np.ndarray) -> None:
    """
    Build stacked bel/beu observed data arrays and initial bmul/bmuu arrays
    for each group.  Stores results in params.group_data.

    Corresponds to the stacking loops in bnet_driver.m / bnet_logpost.m.

    Parameters
    ----------
    params : BrainCGRG
        Modified in-place: params.group_data is populated.
    data : dict
        As returned by load_brain_data(): keys 'ad', 'mci', 'nl', each a
        list of 4x4 numpy arrays.
    rows_u, cols_u, rows_l, cols_l : np.ndarray
        Triangle index arrays from make_triangle_indices().
    """
    group_data: Dict[str, GroupData] = {}
    group_order = [("nl", 0), ("mci", 1), ("ad", 2)]

    for gname, gidx in group_order:
        matrices = data[gname]
        S = len(matrices)

        bel_list, beu_list = [], []
        for mat in matrices:
            bel_list.append(mat[rows_l, cols_l])
            beu_list.append(mat[rows_u, cols_u])

        bel = np.concatenate(bel_list)   # (6*S,)
        beu = np.concatenate(beu_list)   # (6*S,)

        # Initialise mean vectors (will be overwritten by update_means)
        d = params.delta[:, gidx]
        g = params.gamma[:, gidx]
        mmat = d[:, np.newaxis] + g[np.newaxis, :]
        mul_row = mmat[rows_l, cols_l]
        muu_row = mmat[rows_u, cols_u]
        bmul = np.tile(mul_row, S)
        bmuu = np.tile(muu_row, S)

        group_data[gname] = GroupData(
            bel=bel, beu=beu, bmul=bmul, bmuu=bmuu, n_subjects=S
        )

    params.group_data = group_data
