"""
Data loading for the CGRG brain network model.

Expects the data directory to contain:
    AD_list.mat   -- cell array of 4x4 adjacency matrices, 12 subjects
    MCI_list.mat  -- cell array of 4x4 adjacency matrices, 63 subjects
    NL_list.mat   -- cell array of 4x4 adjacency matrices, 37 subjects

Both legacy MATLAB (.mat <= v7) and HDF5-based (.mat v7.3) files are
supported via scipy.io.loadmat and h5py respectively.
"""

import os
from pathlib import Path
from typing import Dict, List

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_mat_legacy(path: str) -> dict:
    """Load a .mat file using scipy.io.loadmat (MATLAB <= 7.2 format)."""
    import scipy.io
    return scipy.io.loadmat(path, squeeze_me=False, struct_as_record=True)


def _load_mat_hdf5(path: str) -> dict:
    """Load a .mat file in HDF5 format (MATLAB v7.3)."""
    import h5py
    out = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            out[key] = f[key]
    return out


def _mat_cell_to_list(mat_obj) -> List[np.ndarray]:
    """
    Convert a MATLAB variable (loaded by scipy.io, struct_as_record=True)
    into a Python list of 2-D numpy arrays.

    The brain data files are stored as MATLAB structs with subject IDs as
    field names (e.g. '0729', '4153', ...).  scipy.io.loadmat with
    struct_as_record=True loads these as numpy void scalars whose dtype
    has named fields.  We iterate over the field names to recover each
    subject's matrix.

    Also handles the legacy object-array case for robustness.
    """
    # Unwrap (1,1) wrapper arrays that scipy sometimes inserts
    while (isinstance(mat_obj, np.ndarray)
           and mat_obj.ndim >= 1
           and mat_obj.dtype != object
           and mat_obj.dtype.names is not None
           and mat_obj.size == 1):
        mat_obj = mat_obj.flat[0]

    # Struct case: numpy void with named fields (subject IDs as field names)
    if isinstance(mat_obj, np.void) or (
            isinstance(mat_obj, np.ndarray) and mat_obj.dtype.names is not None):
        names = mat_obj.dtype.names
        return [np.asarray(mat_obj[n], dtype=float) for n in names]

    # Cell array case: object-dtype ndarray
    if isinstance(mat_obj, np.ndarray) and mat_obj.dtype == object:
        return [np.asarray(mat_obj.flat[i], dtype=float)
                for i in range(mat_obj.size)]

    # Single matrix (edge case: only one subject)
    return [np.asarray(mat_obj, dtype=float)]


def _hdf5_cell_to_list(h5_group) -> List[np.ndarray]:
    """
    Convert an HDF5 cell array (from MATLAB v7.3 .mat) into a list of
    2-D numpy arrays.

    In h5py, MATLAB cell arrays appear as HDF5 groups whose datasets are
    named '#0', '#1', ... or as HDF5 object-reference datasets.
    """
    import h5py

    references = h5_group[()]          # array of HDF5 object references
    root = h5_group.file               # root of the HDF5 file

    matrices = []
    for ref in references.flat:
        arr = root[ref][()]
        # MATLAB stores arrays column-major; transpose to row-major convention
        matrices.append(np.asarray(arr, dtype=float).T)
    return matrices


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_brain_data(data_dir: str) -> Dict[str, List[np.ndarray]]:
    """
    Load fMRI brain connectivity matrices for the three diagnostic groups.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing AD_list.mat, MCI_list.mat, NL_list.mat.

    Returns
    -------
    dict with keys 'ad', 'mci', 'nl', each mapping to a list of 4x4
    numpy float64 arrays (one per subject).

    Raises
    ------
    FileNotFoundError
        If any of the three .mat files is missing.
    RuntimeError
        If the cell variable cannot be extracted from a file.
    """
    data_dir = Path(data_dir)
    file_map = {
        "ad":  data_dir / "AD_list.mat",
        "mci": data_dir / "MCI_list.mat",
        "nl":  data_dir / "NL_list.mat",
    }

    result: Dict[str, List[np.ndarray]] = {}

    for group, fpath in file_map.items():
        if not fpath.exists():
            raise FileNotFoundError(f"Expected data file not found: {fpath}")

        matrices = _load_single_file(str(fpath), group)
        result[group] = matrices
        print(f"  Loaded {group.upper():3s}: {len(matrices)} networks "
              f"(shape {matrices[0].shape})")

    return result


def _load_single_file(path: str, group: str) -> List[np.ndarray]:
    """
    Attempt to load a .mat file, trying legacy format first and falling
    back to HDF5 (MATLAB v7.3) if that fails.
    """
    # Try legacy scipy.io loader first
    try:
        mat = _load_mat_legacy(path)
        # The cell variable name may be e.g. 'AD_list', 'MCI_list', 'NL_list'
        # or simply the group label.  Search for the first non-meta key.
        candidate_keys = [k for k in mat.keys() if not k.startswith("__")]
        if not candidate_keys:
            raise RuntimeError(f"No data variables found in {path}")
        # Prefer a key that contains the group name (case-insensitive)
        preferred = [k for k in candidate_keys
                     if group.lower() in k.lower()]
        key = preferred[0] if preferred else candidate_keys[0]
        return _mat_cell_to_list(mat[key])

    except Exception as legacy_err:
        # Fall back to HDF5 loader (MATLAB v7.3)
        try:
            import h5py
            with h5py.File(path, "r") as f:
                candidate_keys = [k for k in f.keys()
                                  if not k.startswith("#")]
                preferred = [k for k in candidate_keys
                             if group.lower() in k.lower()]
                key = preferred[0] if preferred else candidate_keys[0]
                return _hdf5_cell_to_list(f[key])
        except Exception as hdf5_err:
            raise RuntimeError(
                f"Failed to load {path}.\n"
                f"  Legacy error: {legacy_err}\n"
                f"  HDF5 error:   {hdf5_err}"
            ) from hdf5_err
