"""
Transport energy utilities.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def compute_transport_energy(
    adata,
    *,
    layer: str = "transport_fwd",
    key_added: str = "transport_energy",
    log1p: bool = False,
):
    """
    Compute per-cell transport energy.

    transport_energy_i = sum_g (X_transport - X_original)^2

    Parameters
    ----------
    adata
        AnnData object
    layer
        Transport layer name
    key_added
        Column name stored in adata.obs
    log1p
        Apply log1p transform

    Returns
    -------
    AnnData
        The input object with ``adata.obs[key_added]`` populated.
    """

    if layer not in adata.layers:
        raise KeyError(f"{layer} not found in adata.layers")

    X = adata.X
    Xt = adata.layers[layer]

    if sparse.issparse(X) or sparse.issparse(Xt):
        delta = Xt - X
        if sparse.issparse(delta):
            energy = np.asarray(delta.power(2).sum(axis=1)).ravel()
        else:
            delta = np.asarray(delta)
            energy = np.sum(delta * delta, axis=1)
    else:
        delta = np.asarray(Xt) - np.asarray(X)
        energy = np.sum(delta * delta, axis=1)

    energy = np.asarray(energy).ravel()

    if log1p:
        energy = np.log1p(energy)

    adata.obs[key_added] = energy
    return adata
