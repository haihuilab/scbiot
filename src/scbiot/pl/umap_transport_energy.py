"""
UMAP visualization for per-cell transport energy.
"""

from __future__ import annotations

import scanpy as sc


def umap_transport_energy(
    adata,
    key: str = "transport_energy",
    **kwargs,
):
    """
    Plot per-cell transport energy on UMAP.

    Requires `adata.obs[key]`.

    Example:
        adata.obs["transport_energy"]
    """

    if key not in adata.obs:
        raise KeyError(f"{key} not found in adata.obs")

    sc.pl.umap(
        adata,
        color=key,
        **kwargs,
    )
