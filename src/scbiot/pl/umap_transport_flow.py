"""
UMAP visualization of transport flow arrows.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc


def umap_transport_flow(
    adata,
    *,
    layer: str = "transport_fwd",
    scale: float = 1.0,
    color: str = "black",
    alpha: float = 0.6,
    step: int = 10,
):
    """
    Plot OT transport vectors on UMAP.

    Parameters
    ----------
    adata
        AnnData object.
    layer
        Transport layer used to infer direction.
    scale
        Arrow scaling.
    color
        Arrow color.
    alpha
        Arrow transparency.
    step
        Downsample factor for arrows.
    """

    if "X_umap" not in adata.obsm:
        raise KeyError("UMAP embedding not found in adata.obsm['X_umap']")

    X = adata.X
    Xt = adata.layers[layer]

    delta = Xt - X

    coords = adata.obsm["X_umap"]

    dx = delta.mean(axis=1)

    coords = coords[::step]
    dx = dx[::step]

    plt.scatter(
        adata.obsm["X_umap"][:, 0],
        adata.obsm["X_umap"][:, 1],
        s=5,
        alpha=0.3,
    )

    plt.quiver(
        coords[:, 0],
        coords[:, 1],
        dx,
        dx,
        angles="xy",
        scale_units="xy",
        scale=scale,
        color=color,
        alpha=alpha,
    )

    plt.title("OT Transport Flow on UMAP")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.show()
