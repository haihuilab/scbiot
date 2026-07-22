"""
UMAP visualization helpers for transported gene expression.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import scanpy as sc
from scipy import sparse


def umap_transport(
    adata,
    gene: str | Sequence[str],
    *,
    mode: str = "fwd",
    cmap: str = "viridis",
    vcenter: Optional[float] = None,
    **kwargs,
):
    """
    Visualize transported gene expression on UMAP.

    Parameters
    ----------
    adata
        AnnData with transport layers.
    gene
        Gene name. If a list is provided with `ax`, plots the per-cell mean
        across the selected genes.
    mode
        "fwd", "rev", or "delta".
    cmap
        Colormap.
    vcenter
        Center for diverging maps.
    kwargs
        Passed to `scanpy.pl.umap`.
    """

    if isinstance(gene, str):
        genes = [gene]
    else:
        genes = list(gene)
    if len(genes) == 0:
        raise ValueError("gene must contain at least one gene name")

    if mode == "fwd":
        layer = "transport_fwd"
    elif mode == "rev":
        layer = "transport_rev"
    elif mode == "delta":
        if "transport_fwd" not in adata.layers:
            raise KeyError("transport_fwd layer not found")

        if "transport_delta_fwd" not in adata.layers:
            adata.layers["transport_delta_fwd"] = (
                adata.layers["transport_fwd"] - adata.X
            )

        layer = "transport_delta_fwd"
        if vcenter is None:
            vcenter = 0
    else:
        raise ValueError("mode must be 'fwd', 'rev', or 'delta'")

    use_ax = kwargs.get("ax") is not None
    if use_ax and len(genes) > 1:
        missing = [g for g in genes if g not in adata.var_names]
        if missing:
            raise KeyError(f"Genes not found in adata.var_names: {missing}")
        X = adata[:, genes].layers[layer]
        if sparse.issparse(X):
            score = np.asarray(X.mean(axis=1)).ravel()
        else:
            score = np.asarray(X).mean(axis=1)
        tmp_key = "_scbiot_transport_score"
        restore = tmp_key in adata.obs
        old = adata.obs[tmp_key].copy() if restore else None
        adata.obs[tmp_key] = score
        try:
            return sc.pl.umap(
                adata,
                color=tmp_key,
                cmap=cmap,
                vcenter=vcenter,
                **kwargs,
            )
        finally:
            if restore:
                adata.obs[tmp_key] = old
            else:
                del adata.obs[tmp_key]
    return sc.pl.umap(
        adata,
        color=gene if isinstance(gene, str) else genes,
        layer=layer,
        cmap=cmap,
        vcenter=vcenter,
        **kwargs,
    )
