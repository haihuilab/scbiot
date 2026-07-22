"""
Transport driver gene heatmap along pseudotime.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse


def _as_array(X) -> np.ndarray:
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


def transport_driver_heatmap(
    adata,
    *,
    pseudotime_key: str,
    score_key: str = "transport_score_score",
    n_top: int = 30,
    layer: Optional[str] = None,
    smooth: int = 50,
):
    """
    Plot heatmap of top transport driver genes along pseudotime.

    Parameters
    ----------
    adata
        AnnData object.
    pseudotime_key
        Column in adata.obs with pseudotime.
    score_key
        Gene score column in adata.var.
    n_top
        Number of genes to show.
    layer
        Expression layer to use (e.g. "transport_fwd").
    smooth
        Rolling smoothing window.
    """

    if pseudotime_key not in adata.obs:
        raise KeyError(f"{pseudotime_key} not found in adata.obs")

    if score_key not in adata.var:
        raise KeyError(f"{score_key} not found in adata.var")

    if layer is not None and layer not in adata.layers:
        raise KeyError(f"{layer} not found in adata.layers")

    genes = (
        adata.var.sort_values(score_key, ascending=False)
        .index[:n_top]
        .tolist()
    )

    order = np.argsort(adata.obs[pseudotime_key].values)

    if layer is None:
        X = _as_array(adata[:, genes].X)
    else:
        X = _as_array(adata[:, genes].layers[layer])

    X = X[order]

    if smooth and smooth > 1:
        window = np.ones(int(smooth), dtype=np.float64) / float(smooth)
        for g in range(X.shape[1]):
            X[:, g] = np.convolve(X[:, g], window, mode="same")

    plt.figure(figsize=(6, 8))

    plt.imshow(
        X.T,
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )

    plt.yticks(range(len(genes)), genes)
    plt.xlabel("Cells ordered by pseudotime")
    plt.ylabel("Driver genes")
    plt.title("Transport driver gene heatmap")

    plt.colorbar(label="Expression")

    plt.tight_layout()
    plt.show()
