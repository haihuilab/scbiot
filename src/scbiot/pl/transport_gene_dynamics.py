"""
Pseudotime gene dynamics using transported expression.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def transport_gene_dynamics(
    adata,
    gene: str,
    pseudotime_key: str,
    *,
    layer: str = "transport_fwd",
):
    """
    Plot gene dynamics along pseudotime using transport predictions.
    """

    if pseudotime_key not in adata.obs:
        raise KeyError(f"{pseudotime_key} not found in adata.obs")

    gene_idx = list(adata.var_names).index(gene)

    x = adata.obs[pseudotime_key].values

    y_true = adata.X[:, gene_idx]
    y_transport = adata.layers[layer][:, gene_idx]

    order = np.argsort(x)

    x = x[order]
    y_true = y_true[order]
    y_transport = y_transport[order]

    plt.plot(x, y_true, label="Observed", alpha=0.6)
    plt.plot(x, y_transport, label="Transported", linewidth=2)

    plt.xlabel("Pseudotime")
    plt.ylabel(gene)
    plt.title(f"Transport gene dynamics: {gene}")
    plt.legend()
    plt.show()
