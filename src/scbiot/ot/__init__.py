from __future__ import annotations

from typing import Any, Dict

from .coembedding import (
    assemble_joint_embedding,
    build_aligned_coembedding,
    harmonize_gene_names,
    label_transfer_shared_pca,
    prepare_transfer_layers,
    transfer_labels,
)
from .integrate import integrate_ot
from .integrate_centroids import integrate_centroids
from .integrate_paired import integrate_paired
from ._presets import Preset, get_modality_preset as _get_modality_preset

__all__ = [
    "integrate",
    "integrate_ot",
    "integrate_spatial",
    "integrate_spatial_ot",
    "integrate_centroids",    
    "integrate_paired",
    "build_aligned_coembedding",
    "label_transfer_shared_pca",
    "prepare_transfer_layers",
    "transfer_labels",
    "harmonize_gene_names",
    "assemble_joint_embedding",
]
def integrate(adata: Any, modality: str = "rna", **overrides: Any):
    """
    Run scBIOT OT integration with a modality-specific preset.

    Parameters
    ----------
    adata
        Annotated data matrix.
    modality
        One of ``"rna"``, ``"supervised"``, ``"atac"``, or ``"paired"`` (case-insensitive).
        Determines which preset of hyper-parameters to start from.
    overrides
        Any keyword arguments to override the preset for fine-grained control.

    Returns
    -------
    The return payload of :func:`integrate_ot` or :func:`integrate_paired`, depending
    on the selected modality (typically ``adata`` and a diagnostics dictionary).
    """
    params = {**_get_modality_preset(modality), **overrides}
    mode = modality.lower()
    if mode == 'paired':
        return integrate_paired(adata, **params)
    if mode == 'spatial':
        return integrate_spatial_ot(adata, **params)
    return integrate_ot(adata, **params)
