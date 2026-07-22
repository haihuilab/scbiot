"""
Public preprocessing namespace.
"""

from __future__ import annotations

from .peaks import (
    remove_promoter_proximal_peaks,
    find_variable_features,
    add_iterative_lsi,
    annotate_gene_activity,
    harmonize_gene_names,
    knn_smooth_ga_on_atac,
    ensure_csr_f32,
    create_gene_activity,
)
from .autoencoder import autoencoder, autoencoder_map, LinearAutoencoderModel
from ..tl.integration_transport import integration_transport_score, per_batch_transport

__all__ = [
    "integration_transport_score",
    "per_batch_transport",
    "remove_promoter_proximal_peaks",
    "find_variable_features",
    "add_iterative_lsi",
    "create_gene_activity",
    "annotate_gene_activity",
    "harmonize_gene_names",
    "knn_smooth_ga_on_atac",
    "ensure_csr_f32",
    "autoencoder",
    "autoencoder_map",
    "LinearAutoencoderModel",
    "build_geo_ot",
    "refine_domains",
    "SPATIAL_GRID_KEY",
    "SPATIAL_KNN_KEY",
]
