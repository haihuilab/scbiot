"""
Public preprocessing namespace.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .setup_anndata import (
    AnnDataSetupError,
    ensure_anndata_setup,
    get_anndata_setup,
    setup_anndata as _setup_anndata,
)

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
from .coembedding import coembed_pca

if TYPE_CHECKING:
    from anndata import AnnData

__all__ = [
    "AnnDataSetupError",
    "remove_promoter_proximal_peaks",
    "find_variable_features",
    "add_iterative_lsi",
    "create_gene_activity",
    "annotate_gene_activity",
    "harmonize_gene_names",
    "knn_smooth_ga_on_atac",
    "ensure_csr_f32",
    "coembed_pca",
    "build_geo_ot",
    "refine_domains",
    "SPATIAL_GRID_KEY",
    "SPATIAL_KNN_KEY",
    "ensure_anndata_setup",
    "get_anndata_setup",
    "setup_anndata",
]


def setup_anndata(
    adata: AnnData,
    *,
    var_key: str = "scBIOT_OT",
    batch_key: str = "batch",
    pseudo_key: str | None = "leiden_scBIOT_OT",
    true_key: str | None = None,
    overwrite: bool = False,
) -> dict[str, str | None]:
    return _setup_anndata(
        adata,
        var_key=var_key,
        batch_key=batch_key,
        pseudo_key=pseudo_key,
        true_key=true_key,
        overwrite=overwrite,
    )


setup_anndata.__doc__ = _setup_anndata.__doc__
