"""
Public preprocessing namespace.
"""

from .setup_anndata import (
    AnnDataSetupError,
    ensure_anndata_setup,
    get_anndata_setup,
    setup_anndata,
)

from .peaks import (
    remove_promoter_proximal_peaks,
    find_variable_features,
    add_iterative_lsi,
    annotate_gene_activity,
    harmonize_gene_names,
    knn_smooth_ga_on_atac,
    ensure_csr_f32,
)
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:  # only for type checkers to avoid import cycles at runtime
    from ..ot.coembedding import (
        AtacPreprocessConfig,
        AtacPreprocessResult,
        assemble_joint_embedding,
        preprocess_atac,
    )

__all__ = [
    "remove_promoter_proximal_peaks",
    "find_variable_features",
    "add_iterative_lsi",
    "annotate_gene_activity",
    "harmonize_gene_names",
    "knn_smooth_ga_on_atac",
    "ensure_csr_f32",
    "setup_anndata",
    "preprocess_atac",
    "AtacPreprocessConfig",
    "AtacPreprocessResult",
    "assemble_joint_embedding",
]


def __getattr__(name: str):
    lazy_attrs: Dict[str, str] = {
        "preprocess_atac": "preprocess_atac",
        "AtacPreprocessConfig": "AtacPreprocessConfig",
        "AtacPreprocessResult": "AtacPreprocessResult",
        "assemble_joint_embedding": "assemble_joint_embedding",
    }
    if name in lazy_attrs:
        # Local import to avoid circular initialization issues.
        from ..ot.coembedding import (
            AtacPreprocessConfig,
            AtacPreprocessResult,
            assemble_joint_embedding,
            preprocess_atac,
        )

        globals().update(
            {
                "preprocess_atac": preprocess_atac,
                "AtacPreprocessConfig": AtacPreprocessConfig,
                "AtacPreprocessResult": AtacPreprocessResult,
                "assemble_joint_embedding": assemble_joint_embedding,
            }
        )
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
