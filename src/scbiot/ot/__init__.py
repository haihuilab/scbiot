from __future__ import annotations

from typing import Any, Dict

from .coembedding import (
    assemble_joint_embedding,
    build_aligned_coembedding,
    harmonize_gene_names,
    label_transfer_shared_pca,
)
from .integrate import integrate_ot
from .integrate_paired import integrate_paired

__all__ = [
    "integrate",
    "integrate_ot",
    "integrate_paired",
    "build_aligned_coembedding",
    "label_transfer_shared_pca",
    "harmonize_gene_names",
    "assemble_joint_embedding",
]


Preset = Dict[str, Any]


_MODALITY_PRESETS: Dict[str, Preset] = {
    "rna": {
        "obsm_key": "X_pca",
        "batch_key": "batch",
        "out_key": "scBIOT_OT",
        "reference": "union",
        # OT hyper-parameters
        "K_ref": 1024,
        "K_batch": 448,
        "reg": 0.03,
        "reg_m": 0.40,
        # Connectivity (relaxed)
        "sharpen": 0.15,
        "K_pseudo": 24,
        "pull": 0.75,
        "push": 0.30,
        "lambda0_hi": 0.50,
        "lambda0_lo": 0.35,
        "smin_bulk": 0.75,
        "smax_bulk": 1.65,
        "smin_bridge": 0.85,
        "smax_bridge": 1.25,
        "max_step_local": 1.0,
        "step_lo": 0.75,
        "step_hi": 0.95,
        "q_start": 0.80,
        "q_end": 0.90,
        "overlap0_lo": 0.60,
        "overlap0_hi": 0.70,
        "w_overlap": 0.20,
        "w_strain": 1.0,
        "penalty_gamma": 1.5,
        "use_gpu": True,
        "ot_backend": "torch",
        "verbose": True,
    },
    "supervised": {
        "obsm_key": "X_pca",
        "batch_key": "batch",
        "out_key": "scBIOT_OT",
        "reference": "union",
        # OT hyper-parameters
        "K_ref": 1024,
        "K_batch": 448,
        "reg": 0.03,
        "reg_m": 0.40,
        # Connectivity (relaxed)
        "sharpen": 0.15,
        "K_pseudo": 24,
        "pull": 0.75,
        "push": 0.30,
        "lambda0_hi": 0.50,
        "lambda0_lo": 0.35,
        "smin_bulk": 0.75,
        "smax_bulk": 1.65,
        "smin_bridge": 0.85,
        "smax_bridge": 1.25,
        "max_step_local": 1.0,
        "step_lo": 0.75,
        "step_hi": 0.95,
        "q_start": 0.80,
        "q_end": 0.90,
        "overlap0_lo": 0.60,
        "overlap0_hi": 0.70,
        "w_overlap": 0.20,
        "w_strain": 1.0,
        "penalty_gamma": 1.5,
        # supervised
        "lam_sup": 0.60,
        "lam_repulse": 0.18,
        "use_gpu": True,
        "ot_backend": "torch",
        "verbose": True,
    },        
    "atac": {
    "obsm_key": "X_lsi",
    "batch_key": "batchname_all",
    "out_key": "scBIOT_OT",
    "reference": "largest",
    "K_ref": 1024,
    "K_batch": 512,
    "reg": 0.036,
    "reg_m": 0.30,
    "sharpen": 0.10,
    "K_pseudo": 15,
    "pull": 0.70,
    "push": 0.20,
    "lambda0_hi": 0.58,
    "lambda0_lo": 0.42,
    "smin_bulk": 0.78,
    "smax_bulk": 1.50,
    "smin_bridge": 0.92,
    "smax_bridge": 1.08,
    "max_step_local": 0.80,
    "step_lo": 0.70,
    "step_hi": 0.82,
    "q_start": 0.78,
    "q_end": 0.885,
    "overlap0_lo": 0.64,
    "overlap0_hi": 0.73,
    "w_overlap": 0.35,
    "w_strain": 1.0,
    "penalty_gamma": 1.50,
    "use_gpu": True,
    "ot_backend": "torch",
    "verbose": True
},
"paired": {
    "obsm_key": "X_pca",              # base view for geometry/smoothing
    "batch_key": "batch",
    "out_key": "scBIOT_OT",
    "mode": "ufgw_barycenter",        # NEW: Unbalanced FGW Barycenter
    "view_keys": ("X_pca", "X_lsi"),  # two-view fusion (RNA PCs + ATAC LSI)
    "reference": "union",             # "largest" or "union"
    # OT hyper-parameters
    "K_ref": 960,
    "K_batch": 360,
    "reg": 0.034,
    "reg_m": 0.28,
    # Connectivity (relaxed)
    "sharpen": 0.10,
    "K_pseudo": 20,
    "pull": 0.72,
    "push": 0.24,
    "lambda0_hi": 0.56,
    "lambda0_lo": 0.44,
    "smin_bulk": 0.80,
    "smax_bulk": 1.55,
    "smin_bridge": 0.92,
    "smax_bridge": 1.14,
    "max_step_local": 0.92,
    "step_lo": 0.70,
    "step_hi": 0.88,
    "q_start": 0.78,
    "q_end": 0.88,
    "overlap0_lo": 0.64,
    "overlap0_hi": 0.74,
    "w_overlap": 0.14,
    "w_strain": 1.1,
    "penalty_gamma": 1.6,
    "use_gpu": True,
    "ot_backend": "torch",
    "verbose": True,
}
 

}


def _get_modality_preset(modality: str) -> Preset:
    """Return the preset parameters for a supported modality."""
    key = modality.lower()
    try:
        return _MODALITY_PRESETS[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(_MODALITY_PRESETS))
        raise ValueError(f"Unknown modality '{modality}'. Available presets: {allowed}") from exc


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
    if modality == 'paired':
        return integrate_paired(adata, **params)
    else:
         return integrate_ot(adata, **params)
