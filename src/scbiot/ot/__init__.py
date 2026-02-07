from __future__ import annotations

from inspect import signature
import warnings
from typing import Any, Optional, Literal, Sequence
from .integrate import integrate_ot
from .integrate_centroids import integrate_centroids
from .integrate_paired import integrate_paired

from .supbiot import transfer_labels
from ._presets import get_modality_preset as _get_modality_preset

__all__ = [
    "integrate",
    "integrate_centroids",
    "integrate_paired",
    "supbiot",
]

_UNSET = object()  # Sentinel to detect which args were explicitly provided.
_PAIRED_ARG_KEYS = set(signature(integrate_paired).parameters) - {"adata"}

def integrate(
    adata: Any,
    obsm_key: str = _UNSET,
    batch_key: str = _UNSET,
    out_key: str = _UNSET,
    preset: Literal["rna", "atac", "supervised", "anchor", "centroid", "spatial", "paired"] = _UNSET,
    K_ref: int = _UNSET,
    K_batch: int = _UNSET,
    reg: float = _UNSET,
    reg_m: float = _UNSET,
    sharpen: float = _UNSET,
    K_pseudo: int = _UNSET,
    pull: float = _UNSET,
    push: float = _UNSET,
    lambda0_hi: float = _UNSET,
    lambda0_lo: float = _UNSET,
    smin_bulk: float = _UNSET,
    smax_bulk: float = _UNSET,
    smin_bridge: float = _UNSET,
    smax_bridge: float = _UNSET,
    max_step_local: float = _UNSET,
    step_lo: float = _UNSET,
    step_hi: float = _UNSET,
    q_start: float = _UNSET,
    q_end: float = _UNSET,
    overlap0_lo: float = _UNSET,
    overlap0_hi: float = _UNSET,
    w_overlap: float = _UNSET,
    penalty_gamma: float = _UNSET,
    w_strain: float = _UNSET,
    k_local: int = _UNSET,
    k_eval: int = _UNSET,
    eval_subsample: Optional[int] = _UNSET,
    trust_subsample: Optional[int] = _UNSET,
    max_iter: int = _UNSET,
    patience: int = _UNSET,
    tol: float = _UNSET,
    reference: str = _UNSET,  # supports "largest" or "union"
    reference_category: Optional[str] = _UNSET,
    reference_align: bool = _UNSET,
    label_key: Optional[str] = _UNSET,
    unlabeled_category: Any = _UNSET,
    postscale: bool = _UNSET,
    random_state: int = _UNSET,
    verbose: bool = _UNSET,
    use_gpu: bool = _UNSET,
    gpu_device: int = _UNSET,
    ot_backend: str = _UNSET,
    ot_mode: str = _UNSET,
    approximate_ot: bool = False,
    centroid_ot: bool = False,
    spatial_key: Optional[str] = _UNSET,
    spatial_weight: float = _UNSET,
    # ---- NEW: CORAL pre-alignment for disjoint datasets ----
    prealign: Optional[str] = _UNSET,  # None | "coral"
    coral_strength: float = _UNSET,  # 1.0 full CORAL, 0.3~0.7 partial
    coral_eps: float = _UNSET,
    coral_max_points: int = _UNSET,
    coral_target: str = _UNSET,  # "auto"|"reference"|"global"
    # ---- NEW supervised knobs ----
    lam_sup: float = _UNSET,  # pull to own-class mean (0 disables)
    lam_repulse: float = _UNSET,
    **overrides: Any,
):
    """
    **scBIOT**: optimal transport–based data integration for single-modality and cross-modality embeddings.

    Parameters
    ----------
    adata
        AnnData object containing a low-dimensional representation to align.
    obsm_key
        Key in ``adata.obsm`` with the starting embedding (for example, PCA).
    batch_key
        ``adata.obs`` column containing batch identities.
    out_key
        Destination key in ``adata.obsm`` for the corrected coordinates.
    preset
        Default settings used for integration of scRNA-seq ("rna"), supervised ("supervised"),
        snATAC-seq ("atac"), spatial ("spatial"), multiomics for unpaired workflows ("anchor"),
        matched RNA/ATAC multiomics ("paired"), or centroid OT ("centroid"). For
        ``preset=\"paired\"``, provide ``view_key`` plus optional fusion weights
        (``w_base``, ``w_view``) and pairing prior controls (``prior_strength``, ``diag_mass``).
    K_ref / K_batch
        Reference and batch-specific prototype sizes used for OT coupling.
    reg / reg_m
        Entropic and unbalanced mass-penalty terms for the OT solver.
    sharpen / K_pseudo / pull / push
        Connectivity controls that steer the pseudo-label refinement.
    lambda0_hi / lambda0_lo / smin_* / smax_* / max_step_local / step_*
        Step-size and neighborhood scaling parameters for the iterative updates.
    q_start / q_end / overlap0_lo / overlap0_hi / w_overlap / penalty_gamma
        Hyper-parameters that encourage overlap while penalizing over-correction.
    w_strain
        Weight on the graph-strain regularizer.
    k_local / k_eval
        Neighborhood sizes used for local graph construction and evaluation.
    eval_subsample / trust_subsample
        Optional subsampling for efficiency during scoring.
    max_iter / patience / tol
        Early stopping controls for the outer optimization loop.
    reference
        Reference batch selection strategy (``"largest"`` or ``"union"``). When running
        in supervised mode with ``label_key`` provided, the method forces ``"union"``
        for stability.
    reference_category
        Alias for ``reference`` when you want to pass a batch label string (for example,
        ``"reference"``). If provided, it overrides ``reference``.
    reference_align
        When True, align query cells directly to the reference group and keep the
        reference fixed (reference/query alignment).
    label_key / unlabeled_category
        Optional semi-supervised label column and unlabeled marker used for supervised
        guidance and to compute OT transport metadata for label transfer. If omitted,
        label-transfer metadata is skipped.
    postscale
        Whether to rescale the aligned embedding to unit variance per dimension.
    random_state
        Seed for stochastic steps used in subsampling and initialization.
    verbose
        Print progress information when ``True``.
    use_gpu / gpu_device / ot_backend / ot_mode
        Compute backend selection for OT (torch by default, pot optional) and whether
        to run unbalanced OT (default) or balanced OT for stronger batch mixing. When
        ``ot_mode='balanced'``, the algorithm automatically switches to the ``'union'``
        reference so every batch moves symmetrically.
    approximate_ot
        When True, use the approximate OT solver while keeping the selected preset's
        hyper-parameters and data keys.
    centroid_ot
        When True, run centroid-level OT with FAISS interpolation (scales to very
        large datasets).
    spatial_key
        Optional key in ``adata.obsm`` containing spatial coordinates. When provided,
        those coordinates are concatenated to ``obsm_key`` before OT.
    spatial_weight
        Multiplier applied to the standardized spatial coordinates prior to
        concatenation so that spatial distances and expression distances have
        comparable scale.
    lam_sup / lam_repulse
        Attraction/repulsion strengths when ``label_key`` is provided.

    Returns
    -------
    adata
        The input object with integrated coordinates stored in ``adata.obsm[out_key]``.
    dict
        Diagnostics with mixing metrics (``mix``, ``overlap0``, ``strain``, ``tw``)
        and the iteration that achieved the best score (``it``).

    Notes
    -----
    The function updates ``adata`` in place and also returns it for convenience.
    """
    # Collect user-specified args, then layer: preset < args < overrides
    _args = dict(locals())
    _args.pop("adata")
    overrides = _args.pop("overrides")
    approximate_ot = bool(_args.pop("approximate_ot"))
    centroid_ot = bool(_args.pop("centroid_ot"))

    reference_category = _args.pop("reference_category")
    reference_align = _args.pop("reference_align")
    reference_category_set = reference_category is not _UNSET
    if reference_category_set:
        if _args.get("reference", _UNSET) is not _UNSET:
            raise TypeError(
                "integrate() got multiple values for 'reference' (reference_category also provided)"
            )
        _args["reference"] = reference_category

    preset_value = _args.pop("preset")
    if preset_value is _UNSET:
        if "modality" in overrides:
            preset_value = overrides.pop("modality")
            warnings.warn(
                "`modality` is deprecated; use `preset` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        else:
            preset_value = "rna"
    elif "modality" in overrides:
        raise TypeError("integrate() got multiple values for 'preset' (old name 'modality' also provided)")

    mode = str(preset_value).lower()
    if centroid_ot and approximate_ot:
        raise ValueError(
            "integrate() received both approximate_ot and centroid_ot; enable only one."
        )
    effective_centroid = (centroid_ot or (mode == "centroid")) and mode != "paired"
    _args = {key: value for key, value in _args.items() if value is not _UNSET}
    params = {**_get_modality_preset(mode), **_args, **overrides}
    if approximate_ot:
        params["approximate_ot"] = True
    if centroid_ot and mode == "paired":
        params["centroid_ot"] = True
    if reference_align is not _UNSET:
        params["reference_align"] = bool(reference_align)
    elif reference_category_set and mode == "anchor":
        params["reference_align"] = True

    if params.get("reference_align") and mode == "anchor":
        if "ot_mode" not in _args and "ot_mode" not in overrides:
            params["ot_mode"] = "unbalanced"
        if "postscale" not in _args and "postscale" not in overrides:
            params["postscale"] = False

    if effective_centroid:
        centroid_defaults = _get_modality_preset("centroid")
        centroid_keys = (
            "n_centroids_per_batch",
            "max_samples_per_batch",
            "k_interp",
            "chunk_size",
            "use_gpu",
            "gpu_device",
            "tmp_path",
        )
        for key in centroid_keys:
            if key not in _args and key not in overrides and key in centroid_defaults:
                params[key] = centroid_defaults[key]
    if effective_centroid:
        params.pop("mode", None)
        centroid_mode = None if mode == "centroid" else mode
        return integrate_centroids(adata, modality=centroid_mode, **params)
    if mode == "paired":
        paired_kwargs = {k: params[k] for k in _PAIRED_ARG_KEYS if k in params}
        return integrate_paired(adata, **paired_kwargs)
    params.pop("mode", None)
    return integrate_ot(adata, modality=mode, **params)


def supbiot(
    adata: Any,
    *,
    use_rep: Optional[str] = None,
    label_key: str,
    unlabeled_category: str | Sequence[str] | set[str] = "unknown",
    pred_label_key: str = "pred_cell_type",
    pred_conf_key: str = "pred_confidence",
    min_conf: float = 0.25,
    use_embedding_ref: bool = False,
    embedding_key: Optional[str] = None,
    embedding_k: Optional[int] = None,
    embedding_weight_power: float = 2.0,
    inplace: bool = True,
    batch_key: str = "batch",
    out_key: Optional[str] = None,
    verbose: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Label transfer wrapper around ``transfer_labels`` using OT metadata.

    Parameters
    ----------
    adata
        AnnData object with reference and query cells plus OT metadata.
    use_rep
        Representation in ``adata.obsm`` used for label transfer (alias for ``out_key``).
    label_key
        Column in ``adata.obs`` with labels for reference cells.
    unlabeled_category
        Label value(s) that identify query cells.
    pred_label_key
        Output column in ``adata.obs`` for predicted labels for only query cells within ``unlabeled_category``.
    pred_conf_key
        Output column in ``adata.obs`` for prediction confidence.
    min_conf
        Minimum confidence threshold for assigned labels.
    use_embedding_ref
        If True, project query embeddings into the reference embedding space.
    embedding_key
        ``adata.obsm`` key for embedding projection when ``use_embedding_ref=True``.
    embedding_k
        Number of neighbors for embedding projection.
    embedding_weight_power
        Power for distance weights in embedding projection.
    inplace
        If True, write results back to ``adata`` and return it.
    batch_key
        Batch column name for the joint embedding output.
    out_key
        Override the aligned embedding key from OT metadata.
    verbose
        Print progress messages.
    **kwargs
        Extra args forwarded to ``transfer_labels``.

    Returns
    -------
    Any
        Updated ``adata`` (in place) or a joint AnnData when ``inplace=False``.
    """
    if use_rep is not None:
        if out_key is not None and use_rep != out_key:
            raise ValueError("supbiot received both use_rep and out_key with different values.")
        out_key = use_rep
    return transfer_labels(
        adata,
        label_key=label_key,
        unlabeled_category=unlabeled_category,
        pred_label_key=pred_label_key,
        pred_conf_key=pred_conf_key,
        min_conf=min_conf,
        use_embedding_ref=use_embedding_ref,
        embedding_key=embedding_key,
        embedding_k=embedding_k,
        embedding_weight_power=embedding_weight_power,
        inplace=inplace,
        batch_key=batch_key,
        out_key=out_key,
        verbose=verbose,
        **kwargs,
    )
