from __future__ import annotations

from typing import Any, Optional, Sequence
from .integrate import DEFAULT_MAX_ITER, integrate_ot
from .integrate_centroids import integrate_centroids
from .integrate_paired import integrate_paired

from .supbiot import (
    LOGREG_DEFAULT_C,
    LOGREG_DEFAULT_CLASS_WEIGHT,
    LOGREG_DEFAULT_MAX_ITER,
    LOGREG_DEFAULT_MULTI_CLASS,
    LOGREG_DEFAULT_PENALTY,
    LOGREG_DEFAULT_PRIOR_ALPHA,
    LOGREG_DEFAULT_SOLVER,
    LOGREG_DEFAULT_STANDARDIZE,
    SUPBIOT_DEFAULT_INPUT_REP_KEY,
    SUPBIOT_DEFAULT_INPUT_REP_WEIGHT,
    SUPBIOT_DEFAULT_PROTOTYPE_WEIGHT,
    _pop_predict_pseudo_label_kwargs,
    predict_pseudo_labels,
    transfer_labels,
)

__all__ = [
    "integrate",
    "integrate_ot",
    "integrate_centroids",
    "integrate_paired",
    "DEFAULT_MAX_ITER",
    "supbiot",
]


def integrate(
    adata: Any,
    obsm_key: str = "X_pca",
    batch_key: str = "batch",
    out_key: str = "scBIOT",
    strength: float = 0.5,
    conservation: float = 0.5,
    prototypes: float = 0.5,
    supervision: float = 0.5,
    approximate: bool = False,
    centroid: bool = False,
    reference: str = "auto",
    label_key: Optional[str] = None,
    unlabeled_category: Any = "unknown",
    use_gpu: bool = True,
    gpu_device: int = 0,
    ot_backend: str = "torch",
    random_state: int = 0,
    verbose: bool = True,
    modality: str = "auto",
    spatial_key: Optional[str] = None,
    spatial_weight: float = 0.5,
    time_key: Optional[str] = None,
    time_weight: float = 0.5,
    time_mode: str = "auto",
    prealign: Optional[str] = 'auto',
    prealign_strength: float = 0.5,
    prealign_eps: float = 1e-2,
    prealign_max_points: int = 20000,
    max_iter: int = DEFAULT_MAX_ITER,
    n_centroids_per_batch: int = 2048,
    max_samples_per_batch: int = 500_000,
    k_interp: int = 8,
    chunk_size: int = 500_000,
    tmp_path: Optional[str] = None,
    align_reference: bool = False,
    transfer_mode: str = "knn",
):
    """
    Wrapper around `integrate_ot` with the knob-based public interface.

    Trajectory-aware prealignment
    ------------------------------
    Gaussian OT prealignment (Bures barycenter matching) aligns each batch's full
    distribution (mean + covariance) to a shared target. On a continuous trajectory
    the batches legitimately span different points along that trajectory (e.g.
    different developmental stages), so collapsing their distributions to a common
    Gaussian destroys the very structure the trajectory encodes. Therefore, when a
    continuous-structure key is provided via ``time_key`` and ``prealign`` is left at
    its default (``'auto'``), the Gaussian prealignment is disabled by default. Pass
    ``prealign`` explicitly to override this behaviour.
    """
    has_labels = label_key is not None and hasattr(adata, "obs") and label_key in adata.obs

    # A continuous-structure key (time_key) means the batches span a trajectory; the
    # Gaussian prealign would collapse that structure, so disable it unless the user
    # set `prealign` explicitly.
    if time_key is not None and prealign == 'auto':
        prealign = None
        if verbose:
            print(f"[integrate] continuous-structure key '{time_key}' provided "
                  f"-> Gaussian prealignment disabled to preserve trajectory "
                  f"(pass prealign= explicitly to override).")

    adata, metrics = integrate_ot(
        adata,
        obsm_key=obsm_key,
        batch_key=batch_key,
        out_key=out_key,
        strength=strength,
        conservation=conservation,
        prototypes=prototypes,
        supervision=supervision,
        approximate=approximate,
        centroid=centroid,
        reference=reference,
        label_key=label_key,
        unlabeled_category=unlabeled_category,
        use_gpu=use_gpu,
        gpu_device=gpu_device,
        ot_backend=ot_backend,
        random_state=random_state,
        verbose=verbose,
        modality=modality,
        spatial_key=spatial_key,
        spatial_weight=spatial_weight,
        time_key=time_key,
        time_weight=time_weight,
        time_mode=time_mode,
        prealign=prealign,
        prealign_strength=prealign_strength,
        prealign_eps=prealign_eps,
        prealign_max_points=prealign_max_points,
        max_iter=max_iter,
        n_centroids_per_batch=n_centroids_per_batch,
        max_samples_per_batch=max_samples_per_batch,
        k_interp=k_interp,
        chunk_size=chunk_size,
        tmp_path=tmp_path,
        align_reference=align_reference,
        transfer_mode=transfer_mode,
        _one_stage_supervised=bool(has_labels and (not align_reference) and (not centroid)),
    )
    if has_labels and (not align_reference) and (not centroid) and hasattr(adata, "obsm"):
        adata.obsm["supBIOT_one_stage"] = adata.obsm[out_key].copy()
    return adata, metrics


def supbiot(
    adata: Any,
    *,
    use_rep: Optional[str] = None,
    label_key: str,
    unlabeled_category: str | Sequence[str] | set[str] = "unknown",
    pred_label_key: str = "pred_cell_type",
    pred_conf_key: str = "pred_confidence",
    min_conf: float = 0.25,
    transfer_mode: str = "logreg",
    use_embedding_ref: bool = False,
    embedding_key: Optional[str] = None,
    embedding_k: Optional[int] = 10,
    embedding_weight_power: float = 2.0,
    inplace: bool = True,
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
    transfer_mode
        Transfer backend. ``"knn"`` runs tissue-aware weighted kNN on the selected embedding.
        ``"logreg"`` runs tissue-routed multinomial logistic regression on the selected embedding.
    use_embedding_ref
        If True, project query embeddings into the reference embedding space.
    embedding_key
        ``adata.obsm`` key for embedding projection when ``use_embedding_ref=True``.
        Projected query embeddings are written back to this key.
    embedding_k
        Number of neighbors for embedding projection.
    embedding_weight_power
        Power for distance weights in embedding projection.
    inplace
        If True, write results back to ``adata`` and return it.
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


    Examples
    --------
    Basic usage:

    >>> import scbiot as scb
    >>> adata, metrics = scb.ot.integrate(
    ...     adata,
    ...     obsm_key="X_pca",
    ...     batch_key="batch",    
    ...     out_key="X_supbiot",    
    ...     label_key="cell_type",
    ...     unlabeled_category="Unknown"    
    )
    >>> adata = scb.ot.supbiot(
    ...     adata,
    ...     label_key="cell_type",
    ...     unlabeled_category="Unknown",
    ...     pred_label_key='pred_cell_type',
    ...     pred_conf_key="pred_confidence",
    ...     min_conf=0.25
    )
    >>> sc.pl.violin(adata_query, keys="pred_confidence", groupby="pred_cell_type", rotation=90)
    """
    if use_rep is not None:
        if out_key is not None and use_rep != out_key:
            raise ValueError("supbiot received both use_rep and out_key with different values.")
        out_key = use_rep
    if "transfer_mode" in kwargs:
        transfer_mode_kw = kwargs.pop("transfer_mode")
        if transfer_mode != "logreg" and transfer_mode_kw != transfer_mode:
            raise ValueError("Conflicting transfer_mode specified in argument and kwargs.")
        transfer_mode = transfer_mode_kw
    if "method" in kwargs:
        method_kw = kwargs.pop("method")
        if transfer_mode != "logreg" and method_kw != transfer_mode:
            raise ValueError("Conflicting method and transfer_mode specified.")
        transfer_mode = method_kw
    pred_overrides = _pop_predict_pseudo_label_kwargs(kwargs)
    if inplace and not use_embedding_ref and embedding_key is None and embedding_k is None:
        pred_kwargs = {
            "max_ref": pred_overrides.get("max_ref", 20000),
            "use_gpu": pred_overrides.get("use_gpu", True),
            "gpu_device": pred_overrides.get("gpu_device", 0),
            "transfer_mode": transfer_mode,
            "knn_k": pred_overrides.get("knn_k"),
            "knn_metric": pred_overrides.get("knn_metric", "cosine"),
            "knn_tissue_key": pred_overrides.get("knn_tissue_key", "tissue"),
            "knn_weight_mode": pred_overrides.get("knn_weight_mode", "exp"),
            "knn_prior_alpha": pred_overrides.get("knn_prior_alpha", 0.0),
            "knn_graph_k": pred_overrides.get("knn_graph_k", 20),
            "knn_diffuse_alpha": pred_overrides.get("knn_diffuse_alpha", 0.2),
            "knn_diffuse_steps": pred_overrides.get("knn_diffuse_steps", 1),
            "knn_diffuse_group_key": pred_overrides.get("knn_diffuse_group_key", "modality"),
            "query_group_key": pred_overrides.get("query_group_key", "modality"),
            "query_group_value": pred_overrides.get("query_group_value", "query"),
            "logreg_tissue_key": pred_overrides.get("logreg_tissue_key", "tissue"),
            "logreg_min_cells": pred_overrides.get("logreg_min_cells", 80),
            "logreg_min_classes": pred_overrides.get("logreg_min_classes", 2),
            "logreg_class_weight": pred_overrides.get("logreg_class_weight", LOGREG_DEFAULT_CLASS_WEIGHT),
            "logreg_C": pred_overrides.get("logreg_C", LOGREG_DEFAULT_C),
            "logreg_solver": pred_overrides.get("logreg_solver", LOGREG_DEFAULT_SOLVER),
            "logreg_penalty": pred_overrides.get("logreg_penalty", LOGREG_DEFAULT_PENALTY),
            "logreg_multi_class": pred_overrides.get("logreg_multi_class", LOGREG_DEFAULT_MULTI_CLASS),
            "logreg_max_iter": pred_overrides.get("logreg_max_iter", LOGREG_DEFAULT_MAX_ITER),
            "logreg_prior_alpha": pred_overrides.get("logreg_prior_alpha", LOGREG_DEFAULT_PRIOR_ALPHA),
            "logreg_standardize": pred_overrides.get("logreg_standardize", LOGREG_DEFAULT_STANDARDIZE),
            "input_rep_key": pred_overrides.get("input_rep_key", SUPBIOT_DEFAULT_INPUT_REP_KEY),
            "input_rep_weight": pred_overrides.get("input_rep_weight", SUPBIOT_DEFAULT_INPUT_REP_WEIGHT),
            "prototype_weight": pred_overrides.get("prototype_weight", SUPBIOT_DEFAULT_PROTOTYPE_WEIGHT),
            "random_state": pred_overrides.get("random_state", 0),
        }
        predict_pseudo_labels(
            adata,
            rep=None,
            rep_key=out_key,
            label_key=label_key,
            unlabeled_category=unlabeled_category,
            pred_label_key=pred_label_key,
            pred_conf_key=pred_conf_key,
            min_conf=min_conf,
            return_numpy=True,
            inplace=True,
            **pred_kwargs,
        )
        return adata
    return transfer_labels(
        adata,
        label_key=label_key,
        unlabeled_category=unlabeled_category,
        pred_label_key=pred_label_key,
        pred_conf_key=pred_conf_key,
        min_conf=min_conf,
        transfer_mode=transfer_mode,
        use_embedding_ref=use_embedding_ref,
        embedding_key=embedding_key,
        embedding_k=embedding_k,
        embedding_weight_power=embedding_weight_power,
        inplace=inplace,
        out_key=out_key,
        verbose=verbose,
        **pred_overrides,
    )
