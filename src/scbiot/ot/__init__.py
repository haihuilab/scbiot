from __future__ import annotations

from typing import Any, Optional, Sequence
from .integrate import DEFAULT_MAX_ITER, integrate_ot
from .integrate_centroids import integrate_centroids
from .integrate_paired import integrate_paired

from .supbiot import predict_pseudo_labels, transfer_labels

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
    sharpen: float = 0.5,
    supervision: float = 0.5,
    projector: float = 0.5,
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
    prealign: Optional[str] = None,
    prealign_strength: float = 1.0,
    prealign_eps: float = 1e-2,
    prealign_max_points: int = 20000,
    max_iter: int = DEFAULT_MAX_ITER,
    n_centroids_per_batch: int = 2048,
    max_samples_per_batch: int = 500_000,
    k_interp: int = 8,
    chunk_size: int = 500_000,
    tmp_path: Optional[str] = None,
    align_reference: bool = False,
    run_three_stage: bool = True,
):
    """
    Wrapper around `integrate_ot` with the same knob-based interface.

    If `label_key` is provided (and exists in `adata.obs`), this runs an internal
    three-stage sequence that exactly mimics calling `integrate_ot` externally as:
    supervised → unsupervised → supervised, with projector disabled in the two
    supervised stages. Intermediate embeddings are stored under temporary `obsm`
    keys and removed before returning.
    """

    def _tmp_obsm_key(base: str) -> str:
        candidate = base
        i = 0
        while candidate in adata.obsm:
            i += 1
            candidate = f"{base}_{i}"
        return candidate

    has_labels = label_key is not None and hasattr(adata, "obs") and label_key in adata.obs
    run_three_stage = bool(run_three_stage and (not align_reference) and has_labels)

    if align_reference and has_labels:
        tmp1 = _tmp_obsm_key(f"__scbiot_tmp_sup1_{out_key}")
        try:
            # Stage1: supervised (projector off)
            print("======== Stage1: supervised OT for label propagation ========")
            adata, _ = integrate_ot(
                adata,
                obsm_key=obsm_key,
                batch_key=batch_key,
                out_key=tmp1,
                strength=strength,
                conservation=conservation,
                prototypes=prototypes,
                sharpen=sharpen,
                supervision=supervision,
                projector=projector,
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
                align_reference=False,
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
            )

            # Stage2: query -> reference alignment
            print("\n======== Stage2: Global OT for mapping query to reference ========")
            adata, metrics = integrate_ot(
                adata,
                obsm_key=tmp1,
                batch_key=batch_key,
                out_key=out_key,
                strength=strength,
                conservation=conservation,
                prototypes=prototypes,
                sharpen=sharpen,
                supervision=supervision,
                projector=projector,
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
                align_reference=True,
                prealign="none",
                prealign_strength=prealign_strength,
                prealign_eps=prealign_eps,
                prealign_max_points=prealign_max_points,
                max_iter=max_iter,
                n_centroids_per_batch=n_centroids_per_batch,
                max_samples_per_batch=max_samples_per_batch,
                k_interp=k_interp,
                chunk_size=chunk_size,
                tmp_path=tmp_path,
            )
        finally:
            if hasattr(adata, "obsm") and tmp1 in adata.obsm:
                del adata.obsm[tmp1]

        return adata, metrics

    if not run_three_stage:
        return integrate_ot(
            adata,
            obsm_key=obsm_key,
            batch_key=batch_key,
            out_key=out_key,
            strength=strength,
            conservation=conservation,
            prototypes=prototypes,
            sharpen=sharpen,
            supervision=supervision,
            projector=projector,
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
        )

    tmp1 = _tmp_obsm_key(f"__scbiot_tmp_sup1_{out_key}")
    tmp2 = _tmp_obsm_key(f"__scbiot_tmp_unsup_{out_key}")
    try:
        # Stage1: supervised (projector off)
        print("======== Stage1: supervised OT for label propagation ========")
        adata, _ = integrate_ot(
            adata,
            obsm_key=obsm_key,
            batch_key=batch_key,
            out_key=tmp1,
            strength=strength,
            conservation=conservation,
            prototypes=prototypes,
            sharpen=sharpen,
            supervision=supervision,
            projector=0.,
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
        )       

        # Stage2: unsupervised
        print("\n======== Stage2: unsupervised OT for batch integration ========")
        adata, _ = integrate_ot(
            adata,
            obsm_key=tmp1,
            batch_key=batch_key,
            out_key=tmp2,
            strength=strength,
            conservation=conservation,
            prototypes=prototypes,
            sharpen=sharpen,
            supervision=supervision,
            projector=projector,
            approximate=approximate,
            centroid=centroid,
            reference=reference,
            label_key=None,
            unlabeled_category=unlabeled_category,
            use_gpu=use_gpu,
            gpu_device=gpu_device,
            ot_backend=ot_backend,
            random_state=random_state,
            verbose=verbose,
            modality=modality,
            spatial_key=spatial_key,
            spatial_weight=spatial_weight,
            prealign="none",
            prealign_strength=prealign_strength,
            prealign_eps=prealign_eps,
            prealign_max_points=prealign_max_points,
            max_iter=max_iter,
            n_centroids_per_batch=n_centroids_per_batch,
            max_samples_per_batch=max_samples_per_batch,
            k_interp=k_interp,
            chunk_size=chunk_size,
            tmp_path=tmp_path,
        )

        # Stage3: supervised (projector off) -> final out_key
        print("\n======== Stage3: supervised OT for refinement ========")
        adata, metrics = integrate_ot(
            adata,
            obsm_key=tmp2,
            batch_key=batch_key,
            out_key=out_key,
            strength=strength,
            conservation=conservation,
            prototypes=prototypes,
            sharpen=sharpen,
            supervision=supervision,
            projector=0.0,
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
            prealign="none",
            prealign_strength=prealign_strength,
            prealign_eps=prealign_eps,
            prealign_max_points=prealign_max_points,
            max_iter=max_iter,
            n_centroids_per_batch=n_centroids_per_batch,
            max_samples_per_batch=max_samples_per_batch,
            k_interp=k_interp,
            chunk_size=chunk_size,
            tmp_path=tmp_path,
        )
    finally:
        # Do not leave intermediate embeddings around.
        if hasattr(adata, "obsm"):
            for key in (tmp1, tmp2):
                if key in adata.obsm:
                    del adata.obsm[key]

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
    if inplace and not use_embedding_ref and embedding_key is None and embedding_k is None:
        pred_kwargs = {
            "max_ref": kwargs.get("max_ref", 20000),
            "topk": kwargs.get("topk", 64),
            "use_gpu": kwargs.get("use_gpu", True),
            "gpu_device": kwargs.get("gpu_device", 0),
            "reg": kwargs.get("reg"),
            "reg_m": kwargs.get("reg_m"),
            "cost_clip_q": kwargs.get("cost_clip_q"),
            "ot_backend": kwargs.get("ot_backend", "torch"),
            "ot_iters": kwargs.get("ot_iters", 1000),
            "ot_tol": kwargs.get("ot_tol", 1e-6),
            "chunk_size": kwargs.get("chunk_size", 1024),
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
        use_embedding_ref=use_embedding_ref,
        embedding_key=embedding_key,
        embedding_k=embedding_k,
        embedding_weight_power=embedding_weight_power,
        inplace=inplace,
        out_key=out_key,
        verbose=verbose,
        **kwargs,
    )
