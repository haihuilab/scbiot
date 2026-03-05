# =================== supbiot.py: label transfer ===================
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Sequence

import difflib
import numpy as np
import pandas as pd


def _resolve_ref_query_masks(
    adata: Any,
    *,
    label_key: str,
    unlabeled_category: str | Sequence[str] | set[str] | None,
) -> Tuple[pd.Series, np.ndarray, np.ndarray]:
    if label_key not in adata.obs:
        raise KeyError(f"{label_key!r} not found in adata.obs")
    labels = adata.obs[label_key]
    if unlabeled_category is None:
        query_mask = labels.isna()
    elif isinstance(unlabeled_category, (list, tuple, set)):
        query_mask = labels.isna() | labels.isin(unlabeled_category)
    else:
        query_mask = labels.isna() | labels.eq(unlabeled_category)
    if int(query_mask.sum()) == 0:
        hint = ""
        available_labels = []
        if hasattr(labels, "cat"):
            available_labels = list(labels.cat.categories)
        else:
            available_labels = list(labels.dropna().unique())
        if isinstance(unlabeled_category, str) and available_labels:
            matches = difflib.get_close_matches(
                unlabeled_category,
                [str(label) for label in available_labels],
                n=1,
                cutoff=0.6,
            )
            if matches:
                hint = f" Did you mean {matches[0]!r}?"
        if available_labels:
            preview = ", ".join(repr(label) for label in available_labels[:10])
            suffix = "..." if len(available_labels) > 10 else ""
            hint = f"{hint} Available labels include: {preview}{suffix}"
        raise ValueError(f"Query subset empty; check unlabeled_category.{hint}")
    ref_mask = ~query_mask
    if int(ref_mask.sum()) == 0:
        raise ValueError("Reference subset empty; check unlabeled_category.")
    return labels, np.asarray(ref_mask), np.asarray(query_mask)


def predict_pseudo_labels(
    adata: Any,
    *,
    rep: np.ndarray | None = None,
    rep_key: str | None = None,
    label_key: str,
    unlabeled_category: str | Sequence[str] | set[str] | None = "unknown",
    pred_label_key: str = "pred_cell_type",
    pred_conf_key: str = "pred_confidence",
    min_conf: float = 0.0,
    return_numpy: bool = True,
    inplace: bool = False,
    max_ref: int = 20000,
    topk: int = 64,
    use_gpu: bool = True,
    gpu_device: int = 0,
    reg: float | None = None,
    reg_m: float | None = None,
    cost_clip_q: float | None = None,
    ot_backend: str = "torch",
    ot_iters: int = 1000,
    ot_tol: float = 1e-6,
    chunk_size: Optional[int] = 1024,
) -> Tuple[np.ndarray | pd.Series, np.ndarray | pd.Series]:
    """
    Predict pseudo labels using OT transport weights or on-the-fly OT alignment.
    """
    from ..utils.ot_transfer import label_transfer_shared_pca
    from ..utils.ot_transport import compute_ot_alignment, ot_label_transfer

    labels, ref_mask, query_mask = _resolve_ref_query_masks(
        adata,
        label_key=label_key,
        unlabeled_category=unlabeled_category,
    )
    n_obs = int(adata.n_obs)
    if rep is None:
        meta = adata.uns.get("_supbiot", {})
        rep_key = rep_key or meta.get("rep_key", "X_pca_shared_aligned")
        if rep_key not in adata.obsm:
            raise KeyError(f"{rep_key!r} not found in adata.obsm")

    ot_meta = adata.uns.get("_ot_alignment") if rep is None else None
    if rep is None and ot_meta is None:
        raise RuntimeError("OT alignment metadata missing; run integrate_ot first.")

    pred_labels_query: np.ndarray
    pred_conf_query: np.ndarray
    if rep is None and ot_meta is not None:
        adata_ref = adata[ref_mask].copy()
        adata_query = adata[query_mask].copy()
        adata_query.uns["_ot_alignment"] = ot_meta
        label_transfer_shared_pca(
            adata_ref,
            adata_query,
            label_key=label_key,
            use_rep=rep_key,
            min_conf=min_conf,
        )
        pred_labels_query = adata_query.obs["pred_cell_type"].to_numpy()
        pred_conf_query = adata_query.obs["pred_confidence"].to_numpy()
    else:
        rep_arr = np.asarray(rep if rep is not None else adata.obsm[rep_key], dtype=np.float32, order="C")
        if rep_arr.shape[0] != n_obs:
            raise ValueError("rep must have the same number of rows as adata.obs")
        ref_emb = rep_arr[ref_mask]
        query_emb = rep_arr[query_mask]
        if ref_emb.size == 0 or query_emb.size == 0:
            raise ValueError("Reference/query subset empty; check label_key/unlabeled_category.")
        labels_ref = labels[ref_mask]

        if max_ref is not None and ref_emb.shape[0] > max_ref:
            rng = np.random.default_rng(0)
            keep = rng.choice(ref_emb.shape[0], size=int(max_ref), replace=False)
            ref_emb = ref_emb[keep]
            labels_ref = labels_ref.iloc[keep]

        reg_eff = float(reg) if reg is not None else float(0.05)
        reg_m_eff = float(reg_m) if reg_m is not None else float(0.5)
        cost_clip_eff = float(cost_clip_q) if cost_clip_q is not None else float(0.90)
        topk_eff = int(min(max(1, topk), ref_emb.shape[0]))
        _, transport = compute_ot_alignment(
            query_emb,
            ref_emb,
            reg=reg_eff,
            reg_m=reg_m_eff,
            cost_clip_q=cost_clip_eff,
            clip_big=50.0,
            backend=ot_backend,
            iters=ot_iters,
            tol=ot_tol,
            use_gpu=use_gpu,
            gpu_device=gpu_device,
            transport_topk=topk_eff,
            chunk_size=chunk_size,
        )
        res = ot_label_transfer(
            transport=transport,
            target_labels=labels_ref,
            min_conf=min_conf,
            confidence_mode="entropy",
        )
        pred_labels_query = res["pred_labels"]
        pred_conf_query = res["confidence"]

    pred_labels = np.full(n_obs, None, dtype=object)
    pred_conf = np.full(n_obs, np.nan, dtype=np.float32)
    pred_labels[query_mask] = pred_labels_query
    pred_conf[query_mask] = pred_conf_query

    if inplace:
        if pred_label_key in adata.obs and isinstance(adata.obs[pred_label_key].dtype, pd.CategoricalDtype):
            base_col = adata.obs[pred_label_key]
            extra = pd.unique(pd.Series(pred_labels_query).dropna())
            categories = list(base_col.cat.categories)
            for value in extra:
                if value not in categories:
                    categories.append(value)
            if categories != list(base_col.cat.categories):
                adata.obs[pred_label_key] = base_col.cat.set_categories(categories)
            adata.obs.loc[adata.obs_names[query_mask], pred_label_key] = pred_labels_query
        else:
            if pred_label_key in adata.obs:
                adata.obs.loc[adata.obs_names[query_mask], pred_label_key] = pred_labels_query
            else:
                adata.obs[pred_label_key] = pred_labels

        if pred_conf_key in adata.obs:
            adata.obs.loc[adata.obs_names[query_mask], pred_conf_key] = pred_conf_query
        else:
            adata.obs[pred_conf_key] = pred_conf

    if return_numpy:
        return pred_labels, pred_conf
    pred_labels_s = pd.Series(pred_labels, index=adata.obs_names, name=pred_label_key)
    pred_conf_s = pd.Series(pred_conf, index=adata.obs_names, name=pred_conf_key)
    return pred_labels_s, pred_conf_s


def transfer_labels(
    adata: Any,
    *,
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
    out_key: Optional[str] = None,
    verbose: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Transfer labels from reference to query using OT alignment computed by integrate_ot.
    When ``use_embedding_ref=True``, projected query embeddings are stored under
    ``embedding_key``.
    """
    from ..utils.ot_transfer import (
        _knn_barycentric_project,
        _resolve_obsm_key,
        assemble_joint_embedding,
    )

    meta = adata.uns.get("_supbiot", {})
    rep_key = out_key or meta.get("rep_key", "X_pca_shared_aligned")

    map_key = embedding_key or "X_umap"
    rep_key_map = rep_key
    if use_embedding_ref:
        if embedding_key is not None:
            map_key = embedding_key
    elif embedding_key is not None and verbose:
        print("[supbiot] embedding_key is ignored because use_embedding_ref=False.")

    _, ref_mask, query_mask = _resolve_ref_query_masks(
        adata,
        label_key=label_key,
        unlabeled_category=unlabeled_category,
    )
    
    adata_ref = adata[ref_mask].copy()
    adata_query = adata[query_mask].copy()

    if rep_key not in adata.obsm:
        raise KeyError(f"{rep_key!r} not found in adata.obsm")

    pred_labels, pred_conf = predict_pseudo_labels(
        adata,
        rep=None,
        rep_key=rep_key,
        label_key=label_key,
        unlabeled_category=unlabeled_category,
        pred_label_key=pred_label_key,
        pred_conf_key=pred_conf_key,
        min_conf=min_conf,
        return_numpy=True,
        inplace=False,
        max_ref=kwargs.get("max_ref", 20000),
        topk=kwargs.get("topk", 64),
        use_gpu=kwargs.get("use_gpu", True),
        gpu_device=kwargs.get("gpu_device", 0),
    )
    adata_query.obs["pred_cell_type"] = pred_labels[query_mask]
    adata_query.obs["pred_confidence"] = pred_conf[query_mask]
    if pred_label_key != "pred_cell_type":
        adata_query.obs[pred_label_key] = adata_query.obs["pred_cell_type"]
    if pred_conf_key != "pred_confidence":
        adata_query.obs[pred_conf_key] = adata_query.obs["pred_confidence"]

    ot_meta = adata.uns.get("_ot_alignment")

    mapped_key: Optional[str] = None
    ref_umap_key: Optional[str] = None
    do_map = use_embedding_ref
    if do_map:
        if ot_meta is None:
            raise RuntimeError("OT alignment metadata missing; run integrate_ot first.")
        topk = int(ot_meta.get("params", {}).get("topk", 64))
        k_default = min(max(topk, 1), 50)
        k_map = k_default if embedding_k is None else int(embedding_k)
        k_map = max(k_map, 1)

        rep_ref = adata_ref.obsm[rep_key_map]
        rep_query = adata_query.obsm[rep_key_map]

        ref_umap_key = _resolve_obsm_key(adata_ref, map_key)        
        ref_umap = np.asarray(adata_ref.obsm[ref_umap_key], dtype=np.float32)        
        mapped_key = ref_umap_key
        mapped_query = _knn_barycentric_project(
            rep_ref,
            rep_query,
            ref_umap,
            k=k_map,
            metric="euclidean",
            weight_power=embedding_weight_power,
        )
        adata_query.obsm[mapped_key] = mapped_query

    adata_joint = assemble_joint_embedding(
        rep_key,
        {"Reference": adata_ref, "Query": adata_query},
    )

    if mapped_key is not None:
        ref_umap = adata_ref.obsm[ref_umap_key]
        query_umap = adata_query.obsm[mapped_key]
        if ref_umap.shape[1] != query_umap.shape[1]:
            raise ValueError("UMAP dimensionality mismatch between reference and query.")
        joint_umap = np.full((adata_joint.n_obs, ref_umap.shape[1]), np.nan, dtype=np.float32)
        mask_ref = adata_joint.obs["modality"] == "Reference"
        mask_query = adata_joint.obs["modality"] == "Query"
        joint_umap[mask_ref] = ref_umap
        joint_umap[mask_query] = query_umap
        adata_joint.obsm[mapped_key] = joint_umap

    if inplace:
        for col in adata_query.obs.columns:
            if col in adata.obs and isinstance(adata.obs[col].dtype, pd.CategoricalDtype):
                base_col = adata.obs[col]
                query_col = adata_query.obs[col]
                base_categories = list(base_col.cat.categories)
                if isinstance(query_col.dtype, pd.CategoricalDtype):
                    extra_categories = list(query_col.cat.categories)
                else:
                    extra_categories = [v for v in pd.unique(query_col.dropna())]
                for value in extra_categories:
                    if value not in base_categories:
                        base_categories.append(value)
                if base_categories != list(base_col.cat.categories):
                    adata.obs[col] = base_col.cat.set_categories(base_categories)
                values = query_col.to_numpy()
            else:
                values = adata_query.obs[col].values
            adata.obs.loc[adata_query.obs_names, col] = values
        if mapped_key is not None:
            ref_umap = adata_ref.obsm[ref_umap_key]
            query_umap = adata_query.obsm[mapped_key]
            joint_umap = np.full((adata.n_obs, ref_umap.shape[1]), np.nan, dtype=np.float32)
            joint_umap[np.asarray(ref_mask)] = ref_umap
            joint_umap[np.asarray(query_mask)] = query_umap
            adata.obsm[mapped_key] = joint_umap
        return adata
    return adata_joint
