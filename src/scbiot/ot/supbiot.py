# =================== supbiot.py: label transfer ===================
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Sequence

import difflib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder


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
    batch_key: str = "batch",
    out_key: Optional[str] = None,
    verbose: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Transfer labels from reference to query using OT alignment computed by integrate_ot.
    """
    from ..utils.ot_transfer import (
        _knn_barycentric_project,
        _resolve_obsm_key,
        assemble_joint_embedding,
        label_transfer_shared_pca,
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

    if label_key not in adata.obs:
        raise KeyError(f"{label_key!r} not found in adata.obs")
    labels = adata.obs[label_key]
    if isinstance(unlabeled_category, (list, tuple, set)):
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

    adata_ref = adata[ref_mask].copy()
    adata_query = adata[query_mask].copy()

    if rep_key not in adata.obsm:
        raise KeyError(f"{rep_key!r} not found in adata.obsm")

    ot_meta = adata.uns.get("_ot_alignment")
    if ot_meta is not None:
        adata_query.uns["_ot_alignment"] = ot_meta

    label_transfer_shared_pca(
        adata_ref,
        adata_query,
        label_key=label_key,
        use_rep=rep_key,
        min_conf=min_conf,
    )
    if pred_label_key != "pred_cell_type" and pred_label_key not in adata_query.obs:
        adata_query.obs[pred_label_key] = adata_query.obs["pred_cell_type"]
    if pred_conf_key != "pred_confidence" and pred_conf_key not in adata_query.obs:
        adata_query.obs[pred_conf_key] = adata_query.obs["pred_confidence"]

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
        mapped_key = ref_umap_key if not use_embedding_ref else f"{ref_umap_key}_ref"
        adata_query.obsm[mapped_key] = _knn_barycentric_project(
            rep_ref,
            rep_query,
            ref_umap,
            k=k_map,
            metric="euclidean",
            weight_power=embedding_weight_power,
        )

    adata_joint = assemble_joint_embedding(
        rep_key,
        {"Reference": adata_ref, "Query": adata_query},
    )
    if batch_key not in adata_joint.obs:
        adata_joint.obs[batch_key] = adata_joint.obs["modality"]

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
