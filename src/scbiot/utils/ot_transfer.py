from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from sklearn.neighbors import NearestNeighbors

from .ot_transport import ot_label_transfer


def _stack_matrices(matrices: List[np.ndarray | sp.spmatrix]) -> np.ndarray | sp.spmatrix:
    if any(sp.issparse(m) for m in matrices):
        sparse_blocks = [m if sp.issparse(m) else sp.csr_matrix(m) for m in matrices]
        return sp.vstack(sparse_blocks, format="csr")
    return np.vstack([np.asarray(m) for m in matrices])


def assemble_joint_embedding(rep_key: str, modalities: Dict[str, AnnData]) -> AnnData:
    embeddings: List[np.ndarray] = []
    obs_frames: List[pd.DataFrame] = []
    adata_blocks: List[AnnData] = []
    for name, adata in modalities.items():
        if rep_key not in adata.obsm:
            raise KeyError(f"Representation '{rep_key}' missing for modality '{name}'.")
        Z = np.asarray(adata.obsm[rep_key], dtype=np.float32)
        embeddings.append(Z)
        obs_df = adata.obs.copy()
        obs_df["obs_original"] = adata.obs_names
        obs_df["modality"] = name
        obs_df.index = pd.Index([f"{name}::{idx}" for idx in obs_df.index], name=obs_df.index.name)
        obs_frames.append(obs_df)
        adata_blocks.append(adata)

    Z_all = np.vstack(embeddings)
    obs_all = pd.concat(obs_frames, axis=0, sort=False)
    first = adata_blocks[0]
    if all(adata.var_names.equals(first.var_names) for adata in adata_blocks[1:]):
        X_all = _stack_matrices([adata.X for adata in adata_blocks])
        ad_all = AnnData(X=X_all, var=first.var.copy())
    else:
        ad_all = AnnData(X=np.zeros((Z_all.shape[0], 1), dtype=np.float32))
    ad_all.obs = obs_all
    ad_all.obsm[rep_key] = Z_all
    return ad_all


def label_transfer_shared_pca(
    adata_ref: AnnData,
    adata_query: AnnData,
    *,
    label_key: str,
    use_rep: str,
    k: int = 50,
    metric: str = "cosine",
    min_conf: float,
) -> AnnData:
    if label_key not in adata_ref.obs:
        raise KeyError(f"{label_key!r} not found in adata_ref.obs")
    if use_rep not in adata_ref.obsm or use_rep not in adata_query.obsm:
        raise KeyError(f"{use_rep!r} missing in AnnData.obsm")

    ot_meta = adata_query.uns.get("_ot_alignment")
    if ot_meta is None or "indices" not in ot_meta or "weights" not in ot_meta:
        raise RuntimeError("OT transport weights missing; run integrate_ot first.")

    rna_order = pd.Index(ot_meta["rna_obs"])
    labels_series = adata_ref.obs[label_key].reindex(rna_order)
    if labels_series.isna().any():
        raise ValueError("Missing RNA labels after reindexing; ensure obs names align with OT metadata.")

    transport = {
        "indices": ot_meta["indices"],
        "weights": ot_meta["weights"],
    }
    if ot_meta.get("residual") is not None:
        transport["residual"] = ot_meta["residual"]

    res = ot_label_transfer(
        transport=transport,
        target_labels=labels_series,
        min_conf=min_conf,
        confidence_mode="entropy",
    )

    adata_query.obs["pred_cell_type"] = res["pred_labels"]
    adata_query.obs["pred_confidence"] = res["confidence"]
    adata_query.uns["pred_cell_type_classes"] = res["classes"]
    adata_query.uns["pred_confidence_mode"] = "entropy"
    return adata_query


def _resolve_obsm_key(adata: AnnData, key: str) -> str:
    if key in adata.obsm:
        return key
    alt = f"X_{key}"
    if not key.startswith("X_") and alt in adata.obsm:
        return alt
    tried = f"{key!r}" if key.startswith("X_") else f"{key!r} and {alt!r}"
    raise KeyError(f"{key!r} not found in AnnData.obsm (tried {tried}).")


def _knn_barycentric_project(
    source_rep: np.ndarray,
    target_rep: np.ndarray,
    source_embed: np.ndarray,
    *,
    k: int,
    metric: str = "euclidean",
    weight_power: float = 2.0,
) -> np.ndarray:
    if sp.issparse(source_rep):
        source_rep = source_rep.toarray()
    if sp.issparse(target_rep):
        target_rep = target_rep.toarray()
    source_rep = np.asarray(source_rep, dtype=np.float32, order="C")
    target_rep = np.asarray(target_rep, dtype=np.float32, order="C")
    source_embed = np.asarray(source_embed, dtype=np.float32, order="C")

    if source_rep.shape[0] != source_embed.shape[0]:
        raise ValueError("source_rep and source_embed must have the same number of rows.")
    if source_rep.shape[0] == 0:
        return np.zeros((target_rep.shape[0], source_embed.shape[1]), dtype=np.float32)

    k = int(max(1, min(k, source_rep.shape[0])))
    nn = NearestNeighbors(n_neighbors=k, metric=metric)
    nn.fit(source_rep)
    dist, idx = nn.kneighbors(target_rep, n_neighbors=k, return_distance=True)

    tau = float(np.median(dist)) if dist.size else 1.0
    if not np.isfinite(tau) or tau <= 0:
        tau = 1.0
    weights = np.exp(-dist / (tau + 1e-12))
    if weight_power != 1.0:
        weights = np.power(weights, weight_power)
    mass = weights.sum(axis=1, keepdims=True)
    weights = np.divide(
        weights,
        np.where(mass > 0, mass, 1.0),
        out=np.zeros_like(weights),
        where=mass > 0,
    )

    out = (source_embed[idx] * weights[..., None]).sum(axis=1)
    return out.astype(np.float32, copy=False)
