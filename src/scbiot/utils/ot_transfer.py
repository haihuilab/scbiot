from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors


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
        z = np.asarray(adata.obsm[rep_key], dtype=np.float32)
        embeddings.append(z)
        obs_df = adata.obs.copy()
        obs_df["obs_original"] = adata.obs_names
        obs_df["modality"] = name
        obs_df.index = pd.Index([f"{name}::{idx}" for idx in obs_df.index], name=obs_df.index.name)
        obs_frames.append(obs_df)
        adata_blocks.append(adata)

    z_all = np.vstack(embeddings)
    obs_all = pd.concat(obs_frames, axis=0, sort=False)
    first = adata_blocks[0]
    first_vars = first.var_names
    if all(adata.var_names.equals(first_vars) for adata in adata_blocks[1:]):
        x_all = _stack_matrices([adata.X for adata in adata_blocks])
        ad_all = AnnData(X=x_all, var=first.var.copy())
    else:
        first_set = set(first_vars)
        if all(set(adata.var_names) == first_set for adata in adata_blocks[1:]):
            x_all = _stack_matrices(
                [(adata if adata.var_names.equals(first_vars) else adata[:, first_vars]).X for adata in adata_blocks]
            )
            ad_all = AnnData(X=x_all, var=first.var.copy())
        else:
            shared_set = set(first_vars)
            for adata in adata_blocks[1:]:
                shared_set &= set(adata.var_names)
            if shared_set:
                shared_vars = first_vars[first_vars.isin(shared_set)]
                x_all = _stack_matrices([adata[:, shared_vars].X for adata in adata_blocks])
                ad_all = AnnData(X=x_all, var=first.var.loc[shared_vars].copy())
            else:
                ad_all = AnnData(X=np.zeros((z_all.shape[0], 1), dtype=np.float32))
    ad_all.obs = obs_all
    ad_all.obsm[rep_key] = z_all
    return ad_all


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


def _validate_transfer_mode(transfer_mode: str) -> str:
    mode = transfer_mode.strip().lower()
    allowed = {"knn", "logreg"}
    if mode not in allowed:
        raise ValueError(f"transfer_mode must be one of {sorted(allowed)}.")
    return mode


def _encode_reference_labels(labels_series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    labels_cat = labels_series.astype("category")
    classes = labels_cat.cat.categories.to_numpy(dtype=object)
    codes = labels_cat.cat.codes.to_numpy(dtype=np.int32)
    valid = codes >= 0
    return classes, np.where(valid, codes, -1)


def _global_knn_scores(
    x_ref: np.ndarray,
    y_ref_codes: np.ndarray,
    x_query: np.ndarray,
    *,
    k: int,
    n_classes: int,
    metric: str,
) -> np.ndarray:
    k_eff = int(max(1, min(k, x_ref.shape[0])))
    nn = NearestNeighbors(n_neighbors=k_eff, metric=metric)
    nn.fit(x_ref)
    dist, idx = nn.kneighbors(x_query, n_neighbors=k_eff, return_distance=True)

    tau = float(np.median(dist)) if dist.size else 1.0
    if tau <= 0 or not np.isfinite(tau):
        tau = 1.0

    w = np.exp(-dist / tau).astype(np.float32, copy=False)
    cls = y_ref_codes[idx]

    scores = np.zeros((x_query.shape[0], n_classes), dtype=np.float32)
    row_idx = np.arange(x_query.shape[0])
    for j in range(k_eff):
        np.add.at(scores, (row_idx, cls[:, j]), w[:, j])

    row_sum = scores.sum(axis=1, keepdims=True)
    scores = np.divide(
        scores,
        np.where(row_sum > 0, row_sum, 1.0),
        out=np.zeros_like(scores),
        where=row_sum > 0,
    )
    return scores


def _logreg_scores(
    x_ref: np.ndarray,
    y_ref_codes: np.ndarray,
    x_query: np.ndarray,
    *,
    n_classes: int,
) -> np.ndarray:
    if x_ref.shape[0] == 0:
        return np.zeros((x_query.shape[0], n_classes), dtype=np.float32)

    clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        random_state=0,
    )
    clf.fit(x_ref, y_ref_codes)
    return np.asarray(clf.predict_proba(x_query), dtype=np.float32, order="C")


def label_transfer_shared_pca(
    adata_ref: AnnData,
    adata_query: AnnData,
    *,
    label_key: str,
    use_rep: str,
    transfer_mode: str = "logreg",
    k: int = 50,
    metric: str = "cosine",
    min_conf: float,
    unknown_label: str = "unknown",
) -> AnnData:
    mode = _validate_transfer_mode(transfer_mode)

    if label_key not in adata_ref.obs:
        raise KeyError(f"{label_key!r} not found in adata_ref.obs")

    rep_key = use_rep
    if rep_key not in adata_ref.obsm or rep_key not in adata_query.obsm:
        raise KeyError(f"{rep_key!r} missing in AnnData.obsm")

    x_ref = np.asarray(adata_ref.obsm[rep_key], dtype=np.float32, order="C")
    x_query = np.asarray(adata_query.obsm[rep_key], dtype=np.float32, order="C")

    labels_ref_all = adata_ref.obs[label_key].astype("string")
    keep_ref = labels_ref_all.notna().to_numpy()
    keep_ref &= labels_ref_all.to_numpy() != unknown_label

    x_ref_labeled = x_ref[keep_ref]
    labels_ref = labels_ref_all.iloc[np.where(keep_ref)[0]].astype(str)

    if x_ref_labeled.shape[0] == 0:
        raise ValueError("No labeled reference cells available for transfer.")

    classes, y_ref_codes = _encode_reference_labels(labels_ref)
    n_classes = int(classes.size)
    if mode == "knn":
        class_scores = _global_knn_scores(
            x_ref_labeled,
            y_ref_codes,
            x_query,
            k=k,
            n_classes=n_classes,
            metric=metric,
        )
    else:
        class_scores = _logreg_scores(
            x_ref_labeled,
            y_ref_codes,
            x_query,
            n_classes=n_classes,
        )

    pred_idx = class_scores.argmax(axis=1)
    pred = classes[pred_idx].astype(object)
    conf = class_scores.max(axis=1).astype(np.float32, copy=False)
    pred[conf < min_conf] = unknown_label

    adata_query.obs["pred_cell_type"] = pred
    adata_query.obs["pred_confidence"] = conf
    adata_query.uns["pred_cell_type_classes"] = pd.Index(classes)
    adata_query.uns["pred_confidence_mode"] = "max"

    return adata_query
