"""Lightweight shared-PCA coembedding for RNA + gene-activity inputs (Scanpy-style API)."""

from __future__ import annotations
from typing import Any, Dict, Optional, Sequence, Literal

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from sklearn.decomposition import PCA
import anndata as ad

from .peaks import ensure_csr_f32

import scipy.sparse as sp


def _ensure_csc_f32(X):
    if sp.issparse(X):
        X = X.tocsc()
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        return X
    return np.asarray(X, dtype=np.float32, order="C")


# -----------------------------
# kNN outlier flagger (per-cluster, in-place)
# -----------------------------
from sklearn.neighbors import NearestNeighbors


def _robust_thr_mad(x: np.ndarray, z: float = 3.0, eps: float = 1e-12) -> float:
    """thr = median + z * 1.4826 * MAD (robust)."""
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    mad = max(float(mad), float(eps))
    return float(med + float(z) * 1.4826 * mad)


def _flag_outliers_per_cluster_inplace(
    adata: AnnData,
    *,
    X_key: str,
    label_key: str,
    unknown_label: str = "Unknown",
    k: int = 30,
    z: float = 3.0,
    min_cluster_size: Optional[int] = None,
    store_qc_cols: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    For each labeled cluster (label_key != unknown_label), compute within-cluster self-kNN mean distance.
    Outliers (robust MAD cutoff) are relabeled to unknown_label in-place.

    Writes:
      - adata.obs[f"{label_key}__knn_mean"] (optional)
      - adata.obs[f"{label_key}__knn_outlier"] (optional)
      - adata.uns[f"{label_key}__thr_by_cluster"]
      - adata.uns[f"{label_key}__summary"]
    """
    if X_key not in adata.obsm:
        raise KeyError(f"{X_key!r} not in adata.obsm")
    if label_key not in adata.obs:
        raise KeyError(f"{label_key!r} not in adata.obs")

    X = np.asarray(adata.obsm[X_key], dtype=np.float32, order="C")
    labels = adata.obs[label_key].copy()
    ref_mask = labels.notna() & labels.ne(unknown_label)

    if min_cluster_size is None:
        min_cluster_size = max(k + 2, 10)

    # allow setting Unknown even if categorical
    if isinstance(adata.obs[label_key].dtype, pd.CategoricalDtype):
        adata.obs[label_key] = adata.obs[label_key].astype("object")
        labels = adata.obs[label_key].copy()

    mean_col = f"{label_key}__knn_mean"
    out_col = f"{label_key}__knn_outlier"
    if store_qc_cols:
        if mean_col not in adata.obs:
            adata.obs[mean_col] = np.nan
        if out_col not in adata.obs:
            adata.obs[out_col] = False

    thr_by_cluster: Dict[str, Any] = {}
    summary_rows = []

    clusters = pd.Index(labels[ref_mask].astype(str).unique()).sort_values()

    for cl in clusters:
        mask_cl = ref_mask & labels.astype(str).eq(cl)
        idx = np.flatnonzero(np.asarray(mask_cl))
        n = int(idx.size)
        if n < int(min_cluster_size):
            if verbose:
                print(f"[skip] {cl}: n={n} (<{min_cluster_size})")
            summary_rows.append((str(cl), n, np.nan, 0))
            continue

        kk = int(min(k, n - 1))
        X_cl = X[idx]

        nbrs = NearestNeighbors(n_neighbors=kk + 1, metric="euclidean").fit(X_cl)
        dists, _ = nbrs.kneighbors(X_cl)  # self in col0
        mean_knn = dists[:, 1:].mean(axis=1)

        thr = _robust_thr_mad(mean_knn, z=z)
        outlier = mean_knn > thr
        n_out = int(outlier.sum())

        thr_by_cluster[str(cl)] = {"n": n, "k": kk, "thr": float(thr), "z": float(z)}
        summary_rows.append((str(cl), n, float(thr), n_out))

        if store_qc_cols:
            adata.obs.iloc[idx, adata.obs.columns.get_loc(mean_col)] = mean_knn.astype(np.float32, copy=False)
            adata.obs.iloc[idx, adata.obs.columns.get_loc(out_col)] = outlier

        if n_out > 0:
            out_idx = idx[outlier]
            adata.obs.iloc[out_idx, adata.obs.columns.get_loc(label_key)] = unknown_label

        if verbose:
            print(f"[ok] {cl}: n={n} k={kk} thr={thr:.4g} out={n_out} ({n_out/n:.2%})")

    summary = pd.DataFrame(summary_rows, columns=["cluster", "n", "thr", "n_outliers"]).sort_values(
        ["n_outliers", "n"], ascending=[False, False]
    )
    adata.uns[f"{label_key}__thr_by_cluster"] = thr_by_cluster
    adata.uns[f"{label_key}__summary"] = summary
    return {"summary": summary, "thr_by_cluster": thr_by_cluster}


# -----------------------------
# helpers (internal)
# -----------------------------
def _select_layer(adata: AnnData, layer: Optional[str]) -> str:
    """Return the effective layer name ('X' meaning adata.X) given a user layer."""
    if layer is None:
        return "X"
    if layer in adata.layers:
        return layer
    return "X"


def _normalize_log1p(
    adata: AnnData,
    *,
    layer: Optional[str],
    layer_out: str,
    target_sum: float = 1e4,
    meta_prefix: str = "scbiot:coembed_pca",
) -> None:
    """Normalize + log1p into `adata.layers[layer_out]` with caching in `adata.uns`."""
    src = _select_layer(adata, layer)
    meta_key = f"{meta_prefix}:norm:{layer_out}"
    meta: Optional[Dict[str, Any]] = adata.uns.get(meta_key)  # type: ignore[assignment]

    if (
        layer_out in adata.layers
        and meta
        and meta.get("source") == src
        and float(meta.get("target_sum", target_sum)) == float(target_sum)
        and tuple(meta.get("shape", ())) == (adata.n_obs, adata.n_vars)
    ):
        return

    X0 = adata.layers[src] if src != "X" else adata.X
    X0 = ensure_csr_f32(X0)

    adata.layers[layer_out] = X0.copy()
    sc.pp.normalize_total(adata, target_sum=target_sum, layer=layer_out, inplace=True)
    sc.pp.log1p(adata, layer=layer_out)

    adata.uns[meta_key] = {"source": src, "target_sum": float(target_sum), "shape": (adata.n_obs, adata.n_vars)}


def _auto_ga_layer(adata_ga: AnnData, ga_layer: Optional[str]) -> Optional[str]:
    """Auto-pick GA input layer with the same logic you had before."""
    if ga_layer is not None:
        return ga_layer if ga_layer in adata_ga.layers else None
    if "ga_smooth" in adata_ga.layers:
        return "ga_smooth"
    if "ga" in adata_ga.layers:
        return "ga"
    return None


def _joint_hvgs(
    adata_rna: AnnData,
    adata_ga: AnnData,
    *,
    n_top: int,
    rna_norm: str,
    ga_norm: str,
    batch_key: Optional[str],
    rna_layer: Optional[str],
    ga_layer: Optional[str],
    flavor: Optional[str]= "cell_ranger",
    target_sum: float,
    meta_prefix: str,
) -> pd.Index:
    _normalize_log1p(adata_rna, layer=rna_layer, layer_out=rna_norm, target_sum=target_sum, meta_prefix=meta_prefix)

    ga_in = _auto_ga_layer(adata_ga, ga_layer)
    _normalize_log1p(adata_ga, layer=ga_in, layer_out=ga_norm, target_sum=target_sum, meta_prefix=meta_prefix)

    adata_rna.layers[rna_norm] = _ensure_csc_f32(adata_rna.layers[rna_norm])
    adata_ga.layers[ga_norm] = _ensure_csc_f32(adata_ga.layers[ga_norm])

    use_batch = (
    batch_key
    if (batch_key and batch_key in adata_rna.obs.columns and batch_key in adata_ga.obs.columns)
    else None
    )

    sc.pp.highly_variable_genes(
        adata_rna,
        flavor=flavor,
        n_top_genes=min(n_top, adata_rna.n_vars),
        layer=rna_norm,
        batch_key=use_batch,
        span=0.6,
        inplace=True,
    )
    sc.pp.highly_variable_genes(
        adata_ga,
        flavor=flavor,
        n_top_genes=min(n_top, adata_ga.n_vars),
        layer=ga_norm,
        batch_key=use_batch,
        span=0.6,
        inplace=True,
    )

    rna_hv = set(adata_rna.var_names[adata_rna.var["highly_variable"].values]) 
    ga_hv = set(adata_ga.var_names[adata_ga.var["highly_variable"].values])
    shared = sorted(set(adata_rna.var_names).intersection(adata_ga.var_names))

    if ("dispersions_norm" in adata_rna.var.columns) and ("dispersions_norm" in adata_ga.var.columns):
        col = "dispersions_norm"
    elif ("dispersions" in adata_rna.var.columns) and ("dispersions" in adata_ga.var.columns):
        col = "dispersions"
    else:
        raise ValueError("Expected dispersions(_norm) in .var after highly_variable_genes(cell_ranger).")

    rna_s = np.nan_to_num(adata_rna.var.loc[shared, col].to_numpy(), nan=-np.inf)
    ga_s  = np.nan_to_num(adata_ga.var.loc[shared, col].to_numpy(),  nan=-np.inf)
    score = np.maximum(rna_s, ga_s)

    k = min(int(n_top), len(shared))
    order_all = np.argsort(-score)  # best->worst in shared space

    # prefer shared HVGs first
    candidates = set(rna_hv.union(ga_hv))
    order_pref = [i for i in order_all if shared[i] in candidates]
    order_rest = [i for i in order_all if shared[i] not in candidates]

    pick = (order_pref + order_rest)[:k]
    genes = sorted([shared[i] for i in pick])
    # ------------------------------------------------------------

    if len(genes) < 500:
        raise ValueError("Too few shared HVGs; check name harmonization or GA quality.")

    return pd.Index(genes)


def _shared_pca_from_genes(
    adata_rna: AnnData,
    adata_ga: AnnData,
    genes: Sequence[str],
    *,
    n_comps: int,
    rna_norm: str,
    ga_norm: str,
    rep_key: str,
    loadings_key: str,
    meta_prefix: str,
    pca_solver: str,
    projection_chunk_size: Optional[int],
) -> None:
    # NOTE: keeps your original behavior (dense arrays)
    Xr = ensure_csr_f32(adata_rna[:, genes].layers[rna_norm]).toarray()
    Xg = ensure_csr_f32(adata_ga[:, genes].layers[ga_norm])

    mu = Xr.mean(axis=0, dtype=np.float64)
    var = (Xr**2).mean(axis=0, dtype=np.float64) - mu**2
    sd = np.sqrt(np.maximum(var, 1e-8))
    mu32 = mu.astype(np.float32, copy=False)
    sd32 = sd.astype(np.float32, copy=False)

    Xr_z = (Xr - mu32) / sd32
    np.clip(Xr_z, -10, 10, out=Xr_z)

    k = int(min(n_comps, Xr_z.shape[1], max(2, Xr_z.shape[0] - 1)))
    pca = PCA(n_components=k, svd_solver=pca_solver, random_state=0)
    Zr = pca.fit_transform(Xr_z).astype(np.float32)

    n_query = Xg.shape[0]
    if projection_chunk_size is None or n_query <= projection_chunk_size:
        Xg_z = (Xg.toarray() - mu32) / sd32
        np.clip(Xg_z, -10, 10, out=Xg_z)
        Zg = pca.transform(Xg_z).astype(np.float32)
    else:
        Zg = np.empty((n_query, k), dtype=np.float32)
        for start in range(0, n_query, projection_chunk_size):
            end = min(n_query, start + projection_chunk_size)
            Xg_block = Xg[start:end].toarray()
            Xg_block = (Xg_block - mu32) / sd32
            np.clip(Xg_block, -10, 10, out=Xg_block)
            Zg[start:end] = pca.transform(Xg_block).astype(np.float32, copy=False)

    adata_rna.obsm[rep_key] = Zr
    adata_ga.obsm[rep_key] = Zg

    loadings = pca.components_.T.astype(np.float32, copy=False)
    L_full = np.zeros((adata_rna.n_vars, loadings.shape[1]), np.float32)
    idx = adata_rna.var_names.get_indexer(genes)
    L_full[idx[idx >= 0]] = loadings
    adata_rna.varm[loadings_key] = L_full

    meta = {
        "genes_used": list(genes),
        "n_comps": int(loadings.shape[1]),
        "explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float32).tolist(),
        "norm_layers": {"rna": rna_norm, "ga": ga_norm},
        "rep_key": rep_key,
        "loadings_key": loadings_key,
    }
    adata_rna.uns[f"{meta_prefix}:pca:{rep_key}"] = meta
    adata_rna.uns["shared_pca_meta"] = meta


# -----------------------------
# public API (scanpy-style) + ONLY add flag_outlier behavior
# -----------------------------
def coembed_pca(
    adata_reference: AnnData,
    adata_query: AnnData,
    *,
    out_key: str = "X_pca_shared",
    label: str = "modality",
    mode: Optional[Literal["paired", "unpaired"]] = None,
    keys: Optional[Sequence[str]] = None,
    reference_layer: Optional[str] = None,
    query_layer: Optional[str] = None,
    n_top_genes: int = 4000,
    n_components: int = 50,
    flavor: Optional[str]= "cell_ranger",
    reference_norm_layer: str = "rna_log1p",
    query_norm_layer: str = "ga_log1p",
    batch_key: Optional[str] = None,
    genes: Optional[Sequence[str]] = None,
    pca_solver: str = "randomized",
    projection_chunk_size: Optional[int] = 4096,
    # NEW: outlier flagging (minimal surface)
    label_key: Optional[str] = None,
    unlabeled_category: str = "Unknown",
    flag_outlier: bool = True,
    outlier_k: int = 30,
    outlier_z: float = 3.0,    
    min_cluster_size=None,
    store_qc_cols: bool = False,
    verbose: bool = True,
) -> AnnData:
    """
    Shared PCA coembedding (reference-fitted PCA, query projection) and optional per-cluster outlier flagging.

    Parameters
    ----------
    adata_reference
        Reference modality AnnData (e.g., scRNA / GEX). PCA is fitted on this object.
    adata_query
        Query modality AnnData (e.g., gene-activity / scATAC GA). PCA is projected onto this object.
    out_key
        `.obsm` key for the shared PCA coordinates written into both input objects and the returned joint AnnData.
    label
        Column name created in the joint AnnData (`ad.concat`) indicating modality (e.g., "modality").
    mode
        If "paired", use `join="outer"` when concatenating; otherwise use `join="inner"`.
    keys
        Two strings naming the modalities for `ad.concat` (default: ("reference","query")).
    reference_layer
        Input layer for reference counts. If None, uses `.X`. If provided but missing, falls back to `.X`.
    query_layer
        Input layer for query counts. If None, auto-picks ("ga_smooth" -> "ga" -> `.X`).
    n_top_genes
        Number of HVGs to select per modality before intersecting (shared genes only).
    n_components
        Target number of PCA components (actual k may be smaller if limited by cells/genes).
    reference_norm_layer
        Output layer name for normalized+log1p reference counts (cached).
    query_norm_layer
        Output layer name for normalized+log1p query counts (cached).
    batch_key
        Optional `adata_reference.obs` column used for batch-aware HVG selection (Scanpy HVG `batch_key`).
    genes
        Optional explicit gene list. If provided, skips HVG selection and uses these genes (must exist in both).
    pca_solver
        scikit-learn PCA solver (e.g., "randomized", "full", "auto").
    projection_chunk_size
        If set, project query in chunks of this size to reduce memory spikes.

    label_key
        `adata.obs` column containing cell labels (e.g., "cell_type") used for reference labels and outlier flagging.
    unlabeled_category
        Label value treated as "unlabeled" query and used to relabel outliers (excluded from per-cluster flagging).
    flag_outlier
        If True, run per-cluster kNN outlier detection on the final joint embedding and set outliers to `unlabeled_category`.
    outlier_k
        Number of neighbors for within-cluster kNN distance (uses k+1 internally to exclude self).
    outlier_z
        Robust cutoff multiplier for MAD threshold: `thr = median + outlier_z * 1.4826 * MAD`.
    min_cluster_size
        Minimum labeled cells per cluster required to run outlier detection. If None, uses `max(outlier_k+2, 10)`.
    store_qc_cols
        If True, store QC columns in `adata_joint.obs`:
        - `{label_key}__knn_mean`, `{label_key}__knn_outlier`
        and summary in `adata_joint.uns`.
    verbose
        If True, print per-cluster outlier stats.

    Returns
    -------
    AnnData
        Joint AnnData concatenating reference and query with `adata_joint.obsm[out_key]` stacked.

    Notes
    -----
    - This function DOES NOT clip/modify embeddings for outliers; it only relabels them when `flag_outlier=True`.
    - By default, this sets `adata_query.obs[label_key] = unlabeled_category` (so outlier flagging targets reference clusters).
    
    Examples
    --------
    >>> adata = scb.pp.coembed_pca(
    ...     adata_gex, adata_ga,
    ...     label="modality",
    ...     keys=("reference", "query"),
    ...     reference_layer="counts",
    ...     query_layer="ga_smooth",
    ...     out_key="X_shared_pca",
    ...     label_key="cell_type",
    ...     unlabeled_category="Unknown",
    ...     flag_outlier=True,
    ... )    
    """
    if keys is None:
        keys = ("reference", "query")
    if len(keys) != 2:
        raise ValueError("keys must contain exactly two entries (reference, query).")

    if flag_outlier and (label_key is None):
        raise ValueError("flag_outlier=True requires label_key=...")

    adata_query.obs[label_key] = unlabeled_category

    # ---- HVGs / genes ----
    if genes is None:
        genes_idx = _joint_hvgs(
            adata_reference,
            adata_query,
            n_top=n_top_genes,
            flavor=flavor,
            rna_norm=reference_norm_layer,
            ga_norm=query_norm_layer,
            batch_key=batch_key,
            rna_layer=reference_layer,
            ga_layer=query_layer,
            target_sum=1e4,
            meta_prefix="scbiot:coembed_pca",
        )
    else:
        genes_idx = pd.Index(genes)

    # ---- PCA fit on reference, project query ----
    loadings_key = "PCs_shared" if out_key == "X_pca_shared" else f"PCs_{out_key}"
    _shared_pca_from_genes(
        adata_reference,
        adata_query,
        genes_idx,
        n_comps=n_components,
        rna_norm=reference_norm_layer,
        ga_norm=query_norm_layer,
        rep_key=out_key,
        loadings_key=loadings_key,
        meta_prefix="scbiot:coembed_pca",
        pca_solver=pca_solver,
        projection_chunk_size=projection_chunk_size,
    )

    # ---- concat + joint embedding ----
    if mode == "paired":
        adata_joint = ad.concat(
            [adata_reference, adata_query],
            join="outer",
            index_unique="::",
            label=label,
            keys=list(keys),
        )
    else:
        adata_joint = ad.concat(
            [adata_reference, adata_query],
            join="inner",
            label=label,
            keys=list(keys),
        )

    if "obs_original" not in adata_joint.obs:
        obs_original = np.concatenate([adata_reference.obs_names.to_numpy(), adata_query.obs_names.to_numpy()])
        adata_joint.obs["obs_original"] = obs_original

    Zref = np.asarray(adata_reference.obsm[out_key], dtype=np.float32)
    Zqry = np.asarray(adata_query.obsm[out_key], dtype=np.float32)
    adata_joint.obsm[out_key] = np.vstack([Zref, Zqry])

    # ---- NEW: outlier flagging (ONLY addition) ----
    if flag_outlier:
        if label_key not in adata_joint.obs.columns:
            raise ValueError(f"label_key={label_key!r} not found in joint obs. "
                             "Set it in both adatas (query can be all 'Unknown').")
        _flag_outliers_per_cluster_inplace(
            adata_joint,
            X_key=out_key,
            label_key=label_key,
            unknown_label=unlabeled_category,
            k=outlier_k,
            z=outlier_z,
            min_cluster_size=min_cluster_size,
            store_qc_cols=store_qc_cols,
            verbose=verbose,
        )

    return adata_joint


# =======================
# Usage (as you want)
# =======================
# adata = scb.pp.coembed_pca(
#     adata_gex, adata_ga,
#     label="modality",
#     keys=("reference", "query"),
#     reference_layer="counts",
#     query_layer="ga_smooth",
#     out_key="X_shared_pca",
#     label_key="cell_type",
#     unlabeled_category="Unknown",
#     flag_outlier=True,
# )