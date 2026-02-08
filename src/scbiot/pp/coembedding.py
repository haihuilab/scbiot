"""Lightweight shared-PCA coembedding for RNA + gene-activity inputs."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData
from sklearn.decomposition import PCA

from .peaks import ensure_csr_f32


def _normalize_log1p(adata: AnnData, layer_in: Optional[str], layer_out: str, target_sum: float = 1e4) -> None:
    source_layer = layer_in if (layer_in and layer_in in adata.layers) else "X"
    meta_key = f"_coembed_norm::{layer_out}"
    meta: Optional[Dict[str, Any]] = adata.uns.get(meta_key)  # type: ignore[assignment]
    if (
        layer_out in adata.layers
        and meta
        and meta.get("source_layer") == source_layer
        and float(meta.get("target_sum", target_sum)) == float(target_sum)
        and tuple(meta.get("shape", ())) == (adata.n_obs, adata.n_vars)
    ):
        return

    X0 = adata.layers[layer_in] if (layer_in and layer_in in adata.layers) else adata.X
    X0 = ensure_csr_f32(X0)
    adata.layers[layer_out] = X0.copy()
    sc.pp.normalize_total(adata, target_sum=target_sum, layer=layer_out, inplace=True)
    sc.pp.log1p(adata, layer=layer_out)
    adata.uns[meta_key] = {
        "source_layer": source_layer,
        "target_sum": float(target_sum),
        "shape": (adata.n_obs, adata.n_vars),
    }


def _joint_hvgs(
    adata_rna: AnnData,
    adata_ga: AnnData,
    *,
    n_top: int,
    rna_norm: str,
    ga_norm: str,
    batch_key: Optional[str],
) -> pd.Index:
    _normalize_log1p(adata_rna, layer_in=None, layer_out=rna_norm)
    ga_layer = "ga_smooth" if "ga_smooth" in adata_ga.layers else ("ga" if "ga" in adata_ga.layers else None)
    _normalize_log1p(adata_ga, layer_in=ga_layer, layer_out=ga_norm)

    use_batch = batch_key if (batch_key and batch_key in adata_rna.obs.columns) else None
    sc.pp.highly_variable_genes(
        adata_rna,
        flavor="seurat_v3",
        n_top_genes=min(n_top, adata_rna.n_vars),
        layer=rna_norm,
        batch_key=use_batch,
        span=0.6,
        inplace=True,
    )
    sc.pp.highly_variable_genes(
        adata_ga,
        flavor="seurat_v3",
        n_top_genes=min(n_top, adata_ga.n_vars),
        layer=ga_norm,
        span=0.6,
        inplace=True,
    )
    rna_hv = set(adata_rna.var_names[adata_rna.var["highly_variable"].values])
    ga_hv = set(adata_ga.var_names[adata_ga.var["highly_variable"].values])
    shared = set(adata_rna.var_names).intersection(adata_ga.var_names)
    genes = sorted(shared.intersection(rna_hv.union(ga_hv)))
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
) -> None:
    Xr = ensure_csr_f32(adata_rna[:, genes].layers[rna_norm]).toarray()
    Xg = ensure_csr_f32(adata_ga[:, genes].layers[ga_norm]).toarray()

    mu = Xr.mean(axis=0, dtype=np.float64)
    var = (Xr**2).mean(axis=0, dtype=np.float64) - mu**2
    sd = np.sqrt(np.maximum(var, 1e-8))
    Xr_z = np.clip((Xr - mu) / sd, -10, 10)
    Xg_z = np.clip((Xg - mu) / sd, -10, 10)

    k = int(min(n_comps, Xr_z.shape[1], max(2, Xr_z.shape[0] - 1)))
    pca = PCA(n_components=k, svd_solver="full", random_state=0)
    Zr = pca.fit_transform(Xr_z).astype(np.float32)
    Zg = pca.transform(Xg_z).astype(np.float32)

    adata_rna.obsm[rep_key] = Zr
    adata_ga.obsm[rep_key] = Zg

    loadings = pca.components_.T.astype(np.float32, copy=False)
    L_full = np.zeros((adata_rna.n_vars, loadings.shape[1]), np.float32)
    idx = adata_rna.var_names.get_indexer(genes)
    L_full[idx[idx >= 0]] = loadings
    adata_rna.varm[loadings_key] = L_full

    adata_rna.uns["shared_pca_meta"] = {
        "genes_used": list(genes),
        "n_comps": int(loadings.shape[1]),
        "explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float32).tolist(),
        "norm_layers": {"rna": rna_norm, "ga": ga_norm},
        "rep_key": rep_key,
        "loadings_key": loadings_key,
    }


def coembed_pca(
    adata_rna: AnnData,
    adata_ga: AnnData,
    *,
    out_key: str = "X_pca_shared",
    n_top_genes: int = 4000,
    n_components: int = 50,
    rna_norm_layer: str = "rna_log1p",
    ga_norm_layer: str = "ga_log1p",
    batch_key: Optional[str] = "batch",
    label: str = "modality",
    keys: Optional[Sequence[str]] = None,
    genes: Optional[Sequence[str]] = None,
) -> AnnData:
    """
    Build a shared PCA coembedding (RNA-fitted PCA, GA projection) and return a joint AnnData.

    The PCA basis is learned from RNA and applied to GA after z-scoring with RNA statistics.

    Parameters
    ----------
    adata_rna:
        RNA AnnData with genes in ``.var_names``.
    adata_ga:
        Gene-activity AnnData aligned to the same genes (``.var_names``).
    out_key:
        Key in ``.obsm`` to store the shared PCA coordinates.
    n_top_genes:
        Number of shared highly variable genes to use when ``genes`` is not provided.
    n_components:
        Number of PCA components to compute.
    rna_norm_layer:
        Layer name to store log1p-normalized RNA counts for HVG/PCA.
    ga_norm_layer:
        Layer name to store log1p-normalized GA counts for HVG/PCA.
    batch_key:
        Optional ``adata_rna.obs`` column for batch-aware HVG selection.
    label:
        Column name added to ``adata_joint.obs`` to mark modality.
    keys:
        Labels for the two modalities, in RNA/GA order.
    genes:
        Explicit list of genes to use instead of HVGs.

    Returns
    -------
    AnnData
        Concatenated RNA/GA AnnData with shared PCA coordinates in ``.obsm[out_key]``.

    Notes
    -----
    The inputs are modified in-place: normalized layers are added, ``.obsm[out_key]`` is
    populated on both objects, and PCA loadings/metadata are stored on ``adata_rna``.

    Examples
    --------
    Basic usage:

    >>> import scbiot as scb
    >>> adata_joint = scb.pp.coembed_pca(adata_rna, adata_ga, out_key="X_pca_shared")
    """
    if genes is None:
        genes_idx = _joint_hvgs(
            adata_rna,
            adata_ga,
            n_top=n_top_genes,
            rna_norm=rna_norm_layer,
            ga_norm=ga_norm_layer,
            batch_key=batch_key,
        )
    else:
        genes_idx = pd.Index(genes)

    loadings_key = "PCs_shared" if out_key == "X_pca_shared" else f"PCs_{out_key}"
    _shared_pca_from_genes(
        adata_rna,
        adata_ga,
        genes_idx,
        n_comps=n_components,
        rna_norm=rna_norm_layer,
        ga_norm=ga_norm_layer,
        rep_key=out_key,
        loadings_key=loadings_key,
    )

    if keys is None:
        keys = ["RNA", "ATAC_GA"]
    if len(keys) != 2:
        raise ValueError("keys must contain exactly two entries (one per input AnnData).")

    adata_joint = ad.concat(
        [adata_rna, adata_ga],
        join="inner",
        label=label,
        keys=keys,
    )
    if "obs_original" not in adata_joint.obs:
        obs_original = np.concatenate([adata_rna.obs_names.to_numpy(), adata_ga.obs_names.to_numpy()])
        adata_joint.obs["obs_original"] = obs_original

    Zr = np.asarray(adata_rna.obsm[out_key], dtype=np.float32)
    Zg = np.asarray(adata_ga.obsm[out_key], dtype=np.float32)
    adata_joint.obsm[out_key] = np.vstack([Zr, Zg])
    return adata_joint
