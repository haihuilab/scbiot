from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import warnings

import numpy as np
import ot
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from anndata import AnnData
from numpy.linalg import svd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.utils.sparsefuncs import inplace_column_scale
from sklearn.utils.sparsefuncs_fast import inplace_csr_row_normalize_l1
from ..pp.peaks import (
    add_iterative_lsi,
    annotate_gene_activity,
    ensure_csr_f32,
    find_variable_features,
    harmonize_gene_names,
    knn_smooth_ga_on_atac,
    remove_promoter_proximal_peaks,
)

from ..ot.integrate_unpaired import (    
    _ot_barycentric_gpu,
    _sinkhorn_uot_torch,
    compute_ot_alignment,
    ot_label_transfer
    
)
from .integrate import integrate_ot


__all__ = [
    "AtacPreprocessConfig",
    "AtacPreprocessResult",
    "CoEmbeddingConfig",
    "CoEmbeddingResult",
    "align_lsi_with_ga_anchors",
    "annotate_gene_activity",
    "assemble_joint_embedding",
    "build_aligned_coembedding",
    "build_atac_lsi",
    "compute_ot_alignment",
    "ensure_csr_f32",
    "find_variable_features",
    "harmonize_gene_names",
    "label_transfer_shared_pca",    
    "prepare_transfer_layers",
    "transfer_labels",
    "ot_label_transfer",
    "preprocess_atac",
    "remove_promoter_proximal_peaks",    
]


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class CoEmbeddingConfig:
    n_top_genes: int = 4000
    n_components: int = 50
    use_optimal_transport: bool = True
    ot_reg: float = 0.05
    ot_reg_m: float = 0.60
    ot_cost_clip_q: float = 0.90
    ot_clip_big: float = 50.0
    ot_topk: int = 64
    ot_backend: str = "torch"
    ot_use_gpu: bool = True
    ot_gpu_device: int = 0
    ot_chunk_size: Optional[int] = 1024
    ref_norm_layer: str = "rna_log1p"
    query_norm_layer: str = "ga_log1p"
    batch_key: Optional[str] = "batch"
    rep_in: str = "X_pca_shared"
    rep_out: str = "X_pca_shared_aligned"


@dataclass
class CoEmbeddingResult:
    embedding_key: str
    genes: pd.Index
    config: CoEmbeddingConfig

    def concat_modalities(
        self,
        adata_ref: AnnData,
        adata_query: AnnData,
        *,
        label: str = "modality",
        **kwargs: Any,
    ) -> AnnData:
        params = dict(label=label, join="inner", merge="same")
        params.update(kwargs)
        return sc.concat({"RNA": adata_ref, "ATAC_GA": adata_query}, **params)


@dataclass
class AtacPreprocessConfig:
    batch_key: str = "batch"
    top_peaks: int = 30000
    promoter_filter_up: int = 2000
    promoter_filter_down: int = 500
    lsi_n_components: int = 50
    lsi_random_state: int = 42
    lsi_kwargs: Optional[Dict[str, Any]] = None
    ga_promoter_up: int = 2000
    ga_promoter_down: int = 0
    ga_include_gene_body: bool = False
    ga_weight_by_distance: bool = True
    ga_tss_decay_bp: int = 2000
    ga_promoter_priority: bool = True
    knn_neighbors: int = 50


@dataclass
class AtacPreprocessResult:
    atac: AnnData
    ga: AnnData
    lsi_key: str = "X_lsi"
    ga_layer: str = "ga"
    ga_smooth_layer: str = "ga_smooth"

    def assemble_joint_embedding(self, rep_key: str, modalities: Optional[Dict[str, AnnData]] = None) -> AnnData:
        payload = dict(modalities or {})
        # The aligned representation lives on the gene-activity AnnData, not the raw ATAC matrix.
        payload.setdefault("ATAC", self.ga)
        return assemble_joint_embedding(rep_key, payload)

    def nnz_pct(self, layer: Optional[str] = None) -> float:
        if layer is None:
            matrix = self.ga.layers.get(self.ga_layer, self.ga.X)
        elif layer == "X":
            matrix = self.ga.X
        elif layer in self.ga.layers:
            matrix = self.ga.layers[layer]
        else:
            raise KeyError(f"Layer '{layer}' not found in GA AnnData.")
        return _nnz_pct(matrix)


# --------------------------------------------------------------------------- #
# Shared utilities
# --------------------------------------------------------------------------- #


def _nnz_pct(X: np.ndarray | sp.spmatrix) -> float:
    if sp.issparse(X):
        return 100.0 * X.nnz / (X.shape[0] * X.shape[1])
    arr = np.asarray(X)
    return 100.0 * (arr != 0).sum() / arr.size


def build_atac_lsi(
    atac: AnnData,
    *,
    gtf_file: Path | str,
    batch_key: str,
    n_comps: int = 50,
    promoter_up: int = 2000,
    promoter_down: int = 500,
    top_peaks: int = 30000,
    random_state: int = 0,
    lsi_kwargs: Optional[Dict[str, Any]] = None,
) -> AnnData:
    if batch_key not in atac.obs:
        raise KeyError(f"Batch key '{batch_key}' not found in ATAC obs.")

    atac_filtered = remove_promoter_proximal_peaks(
        atac,
        gtf_file=gtf_file,
        promoter_up=promoter_up,
        promoter_down=promoter_down,
    )
    adata_top = find_variable_features(
        atac_filtered,
        batch_key=batch_key,
        topN=top_peaks,
        add_key="var_features",
        normalize_output=False,
    )
    counts = ensure_csr_f32(adata_top.X).copy()
    counts.data[:] = 1.0
    counts.eliminate_zeros()
    adata_top.X = counts
    adata_top.layers["counts"] = counts

    # tfidf_transform(adata_top, layer_out="tfidf")

    params = dict(lsi_kwargs or {})
    params.setdefault("layer", "counts")
    # lsi_transform now uses `add_key`; support legacy `key_added` for compatibility
    if "key_added" in params and "add_key" not in params:
        params["add_key"] = params.pop("key_added")
    params.setdefault("add_key", "X_lsi")
    drop_first = params.get("drop_first_component", True)
    params.setdefault("random_state", random_state)
    if "n_components" not in params:
        params["n_components"] = int(n_comps + (1 if drop_first else 0))
    # lsi_transform(adata_top, **params)
    add_iterative_lsi(adata_top, n_iter=1,               # no iterative re-selection
    per_cluster_union=False,
    sample_cells_pre=None,  # fit SVD on all cells
    **params)

    atac.obsm["X_lsi"] = adata_top.obsm["X_lsi"].astype(np.float32, copy=False)
    atac.uns["atac_lsi_meta"] = {
        "gtf_file": str(gtf_file),
        "batch_key": batch_key,
        "promoter_up": int(promoter_up),
        "promoter_down": int(promoter_down),
        "top_peaks_requested": int(top_peaks),
        "n_selected_peaks": int(adata_top.n_vars),
        "n_components": int(atac.obsm["X_lsi"].shape[1]),
        "drop_first_component": bool(drop_first),
        "random_state": int(random_state),
        "binarized_counts": True,
    }
    return atac


def assemble_joint_embedding(rep_key: str, modalities: Dict[str, AnnData]) -> AnnData:
    embeddings: List[np.ndarray] = []
    obs_frames: List[pd.DataFrame] = []
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

    Z_all = np.vstack(embeddings)
    obs_all = pd.concat(obs_frames, axis=0, sort=False)
    ad_all = AnnData(X=np.zeros((Z_all.shape[0], 1), dtype=np.float32))
    ad_all.obs = obs_all
    ad_all.obsm[rep_key] = Z_all
    return ad_all


def preprocess_atac(
    adata_atac: AnnData,
    gtf_file: Path | str,
    *,
    config: Optional[AtacPreprocessConfig] = None,
    verbose: bool = True,
) -> AtacPreprocessResult:
    cfg = config or AtacPreprocessConfig()
    build_atac_lsi(
        adata_atac,
        gtf_file=gtf_file,
        batch_key=cfg.batch_key,
        n_comps=cfg.lsi_n_components,
        promoter_up=cfg.promoter_filter_up,
        promoter_down=cfg.promoter_filter_down,
        top_peaks=cfg.top_peaks,
        random_state=cfg.lsi_random_state,
        lsi_kwargs=cfg.lsi_kwargs,
    )

    adata_query = annotate_gene_activity(
        adata_atac,
        gtf_file=gtf_file,
        promoter_up=cfg.ga_promoter_up,
        promoter_down=cfg.ga_promoter_down,
        include_gene_body=cfg.ga_include_gene_body,
        weight_by_distance=cfg.ga_weight_by_distance,
        tss_decay_bp=cfg.ga_tss_decay_bp,
        promoter_priority=cfg.ga_promoter_priority,
        verbose=verbose,
    )
    adata_query.layers["ga"] = ensure_csr_f32(adata_query.X)

    knn_smooth_ga_on_atac(
        adata_atac,
        adata_query,
        n_neighbors=cfg.knn_neighbors,
    )
    return AtacPreprocessResult(atac=adata_atac, ga=adata_query)


# --------------------------------------------------------------------------- #
# Co-embedding utilities
# --------------------------------------------------------------------------- #


def normalize_log1p(adata: AnnData, layer_in: Optional[str], layer_out: str, target_sum: float = 1e4) -> None:
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


def prepare_transfer_layers(
    adata_ref: AnnData,
    adata_query: AnnData,
    *,
    ref_layer: str = "ref_log1p",
    query_layer: str = "query_log1p",
    target_sum: float = 1e4,
) -> Tuple[str, str]:
    sc.pp.normalize_total(adata_ref, target_sum=target_sum)
    sc.pp.log1p(adata_ref)
    adata_ref.layers[ref_layer] = ensure_csr_f32(adata_ref.X).copy()

    sc.pp.normalize_total(adata_query, target_sum=target_sum)
    sc.pp.log1p(adata_query)
    adata_query.layers[query_layer] = ensure_csr_f32(adata_query.X).copy()

    return ref_layer, query_layer


def joint_hvgs(
    adata_ref: AnnData,
    adata_query: AnnData,
    *,
    n_top: int,
    ref_norm: str,
    query_norm: str,
    batch_key: Optional[str],
) -> pd.Index:
    normalize_log1p(adata_ref, layer_in=None, layer_out=ref_norm)
    ga_layer = "ga_smooth" if "ga_smooth" in adata_query.layers else ("ga" if "ga" in adata_query.layers else None)
    normalize_log1p(adata_query, layer_in=ga_layer, layer_out=query_norm)

    use_batch = batch_key if (batch_key and batch_key in adata_ref.obs.columns) else None
    sc.pp.highly_variable_genes(
        adata_ref,
        flavor="seurat_v3",
        n_top_genes=min(n_top, adata_ref.n_vars),
        layer=ref_norm,
        batch_key=use_batch,
    )
    sc.pp.highly_variable_genes(
        adata_query,
        flavor="seurat_v3",
        n_top_genes=min(n_top, adata_query.n_vars),
        layer=query_norm,
        batch_key=None,
    )
    ref_hv = set(adata_ref.var_names[adata_ref.var["highly_variable"].values])
    query_hv = set(adata_query.var_names[adata_query.var["highly_variable"].values])
    shared = set(adata_ref.var_names).intersection(adata_query.var_names)
    genes = sorted(shared.intersection(ref_hv.union(query_hv)))
    if len(genes) < 500:
        raise ValueError("Too few shared HVGs; check name harmonization or GA quality.")
    return pd.Index(genes)


def shared_pca_from_genes(
    adata_ref: AnnData,
    adata_query: AnnData,
    genes: Sequence[str],
    *,
    n_comps: int,
    ref_norm: str,
    query_norm: str,
) -> pd.Index:
    normalize_log1p(adata_ref, layer_in=None, layer_out=ref_norm)
    ga_layer = "ga_smooth" if "ga_smooth" in adata_query.layers else ("ga" if "ga" in adata_query.layers else None)
    normalize_log1p(adata_query, layer_in=ga_layer, layer_out=query_norm)

    genes_idx = pd.Index(genes)
    if genes_idx.has_duplicates:
        genes_idx = genes_idx.drop_duplicates()

    def _first_pos(names: pd.Index) -> Dict[Any, int]:
        pos: Dict[Any, int] = {}
        for i, name in enumerate(names):
            pos.setdefault(name, i)
        return pos

    rna_pos = _first_pos(adata_ref.var_names)
    ga_pos = _first_pos(adata_query.var_names)
    shared = [g for g in genes_idx if g in rna_pos and g in ga_pos]
    if not shared:
        raise ValueError("No shared genes found between RNA and GA inputs.")
    if len(shared) < len(genes_idx):
        warnings.warn(
            "Some genes were missing in one modality or duplicated; using first occurrence only.",
            RuntimeWarning,
        )

    rna_idx = np.array([rna_pos[g] for g in shared], dtype=np.int64)
    ga_idx = np.array([ga_pos[g] for g in shared], dtype=np.int64)

    Xr = ensure_csr_f32(adata_ref[:, rna_idx].layers[ref_norm]).toarray()
    Xg = ensure_csr_f32(adata_query[:, ga_idx].layers[query_norm]).toarray()
    shared = list(shared)

    nonfinite_r = ~np.isfinite(Xr)
    nonfinite_g = ~np.isfinite(Xg)
    if nonfinite_r.any() or nonfinite_g.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mu_nf = np.nanmean(np.where(np.isfinite(Xr), Xr, np.nan), axis=0)
        bad_gene = ~np.isfinite(mu_nf)
        if bad_gene.any():
            dropped = int(bad_gene.sum())
            warnings.warn(
                f"Dropping {dropped} shared genes with no finite RNA values.",
                RuntimeWarning,
            )
            keep = ~bad_gene
            Xr = Xr[:, keep]
            Xg = Xg[:, keep]
            rna_idx = rna_idx[keep]
            ga_idx = ga_idx[keep]
            shared = [g for g, k in zip(shared, keep) if k]
            mu_nf = mu_nf[keep]
            nonfinite_r = nonfinite_r[:, keep]
            nonfinite_g = nonfinite_g[:, keep]
        if not shared:
            raise ValueError("No shared genes with finite RNA values after filtering.")
        if nonfinite_r.any():
            rows, cols = np.where(nonfinite_r)
            Xr[rows, cols] = mu_nf[cols]
        if nonfinite_g.any():
            rows, cols = np.where(nonfinite_g)
            Xg[rows, cols] = mu_nf[cols]
        warnings.warn(
            "Non-finite values found in shared RNA/GA layers; imputing with RNA means.",
            RuntimeWarning,
        )

    mu = Xr.mean(axis=0, dtype=np.float64)
    var = (Xr**2).mean(axis=0, dtype=np.float64) - mu**2
    sd = np.sqrt(np.maximum(var, 1e-8))
    Xr_z = np.clip((Xr - mu) / sd, -10, 10)
    Xg_z = np.clip((Xg - mu) / sd, -10, 10)

    k = int(min(n_comps, Xr_z.shape[1], max(2, Xr_z.shape[0] - 1)))
    pca = PCA(n_components=k, svd_solver="full", random_state=0)
    Zr = pca.fit_transform(Xr_z).astype(np.float32)
    Zg = pca.transform(Xg_z).astype(np.float32)

    adata_ref.obsm["X_pca_shared"] = Zr
    adata_query.obsm["X_pca_shared"] = Zg

    loadings = pca.components_.T.astype(np.float32, copy=False)
    L_full = np.zeros((adata_ref.n_vars, loadings.shape[1]), np.float32)
    L_full[rna_idx] = loadings
    adata_ref.varm["PCs_shared"] = L_full

    adata_ref.uns["shared_pca_meta"] = {
        "genes_used": list(shared),
        "n_comps": int(loadings.shape[1]),
        "explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float32).tolist(),
        "norm_layers": {"rna": ref_norm, "ga": query_norm},
    }
    return pd.Index(shared)


def _compute_coral_alignment(Zr: np.ndarray, Zg: np.ndarray) -> np.ndarray:
    mu_r = Zr.mean(axis=0, keepdims=True)
    mu_g = Zg.mean(axis=0, keepdims=True)
    Xr0 = Zr - mu_r
    Xg0 = Zg - mu_g
    eps = 1e-4
    Cr = (Xr0.T @ Xr0) / max(len(Zr) - 1, 1) + eps * np.eye(Zr.shape[1], dtype=np.float32)
    Cg = (Xg0.T @ Xg0) / max(len(Zg) - 1, 1) + eps * np.eye(Zg.shape[1], dtype=np.float32)
    Ur, Sr, Vr = svd(Cr, full_matrices=False)
    Ug, Sg, Vg = svd(Cg, full_matrices=False)
    Cr_sqrt = (Ur * np.sqrt(Sr)) @ Vr
    Cg_isqrt = (Ug * (1.0 / np.sqrt(Sg))) @ Vg
    return (Xg0 @ Cg_isqrt) @ Cr_sqrt + mu_r


# --------------------------------------------------------------------------- #
# Optimal transport helpers
# --------------------------------------------------------------------------- #


def _torch_device(use_gpu: bool, gpu_device: int) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_device}")
    return torch.device("cpu")


def _to_torch(
    x: np.ndarray,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype).contiguous()
    try:
        return torch.as_tensor(x, device=device, dtype=dtype).contiguous()
    except TypeError:
        return torch.as_tensor(x, dtype=dtype).to(device=device).contiguous()


# --------------------------------------------------------------------------- #
# Alignment & label transfer
# --------------------------------------------------------------------------- #
def align_shared_pca(
    adata_ref: AnnData,
    adata_query: AnnData,
    *,
    rep_in: str,
    rep_out: str,
    use_ot: bool,
    ot_reg: float,
    ot_reg_m: float,
    ot_cost_clip_q: float,
    ot_clip_big: float,
    ot_backend: str,
    ot_use_gpu: bool,
    ot_gpu_device: int,
    ot_topk: int,
    ot_chunk_size: Optional[int],
    verbose: bool,
) -> str:
    Zr, Zg = adata_ref.obsm[rep_in], adata_query.obsm[rep_in]
    if sp.issparse(Zr):
        Zr = Zr.toarray()
    if sp.issparse(Zg):
        Zg = Zg.toarray()
    Zr = np.asarray(Zr, dtype=np.float32, order="C")
    Zg = np.asarray(Zg, dtype=np.float32, order="C")

    mu_r = Zr.mean(axis=0, keepdims=True)
    sd_r = Zr.std(axis=0, keepdims=True)

    Zg_coral = _compute_coral_alignment(Zr, Zg)

    if not use_ot:
        adata_ref.obsm[rep_out] = Zr
        adata_query.obsm[rep_out] = Zg_coral.astype(np.float32, copy=False)
        adata_query.uns.pop("_ot_alignment", None)
        if verbose:
            print(f"[CORAL] '{rep_out}' updated using CORAL alignment only; OT skipped.")
        return rep_out

    Zg_aligned, transport = compute_ot_alignment(
        Zg_coral,
        Zr,
        reg=ot_reg,
        reg_m=ot_reg_m,
        cost_clip_q=ot_cost_clip_q,
        clip_big=ot_clip_big,
        backend=ot_backend,
        use_gpu=ot_use_gpu,
        gpu_device=ot_gpu_device,
        transport_topk=ot_topk,
        chunk_size=ot_chunk_size,
    )

    adata_ref.obsm[rep_out] = Zr
    adata_query.obsm[rep_out] = Zg_aligned.astype(np.float32, copy=False)

    ot_meta: Dict[str, Any] = {
        "indices": transport["indices"].astype(np.int32, copy=False),
        "weights": transport["weights"].astype(np.float32, copy=False),
        "residual": (
            transport.get("residual").astype(np.float32, copy=False)
            if transport.get("residual") is not None
            else None
        ),
        "rna_obs": np.asarray(adata_ref.obs_names, dtype=object),
        "params": {
            "reg": float(ot_reg),
            "reg_m": float(ot_reg_m),
            "cost_clip_q": None if ot_cost_clip_q is None else float(ot_cost_clip_q),
            "clip_big": float(ot_clip_big),
            "backend": ot_backend,
            "topk": int(ot_topk),
        },
        "center": mu_r.astype(np.float32, copy=False),
        "scale": sd_r.astype(np.float32, copy=False),
    }
    adata_query.uns["_ot_alignment"] = ot_meta

    if verbose:
        print(f"[OT] '{rep_out}' updated with query→reference barycentric OT (top-{ot_topk} transport stored).")
    return rep_out


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
        raise RuntimeError("OT transport weights missing; run align_shared_pca with OT first.")

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
    )

    adata_query.obs["pred_cell_type"] = res["pred_labels"]
    adata_query.obs["pred_confidence"] = res["confidence"]
    adata_query.obs["pred_support"] = res["support"]
    adata_query.obs["pred_unknown_prob"] = res["unknown_prob"]
    adata_query.obsm["pred_cell_type_proba"] = res["proba"]
    adata_query.uns["pred_cell_type_classes"] = list(res["classes"])
    return adata_query


def build_aligned_coembedding(
    adata_ref: AnnData,
    adata_query: AnnData,
    *,
    config: Optional[CoEmbeddingConfig] = None,
    genes: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> CoEmbeddingResult:
    cfg = config or CoEmbeddingConfig()
    if genes is None:
        genes_idx = joint_hvgs(
            adata_ref,
            adata_query,
            n_top=cfg.n_top_genes,
            ref_norm=cfg.ref_norm_layer,
            query_norm=cfg.query_norm_layer,
            batch_key=cfg.batch_key,
        )
    else:
        genes_idx = pd.Index(genes)

    genes_idx = shared_pca_from_genes(
        adata_ref,
        adata_query,
        genes_idx,
        n_comps=cfg.n_components,
        ref_norm=cfg.ref_norm_layer,
        query_norm=cfg.query_norm_layer,
    )
    rep_key = align_shared_pca(
        adata_ref,
        adata_query,
        rep_in=cfg.rep_in,
        rep_out=cfg.rep_out,
        use_ot=cfg.use_optimal_transport,
        ot_reg=cfg.ot_reg,
        ot_reg_m=cfg.ot_reg_m,
        ot_cost_clip_q=cfg.ot_cost_clip_q,
        ot_clip_big=cfg.ot_clip_big,
        ot_backend=cfg.ot_backend,
        ot_use_gpu=cfg.ot_use_gpu,
        ot_gpu_device=cfg.ot_gpu_device,
        ot_topk=cfg.ot_topk,
        ot_chunk_size=cfg.ot_chunk_size,
        verbose=verbose,
    )
    return CoEmbeddingResult(embedding_key=rep_key, genes=genes_idx, config=cfg)


def transfer_labels(
    adata_ref: AnnData,
    adata_query: AnnData,
    *,
    ref_label_key: str,
    query_label_key: str = "pred_cell_type",
    pred_conf_key: str = "pred_confidence",
    genes: Optional[Sequence[str] | str] = None,
    min_conf: float = 0.25,
    coembed_config: Optional[CoEmbeddingConfig] = None,
    batch_key: str = "modality",
    out_key: str = "X_pca_shared_aligned",
    integrate_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Tuple[CoEmbeddingResult, AnnData, Dict[str, float | int]]:
    """
    Build an aligned RNA/ATAC co-embedding, transfer labels, and integrate jointly with scBIOT.
    `ref_label_key` is read from the reference; predictions are stored on the query under
    `query_label_key` (default: "pred_cell_type") with confidence in `pred_conf_key`.
    """
    cfg = replace(coembed_config) if coembed_config is not None else CoEmbeddingConfig()
    ref_layer, query_layer = prepare_transfer_layers(adata_ref, adata_query)
    cfg.ref_norm_layer = ref_layer
    cfg.query_norm_layer = query_layer
    if not cfg.use_optimal_transport:
        raise ValueError("transfer_labels requires use_optimal_transport=True for OT label transfer.")

    genes_resolved: Optional[Sequence[str]]
    if isinstance(genes, str):
        if genes.lower() != "shared":
            raise ValueError("genes must be a sequence or the string 'shared'.")
        genes_resolved = adata_ref.var_names.intersection(adata_query.var_names)
        if len(genes_resolved) == 0:
            raise ValueError("No shared genes between reference and query inputs.")
        if verbose:
            print(f"[coembed] shared genes={len(genes_resolved)}")
    else:
        genes_resolved = genes

    coembed = build_aligned_coembedding(
        adata_ref,
        adata_query,
        genes=genes_resolved,
        config=cfg,
        verbose=verbose,
    )

    label_transfer_shared_pca(
        adata_ref,
        adata_query,
        label_key=ref_label_key,
        use_rep=coembed.embedding_key,
        min_conf=min_conf,
    )
    if query_label_key != "pred_cell_type" and query_label_key not in adata_query.obs:
        adata_query.obs[query_label_key] = adata_query.obs["pred_cell_type"]
    if pred_conf_key != "pred_confidence" and pred_conf_key not in adata_query.obs:
        adata_query.obs[pred_conf_key] = adata_query.obs["pred_confidence"]

    adata_joint = assemble_joint_embedding(
        coembed.embedding_key,
        {"Reference": adata_ref, "Query": adata_query},
    )
    if batch_key not in adata_joint.obs:
        adata_joint.obs[batch_key] = adata_joint.obs["modality"]   

    
    return coembed, adata_joint
