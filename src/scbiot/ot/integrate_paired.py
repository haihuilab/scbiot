# integrate_paired.py
# ==================
# Paired RNA+ATAC (Multiome) joint coembedding via paired-aware OT barycentric projection
# Improvements vs your original:
#   1) Optional paired pre-alignment (Procrustes or ridge) to make PCA vs LSI commensurate
#   2) Safer unbalanced OT barycentric projection with row-mass fallback (avoids NaNs / explosions)
#   3) Weight normalization (w_base + w_view = 1)
#   4) Lower default "full OT" threshold; auto-switch to approximate OT earlier
#   5) Adds paired disagreement metrics

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Literal

import numpy as np
import torch
from sklearn.metrics import pairwise_distances_argmin
from sklearn.preprocessing import LabelEncoder

try:
    import ot  # type: ignore

    _POT_AVAILABLE = True
except ModuleNotFoundError:
    ot = None  # type: ignore
    _POT_AVAILABLE = False

from ..utils.ot_helpers import (
    _FAISS_AVAIL,
    _as_nd_f32_c,
    _faiss_knn_search,
    _knn_overlap,
    _minikm_centers,
    _neighbor_batch_entropy_per_cell,
    _sinkhorn_uot_torch,
    _to_torch,
    _torch_device,
)


# -------------------- Utilities --------------------


def _zscore(X: np.ndarray) -> np.ndarray:
    X = _as_nd_f32_c(X)
    mean = X.mean(0, keepdims=True)
    std = X.std(0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (X - mean) / std


def _normalize_weights(w_base: float, w_view: float) -> Tuple[float, float]:
    wb = float(w_base)
    wv = float(w_view)
    s = wb + wv
    if not np.isfinite(s) or s <= 0.0:
        raise ValueError("w_base + w_view must be a finite number > 0.")
    return wb / s, wv / s


def _trim_components(
    X_base: np.ndarray,
    X_view: np.ndarray,
    n_components: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, int]:
    d_base = int(X_base.shape[1])
    d_view = int(X_view.shape[1])
    n_max = int(min(d_base, d_view))
    if n_max <= 0:
        raise ValueError("Embeddings must have at least 1 dimension.")

    if n_components is None:
        n_use = n_max
    else:
        n_use = int(n_components)
        if n_use <= 0:
            raise ValueError("n_components must be a positive integer.")
        n_use = min(n_use, n_max)

    return X_base[:, :n_use], X_view[:, :n_use], n_use


def _sqeuclidean_cost(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = _as_nd_f32_c(A)
    B = _as_nd_f32_c(B)
    A2 = (A * A).sum(1, keepdims=True)
    B2 = (B * B).sum(1, keepdims=True).T
    C = A2 + B2 - 2.0 * (A @ B.T)
    return np.maximum(C, 0.0).astype(np.float32, copy=False)


def _paired_cost_with_prior(
    X_view_z: np.ndarray,
    X_base_z: np.ndarray,
    diag_mass: float,
    prior_strength: float,
) -> np.ndarray:
    """
    Cost with a soft diagonal prior implemented as a log-prior bonus.

    Implementation detail:
    - We keep every entry positive by adding a small "off" mass everywhere.
      This avoids log(0) and makes the prior compatible with Sinkhorn.
    """
    C = _sqeuclidean_cost(X_view_z, X_base_z)
    prior_strength = float(prior_strength)
    diag_mass = float(diag_mass)

    if prior_strength <= 0.0 or diag_mass <= 0.0:
        return C

    if not (0.0 <= diag_mass <= 1.0):
        raise ValueError("diag_mass must be within [0, 1].")
    if X_view_z.shape[0] != X_base_z.shape[0]:
        raise ValueError("paired OT requires view/base to have the same number of rows.")

    n = int(X_view_z.shape[0])
    if n <= 0:
        return C

    # Keep support everywhere:
    off = (1.0 - diag_mass) / (n * n)
    diag = (diag_mass / n) + off
    log_off = np.log(off + 1e-12)
    log_diag = np.log(diag + 1e-12)

    # baseline shift
    C = C - prior_strength * log_off
    # diagonal bonus
    diag_adjust = -prior_strength * (log_diag - log_off)
    idx = np.arange(n)
    C[idx, idx] += diag_adjust
    return C


def _uot_from_cost(
    M: np.ndarray,
    reg: float,
    reg_m: float,
    backend: str,
    use_gpu: bool,
    gpu_device: int,
) -> np.ndarray:
    backend = str(backend).lower()
    if backend == "ot":
        backend = "pot"
    if backend not in {"pot", "torch"}:
        raise ValueError("ot_backend must be 'torch' or 'pot'.")

    N, K = int(M.shape[0]), int(M.shape[1])
    a = np.full(N, 1.0 / max(N, 1), dtype=np.float64)
    b = np.full(K, 1.0 / max(K, 1), dtype=np.float64)

    if backend == "pot":
        if not _POT_AVAILABLE or ot is None:
            raise ModuleNotFoundError(
                "POT is required when ot_backend='pot'. Install it via `pip install POT`."
            )
        M64 = np.asarray(M, dtype=np.float64, order="C")
        try:
            T = ot.unbalanced.sinkhorn_unbalanced(
                a,
                b,
                M64,
                reg,
                reg_m,
                method="sinkhorn_stabilized",
                numItermax=1000,
                stopThr=1e-6,
                verbose=False,
            )
        except TypeError:
            T = ot.unbalanced.sinkhorn_unbalanced(
                a,
                b,
                M64,
                reg,
                reg_m,
                numItermax=1000,
                stopThr=1e-6,
            )
        return np.asarray(T, dtype=np.float32, order="C")

    device = _torch_device(use_gpu, gpu_device)
    dtype = torch.float32
    a_t = torch.full((N,), 1.0 / max(N, 1), dtype=dtype, device=device)
    b_t = torch.full((K,), 1.0 / max(K, 1), dtype=dtype, device=device)
    M_t = _to_torch(np.asarray(M, dtype=np.float32, order="C"), device=device, dtype=dtype)

    # heuristic mapping between reg_m and tau (consistent with your helper)
    tau = float(reg_m / (reg_m + reg)) if reg_m > 0 else 0.0
    u, v, Kt = _sinkhorn_uot_torch(M_t, a_t, b_t, eps=float(reg), tau=tau)
    T_t = (u.view(-1, 1) * Kt) * v.view(1, -1)
    return T_t.detach().cpu().numpy().astype(np.float32, copy=False)


def _barycentric_project(T: np.ndarray, X_base: np.ndarray) -> np.ndarray:
    row_sum = T.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum > 0, row_sum, 1e-12)
    return (T @ X_base) / row_sum


def _compute_ot_stats(T: np.ndarray, C: np.ndarray) -> Dict[str, float]:
    row_sum = T.sum(axis=1, keepdims=True) + 1e-12
    row_cost = (T * C).sum(axis=1) / row_sum[:, 0]
    row_norm = T / row_sum
    row_entropy = -(row_norm * np.log(row_norm + 1e-12)).sum(axis=1)
    return {
        "ot_cost_mean": float(row_cost.mean()) if row_cost.size else 0.0,
        "ot_cost_p50": float(np.percentile(row_cost, 50)) if row_cost.size else 0.0,
        "ot_cost_p90": float(np.percentile(row_cost, 90)) if row_cost.size else 0.0,
        "transport_entropy": float(row_entropy.mean()) if row_entropy.size else 0.0,
    }


def _group_centroids(
    X_base_z: np.ndarray,
    X_view_z: np.ndarray,
    group_ids: np.ndarray,
    n_groups: int,
) -> Tuple[np.ndarray, np.ndarray]:
    d_base = int(X_base_z.shape[1])
    d_view = int(X_view_z.shape[1])
    base_centroids = np.zeros((n_groups, d_base), dtype=np.float32)
    view_centroids = np.zeros((n_groups, d_view), dtype=np.float32)

    for g in range(n_groups):
        mask = group_ids == g
        if np.any(mask):
            base_centroids[g] = X_base_z[mask].mean(axis=0)
            view_centroids[g] = X_view_z[mask].mean(axis=0)
        else:
            base_centroids[g] = X_base_z.mean(axis=0)
            view_centroids[g] = X_view_z.mean(axis=0)
    return base_centroids, view_centroids


def _assign_to_centroids(
    X: np.ndarray,
    centroids: np.ndarray,
    use_gpu: bool,
    gpu_device: int,
) -> np.ndarray:
    if _FAISS_AVAIL:
        _, nn_idx = _faiss_knn_search(X, centroids, 1, use_gpu=use_gpu, device=gpu_device)
        return nn_idx.ravel().astype(np.int32, copy=False)
    return pairwise_distances_argmin(X, centroids, metric="euclidean").astype(np.int32, copy=False)


def _procrustes_align(X_view_z: np.ndarray, X_base_z: np.ndarray) -> np.ndarray:
    """
    Orthogonal Procrustes alignment: find rotation R s.t. X_view_z @ R ~ X_base_z.
    Assumes X_view_z and X_base_z are already centered/scaled similarly (z-scored).
    """
    # Solve argmax_R tr(R^T X_view^T X_base), R orthogonal
    M = X_view_z.T @ X_base_z
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    return (X_view_z @ R).astype(np.float32, copy=False)


def _ridge_align(X_view_z: np.ndarray, X_base_z: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Ridge regression mapping: W = argmin ||X_view W - X_base||^2 + alpha||W||^2
    """
    alpha = float(alpha)
    if alpha < 0:
        raise ValueError("ridge_alpha must be >= 0.")
    d = int(X_view_z.shape[1])
    A = (X_view_z.T @ X_view_z) + (alpha * np.eye(d, dtype=np.float32))
    B = (X_view_z.T @ X_base_z).astype(np.float32, copy=False)
    W = np.linalg.solve(A.astype(np.float32, copy=False), B).astype(np.float32, copy=False)
    return (X_view_z @ W).astype(np.float32, copy=False)


# -------------------- Main integration --------------------


def integrate_paired(
    adata: Any,
    obsm_key: str = "X_pca",
    view_key: str = "X_lsi",
    batch_key: str = "batch",
    out_key: str = "scBIOT",
    approximate_ot: bool = False,
    centroid_ot: bool = False,
    K_ref: int = 1024,
    K_batch: int = 512,
    reg: float = 0.028,
    reg_m: float = 0.40,
    prior_strength: float = 1.0,
    diag_mass: float = 0.2,
    w_base: float = 0.5,
    w_view: float = 0.5,
    n_components: Optional[int] = None,
    random_state: int = 0,
    verbose: bool = True,
    use_gpu: bool = True,
    gpu_device: int = 0,
    ot_backend: str = "torch",
    # ---- paired improvements ----
    prealign: Optional[Literal["procrustes", "ridge", "none"]] = "procrustes",
    ridge_alpha: float = 1.0,
    min_row_mass: float = 1e-3,
    max_obs_full_ot: int = 10_000,
) -> Tuple[Any, Dict[str, float | int | str | Dict[str, float]]]:
    """
    Integrate paired multiome embeddings via paired-aware OT barycentric projection.

    IMPORTANT:
    - If view_key is ATAC LSI and obsm_key is RNA PCA, they are not commensurate.
      Use `prealign="procrustes"` (default) or `prealign="ridge"` to make cross-modal
      distances meaningful before OT.

    Parameters
    ----------
    prealign:
        "procrustes" (default): orthogonal alignment of view->base using paired cells.
        "ridge": ridge regression mapping view->base.
        "none": no prealignment (only recommended if view is already in base coordinates,
                e.g. ATAC gene-activity projected into RNA PCA).
    min_row_mass:
        If unbalanced OT assigns too little mass to a row, barycentric projection becomes unstable.
        For paired data, we fall back to identity (use base embedding) for those cells/rows.
    max_obs_full_ot:
        Maximum number of observations to allow dense full OT (n x n) before auto-approximate.
    """
    if approximate_ot and centroid_ot:
        raise ValueError("integrate_paired received both approximate_ot and centroid_ot; enable only one.")

    if obsm_key not in adata.obsm:
        raise ValueError(f"obsm_key '{obsm_key}' not found in adata.obsm.")
    if view_key not in adata.obsm:
        raise ValueError(f"view_key '{view_key}' not found in adata.obsm.")
    if view_key == out_key:
        raise ValueError("view_key must not match out_key.")
    if batch_key not in adata.obs:
        raise ValueError(f"batch_key '{batch_key}' not found in adata.obs.")

    w_base, w_view = _normalize_weights(w_base, w_view)

    X_base = _as_nd_f32_c(adata.obsm[obsm_key])
    X_view = _as_nd_f32_c(adata.obsm[view_key])

    if X_base.shape[0] != adata.n_obs:
        raise ValueError(f"obsm_key '{obsm_key}' has {X_base.shape[0]} rows; expected {adata.n_obs}.")
    if X_view.shape[0] != adata.n_obs:
        raise ValueError(f"view_key '{view_key}' has {X_view.shape[0]} rows; expected {adata.n_obs}.")

    n_obs = int(X_base.shape[0])

    auto_approximate = False
    max_obs = int(max_obs_full_ot)
    if not approximate_ot and not centroid_ot and n_obs > max_obs:
        approximate_ot = True
        auto_approximate = True
        if verbose:
            print(f"[paired] n_obs={n_obs} exceeds {max_obs}; switching to approximate_ot=True")

    X_base, X_view, n_components_eff = _trim_components(X_base, X_view, n_components)
    X_base_z = _zscore(X_base)
    X_view_z = _zscore(X_view)

    # ---- paired prealignment (critical when base=PCA and view=LSI) ----
    palign = "none" if prealign is None else str(prealign).lower()
    if palign not in {"procrustes", "ridge", "none"}:
        raise ValueError("prealign must be one of: 'procrustes', 'ridge', 'none'.")

    if palign == "procrustes":
        X_view_z = _procrustes_align(X_view_z, X_base_z)
    elif palign == "ridge":
        X_view_z = _ridge_align(X_view_z, X_base_z, alpha=ridge_alpha)

    # ---- OT backend ----
    ot_backend = str(ot_backend).lower()
    if ot_backend == "ot":
        ot_backend = "pot"

    # ---- choose OT path ----
    if approximate_ot:
        n_obs2 = int(X_base_z.shape[0])
        # heuristic: ~O(sqrt(n)) centroids, clipped by K_ref/K_batch
        K_base = int(min(K_ref, max(16, 2 * np.sqrt(max(n_obs2, 1)))))
        K_view = int(min(K_batch, max(8, 2 * np.sqrt(max(n_obs2, 1)))))
        K_eff = int(max(1, min(K_base, K_view, n_obs2)))

        base_centroids = _minikm_centers(
            X_base_z,
            n_clusters=K_eff,
            seed=int(random_state) + 17,
            use_gpu=use_gpu,
            device=gpu_device,
        )
        group_ids = _assign_to_centroids(X_base_z, base_centroids, use_gpu=use_gpu, gpu_device=gpu_device)
        _, view_centroids = _group_centroids(X_base_z, X_view_z, group_ids, K_eff)

    elif centroid_ot:
        b_raw = adata.obs[batch_key].to_numpy()
        le = LabelEncoder()
        group_ids = le.fit_transform(b_raw).astype(np.int32, copy=False)
        n_groups = int(group_ids.max()) + 1 if len(group_ids) else 1
        base_centroids, view_centroids = _group_centroids(X_base_z, X_view_z, group_ids, n_groups)

    else:
        base_centroids = X_base_z
        view_centroids = X_view_z
        group_ids = None

    # ---- paired OT with diagonal prior ----
    C = _paired_cost_with_prior(
        view_centroids, base_centroids, diag_mass=float(diag_mass), prior_strength=float(prior_strength)
    )
    # --- Numerical stabilization: shift + scale cost ---
    C = C - float(C.min())                    # min -> 0
    scale = float(np.median(C))               # robust scale
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = float(C.mean()) if float(C.mean()) > 1e-8 else 1.0
    C = (C / scale).astype(np.float32, copy=False)
    
    print("[debug] C min/median/p90/max:",
      float(C.min()),
      float(np.median(C)),
      float(np.percentile(C, 90)),
      float(C.max()))
    print("[debug] suggested eps ~ 0.05*median(C) =", float(0.05*np.median(C))) 
    T = _uot_from_cost(
        C, reg=float(reg), reg_m=float(reg_m), backend=ot_backend, use_gpu=use_gpu, gpu_device=gpu_device
    )
    row_mass = T.sum(axis=1)
    expected = 1.0 / max(T.shape[0], 1)
    row_mass = T.sum(axis=1)
    print("[debug] row_mass min/median/max:", row_mass.min(), np.median(row_mass), row_mass.max())
    print("[debug] expected 1/N:", expected)
    print("[debug] frac < 0.1*(1/N):", float((row_mass < 0.1*expected).mean()))

    # barycentric projection (view->base)
    view_to_base_centroids = _barycentric_project(T, base_centroids).astype(np.float32, copy=False)

    # broadcast to cells + row-mass fallback
    row_mass_cent = T.sum(axis=1).astype(np.float32, copy=False)

    if group_ids is None:
        X_view_to_base = view_to_base_centroids
        row_mass_cell = row_mass_cent
    else:
        X_view_to_base = view_to_base_centroids[group_ids]
        row_mass_cell = row_mass_cent[group_ids]

    # row_mass_cell is the OT mass per cell-row in the plan
    row_mass_cell = row_mass_cell.astype(np.float32, copy=False)

    # expected mass per row under uniform marginals
    expected = 1.0 / max(len(row_mass_cent), 1)  # centroids or cells, depending on path

    # interpret min_row_mass as a *fraction of expected* if <= 1, else as absolute
    min_rm = float(min_row_mass)
    thr = (min_rm * expected) if min_rm <= 1.0 else min_rm

    low = row_mass_cell < thr
    fallback_frac = float(low.mean())
    # store this so you can see if fallback is actually happening
    # (add into metrics later)
    # metrics["fallback_frac"] = fallback_frac
    # metrics["row_mass_expected"] = float(expected)
    # metrics["row_mass_thr"] = float(thr)

    if np.any(low):
        X_view_to_base = X_view_to_base.copy()
        X_view_to_base[low] = X_base_z[low]

    # fused joint embedding
    X_joint = (w_base * X_base_z + w_view * X_view_to_base).astype(np.float32, copy=False)
    adata.obsm[out_key] = X_joint

    # ---- metrics ----
    metrics: Dict[str, float | int | str | Dict[str, float]] = {
        "preset": "paired",
        "n_obs": int(X_joint.shape[0]),
        "n_components": int(n_components_eff),
        "base_key": str(obsm_key),
        "view_key": str(view_key),
        "out_key": str(out_key),
        "approximate_ot": bool(approximate_ot),
        "centroid_ot": bool(centroid_ot),
        "prealign": str(palign),
        "fusion_weights": {"w_base": float(w_base), "w_view": float(w_view)},
        "diag_mass_used": float(diag_mass),
        "prior_strength": float(prior_strength),
        "min_row_mass": float(min_rm),
        "max_obs_full_ot": int(max_obs),
    }
    if auto_approximate:
        metrics["auto_approximate_ot"] = True

    metrics.update(_compute_ot_stats(T, C))

    # paired disagreement (cell-level)
    d = np.linalg.norm(X_view_to_base - X_base_z, axis=1)
    metrics["paired_disagreement_mean"] = float(d.mean()) if d.size else 0.0
    metrics["paired_disagreement_p90"] = float(np.percentile(d, 90)) if d.size else 0.0

    # batch mixing diagnostic (still useful if batch_key is sample)
    b_raw = adata.obs[batch_key].to_numpy()
    le = LabelEncoder()
    b = le.fit_transform(b_raw).astype(np.int32, copy=False)
    k_eval = int(min(15, max(2, X_joint.shape[0] - 1)))
    _, batch_entropy_mean = _neighbor_batch_entropy_per_cell(
        X_joint, b, k=k_eval, use_gpu=use_gpu, device=gpu_device
    )
    metrics["batch_entropy_per_cell_mean"] = float(batch_entropy_mean)

    # knn overlap with base embedding (small n only)
    if X_joint.shape[0] <= 20_000:
        metrics["knn_overlap"] = float(
            _knn_overlap(X_base_z, X_joint, k=min(30, k_eval), use_gpu=use_gpu, device=gpu_device)
        )

    if verbose:
        path = "full"
        if centroid_ot:
            path = "centroid"
        elif approximate_ot:
            path = "approximate"
        print(
            f"[paired] path={path} prealign={palign} n={X_joint.shape[0]} d={X_joint.shape[1]} "
            f"w=({w_base:.3f},{w_view:.3f})"
        )

    return adata, metrics