# =================== paste-and-run: OTIntegration_rare_aware_sup.py (rare-safe + supervised option) ===================
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Sequence

import numpy as np
import torch
import scipy.sparse as sp
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from .torch_utils import _sinkhorn_uot_torch, _to_torch, _torch_device
try:  # POT (only used if you flip ot_backend="pot", otherwise unused)
    import ot  # type: ignore
    _POT_AVAILABLE = True
except ModuleNotFoundError:
    ot = None  # type: ignore
    _POT_AVAILABLE = False
from scipy.cluster.vq import kmeans2

# -------------------- Optional FAISS backend --------------------
try:
    import faiss
    _FAISS_AVAIL = True
    try:
        _FAISS_GPU = bool(getattr(faiss, "get_num_gpus", lambda: 0)() > 0 and hasattr(faiss, "StandardGpuResources"))
    except Exception:
        _FAISS_GPU = False
except Exception:
    _FAISS_AVAIL, _FAISS_GPU = False, False
    faiss = None  # type: ignore

# Shared FAISS resources keyed by device id.
_FAISS_GPU_RESOURCES: Dict[int, "faiss.StandardGpuResources"] = {}

# -------------------- Utilities --------------------

def _as_nd_f32_c(a: np.ndarray) -> np.ndarray:
    """Return 2D float32 C-contiguous numpy array; densifies sparse/pandas inputs."""
    if hasattr(a, "toarray"):
        a = a.toarray()
    try:
        import pandas as _pd  # type: ignore
        if isinstance(a, (_pd.DataFrame, _pd.Series)):
            a = a.to_numpy()
    except Exception:
        pass
    if isinstance(a, np.matrix):
        a = np.asarray(a)
    a = np.asarray(a, dtype=np.float32, order="C")
    if a.ndim != 2:
        a = np.atleast_2d(a)
    return a


def _resolve_reference_query_masks(
    adata: Any,
    *,
    batch_key: str,
    label_key: Optional[str],
    unlabeled_category: Any,
    reference: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if label_key is not None and label_key in adata.obs:
        labels = adata.obs[label_key]
        if unlabeled_category is None:
            query_mask = labels.isna()
        elif isinstance(unlabeled_category, (list, tuple, set)):
            query_mask = labels.isna() | labels.isin(unlabeled_category)
        else:
            query_mask = labels.isna() | labels.eq(unlabeled_category)
        ref_mask = ~query_mask
        if int(query_mask.sum()) == 0 or int(ref_mask.sum()) == 0:
            raise ValueError("Reference/query subset empty; check label_key/unlabeled_category.")
        return np.asarray(ref_mask), np.asarray(query_mask)

    if batch_key not in adata.obs:
        raise KeyError(f"Neither label_key nor batch_key='{batch_key}' found in adata.obs.")

    batches = adata.obs[batch_key].astype(str)
    unique = set(batches.unique())
    label_map = {str(label).lower(): str(label) for label in unique}
    ref_label: Optional[str] = None
    ref_pref = str(reference).lower()
    if ref_pref in label_map:
        ref_label = label_map[ref_pref]
    elif "largest" in ref_pref or ref_pref == "auto":
        ref_label = batches.value_counts().idxmax()
    else:
        ref_label = batches.value_counts().idxmax()

    ref_mask = batches.eq(ref_label)
    query_mask = ~ref_mask
    if int(query_mask.sum()) == 0 or int(ref_mask.sum()) == 0:
        raise ValueError("Reference/query subset empty; check batch_key and reference.")
    return np.asarray(ref_mask), np.asarray(query_mask)


def _faiss_ready(a: np.ndarray) -> np.ndarray:
    return _as_nd_f32_c(a)


def _get_faiss_index(d: int, use_gpu: bool, device: int):
    if not _FAISS_AVAIL:
        raise RuntimeError("FAISS not available")
    cpu_index = faiss.IndexFlatL2(d)
    if use_gpu and _FAISS_GPU:
        res = _FAISS_GPU_RESOURCES.get(device)
        if res is None:
            res = faiss.StandardGpuResources()
            _FAISS_GPU_RESOURCES[device] = res
        return faiss.index_cpu_to_gpu(res, int(device), cpu_index)
    return cpu_index


def _faiss_knn_search(
    query: np.ndarray, base: np.ndarray, k: int, use_gpu: bool = True, device: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    xb = _faiss_ready(base)
    xq = _faiss_ready(query)
    index = _get_faiss_index(xb.shape[1], use_gpu=use_gpu, device=device)
    index.add(xb)
    D2, I = index.search(xq, k)
    return D2, I


def _ot_barycentric_gpu(
    Bi: np.ndarray,
    R: np.ndarray,
    reg: float = 0.05,
    reg_m: float = 0.5,
    cost_clip_q: Optional[float] = 0.90,
    clip_big: float = 50.0,
    ot_backend: str = "torch",
    iters: int = 1000,
    tol: float = 1e-6,
    use_gpu: bool = True,
    gpu_device: int = 0,
    return_transport: bool = False,
    transport_topk: int = 10,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, np.ndarray]]:
    # Backward-compat: map any 'keops' request to 'torch' (KeOps path removed).
    if isinstance(ot_backend, str) and ot_backend.lower() == "keops":
        ot_backend = "torch"

    assert ot_backend in {"torch", "pot"}

    if len(Bi) == 0:
        return Bi.copy()
    if isinstance(R, dict):
        # caller should have selected class-specific Rc already
        raise ValueError("R must be an ndarray here (class subset should be chosen upstream).")
    if len(R) == 0:
        return Bi.copy()

    if ot_backend == "pot":
        return _ot_barycentric_pot(
            Bi,
            R,
            reg=reg,
            reg_m=reg_m,
            cost_clip_q=cost_clip_q,
            clip_big=clip_big,
            return_transport=return_transport,
            transport_topk=transport_topk,
        )

    device = _torch_device(use_gpu, gpu_device)
    dtype = torch.float32

    N, d = Bi.shape
    M = R.shape[0]
    Bi_t = _to_torch(Bi, device=device, dtype=dtype)
    R_t = _to_torch(R, device=device, dtype=dtype)

    a = torch.full((N,), 1.0 / max(N, 1), dtype=dtype, device=device)
    b = torch.full((M,), 1.0 / max(M, 1), dtype=dtype, device=device)

    M_full = torch.cdist(Bi_t, R_t, p=2).pow_(2)
    std_val = M_full.std().clamp_min(1e-8)

    if cost_clip_q is not None:
        thr = torch.quantile(M_full, q=float(cost_clip_q), dim=1, keepdim=True)
        M_full = torch.where(M_full > thr, thr + clip_big, M_full)
    M_norm = M_full / std_val
    tau = float(reg_m / (reg_m + reg))
    u, v, K = _sinkhorn_uot_torch(
        M_norm,
        a,
        b,
        eps=reg,
        tau=tau,
        iters=iters,
        tol=tol,
    )
    num = torch.matmul(K, v[:, None] * R_t)
    den = torch.matmul(K, v).clamp_min(torch.finfo(dtype).eps)
    out = num / den[:, None]
    out_np = out.detach().cpu().to(dtype=torch.float32).numpy()

    if not return_transport:
        return out_np

    T = (u[:, None] * K) * v[None, :]
    topk = int(min(max(1, transport_topk), T.shape[1]))
    if topk < T.shape[1]:
        w_top, idx_top = torch.topk(T, k=topk, dim=1)
    else:
        idx_top = torch.arange(T.shape[1], device=T.device)[None, :].expand(T.shape[0], -1)
        w_top = T
    row_sum = w_top.sum(dim=1, keepdim=True).clamp_min(torch.finfo(dtype).eps)
    w_norm = w_top / row_sum
    transport = {
        "indices": idx_top.detach().cpu().to(dtype=torch.int32).numpy(),
        "weights": w_norm.detach().cpu().to(dtype=torch.float32).numpy(),
    }
    return out_np, transport


def _ot_barycentric_pot(
    Bi: np.ndarray,
    R: np.ndarray,
    reg: float = 0.05,
    reg_m: float = 0.5,
    cost_clip_q: Optional[float] = 0.90,
    clip_big: float = 50.0,
    return_transport: bool = False,
    transport_topk: int = 10,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, np.ndarray]]:
    if not _POT_AVAILABLE or ot is None:
        raise ModuleNotFoundError(
            "POT is required when ot_backend='pot'. Install it via `pip install POT` or "
            "`pip install scbiot[analysis]`."
        )
    if len(Bi) == 0 or len(R) == 0:
        empty = Bi.copy()
        if not return_transport:
            return empty
        return empty, {
            "indices": np.zeros((0, 1), dtype=np.int32),
            "weights": np.zeros((0, 1), dtype=np.float32),
        }
    Bi64 = np.asarray(Bi, dtype=np.float64, order="C")
    R64 = np.asarray(R, dtype=np.float64, order="C")
    M = ot.dist(Bi64, R64, metric="sqeuclidean")
    M /= (M.std() + 1e-8)
    if cost_clip_q is not None:
        thr = np.quantile(M, cost_clip_q, axis=1, keepdims=True)
        M = np.where(M > thr, thr + clip_big, M)
    a = np.full(Bi.shape[0], 1.0 / max(Bi.shape[0], 1), dtype=np.float64)
    b = np.full(R.shape[0], 1.0 / max(R.shape[0], 1), dtype=np.float64)
    try:
        T = ot.unbalanced.sinkhorn_unbalanced(
            a,
            b,
            M,
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
            M,
            reg,
            reg_m,
            numItermax=1000,
            stopThr=1e-6,
        )
    row_sum = T.sum(1, keepdims=True) + 1e-12
    Bi_to_R = (T / row_sum) @ R64
    out = Bi_to_R.astype(Bi.dtype, copy=False)

    if not return_transport:
        return out

    topk = int(min(max(1, transport_topk), T.shape[1]))
    if topk < T.shape[1]:
        idx_top = np.argpartition(T, -topk, axis=1)[:, -topk:]
        rows = np.arange(T.shape[0])[:, None]
        w_top = T[rows, idx_top]
    else:
        idx_top = np.broadcast_to(np.arange(T.shape[1]), T.shape).copy()
        w_top = T.copy()
    w_sum = w_top.sum(axis=1, keepdims=True)
    w_top = w_top / (w_sum + 1e-12)
    transport = {
        "indices": idx_top.astype(np.int32, copy=False),
        "weights": w_top.astype(np.float32, copy=False),
    }
    return out, transport


# -------------------- KNN / graphs --------------------

def _ptp(x: np.ndarray) -> float:
    return float(np.ptp(x)) if len(x) else 0.0


def _lerp(a: float, b: float, t: float) -> float:
    return (1.0 - t) * a + t * b


def _knn_idx(X: np.ndarray, k: int, use_gpu: bool = True, device: int = 0) -> np.ndarray:
    N = len(X)
    if N <= 1:
        return np.zeros((N, 0), dtype=int)
    k = max(1, min(k, N - 1))
    if _FAISS_AVAIL:
        D2, I = _faiss_knn_search(X, X, k + 1, use_gpu=use_gpu, device=device)
        return I[:, 1:]
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X)
    _, idx = nn.kneighbors(X)
    return idx[:, 1:]


def _knn_graph(
    X: np.ndarray, k: int, use_gpu: bool = True, device: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    N = len(X)
    if N <= 1:
        return np.zeros((N, 0), dtype=X.dtype), np.zeros((N, 0), dtype=int)
    k_eff = int(max(1, min(k, N - 1)))
    if _FAISS_AVAIL:
        D2, I = _faiss_knn_search(X, X, k_eff + 1, use_gpu=use_gpu, device=device)
        d = np.sqrt(np.maximum(D2[:, 1:], 0.0)).astype(X.dtype, copy=False)
        idx = I[:, 1:]
        return d, idx
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean").fit(X)
    d, idx = nn.kneighbors(X)
    return d[:, 1:], idx[:, 1:]


def _neighbor_batch_entropy_per_cell(
    X: np.ndarray,
    batch_enc: np.ndarray,
    k: int = 15,
    use_gpu: bool = True,
    device: int = 0,
) -> Tuple[np.ndarray, float]:
    if len(X) == 0:
        return np.zeros(0, dtype=np.float32), 0.0
    idx = _knn_idx(X, k, use_gpu=use_gpu, device=device)
    n_classes = int(batch_enc.max()) + 1 if len(batch_enc) else 1
    N = X.shape[0]
    counts = np.zeros((N, n_classes), dtype=np.int32)
    if idx.size > 0:
        row_ids = np.repeat(np.arange(N), idx.shape[1])
        np.add.at(counts, (row_ids, batch_enc[idx].ravel()), 1)
    p = counts / (counts.sum(1, keepdims=True) + 1e-12)
    H = -(p * np.log(p + 1e-12)).sum(1)
    return H.astype(np.float32, copy=False), float(H.mean())


def _knn_overlap(
    X_prev: np.ndarray,
    X_now: np.ndarray,
    k: int = 30,
    subsample: Optional[int] = None,
    rng: int = 0,
    use_gpu: bool = True,
    device: int = 0,
) -> float:
    if len(X_prev) == 0:
        return 1.0
    rng_state = np.random.default_rng(rng)
    N = X_prev.shape[0]
    if subsample is not None and subsample < N:
        sub = rng_state.choice(N, size=subsample, replace=False)
        A0, A1 = X_prev[sub], X_now[sub]
    else:
        A0, A1 = X_prev, X_now
    k = max(1, min(k, len(A0) - 1))
    if k <= 0:
        return 1.0
    i0 = _knn_idx(A0, k, use_gpu=use_gpu, device=device)
    i1 = _knn_idx(A1, k, use_gpu=use_gpu, device=device)
    inter_counts = (i0[:, :, None] == i1[:, None, :]).sum(axis=(1, 2)).astype(np.float32)
    return float((inter_counts / k).mean())


# -------------------- Rare-aware prototypes --------------------

def _local_knn_density(X: np.ndarray, k: int = 15, use_gpu: bool = True, device: int = 0) -> np.ndarray:
    """Return density-derived weights (larger for sparser points)."""
    d, _ = _knn_graph(X, k, use_gpu=use_gpu, device=device)
    m = d.mean(axis=1) if d.size else np.zeros(len(X), dtype=X.dtype)
    w = m / (m.mean() + 1e-8)
    return (w + 1e-8).astype(np.float32, copy=False)


def _minikm_centers(
    X: np.ndarray,
    n_clusters: int,
    seed: int = 0,
    use_gpu: bool = True,
    device: int = 0,
    weights: Optional[np.ndarray] = None,
    return_labels: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """K-means centers with optional sample weights (falls back if unsupported)."""
    del use_gpu, device  # GPU unused but kept for signature parity
    n_clusters = int(max(2, min(n_clusters, len(X)))) if len(X) > 1 else 1
    if n_clusters <= 1:
        centers = X.mean(0, keepdims=True).astype(X.dtype, copy=False)
        if return_labels:
            labels = np.zeros(len(X), dtype=np.int32)
            return centers, labels
        return centers

    xp = np.asarray(X, dtype=np.float32, order="C")
    labels: Optional[np.ndarray] = None
    try:
        km = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=seed,
            batch_size=min(4096, max(n_clusters * 4, 256)),
            max_iter=25,
            n_init=1,
        )
        try:
            km.fit(xp, sample_weight=weights)
        except TypeError:
            km.fit(xp)
        centers = km.cluster_centers_.astype(X.dtype, copy=False)
        if return_labels:
            try:
                labels = km.labels_
            except Exception:
                labels = km.predict(xp)
    except Exception:
        rng = np.random.default_rng(seed)
        init_idx = rng.choice(len(xp), size=n_clusters, replace=False)
        init = xp[init_idx]
        centers, _ = kmeans2(xp, init, iter=20, minit="matrix")
        centers = centers.astype(X.dtype, copy=False)
        if return_labels:
            labels = pairwise_distances_argmin(xp, centers, metric="euclidean")
    if return_labels:
        if labels is None:
            labels = pairwise_distances_argmin(xp, centers, metric="euclidean")
        return centers, labels.astype(np.int32, copy=False)
    return centers


# ===================== Supervised helpers (from code2) =====================

def _class_means(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    use_gpu: bool = False,
    device: int = 0,
) -> np.ndarray:
    D = X.shape[1]
    if not use_gpu or not torch.cuda.is_available():
        C = np.full((n_classes, D), np.nan, dtype=X.dtype)
        for c in range(n_classes):
            idx = np.where(y == c)[0]
            if len(idx) > 0:
                C[c] = X[idx].mean(0)
        return C

    device_t = _torch_device(use_gpu=True, gpu_device=device)
    X_t = torch.as_tensor(X, dtype=torch.float32, device=device_t)
    y_t = torch.as_tensor(y, dtype=torch.int64, device=device_t)
    valid = (y_t >= 0) & (y_t < n_classes)
    if not torch.any(valid):
        return np.full((n_classes, D), np.nan, dtype=X.dtype)

    sums = torch.zeros((n_classes, D), device=device_t, dtype=torch.float32)
    counts = torch.zeros((n_classes,), device=device_t, dtype=torch.float32)
    idx = y_t[valid]
    sums.index_add_(0, idx, X_t[valid])
    counts.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
    means = sums / counts.clamp_min(1.0)[:, None]
    means[counts == 0] = float("nan")
    return means.cpu().numpy().astype(X.dtype, copy=False)


def _nearest_other_class_index(
    X: np.ndarray,
    y: np.ndarray,
    C: np.ndarray,
    use_gpu: bool = False,
    device: int = 0,
    chunk_size: int = 65536,
) -> np.ndarray:
    N = len(X)
    if N == 0 or C.size == 0:
        return np.full(N, -1, dtype=int)
    if not use_gpu or not torch.cuda.is_available():
        D = pairwise_distances(X, C, metric="euclidean")
        res = np.full(N, -1, dtype=int)
        for i in range(N):
            yi = y[i]
            if yi < 0 or yi >= C.shape[0] or np.isnan(C[yi]).any():
                continue
            Di = D[i].copy()
            Di[yi] = np.inf
            j = int(np.argmin(Di))
            if np.isfinite(Di[j]) and not np.isnan(C[j]).any():
                res[i] = j
        return res

    device_t = _torch_device(use_gpu=True, gpu_device=device)
    C_t = torch.as_tensor(C, dtype=torch.float32, device=device_t)
    valid_class = ~torch.isnan(C_t).any(dim=1)
    if not torch.any(valid_class):
        return np.full(N, -1, dtype=int)

    n_classes = int(C_t.shape[0])
    res = np.full(N, -1, dtype=int)
    chunk_size = int(max(1, chunk_size))
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        X_t = torch.as_tensor(X[start:end], dtype=torch.float32, device=device_t)
        dist = torch.cdist(X_t, C_t, p=2)
        if not torch.all(valid_class):
            dist[:, ~valid_class] = float("inf")
        y_chunk = torch.as_tensor(y[start:end], dtype=torch.int64, device=device_t)
        valid_y = (y_chunk >= 0) & (y_chunk < n_classes)
        if torch.any(valid_y):
            rows = torch.nonzero(valid_y, as_tuple=False).squeeze(1)
            cls = y_chunk[rows]
            valid_rows = valid_class[cls]
            if torch.any(valid_rows):
                rows = rows[valid_rows]
                cls = cls[valid_rows]
                dist[rows, cls] = float("inf")
        min_val, min_idx = torch.min(dist, dim=1)
        min_idx_np = min_idx.cpu().numpy()
        min_val_np = min_val.cpu().numpy()
        min_idx_np[~np.isfinite(min_val_np)] = -1
        res[start:end] = min_idx_np
    return res


def _get_semi_labels(adata, label_key: Optional[str], unknown_aliases=(
    "unknown","unlabeled","unlabelled","unk","na","nan","none",""
)):
    """
    Normalize a semi-supervised label column so unlabeled cells become the string 'unknown'
    in adata.obs[label_key], and return integer labels y with -1 for unknown.
    """
    if label_key is None or label_key not in adata.obs:
        return None, 0, None  # no labels provided

    s = adata.obs[label_key]
    s_norm = s.astype(str).str.strip().str.lower()
    unk_mask = s.isna() | s_norm.isin(set(a.lower() for a in unknown_aliases))
    s_norm = s_norm.mask(unk_mask, "unknown")

    # write back normalized labels so AnnData shows 'unknown' for unlabeled
    adata.obs[label_key] = s_norm.astype("category")

    known = ~s_norm.eq("unknown")
    y = np.full(len(s_norm), -1, dtype=int)
    n_labels = 0
    le = None
    if known.any():
        le = LabelEncoder().fit(s_norm[known])
        y[known] = le.transform(s_norm[known])
        n_labels = int(y.max()) + 1
    return y, n_labels, le


# ===================== Prototypes (unsup + supervised) =====================

def _compute_prototypes(
    X: np.ndarray,
    b: np.ndarray,
    ref_label: int,
    K_ref: int = 512,
    K_batch: int = 256,
    seed: int = 0,
    use_gpu: bool = True,
    device: int = 0,
    y: Optional[np.ndarray] = None,
) -> Tuple[object, List[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[int]]], np.ndarray]:
    """
    If y is None or all -1: original behavior (unsupervised). Returns:
      R: ndarray (reference prototypes), packs: [(idx, Bi, nn_idx, None), ...]
    If y provided (>=0 are known classes): supervised class-conditional prototypes.
      R: dict[int -> ndarray] of reference prototypes per class,
      packs: [(idx, Bi, nn_idx, c), ...] for each batch/class present.
    """
    if (y is None) or (np.all(y < 0)):
        # ---- original (unsupervised) ----
        ref_idx = np.where(b == ref_label)[0]
        X_ref = X[ref_idx]
        Kref_eff = int(min(K_ref, max(16, 2 * np.sqrt(max(len(ref_idx), 1)))))
        w_ref = _local_knn_density(X_ref, k=15, use_gpu=use_gpu, device=device) if len(X_ref) else None
        R = _minikm_centers(X_ref, Kref_eff, seed, use_gpu=use_gpu, device=device, weights=w_ref)

        packs: List[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[int]]] = []
        for lbl in np.unique(b):
            if lbl == ref_label:
                continue
            idx = np.where(b == lbl)[0]
            if len(idx) == 0:
                continue
            Xi = X[idx]
            Kb_eff = int(min(K_batch, max(8, 2 * np.sqrt(len(idx)))))
            w_i = _local_knn_density(Xi, k=15, use_gpu=use_gpu, device=device)
            Bi = _minikm_centers(Xi, Kb_eff, seed + 7, use_gpu=use_gpu, device=device, weights=w_i)
            if _FAISS_AVAIL:
                _, nn_idx = _faiss_knn_search(Xi, Bi, 1, use_gpu=use_gpu, device=device)
                nn_idx = nn_idx.ravel()
            else:
                nn_idx = pairwise_distances_argmin(Xi, Bi, metric="euclidean")
            packs.append((idx, Bi, nn_idx, None))
        return R, packs, ref_idx

    # ---- supervised (class-conditional) ----
    classes = np.unique(y[y >= 0])
    R_dict: Dict[int, np.ndarray] = {}
    packs: List[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[int]]] = []

    # build class-specific reference prototypes (prefer reference batch if available)
    for c in classes:
        ref_idx_c = np.where((b == ref_label) & (y == c))[0]
        if len(ref_idx_c) == 0:
            ref_idx_c = np.where(y == c)[0]
        X_ref_c = X[ref_idx_c]
        Kref_eff = int(min(K_ref, max(8, int(2 * np.sqrt(max(len(ref_idx_c), 1))))))
        w_ref_c = _local_knn_density(X_ref_c, k=15, use_gpu=use_gpu, device=device) if len(X_ref_c) else None
        R_dict[int(c)] = _minikm_centers(X_ref_c, Kref_eff, seed + int(c),
                                         use_gpu=use_gpu, device=device, weights=w_ref_c)

    # per (batch, class) batches
    for lbl in np.unique(b):
        if lbl == ref_label:
            continue
        for c in classes:
            idx = np.where((b == lbl) & (y == c))[0]
            if len(idx) == 0:
                continue
            Xi = X[idx]
            Kb_eff = int(min(K_batch, max(8, int(2 * np.sqrt(len(idx))))))
            w_i = _local_knn_density(Xi, k=15, use_gpu=use_gpu, device=device)
            Bi = _minikm_centers(Xi, Kb_eff, seed + 7 + int(c), use_gpu=use_gpu, device=device, weights=w_i)
            if _FAISS_AVAIL:
                _, nn_idx = _faiss_knn_search(Xi, Bi, 1, use_gpu=use_gpu, device=device)
                nn_idx = nn_idx.ravel()
            else:
                nn_idx = pairwise_distances_argmin(Xi, Bi, metric="euclidean")
            packs.append((idx, Bi, nn_idx, int(c)))

    ref_idx_all = np.where(b == ref_label)[0]
    return R_dict, packs, ref_idx_all


def _compute_prototypes_union(
    X: np.ndarray,
    b: np.ndarray,
    K_ref: int = 1024,
    K_batch: int = 448,
    seed: int = 0,
    use_gpu: bool = True,
    device: int = 0,
    y: Optional[np.ndarray] = None,
) -> Tuple[object, List[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[int]]], np.ndarray]:
    if (y is None) or (np.all(y < 0)):
        # ---- original union (unsupervised) ----
        all_B = []
        packs: List[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[int]]] = []
        for lbl in np.unique(b):
            idx = np.where(b == lbl)[0]
            if len(idx) == 0:
                continue
            Xi = X[idx]
            Kb_eff = int(min(K_batch, max(8, 2 * np.sqrt(len(idx)))))
            w_i = _local_knn_density(Xi, k=15, use_gpu=use_gpu, device=device)
            Bi = _minikm_centers(Xi, Kb_eff, seed + 7, use_gpu=use_gpu, device=device, weights=w_i)
            all_B.append(Bi)
            if _FAISS_AVAIL:
                _, nn_idx = _faiss_knn_search(Xi, Bi, 1, use_gpu=use_gpu, device=device)
                nn_idx = nn_idx.ravel()
            else:
                nn_idx = pairwise_distances_argmin(Xi, Bi, metric="euclidean")
            packs.append((idx, Bi, nn_idx, None))
        if len(all_B) == 0:
            R = _minikm_centers(X, max(16, 2 * int(np.sqrt(max(len(X), 1)))), seed, use_gpu=use_gpu, device=device)
        else:
            Bstk = np.vstack(all_B)
            Kref_eff = int(min(K_ref, max(32, 2 * np.sqrt(len(Bstk)))))
            w_union = _local_knn_density(Bstk, k=15, use_gpu=use_gpu, device=device)
            R = _minikm_centers(Bstk, Kref_eff, seed, use_gpu=use_gpu, device=device, weights=w_union)
        ref_idx = np.arange(X.shape[0])
        return R, packs, ref_idx

    # ---- supervised union: class-conditional union prototypes ----
    classes = np.unique(y[y >= 0])
    all_B_dict: Dict[int, List[np.ndarray]] = {int(c): [] for c in classes}
    packs: List[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[int]]] = []

    for lbl in np.unique(b):
        for c in classes:
            idx = np.where((b == lbl) & (y == c))[0]
            if len(idx) == 0:
                continue
            Xi = X[idx]
            Kb_eff = int(min(K_batch, max(8, int(2 * np.sqrt(len(idx))))))
            w_i = _local_knn_density(Xi, k=15, use_gpu=use_gpu, device=device)
            Bi = _minikm_centers(Xi, Kb_eff, seed + 7 + int(c), use_gpu=use_gpu, device=device, weights=w_i)
            all_B_dict[int(c)].append(Bi)
            if _FAISS_AVAIL:
                _, nn_idx = _faiss_knn_search(Xi, Bi, 1, use_gpu=use_gpu, device=device)
                nn_idx = nn_idx.ravel()
            else:
                nn_idx = pairwise_distances_argmin(Xi, Bi, metric="euclidean")
            packs.append((idx, Bi, nn_idx, int(c)))

    R_dict: Dict[int, np.ndarray] = {}
    for c in classes:
        Bs = all_B_dict[int(c)]
        if len(Bs) == 0:
            R_dict[int(c)] = np.zeros((0, X.shape[1]), dtype=X.dtype)
            continue
        Bstk = np.vstack(Bs)
        Kref_eff = int(min(K_ref, max(32, 2 * np.sqrt(len(Bstk)))))
        w_union = _local_knn_density(Bstk, k=15, use_gpu=use_gpu, device=device)
        R_dict[int(c)] = _minikm_centers(Bstk, Kref_eff, seed + int(c), use_gpu=use_gpu, device=device, weights=w_union)

    ref_idx = np.arange(X.shape[0])  # placeholder, not used downstream
    return R_dict, packs, ref_idx


# -------------------- Field shaping & guards --------------------

def _cluster_sharpen_field(
    X: np.ndarray,
    K: int = 24,
    seed: int = 0,
    pull: float = 0.70,
    push: float = 0.35,
    bridge_score: Optional[np.ndarray] = None,
    gate: float = 0.7,
    use_gpu: bool = True,
    device: int = 0,
) -> np.ndarray:
    N = len(X)
    if N == 0:
        return np.zeros_like(X)
    K = int(max(8, min(K, N)))
    C = _minikm_centers(X, K, seed, use_gpu=use_gpu, device=device)
    if _FAISS_AVAIL:
        D2, I = _faiss_knn_search(X, C, 2, use_gpu=use_gpu, device=device)
        d01 = np.sqrt(np.maximum(D2, 0.0))
        lab, other = I[:, 0], I[:, 1]
        d1, d2 = d01[:, 0], d01[:, 1]
    else:
        D = pairwise_distances(X, C, metric="euclidean")
        ord2 = np.argpartition(D, kth=(0, 1), axis=1)[:, :2]
        d01 = np.take_along_axis(D, ord2, axis=1)
        lab, other = ord2[:, 0], ord2[:, 1]
        d1, d2 = d01[:, 0], d01[:, 1]
    margin = (d2 - d1) / (np.median(d2) + 1e-8)
    g = 1.0 / (1.0 + np.exp((margin - 1.0) / 0.8))
    disp = pull * (C[lab] - X) + (push * g)[:, None] * (X - C[other])
    if bridge_score is not None:
        disp = ((1 - 0.4 * bridge_score)[:, None]) * disp
        near_bridge = (bridge_score > gate).astype(X.dtype)[:, None]
        disp = disp - near_bridge * (0.6 * (X - C[other]))
    return disp.astype(X.dtype, copy=False)


def _smooth_by_knn(
    field: np.ndarray,
    idx: np.ndarray,
    lam: float = 0.3,
    use_gpu: bool = False,
    device: int = 0,
) -> np.ndarray:
    if lam <= 0 or idx.size == 0:
        return field
    if not use_gpu or not torch.cuda.is_available():
        neigh = field[idx]
        avg = neigh.mean(axis=1)
        return (1.0 - lam) * field + lam * avg
    device_t = _torch_device(use_gpu=True, gpu_device=device)
    field_t = torch.as_tensor(field, dtype=torch.float32, device=device_t)
    idx_t = torch.as_tensor(idx, dtype=torch.int64, device=device_t)
    neigh = field_t[idx_t]
    avg = neigh.mean(dim=1)
    out = (1.0 - lam) * field_t + lam * avg
    return out.cpu().numpy().astype(field.dtype, copy=False)


def _cap_step_local(
    move: np.ndarray,
    knn_mean_dist: np.ndarray,
    max_step_local: float = 1.1,
    use_gpu: bool = False,
    device: int = 0,
) -> np.ndarray:
    if max_step_local <= 0:
        return move
    if not use_gpu or not torch.cuda.is_available():
        cap = max_step_local * (knn_mean_dist + 1e-8)
        nrm = np.linalg.norm(move, axis=1) + 1e-12
        scale = np.minimum(1.0, cap / nrm)
        return move * scale[:, None]
    device_t = _torch_device(use_gpu=True, gpu_device=device)
    move_t = torch.as_tensor(move, dtype=torch.float32, device=device_t)
    dist_t = torch.as_tensor(knn_mean_dist, dtype=torch.float32, device=device_t)
    cap = max_step_local * (dist_t + 1e-8)
    nrm = torch.linalg.norm(move_t, dim=1) + 1e-12
    scale = torch.minimum(torch.ones_like(cap), cap / nrm)
    out = move_t * scale[:, None]
    return out.cpu().numpy().astype(move.dtype, copy=False)


def _guard_edge_stretch_weighted(
    X: np.ndarray,
    move: np.ndarray,
    idx0: np.ndarray,
    d0: np.ndarray,
    smin_i: np.ndarray,
    smax_i: np.ndarray,
    rounds: int = 2,
    use_gpu: bool = False,
    device: int = 0,
    chunk_size: int = 65536,
) -> np.ndarray:
    if idx0.size == 0:
        return move
    if not use_gpu or not torch.cuda.is_available():
        eps = 1e-8
        for _ in range(int(max(1, rounds))):
            Xcand_i = X[:, None, :] + move[:, None, :]
            Xcand_j = X[idx0] + move[idx0]
            dij_new = np.linalg.norm(Xcand_i - Xcand_j, axis=2)
            r = dij_new / (d0 + eps)
            r_max = r.max(axis=1)
            r_min = r.min(axis=1)
            f_high = np.minimum(1.0, smax_i / (r_max + eps))
            f_low = np.minimum(1.0, (r_min + eps) / smin_i)
            f = np.minimum(f_high, f_low).astype(move.dtype)
            if np.all(f >= 0.999):
                break
            move *= f[:, None]
        return move

    device_t = _torch_device(use_gpu=True, gpu_device=device)
    X_t = torch.as_tensor(X, dtype=torch.float32, device=device_t)
    move_t = torch.as_tensor(move, dtype=torch.float32, device=device_t)
    idx_t = torch.as_tensor(idx0, dtype=torch.int64, device=device_t)
    d0_t = torch.as_tensor(d0, dtype=torch.float32, device=device_t)
    smin_t = torch.as_tensor(smin_i, dtype=torch.float32, device=device_t)
    smax_t = torch.as_tensor(smax_i, dtype=torch.float32, device=device_t)
    eps = 1e-8
    n = X_t.shape[0]
    chunk_size = int(max(1, chunk_size))

    for _ in range(int(max(1, rounds))):
        f_all = torch.ones(n, device=device_t, dtype=torch.float32)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            Xi = X_t[start:end]
            move_i = move_t[start:end]
            idx_chunk = idx_t[start:end]
            Xcand_i = Xi[:, None, :] + move_i[:, None, :]
            Xcand_j = X_t[idx_chunk] + move_t[idx_chunk]
            dij_new = torch.linalg.norm(Xcand_i - Xcand_j, dim=2)
            r = dij_new / (d0_t[start:end] + eps)
            r_max = r.max(dim=1).values
            r_min = r.min(dim=1).values
            f_high = torch.minimum(torch.ones_like(r_max), smax_t[start:end] / (r_max + eps))
            f_low = torch.minimum(torch.ones_like(r_min), (r_min + eps) / smin_t[start:end])
            f_all[start:end] = torch.minimum(f_high, f_low)
        if torch.all(f_all >= 0.999):
            break
        move_t = move_t * f_all[:, None]
    return move_t.cpu().numpy().astype(move.dtype, copy=False)


def _graph_strain(
    X: np.ndarray,
    idx0: np.ndarray,
    d0: np.ndarray,
    clip: float = 1.0,
    use_gpu: bool = False,
    device: int = 0,
    chunk_size: int = 65536,
) -> float:
    if idx0.size == 0:
        return 0.0
    if not use_gpu or not torch.cuda.is_available():
        Xi = X[:, None, :]
        Xj = X[idx0]
        dij = np.linalg.norm(Xi - Xj, axis=2)
        r = dij / (d0 + 1e-8)
        dev = np.clip(r - 1.0, -clip, clip)
        return float(np.mean(dev * dev))

    device_t = _torch_device(use_gpu=True, gpu_device=device)
    X_t = torch.as_tensor(X, dtype=torch.float32, device=device_t)
    idx_t = torch.as_tensor(idx0, dtype=torch.int64, device=device_t)
    d0_t = torch.as_tensor(d0, dtype=torch.float32, device=device_t)
    n = X_t.shape[0]
    chunk_size = int(max(1, chunk_size))
    total = 0.0
    count = 0
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        Xi = X_t[start:end]
        Xj = X_t[idx_t[start:end]]
        dij = torch.linalg.norm(Xi[:, None, :] - Xj, dim=2)
        r = dij / (d0_t[start:end] + 1e-8)
        dev = torch.clamp(r - 1.0, -clip, clip)
        total += float(torch.sum(dev * dev).item())
        count += int(dev.numel())
    return float(total / max(1, count))


def _trustworthiness_score(
    X_orig: np.ndarray,
    X_emb: np.ndarray,
    n_neighbors: int,
    use_gpu: bool,
    gpu_device: int,
) -> float:
    n = X_orig.shape[0]
    if n <= 1 or n_neighbors <= 0:
        return 1.0
    if n_neighbors >= n / 2:
        n_neighbors = max(1, (n // 2) - 1)
    n_neighbors = min(n_neighbors, n - 1)
    if n_neighbors <= 0:
        return 1.0

    X_orig32 = np.asarray(X_orig, dtype=np.float32, order="C")
    X_emb32 = np.asarray(X_emb, dtype=np.float32, order="C")

    ind_X: Optional[np.ndarray]
    ind_Y: Optional[np.ndarray]

    if use_gpu and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_device}")
        try:
            with torch.no_grad():
                X_o_t = torch.as_tensor(X_orig32, dtype=torch.float32, device=device)
                X_e_t = torch.as_tensor(X_emb32, dtype=torch.float32, device=device)
                dist_o = torch.cdist(X_o_t, X_o_t, p=2)
                dist_o.fill_diagonal_(float("inf"))
                ind_X_t = torch.argsort(dist_o, dim=1)
                dist_e = torch.cdist(X_e_t, X_e_t, p=2)
                dist_e.fill_diagonal_(float("inf"))
                ind_Y_t = torch.topk(dist_e, k=n_neighbors, largest=False).indices
            ind_X = ind_X_t.cpu().numpy()
            ind_Y = ind_Y_t.cpu().numpy()
            del dist_o, dist_e, ind_X_t, ind_Y_t, X_o_t, X_e_t
        except RuntimeError:
            torch.cuda.empty_cache()
            ind_X = None
            ind_Y = None
    else:
        ind_X = None
        ind_Y = None

    if ind_X is None or ind_Y is None:
        return float(trustworthiness(X_orig32, X_emb32, n_neighbors=n_neighbors))

    ordered = np.arange(n, dtype=np.int32)
    ranks_template = np.arange(1, n + 1, dtype=np.int32)
    inverted_index = np.empty((n, n), dtype=np.int32)
    inverted_index[ordered[:, None], ind_X] = ranks_template
    ranks = inverted_index[ordered[:, None], ind_Y] - n_neighbors
    ranks = ranks.astype(np.int64, copy=False)
    penalty = ranks[ranks > 0].sum(dtype=np.int64)
    factor = 2.0 / (n * n_neighbors * (2.0 * n - 3.0 * n_neighbors - 1.0))
    return float(1.0 - factor * penalty)
