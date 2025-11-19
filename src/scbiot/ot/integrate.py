# =================== paste-and-run: OTIntegration_rare_aware_sup.py (rare-safe + supervised option) ===================
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Sequence

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
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
    except TypeError:  # older torch without device kwarg
        return torch.as_tensor(x, dtype=dtype).to(device=device).contiguous()


# -------------------- Unbalanced Sinkhorn (Torch) --------------------
@torch.no_grad()
def _sinkhorn_uot_torch(
    M: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float = 0.05,
    tau: float = 0.5,
    iters: int = 1000,
    tol: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = M.dtype
    tiny = torch.finfo(dtype).eps
    K = torch.exp(-M / eps)
    v = torch.ones_like(b)
    u = torch.ones_like(a)

    for _ in range(iters):
        Kv = torch.matmul(K, v).clamp_min(tiny)
        u_new = torch.pow(a / Kv, tau)

        KTu = torch.matmul(K.T, u_new).clamp_min(tiny)
        v_new = torch.pow(b / KTu, tau)

        if (
            torch.max(torch.abs(torch.log(u_new) - torch.log(u))) < tol
            and torch.max(torch.abs(torch.log(v_new) - torch.log(v))) < tol
        ):
            u, v = u_new, v_new
            break
        u, v = u_new, v_new
    return u, v, K


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
) -> np.ndarray:
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
    tau = float(reg_m / (reg_m + reg))

    if cost_clip_q is not None:
        thr = torch.quantile(M_full, q=float(cost_clip_q), dim=1, keepdim=True)
        M_full = torch.where(M_full > thr, thr + clip_big, M_full)
    M_norm = M_full / std_val
    _, v, K = _sinkhorn_uot_torch(
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
    return out.detach().cpu().to(dtype=torch.float32).numpy()


def _ot_barycentric_pot(
    Bi: np.ndarray,
    R: np.ndarray,
    reg: float = 0.05,
    reg_m: float = 0.5,
    cost_clip_q: Optional[float] = 0.90,
    clip_big: float = 50.0,
) -> np.ndarray:
    if not _POT_AVAILABLE or ot is None:
        raise ModuleNotFoundError(
            "POT is required when ot_backend='pot'. Install it via `pip install POT` or "
            "`pip install scbiot[analysis]`."
        )
    if len(Bi) == 0 or len(R) == 0:
        return Bi.copy()
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
    return Bi_to_R.astype(Bi.dtype, copy=False)


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
) -> np.ndarray:
    """K-means centers with optional sample weights (falls back if unsupported)."""
    del use_gpu, device  # GPU unused but kept for signature parity
    n_clusters = int(max(2, min(n_clusters, len(X)))) if len(X) > 1 else 1
    if n_clusters <= 1:
        return X.mean(0, keepdims=True).astype(X.dtype, copy=False)

    xp = np.asarray(X, dtype=np.float32, order="C")
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
    except Exception:
        rng = np.random.default_rng(seed)
        init_idx = rng.choice(len(xp), size=n_clusters, replace=False)
        init = xp[init_idx]
        centers, _ = kmeans2(xp, init, iter=20, minit="matrix")
        centers = centers.astype(X.dtype, copy=False)
    return centers


# ===================== Supervised helpers (from code2) =====================

def _class_means(X: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    D = X.shape[1]
    C = np.full((n_classes, D), np.nan, dtype=X.dtype)
    for c in range(n_classes):
        idx = np.where(y == c)[0]
        if len(idx) > 0:
            C[c] = X[idx].mean(0)
    return C


def _nearest_other_class_index(X: np.ndarray, y: np.ndarray, C: np.ndarray) -> np.ndarray:
    N = len(X)
    if N == 0 or C.size == 0:
        return np.full(N, -1, dtype=int)
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


def _smooth_by_knn(field: np.ndarray, idx: np.ndarray, lam: float = 0.3) -> np.ndarray:
    if lam <= 0 or idx.size == 0:
        return field
    neigh = field[idx]
    avg = neigh.mean(axis=1)
    return (1.0 - lam) * field + lam * avg


def _cap_step_local(
    move: np.ndarray, knn_mean_dist: np.ndarray, max_step_local: float = 1.1
) -> np.ndarray:
    if max_step_local <= 0:
        return move
    cap = max_step_local * (knn_mean_dist + 1e-8)
    nrm = np.linalg.norm(move, axis=1) + 1e-12
    scale = np.minimum(1.0, cap / nrm)
    return move * scale[:, None]


def _guard_edge_stretch_weighted(
    X: np.ndarray,
    move: np.ndarray,
    idx0: np.ndarray,
    d0: np.ndarray,
    smin_i: np.ndarray,
    smax_i: np.ndarray,
    rounds: int = 2,
) -> np.ndarray:
    if idx0.size == 0:
        return move
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


def _graph_strain(X: np.ndarray, idx0: np.ndarray, d0: np.ndarray, clip: float = 1.0) -> float:
    if idx0.size == 0:
        return 0.0
    Xi = X[:, None, :]
    Xj = X[idx0]
    dij = np.linalg.norm(Xi - Xj, axis=2)
    r = dij / (d0 + 1e-8)
    dev = np.clip(r - 1.0, -clip, clip)
    return float(np.mean(dev * dev))


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


# -------------------- Main integration (now supervised-ready) --------------------

def integrate_ot(
    adata: Any,
    obsm_key: str = "X_pca",
    batch_key: str = "batch",
    out_key: str = "scBIOT",
    K_ref: int = 1024,
    K_batch: int = 448,
    reg: float = 0.028,
    reg_m: float = 0.40,
    sharpen: float = 0.22,
    K_pseudo: int = 28,
    pull: float = 0.78,
    push: float = 0.34,
    lambda0_hi: float = 0.52,
    lambda0_lo: float = 0.38,
    smin_bulk: float = 0.72,
    smax_bulk: float = 1.65,
    smin_bridge: float = 0.88,
    smax_bridge: float = 1.24,
    max_step_local: float = 1.05,
    step_lo: float = 0.78,
    step_hi: float = 0.96,
    q_start: float = 0.80,
    q_end: float = 0.90,
    overlap0_lo: float = 0.60,
    overlap0_hi: float = 0.68,
    w_overlap: float = 0.18,
    penalty_gamma: float = 1.4,
    w_strain: float = 1.0,
    k_local: int = 15,
    k_eval: int = 30,
    eval_subsample: Optional[int] = 5000,
    trust_subsample: Optional[int] = 2500,
    max_iter: int = 15,
    patience: int = 3,
    tol: float = 1e-3,
    reference: str = "largest",   # supports "largest" or "union"
    postscale: bool = True,
    random_state: int = 0,
    verbose: bool = True,
    use_gpu: bool = True,
    gpu_device: int = 0,
    ot_backend: str = "torch",
    # ---- NEW supervised knobs ----
    true_label_key: Optional[str] = None,  # e.g., "cell_type". If None -> unsupervised as before.
    lam_sup: float = 0.60,                 # pull to own-class mean (0 disables)
    lam_repulse: float = 0.18,             # repel from nearest other-class mean (0 disables)
) -> Tuple[Any, Dict[str, float | int]]:
    """
    Batch-correct a single-modality embedding with scBIOT optimal transport.

    Parameters
    ----------
    adata
        AnnData object containing a low-dimensional representation to align.
    obsm_key
        Key in ``adata.obsm`` with the starting embedding (for example, PCA).
    batch_key
        ``adata.obs`` column containing batch identities.
    out_key
        Destination key in ``adata.obsm`` for the corrected coordinates.
    K_ref / K_batch
        Reference and batch-specific prototype sizes used for OT coupling.
    reg / reg_m
        Entropic and unbalanced mass-penalty terms for the OT solver.
    sharpen / K_pseudo / pull / push
        Connectivity controls that steer the pseudo-label refinement.
    lambda0_hi / lambda0_lo / smin_* / smax_* / max_step_local / step_*
        Step-size and neighborhood scaling parameters for the iterative updates.
    q_start / q_end / overlap0_lo / overlap0_hi / w_overlap / penalty_gamma
        Hyper-parameters that encourage overlap while penalizing over-correction.
    w_strain
        Weight on the graph-strain regularizer.
    k_local / k_eval
        Neighborhood sizes used for local graph construction and evaluation.
    eval_subsample / trust_subsample
        Optional subsampling for efficiency during scoring.
    max_iter / patience / tol
        Early stopping controls for the outer optimization loop.
    reference
        Reference batch selection strategy (``"largest"`` or ``"union"``).
    postscale
        Whether to rescale the aligned embedding to unit variance per dimension.
    random_state
        Seed for stochastic steps used in subsampling and initialization.
    verbose
        Print progress information when ``True``.
    use_gpu / gpu_device / ot_backend
        Compute backend selection for OT (Torch by default, POT when requested).
    true_label_key
        Optional ``adata.obs`` column for semi-supervised guidance.
    lam_sup / lam_repulse
        Attraction/repulsion strengths when ``true_label_key`` is provided.

    Returns
    -------
    adata
        The input object with integrated coordinates stored in ``adata.obsm[out_key]``.
    dict
        Diagnostics with mixing metrics (``mix``, ``overlap0``, ``strain``, ``tw``)
        and the iteration that achieved the best score (``it``).

    Notes
    -----
    The function updates ``adata`` in place and also returns it for convenience.
    """
    X0 = _as_nd_f32_c(adata.obsm[obsm_key])
    b_raw = adata.obs[batch_key].to_numpy()
    le = LabelEncoder()
    b = le.fit_transform(b_raw).astype(np.int32, copy=False)
    n_classes = max(1, int(b.max()) + 1)

    # optional supervised labels
    y: Optional[np.ndarray]
    n_labels = 0
    if true_label_key is not None and true_label_key in adata.obs:
        y, n_labels, _ = _get_semi_labels(
            adata,
            true_label_key,
            unknown_aliases=("unknown","Unknown","unlabeled","unlabelled","UNK","NA","NaN","None","")
        )
    else:
        y, n_labels = None, 0

    # reference batch selection
    if str(reference).lower() == "largest":
        vals, counts = np.unique(b_raw, return_counts=True)
        ref_label_raw = vals[np.argmax(counts)]
        ref_label_enc = int(np.where(le.classes_ == ref_label_raw)[0][0])
    else:
        # if integer was passed, use it; else fallback to 0
        try:
            ref_label_enc = int(reference)
        except Exception:
            ref_label_enc = int(0)

    X = X0.copy()
    mu0 = X.mean(0)
    sd0 = X.std(0) + 1e-8

    d0, idx0 = _knn_graph(X0, k=max(15, k_local + 5), use_gpu=use_gpu, device=gpu_device)
    knn_mean_dist0 = d0.mean(axis=1) if d0.size else np.zeros(len(X), dtype=X.dtype)
    dens0 = (knn_mean_dist0 - (knn_mean_dist0.min() if len(knn_mean_dist0) else 0.0)) / (
        _ptp(knn_mean_dist0) + 1e-12
    )

    H0_i, _ = _neighbor_batch_entropy_per_cell(
        X0, b, k=min(15, k_eval), use_gpu=use_gpu, device=gpu_device
    )
    H0_max = np.log(n_classes + 1e-12)
    H0_norm = np.clip(H0_i / (H0_max + 1e-12), 0.0, 1.0)
    bridge_score = 0.5 * H0_norm + 0.5 * dens0

    smin_i = _lerp(smin_bulk, smin_bridge, bridge_score).astype(np.float32)
    smax_i = _lerp(smax_bulk, smax_bridge, bridge_score).astype(np.float32)

    mix0 = _neighbor_batch_entropy_per_cell(
        X0, b, k=k_eval, use_gpu=use_gpu, device=gpu_device
    )[1]
    strain0 = _graph_strain(X0, idx0, d0)
    best = dict(J=-np.inf, X=X.copy(), mix=mix0, overlap0=1.0, strain=strain0, it=0)

    if verbose:
        backend = "FAISS-GPU" if (_FAISS_AVAIL and use_gpu and _FAISS_GPU) else (
            "FAISS-CPU" if _FAISS_AVAIL else "sklearn"
        )
        print(f"[baseline] KNN backend={backend} mix={mix0:.4f} strain={strain0:.5f}")

    no_imp = 0
    for it in range(1, max_iter + 1):
        t = (it - 1) / max(1, max_iter - 1)
        lambda_graph0 = _lerp(lambda0_hi, lambda0_lo, t)
        overlap0_floor = _lerp(overlap0_lo, overlap0_hi, t)
        step = _lerp(step_lo, step_hi, t)
        cost_clip_q = _lerp(q_start, q_end, t)

        ref_mode = str(reference).lower()
        if ref_mode == "union":
            R, packs, _ = _compute_prototypes_union(
                X, b, K_ref, K_batch, random_state + it, use_gpu=use_gpu, device=gpu_device, y=y
            )
        else:
            R, packs, _ = _compute_prototypes(
                X, b, ref_label_enc, K_ref, K_batch, random_state + it, use_gpu=use_gpu, device=gpu_device, y=y
            )

        shift = np.zeros_like(X, dtype=X.dtype)
        alpha = np.ones(len(X), dtype=X.dtype)

        # OT barycentric mapping (supervised-aware)
        for pack in packs:
            # packs elements are either (idx, Bi, nn_idx, None) or (idx, Bi, nn_idx, cls)
            if len(pack) == 4:
                idx, Bi, nn_idx, cls = pack
            else:
                idx, Bi, nn_idx = pack  # type: ignore
                cls = None

            if len(Bi) == 0:
                continue

            if isinstance(R, dict):
                Rc = R.get(int(cls), None) if cls is not None else None
                if Rc is None or len(Rc) == 0:
                    # if no prototypes for this class, skip transport (identity)
                    continue
            else:
                Rc = R

            Bi_to_R = _ot_barycentric_gpu(
                Bi,
                Rc,  # ndarray (class-specific or global)
                reg=reg,
                reg_m=reg_m,
                cost_clip_q=cost_clip_q,
                clip_big=50.0,
                ot_backend=ot_backend,
                iters=1000,
                tol=1e-6,
                use_gpu=use_gpu,
                gpu_device=gpu_device,
            )
            disp_proto = Bi_to_R - Bi
            norm_move = np.linalg.norm(disp_proto, axis=1)
            s_dist = 1.0 / (1.0 + (norm_move / (norm_move.std() + 1e-8)))
            alpha_i = s_dist[nn_idx] * (1.0 - 0.35 * bridge_score[idx])
            alpha[idx] = alpha_i.astype(X.dtype, copy=False)
            shift[idx] = disp_proto[nn_idx].astype(X.dtype, copy=False)

        sharp = np.zeros_like(X)
        if sharpen > 0:
            sharp = _cluster_sharpen_field(
                X,
                K=K_pseudo,
                seed=abs(random_state) + 31,
                pull=pull,
                push=push,
                bridge_score=bridge_score,
                gate=0.7,
                use_gpu=use_gpu,
                device=gpu_device,
            )
        shift = alpha[:, None] * shift + (sharpen * sharp if sharpen > 0 else 0.0)

        # # ---- NEW: supervised pull/repulse boosts class compactness (optional) ----
        if (y is not None) and np.any(y >= 0) and (lam_sup > 0.0 or lam_repulse > 0.0):
            C = _class_means(X, y, n_labels)
            known = (y >= 0)
            # only apply where class mean is valid
            known &= ~np.isnan(C[np.clip(y, 0, n_labels - 1)]).any(axis=1)
            if known.any():
                if lam_sup > 0.0:
                    pull_vec = (C[y[known]] - X[known]) * (1.0 - 0.20 * bridge_score[known])[:, None]
                    shift[known] += lam_sup * pull_vec
                if lam_repulse > 0.0 and n_labels > 1:
                    near_other = _nearest_other_class_index(X, y, C)
                    ok = known & (near_other >= 0)
                    if ok.any():
                        repulse_vec = (X[ok] - C[near_other[ok]]) * (1.0 - 0.20 * bridge_score[ok])[:, None]
                        shift[ok] += lam_repulse * repulse_vec

        

        # rare-friendly smoothing: avoid oversmoothing sparse islands
        dens0_q = np.quantile(dens0, 0.85) if len(dens0) else 1.0
        rare_mask = dens0 >= dens0_q
        if lambda_graph0 > 0 and idx0.size > 0:
            shift_sm = _smooth_by_knn(shift, idx0, lam=lambda_graph0)
            shift[~rare_mask] = shift_sm[~rare_mask]

        shift *= (1.0 - 0.12 * bridge_score)[:, None]
        move = _cap_step_local(step * shift, knn_mean_dist0, max_step_local=max_step_local)
        move = _guard_edge_stretch_weighted(
            X, move, idx0, d0, smin_i, smax_i, rounds=2
        )

        Xcand = X + move
        if postscale:
            Xcand = (Xcand - Xcand.mean(0)) * (sd0 / (Xcand.std(0) + 1e-8)) + mu0

        mix_c = _neighbor_batch_entropy_per_cell(
            Xcand, b, k=k_eval, use_gpu=use_gpu, device=gpu_device
        )[1]
        overlap0 = _knn_overlap(
            X0,
            Xcand,
            k=k_eval,
            subsample=eval_subsample,
            rng=random_state + it,
            use_gpu=use_gpu,
            device=gpu_device,
        )
        strain_c = _graph_strain(Xcand, idx0, d0)

        penalty_floor = penalty_gamma * max(0.0, float(overlap0_floor - overlap0)) ** 2
        penalty_rel = 0.45 * max(0.0, float(best["overlap0"] - overlap0))
        J = (
            (mix_c - mix0)
            + (w_overlap * overlap0)
            - (w_strain * (strain_c - strain0))
            - (penalty_floor + penalty_rel)
        )

        if J > best["J"] + tol:
            X = Xcand
            if J > best["J"]:
                best.update(J=J, X=X.copy(), mix=mix_c, overlap0=overlap0, strain=strain_c, it=it)
            no_imp = 0
        else:
            no_imp += 1

        if verbose:
            print(
                f"[iter {it:02d}] mix={mix_c:.3f} overlap0={overlap0:.3f} "
                f"strain={strain_c:.5f} floor~{overlap0_floor:.3f} J={J:.3f} "
                f"best_it={best['it']}"
            )

        if no_imp >= patience:
            if verbose:
                print("[early stop] plateau reached.")
            break

    X_best = best["X"]
    n_obs = len(X_best)
    if trust_subsample is not None and n_obs > trust_subsample:
        rng_tw = np.random.default_rng(random_state)
        sample_tw = rng_tw.choice(n_obs, size=trust_subsample, replace=False)
        X0_eval = X0[sample_tw]
        X_best_eval = X_best[sample_tw]
    elif eval_subsample is not None and n_obs > eval_subsample:
        rng_tw = np.random.default_rng(random_state + 17)
        sample_tw = rng_tw.choice(n_obs, size=eval_subsample, replace=False)
        X0_eval = X0[sample_tw]
        X_best_eval = X_best[sample_tw]
    else:
        X0_eval = X0
        X_best_eval = X_best

    k_tw = min(k_eval, max(1, len(X_best_eval) - 1))
    tw = _trustworthiness_score(
        X0_eval,
        X_best_eval,
        n_neighbors=k_tw,
        use_gpu=use_gpu,
        gpu_device=gpu_device,
    )
    if verbose:
        print(
            f"[final] it*={best['it']} mix={best['mix']:.3f} "
            f"overlap0={best['overlap0']:.3f} strain={best['strain']:.5f} tw={tw:.3f}"
        )

    adata.obsm[out_key] = X_best
    return adata, dict(mix=best["mix"], overlap0=best["overlap0"], strain=best["strain"], tw=tw, it=best["it"]) 


# -------------------- Example usage --------------------
# Unsupervised: RNA
# adata, metrics = integrate_ot(
#     adata, obsm_key="X_pca", batch_key="batch", out_key="scBIOT_OT", reference='union', 
#     # OT
#     K_ref=1024, K_batch=448, reg=0.03, reg_m=0.40,
#     # Connectivity (relaxed)
#     sharpen=0.2, K_pseudo=24, pull=0.8, push=0.3,
#     lambda0_hi=0.5, lambda0_lo=0.35,
#     smin_bulk=0.75, smax_bulk=1.65,
#     smin_bridge=0.85, smax_bridge=1.25,
#     max_step_local=1.0,
#     step_lo=0.75, step_hi=0.95,
#     q_start=0.80, q_end=0.90,
#     overlap0_lo=0.60, overlap0_hi=0.7,
#     w_overlap=0.2, w_strain=1.0, penalty_gamma=1.5,
#     verbose=True
# )
# print(metrics)

# Supervised: RNA
# adata, metrics = integrate_ot(
#     adata, obsm_key="X_pca", batch_key="batch", out_key="scBIOT_OT", reference="union",
#     # OT
#     K_ref=1024, K_batch=448, reg=0.03, reg_m=0.40,
#     # Connectivity (relaxed)
#     sharpen=0.15, K_pseudo=24, pull=0.75, push=0.30,
#     lambda0_hi=0.50, lambda0_lo=0.35,
#     smin_bulk=0.75, smax_bulk=1.65,
#     smin_bridge=0.85, smax_bridge=1.25,
#     max_step_local=1.0,
#     step_lo=0.75, step_hi=0.95,
#     q_start=0.80, q_end=0.90,
#     overlap0_lo=0.60, overlap0_hi=0.70,
#     w_overlap=0.20, w_strain=1.0, penalty_gamma=1.5,
#     # --- supervised ---
#     true_label_key="semi_cell_type",
#     lam_sup=0.60,
#     lam_repulse=0.18,
#     use_gpu=True, ot_backend="torch", verbose=True
# )
# print(metrics)

# unsupervised: ATAC
# adata, metrics = integrate_ot(
#     adata,
#     obsm_key="X_lsi",
#     batch_key="batchname_all",
#     out_key="scBIOT_OT",
#     reference="largest",   
#     K_ref=960,
#     K_batch=360,
#     reg=0.036,
#     reg_m=0.30,
#     sharpen=0.15,
#     K_pseudo=20,
#     pull=0.76,
#     push=0.24,
#     lambda0_hi=0.58,
#     lambda0_lo=0.42,
#     smin_bulk=0.78,
#     smax_bulk=1.50,
#     smin_bridge=0.90,
#     smax_bridge=1.16,
#     max_step_local=0.88,
#     step_lo=0.70,
#     step_hi=0.88,
#     q_start=0.78,
#     q_end=0.885,
#     overlap0_lo=0.64,
#     overlap0_hi=0.73,
#     w_overlap=0.30,
#     w_strain=1.0,
#     penalty_gamma=1.30,
#     verbose=True,
#     use_gpu=True,
#     gpu_device=0,
#     ot_backend="torch",
# )
# print(metrics)
