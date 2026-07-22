from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy.stats import rankdata
from sklearn.manifold import trustworthiness as sk_trustworthiness
from sklearn.neighbors import NearestNeighbors

from ..utils.optional_backends import load_faiss

try:
    from ..utils.ot_helpers import _trustworthiness_score as _torch_trustworthiness_score

    _FAST_TRUSTWORTHINESS_AVAIL = True
except Exception:
    _torch_trustworthiness_score = None  # type: ignore
    _FAST_TRUSTWORTHINESS_AVAIL = False

faiss, _FAISS_AVAIL, _FAISS_GPU = load_faiss()

try:
    import torch  # type: ignore

    _TORCH_AVAIL = True
    _TORCH_GPU = bool(torch.cuda.is_available())
except Exception:
    torch = None  # type: ignore
    _TORCH_AVAIL = False
    _TORCH_GPU = False

_FAISS_GPU_RESOURCES: dict[int, "faiss.StandardGpuResources"] = {}
_FAISS_GPU_MIN_ROWS = max(1, int(os.getenv("SCBIOT_FAISS_GPU_MIN_ROWS", "2048")))
_TORCH_GPU_MIN_ROWS = max(1, int(os.getenv("SCBIOT_TORCH_GPU_MIN_ROWS", "4096")))
_TORCH_CDIST_CHUNK = max(64, int(os.getenv("SCBIOT_TORCH_CDIST_CHUNK", "2048")))


@dataclass
class StructureMetricResult:
    score: float
    per_cell: np.ndarray
    valid_mask: np.ndarray
    name: str


def _as_faiss_f32(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32, order="C")
    if X.ndim != 2:
        raise ValueError("Expected 2D array for FAISS KNN.")
    return X


def _get_faiss_index(d: int, use_gpu: bool = True, device: int = 0):
    if not _FAISS_AVAIL:
        raise RuntimeError("FAISS backend requested but faiss is not available.")
    cpu_index = faiss.IndexFlatL2(d)
    if use_gpu and _FAISS_GPU:
        res = _FAISS_GPU_RESOURCES.get(device)
        if res is None:
            res = faiss.StandardGpuResources()
            _FAISS_GPU_RESOURCES[device] = res
        return faiss.index_cpu_to_gpu(res, int(device), cpu_index)
    return cpu_index


def _faiss_knn_search(
    query: np.ndarray,
    base: np.ndarray,
    k: int,
    use_gpu: bool = True,
    device: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    xq = _as_faiss_f32(query)
    xb = _as_faiss_f32(base)
    index = _get_faiss_index(xb.shape[1], use_gpu=use_gpu, device=device)
    index.add(xb)
    D2, I = index.search(xq, int(k))
    return D2, I


def _torch_knn_search(
    query: np.ndarray,
    base: np.ndarray,
    k: int,
    device: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    if not (_TORCH_AVAIL and _TORCH_GPU):
        raise RuntimeError("Torch GPU backend requested but torch.cuda is not available.")

    same_space = (query is base) or np.shares_memory(np.asarray(query), np.asarray(base))
    xq = np.asarray(query, dtype=np.float32, order="C")
    xb = np.asarray(base, dtype=np.float32, order="C")
    if xq.ndim != 2 or xb.ndim != 2:
        raise ValueError("Expected 2D arrays for torch KNN.")

    dev = torch.device(f"cuda:{int(device)}")
    tq = torch.as_tensor(xq, dtype=torch.float32, device=dev)
    tb = torch.as_tensor(xb, dtype=torch.float32, device=dev)

    nq = tq.shape[0]
    k_eff = min(int(k), int(tb.shape[0]))
    if k_eff <= 0:
        raise ValueError("k must be >= 1.")

    out_dist = np.empty((nq, k_eff), dtype=np.float64)
    out_ind = np.empty((nq, k_eff), dtype=np.int64)

    same_space = same_space and (xq.shape[0] == xb.shape[0])
    row_ids = torch.arange(nq, device=dev, dtype=torch.long)

    for start in range(0, nq, _TORCH_CDIST_CHUNK):
        stop = min(start + _TORCH_CDIST_CHUNK, nq)
        d = torch.cdist(tq[start:stop], tb, p=2.0)
        if same_space:
            local_rows = torch.arange(stop - start, device=dev, dtype=torch.long)
            d[local_rows, row_ids[start:stop]] = float("inf")

        vals, idx = torch.topk(d, k=k_eff, dim=1, largest=False, sorted=True)
        out_dist[start:stop] = vals.detach().cpu().numpy().astype(np.float64, copy=False)
        out_ind[start:stop] = idx.detach().cpu().numpy().astype(np.int64, copy=False)

    return out_dist, out_ind


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _group_indices(labels: Iterable[Any]) -> dict[Any, np.ndarray]:
    labels = np.asarray(list(labels), dtype=object)
    out: dict[Any, np.ndarray] = {}
    for lab in np.unique(labels):
        out[lab] = np.flatnonzero(labels == lab).astype(np.int64)
    return out


def _topk_neighbors(
    X: np.ndarray,
    k: int,
    metric: str = "euclidean",
    exclude_self: bool = True,
) -> np.ndarray:
    n = X.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 cells.")

    k_query = min(k + int(exclude_self), n)
    ind: np.ndarray

    if metric == "euclidean" and _FAISS_AVAIL:
        use_gpu = _FAISS_GPU and n >= _FAISS_GPU_MIN_ROWS
        _, ind = _faiss_knn_search(X, X, k_query, use_gpu=use_gpu, device=0)
        ind = ind.astype(np.int64, copy=False)
    elif metric == "euclidean" and _TORCH_AVAIL and _TORCH_GPU and n >= _TORCH_GPU_MIN_ROWS:
        _, ind = _torch_knn_search(X, X, k_query, device=0)
    else:
        nn = NearestNeighbors(n_neighbors=k_query, metric=metric)
        nn.fit(X)
        ind = nn.kneighbors(X, return_distance=False).astype(np.int64, copy=False)

    if not exclude_self:
        return ind[:, : min(k, n)]

    k_eff = min(k, n - 1)
    self_mask = ind == np.arange(n, dtype=np.int64)[:, None]
    if np.all(np.sum(self_mask, axis=1) == 1):
        compact = ind[~self_mask].reshape(n, -1)
        return compact[:, :k_eff]

    out = np.empty((n, k_eff), dtype=np.int64)
    for i in range(n):
        row = ind[i]
        row = row[row != i]
        out[i] = row[:k_eff]
    return out


def _knn_distances(
    X: np.ndarray,
    k: int,
    metric: str = "euclidean",
    exclude_self: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 cells.")

    k_query = min(k + int(exclude_self), n)
    if metric == "euclidean" and _FAISS_AVAIL:
        use_gpu = _FAISS_GPU and n >= _FAISS_GPU_MIN_ROWS
        D2, ind = _faiss_knn_search(X, X, k_query, use_gpu=use_gpu, device=0)
        dist = np.sqrt(np.maximum(D2, 0.0)).astype(np.float64, copy=False)
        ind = ind.astype(np.int64, copy=False)
    elif metric == "euclidean" and _TORCH_AVAIL and _TORCH_GPU and n >= _TORCH_GPU_MIN_ROWS:
        dist, ind = _torch_knn_search(X, X, k_query, device=0)
    else:
        nn = NearestNeighbors(n_neighbors=k_query, metric=metric)
        nn.fit(X)
        dist, ind = nn.kneighbors(X, return_distance=True)
        dist = dist.astype(np.float64, copy=False)
        ind = ind.astype(np.int64, copy=False)

    if not exclude_self:
        k_eff = min(k, n)
        return dist[:, :k_eff], ind[:, :k_eff]

    k_eff = min(k, n - 1)
    self_mask = ind == np.arange(n, dtype=np.int64)[:, None]
    if np.all(np.sum(self_mask, axis=1) == 1):
        compact_ind = ind[~self_mask].reshape(n, -1)
        compact_dist = dist[~self_mask].reshape(n, -1)
        return compact_dist[:, :k_eff], compact_ind[:, :k_eff]

    out_dist = np.empty((n, k_eff), dtype=np.float64)
    out_ind = np.empty((n, k_eff), dtype=np.int64)
    for i in range(n):
        row_ind = ind[i]
        row_dist = dist[i]
        keep = row_ind != i
        row_ind = row_ind[keep][:k_eff]
        row_dist = row_dist[keep][:k_eff]
        out_ind[i] = row_ind
        out_dist[i] = row_dist
    return out_dist, out_ind


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if a.size < 2 or b.size < 2:
        return np.nan
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return np.nan

    am = a.mean()
    bm = b.mean()
    av = a - am
    bv = b - bm
    denom = np.sqrt(np.sum(av * av) * np.sum(bv * bv))
    if denom <= 0:
        return np.nan
    return float(np.sum(av * bv) / denom)


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        return np.nan
    ra = rankdata(a, method="average")
    rb = rankdata(b, method="average")
    return _safe_corr(ra, rb)


def _rowwise_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape:
        raise ValueError("a and b must be 2D arrays with identical shape.")
    if a.shape[1] < 2:
        return np.full(a.shape[0], np.nan, dtype=np.float64)

    av = a - np.mean(a, axis=1, keepdims=True)
    bv = b - np.mean(b, axis=1, keepdims=True)
    denom = np.sqrt(np.sum(av * av, axis=1) * np.sum(bv * bv, axis=1))
    out = np.full(a.shape[0], np.nan, dtype=np.float64)
    keep = denom > 0
    out[keep] = np.sum(av[keep] * bv[keep], axis=1) / denom[keep]
    return out


def _rowwise_spearman(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    try:
        ra = rankdata(a, method="average", axis=1)
        rb = rankdata(b, method="average", axis=1)
        return _rowwise_corr(ra, rb)
    except TypeError:
        out = np.full(a.shape[0], np.nan, dtype=np.float64)
        for i in range(a.shape[0]):
            out[i] = _safe_spearman(a[i], b[i])
        return out


def _rowwise_jaccard_with_topm(
    neighbors_a: np.ndarray,
    neighbors_b: np.ndarray,
    top_m: np.ndarray,
    *,
    chunk_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    neighbors_a = np.asarray(neighbors_a, dtype=np.int64)
    neighbors_b = np.asarray(neighbors_b, dtype=np.int64)
    top_m = np.asarray(top_m, dtype=np.int64)

    if neighbors_a.ndim != 2 or neighbors_b.ndim != 2:
        raise ValueError("neighbors_a and neighbors_b must be 2D.")
    if neighbors_a.shape[0] != neighbors_b.shape[0] or neighbors_a.shape[0] != top_m.shape[0]:
        raise ValueError("neighbors_a, neighbors_b, and top_m must share n_rows.")

    n = neighbors_a.shape[0]
    k_max = min(neighbors_a.shape[1], neighbors_b.shape[1])
    top_m = np.clip(top_m, 0, k_max)

    scores = np.full(n, np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    if n == 0 or k_max == 0:
        return scores, valid

    base_cols = np.arange(k_max, dtype=np.int64)[None, :]
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        a = neighbors_a[start:stop, :k_max]
        b = neighbors_b[start:stop, :k_max]
        m = top_m[start:stop]

        keep_rows = m > 0
        if not np.any(keep_rows):
            continue

        mask = base_cols < m[:, None]
        eq = a[:, :, None] == b[:, None, :]
        inter = np.sum(np.any(eq & mask[:, None, :], axis=2) & mask, axis=1, dtype=np.int64)
        union = (2 * m) - inter
        good = keep_rows & (union > 0)
        if not np.any(good):
            continue

        rows = np.flatnonzero(good) + start
        scores[rows] = inter[good] / union[good]
        valid[rows] = True
    return scores, valid


def _global_structure_result(
    score: float,
    n_cells: int,
    name: str,
) -> StructureMetricResult:
    per_cell = np.full(n_cells, float(score), dtype=np.float64)
    valid = np.isfinite(per_cell)
    return StructureMetricResult(score=float(score), per_cell=per_cell, valid_mask=valid, name=name)


def _subset_topk_neighbors(
    X: np.ndarray,
    subset_idx: np.ndarray,
    k: int,
    metric: str = "euclidean",
) -> np.ndarray:
    subset_idx = np.asarray(subset_idx, dtype=np.int64)
    Xs = X[subset_idx]
    neigh_local = _topk_neighbors(Xs, k=k, metric=metric, exclude_self=True)
    return subset_idx[neigh_local]


def trustworthiness(
    X_pre: np.ndarray,
    X_post: np.ndarray,
    *,
    n_neighbors: int = 15,
    metric: str = "euclidean",
    sample_size: int | None = None,
    name: str = "Trust-worthiness",
) -> StructureMetricResult:
    """
    Embedding trustworthiness in [0, 1], higher is better.

    If ``sample_size`` is set and smaller than ``n_cells``, computes an
    approximation on a deterministic random subset for speed.
    """
    X_pre = _to_numpy(X_pre)
    X_post = _to_numpy(X_post)

    n = X_pre.shape[0]
    if X_post.shape[0] != n:
        raise ValueError("X_pre and X_post must have matching n_cells.")
    if n < 2:
        raise ValueError("Need at least 2 cells.")

    X_pre_eval = X_pre
    X_post_eval = X_post
    n_eval = n
    if sample_size is not None:
        s = int(sample_size)
        if s < 2:
            raise ValueError("sample_size must be >= 2 when provided.")
        if s < n:
            rng = np.random.default_rng(0)
            idx = rng.choice(n, size=s, replace=False)
            X_pre_eval = X_pre[idx]
            X_post_eval = X_post[idx]
            n_eval = s

    k = int(n_neighbors)
    if k <= 0:
        raise ValueError("n_neighbors must be >= 1.")
    if k >= n_eval / 2:
        k = max(1, (n_eval // 2) - 1)
    k = min(k, n_eval - 1)

    if metric == "euclidean" and _FAST_TRUSTWORTHINESS_AVAIL:
        score = float(
            _torch_trustworthiness_score(
                X_pre_eval,
                X_post_eval,
                n_neighbors=k,
                use_gpu=True,
                gpu_device=0,
            )
        )
    else:
        score = float(sk_trustworthiness(X_pre_eval, X_post_eval, n_neighbors=k, metric=metric))
    return _global_structure_result(score=score, n_cells=n, name=name)


def calculate_mean_ja(
    X_pre: np.ndarray,
    X_post: np.ndarray,
    *,
    k: int = 30,
    metric: str = "euclidean",
    min_valid_k: int = 1,
    name: str = "Jaccard index",
) -> StructureMetricResult:
    """
    Mean neighborhood Jaccard agreement between pre and post embeddings.
    """
    X_pre = _to_numpy(X_pre)
    X_post = _to_numpy(X_post)

    n = X_pre.shape[0]
    if X_post.shape[0] != n:
        raise ValueError("X_pre and X_post must have matching n_cells.")

    neigh_pre = _topk_neighbors(X_pre, k=k, metric=metric, exclude_self=True)
    neigh_post = _topk_neighbors(X_post, k=k, metric=metric, exclude_self=True)

    k_eff = min(neigh_pre.shape[1], neigh_post.shape[1])
    top_m = np.full(n, k_eff if k_eff >= min_valid_k else 0, dtype=np.int64)
    scores, valid = _rowwise_jaccard_with_topm(neigh_pre, neigh_post, top_m)

    score = float(np.nanmean(scores)) if valid.any() else np.nan
    return StructureMetricResult(score=score, per_cell=scores, valid_mask=valid, name=name)


def within_label_neighborhood_preservation(
    X_pre: np.ndarray,
    X_post: np.ndarray,
    labels: Iterable[Any],
    *,
    k: int = 30,
    metric: str = "euclidean",
    min_label_size: int = 10,
    min_valid_k: int = 5,
    macro_average: bool = True,
    name: str = "WLNP",
) -> StructureMetricResult:
    """
    Within-Label Neighborhood Preservation.
    """
    X_pre = _to_numpy(X_pre)
    X_post = _to_numpy(X_post)
    labels = np.asarray(list(labels), dtype=object)

    n = X_pre.shape[0]
    if X_post.shape[0] != n or len(labels) != n:
        raise ValueError("X_pre, X_post, and labels must have matching n_cells.")

    scores = np.full(n, np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)

    label_scores = []
    label_groups = _group_indices(labels)

    for idx in label_groups.values():
        if len(idx) < max(min_label_size, 2):
            continue

        k_eff = min(k, len(idx) - 1)
        if k_eff < min_valid_k:
            continue

        neigh_pre = _subset_topk_neighbors(X_pre, idx, k=k_eff, metric=metric)
        neigh_post = _subset_topk_neighbors(X_post, idx, k=k_eff, metric=metric)

        top_m = np.full(len(idx), k_eff, dtype=np.int64)
        local_scores, local_valid = _rowwise_jaccard_with_topm(neigh_pre, neigh_post, top_m)
        if local_valid.any():
            scores[idx[local_valid]] = local_scores[local_valid]
            valid[idx[local_valid]] = True
            label_scores.append(float(np.mean(local_scores[local_valid])))

    if macro_average:
        score = float(np.mean(label_scores)) if label_scores else np.nan
    else:
        score = float(np.nanmean(scores)) if valid.any() else np.nan

    return StructureMetricResult(score=score, per_cell=scores, valid_mask=valid, name=name)


def branch_transition_preservationwithin_label_neighborhood_preservation(
    X_pre: np.ndarray,
    X_post: np.ndarray,
    labels: Iterable[Any],
    *,
    k: int = 30,
    metric: str = "euclidean",
    min_label_size: int = 10,
    min_valid_k: int = 5,
    macro_average: bool = True,
    name: str = "WLNP",
) -> StructureMetricResult:
    """
    Backward-compatible alias for within_label_neighborhood_preservation.
    """
    return within_label_neighborhood_preservation(
        X_pre,
        X_post,
        labels=labels,
        k=k,
        metric=metric,
        min_label_size=min_label_size,
        min_valid_k=min_valid_k,
        macro_average=macro_average,
        name=name,
    )


def local_geometry_distortion_score(
    X_pre: np.ndarray,
    X_post: np.ndarray,
    labels: Iterable[Any],
    *,
    k: int = 30,
    metric: str = "euclidean",
    min_label_size: int = 10,
    min_valid_k: int = 5,
    use_pre_neighbors_for_both: bool = True,
    rank_based: bool = False,
    macro_average: bool = True,
    name: str = "LGDS",
) -> StructureMetricResult:
    """
    Local Geometry Distortion Score.

    For each cell within its own label:
      1. choose same-label neighbors
      2. compare local distances in pre vs post
    """
    X_pre = _to_numpy(X_pre)
    X_post = _to_numpy(X_post)
    labels = np.asarray(list(labels), dtype=object)

    n = X_pre.shape[0]
    if X_post.shape[0] != n or len(labels) != n:
        raise ValueError("X_pre, X_post, and labels must have matching n_cells.")

    scores = np.full(n, np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)

    label_scores = []
    label_groups = _group_indices(labels)

    for idx in label_groups.values():
        if len(idx) < max(min_label_size, 2):
            continue

        k_eff = min(k, len(idx) - 1)
        if k_eff < min_valid_k:
            continue

        Xpre_lab = X_pre[idx]
        Xpost_lab = X_post[idx]

        dpre, npre = _knn_distances(Xpre_lab, k=k_eff, metric=metric, exclude_self=True)
        if use_pre_neighbors_for_both:
            a = dpre
            b = np.linalg.norm(Xpost_lab[npre] - Xpost_lab[:, None, :], axis=2)
        else:
            dpost, _ = _knn_distances(Xpost_lab, k=k_eff, metric=metric, exclude_self=True)
            a = dpre
            b = dpost

        m = min(a.shape[1], b.shape[1])
        if m < min_valid_k:
            continue

        a = a[:, :m]
        b = b[:, :m]
        local_scores = _rowwise_spearman(a, b) if rank_based else _rowwise_corr(a, b)
        local_valid = np.isfinite(local_scores)
        if local_valid.any():
            scores[idx[local_valid]] = local_scores[local_valid]
            valid[idx[local_valid]] = True
            label_scores.append(float(np.mean(local_scores[local_valid])))

    if macro_average:
        score = float(np.mean(label_scores)) if label_scores else np.nan
    else:
        score = float(np.nanmean(scores)) if valid.any() else np.nan

    return StructureMetricResult(score=score, per_cell=scores, valid_mask=valid, name=name)


def cross_batch_within_label_neighborhood_preservation(
    X_pre: np.ndarray,
    X_post: np.ndarray,
    labels: Iterable[Any],
    batch: Iterable[Any],
    *,
    k: int = 30,
    metric: str = "euclidean",
    min_label_size: int = 10,
    min_valid_k: int = 5,
    max_candidates_multiplier: int = 5,
    macro_average: bool = True,
    name: str = "XB-WLNP",
) -> StructureMetricResult:
    """
    Cross-Batch Within-Label Neighborhood Preservation.

    Same-label, cross-batch-only version of WLNP.
    """
    X_pre = _to_numpy(X_pre)
    X_post = _to_numpy(X_post)
    labels = np.asarray(list(labels), dtype=object)
    batch = np.asarray(list(batch), dtype=object)

    n = X_pre.shape[0]
    if X_post.shape[0] != n or len(labels) != n or len(batch) != n:
        raise ValueError("X_pre, X_post, labels, batch must match n_cells.")

    scores = np.full(n, np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)

    label_scores = []
    label_groups = _group_indices(labels)

    for idx in label_groups.values():
        if len(idx) < max(min_label_size, 2):
            continue

        k_cand = min(max(k * max_candidates_multiplier, k), len(idx) - 1)
        if k_cand < min_valid_k:
            continue

        neigh_pre = _subset_topk_neighbors(X_pre, idx, k=k_cand, metric=metric)
        neigh_post = _subset_topk_neighbors(X_post, idx, k=k_cand, metric=metric)

        batch_local = batch[idx]
        cross_pre = batch[neigh_pre] != batch_local[:, None]
        cross_post = batch[neigh_post] != batch_local[:, None]

        order_pre = np.argsort(~cross_pre, axis=1, kind="stable")
        order_post = np.argsort(~cross_post, axis=1, kind="stable")

        pre_sorted = np.take_along_axis(neigh_pre, order_pre, axis=1)
        post_sorted = np.take_along_axis(neigh_post, order_post, axis=1)
        cross_pre_sorted = np.take_along_axis(cross_pre, order_pre, axis=1)
        cross_post_sorted = np.take_along_axis(cross_post, order_post, axis=1)

        pre_top = pre_sorted[:, :k]
        post_top = post_sorted[:, :k]
        pre_count = np.sum(cross_pre_sorted[:, :k], axis=1, dtype=np.int64)
        post_count = np.sum(cross_post_sorted[:, :k], axis=1, dtype=np.int64)

        top_m = np.minimum(np.minimum(pre_count, post_count), k).astype(np.int64, copy=False)
        top_m[top_m < min_valid_k] = 0

        local_scores, local_valid = _rowwise_jaccard_with_topm(pre_top, post_top, top_m)
        if local_valid.any():
            scores[idx[local_valid]] = local_scores[local_valid]
            valid[idx[local_valid]] = True
            label_scores.append(float(np.mean(local_scores[local_valid])))

    if macro_average:
        score = float(np.mean(label_scores)) if label_scores else np.nan
    else:
        score = float(np.nanmean(scores)) if valid.any() else np.nan

    return StructureMetricResult(score=score, per_cell=scores, valid_mask=valid, name=name)


__all__ = [
    "StructureMetricResult",
    "trustworthiness",
    "calculate_mean_ja",
    "within_label_neighborhood_preservation",
    "branch_transition_preservationwithin_label_neighborhood_preservation",
    "local_geometry_distortion_score",
    "cross_batch_within_label_neighborhood_preservation",
]
