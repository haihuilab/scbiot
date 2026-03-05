from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch

from .ot_helpers import _FAISS_AVAIL, _FAISS_GPU, _as_nd_f32_c, _knn_graph


class ResidualProjector(torch.nn.Module):
    def __init__(self, d_in: int, hidden: int = 512, res_scale: float = 0.05) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, d_in),
            torch.nn.LayerNorm(d_in),
        )
        self.res_scale = float(res_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.res_scale * self.net(x)


def pairs_from_coupling(
    P: object,
    topk: int = 10,
    symmetric: bool = True,
    max_pairs: int = 2_000_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if P is None:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)

    if isinstance(P, dict) and {"indices", "weights"}.issubset(P):
        idx = np.asarray(P["indices"], dtype=np.int64)
        w = np.asarray(P["weights"], dtype=np.float32)
    elif isinstance(P, tuple) and len(P) == 2:
        idx = np.asarray(P[0], dtype=np.int64)
        w = np.asarray(P[1], dtype=np.float32)
    elif sp.issparse(P):
        P_csr = P.tocsr()
        src_list = []
        dst_list = []
        w_list = []
        for i in range(P_csr.shape[0]):
            start = P_csr.indptr[i]
            end = P_csr.indptr[i + 1]
            if end <= start:
                continue
            row_idx = P_csr.indices[start:end]
            row_w = P_csr.data[start:end]
            if row_w.size > topk:
                keep = np.argpartition(row_w, -topk)[-topk:]
                row_idx = row_idx[keep]
                row_w = row_w[keep]
            row_sum = float(row_w.sum())
            if row_sum <= 0:
                continue
            row_w = row_w / row_sum
            src_list.append(np.full(row_idx.shape[0], i, dtype=np.int64))
            dst_list.append(row_idx.astype(np.int64, copy=False))
            w_list.append(row_w.astype(np.float32, copy=False))
        if not src_list:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
        src = np.concatenate(src_list)
        dst = np.concatenate(dst_list)
        w = np.concatenate(w_list)
        return _finalize_pairs(src, dst, w, symmetric=symmetric, max_pairs=max_pairs)
    else:
        P_arr = np.asarray(P)
        if P_arr.ndim != 2:
            raise ValueError("Coupling must be 2D.")
        topk = int(min(max(1, topk), P_arr.shape[1]))
        idx = np.argpartition(P_arr, -topk, axis=1)[:, -topk:]
        rows = np.arange(P_arr.shape[0])[:, None]
        w = P_arr[rows, idx].astype(np.float32, copy=False)

    row_sum = w.sum(axis=1, keepdims=True)
    w = w / (row_sum + 1e-12)
    src = np.repeat(np.arange(idx.shape[0], dtype=np.int64), idx.shape[1])
    dst = idx.reshape(-1).astype(np.int64, copy=False)
    w = w.reshape(-1).astype(np.float32, copy=False)
    return _finalize_pairs(src, dst, w, symmetric=symmetric, max_pairs=max_pairs)


def laplacian_from_knn(
    X: np.ndarray,
    k: int = 30,
    backend: str = "faiss",
    sym: bool = True,
) -> sp.csr_matrix:
    X = _as_nd_f32_c(X)
    n_obs = X.shape[0]
    if n_obs <= 1:
        return sp.csr_matrix((n_obs, n_obs), dtype=np.float32)

    k = int(max(1, min(k, n_obs - 1)))
    if backend == "faiss" and _FAISS_AVAIL:
        d, idx = _knn_graph(X, k=k, use_gpu=_FAISS_GPU, device=0)
    else:
        d, idx = _knn_graph(X, k=k, use_gpu=False, device=0)

    sigma = float(np.mean(d)) if d.size else 1.0
    if sigma <= 0:
        sigma = 1.0
    w = np.exp(-d / (sigma + 1e-8)).astype(np.float32, copy=False)

    row = np.repeat(np.arange(n_obs), idx.shape[1])
    col = idx.reshape(-1)
    data = w.reshape(-1)
    W = sp.csr_matrix((data, (row, col)), shape=(n_obs, n_obs), dtype=np.float32)
    if sym:
        W = W.maximum(W.T)
    W.setdiag(0.0)
    D = sp.diags(np.asarray(W.sum(axis=1)).ravel(), dtype=np.float32)
    return (D - W).tocsr()


def match_loss(Z: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    diff = Z[src] - Z[dst]
    return torch.mean(w * torch.sum(diff * diff, dim=1))


def laplacian_loss(Z: torch.Tensor, L_sparse: torch.Tensor) -> torch.Tensor:
    LZ = torch.sparse.mm(L_sparse, Z)
    return torch.sum(Z * LZ) / max(1, Z.shape[0])


def batch_orth_loss(Z: torch.Tensor, batch_codes: torch.Tensor) -> torch.Tensor:
    n_batches = int(batch_codes.max().item()) + 1 if batch_codes.numel() else 1
    B = torch.nn.functional.one_hot(batch_codes, num_classes=n_batches).to(dtype=Z.dtype)
    B = B - B.mean(dim=0, keepdim=True)
    proj = torch.matmul(B.T, Z)
    return torch.sum(proj * proj) / max(1, Z.shape[0])


def batch_orth_loss_per_label(
    Z: torch.Tensor,
    batch_codes: torch.Tensor,
    label_codes: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if label_codes.numel() == 0:
        return torch.tensor(0.0, device=Z.device, dtype=Z.dtype)
    valid = label_codes >= 0
    if not torch.any(valid):
        return torch.tensor(0.0, device=Z.device, dtype=Z.dtype)

    if weights is None:
        weights = torch.ones_like(label_codes, dtype=Z.dtype)
    else:
        weights = weights.to(dtype=Z.dtype)
    weights = torch.clamp(weights, min=0.0)
    total_w = torch.sum(weights[valid])
    if total_w <= 0:
        return torch.tensor(0.0, device=Z.device, dtype=Z.dtype)

    n_batches = int(batch_codes.max().item()) + 1 if batch_codes.numel() else 1
    loss = torch.tensor(0.0, device=Z.device, dtype=Z.dtype)
    labels = torch.unique(label_codes[valid])
    for lbl in labels:
        mask = valid & (label_codes == lbl)
        if not torch.any(mask):
            continue
        w = weights[mask]
        w_sum = torch.sum(w)
        if w_sum <= 0:
            continue
        B = torch.nn.functional.one_hot(batch_codes[mask], num_classes=n_batches).to(dtype=Z.dtype)
        B_mean = torch.sum(B * w[:, None], dim=0, keepdim=True) / (w_sum + 1e-12)
        B_centered = B - B_mean
        Z_w = Z[mask] * w[:, None]
        proj = torch.matmul(B_centered.T, Z_w)
        loss = loss + torch.sum(proj * proj)
    return loss / torch.clamp(total_w, min=1.0)


def variance_loss(Z: torch.Tensor, target_std: torch.Tensor) -> torch.Tensor:
    z_std = Z.std(dim=0, unbiased=False)
    return torch.mean((z_std - target_std) ** 2)


def train_projector(
    X: np.ndarray,
    batch: np.ndarray,
    pairs: Tuple[np.ndarray, np.ndarray, np.ndarray],
    L: sp.csr_matrix,
    strength: float,
    device: torch.device | str,
    epochs: int = 2,
    pair_batch: int = 65536,
    seed: int = 0,
    label_codes: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    X = _as_nd_f32_c(X)
    n_obs = X.shape[0]
    if n_obs == 0:
        return X

    s = float(strength)
    if s <= 0:
        return X

    src_idx, dst_idx, weight = pairs
    if src_idx.size == 0:
        return X

    res_scale = min(0.20, 0.05 * s)
    w_match = 1.0
    w_lap = 0.10 * s
    w_batch = 0.15 * s if label_codes is not None else 0.10 * s
    w_var = 0.01 * s

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device_t = torch.device(device)
    X_t = torch.as_tensor(X, device=device_t, dtype=torch.float32)
    batch_t = torch.as_tensor(batch, device=device_t, dtype=torch.int64)
    label_t = None
    weight_t = None
    if label_codes is not None:
        label_t = torch.as_tensor(label_codes, device=device_t, dtype=torch.int64)
        if weights is not None:
            weight_t = torch.as_tensor(weights, device=device_t, dtype=torch.float32)
    target_std = torch.as_tensor(1.05 * X.std(axis=0), device=device_t, dtype=torch.float32)

    projector = ResidualProjector(d_in=X.shape[1], hidden=512, res_scale=res_scale).to(device_t)
    optimizer = torch.optim.Adam(projector.parameters(), lr=1e-3)

    L_t = _torch_sparse_from_scipy(L, device_t)

    edges = _sample_laplacian_edges(L, max_edges=200_000, seed=seed)
    base_neighbor_dist = _mean_edge_distance(X, edges) if edges.size else None

    prev_loss: Optional[float] = None
    n_pairs = int(src_idx.shape[0])
    pair_batch = int(max(1024, pair_batch))

    for _ in range(int(max(1, epochs))):
        perm = rng.permutation(n_pairs) if n_pairs > pair_batch else np.arange(n_pairs)
        epoch_loss = 0.0
        n_steps = 0

        for start in range(0, n_pairs, pair_batch):
            sel = perm[start : start + pair_batch]
            src = torch.as_tensor(src_idx[sel], device=device_t, dtype=torch.int64)
            dst = torch.as_tensor(dst_idx[sel], device=device_t, dtype=torch.int64)
            w = torch.as_tensor(weight[sel], device=device_t, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)
            use_amp = device_t.type == "cuda"
            with torch.amp.autocast("cuda", enabled=use_amp):
                Z = projector(X_t)
                loss = w_match * match_loss(Z, src, dst, w)
                if label_t is None:
                    loss = loss + w_batch * batch_orth_loss(Z, batch_t)
                else:
                    loss = loss + w_batch * batch_orth_loss_per_label(Z, batch_t, label_t, weight_t)
                loss = loss + w_var * variance_loss(Z, target_std)
            if w_lap > 0:
                with torch.amp.autocast("cuda", enabled=False):
                    Z_lap = Z.float()
                    loss = loss + w_lap * laplacian_loss(Z_lap, L_t)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            n_steps += 1

        if n_steps:
            epoch_loss /= n_steps
        if prev_loss is not None and epoch_loss >= prev_loss - 1e-4 * max(1.0, prev_loss):
            break
        prev_loss = epoch_loss

    with torch.no_grad():
        Z = projector(X_t).detach().cpu().to(dtype=torch.float32).numpy()

    if base_neighbor_dist is not None:
        new_dist = _mean_edge_distance(Z, edges)
        if new_dist is not None and new_dist < 0.7 * base_neighbor_dist:
            res_scale *= 0.5
            projector.res_scale = float(res_scale)
            with torch.no_grad():
                Z = projector(X_t).detach().cpu().to(dtype=torch.float32).numpy()

    return Z


def _torch_sparse_from_scipy(L: sp.csr_matrix, device: torch.device) -> torch.Tensor:
    if not sp.issparse(L):
        L = sp.csr_matrix(L, dtype=np.float32)
    coo = L.tocoo()
    if coo.nnz == 0:
        return torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.int64, device=device),
            torch.zeros((0,), dtype=torch.float32, device=device),
            size=coo.shape,
        )
    indices = torch.as_tensor(np.vstack([coo.row, coo.col]), device=device, dtype=torch.int64)
    values = torch.as_tensor(coo.data, device=device, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, size=coo.shape).coalesce()


def _sample_laplacian_edges(L: sp.csr_matrix, max_edges: int, seed: int) -> np.ndarray:
    if not sp.issparse(L) or L.nnz == 0:
        return np.zeros((0, 2), dtype=np.int64)
    coo = L.tocoo()
    mask = coo.row != coo.col
    rows = coo.row[mask]
    cols = coo.col[mask]
    if rows.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    edges = np.stack([rows, cols], axis=1).astype(np.int64, copy=False)
    if edges.shape[0] > max_edges:
        rng = np.random.default_rng(seed)
        keep = rng.choice(edges.shape[0], size=max_edges, replace=False)
        edges = edges[keep]
    return edges


def _mean_edge_distance(X: np.ndarray, edges: np.ndarray) -> Optional[float]:
    if edges.size == 0:
        return None
    diff = X[edges[:, 0]] - X[edges[:, 1]]
    dist = np.linalg.norm(diff, axis=1)
    return float(np.mean(dist)) if dist.size else None


def _finalize_pairs(
    src: np.ndarray,
    dst: np.ndarray,
    w: np.ndarray,
    *,
    symmetric: bool,
    max_pairs: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if symmetric and src.size:
        src_sym = np.concatenate([src, dst])
        dst_sym = np.concatenate([dst, src])
        w_sym = np.concatenate([w, w])
        src, dst, w = src_sym, dst_sym, w_sym
        w = np.concatenate([w, w])

    if src.size:
        pairs = np.stack([src, dst], axis=1)
        _, unique_idx = np.unique(pairs, axis=0, return_index=True)
        src = src[unique_idx]
        dst = dst[unique_idx]
        w = w[unique_idx]

    if src.size > max_pairs:
        rng = np.random.default_rng(0)
        keep = rng.choice(src.size, size=max_pairs, replace=False)
        src = src[keep]
        dst = dst[keep]
        w = w[keep]

    if src.size:
        w_sum = np.zeros(int(src.max()) + 1, dtype=np.float32)
        np.add.at(w_sum, src, w)
        w = w / (w_sum[src] + 1e-12)
    return src.astype(np.int64, copy=False), dst.astype(np.int64, copy=False), w.astype(np.float32, copy=False)
