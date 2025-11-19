from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import ot
import pandas as pd
import torch


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
    except TypeError:  # for very old torch versions
        return torch.as_tensor(x, dtype=dtype).to(device=device).contiguous()


@torch.no_grad()
def _sinkhorn_uot_torch(
    M: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float,
    tau: float,
    iters: int,
    tol: float,
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
    *,
    reg: float,
    reg_m: float,
    cost_clip_q: Optional[float],
    clip_big: float,
    ot_backend: str,
    iters: int,
    tol: float,
    use_gpu: bool,
    gpu_device: int,
    return_transport: bool,
    transport_topk: int,
    chunk_size: Optional[int],
    _chunked: bool = False,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, np.ndarray]]:
    if len(Bi) == 0 or len(R) == 0:
        return Bi.copy() if not return_transport else (Bi.copy(), {"indices": np.zeros((0, 1), dtype=np.int32), "weights": np.zeros((0, 1), dtype=np.float32)})
    if isinstance(R, dict):
        raise ValueError("R must be a dense ndarray; provide modality-specific prototypes upstream.")

    if chunk_size is not None and len(Bi) > chunk_size and not _chunked:
        outputs: List[np.ndarray] = []
        idx_chunks: List[np.ndarray] = []
        weight_chunks: List[np.ndarray] = []
        residual_chunks: List[np.ndarray] = []
        has_residual = False
        for start in range(0, len(Bi), chunk_size):
            chunk = slice(start, min(len(Bi), start + chunk_size))
            result = _ot_barycentric_gpu(
                Bi[chunk],
                R,
                reg=reg,
                reg_m=reg_m,
                cost_clip_q=cost_clip_q,
                clip_big=clip_big,
                ot_backend=ot_backend,
                iters=iters,
                tol=tol,
                use_gpu=use_gpu,
                gpu_device=gpu_device,
                return_transport=return_transport,
                transport_topk=transport_topk,
                chunk_size=chunk_size,
                _chunked=True,
            )
            if return_transport:
                chunk_out, chunk_trans = result  # type: ignore[assignment]
                outputs.append(np.asarray(chunk_out))
                idx_chunks.append(np.asarray(chunk_trans["indices"], dtype=np.int32))
                weight_chunks.append(np.asarray(chunk_trans["weights"], dtype=np.float32))
                residual = chunk_trans.get("residual")
                if residual is not None:
                    residual_chunks.append(np.asarray(residual, dtype=np.float32))
                    has_residual = True
            else:
                outputs.append(np.asarray(result))
        combined = np.vstack(outputs).astype(Bi.dtype, copy=False) if outputs else Bi.copy()
        if not return_transport:
            return combined
        transport: Dict[str, np.ndarray] = {
            "indices": np.vstack(idx_chunks).astype(np.int32, copy=False),
            "weights": np.vstack(weight_chunks).astype(np.float32, copy=False),
        }
        if has_residual:
            transport["residual"] = np.concatenate(residual_chunks).astype(np.float32, copy=False)
        return combined, transport

    device = _torch_device(use_gpu=(ot_backend == "torch") and use_gpu, gpu_device=gpu_device)
    Bi64 = np.asarray(Bi, dtype=np.float64, order="C")
    R64 = np.asarray(R, dtype=np.float64, order="C")
    M = ot.dist(Bi64, R64, metric="sqeuclidean")
    M /= (M.std() + 1e-8)
    if cost_clip_q is not None:
        thr = np.quantile(M, cost_clip_q, axis=1, keepdims=True)
        M = np.where(M > thr, thr + clip_big, M)

    a = np.full(Bi.shape[0], 1.0 / max(Bi.shape[0], 1), dtype=np.float64)
    b = np.full(R.shape[0], 1.0 / max(R.shape[0], 1), dtype=np.float64)

    if ot_backend == "torch":
        M_t = _to_torch(M, device=device, dtype=torch.float64)
        a_t = _to_torch(a, device=device, dtype=torch.float64)
        b_t = _to_torch(b, device=device, dtype=torch.float64)
        tau = reg / (reg + reg_m)
        u, v, K = _sinkhorn_uot_torch(M_t, a_t, b_t, float(reg), float(tau), iters, tol)
        # Avoid forming dense diagonal matrices on GPU; multiply in-place instead.
        T = (u[:, None] * K) * v[None, :]
        T = T.cpu().numpy()
    else:
        try:
            T = ot.unbalanced.sinkhorn_unbalanced(
                a,
                b,
                M,
                reg,
                reg_m,
                method="sinkhorn_stabilized",
                numItermax=iters,
                stopThr=tol,
                verbose=False,
            )
        except TypeError:
            T = ot.unbalanced.sinkhorn_unbalanced(
                a,
                b,
                M,
                reg,
                reg_m,
                numItermax=iters,
                stopThr=tol,
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
        w_top_raw = T[rows, idx_top]
    else:
        idx_top = np.broadcast_to(np.arange(T.shape[1]), T.shape).copy()
        w_top_raw = T.copy()
    sum_top = w_top_raw.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        residual = np.where(
            row_sum > 0,
            np.clip((row_sum - sum_top) / (row_sum + 1e-12), 0.0, 1.0),
            1.0,
        ).astype(np.float32, copy=False)
    w_top_norm = w_top_raw / (sum_top + 1e-12)
    transport = {
        "indices": idx_top.astype(np.int32, copy=False),
        "weights": w_top_norm.astype(np.float32, copy=False),
        "residual": residual[:, 0],
    }
    return out, transport


def compute_ot_alignment(
    source: np.ndarray,
    reference: np.ndarray,
    *,
    reg: float = 0.05,
    reg_m: float = 0.5,
    cost_clip_q: Optional[float] = 0.90,
    clip_big: float = 50.0,
    backend: str = "torch",
    iters: int = 1000,
    tol: float = 1e-6,
    use_gpu: bool = True,
    gpu_device: int = 0,
    transport_topk: int = 64,
    chunk_size: Optional[int] = 1024,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    aligned, transport = _ot_barycentric_gpu(
        source,
        reference,
        reg=reg,
        reg_m=reg_m,
        cost_clip_q=cost_clip_q,
        clip_big=clip_big,
        ot_backend=backend,
        iters=iters,
        tol=tol,
        use_gpu=use_gpu,
        gpu_device=gpu_device,
        return_transport=True,
        transport_topk=transport_topk,
        chunk_size=chunk_size,
    )
    return aligned.astype(np.float32, copy=False), transport


def ot_label_transfer(
    *,
    transport: Dict[str, np.ndarray],
    target_labels: pd.Series,
    min_conf: float = 0.55,
    unknown_label: str = "unknown",
) -> Dict[str, Any]:
    if not {"indices", "weights"}.issubset(transport):
        raise KeyError("transport must contain 'indices' and 'weights'.")

    labels_cat = target_labels.astype("category")
    classes = labels_cat.cat.categories
    codes = labels_cat.cat.codes.to_numpy()

    idx = np.asarray(transport["indices"], dtype=np.int64)
    weights = np.asarray(transport["weights"], dtype=np.float32)
    if weights.shape != idx.shape:
        raise ValueError("Transport indices/weights shape mismatch.")

    label_codes = codes[idx]
    valid = label_codes >= 0
    weights_valid = np.where(valid, weights, 0.0)
    mass = weights_valid.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        weights_norm = np.divide(
            weights_valid,
            np.where(mass > 0, mass, 1.0),
            out=np.zeros_like(weights_valid),
            where=mass > 0,
        )

    n_classes = len(classes)
    proba = np.zeros((idx.shape[0], n_classes), dtype=np.float32)
    for j in range(n_classes):
        proba[:, j] = (weights_norm * (label_codes == j)).sum(axis=1)

    support = mass.squeeze(1).astype(np.float32)
    conf = proba.max(axis=1)
    best_idx = proba.argmax(axis=1)
    predictions = np.full(idx.shape[0], unknown_label, dtype=object)
    mask = support > 0
    predictions[mask] = classes.to_numpy()[best_idx[mask]]
    predictions[conf < float(min_conf)] = unknown_label

    unknown_prob = np.clip(1.0 - proba.sum(axis=1), 0.0, 1.0)
    return {
        "pred_labels": pd.Categorical(predictions, categories=list(classes) + [unknown_label]),
        "confidence": conf,
        "proba": proba,
        "classes": classes,
        "support": support,
        "unknown_prob": unknown_prob,
    }



__all__ = ["_sinkhorn_uot_torch", "_ot_barycentric_gpu", "compute_ot_alignment", "ot_label_transfer"]
