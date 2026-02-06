from __future__ import annotations

from typing import Tuple

import numpy as np
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
    except TypeError:  # older torch without device kwarg
        return torch.as_tensor(x, dtype=dtype).to(device=device).contiguous()


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
