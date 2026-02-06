# =================== paste-and-run: OTIntegration_rare_aware_sup.py (rare-safe + supervised option) ===================
from __future__ import annotations
from typing import Optional

import numpy as np

from .ot_helpers import _as_nd_f32_c

# -------------------- CORAL pre-alignment (NEW) --------------------

def _cov_psd(X: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """PSD covariance with ridge eps*I for stability."""
    Xc = X - X.mean(0, keepdims=True)
    n = Xc.shape[0]
    if n <= 1:
        # fallback: identity
        d = Xc.shape[1]
        return np.eye(d, dtype=np.float64)
    C = (Xc.T @ Xc) / max(n - 1, 1)
    d = C.shape[0]
    C = C + float(eps) * np.eye(d, dtype=np.float64)
    return C


def _sqrtm_psd(C: np.ndarray) -> np.ndarray:
    """Matrix square root for PSD matrix via eigendecomposition."""
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-12, None)
    return (V * np.sqrt(w)[None, :]) @ V.T


def _invsqrtm_psd(C: np.ndarray) -> np.ndarray:
    """Inverse matrix square root for PSD matrix via eigendecomposition."""
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-12, None)
    return (V * (1.0 / np.sqrt(w))[None, :]) @ V.T


def _subsample_rows(X: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if max_points is None or max_points <= 0 or X.shape[0] <= max_points:
        return X
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(X.shape[0], size=int(max_points), replace=False)
    return X[idx]


def _coral_transform(
    Xs: np.ndarray,
    mu_t: np.ndarray,
    Ct_sqrt: np.ndarray,
    eps: float = 1e-3,
) -> np.ndarray:
    """
    CORAL: Xs -> (Xs - mu_s) * Cs^{-1/2} * Ct^{1/2} + mu_t
    Inputs:
      - mu_t: target mean
      - Ct_sqrt: target covariance sqrt
    """
    Xs64 = np.asarray(Xs, dtype=np.float64, order="C")
    mu_s = Xs64.mean(0, keepdims=True)
    Cs = _cov_psd(Xs64, eps=eps)
    Cs_invsqrt = _invsqrtm_psd(Cs)
    Xw = (Xs64 - mu_s) @ Cs_invsqrt
    Xr = Xw @ Ct_sqrt + mu_t
    return Xr.astype(np.float32, copy=False)


def _coral_prealign(
    X: np.ndarray,
    b: np.ndarray,
    ref_label_enc: int,
    ref_mode: str,
    strength: float = 1.0,         # 1.0 = full CORAL, 0.0 = no-op
    eps: float = 1e-3,
    max_points: int = 20000,
    seed: int = 0,
    target: str = "auto",          # "auto"|"reference"|"global"
    apply_dims: Optional[slice] = None,  # apply CORAL only on subset dims
) -> np.ndarray:
    """
    Pre-align batches with CORAL to create overlap for OT in disjoint datasets.

    target:
      - "reference": align all non-ref batches to ref batch (largest)
      - "global": align each batch to global pooled stats
      - "auto": if ref_mode=="union" -> global, else -> reference
    """
    s = float(np.clip(strength, 0.0, 1.0))
    if s <= 0.0:
        return X

    Xf = _as_nd_f32_c(X)
    out = Xf.copy()

    if apply_dims is None:
        apply_dims = slice(0, Xf.shape[1])

    X_sub = Xf[:, apply_dims]
    target_norm = str(target).lower()
    if target_norm == "auto":
        target_norm = "global" if str(ref_mode).lower() == "union" else "reference"

    # build target moments
    if target_norm == "reference":
        Xt = X_sub[b == int(ref_label_enc)]
        Xt = _subsample_rows(Xt, max_points=max_points, seed=seed + 11)
    elif target_norm == "global":
        Xt = _subsample_rows(X_sub, max_points=max_points, seed=seed + 13)
    else:
        raise ValueError(f"target must be 'auto'|'reference'|'global' (got {target!r})")

    if Xt.shape[0] <= 1:
        return Xf  # not enough to estimate target stats

    Xt64 = np.asarray(Xt, dtype=np.float64, order="C")
    mu_t = Xt64.mean(0, keepdims=True)
    Ct = _cov_psd(Xt64, eps=eps)
    Ct_sqrt = _sqrtm_psd(Ct)

    # transform each batch
    for lbl in np.unique(b):
        lbl = int(lbl)
        idx = np.where(b == lbl)[0]
        if idx.size == 0:
            continue
        if target_norm == "reference" and lbl == int(ref_label_enc):
            continue  # keep reference as-is

        Xs = X_sub[idx]
        Xs_aligned = _coral_transform(Xs, mu_t=mu_t, Ct_sqrt=Ct_sqrt, eps=eps)

        # blend (minimal safety knob): keep some of original geometry if desired
        if s < 1.0:
            Xs_aligned = (1.0 - s) * Xs + s * Xs_aligned

        out[idx, apply_dims] = Xs_aligned.astype(out.dtype, copy=False)

    return out
