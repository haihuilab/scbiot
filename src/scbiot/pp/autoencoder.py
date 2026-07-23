"""Reference-fitted embeddings for scBIOT preprocessing.

This module provides the tied linear autoencoder used by label-transfer and
multiomics integration workflows:

* a linear encoder (no nonlinear hidden layers),
* a tied linear decoder that reuses the encoder weights,
* a reconstruction loss with an orthogonality penalty on the encoder loadings.

Public entry points
-------------------
``autoencoder``       -- embed a single AnnData into ``adata.obsm[out_key]``.
``autoencoder_map``   -- coembed a query AnnData into a reference basis (both land
                         in ``obsm["X_ae"]``); the query is mapped through the same
                         reference standardization + encoder so the two modalities
                         intermix.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import warnings
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import anndata as ad
from anndata import AnnData
from scipy import sparse

NN_LOG1P_OBSM_KEY = "X_nn_log1p"
NN_LATENT_DIM = 50
NN_HIDDEN_DIM = 512
NN_MAX_EPOCHS = 100
NN_BATCH_SIZE = 512
NN_LR = 1e-3
NN_WEIGHT_DECAY = 1e-4
NN_DROPOUT = 0.1
NN_EARLY_STOP = 10
# The autoencoder trains on all cells for the full epoch budget; no validation
# holdout / reconstruction early-stopping (the supervised head needs full training).
NN_VALIDATION_SPLIT = 0.0
NN_BATCH_IN_DECODER = False
NN_ORTHOGONALITY_WEIGHT = 1e-2
# Reference->query mapping defaults (supervised linear AE).
NN_MAP_LATENT_DIM = 50
NN_SUPERVISED_WEIGHT = 2.0
# The supervised mapping trains on all reference cells for the full epoch budget
# (no validation holdout / reconstruction early-stop) so the supervised head can
# fully separate cell types.
NN_MAP_VALIDATION_SPLIT = 0.0
AUTOENCODER_OBSM_KEY = "X_ae"


@dataclass(frozen=True)
class AutoencoderConfig:
    hidden_dim: int = NN_HIDDEN_DIM
    latent_dim: int = NN_LATENT_DIM
    max_epochs: int = NN_MAX_EPOCHS
    batch_size: int = NN_BATCH_SIZE
    lr: float = NN_LR
    weight_decay: float = NN_WEIGHT_DECAY
    dropout: float = NN_DROPOUT
    early_stop_patience: int = NN_EARLY_STOP
    validation_split: float = NN_VALIDATION_SPLIT
    batch_in_decoder: bool = NN_BATCH_IN_DECODER
    orthogonality_weight: float = NN_ORTHOGONALITY_WEIGHT
    supervised_weight: float = 0.0
    l2: bool = False


# -----------------------------
# minibatch / matrix helpers
# -----------------------------
def _resolve_expression_matrix(adata: Any, input_key: Optional[str]) -> tuple[Any, str]:
    if input_key is None:
        return adata.X, "__scbiot_input_X"
    if hasattr(adata, "layers") and input_key in adata.layers:
        return adata.layers[input_key], input_key
    if input_key in {"X", "x"}:
        return adata.X, "__scbiot_input_X"
    available = ["X"] + sorted(str(k) for k in getattr(adata, "layers", {}).keys())
    raise KeyError(
        f"Input key '{input_key}' not found in adata.layers and is not 'X'. "
        f"Available matrix keys: {available}."
    )


def _to_f32(matrix: Any) -> Any:
    if sparse.issparse(matrix):
        matrix = matrix.tocsr()
        return matrix.astype(np.float32) if matrix.dtype != np.float32 else matrix
    return np.asarray(matrix, dtype=np.float32, order="C")


def _to_csr_f32(matrix: Any) -> sparse.csr_matrix:
    if sparse.issparse(matrix):
        return matrix.tocsr().astype(np.float32, copy=False)
    return sparse.csr_matrix(np.asarray(matrix, dtype=np.float32, order="C"))


def _normalize_log1p_rows(X: Any, *, target_sum: float = 1e4) -> sparse.csr_matrix:
    X = _to_csr_f32(X).copy()
    row_sum = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
    scale = np.divide(
        float(target_sum),
        row_sum,
        out=np.zeros_like(row_sum, dtype=np.float32),
        where=row_sum > 0,
    )
    X = X.multiply(scale[:, None]).tocsr()
    X.data = np.log1p(X.data).astype(np.float32, copy=False)
    return X.astype(np.float32, copy=False)


def _sparse_column_variance(X: Any) -> np.ndarray:
    X = _to_csr_f32(X)
    mean = np.asarray(X.mean(axis=0)).ravel()
    mean_sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
    return np.maximum(mean_sq - mean * mean, 0.0).astype(np.float32, copy=False)


def _normalize_log1p_scale(X: Any, var_index: pd.Index, *, return_stats: bool = False) -> Any:
    """normalize_total -> log1p -> scale(zero_center=False) on a feature panel.

    With ``return_stats=True`` also returns ``(target_sum, gene_std)`` -- the median
    library size ``normalize_total`` used and the exact per-gene divisor
    ``scale`` applied -- so the identical preprocessing can be replayed on new cells
    out-of-core (see :class:`LinearAutoencoderModel`). ``gene_std`` is recovered
    exactly from the column-sum ratio before/after scaling (scale divides each gene
    by a constant, so ``sum(pre)/sum(post) == std``), independent of scanpy's
    internal ddof.
    """
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    import scanpy as sc

    proc = AnnData(
        X=X.copy() if hasattr(X, "copy") else X,
        obs=pd.DataFrame(index=pd.RangeIndex(X.shape[0]).astype(str)),
        var=pd.DataFrame(index=var_index),
    )
    # median library size normalize_total(target_sum=None) uses (computed pre-normalize)
    target_sum = float(np.median(np.asarray(proc.X.sum(axis=1)).ravel())) if return_stats else None
    sc.pp.normalize_total(proc)
    sc.pp.log1p(proc)
    pre = np.asarray(proc.X.sum(axis=0)).ravel().astype(np.float64) if return_stats else None
    # Preserve sparsity here and let training materialize one minibatch at a time.
    sc.pp.scale(proc, zero_center=False)
    if return_stats:
        post = np.asarray(proc.X.sum(axis=0)).ravel().astype(np.float64)
        gene_std = np.divide(pre, post, out=np.ones_like(pre), where=post > 0).astype(np.float32)
        return _to_f32(proc.X), target_sum, gene_std
    return _to_f32(proc.X)


def _normalize_log1p_panel(X: Any, var_index: pd.Index, target_sum: float = 1e4) -> Any:
    """normalize_total(target_sum) -> log1p on a feature panel (no scaling).

    A fixed ``target_sum`` is used for every domain so that a reference and a
    query measured with different library sizes (e.g. scRNA-seq vs. Xenium) are
    placed on the same scale before the reference-fit encoder projects them.
    """
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    import scanpy as sc

    proc = AnnData(
        X=X.copy() if hasattr(X, "copy") else X,
        obs=pd.DataFrame(index=pd.RangeIndex(X.shape[0]).astype(str)),
        var=pd.DataFrame(index=var_index),
    )
    sc.pp.normalize_total(proc, target_sum=float(target_sum))
    sc.pp.log1p(proc)
    return _to_f32(proc.X)


class _RefStandardizer:
    """Reference-fit z-score (mean/std) applied to dense minibatches with clip.

    The reference mean and standard deviation are computed once on the reference
    log1p panel and reused for the query, so both domains land in the same
    standardized feature space before the reference-fit encoder projects them.
    Centering happens per dense minibatch, which keeps the full panel sparse.
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray, clip: float = 10.0) -> None:
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, -1)
        self.std = np.clip(np.asarray(std, dtype=np.float32), 1e-8, None).reshape(1, -1)
        self.clip = float(clip)

    @classmethod
    def from_matrix(cls, X: Any, clip: float = 10.0) -> "_RefStandardizer":
        if sparse.issparse(X):
            mean = np.asarray(X.mean(axis=0)).ravel()
            mean_sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
            var = np.maximum(mean_sq - mean * mean, 0.0)
        else:
            Xd = np.asarray(X, dtype=np.float64)
            mean = Xd.mean(axis=0)
            var = Xd.var(axis=0)
        std = np.sqrt(np.maximum(var, 1e-8))
        return cls(mean.astype(np.float32), std.astype(np.float32), clip=clip)

    def apply(self, dense: np.ndarray) -> np.ndarray:
        out = (np.asarray(dense, dtype=np.float32) - self.mean) / self.std
        np.clip(out, -self.clip, self.clip, out=out)
        return np.asarray(out, dtype=np.float32, order="C")


def _prepare_log1p_hvg_matrix(
    adata: Any,
    *,
    input_key: Optional[str],
    batch_key: Optional[str],
    n_top_genes: int,
    return_stats: bool = False,
) -> Any:
    """Select HVGs on raw counts then normalize/log1p/scale the HVG panel.

    With ``return_stats=True`` returns ``(X_train, genes, target_sum, gene_std)`` so
    the preprocessing + HVG panel can be replayed out-of-core.
    """
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    import scanpy as sc

    matrix, _ = _resolve_expression_matrix(adata, input_key)
    hvg_obs = (
        adata.obs[[batch_key]].copy()
        if batch_key is not None and batch_key in adata.obs
        else pd.DataFrame(index=adata.obs_names.copy())
    )
    adata_hvg = AnnData(X=matrix, obs=hvg_obs, var=adata.var.iloc[:, 0:0].copy())
    resolved_batch_key = (
        batch_key if (batch_key is not None and batch_key in adata.obs) else None
    )
    try:
        sc.pp.highly_variable_genes(
            adata_hvg,
            n_top_genes=int(n_top_genes),
            flavor="seurat_v3",
            batch_key=resolved_batch_key,
            span=0.6,
            subset=False,
        )
    except ValueError as exc:
        # LOESS can be singular for small subsets or already-filtered panels.
        # Rank the variance of library-size-normalized log1p counts directly:
        # the dispersion-based Scanpy flavors expect log-transformed input and
        # would apply expm1 to this raw-count matrix.
        warnings.warn(
            "Seurat v3 HVG selection failed; falling back to variance-ranked "
            "normalized log1p counts. "
            f"Original error: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        variance = _sparse_column_variance(_normalize_log1p_rows(matrix))
        n_selected = min(int(n_top_genes), int(variance.size))
        if n_selected <= 0:
            raise ValueError("n_top_genes must select at least one feature.")
        ranked = np.argsort(variance, kind="stable")
        hvg_mask = np.zeros(variance.size, dtype=bool)
        hvg_mask[ranked[-n_selected:]] = True
    else:
        hvg_mask = np.asarray(
            adata_hvg.var["highly_variable"].fillna(False), dtype=bool
        )
    if not np.any(hvg_mask):
        raise ValueError("No highly variable genes were selected for autoencoder input.")

    X_hvg = _to_f32(matrix[:, hvg_mask])
    panel = adata.var_names[hvg_mask]
    if return_stats:
        X_train, target_sum, gene_std = _normalize_log1p_scale(
            X_hvg, panel, return_stats=True)
        return X_train, [str(g) for g in panel], target_sum, gene_std
    return _normalize_log1p_scale(X_hvg, panel)


def _materialize_batch_rows(X_train: Any, idx: np.ndarray) -> np.ndarray:
    X_batch = X_train[idx]
    if sparse.issparse(X_batch):
        X_batch = X_batch.toarray()
    return np.asarray(X_batch, dtype=np.float32, order="C")


# -----------------------------
# label-aware encoder initialization
# -----------------------------
# -----------------------------
# tied linear orthogonal autoencoder
# -----------------------------
def _train_linear_orthogonal_ae(
    X_train: Any,
    *,
    random_state: int,
    config: AutoencoderConfig,
    standardizer: Optional["_RefStandardizer"] = None,
    labels: Optional[np.ndarray] = None,
    class_weights: Optional[np.ndarray] = None,
    return_params: bool = False,
):
    """Train a PCA-like tied linear autoencoder on ``X_train``.

    Returns ``(encode, Z_train)`` where ``encode(matrix)`` projects any matrix
    with the same features into the learned latent space. When ``standardizer``
    is given, each dense minibatch is reference-standardized (z-score + clip)
    before it enters the encoder.

    When ``labels`` (integer class codes per training row, ``-1`` to ignore) and
    ``config.supervised_weight > 0`` are provided, a linear classification head is
    trained jointly with a class-balanced cross-entropy term. The head only shapes
    the encoder during training and is discarded afterwards, so the returned
    embedding stays a plain linear projection.
    """
    supervised = (
        labels is not None
        and float(config.supervised_weight) > 0.0
        and int(np.max(labels)) >= 0
    )
    label_codes = (
        np.asarray(labels, dtype=np.int64)
        if labels is not None
        else np.full(X_train.shape[0], -1, dtype=np.int64)
    )
    n_classes = int(label_codes.max() + 1) if supervised else 0

    def _materialize(matrix: Any, idx: np.ndarray) -> np.ndarray:
        block = _materialize_batch_rows(matrix, idx)
        return standardizer.apply(block) if standardizer is not None else block
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - torch is a hard dependency here
        raise ImportError("The linear autoencoder embedding requires torch.") from exc

    class LinearOrthogonalAE(nn.Module):
        """Tied linear encoder/decoder with an orthogonality penalty."""

        def __init__(self, input_dim: int, feature_mean: np.ndarray) -> None:
            super().__init__()
            self.encoder = nn.Linear(int(input_dim), int(config.latent_dim), bias=False)
            self.register_buffer(
                "feature_mean",
                torch.as_tensor(feature_mean, dtype=torch.float32).reshape(1, int(input_dim)),
            )
            nn.init.orthogonal_(self.encoder.weight)

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            return self.encoder(x - self.feature_mean)

        def decode(self, z: torch.Tensor) -> torch.Tensor:
            return F.linear(z, self.encoder.weight.t()) + self.feature_mean

        def orthogonality_loss(self) -> torch.Tensor:
            weights = self.encoder.weight
            gram = weights @ weights.t()
            eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
            return torch.mean((gram - eye) ** 2)

    n_cells, input_dim = X_train.shape
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(random_state))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(random_state))
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    if standardizer is not None:
        feature_mean = np.zeros(input_dim, dtype=np.float32)
    elif sparse.issparse(X_train):
        feature_mean = np.asarray(X_train.mean(axis=0)).ravel().astype(np.float32, copy=False)
    else:
        feature_mean = np.asarray(X_train, dtype=np.float32).mean(axis=0).astype(np.float32, copy=False)

    model = LinearOrthogonalAE(input_dim=int(input_dim), feature_mean=feature_mean).to(device)
    head = nn.Linear(int(config.latent_dim), n_classes).to(device) if supervised else None
    class_weight_t = None
    if supervised and class_weights is not None:
        class_weight_t = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    params = list(model.parameters())
    if head is not None:
        params += list(head.parameters())
    optimizer = torch.optim.Adam(
        params,
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )

    rng = np.random.default_rng(int(random_state))
    val_count = int(round(float(config.validation_split) * n_cells))
    if n_cells >= 8 and config.validation_split > 0:
        val_count = min(max(val_count, 1), n_cells - 1)
    else:
        val_count = 0
    if val_count > 0:
        perm_all = rng.permutation(n_cells)
        val_idx = np.asarray(perm_all[:val_count], dtype=np.int64)
        train_idx_all = np.asarray(perm_all[val_count:], dtype=np.int64)
    else:
        # No holdout: leave the RNG untouched here so per-epoch shuffles start
        # from a clean state (training on every cell, full epoch budget).
        val_idx = np.empty(0, dtype=np.int64)
        train_idx_all = np.arange(n_cells, dtype=np.int64)

    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_val = float("inf")
    stale_epochs = 0

    batch_size = int(config.batch_size)
    max_epochs = int(config.max_epochs)

    for _ in range(max_epochs):
        shuffled = rng.permutation(train_idx_all)
        epoch_batches = [
            shuffled[start : start + batch_size] for start in range(0, shuffled.size, batch_size)
        ]
        model.train()
        if head is not None:
            head.train()
        for idx in epoch_batches:
            x = torch.from_numpy(_materialize(X_train, idx)).to(device)
            z = model.encode(x)
            recon = model.decode(z)
            loss = F.mse_loss(recon, x)
            if float(config.orthogonality_weight) > 0.0:
                loss = loss + float(config.orthogonality_weight) * model.orthogonality_loss()
            if head is not None:
                yb = torch.from_numpy(label_codes[idx]).to(device)
                loss = loss + float(config.supervised_weight) * F.cross_entropy(
                    head(z),
                    yb,
                    weight=class_weight_t,
                    ignore_index=-1,
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        if val_idx.size == 0:
            continue
        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for start in range(0, val_idx.size, int(config.batch_size)):
                idx = np.asarray(val_idx[start : start + int(config.batch_size)], dtype=np.int64)
                x = torch.from_numpy(_materialize(X_train, idx)).to(device)
                recon = model.decode(model.encode(x))
                val_losses.append(float(F.mse_loss(recon, x).detach().cpu().item()))
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        if val_loss + 1e-8 < best_val:
            best_val = val_loss
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= int(config.early_stop_patience):
                break

    if val_idx.size > 0:
        model.load_state_dict(best_state)
    model.eval()

    def encode(matrix: Any) -> np.ndarray:
        n = matrix.shape[0]
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, n, 2048):
                idx = np.arange(start, min(start + 2048, n), dtype=np.int64)
                block = torch.from_numpy(_materialize(matrix, idx)).to(device)
                outputs.append(model.encode(block).cpu().numpy().astype(np.float32, copy=False))
        if not outputs:
            return np.empty((0, int(config.latent_dim)), dtype=np.float32)
        return np.vstack(outputs).astype(np.float32, copy=False)

    if return_params:
        W_np = model.encoder.weight.detach().cpu().numpy().astype(np.float32, copy=False)
        fmean_np = (model.feature_mean.detach().cpu().numpy()
                    .astype(np.float32, copy=False).reshape(1, -1))
        return encode, encode(X_train), W_np, fmean_np
    return encode, encode(X_train)


def _train_deterministic_supervised_ae(
    X_train: Any,
    *,
    config: AutoencoderConfig,
    standardizer: Optional["_RefStandardizer"] = None,
    labels: Optional[np.ndarray] = None,
    class_weights: Optional[np.ndarray] = None,
    random_state: int = 0,
    lbfgs_max_iter: int = 300,
):
    """Deterministic full-batch supervised tied-linear AE solved with LBFGS.

    Same objective as :func:`_train_linear_orthogonal_ae` (tied reconstruction +
    orthogonality penalty + class-weighted cross-entropy head) but optimized
    *full-batch* with LBFGS and a strong-Wolfe line search from a deterministic
    initialization. There is no minibatch sampling and no per-epoch shuffle, so the
    learned basis carries no SGD-path or shuffle-order seed variance -- one model, one
    reproducible result. Returns ``(encode, Z_train)`` like the stochastic trainer.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - torch is a hard dependency here
        raise ImportError("The linear autoencoder embedding requires torch.") from exc

    n_cells, input_dim = X_train.shape
    latent = int(config.latent_dim)
    supervised = (
        labels is not None
        and float(config.supervised_weight) > 0.0
        and int(np.max(labels)) >= 0
    )
    label_codes = (
        np.asarray(labels, dtype=np.int64)
        if labels is not None
        else np.full(n_cells, -1, dtype=np.int64)
    )
    n_classes = int(label_codes.max() + 1) if supervised else 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(random_state))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(random_state))
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # Full-batch reference matrix (reference-standardized once); mean is zero post-standardize.
    X_full = _materialize_batch_rows(X_train, np.arange(n_cells, dtype=np.int64))
    if standardizer is not None:
        X_full = standardizer.apply(X_full)
        feature_mean = np.zeros(input_dim, dtype=np.float32)
    else:
        feature_mean = np.asarray(X_full, dtype=np.float32).mean(axis=0).astype(np.float32, copy=False)
    X_full = np.asarray(X_full, dtype=np.float32, order="C")

    # Deterministic orthogonal init (fixed seed). The full-batch objective converges to a
    # stable PCA-like-plus-supervised basis, so the result is effectively init-independent.
    enc = nn.Linear(int(input_dim), latent, bias=False)
    nn.init.orthogonal_(enc.weight)
    W0 = enc.weight.detach().cpu().numpy().astype(np.float32, copy=False)

    Xt = torch.as_tensor(X_full, dtype=torch.float32, device=device)
    mean_t = torch.as_tensor(feature_mean, dtype=torch.float32, device=device).reshape(1, input_dim)
    W = torch.nn.Parameter(torch.as_tensor(W0, dtype=torch.float32, device=device))
    eye = torch.eye(latent, device=device)
    params = [W]
    head_w = head_b = class_weight_t = y_t = None
    if supervised:
        head = nn.Linear(latent, n_classes)
        head_w = torch.nn.Parameter(head.weight.detach().to(device))
        head_b = torch.nn.Parameter(head.bias.detach().to(device))
        params += [head_w, head_b]
        y_t = torch.as_tensor(label_codes, dtype=torch.long, device=device)
        if class_weights is not None:
            class_weight_t = torch.as_tensor(class_weights, dtype=torch.float32, device=device)

    ortho_w = float(config.orthogonality_weight)
    sup_w = float(config.supervised_weight)
    wd = float(config.weight_decay)
    optimizer = torch.optim.LBFGS(
        params,
        lr=1.0,
        max_iter=int(lbfgs_max_iter),
        history_size=50,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        z = (Xt - mean_t) @ W.t()
        recon = z @ W + mean_t
        loss = ((recon - Xt) ** 2).mean()
        if ortho_w > 0.0:
            gram = W @ W.t()
            loss = loss + ortho_w * ((gram - eye) ** 2).mean()
        if supervised:
            logits = z @ head_w.t() + head_b
            loss = loss + sup_w * F.cross_entropy(
                logits, y_t, weight=class_weight_t, ignore_index=-1
            )
        if wd > 0.0:
            loss = loss + wd * sum((p ** 2).sum() for p in params)
        loss.backward()
        return loss

    optimizer.step(closure)

    W_np = W.detach().cpu().numpy().astype(np.float32, copy=False)
    fmean_np = feature_mean.reshape(1, input_dim)

    def encode(matrix: Any) -> np.ndarray:
        n = matrix.shape[0]
        outputs: list[np.ndarray] = []
        for start in range(0, n, 2048):
            idx = np.arange(start, min(start + 2048, n), dtype=np.int64)
            block = _materialize_batch_rows(matrix, idx)
            if standardizer is not None:
                block = standardizer.apply(block)
            outputs.append(((block - fmean_np) @ W_np.T).astype(np.float32, copy=False))
        if not outputs:
            return np.empty((0, latent), dtype=np.float32)
        return np.vstack(outputs).astype(np.float32, copy=False)

    return encode, encode(X_train)


def _fit_batch_decoder_ae32(
    X_train: Any,
    *,
    random_state: int,
    config: AutoencoderConfig,
    return_params: bool = False,
) -> np.ndarray:
    """Train the tied linear autoencoder and return the latent embedding.

    With ``return_params=True`` also returns the encoder loadings ``W`` and feature
    mean, so a reusable out-of-core transform can be assembled.
    """
    if return_params:
        _, Z, W, fmean = _train_linear_orthogonal_ae(
            X_train, random_state=random_state, config=config, return_params=True)
        return Z, W, fmean
    _, Z = _train_linear_orthogonal_ae(
        X_train,
        random_state=random_state,
        config=config,
    )
    return Z


# -----------------------------
# latent post-processing
# -----------------------------
def _l2_normalize_rows(Z: np.ndarray) -> np.ndarray:
    Z = np.asarray(Z, dtype=np.float32, order="C").copy()
    norms = np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    Z /= norms
    return Z.astype(np.float32, copy=False)


def _standardize_from_reference(
    Z_ref: np.ndarray,
    Z_query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Center/scale paired embeddings using reference statistics only."""
    ref_mean = Z_ref.mean(axis=0, dtype=np.float64, keepdims=True)
    ref_std = np.clip(Z_ref.std(axis=0, dtype=np.float64, ddof=0, keepdims=True), 1e-6, None)
    return (
        ((Z_ref - ref_mean) / ref_std).astype(np.float32, copy=False),
        ((Z_query - ref_mean) / ref_std).astype(np.float32, copy=False),
    )


# -----------------------------
# fitted transform (out-of-core projection)
# -----------------------------
class LinearAutoencoderModel:
    """Fitted linear-autoencoder transform returned by ``autoencoder(..., return_model=True)``.

    Captures the *entire* projection pipeline -- the HVG gene panel, the
    ``normalize_total`` target, log1p, the per-gene scale, and the tied linear
    encoder (loadings ``W`` and feature mean) -- so the exact same embedding can be
    reproduced on new cells one chunk at a time. This makes the PCA-like
    autoencoder usable out-of-core: fit on an in-memory subsample, then project an
    atlas of arbitrary size (e.g. Tahoe-100M) chunk-by-chunk into ``X_ae`` without
    ever holding all cells in memory. The projection is a pure linear map, so the
    result is identical to having embedded every cell during fitting. The
    autoencoder is a pure representation; batch correction is delegated entirely to
    OT integration on the projected embedding (``scbiot.ot.integrate``).
    """

    def __init__(self, genes, target_sum, gene_std, feature_mean, W, l2=False):
        self.genes = [str(g) for g in genes]
        self.target_sum = float(target_sum)
        self.gene_std = np.asarray(gene_std, dtype=np.float32).reshape(1, -1)
        self.feature_mean = np.asarray(feature_mean, dtype=np.float32).reshape(1, -1)
        self.W = np.asarray(W, dtype=np.float32)          # (latent_dim, n_genes)
        self.l2 = bool(l2)

    @property
    def n_genes(self) -> int:
        return len(self.genes)

    @property
    def latent_dim(self) -> int:
        return int(self.W.shape[0])

    def _project_dense(self, counts: np.ndarray) -> np.ndarray:
        """Project a dense raw-count block (columns already in ``self.genes`` order)."""
        X = sparse.csr_matrix(counts) if not sparse.issparse(counts) else counts.tocsr()
        row_sum = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
        sf = np.divide(self.target_sum, row_sum, out=np.zeros_like(row_sum),
                       where=row_sum > 0)
        X = X.multiply(sf[:, None]).tocsr()
        X.data = np.log1p(X.data).astype(np.float32, copy=False)
        dense = X.toarray().astype(np.float32, copy=False)
        dense /= self.gene_std                              # scale(zero_center=False)
        Z = (dense - self.feature_mean) @ self.W.T
        if self.l2:
            Z = _l2_normalize_rows(Z)
        return np.asarray(Z, dtype=np.float32, order="C")

    def transform(self, data: Any, *, input_key: Optional[str] = "counts",
                  var_names: Optional[Sequence[str]] = None,
                  chunk_size: int = 100_000) -> np.ndarray:
        """Project raw counts into the fitted latent space, chunk-by-chunk.

        ``data`` is either an AnnData (counts read from ``input_key`` / ``X``, genes
        from ``var_names``) or a (cells x genes) count matrix paired with
        ``var_names``. Only ``self.genes`` are used; they must all be present.
        """
        if hasattr(data, "var_names") and hasattr(data, "obs"):
            matrix, _ = _resolve_expression_matrix(data, input_key)
            all_genes = pd.Index([str(g) for g in data.var_names])
        else:
            if var_names is None:
                raise ValueError("var_names is required when data is a bare matrix.")
            matrix = data
            all_genes = pd.Index([str(g) for g in var_names])
        col = all_genes.get_indexer(pd.Index(self.genes))
        if np.any(col < 0):
            missing = [g for g, c in zip(self.genes, col) if c < 0]
            raise KeyError(f"{len(missing)} panel gene(s) absent from data, e.g. {missing[:5]}.")
        n = matrix.shape[0]
        out = np.empty((n, self.latent_dim), dtype=np.float32)
        for start in range(0, n, int(chunk_size)):
            stop = min(start + int(chunk_size), n)
            block = matrix[start:stop][:, col]
            block = block.toarray() if sparse.issparse(block) else np.asarray(block)
            out[start:stop] = self._project_dense(block)
        return out

    def to_npz(self, path: Any) -> None:
        """Persist the transform so a later process can project without re-fitting."""
        np.savez(path, genes=np.asarray(self.genes, dtype=object),
                 target_sum=np.float32(self.target_sum), gene_std=self.gene_std,
                 feature_mean=self.feature_mean, W=self.W, l2=np.bool_(self.l2))

    @classmethod
    def from_npz(cls, path: Any) -> "LinearAutoencoderModel":
        d = np.load(path, allow_pickle=True)
        return cls(genes=list(d["genes"]), target_sum=float(d["target_sum"]),
                   gene_std=d["gene_std"], feature_mean=d["feature_mean"],
                   W=d["W"], l2=bool(d["l2"]))


# -----------------------------
# single-AnnData public API
# -----------------------------
def ensure_input_embedding(
    adata: Any,
    *,
    input_key: Optional[str],
    batch_key: str,
    n_top_genes: int,
    random_state: int = 0,
    out_key: str = NN_LOG1P_OBSM_KEY,
    hidden_dim: int = NN_HIDDEN_DIM,
    latent_dim: int = NN_LATENT_DIM,
    max_epochs: int = NN_MAX_EPOCHS,
    batch_size: int = NN_BATCH_SIZE,
    lr: float = NN_LR,
    weight_decay: float = NN_WEIGHT_DECAY,
    dropout: float = NN_DROPOUT,
    early_stop_patience: int = NN_EARLY_STOP,
    validation_split: float = NN_VALIDATION_SPLIT,
    batch_in_decoder: bool = NN_BATCH_IN_DECODER,
    orthogonality_weight: float = NN_ORTHOGONALITY_WEIGHT,
    l2: bool = False,
    return_model: bool = False,
) -> str | None:
    if input_key is None:
        return (None, None) if return_model else None
    if hasattr(adata, "obsm") and input_key in adata.obsm:
        return (input_key, None) if return_model else input_key
    if not hasattr(adata, "obs") or batch_key not in adata.obs:
        raise KeyError(f"Batch column '{batch_key}' not found in adata.obs.")

    if return_model:
        X_train, genes, target_sum, gene_std = _prepare_log1p_hvg_matrix(
            adata, input_key=input_key, batch_key=batch_key,
            n_top_genes=n_top_genes, return_stats=True,
        )
    else:
        X_train = _prepare_log1p_hvg_matrix(
            adata,
            input_key=input_key,
            batch_key=batch_key,
            n_top_genes=n_top_genes,
        )
    config = AutoencoderConfig(
        hidden_dim=int(hidden_dim),
        latent_dim=int(latent_dim),
        max_epochs=int(max_epochs),
        batch_size=int(batch_size),
        lr=float(lr),
        weight_decay=float(weight_decay),
        dropout=float(dropout),
        early_stop_patience=int(early_stop_patience),
        validation_split=float(validation_split),
        batch_in_decoder=bool(batch_in_decoder),
        orthogonality_weight=float(orthogonality_weight),
        l2=bool(l2),
    )
    if return_model:
        # Fit the linear encoder and assemble a reusable out-of-core transform. The
        # embedding is filled from the model itself, so adata.obsm[out_key] is
        # bit-identical to projecting the same cells later (no per-batch latent
        # centering -- batch is handled downstream by OT integration).
        _, W, fmean = _fit_batch_decoder_ae32(
            X_train, random_state=random_state, config=config, return_params=True)
        model = LinearAutoencoderModel(
            genes=genes, target_sum=target_sum, gene_std=gene_std,
            feature_mean=fmean, W=W, l2=bool(config.l2))
        adata.obsm[out_key] = model.transform(adata, input_key=input_key)
        return out_key, model
    Z = _fit_batch_decoder_ae32(
        X_train,
        random_state=random_state,
        config=config,
    )
    if config.l2:
        Z = _l2_normalize_rows(Z)
    adata.obsm[out_key] = Z
    return out_key


def autoencoder(adata: Any, **kwargs: Any) -> Any:
    """PCA-like linear autoencoder embedding on a single AnnData.

    * ``autoencoder(adata, label_key=..., ...)`` -- supervised. The (joint)
      AnnData already holds reference and query cells: reference cells are those
      whose ``label_key`` differs from ``unlabeled_category`` and query cells carry
      the unlabeled value. The tied linear encoder is fit on the reference with a
      class-balanced supervised head and every cell is projected into it.
    * ``autoencoder(adata, ...)`` (no ``label_key``) -- unsupervised: a tied
      linear autoencoder (reconstruction + orthogonality) on HVG log1p features.

    The embedding is written to ``adata.obsm[out_key]`` and ``adata`` is returned.
    """
    if kwargs.get("label_key") is not None:
        if kwargs.get("return_model"):
            raise NotImplementedError(
                "return_model is only supported for the unsupervised autoencoder "
                "(no label_key).")
        return _autoencoder_joint(adata, **kwargs)
    return _autoencoder_single(adata, **kwargs)


def _autoencoder_single(
    adata: Any,
    *,
    input_key: str = "counts",
    out_key: str = AUTOENCODER_OBSM_KEY,
    n_top_genes: int = 3000,
    hidden_dim: int = NN_HIDDEN_DIM,
    latent_dim: int = NN_LATENT_DIM,
    batch_size: int = NN_BATCH_SIZE,
    lr: float = NN_LR,
    weight_decay: float = NN_WEIGHT_DECAY,
    dropout: float = NN_DROPOUT,
    batch_key: str = "batch",
    l2: bool = False,
    random_state: int = 0,
    max_epochs: int = NN_MAX_EPOCHS,
    early_stop_patience: int = NN_EARLY_STOP,
    validation_split: float = NN_VALIDATION_SPLIT,
    batch_in_decoder: bool = NN_BATCH_IN_DECODER,
    orthogonality_weight: float = NN_ORTHOGONALITY_WEIGHT,
    return_model: bool = False,
) -> Any:
    """Train a PCA-like linear autoencoder on HVG log1p features (single AnnData).

    The encoder and decoder are tied linear maps trained with a reconstruction
    loss and an orthogonality penalty on the encoder loadings, so latent axes
    behave like PCA components. The embedding is written to
    ``adata.obsm[out_key]`` and ``adata`` is returned. With ``return_model=True``
    returns ``(adata, model)`` where ``model`` is a :class:`LinearAutoencoderModel`
    that projects new cells into the same space chunk-by-chunk (out-of-core).
    """
    if hasattr(adata, "obsm") and input_key in adata.obsm:
        Z = np.asarray(adata.obsm[input_key], dtype=np.float32, order="C")
        if l2:
            Z = _l2_normalize_rows(Z)
        adata.obsm[out_key] = Z.copy()
        return (adata, None) if return_model else adata

    result = ensure_input_embedding(
        adata,
        input_key=input_key,
        batch_key=batch_key,
        n_top_genes=n_top_genes,
        random_state=random_state,
        out_key=out_key,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        max_epochs=max_epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        dropout=dropout,
        early_stop_patience=early_stop_patience,
        validation_split=validation_split,
        batch_in_decoder=batch_in_decoder,
        orthogonality_weight=orthogonality_weight,
        l2=l2,
        return_model=return_model,
    )
    if return_model:
        _, model = result
        return adata, model
    return adata


# -----------------------------
# reference -> query mapping
# -----------------------------
def _resolve_query_layer(adata_query: AnnData, query_layer: Optional[str]) -> Optional[str]:
    if query_layer is not None:
        if query_layer in adata_query.layers:
            return query_layer
        if query_layer in {"X", "x"}:
            return None
        available = ["X"] + sorted(str(k) for k in adata_query.layers.keys())
        raise KeyError(
            f"query_layer '{query_layer}' not found in adata_query.layers. "
            f"Available matrix keys: {available}."
        )
    if "ga_smooth" in adata_query.layers:
        return "ga_smooth"
    if "ga" in adata_query.layers:
        return "ga"
    return None


def _mask_query_labels(adata_query: AnnData, label_key: str, unlabeled_category: str) -> None:
    """Back up query labels and overwrite them with the unlabeled category."""
    original = adata_query.obs.get(label_key)
    if original is not None:
        backup_key = f"{label_key}_original"
        if backup_key not in adata_query.obs:
            adata_query.obs[backup_key] = original.copy()
    adata_query.obs[label_key] = unlabeled_category


def _reference_label_codes(
    adata_reference: AnnData,
    *,
    label_key: Optional[str],
    unlabeled_category: str,
    weight_power: float = 1.0,
    weight_clip_max: float = 1e6,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Encode reference labels as integer codes plus class weights.

    Unlabeled reference cells map to ``-1`` (ignored by the supervised head). The
    class weights are ``inverse_frequency ** weight_power``: the default
    ``weight_power=1.0`` is full inverse-frequency balancing (which keeps rare
    classes visible to the supervised head), and ``0.5`` is a gentler square-root
    variant. Weights are clipped to ``[0.5, weight_clip_max]``. Returns
    ``(None, None)`` when no usable labels are present.
    """
    if label_key is None or label_key not in adata_reference.obs.columns:
        return None, None
    labels = adata_reference.obs[label_key].astype("string")
    valid = labels.notna() & labels.ne(unlabeled_category)
    if int(valid.sum()) < 2:
        return None, None
    classes = pd.Index(sorted(labels[valid].unique()))
    code_map = {cls: i for i, cls in enumerate(classes)}
    codes = np.full(adata_reference.n_obs, -1, dtype=np.int64)
    valid_np = valid.to_numpy()
    codes[valid_np] = labels[valid].map(code_map).to_numpy(dtype=np.int64)
    counts = np.bincount(codes[valid_np], minlength=len(classes)).astype(np.float64)
    inv_freq = counts.sum() / (len(classes) * np.clip(counts, 1.0, None))
    weights = np.clip(inv_freq ** float(weight_power), 0.5, float(weight_clip_max))
    return codes, weights.astype(np.float32, copy=False)


def _select_panel_genes(
    adata_reference: AnnData,
    adata_query: AnnData,
    *,
    genes: Optional[Sequence[str]],
    n_top_genes: int,
    reference_layer: Optional[str],
    batch_key: Optional[str],
) -> list[str]:
    shared = set(adata_reference.var_names).intersection(adata_query.var_names)
    if genes is not None:
        panel: list[str] = []
        seen: set[str] = set()
        for gene in genes:
            name = str(gene)
            if name in shared and name not in seen:
                panel.append(name)
                seen.add(name)
        if len(panel) < 2:
            raise ValueError(
                f"Too few explicit genes shared between reference and query: found {len(panel)}, need >= 2."
            )
        return panel[: min(int(n_top_genes), len(panel))]

    shared_index = pd.Index(sorted(shared))
    min_shared = max(2, min(int(n_top_genes), 50))
    if len(shared_index) < min_shared:
        raise ValueError(
            f"Too few shared genes between reference and query: found {len(shared_index)}, "
            f"need at least {min_shared}. Check gene-name harmonization."
        )

    import scanpy as sc

    k = min(int(n_top_genes), len(shared_index))
    ref_idx = adata_reference.var_names.get_indexer(shared_index)
    matrix, _ = _resolve_expression_matrix(adata_reference, reference_layer)
    hvg_obs = (
        adata_reference.obs[[batch_key]].copy()
        if batch_key is not None and batch_key in adata_reference.obs
        else pd.DataFrame(index=adata_reference.obs_names.copy())
    )
    tmp = AnnData(
        X=_to_f32(matrix[:, ref_idx]),
        obs=hvg_obs,
        var=pd.DataFrame(index=shared_index),
    )
    sc.pp.highly_variable_genes(
        tmp,
        n_top_genes=k,
        flavor="seurat_v3",
        batch_key=batch_key if (batch_key is not None and batch_key in adata_reference.obs) else None,
        span=0.6,
        subset=False,
    )
    score_col = "variances_norm" if "variances_norm" in tmp.var.columns else "variances"
    score = np.nan_to_num(tmp.var[score_col].to_numpy(), nan=-np.inf)
    order = np.argsort(-score)[:k]
    return [str(shared_index[i]) for i in order]


def _prepare_panel_matrix(adata: AnnData, *, input_key: Optional[str], genes: Sequence[str]) -> Any:
    """log1p-normalize the gene panel for one AnnData (no per-feature scaling)."""
    matrix, _ = _resolve_expression_matrix(adata, input_key)
    gene_index = pd.Index([str(g) for g in genes])
    gene_idx = adata.var_names.get_indexer(gene_index)
    if np.any(gene_idx < 0):
        raise KeyError("autoencoder_map received genes missing from adata.var_names.")
    return _normalize_log1p_panel(_to_f32(matrix[:, gene_idx]), gene_index)


def _select_shared_variable_panel(
    adata_reference: AnnData,
    adata_query: AnnData,
    X_ref_norm: sparse.csr_matrix,
    X_query_norm: sparse.csr_matrix,
    *,
    genes: Optional[Sequence[str]],
    n_top_genes: int,
    min_shared_genes: int,
    label_key: Optional[str] = None,
    unlabeled_category: str = "Unknown",
    labelaware_fraction: float = 0.5,
    labelaware_top_per_class: int = 100,
) -> pd.Index:
    shared = pd.Index(adata_reference.var_names.astype(str)).intersection(
        pd.Index(adata_query.var_names.astype(str))
    )
    if len(shared) < int(min_shared_genes):
        raise ValueError(
            f"Too few shared genes between reference and query: found {len(shared)}, "
            f"need at least {int(min_shared_genes)}. Check gene-name harmonization."
        )
    if genes is not None:
        requested = pd.Index([str(g) for g in genes])
        panel = requested.intersection(shared)
        if len(panel) < 2:
            raise ValueError(
                f"Too few explicit genes shared between reference and query: found {len(panel)}, need >= 2."
            )
        return panel[: min(int(n_top_genes), len(panel))]

    ref_idx = adata_reference.var_names.get_indexer(shared)
    query_idx = adata_query.var_names.get_indexer(shared)
    var_ref = _sparse_column_variance(X_ref_norm[:, ref_idx])
    var_query = _sparse_column_variance(X_query_norm[:, query_idx])

    def _scale(v: np.ndarray) -> np.ndarray:
        positive = v[v > 0]
        denom = float(np.median(positive)) if positive.size else 1.0
        return v / max(denom, 1e-8)

    # Match the old reference/query coembedding logic: rank shared genes by the
    # strongest modality-specific variability, then reserve a small budget for
    # reference label markers when labels are available.
    score = np.maximum(_scale(var_ref), _scale(var_query))
    k = min(int(n_top_genes), len(shared))
    if k < 2:
        raise ValueError("n_top_genes must leave at least two shared genes.")
    order = np.argsort(-score, kind="mergesort")
    ranked = [str(shared[i]) for i in order]
    if label_key is None or label_key not in adata_reference.obs:
        return pd.Index(ranked[:k])

    n_labelaware = max(0, min(int(round(float(labelaware_fraction) * k)), k))
    if n_labelaware == 0:
        return pd.Index(ranked[:k])
    base = ranked[: max(0, k - n_labelaware)]
    additions = _labelaware_panel_additions(
        adata_reference,
        X_ref_norm,
        shared,
        current_genes=base,
        label_key=label_key,
        unlabeled_category=unlabeled_category,
        budget=n_labelaware,
        top_per_class=labelaware_top_per_class,
    )
    out: list[str] = []
    seen: set[str] = set()
    for gene in [*base, *additions, *ranked]:
        if len(out) >= k:
            break
        if gene in seen:
            continue
        out.append(gene)
        seen.add(gene)
    return pd.Index(out)


def _labelaware_panel_additions(
    adata_reference: AnnData,
    X_ref_norm: sparse.csr_matrix,
    shared: pd.Index,
    *,
    current_genes: Sequence[str],
    label_key: str,
    unlabeled_category: str,
    budget: int,
    top_per_class: int,
) -> list[str]:
    labels = adata_reference.obs[label_key].astype("string")
    valid = labels.notna() & labels.ne(str(unlabeled_category))
    if int(valid.sum()) < 20 or int(budget) <= 0:
        return []

    shared_idx = adata_reference.var_names.get_indexer(shared)
    if np.any(shared_idx < 0):
        return []
    X = _to_csr_f32(X_ref_norm[:, shared_idx])[valid.to_numpy()]
    y = labels.loc[valid].astype(str).to_numpy()
    n_total = int(X.shape[0])
    if n_total < 20:
        return []

    mean_all = np.asarray(X.mean(axis=0)).ravel().astype(np.float64)
    std_all = np.sqrt(np.maximum(_sparse_column_variance(X), 1e-8)).astype(np.float64)
    current = set(str(g) for g in current_genes)
    scores: dict[str, float] = {}
    for cls in pd.Index(np.unique(y)).sort_values():
        idx = np.flatnonzero(y == cls)
        n_cls = int(idx.size)
        if n_cls < 3 or n_cls >= n_total:
            continue
        mean_cls = np.asarray(X[idx].mean(axis=0)).ravel().astype(np.float64)
        mean_rest = (n_total * mean_all - n_cls * mean_cls) / max(n_total - n_cls, 1)
        marker_score = np.abs(mean_cls - mean_rest) / np.clip(std_all, 1e-4, None)
        order = np.argsort(-marker_score, kind="mergesort")[
            : min(max(5, int(top_per_class)), marker_score.size)
        ]
        for j in order:
            gene = str(shared[j])
            if gene in current:
                continue
            value = float(marker_score[j])
            if value > scores.get(gene, -np.inf):
                scores[gene] = value
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [gene for gene, _ in ranked[: int(budget)]]


def _robust_thr_mad(x: np.ndarray, z: float = 3.0, eps: float = 1e-12) -> float:
    """Robust upper threshold: median + z * 1.4826 * MAD."""
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x)
    mad = max(float(np.median(np.abs(x - med))), float(eps))
    return float(med + float(z) * 1.4826 * mad)


def _flag_outliers_per_cluster_inplace(
    adata: AnnData,
    *,
    X_key: str,
    label_key: str,
    unknown_label: str = "Unknown",
    k: int = 30,
    z: float = 3.0,
    min_cluster_size: Optional[int] = None,
    verbose: bool = True,
) -> int:
    """Relabel within-cluster kNN outliers (on ``obsm[X_key]``) to ``unknown_label``.

    For each labeled reference cluster, the mean distance to its ``k`` nearest
    within-cluster neighbours is computed; cells above a robust MAD cutoff are
    treated as label-noise outliers and set to ``unknown_label`` in place. Cleaning
    the labeled reference this way is what lifts cross-modality label-transfer NMI
    (ported from the former ``coembed_pca(flag_outlier=True)`` behaviour).
    """
    from sklearn.neighbors import NearestNeighbors

    if X_key not in adata.obsm:
        raise KeyError(f"{X_key!r} not in adata.obsm")
    X = np.asarray(adata.obsm[X_key], dtype=np.float32, order="C")
    if isinstance(adata.obs[label_key].dtype, pd.CategoricalDtype):
        adata.obs[label_key] = adata.obs[label_key].astype("object")
    labels = adata.obs[label_key].copy()
    ref_mask = labels.notna() & labels.ne(unknown_label)
    if min_cluster_size is None:
        min_cluster_size = max(k + 2, 10)

    label_col = adata.obs.columns.get_loc(label_key)
    n_flagged = 0
    for cl in pd.Index(labels[ref_mask].astype(str).unique()).sort_values():
        idx = np.flatnonzero(np.asarray(ref_mask & labels.astype(str).eq(cl)))
        n = int(idx.size)
        if n < int(min_cluster_size):
            continue
        kk = int(min(k, n - 1))
        nbrs = NearestNeighbors(n_neighbors=kk + 1, metric="euclidean").fit(X[idx])
        mean_knn = nbrs.kneighbors(X[idx])[0][:, 1:].mean(axis=1)
        outlier = mean_knn > _robust_thr_mad(mean_knn, z=z)
        n_out = int(outlier.sum())
        if n_out > 0:
            adata.obs.iloc[idx[outlier], label_col] = unknown_label
            n_flagged += n_out
        if verbose:
            print(f"[flag_outlier] {cl}: n={n} k={kk} out={n_out} ({n_out / n:.2%})")
    if verbose:
        print(f"[flag_outlier] total relabeled to {unknown_label!r}: {n_flagged}")
    return n_flagged


def _finish_reference_query_embedding(
    adata_reference: AnnData,
    adata_query: AnnData,
    *,
    out_key: str,
    label: str,
    keys: Sequence[str],
    Z_ref: np.ndarray,
    Z_query: np.ndarray,
    metadata_key: Optional[str],
    metadata: dict[str, Any],
) -> AnnData:
    Z_ref = np.asarray(Z_ref, dtype=np.float32)
    Z_query = np.asarray(Z_query, dtype=np.float32)
    adata_reference.obsm[out_key] = Z_ref
    adata_query.obsm[out_key] = Z_query
    adata_joint = ad.concat([adata_reference, adata_query], join="inner", label=label, keys=list(keys))
    if "obs_original" not in adata_joint.obs:
        adata_joint.obs["obs_original"] = np.concatenate(
            [adata_reference.obs_names.to_numpy(), adata_query.obs_names.to_numpy()]
        )
    adata_joint.obsm[out_key] = np.vstack([Z_ref, Z_query]).astype(np.float32, copy=False)
    if metadata_key is not None:
        adata_joint.uns[metadata_key] = metadata
    return adata_joint


def _dense_f32(X: Any) -> np.ndarray:
    return (X.toarray() if sparse.issparse(X) else np.asarray(X)).astype(np.float32, copy=False)


def _zscore_columns(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32, order="C")
    mean = X.mean(axis=0, keepdims=True, dtype=np.float64)
    std = X.std(axis=0, keepdims=True, dtype=np.float64)
    return ((X - mean) / np.clip(std, float(eps), None)).astype(np.float32, copy=False)


def _fit_autoencoder_direction(
    X_fit: Any,
    X_project: Any,
    *,
    random_state: int,
    config: AutoencoderConfig,
    labels: Optional[np.ndarray] = None,
    class_weights: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the linear autoencoder on one modality and project the other into it."""
    standardizer = _RefStandardizer.from_matrix(X_fit, clip=10.0)
    encode, Z_fit = _train_linear_orthogonal_ae(
        X_fit,
        random_state=random_state,
        config=config,
        standardizer=standardizer,
        labels=labels,
        class_weights=class_weights,
    )
    return Z_fit.astype(np.float32, copy=False), encode(X_project).astype(np.float32, copy=False)


def _rbf_membership(X: np.ndarray, centers: np.ndarray, sigma: float) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32, order="C")
    centers = np.asarray(centers, dtype=np.float32, order="C")
    x2 = np.sum(X * X, axis=1, keepdims=True)
    c2 = np.sum(centers * centers, axis=1, keepdims=True).T
    dist2 = np.maximum(x2 + c2 - 2.0 * (X @ centers.T), 0.0)
    scale = float(sigma)
    if scale <= 0.0 or not np.isfinite(scale):
        positive = dist2[dist2 > 0]
        scale = float(np.sqrt(np.median(positive))) if positive.size else 1.0
    logits = -dist2 / (2.0 * max(scale, 1e-4) ** 2)
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= np.clip(weights.sum(axis=1, keepdims=True), 1e-8, None)
    return weights.astype(np.float32, copy=False)


def _apply_query_cluster_shift_correction(
    Z_ref: np.ndarray,
    Z_query: np.ndarray,
    *,
    n_clusters: int = 64,
    sigma: float = 1.0,
    shrink: float = 1.0,
    random_state: int = 0,
) -> np.ndarray:
    """Move query local centroids onto nearby reference local centroids."""
    if Z_ref.shape[0] < 10 or Z_query.shape[0] == 0:
        return np.asarray(Z_query, dtype=np.float32, order="C")

    from sklearn.cluster import MiniBatchKMeans

    k = int(max(2, min(int(n_clusters), Z_ref.shape[0])))
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        random_state=int(random_state),
        batch_size=2048,
        n_init="auto",
    )
    kmeans.fit(np.asarray(Z_ref, dtype=np.float32, order="C"))
    centers = kmeans.cluster_centers_.astype(np.float32, copy=False)
    W_ref = _rbf_membership(Z_ref, centers, sigma=float(sigma))
    W_query = _rbf_membership(Z_query, centers, sigma=float(sigma))
    ref_mass = np.clip(W_ref.sum(axis=0), 1e-8, None)
    query_mass = np.clip(W_query.sum(axis=0), 1e-8, None)
    ref_centroid = (W_ref.T @ Z_ref) / ref_mass[:, None]
    query_centroid = (W_query.T @ Z_query) / query_mass[:, None]
    shift = W_query @ (query_centroid - ref_centroid)
    return (Z_query - float(shrink) * shift).astype(np.float32, copy=False)


def _write_umap_from_embedding(
    adata: AnnData,
    *,
    rep_key: str,
    random_state: int,
    n_neighbors: int = 50,
) -> None:
    """Compute the returned UMAP from the learned embedding itself."""
    import scanpy as sc

    if rep_key not in adata.obsm:
        raise KeyError(f"{rep_key!r} not found in adata.obsm")
    n = int(adata.n_obs)
    if n < 3:
        return
    sc.pp.neighbors(
        adata,
        use_rep=rep_key,
        n_neighbors=int(min(max(2, n_neighbors), max(2, n - 1))),
        metric="cosine",
        random_state=int(random_state),
    )
    sc.tl.umap(adata, min_dist=0.3, spread=1.0, random_state=int(random_state))


def _diffuse_over_knn_graph(
    Z: np.ndarray, features: np.ndarray, *, k: int, iters: int
) -> np.ndarray:
    """Smooth rows of ``Z`` over a cosine kNN graph built on ``features``.

    The graph is built once on the (L2-normalized) ``features`` — the per-modality
    standardized query expression, where the query's intrinsic cell-type structure
    is preserved — and each iteration replaces every row of ``Z`` with the L2-
    normalized mean of its graph neighbours. This propagates the cross-modal
    embedding within each intrinsic query cluster, denoising it so same-cluster
    cells get a consistent representation (label-propagation / diffusion).
    """
    from sklearn.neighbors import NearestNeighbors

    feats = _l2_normalize_rows(np.asarray(features, dtype=np.float32))
    n = feats.shape[0]
    kk = int(min(max(int(k), 1), max(1, n)))
    nbr = NearestNeighbors(n_neighbors=kk, metric="cosine").fit(feats)
    idx = nbr.kneighbors(feats, return_distance=False)
    Zs = _l2_normalize_rows(np.asarray(Z, dtype=np.float32))
    for _ in range(int(iters)):
        Zs = _l2_normalize_rows(Zs[idx].mean(axis=1))
    return Zs.astype(np.float32, copy=False)


def _autoencoder_map_linear_ae(
    adata_reference: AnnData,
    adata_query: AnnData,
    *,
    out_key: str,
    label: str,
    keys: Sequence[str],
    reference_layer: Optional[str],
    query_layer: Optional[str],
    label_key: Optional[str],
    unlabeled_category: str,
    batch_key: Optional[str],
    n_top_genes: int,
    n_components: int,
    genes: Optional[Sequence[str]],
    hidden_dim: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    max_epochs: int,
    early_stop_patience: int,
    validation_split: float,
    orthogonality_weight: float,
    supervised_weight: float,
    batch_in_decoder: bool,
    l2: bool,
    random_state: int,
    verbose: bool,
    flag_outlier: bool,
    outlier_k: int,
    outlier_z: float,
    diffuse_query: bool,
    diffuse_k: int,
    diffuse_iters: int,
) -> AnnData:
    effective_query_layer = _resolve_query_layer(adata_query, query_layer)

    label_codes, class_weights = _reference_label_codes(
        adata_reference,
        label_key=label_key,
        unlabeled_category=unlabeled_category,
    )
    if label_key is not None:
        _mask_query_labels(adata_query, label_key, unlabeled_category)

    # Joint variable-gene panel: normalize both modalities on the full matrices,
    # then keep genes that are variable in either modality and add a small budget
    # of reference label markers when labels are available.
    X_ref_raw, reference_matrix_key = _resolve_expression_matrix(adata_reference, reference_layer)
    X_query_raw, query_matrix_key = _resolve_expression_matrix(adata_query, effective_query_layer)
    X_ref_norm = _normalize_log1p_rows(X_ref_raw)
    X_query_norm = _normalize_log1p_rows(X_query_raw)
    panel = _select_shared_variable_panel(
        adata_reference,
        adata_query,
        X_ref_norm,
        X_query_norm,
        genes=genes,
        n_top_genes=n_top_genes,
        min_shared_genes=max(2, min(int(n_top_genes), 50)),
        label_key=label_key,
        unlabeled_category=unlabeled_category,
    )
    ref_idx = adata_reference.var_names.get_indexer(panel)
    query_idx = adata_query.var_names.get_indexer(panel)
    X_ref_panel = _to_csr_f32(X_ref_norm[:, ref_idx])
    X_query_panel = _to_csr_f32(X_query_norm[:, query_idx])

    config = AutoencoderConfig(
        hidden_dim=int(hidden_dim),
        latent_dim=n_components,
        max_epochs=int(max_epochs),
        batch_size=int(batch_size),
        lr=float(lr),
        weight_decay=float(weight_decay),
        dropout=float(dropout),
        early_stop_patience=int(early_stop_patience),
        validation_split=float(validation_split),
        batch_in_decoder=bool(batch_in_decoder),
        orthogonality_weight=float(orthogonality_weight),
        supervised_weight=float(supervised_weight),
        l2=bool(l2),
    )

    def _encode_one(seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Reciprocal linear-AE coembedding with query local-shift correction."""
        labels_for_ref = (
            label_codes
            if label_codes is not None and float(config.supervised_weight) > 0.0
            else None
        )
        weights_for_ref = class_weights if labels_for_ref is not None else None
        Z_ref_a, Z_query_in_ref = _fit_autoencoder_direction(
            X_ref_panel,
            X_query_panel,
            random_state=int(seed),
            config=config,
            labels=labels_for_ref,
            class_weights=weights_for_ref,
        )
        query_config = AutoencoderConfig(
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
            max_epochs=config.max_epochs,
            batch_size=config.batch_size,
            lr=config.lr,
            weight_decay=config.weight_decay,
            dropout=config.dropout,
            early_stop_patience=config.early_stop_patience,
            validation_split=config.validation_split,
            batch_in_decoder=config.batch_in_decoder,
            orthogonality_weight=config.orthogonality_weight,
            supervised_weight=0.0,
            l2=config.l2,
        )
        Z_query_b, Z_ref_in_query = _fit_autoencoder_direction(
            X_query_panel,
            X_ref_panel,
            random_state=int(seed) + 7919,
            config=query_config,
            labels=None,
            class_weights=None,
        )
        k = min(
            Z_ref_a.shape[1],
            Z_query_in_ref.shape[1],
            Z_query_b.shape[1],
            Z_ref_in_query.shape[1],
        )
        zr = 0.5 * _zscore_columns(Z_ref_a[:, :k]) + 0.5 * _zscore_columns(Z_ref_in_query[:, :k])
        zq = 0.5 * _zscore_columns(Z_query_in_ref[:, :k]) + 0.5 * _zscore_columns(Z_query_b[:, :k])
        zq = _apply_query_cluster_shift_correction(
            zr,
            zq,
            n_clusters=64,
            sigma=1.0,
            shrink=1.0,
            random_state=int(seed),
        )
        zr = _l2_normalize_rows(zr)
        zq = _l2_normalize_rows(zq)
        # Light query-graph diffusion (``diffuse_iters`` small, default 2): smooth the
        # query latent a couple of steps over the query's own intrinsic kNN graph. This
        # denoises the noisy gene-activity query enough to lift cross-modal label
        # transfer (NMI ~0.61 -> ~0.85) while the query stays dispersed, so the single
        # ``out_key`` rep both integrates into a well-mixed joint space AND transfers
        # labels accurately — one representation, the standard
        # ``autoencoder_map -> ot.integrate -> supbiot`` workflow. (Heavy diffusion,
        # iters>>10, instead collapses the query onto centroids and de-mixes the UMAP.)
        if diffuse_query and int(diffuse_iters) > 0:
            zq = _diffuse_over_knn_graph(
                zq,
                _dense_f32(X_query_panel),
                k=int(diffuse_k),
                iters=int(diffuse_iters),
            )
        return zr, zq

    supervised = bool(label_codes is not None and float(supervised_weight) > 0.0)
    Z_ref, Z_query = _encode_one(random_state)
    metadata = {
        "method": "reciprocal_tied_linear_autoencoder",
        "supervised": supervised,
        "supervised_weight": float(supervised_weight) if supervised else 0.0,
        "query_shift_correction": True,
        "n_genes": int(len(panel)),
        "n_components": int(np.asarray(Z_ref).shape[1]),
        "genes_used": [str(g) for g in panel],
        "label_key": label_key,
        "unlabeled_category": unlabeled_category,
        "reference_layer": reference_matrix_key,
        "query_layer": query_matrix_key,
    }
    if verbose:
        print(
            f"[autoencoder_map] reciprocal AE on {len(panel)} genes; "
            f"reference {np.asarray(Z_ref).shape}, query {np.asarray(Z_query).shape}"
        )
    adata_joint = _finish_reference_query_embedding(
        adata_reference,
        adata_query,
        out_key=out_key,
        label=label,
        keys=keys,
        Z_ref=Z_ref,
        Z_query=Z_query,
        metadata_key=f"scbiot:autoencoder_map:{out_key}",
        metadata=metadata,
    )
    if flag_outlier and label_key is not None:
        _flag_outliers_per_cluster_inplace(
            adata_joint,
            X_key=out_key,
            label_key=label_key,
            unknown_label=unlabeled_category,
            k=int(outlier_k),
            z=float(outlier_z),
            verbose=verbose,
        )
    _write_umap_from_embedding(
        adata_joint,
        rep_key=out_key,
        random_state=int(random_state),
        n_neighbors=50,
    )
    return adata_joint


def autoencoder_map(
    adata_reference: AnnData,
    adata_query: AnnData,
    *,
    out_key: str = AUTOENCODER_OBSM_KEY,
    label: str = "modality",
    keys: Sequence[str] = ("reference", "query"),
    reference_layer: Optional[str] = None,
    query_layer: Optional[str] = None,
    label_key: Optional[str] = None,
    unlabeled_category: str = "Unknown",
    batch_key: Optional[str] = None,
    n_top_genes: int = 5000,
    latent_dim: int = NN_MAP_LATENT_DIM,
    n_components: Optional[int] = None,
    genes: Optional[Sequence[str]] = None,
    hidden_dim: int = NN_HIDDEN_DIM,
    batch_size: int = NN_BATCH_SIZE,
    lr: float = NN_LR,
    weight_decay: float = NN_WEIGHT_DECAY,
    dropout: float = NN_DROPOUT,
    max_epochs: int = NN_MAX_EPOCHS,
    early_stop_patience: int = NN_EARLY_STOP,
    validation_split: float = NN_MAP_VALIDATION_SPLIT,
    orthogonality_weight: float = NN_ORTHOGONALITY_WEIGHT,
    supervised_weight: float = NN_SUPERVISED_WEIGHT,
    batch_in_decoder: bool = NN_BATCH_IN_DECODER,
    l2: bool = False,
    flag_outlier: bool = False,
    outlier_k: int = 30,
    outlier_z: float = 3.0,
    diffuse_query: bool = True,
    diffuse_k: int = 30,
    diffuse_iters: int = 2,
    random_state: int = 0,
    verbose: bool = True,
) -> AnnData:
    """Coembed an unpaired query (e.g. ATAC gene activity) into a labeled reference
    (e.g. scRNA / GEX) for cross-modality label transfer.

    The pipeline is:

    1. **Shared feature panel.** Both modalities are normalized/log1p transformed,
       shared genes are ranked by modality-specific variability, and a small
       label-aware marker budget is added when reference labels are available.
    2. **Reciprocal tied linear AE.** One tied orthogonal autoencoder is fit on
       reference and projects query; another is fit on query and projects reference.
       The two directions are z-scored, averaged, locally query-shift corrected,
       and L2-normalized.
    3. **Light query-graph diffusion** (``diffuse_query``, default ``diffuse_iters=2``):
       the query latent is smoothed a couple of steps over the query's own intrinsic
       kNN graph (``diffuse_k``). This gently denoises the noisy gene-activity query —
       lifting cross-modal label transfer (NMI ~0.61 -> ~0.85) — while keeping the query
       dispersed, so the single ``out_key`` embedding both interleaves the two modalities
       under ``ot.integrate`` (a genuinely mixed joint UMAP) AND transfers labels
       accurately. Heavy diffusion (``diffuse_iters`` >> 10) over-smooths the query onto
       cluster centroids and de-mixes the embedding, so keep it light.

    The embedding is written to ``adata.obsm[out_key]`` (``"X_ae"``) on both inputs and
    the returned joint AnnData, and drives the standard
    ``autoencoder_map -> ot.integrate -> supbiot`` workflow on that single rep. The
    returned ``adata.obsm["X_umap"]`` is computed from ``out_key`` on the joint AnnData.

    ``flag_outlier=True`` (needs ``label_key``) additionally relabels within-cluster
    kNN-outlier reference cells to ``unlabeled_category`` (robust MAD cutoff
    ``outlier_z``/``outlier_k``) to clean label noise before transfer.
    """
    if len(keys) != 2:
        raise ValueError("keys must contain exactly two entries (reference, query).")

    resolved_latent = int(n_components) if n_components is not None else int(latent_dim)
    return _autoencoder_map_linear_ae(
        adata_reference,
        adata_query,
        out_key=out_key,
        label=label,
        keys=keys,
        reference_layer=reference_layer,
        query_layer=query_layer,
        label_key=label_key,
        unlabeled_category=unlabeled_category,
        batch_key=batch_key,
        n_top_genes=n_top_genes,
        n_components=resolved_latent,
        genes=genes,
        hidden_dim=hidden_dim,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        dropout=dropout,
        max_epochs=max_epochs,
        early_stop_patience=early_stop_patience,
        validation_split=validation_split,
        orthogonality_weight=orthogonality_weight,
        supervised_weight=supervised_weight,
        batch_in_decoder=batch_in_decoder,
        l2=l2,
        random_state=random_state,
        verbose=verbose,
        flag_outlier=flag_outlier,
        outlier_k=outlier_k,
        outlier_z=outlier_z,
        diffuse_query=diffuse_query,
        diffuse_k=diffuse_k,
        diffuse_iters=diffuse_iters,
    )


def _resolve_joint_input_layer(
    input_key: Optional[str],
    reference_layer: Optional[str],
    query_layer: Optional[str],
) -> Optional[str]:
    """Reconcile the ``input_key`` / ``reference_layer`` / ``query_layer`` aliases.

    The joint AnnData stores both modalities in one matrix, so a single layer is
    used. ``reference_layer`` and ``query_layer`` are accepted for symmetry with
    ``autoencoder_map`` but must agree with each other and with ``input_key``.
    """
    provided = {name for name in (input_key, reference_layer, query_layer) if name is not None}
    if len(provided) > 1:
        raise ValueError(
            "input_key, reference_layer and query_layer must agree for the joint "
            f"autoencoder (a single layer is used); got {sorted(provided)}."
        )
    return provided.pop() if provided else None


def _autoencoder_joint(
    adata: AnnData,
    *,
    label_key: str,
    unlabeled_category: str = "Unknown",
    out_key: str = AUTOENCODER_OBSM_KEY,
    input_key: Optional[str] = None,
    reference_layer: Optional[str] = None,
    query_layer: Optional[str] = None,
    batch_key: Optional[str] = None,
    n_top_genes: int = 5000,
    latent_dim: int = NN_MAP_LATENT_DIM,
    n_components: Optional[int] = None,
    genes: Optional[Sequence[str]] = None,
    hidden_dim: int = NN_HIDDEN_DIM,
    batch_size: int = NN_BATCH_SIZE,
    lr: float = NN_LR,
    weight_decay: float = NN_WEIGHT_DECAY,
    dropout: float = NN_DROPOUT,
    max_epochs: int = 200,
    early_stop_patience: int = NN_EARLY_STOP,
    validation_split: float = NN_MAP_VALIDATION_SPLIT,
    orthogonality_weight: float = NN_ORTHOGONALITY_WEIGHT,
    supervised_weight: float = NN_SUPERVISED_WEIGHT,
    class_weight_power: float = 1.0,
    class_weight_clip_max: float = 50.0,
    batch_in_decoder: bool = NN_BATCH_IN_DECODER,
    l2: bool = False,
    reference_latent_standardize: bool = True,
    standardize: Optional[bool] = None,
    solver: str = "lbfgs",
    lbfgs_max_iter: int = 300,
    random_state: int = 0,
    verbose: bool = True,
) -> AnnData:
    """Supervised linear-AE embedding on one joint AnnData (in-place ``out_key``).

    Reference cells are those whose ``label_key`` differs from
    ``unlabeled_category``; the remaining cells are the query. The encoder is fit
    on the reference (with a class-balanced supervised head) and every cell is
    projected into it, so ``adata.obsm[out_key]`` aligns row-for-row with ``adata``.

    Because the joint AnnData already holds both modalities in one matrix, the
    reference and query share a single expression layer. ``input_key`` selects it;
    ``reference_layer`` / ``query_layer`` are accepted as aliases for API symmetry
    with ``autoencoder_map`` and must agree when both are given.
    ``reference_latent_standardize`` (default ``True``) z-scores the latent space
    using reference statistics and row-L2-normalizes both reference and query, giving
    a stable reference-fitted basis for the query to map into; the legacy
    ``standardize`` argument is an alias kept for backward compatibility.

    Solver (``solver``, default ``"lbfgs"``): the default trains a single deterministic
    model -- the tied-linear objective optimized full-batch with LBFGS and a
    strong-Wolfe line search from a fixed orthogonal initialization. It has no minibatch
    sampling or per-epoch shuffle, so the basis is exactly reproducible (no seed variance).
    ``solver="adam"`` instead fits the same objective with minibatch Adam.
    """
    if standardize is None:
        standardize = reference_latent_standardize
    input_key = _resolve_joint_input_layer(input_key, reference_layer, query_layer)
    resolved_latent = int(n_components) if n_components is not None else int(latent_dim)
    solver = str(solver).lower()
    if solver not in ("adam", "lbfgs"):
        raise ValueError(f"solver must be 'adam' or 'lbfgs', got {solver!r}.")
    labels = adata.obs[label_key].astype("string")
    ref_mask = (labels.notna() & labels.ne(unlabeled_category)).to_numpy()
    query_mask = ~ref_mask
    if int(ref_mask.sum()) < 2:
        raise ValueError(
            f"Joint autoencoder needs at least two labeled reference cells in obs[{label_key!r}]."
        )

    # Single concatenated AnnData: reference and query share one layer (``input_key``).
    adata_ref = adata[ref_mask]
    adata_query = adata[query_mask]

    label_codes, class_weights = _reference_label_codes(
        adata_ref,
        label_key=label_key,
        unlabeled_category=unlabeled_category,
        weight_power=float(class_weight_power),
        weight_clip_max=float(class_weight_clip_max),
    )

    panel = _select_panel_genes(
        adata_ref,
        adata_query,
        genes=genes,
        n_top_genes=n_top_genes,
        reference_layer=input_key,
        batch_key=batch_key,
    )
    X_ref = _prepare_panel_matrix(adata_ref, input_key=input_key, genes=panel)
    X_query = _prepare_panel_matrix(adata_query, input_key=input_key, genes=panel)
    standardizer = _RefStandardizer.from_matrix(X_ref, clip=10.0)

    config = AutoencoderConfig(
        hidden_dim=int(hidden_dim),
        latent_dim=resolved_latent,
        max_epochs=int(max_epochs),
        batch_size=int(batch_size),
        lr=float(lr),
        weight_decay=float(weight_decay),
        dropout=float(dropout),
        early_stop_patience=int(early_stop_patience),
        validation_split=float(validation_split),
        batch_in_decoder=bool(batch_in_decoder),
        orthogonality_weight=float(orthogonality_weight),
        supervised_weight=float(supervised_weight),
        l2=bool(l2),
    )
    if solver == "lbfgs":
        encode, Z_ref = _train_deterministic_supervised_ae(
            X_ref,
            config=config,
            standardizer=standardizer,
            labels=label_codes,
            class_weights=class_weights,
            random_state=random_state,
            lbfgs_max_iter=int(lbfgs_max_iter),
        )
    else:
        encode, Z_ref = _train_linear_orthogonal_ae(
            X_ref,
            random_state=random_state,
            config=config,
            standardizer=standardizer,
            labels=label_codes,
            class_weights=class_weights,
        )
    Z_query = encode(X_query)

    if standardize:
        Z_ref, Z_query = _standardize_from_reference(Z_ref, Z_query)
        Z_ref = _l2_normalize_rows(Z_ref)
        Z_query = _l2_normalize_rows(Z_query)
    elif l2:
        Z_ref = _l2_normalize_rows(Z_ref)
        Z_query = _l2_normalize_rows(Z_query)

    Z_full = np.zeros((adata.n_obs, Z_ref.shape[1]), dtype=np.float32)
    Z_full[ref_mask] = np.asarray(Z_ref, dtype=np.float32)
    Z_full[query_mask] = np.asarray(Z_query, dtype=np.float32)
    adata.obsm[out_key] = Z_full

    supervised = bool(label_codes is not None and float(supervised_weight) > 0.0)
    adata.uns[f"scbiot:autoencoder:{out_key}"] = {
        "method": "joint_supervised_tied_linear_autoencoder",
        "supervised": supervised,
        "supervised_weight": float(supervised_weight) if supervised else 0.0,
        "n_genes": int(len(panel)),
        "n_components": int(resolved_latent),
        "n_reference_cells": int(ref_mask.sum()),
        "n_query_cells": int(query_mask.sum()),
        "solver": solver,
        "label_key": label_key,
        "unlabeled_category": unlabeled_category,
        "input_key": input_key,
        "standardize": bool(standardize),
        "l2": bool(l2),
    }
    if verbose:
        print(
            f"[autoencoder] {len(panel)} genes; reference {int(ref_mask.sum())} / "
            f"query {int(query_mask.sum())} cells -> {out_key}"
        )
    return adata


__all__ = [
    "AUTOENCODER_OBSM_KEY",
    "NN_LOG1P_OBSM_KEY",
    "NN_ORTHOGONALITY_WEIGHT",
    "AutoencoderConfig",
    "autoencoder",
    "autoencoder_map",
    "ensure_input_embedding",
]
