from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.preprocessing import LabelEncoder

from .integrate import integrate_ot
from ..utils.ot_helpers import (
    _FAISS_AVAIL,
    _as_nd_f32_c,
    _faiss_ready,
    _get_faiss_index,
    _local_knn_density,
    _minikm_centers,
)
from ._presets import get_modality_preset

# ===================== Centroid-level OT helper for very large datasets (memory-friendly) =====================

def integrate_centroids(
    adata_full: Any,
    obsm_key: str = "X_pca",
    batch_key: str = "batch",
    out_key: str = "scBIOT",
    n_centroids_per_batch: int = 2048,
    max_samples_per_batch: int = 500_000,
    k_interp: int = 8,
    chunk_size: int = 500_000,
    use_gpu: bool = True,
    gpu_device: int = 0,
    tmp_path: Optional[str] = None,   # if not None -> write output to a memmap file
    modality: Optional[str] = None,
    **integrate_kwargs: Any,
) -> Tuple[Any, Dict[str, float | int]]:
    """
    Centroid-level OT + FAISS interpolation for very large datasets (e.g. Tahoe 100M),
    implemented in a memory-friendly way:

      * Does NOT materialize the full X_pca as a single numpy array.
      * Builds centroids by reading only the subset of rows needed per batch.
      * Interpolates back to all cells in chunks, reading each chunk directly from adata.obsm.

    Parameters
    ----------
    adata_full
        Full AnnData with all cells (e.g. Tahoe 100M). Must have `obsm[obsm_key]` and `obs[batch_key]`.
    obsm_key
        Embedding key to use as the OT space (typically "X_pca").
    batch_key
        Batch column in `adata_full.obs`.
    out_key
        Output key to store the integrated coordinates in `adata_full.obsm[out_key]`.
    n_centroids_per_batch
        Target number of centroids per batch (upper bound; smaller batches get fewer).
    max_samples_per_batch
        Maximum number of raw cells per batch used to fit KMeans (for speed).
    k_interp
        Number of nearest centroids used when interpolating the displacement field.
    chunk_size
        Number of cells per chunk when interpolating over the full dataset.
    use_gpu / gpu_device
        Controls FAISS GPU usage (if available) and OT backend choice inside `integrate_ot`.
    tmp_path
        If provided, the full integrated embedding is stored as a memmap file at this path
        to limit peak RAM usage. If None, a regular in-memory numpy array is used.
    modality
        Optional OT preset ("rna", "supervised", "atac", or "anchor") to initialize OT
        hyper-parameters. Use ``"centroid"`` to keep centroid defaults while relying on
        explicit OT keyword arguments.
    **integrate_kwargs
        Forwarded directly to `integrate_ot` (e.g. reg, reg_m, reference, true_label_key, etc.).

    Returns
    -------
    adata_full
        The same AnnData, with integrated coordinates stored in `adata_full.obsm[out_key]`.
    metrics
        Metrics dictionary returned by `integrate_ot`, augmented with `n_centroids`.

    Examples
    --------
    Basic usage:

    >>> import scbiot as scb
    >>> adata, metrics = scb.ot.integrate_centroids(
    ...     adata,
    ...     obsm_key="X_pca",
    ...     batch_key="batch",
    ...     out_key="X_ot",
    ...     n_centroids_per_batch=2048,
    ... )
    """
    if modality is not None:
        modality_norm = str(modality).lower()
        if modality_norm == "paired":
            raise ValueError(
                "integrate_centroids supports presets rna, supervised, atac, or anchor (paired not supported)."
            )
        if modality_norm == "centroid":
            preset_kwargs = {}
        else:
            preset_kwargs = dict(get_modality_preset(modality_norm))
    else:
        preset_kwargs = {}
    ot_kwargs: Dict[str, Any] = {**preset_kwargs, **integrate_kwargs}

    seed = int(ot_kwargs.get("random_state", 0))

    # 1) Batch labels (cheap, no big arrays)
    b_raw_full = adata_full.obs[batch_key].to_numpy()
    le = LabelEncoder()
    b_enc_full = le.fit_transform(b_raw_full).astype(np.int32, copy=False)

    rng = np.random.default_rng(seed)
    unique_batches = np.unique(b_enc_full)

    centers_list: List[np.ndarray] = []
    batches_anchor: List[np.ndarray] = []

    # For dimension: only peek at the first row of the embedding
    first_row = np.asarray(adata_full.obsm[obsm_key][0:1], dtype=np.float32, order="C")
    d_embed = first_row.shape[1]

    # 2) Build centroids per batch WITHOUT ever assembling X_full
    for lbl in unique_batches:
        idx_batch = np.where(b_enc_full == lbl)[0]
        if len(idx_batch) == 0:
            continue

        # Subsample raw cells for KMeans if the batch is huge.
        if len(idx_batch) > max_samples_per_batch:
            idx_sub = rng.choice(idx_batch, size=max_samples_per_batch, replace=False)
        else:
            idx_sub = idx_batch

        # Pull only the needed rows for this batch
        X_batch = np.asarray(
            adata_full.obsm[obsm_key][idx_sub],
            dtype=np.float32,
            order="C",
        )

        n_clusters = int(max(1, min(n_centroids_per_batch, len(idx_sub))))

        # Optional: density-based weights to emphasize bridges/rare regions (can set to None to disable)
        w_batch = _local_knn_density(
            X_batch,
            k=15,
            use_gpu=use_gpu,
            device=gpu_device,
        )

        C = _minikm_centers(
            X_batch,
            n_clusters=n_clusters,
            seed=seed + int(lbl),
            use_gpu=use_gpu,
            device=gpu_device,
            weights=w_batch,
        )
        centers_list.append(C)

        # Use original batch *string* for each centroid.
        batch_label_raw = b_raw_full[idx_batch[0]]
        batches_anchor.append(
            np.full(C.shape[0], batch_label_raw, dtype=b_raw_full.dtype)
        )

    if len(centers_list) == 0:
        raise ValueError("No centroids could be computed; check your batch_key and data.")

    X_anchor = np.vstack(centers_list).astype(np.float32, order="C")
    b_raw_anchor = np.concatenate(batches_anchor)

    # 3) Centroid-level AnnData and run the original integrate_ot on it.
    try:
        import anndata as ad
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "anndata and pandas are required for integrate_centroids. "
            "Install them via `pip install anndata pandas`."
        ) from e

    obs_anchor = pd.DataFrame({batch_key: b_raw_anchor})
    adata_anchor = ad.AnnData(
        np.zeros((X_anchor.shape[0], 1), dtype=np.float32),
        obs=obs_anchor,
    )
    adata_anchor.obsm[obsm_key] = X_anchor.copy()

    call_kwargs: Dict[str, Any] = dict(ot_kwargs)
    call_kwargs.update(
        obsm_key=obsm_key,
        batch_key=batch_key,
        out_key=out_key,
        use_gpu=use_gpu,
        gpu_device=gpu_device,
    )

    adata_anchor, metrics = integrate_ot(adata_anchor, **call_kwargs)

    X_anchor_ot = _as_nd_f32_c(adata_anchor.obsm[out_key])
    disp_anchor = X_anchor_ot - X_anchor  # (N_centroids, d_embed)

    # 4) Interpolate centroid displacement back to ALL cells with FAISS, in chunks.
    if not _FAISS_AVAIL:
        raise RuntimeError(
            "FAISS is required for centroid-level interpolation. "
            "Install it (e.g. `pip install faiss-gpu` or `faiss-cpu`)."
        )

    xb = _faiss_ready(X_anchor)  # centroids as FAISS base
    d = xb.shape[1]
    index = _get_faiss_index(d, use_gpu=use_gpu, device=gpu_device)
    index.add(xb)

    N_total = adata_full.n_obs
    k = int(max(1, min(k_interp, xb.shape[0])))

    # Decide where to store the full corrected embedding
    if tmp_path is not None:
        # Memmap on disk -> low peak RAM, good for 100M cells
        X_ot_full = np.memmap(
            tmp_path,
            mode="w+",
            dtype=np.float32,
            shape=(N_total, d_embed),
        )
    else:
        # Regular in-memory array (OK if you have enough RAM)
        X_ot_full = np.empty((N_total, d_embed), dtype=np.float32)

    for start in range(0, N_total, chunk_size):
        end = min(start + chunk_size, N_total)

        # Read only this chunk from the embedding
        X_chunk = np.asarray(
            adata_full.obsm[obsm_key][start:end],
            dtype=np.float32,
            order="C",
        )

        D2, I = index.search(X_chunk, k)     # (chunk, k)
        D2 = np.maximum(D2, 1e-6)

        # Inverse-distance weights (normalized)
        D = np.sqrt(D2)
        D = np.maximum(D, 1e-6)
        W = 1.0 / D
        W /= W.sum(axis=1, keepdims=True)

        disp_neighbors = disp_anchor[I]     # (chunk, k, d_embed)
        disp_chunk = (disp_neighbors * W[..., None]).sum(axis=1)  # (chunk, d_embed)

        X_ot_full[start:end] = X_chunk + disp_chunk.astype(np.float32, copy=False)

    # 5) Write result back into the full AnnData.
    # NOTE: this will load X_ot_full into memory, but by this point we are done with
    # the expensive OT / FAISS steps; if that still exceeds RAM, you can instead
    # keep the memmap and avoid holding the whole array in memory at once.
    adata_full.obsm[out_key] = np.asarray(X_ot_full, dtype=np.float32)

    metrics = dict(metrics)
    metrics["n_centroids"] = int(X_anchor.shape[0])
    return adata_full, metrics
