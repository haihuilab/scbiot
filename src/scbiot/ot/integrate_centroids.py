from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.preprocessing import LabelEncoder

from .integrate import DEFAULT_MAX_ITER, integrate_ot, _infer_modality, _normalize_embedding, _MODALITY_DEFAULTS
from ..utils.ot_helpers import (
    _FAISS_AVAIL,
    _ensure_at_least_two_prototypes,
    _as_nd_f32_c,
    _faiss_ready,
    _get_faiss_index,
    _local_knn_density,
    _minikm_centers,
)

# ===================== Centroid-level OT helper for very large datasets (memory-friendly) =====================

def integrate_centroids(
    adata_full: Any,
    obsm_key: str = "X_pca",
    batch_key: str = "batch",
    out_key: str = "scBIOT",
    modality: str = "auto",
    strength: float = 0.5,
    conservation: float = 0.5,
    prototypes: float = 0.5,
    supervision: float = 0.5,
    approximate: bool = False,
    reference: str = "auto",
    label_key: Optional[str] = None,
    unlabeled_category: Any = "unknown",
    random_state: int = 0,
    n_centroids_per_batch: int = 2048,
    max_samples_per_batch: int = 500_000,
    k_interp: int = 8,
    chunk_size: int = 500_000,
    use_gpu: bool = True,
    gpu_device: int = 0,
    ot_backend: str = "torch",
    max_iter: int = DEFAULT_MAX_ITER,
    verbose: bool = True,
    tmp_path: Optional[str] = None,   # if not None -> write output to a memmap file
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
    modality
        Modality hint ("auto", "rna", "atac") passed through to normalization and OT.
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
    strength / conservation / prototypes / supervision
        Semantic 0–1 knobs forwarded to `integrate_ot` on centroid embeddings.
    approximate / reference
        OT solver and reference selection controls forwarded to `integrate_ot`.
    max_iter
        Maximum number of outer optimization iterations (forwarded to `integrate_ot`).

    Returns
    -------
    adata_full
        The same AnnData, with integrated coordinates stored in `adata_full.obsm[out_key]`.
    metrics
        Metrics dictionary returned by `integrate_ot`, augmented with `n_centroids`.
    """
    try:
        import anndata as ad
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "anndata and pandas are required for integrate_centroids. "
            "Install them via `pip install anndata pandas`."
        ) from e

    seed = int(random_state)
    use_labels = label_key is not None
    labels_full = None
    unknown_label: Any = "unknown"
    unknown_aliases: Tuple[str, ...] = (
        "unknown",
        "Unknown",
        "unlabeled",
        "unlabelled",
        "UNK",
        "NA",
        "NaN",
        "None",
        "",
    )
    if use_labels:
        if label_key not in adata_full.obs:
            raise KeyError(f"Label key '{label_key}' not found in adata_full.obs.")
        labels_full = adata_full.obs[label_key].to_numpy()
        if unlabeled_category is None:
            unknown_label = np.nan
        elif isinstance(unlabeled_category, (list, tuple, set)):
            if len(unlabeled_category):
                unknown_label = sorted(str(v) for v in unlabeled_category)[0]
            else:
                unknown_label = "unknown"
            unknown_aliases = unknown_aliases + tuple(str(v) for v in unlabeled_category)
        else:
            unknown_label = str(unlabeled_category)
            unknown_aliases = unknown_aliases + (str(unlabeled_category),)
    unknown_is_nan = bool(pd.isna(unknown_label)) if use_labels else False

    # 1) Batch labels (cheap, no big arrays)
    b_raw_full = adata_full.obs[batch_key].to_numpy()
    le = LabelEncoder()
    b_enc_full = le.fit_transform(b_raw_full).astype(np.int32, copy=False)

    rng = np.random.default_rng(seed)
    unique_batches = np.unique(b_enc_full)

    centers_list: List[np.ndarray] = []
    batches_anchor: List[np.ndarray] = []
    labels_anchor: List[np.ndarray] = []

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

        if use_labels:
            C, assign = _minikm_centers(
                X_batch,
                n_clusters=n_clusters,
                seed=seed + int(lbl),
                use_gpu=use_gpu,
                device=gpu_device,
                weights=w_batch,
                return_labels=True,
            )
        else:
            C = _minikm_centers(
                X_batch,
                n_clusters=n_clusters,
                seed=seed + int(lbl),
                use_gpu=use_gpu,
                device=gpu_device,
                weights=w_batch,
            )
        C = _ensure_at_least_two_prototypes(C, seed=seed + 101 + int(lbl))
        centers_list.append(C)

        # Use original batch *string* for each centroid.
        batch_label_raw = b_raw_full[idx_batch[0]]
        batches_anchor.append(
            np.full(C.shape[0], batch_label_raw, dtype=b_raw_full.dtype)
        )
        if use_labels and labels_full is not None:
            labels_sub = labels_full[idx_sub]
            labels_clean = np.asarray(labels_sub, dtype=object)
            labels_norm = pd.Series(labels_clean).astype(str).str.strip().str.lower()
            unk_set = {str(a).strip().lower() for a in unknown_aliases}
            unk_mask = pd.isna(labels_clean) | labels_norm.isin(unk_set)
            labels_clean = labels_clean.copy()
            labels_clean[unk_mask] = unknown_label

            centroid_labels = np.full(C.shape[0], unknown_label, dtype=object)
            for c in range(C.shape[0]):
                mask = assign == c
                if not np.any(mask):
                    continue
                lab = labels_clean[mask]
                if unknown_is_nan:
                    known = lab[~pd.isna(lab)]
                else:
                    known = lab[lab != unknown_label]
                if known.size == 0:
                    continue
                vals, counts = np.unique(known, return_counts=True)
                centroid_labels[c] = vals[int(np.argmax(counts))]
            labels_anchor.append(centroid_labels)

    if len(centers_list) == 0:
        raise ValueError("No centroids could be computed; check your batch_key and data.")

    X_anchor = np.vstack(centers_list).astype(np.float32, order="C")
    b_raw_anchor = np.concatenate(batches_anchor)

    # 3) Centroid-level AnnData and run the original integrate_ot on it.
    obs_anchor = pd.DataFrame({batch_key: b_raw_anchor})
    if use_labels and labels_anchor:
        obs_anchor[label_key] = np.concatenate(labels_anchor)
    adata_anchor = ad.AnnData(
        np.zeros((X_anchor.shape[0], 1), dtype=np.float32),
        obs=obs_anchor,
    )
    adata_anchor.obsm[obsm_key] = X_anchor.copy()

    if modality == "auto" and hasattr(adata_full, "uns") and isinstance(getattr(adata_full, "uns"), dict):
        for key in ("scbiot_modality", "modality"):
            value = str(adata_full.uns.get(key, "")).lower()
            if value in _MODALITY_DEFAULTS:
                modality = value
                break
    n_total = int(adata_full.n_obs)
    sample_n = int(min(4096, n_total))
    X_mod = np.asarray(
        adata_full.obsm[obsm_key][:sample_n],
        dtype=np.float32,
        order="C",
    )
    modality = _infer_modality(modality, obsm_key, X_mod)
    X_anchor_norm, norm_params = _normalize_embedding(X_anchor, modality)

    adata_anchor, metrics = integrate_ot(
        adata_anchor,
        obsm_key=obsm_key,
        batch_key=batch_key,
        out_key=out_key,
        strength=strength,
        conservation=conservation,
        prototypes=prototypes,
        supervision=supervision,
        approximate=approximate,
        reference=reference,
        label_key=label_key,
        unlabeled_category=unlabeled_category,
        use_gpu=use_gpu,
        gpu_device=gpu_device,
        ot_backend=ot_backend,
        modality=modality,
        random_state=random_state,
        max_iter=max_iter,
        verbose=verbose,
    )

    X_anchor_ot = _as_nd_f32_c(adata_anchor.obsm[out_key])
    disp_anchor = X_anchor_ot - X_anchor_norm  # (N_centroids, d_embed)

    # 4) Interpolate centroid displacement back to ALL cells with FAISS, in chunks.
    if not _FAISS_AVAIL:
        raise RuntimeError(
            "FAISS is required for centroid-level interpolation. "
            "Install it (e.g. `pip install faiss-gpu` or `faiss-cpu`)."
        )

    xb = _faiss_ready(X_anchor_norm)  # centroids as FAISS base
    d = xb.shape[1]
    index = _get_faiss_index(d, use_gpu=use_gpu, device=gpu_device)
    index.add(xb)

    N_total = n_total
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
        X_chunk_norm, _ = _normalize_embedding(X_chunk, modality, params=norm_params)

        D2, I = index.search(X_chunk_norm, k)     # (chunk, k)
        D2 = np.maximum(D2, 1e-6)

        # Inverse-distance weights (normalized)
        D = np.sqrt(D2)
        D = np.maximum(D, 1e-6)
        W = 1.0 / D
        W /= W.sum(axis=1, keepdims=True)

        disp_neighbors = disp_anchor[I]     # (chunk, k, d_embed)
        disp_chunk = (disp_neighbors * W[..., None]).sum(axis=1)  # (chunk, d_embed)

        X_ot_full[start:end] = X_chunk_norm + disp_chunk.astype(np.float32, copy=False)

    # 5) Write result back into the full AnnData.
    # NOTE: this will load X_ot_full into memory, but by this point we are done with
    # the expensive OT / FAISS steps; if that still exceeds RAM, you can instead
    # keep the memmap and avoid holding the whole array in memory at once.
    adata_full.obsm[out_key] = np.asarray(X_ot_full, dtype=np.float32)
    metrics = dict(metrics)
    metrics["n_centroids"] = int(X_anchor.shape[0])
    return adata_full, metrics
