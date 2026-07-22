from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
from sklearn.preprocessing import LabelEncoder

from ..ot._time import _time_to_numeric, _standardize_time, _bin_time_vector
from ..utils.ot_helpers import (
    _FAISS_AVAIL,
    _faiss_ready,
    _get_faiss_index,
    _local_knn_density,
    _minikm_centers,
)
from ..utils.ot_transport import compute_ot_alignment


def _resolve_lineage_names(adata_full: Any, lineage_key: str, n_lineages: int) -> list[str]:
    names = None
    if hasattr(adata_full, "uns") and isinstance(adata_full.uns, dict):
        candidate = adata_full.uns.get(f"{lineage_key}_names")
        if isinstance(candidate, (list, tuple)) and len(candidate) == n_lineages:
            names = [str(v) for v in candidate]
    if names is None:
        names = [f"lineage_{i}" for i in range(n_lineages)]
    return names


def _build_centroids(
    *,
    idx_group: np.ndarray,
    time_raw: np.ndarray,
    n_centroids_per_bin: int,
    max_samples_per_bin: int,
    rng: np.random.Generator,
    seed: int,
    fetch_features,
    weights_group: Optional[np.ndarray],
    density_weight: bool,
    density_k: int,
    use_gpu: bool,
    gpu_device: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    n_samples = int(len(idx_group))
    if n_samples == 0:
        empty = np.zeros((0,), dtype=np.int32)
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            empty,
            empty,
            None,
        )

    if max_samples_per_bin > 0 and n_samples > max_samples_per_bin:
        if weights_group is not None and np.isfinite(weights_group).any() and weights_group.sum() > 0:
            p = weights_group / float(weights_group.sum())
            sub_pos = rng.choice(n_samples, size=max_samples_per_bin, replace=False, p=p)
        else:
            sub_pos = rng.choice(n_samples, size=max_samples_per_bin, replace=False)
        idx_sub = idx_group[sub_pos]
        weights_sub = weights_group[sub_pos] if weights_group is not None else None
    else:
        idx_sub = idx_group
        weights_sub = weights_group

    X_sub = fetch_features(idx_sub)
    w_density = (
        _local_knn_density(X_sub, k=density_k, use_gpu=use_gpu, device=gpu_device)
        if density_weight
        else None
    )
    if weights_sub is None:
        weights = w_density
    else:
        weights = weights_sub.astype(np.float32, copy=False)
        if w_density is not None:
            weights = weights * w_density

    n_clusters = int(max(1, min(n_centroids_per_bin, len(idx_sub))))
    C, assign = _minikm_centers(
        X_sub,
        n_clusters=n_clusters,
        seed=seed,
        use_gpu=use_gpu,
        device=gpu_device,
        weights=weights,
        return_labels=True,
    )

    t_sub = time_raw[idx_sub]
    if not np.isfinite(t_sub).any():
        centroid_time = np.zeros(C.shape[0], dtype=np.float32)
    else:
        centroid_time = np.full(C.shape[0], float(np.nanmean(t_sub)), dtype=np.float32)
        for c in range(C.shape[0]):
            mask = assign == c
            if not np.any(mask):
                continue
            vals = t_sub[mask]
            if np.isfinite(vals).any():
                centroid_time[c] = float(np.nanmean(vals))
    return (
        C.astype(np.float32, copy=False),
        centroid_time,
        np.asarray(assign, dtype=np.int32),
        np.asarray(idx_sub, dtype=np.int32),
        weights_sub,
    )


def velocity_field_sb_centroids(
    adata_full: Any,
    obsm_key: str,
    time_key: str,
    lineage_key: str,
    spatial_key: Optional[str] = None,
    domain_key: Optional[str] = None,
    *,
    out_vel_key: str = "velocity_sb",
    out_vel_uns_key: str = "velocity_sb_meta",
    time_mode: str = "auto",
    time_bins: Optional[int] = 20,
    n_centroids_per_bin: int = 2048,
    max_samples_per_bin: int = 500_000,
    k_interp: int = 8,
    chunk_size: int = 500_000,
    density_weight: bool = True,
    density_k: int = 15,
    soft_lineage_min_weight: float = 1e-3,
    normalize_soft_lineage: bool = True,
    spatial_weight: float = 1.0,
    use_gpu: bool = True,
    gpu_device: int = 0,
    ot_backend: str = "torch",
    ot_reg: float = 0.05,
    ot_reg_m: float = 0.5,
    ot_cost_clip_q: Optional[float] = 0.90,
    ot_clip_big: float = 50.0,
    ot_iters: int = 1000,
    ot_tol: float = 1e-6,
    ot_topk: int = 64,
    ot_chunk_size: Optional[int] = 1024,
    random_state: int = 0,
    tmp_path: Optional[str] = None,
    tmp_weight_path: Optional[str] = None,
) -> Any:
    """
    Compute lineage-specific Schrödinger-bridge-style velocity fields using centroid OT.

    Centroids are built per (time_bin, lineage), aligned across adjacent time bins via
    entropic OT, and interpolated back to all cells with FAISS in chunks.
    """
    if time_key is None:
        raise ValueError("time_key is required for trajectory velocity fields.")
    if time_key not in adata_full.obs:
        raise KeyError(f"time_key '{time_key}' not found in adata_full.obs.")
    if obsm_key not in adata_full.obsm:
        raise KeyError(f"obsm_key '{obsm_key}' not found in adata_full.obsm.")
    if spatial_key is not None and spatial_key not in adata_full.obsm:
        raise KeyError(f"spatial_key '{spatial_key}' not found in adata_full.obsm.")
    if lineage_key not in adata_full.obs and lineage_key not in adata_full.obsm:
        raise KeyError(
            f"lineage_key '{lineage_key}' not found in adata_full.obs or adata_full.obsm."
        )

    n_obs = int(adata_full.n_obs)
    time_raw, time_meta = _time_to_numeric(adata_full.obs[time_key], time_mode)
    if time_bins is None and time_meta.get("mode") == "continuous":
        raise ValueError("time_bins must be set for continuous time values.")
    time_bin, time_bin_meta = _bin_time_vector(time_raw, time_bins, time_meta["mode"])
    time_std, time_mean, time_sd = _standardize_time(time_raw)

    if np.issubdtype(time_bin.dtype, np.floating):
        time_bin = time_bin.astype(np.int32, copy=False)
    valid_time = time_bin >= 0
    if not np.any(valid_time):
        raise ValueError("No valid time bins found; check time_key/time_bins.")
    time_levels = np.sort(np.unique(time_bin[valid_time])).astype(np.int32, copy=False)

    lineage_mode = "hard"
    lineage_names: list[str]
    lineage_ids: Optional[np.ndarray] = None
    lineage_mat: Optional[np.ndarray] = None
    if lineage_key in adata_full.obsm:
        lineage_mode = "soft"
        lineage_mat = np.asarray(adata_full.obsm[lineage_key], dtype=np.float32)
        if lineage_mat.ndim != 2 or lineage_mat.shape[0] != n_obs:
            raise ValueError("Soft lineage matrix must be 2D with shape (n_obs, n_lineages).")
        lineage_mat = np.nan_to_num(lineage_mat, nan=0.0, posinf=0.0, neginf=0.0)
        n_lineages = int(lineage_mat.shape[1])
        lineage_names = _resolve_lineage_names(adata_full, lineage_key, n_lineages)
    else:
        labels = np.asarray(adata_full.obs[lineage_key])
        le = LabelEncoder()
        lineage_ids = le.fit_transform(labels).astype(np.int32, copy=False)
        lineage_names = [str(v) for v in le.classes_]
        n_lineages = int(len(lineage_names))

    domain_ids: Optional[np.ndarray] = None
    domain_labels: Optional[list[str]] = None
    if domain_key is None:
        for candidate in ("spatial_cluster", "annotation_true", "annotation", lineage_key):
            if hasattr(adata_full, "obs") and candidate in adata_full.obs:
                domain_key = candidate
                break
    if domain_key is not None and hasattr(adata_full, "obs") and domain_key in adata_full.obs:
        domain_series = adata_full.obs[domain_key]
        if hasattr(domain_series, "cat"):
            domain_labels = [str(v) for v in domain_series.cat.categories]
            domain_ids = domain_series.cat.codes.to_numpy()
            domain_ids = domain_ids.astype(np.int32, copy=False)
        else:
            domain_vals = np.asarray(domain_series, dtype=object)
            le_domain = LabelEncoder()
            domain_ids = le_domain.fit_transform(domain_vals).astype(np.int32, copy=False)
            domain_labels = [str(v) for v in le_domain.classes_]

    first_embed = np.asarray(adata_full.obsm[obsm_key][0:1], dtype=np.float32, order="C")
    d_embed = int(first_embed.shape[1])
    if spatial_key is not None:
        first_spatial = np.asarray(adata_full.obsm[spatial_key][0:1], dtype=np.float32, order="C")
        d_spatial = int(first_spatial.shape[1])
    else:
        d_spatial = 0
    d_feat = d_embed + d_spatial
    d_out = d_embed
    spatial_weight = float(spatial_weight)

    def _fetch_features(indices: np.ndarray) -> np.ndarray:
        X_embed = np.asarray(adata_full.obsm[obsm_key][indices], dtype=np.float32, order="C")
        if spatial_key is None:
            return X_embed
        X_spatial = np.asarray(adata_full.obsm[spatial_key][indices], dtype=np.float32, order="C")
        if spatial_weight != 1.0:
            X_spatial = X_spatial * spatial_weight
        return np.concatenate([X_embed, X_spatial], axis=1)

    rng = np.random.default_rng(int(random_state))

    centroids: Dict[Tuple[int, int], np.ndarray] = {}
    centroid_times: Dict[Tuple[int, int], np.ndarray] = {}
    centroid_counts: Dict[Tuple[int, int], int] = {}
    centroid_domains: Dict[Tuple[int, int], np.ndarray] = {}
    centroid_weights: Dict[Tuple[int, int], np.ndarray] = {}

    def _summarize_centroid_domains(
        *,
        key: Tuple[int, int],
        idx_group: np.ndarray,
        idx_sub: np.ndarray,
        assign: np.ndarray,
        weights_sub: Optional[np.ndarray],
        weights_group: Optional[np.ndarray],
        n_clusters: int,
    ) -> None:
        if domain_ids is None or domain_labels is None:
            return
        if n_clusters == 0:
            return
        domain_ids_sub = domain_ids[idx_sub]
        n_domains = int(len(domain_labels))
        counts = np.zeros((n_clusters, n_domains), dtype=np.float64)
        if weights_sub is None:
            weights_local = np.ones(len(domain_ids_sub), dtype=np.float32)
            sub_weight = float(len(domain_ids_sub))
            group_weight = float(len(idx_group))
        else:
            weights_local = np.asarray(weights_sub, dtype=np.float32)
            sub_weight = float(np.sum(weights_local))
            group_weight = float(np.sum(weights_group)) if weights_group is not None else float(len(idx_group))
        for row, dom in enumerate(domain_ids_sub):
            if dom < 0:
                continue
            counts[int(assign[row]), int(dom)] += float(weights_local[row])
        row_sums = counts.sum(axis=1)
        scale = group_weight / sub_weight if sub_weight > 0 else 1.0
        row_sums = row_sums * scale
        domain_idx = np.full(n_clusters, -1, dtype=np.int32)
        valid = row_sums > 0
        if np.any(valid):
            domain_idx[valid] = np.argmax(counts[valid], axis=1).astype(np.int32, copy=False)
        centroid_domains[key] = domain_idx
        centroid_weights[key] = row_sums.astype(np.float32, copy=False)

    for t in time_levels:
        idx_time = np.where(time_bin == t)[0]
        if len(idx_time) == 0:
            continue
        if lineage_mode == "hard":
            lineage_ids_time = lineage_ids[idx_time]
            order = np.argsort(lineage_ids_time, kind="mergesort")
            sorted_idx = idx_time[order]
            sorted_lineage = lineage_ids_time[order]
            boundaries = np.flatnonzero(np.diff(sorted_lineage)) + 1
            starts = np.concatenate(([0], boundaries))
            ends = np.concatenate((boundaries, [len(sorted_lineage)]))
            for start, end in zip(starts, ends):
                lineage_idx = int(sorted_lineage[start])
                idx_group = sorted_idx[start:end]
                C, t_cent, assign, idx_sub, weights_sub = _build_centroids(
                    idx_group=idx_group,
                    time_raw=time_raw,
                    n_centroids_per_bin=n_centroids_per_bin,
                    max_samples_per_bin=max_samples_per_bin,
                    rng=rng,
                    seed=int(random_state + lineage_idx * 97 + int(t)),
                    fetch_features=_fetch_features,
                    weights_group=None,
                    density_weight=density_weight,
                    density_k=density_k,
                    use_gpu=use_gpu,
                    gpu_device=gpu_device,
                )
                if C.size == 0:
                    continue
                key = (lineage_idx, int(t))
                centroids[key] = C
                centroid_times[key] = t_cent
                centroid_counts[key] = int(C.shape[0])
                _summarize_centroid_domains(
                    key=key,
                    idx_group=idx_group,
                    idx_sub=idx_sub,
                    assign=assign,
                    weights_sub=weights_sub,
                    weights_group=None,
                    n_clusters=int(C.shape[0]),
                )
        else:
            for lineage_idx in range(n_lineages):
                weights_time = lineage_mat[idx_time, lineage_idx]
                mask = weights_time > float(soft_lineage_min_weight)
                if not np.any(mask):
                    continue
                idx_group = idx_time[mask]
                weights_group = weights_time[mask]
                C, t_cent, assign, idx_sub, weights_sub = _build_centroids(
                    idx_group=idx_group,
                    time_raw=time_raw,
                    n_centroids_per_bin=n_centroids_per_bin,
                    max_samples_per_bin=max_samples_per_bin,
                    rng=rng,
                    seed=int(random_state + lineage_idx * 97 + int(t)),
                    fetch_features=_fetch_features,
                    weights_group=weights_group,
                    density_weight=density_weight,
                    density_k=density_k,
                    use_gpu=use_gpu,
                    gpu_device=gpu_device,
                )
                if C.size == 0:
                    continue
                key = (lineage_idx, int(t))
                centroids[key] = C
                centroid_times[key] = t_cent
                centroid_counts[key] = int(C.shape[0])
                _summarize_centroid_domains(
                    key=key,
                    idx_group=idx_group,
                    idx_sub=idx_sub,
                    assign=assign,
                    weights_sub=weights_sub,
                    weights_group=weights_group,
                    n_clusters=int(C.shape[0]),
                )

    if not centroids:
        raise ValueError("No centroids were constructed; check lineage_key/time_key settings.")

    velocities: Dict[Tuple[int, int], np.ndarray] = {}
    alignment_counts: Dict[Tuple[int, int], Tuple[int, int]] = {}
    flux_mats: Dict[Tuple[object, object], np.ndarray] = {}
    lineage_flux: Dict[str, Dict[Tuple[object, object], np.ndarray]] = {}

    time_values = np.asarray(adata_full.obs[time_key], dtype=object)
    time_bin_labels: Dict[int, object] = {}
    for t in time_levels:
        mask = time_bin == t
        if not np.any(mask):
            continue
        vals = time_values[mask]
        if len(vals) == 0:
            continue
        unique_vals, counts = np.unique(vals, return_counts=True)
        time_bin_labels[int(t)] = unique_vals[int(np.argmax(counts))]

    def _time_pair_key(t_src: int, t_ref: int) -> Tuple[object, object]:
        return (time_bin_labels.get(int(t_src), int(t_src)), time_bin_labels.get(int(t_ref), int(t_ref)))

    def _transport_flux(
        transport: Dict[str, np.ndarray],
        src_domains: np.ndarray,
        tgt_domains: np.ndarray,
        src_weights: Optional[np.ndarray],
        n_domains: int,
    ) -> np.ndarray:
        idx = np.asarray(transport.get("indices"), dtype=np.int32)
        weights = np.asarray(transport.get("weights"), dtype=np.float32)
        residual = transport.get("residual")
        if idx.ndim != 2 or weights.ndim != 2 or idx.shape != weights.shape:
            raise ValueError("Transport indices/weights shape mismatch.")
        flux = np.zeros((n_domains, n_domains), dtype=np.float32)
        for i in range(len(src_domains)):
            src_dom = int(src_domains[i])
            if src_dom < 0 or src_dom >= n_domains:
                continue
            mass = float(src_weights[i]) if src_weights is not None else 1.0
            if residual is not None and i < len(residual):
                mass *= max(0.0, 1.0 - float(residual[i]))
            if mass <= 0:
                continue
            tgt_idx = idx[i]
            tgt_w = weights[i]
            for j, weight in zip(tgt_idx, tgt_w):
                j_int = int(j)
                if j_int < 0 or j_int >= len(tgt_domains):
                    continue
                tgt_dom = int(tgt_domains[j_int])
                if tgt_dom < 0 or tgt_dom >= n_domains:
                    continue
                flux[src_dom, tgt_dom] += mass * float(weight)
        return flux
    for lineage_idx in range(n_lineages):
        for i in range(len(time_levels) - 1):
            t_src = int(time_levels[i])
            t_ref = int(time_levels[i + 1])
            key_src = (lineage_idx, t_src)
            key_ref = (lineage_idx, t_ref)
            if key_src not in centroids or key_ref not in centroids:
                continue
            C_src = centroids[key_src]
            C_ref = centroids[key_ref]
            aligned, transport = compute_ot_alignment(
                C_src,
                C_ref,
                reg=float(ot_reg),
                reg_m=float(ot_reg_m),
                cost_clip_q=ot_cost_clip_q,
                clip_big=float(ot_clip_big),
                backend=ot_backend,
                iters=int(ot_iters),
                tol=float(ot_tol),
                use_gpu=use_gpu,
                gpu_device=gpu_device,
                transport_topk=int(ot_topk),
                chunk_size=ot_chunk_size,
            )
            vel = aligned - C_src
            velocities[key_src] = vel[:, :d_out].astype(np.float32, copy=False)
            alignment_counts[key_src] = (int(C_src.shape[0]), int(C_ref.shape[0]))
            if domain_labels is not None:
                src_domains = centroid_domains.get(key_src)
                tgt_domains = centroid_domains.get(key_ref)
                src_weights = centroid_weights.get(key_src)
                if src_domains is not None and tgt_domains is not None:
                    time_pair = _time_pair_key(t_src, t_ref)
                    flux = _transport_flux(
                        transport,
                        src_domains,
                        tgt_domains,
                        src_weights,
                        n_domains=len(domain_labels),
                    )
                    if time_pair not in flux_mats:
                        flux_mats[time_pair] = flux
                    else:
                        flux_mats[time_pair] += flux
                    lineage_name = lineage_names[lineage_idx]
                    lineage_flux.setdefault(lineage_name, {})[time_pair] = flux

    for key, C in centroids.items():
        if key not in velocities:
            velocities[key] = np.zeros((C.shape[0], d_out), dtype=np.float32)

    if not _FAISS_AVAIL:
        raise RuntimeError(
            "FAISS is required for centroid-level interpolation. "
            "Install it (e.g. `pip install faiss-gpu` or `faiss-cpu`)."
        )

    if tmp_path is not None:
        vel_full = np.memmap(tmp_path, mode="w+", dtype=np.float32, shape=(n_obs, d_out))
        vel_full[:] = 0.0
    else:
        vel_full = np.zeros((n_obs, d_out), dtype=np.float32)

    if lineage_mode == "soft":
        if tmp_weight_path is not None:
            weight_sum = np.memmap(tmp_weight_path, mode="w+", dtype=np.float32, shape=(n_obs,))
            weight_sum[:] = 0.0
        else:
            weight_sum = np.zeros((n_obs,), dtype=np.float32)

    def _interpolate_group(idx_group: np.ndarray, C: np.ndarray, V: np.ndarray) -> np.ndarray:
        k = int(max(1, min(k_interp, C.shape[0])))
        xb = _faiss_ready(C)
        index = _get_faiss_index(xb.shape[1], use_gpu=use_gpu, device=gpu_device)
        index.add(xb)
        out = np.zeros((len(idx_group), d_out), dtype=np.float32)
        for start in range(0, len(idx_group), chunk_size):
            end = min(start + chunk_size, len(idx_group))
            idx_chunk = idx_group[start:end]
            X_chunk = _fetch_features(idx_chunk)
            D2, I = index.search(_faiss_ready(X_chunk), k)
            D2 = np.maximum(D2, 1e-6)
            D = np.sqrt(D2)
            D = np.maximum(D, 1e-6)
            W = 1.0 / D
            W /= W.sum(axis=1, keepdims=True)
            V_neighbors = V[I]
            out[start:end] = (V_neighbors * W[..., None]).sum(axis=1)
        return out

    for t in time_levels:
        idx_time = np.where(time_bin == t)[0]
        if len(idx_time) == 0:
            continue
        if lineage_mode == "hard":
            lineage_ids_time = lineage_ids[idx_time]
            order = np.argsort(lineage_ids_time, kind="mergesort")
            sorted_idx = idx_time[order]
            sorted_lineage = lineage_ids_time[order]
            boundaries = np.flatnonzero(np.diff(sorted_lineage)) + 1
            starts = np.concatenate(([0], boundaries))
            ends = np.concatenate((boundaries, [len(sorted_lineage)]))
            for start, end in zip(starts, ends):
                lineage_idx = int(sorted_lineage[start])
                idx_group = sorted_idx[start:end]
                key = (lineage_idx, int(t))
                C = centroids.get(key)
                V = velocities.get(key)
                if C is None or V is None:
                    continue
                vel_group = _interpolate_group(idx_group, C, V)
                vel_full[idx_group] = vel_group
        else:
            for lineage_idx in range(n_lineages):
                weights_time = lineage_mat[idx_time, lineage_idx]
                mask = weights_time > float(soft_lineage_min_weight)
                if not np.any(mask):
                    continue
                idx_group = idx_time[mask]
                key = (lineage_idx, int(t))
                C = centroids.get(key)
                V = velocities.get(key)
                if C is None or V is None:
                    continue
                vel_group = _interpolate_group(idx_group, C, V)
                w_cells = weights_time[mask].astype(np.float32, copy=False)
                vel_full[idx_group] += vel_group * w_cells[:, None]
                weight_sum[idx_group] += w_cells

    if lineage_mode == "soft" and normalize_soft_lineage:
        w = np.asarray(weight_sum, dtype=np.float32)
        vel_full = np.divide(vel_full, w[:, None], out=vel_full, where=w[:, None] > 0)

    adata_full.obsm[out_vel_key] = np.asarray(vel_full, dtype=np.float32)

    if "lineage_probs" not in adata_full.obsm:
        if lineage_mode == "soft":
            lineage_probs = np.asarray(lineage_mat, dtype=np.float32)
            lineage_probs = np.clip(lineage_probs, 0.0, None)
            row_sum = lineage_probs.sum(axis=1, keepdims=True)
            if normalize_soft_lineage:
                lineage_probs = np.divide(
                    lineage_probs,
                    row_sum,
                    out=lineage_probs,
                    where=row_sum > 0,
                )
        else:
            lineage_probs = np.zeros((n_obs, n_lineages), dtype=np.float32)
            lineage_probs[np.arange(n_obs), lineage_ids] = 1.0
        adata_full.obsm["lineage_probs"] = lineage_probs
    if hasattr(adata_full, "uns"):
        if "lineage_probs_labels" not in adata_full.uns:
            adata_full.uns["lineage_probs_labels"] = lineage_names

    centroid_counts_out: Dict[str, Dict[str, int]] = {}
    for lineage_idx, name in enumerate(lineage_names):
        lineage_map: Dict[str, int] = {}
        for t in time_levels:
            key = (lineage_idx, int(t))
            if key in centroid_counts:
                lineage_map[str(int(t))] = int(centroid_counts[key])
        centroid_counts_out[name] = lineage_map

    align_counts_out: Dict[str, Dict[str, Tuple[int, int]]] = {}
    for lineage_idx, name in enumerate(lineage_names):
        lineage_map = {}
        for t in time_levels[:-1]:
            key = (lineage_idx, int(t))
            if key in alignment_counts:
                lineage_map[str(int(t))] = alignment_counts[key]
        align_counts_out[name] = lineage_map

    meta: Dict[str, Any] = {
        "obsm_key": obsm_key,
        "spatial_key": spatial_key,
        "time_key": time_key,
        "lineage_key": lineage_key,
        "domain_key": domain_key,
        "lineage_mode": lineage_mode,
        "lineage_names": lineage_names,
        "time_levels": [int(v) for v in time_levels.tolist()],
        "time_meta": time_meta,
        "time_bin_meta": time_bin_meta,
        "time_mean": float(time_mean),
        "time_std": float(time_sd),
        "feature_dim": int(d_feat),
        "velocity_dim": int(d_out),
        "centroid_counts": centroid_counts_out,
        "alignment_counts": align_counts_out,
        "params": {
            "time_bins": time_bins,
            "n_centroids_per_bin": int(n_centroids_per_bin),
            "max_samples_per_bin": int(max_samples_per_bin),
            "k_interp": int(k_interp),
            "chunk_size": int(chunk_size),
            "density_weight": bool(density_weight),
            "density_k": int(density_k),
            "soft_lineage_min_weight": float(soft_lineage_min_weight),
            "normalize_soft_lineage": bool(normalize_soft_lineage),
            "spatial_weight": float(spatial_weight),
            "ot_backend": ot_backend,
            "ot_reg": float(ot_reg),
            "ot_reg_m": float(ot_reg_m),
            "ot_cost_clip_q": ot_cost_clip_q,
            "ot_clip_big": float(ot_clip_big),
            "ot_iters": int(ot_iters),
            "ot_tol": float(ot_tol),
            "ot_topk": int(ot_topk),
            "ot_chunk_size": ot_chunk_size,
        },
    }
    adata_full.uns[out_vel_uns_key] = meta

    if domain_labels is not None and flux_mats and hasattr(adata_full, "uns"):
        flux_store: Dict[str, Any] = {
            "flux": flux_mats,
            "domain_labels": domain_labels,
            "domain_key": domain_key,
            "time_pairs": list(flux_mats.keys()),
        }
        if lineage_flux:
            flux_store["lineage_flux"] = lineage_flux
        adata_full.uns["_sb_flux"] = flux_store
    return adata_full
