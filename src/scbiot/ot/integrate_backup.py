# =================== integrate.py: unified pipeline (rare-protection + supervised option) ===================
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Sequence, Literal

import numpy as np
from sklearn.preprocessing import LabelEncoder

from ..utils.coral_helpers import _coral_prealign
from ..utils.ot_helpers import (
    _FAISS_AVAIL,
    _FAISS_GPU,
    _as_nd_f32_c,
    _cap_step_local,
    _class_means,
    _cluster_sharpen_field,
    _compute_prototypes,
    _compute_prototypes_union,
    _graph_strain,
    _guard_edge_stretch_weighted,
    _knn_graph,
    _knn_overlap,
    _lerp,
    _neighbor_batch_entropy_per_cell,
    _nearest_other_class_index,
    _ot_barycentric_gpu,
    _ptp,
    _smooth_by_knn,
    _trustworthiness_score,
)
from ..utils.ot_transport import compute_ot_alignment

# -------------------- Main integration (now supervised-ready) --------------------


def integrate_ot(
    adata: Any,
    obsm_key: str = "X_pca",
    batch_key: str = "batch",
    out_key: str = "scBIOT",
    modality: Literal["rna", "atac", "supervised", "anchor"] | None = None,        
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
    reference: str = "union",   # supports "largest" or "union"
    label_key: Optional[str] = None,
    unlabeled_category: Any = "unknown",
    postscale: bool = True,
    random_state: int = 0,
    verbose: bool = True,
    use_gpu: bool = True,
    gpu_device: int = 0,
    ot_backend: str = "torch",   
    ot_mode: str = "unbalanced",  
    spatial_key: Optional[str] = None,
    spatial_weight: float = 0.5,    
    # ---- NEW: CORAL pre-alignment for disjoint datasets ----
    prealign: Optional[str] = None,        # None | "coral"
    coral_strength: float = 1.0,   # 1.0 full CORAL, 0.3~0.7 partial
    coral_eps: float = 1e-3,
    coral_max_points: int = 20000,
    coral_target: str = "auto",    # "auto"|"reference"|"global"
    # ---- NEW supervised knobs ----
    lam_sup: float = 0.60,                 # pull to own-class mean (0 disables)
    lam_repulse: float = 0.18,             # repel from nearest other-class mean (0 disables)
) -> Tuple[Any, Dict[str, float | int]]:
    """
    **scBIOT**: optimal transport–based data integration for single-modality and cross-modality embeddings.

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
    modality
        Default settings used for integration of scRNA-seq ("RNA"), supervised ("supervised"), snATAC-seq ("atac"), multiomics ("anchor")
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
        Reference batch selection strategy (``"largest"`` or ``"union"``). When running
        in supervised mode with ``label_key`` provided, the method forces ``"union"``
        for stability.
    label_key / unlabeled_category
        Optional semi-supervised label column and unlabeled marker used for supervised
        guidance and to compute OT transport metadata for label transfer. If omitted,
        label-transfer metadata is skipped.
    postscale
        Whether to rescale the aligned embedding to unit variance per dimension.
    random_state
        Seed for stochastic steps used in subsampling and initialization.
    verbose
        Print progress information when ``True``.
    use_gpu / gpu_device / ot_backend / ot_mode
        Compute backend selection for OT (torch by default, pot optional) and whether
        to run unbalanced OT (default) or balanced OT for stronger batch mixing. When
        ``ot_mode='balanced'``, the algorithm automatically switches to the ``'union'``
        reference so every batch moves symmetrically.
    spatial_key
        Optional key in ``adata.obsm`` containing spatial coordinates. When provided,
        those coordinates are concatenated to ``obsm_key`` before OT.
    spatial_weight
        Multiplier applied to the standardized spatial coordinates prior to
        concatenation so that spatial distances and expression distances have
        comparable scale.
    lam_sup / lam_repulse
        Attraction/repulsion strengths when ``label_key`` is provided.

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
    mode_norm = str(ot_mode).lower()
    if mode_norm not in {"balanced", "unbalanced"}:
        raise ValueError("ot_mode must be 'balanced' or 'unbalanced'")

    X0 = _as_nd_f32_c(adata.obsm[obsm_key])
    if spatial_key is not None:
        if spatial_key not in adata.obsm:
            raise KeyError(f"Spatial key '{spatial_key}' not found in adata.obsm.")
        coords = _as_nd_f32_c(adata.obsm[spatial_key])
        if coords.shape[0] != X0.shape[0]:
            raise ValueError(
                f"Spatial coordinates '{spatial_key}' have {coords.shape[0]} rows; expected {X0.shape[0]}."
            )
        coords = coords - coords.mean(0, keepdims=True)
        coords = coords / (coords.std(0, keepdims=True) + 1e-6)
        X0 = np.hstack([X0, coords * float(spatial_weight)])
    b_raw = adata.obs[batch_key].to_numpy()
    le_batch = LabelEncoder()
    b = le_batch.fit_transform(b_raw).astype(np.int32, copy=False)
    n_classes = max(1, int(b.max()) + 1)

    # optional supervised labels
    y: Optional[np.ndarray]
    has_label_key = label_key is not None and label_key in adata.obs
    n_labels = 0
    base_unknown_aliases = ("unknown", "Unknown", "unlabeled", "unlabelled", "UNK", "NA", "NaN", "None", "")
    extra_aliases: Tuple[str, ...] = ()
    if unlabeled_category is not None:
        if isinstance(unlabeled_category, (list, tuple, set)):
            extra_aliases = tuple(str(v) for v in unlabeled_category)
        else:
            extra_aliases = (str(unlabeled_category),)
    unknown_aliases = base_unknown_aliases + extra_aliases
    if has_label_key:
        labels = adata.obs[label_key]
        labels_norm = labels.astype(str).str.strip().str.lower()
        unk_set = set(a.lower() for a in unknown_aliases)
        unk_mask = labels.isna() | labels_norm.isin(unk_set)
        labels_norm = labels_norm.mask(unk_mask, "unknown")

        known = ~labels_norm.eq("unknown")
        y = np.full(len(labels_norm), -1, dtype=int)
        n_labels = 0
        if known.any():
            le_labels = LabelEncoder().fit(labels_norm[known])
            y[known] = le_labels.transform(labels_norm[known])
            n_labels = int(y.max()) + 1
    else:
        y, n_labels = None, 0

    reference_norm = str(reference).lower()
    force_union = (
        (mode_norm == "balanced" and reference_norm == "largest")
        or (mode_norm == "supervised" and has_label_key and reference_norm != "union")
    )
    ref_mode = "union" if (reference_norm == "union" or force_union) else reference_norm

    if ref_mode == "largest":
        vals, counts = np.unique(b_raw, return_counts=True)
        ref_label_raw = vals[np.argmax(counts)]
        ref_label_enc = int(np.where(le_batch.classes_ == ref_label_raw)[0][0])
    elif ref_mode == "union":
        ref_label_enc = 0
    else:
        # if integer was passed, use it; else fallback to 0
        try:
            ref_label_enc = int(reference)
        except Exception:
            ref_label_enc = int(0)

        # --- CORAL pre-align (NEW): force overlap for disjoint batches ---
    d_embed = _as_nd_f32_c(adata.obsm[obsm_key]).shape[1]  # original embedding dims (before spatial concat)
    apply_dims = slice(0, d_embed) if spatial_key is not None else None

    prealign_norm = "none" if prealign is None else str(prealign).lower()
    if prealign_norm == "coral":
        X0 = _coral_prealign(
            X0,
            b,
            ref_label_enc=ref_label_enc,
            ref_mode=ref_mode,
            strength=coral_strength,
            eps=coral_eps,
            max_points=coral_max_points,
            seed=random_state + 991,
            target=coral_target,
            apply_dims=apply_dims,
        )
        if verbose:
            print(f"[prealign] CORAL enabled target={coral_target} strength={coral_strength}")


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

    if force_union and verbose:
        if mode_norm == "balanced" and reference_norm == "largest":
            print("[ot] balanced mode -> using reference='union' for symmetric mixing")
        elif mode_norm == "supervised" and has_label_key and reference_norm != "union":
            print("[ot] supervised labels -> using reference='union' for stability")

    no_imp = 0
    for it in range(1, max_iter + 1):
        t = (it - 1) / max(1, max_iter - 1)
        lambda_graph0 = _lerp(lambda0_hi, lambda0_lo, t)
        overlap0_floor = _lerp(overlap0_lo, overlap0_hi, t)
        step = _lerp(step_lo, step_hi, t)
        cost_clip_q = _lerp(q_start, q_end, t)

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
                ot_mode=ot_mode,
                iters=1000,
                tol=1e-6,
                use_gpu=use_gpu,
                gpu_device=gpu_device,
            )
            disp_proto = Bi_to_R - Bi
            norm_move = np.linalg.norm(disp_proto, axis=1)
            s_dist = 1.0 / (1.0 + (norm_move / (norm_move.std() + 1e-8)))
            bridge_damp = 0.20 if mode_norm == "balanced" else 0.35
            alpha_i = s_dist[nn_idx] * (1.0 - bridge_damp * bridge_score[idx])
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
    if label_key is None or label_key not in adata.obs:
        if verbose:
            print("[label transfer] skipped; pass label_key to compute alignment metadata")
        return adata, dict(mix=best["mix"], overlap0=best["overlap0"], strain=best["strain"], tw=tw, it=best["it"])
    labels = adata.obs[label_key]
    if unlabeled_category is None:
        query_mask = labels.isna()
    elif isinstance(unlabeled_category, (list, tuple, set)):
        query_mask = labels.isna() | labels.isin(unlabeled_category)
    else:
        query_mask = labels.isna() | labels.eq(unlabeled_category)
    ref_mask = ~query_mask

    ref_mask = np.asarray(ref_mask)
    query_mask = np.asarray(query_mask)
    rep = np.asarray(adata.obsm[out_key], dtype=np.float32)
    ref_emb = rep[ref_mask]
    query_emb = rep[query_mask]
    if ref_emb.size == 0 or query_emb.size == 0:
        raise ValueError("Reference/query subset empty; check label_key/unlabeled_category.")
    if ref_emb.shape[1] != query_emb.shape[1]:
        raise ValueError("Reference/query embedding dimensionality mismatch.")

    align_topk = int(min(64, max(1, ref_emb.shape[0])))
    align_chunk = 1024
    _, transport = compute_ot_alignment(
        query_emb,
        ref_emb,
        reg=reg,
        reg_m=reg_m,
        cost_clip_q=float(q_end),
        clip_big=50.0,
        backend=ot_backend,
        use_gpu=use_gpu,
        gpu_device=gpu_device,
        transport_topk=align_topk,
        chunk_size=align_chunk,
    )
    ot_meta: Dict[str, Any] = {
        "indices": transport["indices"].astype(np.int32, copy=False),
        "weights": transport["weights"].astype(np.float32, copy=False),
        "residual": (
            transport.get("residual").astype(np.float32, copy=False)
            if transport.get("residual") is not None
            else None
        ),
        "rna_obs": np.asarray(adata.obs_names[ref_mask], dtype=object),
        "params": {
            "reg": float(reg),
            "reg_m": float(reg_m),
            "cost_clip_q": float(q_end),
            "clip_big": float(50.0),
            "backend": ot_backend,
            "topk": align_topk,
        },
        "center": ref_emb.mean(axis=0, keepdims=True).astype(np.float32, copy=False),
        "scale": ref_emb.std(axis=0, keepdims=True).astype(np.float32, copy=False),
    }
    adata.uns["_ot_alignment"] = ot_meta
    adata.uns["_supbiot"] = {
        "batch_key": batch_key,
        "rep_key": out_key,
        "modality": modality,
        "label_key": label_key,
        "unlabeled_category": unlabeled_category,
    }
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
#     label_key="semi_cell_type",
#     unlabeled_category="unknown",
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
