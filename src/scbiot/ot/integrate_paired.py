from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

import ot  # POT used only if you set ot_backend="ot"

from .integrate import (
    _FAISS_AVAIL,
    _FAISS_GPU,
    _as_nd_f32_c,
    _cap_step_local,
    _cluster_sharpen_field,
    _faiss_knn_search,
    _graph_strain,
    _guard_edge_stretch_weighted,
    _knn_graph,
    _knn_overlap,
    _lerp,
    _minikm_centers,
    _neighbor_batch_entropy_per_cell,
    _ot_barycentric_gpu,
    _ptp,
    _smooth_by_knn,
    _sinkhorn_uot_torch,
    _to_torch,
    _torch_device,
    _trustworthiness_score,
)

# -------------------- Utilities --------------------

def _row_unit_norm(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

def _zscore(X: np.ndarray) -> np.ndarray:
    X = _as_nd_f32_c(X)
    return (X - X.mean(0)) / (X.std(0) + 1e-8)

def _prep_view(name: str, X: np.ndarray) -> np.ndarray:
    n = name.lower()
    if "lsi" in n or n.endswith("lsi"):
        return _row_unit_norm(_as_nd_f32_c(X))  # cosine geometry
    return _zscore(_as_nd_f32_c(X))            # PCs: euclidean

def _views_from_keys(adata, view_keys: Tuple[str, ...]) -> Dict[str, np.ndarray]:
    return {k: _prep_view(k, adata.obsm[k]) for k in view_keys}


def _uot_from_cost(
    M: np.ndarray,
    reg: float,
    reg_m: float,
    backend: str,
    use_gpu: bool,
    gpu_device: int,
) -> np.ndarray:
    """
    Return the *coupling matrix* T (numpy) for entropic unbalanced OT on cost M.
    Uniform marginals.
    """
    backend = backend.lower()
    if backend == "pot":
        backend = "ot"
    N, K = M.shape
    a = np.full(N, 1.0 / max(N, 1), dtype=np.float64)
    b = np.full(K, 1.0 / max(K, 1), dtype=np.float64)

    if backend == "ot":
        M64 = np.asarray(M, dtype=np.float64, order="C")
        try:
            T = ot.unbalanced.sinkhorn_unbalanced(
                a, b, M64, reg, reg_m, method="sinkhorn_stabilized",
                numItermax=1000, stopThr=1e-6, verbose=False
            )
        except TypeError:
            T = ot.unbalanced.sinkhorn_unbalanced(
                a, b, M64, reg, reg_m, numItermax=1000, stopThr=1e-6
            )
        return np.asarray(T, dtype=np.float32, order="C")

    # torch path
    device = _torch_device(use_gpu, gpu_device)
    dtype = torch.float32
    a_t = torch.full((N,), 1.0 / max(N, 1), dtype=dtype, device=device)
    b_t = torch.full((K,), 1.0 / max(K, 1), dtype=dtype, device=device)
    M_t = _to_torch(M.astype(np.float32, copy=False), device=device, dtype=dtype)
    u, v, Kt = _sinkhorn_uot_torch(M_t, a_t, b_t, eps=reg, tau=float(reg_m / (reg_m + reg)))
    T_t = (u.view(-1, 1) * Kt) * v.view(1, -1)
    return T_t.detach().cpu().numpy().astype(np.float32, copy=False)

# ---- TWO-VIEW prototypes for ALL batches (for U-FGW-BC) ----
def _compute_batch_prototypes_per_view(
    X_views: Dict[str, np.ndarray],
    b: np.ndarray,
    K_ref: int,
    K_batch: int,
    seed: int = 0,
    base_view: str = "X_pca",
    use_gpu: bool = True,
    device: int = 0,
):
    """
    Returns:
      - batches: sorted unique labels
      - proto_by_batch[v][lbl] -> array [Kb, dv]
      - assign_by_batch[lbl] -> (idx_cells, nn_idx_to_proto) based on base_view
    """
    batches = np.unique(b)
    proto_by_batch: Dict[str, Dict[int, np.ndarray]] = {vk: {} for vk in X_views.keys()}
    assign_by_batch: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    # choose per-batch K
    counts = {lbl: int(np.sum(b == lbl)) for lbl in batches}
    ref_label = max(counts, key=counts.get)
    for lbl in batches:
        idx = np.where(b == lbl)[0]
        Kb_eff = int(min(K_batch if lbl != ref_label else K_ref, max(8, 2 * np.sqrt(len(idx)))))
        # per-view prototypes
        for vk, Xv in X_views.items():
            proto_by_batch[vk][lbl] = _minikm_centers(Xv[idx], Kb_eff, seed + 7 + int(lbl), use_gpu=use_gpu, device=device)
        # base-view assignment
        base = X_views[base_view][idx]
        if _FAISS_AVAIL:
            _, nn_idx = _faiss_knn_search(base, proto_by_batch[base_view][lbl], 1, use_gpu=use_gpu, device=device)
            nn_idx = nn_idx.ravel()
        else:
            nn_idx = pairwise_distances_argmin(base, proto_by_batch[base_view][lbl], metric="euclidean")
        assign_by_batch[lbl] = (idx, nn_idx)
    return batches, proto_by_batch, assign_by_batch

# ------------- Lightweight structural signatures & fused costs (FGW-lite) -------------
def _struct_signature(X: np.ndarray, k: int = 10) -> np.ndarray:
    """
    For each row, take distances to k nearest other rows (in the same space),
    normalize to unit L2. This is a cheap local topology descriptor.
    """
    n = len(X)
    if n <= 1:
        return np.zeros((n, k), dtype=np.float32)
    k = int(max(1, min(k, n - 1)))
    if _FAISS_AVAIL:
        D2, I = _faiss_knn_search(X, X, k + 1, use_gpu=False, device=0)
        d = np.sqrt(np.maximum(D2[:, 1:], 0.0))
    else:
        nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X)
        d, I = nn.kneighbors(X)
        d = d[:, 1:]
    s = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
    return s.astype(np.float32, copy=False)

def _struct_cost_from_signatures(SB: np.ndarray, SR: np.ndarray) -> np.ndarray:
    # Pairwise squared L2 via expansion: ||a-b||^2 = ||a||^2 + ||b||^2 - 2a·b
    SB2 = (SB * SB).sum(1, keepdims=True)     # [Nb,1]
    SR2 = (SR * SR).sum(1, keepdims=True).T   # [1, Nr]
    C = SB2 + SR2 - 2.0 * (SB @ SR.T)
    return np.maximum(C, 0.0).astype(np.float32, copy=False)

def _feature_cost_two_view(
    B1: np.ndarray,
    R1: np.ndarray,
    B2: np.ndarray,
    R2: np.ndarray,
    w1: float = 0.5,
    *,
    view_names: Tuple[str, str] = ("view1", "view2"),
) -> np.ndarray:
    """
    Fused feature cost for two prototype sets. The first view uses squared
    Euclidean; the second uses cosine if the name suggests LSI, otherwise
    squared Euclidean as well.
    """
    v1, v2 = view_names
    # View 1: Euclidean^2
    D1 = ot.dist(B1.astype(np.float64, copy=False), R1.astype(np.float64, copy=False), metric="sqeuclidean").astype(np.float32)
    # View 2: cosine for LSI-like inputs, else Euclidean^2
    if "lsi" in v2.lower():
        B2n = _row_unit_norm(B2)
        R2n = _row_unit_norm(R2)
        D2 = (1.0 - np.clip(B2n @ R2n.T, -1.0, 1.0)) * 2.0
    else:
        D2 = ot.dist(B2.astype(np.float64, copy=False), R2.astype(np.float64, copy=False), metric="sqeuclidean").astype(np.float32)

    # normalize each to comparable scale (row std) and fuse
    def _norm_rows(M: np.ndarray) -> np.ndarray:
        s = M.std(axis=1, keepdims=True)
        s[s < 1e-8] = 1.0
        return M / s

    D1n = _norm_rows(D1)
    D2n = _norm_rows(D2)
    w1 = float(w1)
    return (w1 * D1n + (1.0 - w1) * D2n).astype(np.float32, copy=False)

# -------------------- U-FGW-BC on prototypes (two-view) --------------------
def _ufgw_barycenter(
    batches: np.ndarray,
    proto_by_batch: Dict[str, Dict[int, np.ndarray]],
    K_bar: int,
    *,
    reg: float,
    reg_m: float,
    ot_backend: str,
    use_gpu: bool,
    gpu_device: int,
    iters_outer: int = 5,
    k_struct: int = 10,
    lambda_lo: float = 0.3,
    lambda_hi: float = 0.7,
    cost_clip_q: Optional[float] = 0.90,
    view_order: Tuple[str, str] = ("X_pca", "X_lsi"),
) -> Tuple[Dict[str, np.ndarray], Dict[int, np.ndarray]]:
    """
    Returns:
      R_views: {view: [K_bar, d_view]} barycenter prototypes per view
      T_by_batch: {lbl: [Kb, K_bar]} couplings from batch prototypes to barycenter
    """
    view_order = tuple(view_order)
    assert len(view_order) == 2, "two views required for U-FGW barycenter"
    for vk in view_order:
        if vk not in proto_by_batch:
            raise KeyError(f"missing view '{vk}' in proto_by_batch")

    # --- init barycenter by kmeans on pooled prototypes per view ---
    R_views: Dict[str, np.ndarray] = {}
    rng = np.random.default_rng(0)
    for vk in view_order:
        pool = np.concatenate([proto_by_batch[vk][lbl] for lbl in batches], axis=0)
        R_views[vk] = _minikm_centers(pool, int(K_bar), seed=17, use_gpu=False, device=0)

    # outer alternation
    T_by_batch: Dict[int, np.ndarray] = {}
    for it in range(1, iters_outer + 1):
        t = (it - 1) / max(1, iters_outer - 1)
        lam = _lerp(lambda_lo, lambda_hi, t)

        # structural signatures averaged across the two views
        S_R_dict = {vk: _struct_signature(R_views[vk], k=min(k_struct, max(1, len(R_views[vk]) - 1))) for vk in view_order}
        S_R = sum(S_R_dict.values()) / float(len(S_R_dict))

        # compute couplings for each batch
        col_sum_acc = None
        num_update: Dict[str, np.ndarray] = {vk: np.zeros_like(R_views[vk]) for vk in view_order}
        for lbl in batches:
            B_by_view = {vk: proto_by_batch[vk][lbl] for vk in view_order}
            B1 = B_by_view[view_order[0]]
            B2 = B_by_view[view_order[1]]
            # feature cost
            M_feat = _feature_cost_two_view(
                B1,
                R_views[view_order[0]],
                B2,
                R_views[view_order[1]],
                w1=0.6,
                view_names=view_order,
            )
            # structure cost (FGW-lite): compare local distance signatures
            S_B_dict = {
                vk: _struct_signature(proto_by_batch[vk][lbl], k=min(k_struct, max(1, len(proto_by_batch[vk][lbl]) - 1)))
                for vk in view_order
            }
            S_B = sum(S_B_dict.values()) / float(len(S_B_dict))
            M_str = _struct_cost_from_signatures(S_B, S_R)

            # fuse + (optional) clip
            M = (1.0 - lam) * M_feat + lam * M_str
            if cost_clip_q is not None:
                thr = np.quantile(M, float(cost_clip_q), axis=1, keepdims=True)
                M = np.where(M > thr, thr + 50.0, M)

            # unbalanced OT coupling (Nb x Kbar)
            T = _uot_from_cost(M, reg=reg, reg_m=reg_m, backend=ot_backend, use_gpu=use_gpu, gpu_device=gpu_device)
            T_by_batch[lbl] = T

            # barycenter update accumulators: R_new = (T^T @ B) / (T^T @ 1)
            col_sum = T.sum(0, keepdims=True).T  # [Kbar,1]
            if col_sum_acc is None:
                col_sum_acc = col_sum
            else:
                col_sum_acc += col_sum
            for vk in view_order:
                num_update[vk] += T.T @ B_by_view[vk]

        # update R per view
        for vk in view_order:
            denom = np.maximum(col_sum_acc, 1e-12)
            R_views[vk] = (num_update[vk] / denom).astype(np.float32, copy=False)

    return R_views, T_by_batch

# -------------------- Main integration --------------------

def integrate_paired(
    adata: Any,
    obsm_key: str = "X_pca",            # base view used for KNN/smoothing & cell->proto assignment
    batch_key: str = "batch",
    out_key: str = "scBIOT",
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
    reference: str | int = "largest",   # used only in mode="reference"  (now also accepts "union")
    postscale: bool = True,
    random_state: int = 0,
    verbose: bool = True,
    use_gpu: bool = True,
    gpu_device: int = 0,
    ot_backend: str = "torch",          # {'torch','ot','pot'}
    # ---- NEW: two-view + barycenter toggle ----
    mode: str = "reference",            # {'reference','ufgw_barycenter'}
    view_keys: Tuple[str, str] = ("X_pca", "X_lsi"),
) -> Tuple[Any, Dict[str, float | int]]:
    """
    Integrate paired RNA/ATAC multiome data with unbalanced OT (single or two views).

    Parameters
    ----------
    adata
        AnnData object that already contains paired modalities in ``adata.obsm``.
    obsm_key
        Base view used for geometry and degree smoothing, typically RNA PCs.
    batch_key
        ``adata.obs`` column with batch labels.
    out_key
        Destination key in ``adata.obsm`` for the aligned embedding.
    K_ref / K_batch
        Reference and batch-local prototype counts for the OT solver.
    reg / reg_m
        Entropic and mass regularization strengths for unbalanced OT.
    sharpen / K_pseudo / pull / push
        Connectivity hyper-parameters for pseudo-label refinement.
    lambda0_hi / lambda0_lo / smin_* / smax_* / max_step_local / step_*
        Controls for step sizes and adaptive neighborhood scaling.
    q_start / q_end / overlap0_lo / overlap0_hi / w_overlap / penalty_gamma
        Scores that encourage batch mixing without over-correction.
    w_strain
        Weight placed on graph-strain regularization.
    k_local / k_eval
        Neighborhood sizes used for graph construction and evaluation.
    eval_subsample / trust_subsample
        Optional subsampling for scoring to reduce runtime.
    max_iter / patience / tol
        Early stopping conditions for the outer loop.
    reference
        Reference batch to anchor the alignment (``"largest"`` or ``"union"``).
    postscale
        Whether to z-score the output embedding per dimension.
    random_state
        Seed controlling all stochastic procedures.
    verbose
        Log progress when ``True``.
    use_gpu / gpu_device / ot_backend
        Backend selection for the OT solver (Torch by default, POT when requested).
    mode
        ``"reference"`` for single-view OT, ``"ufgw_barycenter"`` for two-view fusion.
    view_keys
        Tuple of two embedding keys (for example, RNA PCA and ATAC LSI) when
        running barycentric two-view alignment.

    Returns
    -------
    adata
        Input object with the fused embedding stored in ``adata.obsm[out_key]``.
    dict
        Diagnostics mirroring the single-view integration (``mix``, ``overlap0``,
        ``strain``, ``tw``) and the iteration that reached the best score.
    """
    ot_backend = str(ot_backend).lower()
    if ot_backend == "ot":
        ot_backend = "pot"

    # ---- Prepare views ----
    X_views = _views_from_keys(adata, view_keys)
    base_view = obsm_key  # which view to treat as base for geometry
    X0 = X_views[base_view]

    # batches
    b_raw = adata.obs[batch_key].to_numpy()
    le = LabelEncoder()
    b = le.fit_transform(b_raw).astype(np.int32, copy=False)
    n_classes = max(1, int(b.max()) + 1)

    # --- reference selection (only used if mode='reference') ---
    # Support 'largest', 'union', or explicit int
    ref_label_enc: Optional[int]
    if isinstance(reference, str):
        r = reference.lower()
        if r == "largest":
            vals, counts = np.unique(b_raw, return_counts=True)
            ref_label_raw = vals[np.argmax(counts)]
            ref_label_enc = int(np.where(le.classes_ == ref_label_raw)[0][0])
        elif r == "union":
            ref_label_enc = None  # sentinel for union
        else:
            raise ValueError("reference must be 'largest', 'union', or an integer label")
    else:
        ref_label_enc = int(reference)

    # working copies for base view only (old-style outer loop)
    X = X0.copy()
    mu0 = X.mean(0)
    sd0 = X.std(0) + 1e-8

    d0, idx0 = _knn_graph(X0, k=max(15, k_local + 5), use_gpu=use_gpu, device=gpu_device)
    knn_mean_dist0 = d0.mean(axis=1) if d0.size else np.zeros(len(X), dtype=X.dtype)
    dens0 = (knn_mean_dist0 - (knn_mean_dist0.min() if len(knn_mean_dist0) else 0.0)) / (_ptp(knn_mean_dist0) + 1e-12)

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
        backend = "FAISS-GPU" if (_FAISS_AVAIL and use_gpu and _FAISS_GPU) else ("FAISS-CPU" if _FAISS_AVAIL else "sklearn")
        print(f"[baseline] KNN backend={backend} mix={mix0:.4f} strain={strain0:.5f} mode={mode}")

    # --------------- OUTER OPTIMIZATION LOOP (unchanged scaffolding) ---------------
    no_imp = 0
    for it in range(1, max_iter + 1):
        t = (it - 1) / max(1, max_iter - 1)
        lambda_graph0 = _lerp(lambda0_hi, lambda0_lo, t)
        overlap0_floor = _lerp(overlap0_lo, overlap0_hi, t)
        step = _lerp(step_lo, step_hi, t)
        cost_clip_q = _lerp(q_start, q_end, t)

        shift = np.zeros_like(X, dtype=X.dtype)
        alpha = np.ones(len(X), dtype=X.dtype)

        if mode == "reference":
            # ---- OLD REFERENCE MODE (now supports 'largest' and 'union') ----
            def _compute_prototypes_unionaware(
                X_base: np.ndarray,
                b_enc: np.ndarray,
                ref_label: Optional[int],   # None => union
                K_ref: int,
                K_batch: int,
                seed: int,
            ) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
                if ref_label is None:
                    # UNION: reference = k-means on ALL cells (base view)
                    Kref_eff = int(min(K_ref, max(16, 2 * np.sqrt(max(len(X_base), 1)))))
                    R = _minikm_centers(X_base, Kref_eff, seed, use_gpu=use_gpu, device=gpu_device)
                    target_batches = np.unique(b_enc)
                else:
                    # LARGEST / explicit label: reference = k-means on that batch only
                    ref_idx = np.where(b_enc == ref_label)[0]
                    X_ref = X_base[ref_idx]
                    Kref_eff = int(min(K_ref, max(16, 2 * np.sqrt(max(len(ref_idx), 1)))))
                    R = _minikm_centers(X_ref, Kref_eff, seed, use_gpu=use_gpu, device=gpu_device)
                    target_batches = np.array([lbl for lbl in np.unique(b_enc) if lbl != ref_label], dtype=b_enc.dtype)

                packs: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
                for lbl in target_batches:
                    idx = np.where(b_enc == lbl)[0]
                    if len(idx) == 0:
                        continue
                    Xi = X_base[idx]
                    Kb_eff = int(min(K_batch, max(8, 2 * np.sqrt(len(idx)))))
                    Bi = _minikm_centers(Xi, Kb_eff, seed + 7, use_gpu=use_gpu, device=gpu_device)
                    if _FAISS_AVAIL:
                        _, nn_idx = _faiss_knn_search(Xi, Bi, 1, use_gpu=use_gpu, device=gpu_device)
                        nn_idx = nn_idx.ravel()
                    else:
                        nn_idx = pairwise_distances_argmin(Xi, Bi, metric="euclidean")
                    packs.append((idx, Bi, nn_idx))
                return R, packs

            R, packs = _compute_prototypes_unionaware(X, b, ref_label_enc, K_ref, K_batch, random_state + it)

            for (idx, Bi, nn_idx) in packs:
                Bi_to_R = _ot_barycentric_gpu(
                    Bi, R,
                    reg=reg, reg_m=reg_m,
                    cost_clip_q=cost_clip_q, clip_big=50.0,
                    ot_backend=ot_backend, iters=1000, tol=1e-6,
                    use_gpu=use_gpu, gpu_device=gpu_device,
                )
                disp_proto = Bi_to_R - Bi
                norm_move = np.linalg.norm(disp_proto, axis=1)
                s_dist = 1.0 / (1.0 + (norm_move / (norm_move.std() + 1e-8)))
                alpha_i = s_dist[nn_idx] * (1.0 - 0.35 * bridge_score[idx])
                alpha[idx] = alpha_i.astype(X.dtype, copy=False)
                shift[idx] = disp_proto[nn_idx].astype(X.dtype, copy=False)

        else:
            # ---- NEW: U-FGW-BC (two-view) on batch prototypes ----
            batches, proto_by_batch, assign_by_batch = _compute_batch_prototypes_per_view(
                X_views, b, K_ref, K_batch, seed=random_state + it, base_view=base_view,
                use_gpu=use_gpu, device=gpu_device
            )
            R_views, T_by_batch = _ufgw_barycenter(
                batches, proto_by_batch, K_bar=K_ref,
                reg=reg, reg_m=reg_m, ot_backend=ot_backend,
                use_gpu=use_gpu, gpu_device=gpu_device,
                iters_outer=5, k_struct=10, lambda_lo=0.3, lambda_hi=0.7,
                cost_clip_q=cost_clip_q,
                view_order=view_keys,
            )

            for lbl in batches:
                idx_cells, nn_idx = assign_by_batch[lbl]
                Bp = proto_by_batch[base_view][lbl]
                T = T_by_batch[lbl]
                denom = T.sum(1, keepdims=True) + 1e-12
                B_to_R = (T @ R_views[base_view]) / denom
                disp_proto = B_to_R - Bp
                norm_move = np.linalg.norm(disp_proto, axis=1)
                s_dist = 1.0 / (1.0 + (norm_move / (norm_move.std() + 1e-8)))
                alpha_i = s_dist[nn_idx] * (1.0 - 0.32 * bridge_score[idx_cells]) * (1.0 + 0.12 * dens0[idx_cells])
                alpha[idx_cells] = alpha_i.astype(X.dtype, copy=False)
                shift[idx_cells] = disp_proto[nn_idx].astype(X.dtype, copy=False)

        # finalize move in base view (old shaping pipeline)
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
            shift = alpha[:, None] * shift + sharpen * sharp
        else:
            shift = alpha[:, None] * shift

        shift *= (1.0 - 0.12 * bridge_score)[:, None]
        shift = _smooth_by_knn(shift, idx0, lam=lambda_graph0)

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
                f"strain={strain_c:.5f} J={J:.3f} best_it={best['it']}"
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
    return adata, dict(mix=best["mix"], overlap0=best["overlap0"], strain=best["strain"], tw=tw, it=best["it"])

# -------------------- Example usage --------------------
# Example here shows the NEW barycenter core on two views (recommended for multiome):
# adata, metrics = integrate_paired(
#     adata,
#     obsm_key="X_pca",              # base view for geometry/smoothing
#     batch_key="batch",
#     out_key="scBIOT_OT",
#     mode="ufgw_barycenter",        # <--- NEW: Unbalanced FGW Barycenter
#     view_keys=("X_pca", "X_lsi"),  # two-view fusion (RNA PCs + ATAC LSI)
#     reference="union",           # "largest" or "union"
#     K_ref=1200,
#     K_batch=420,
#     reg=0.034,
#     reg_m=0.28,
#     sharpen=0.10,
#     K_pseudo=20,
#     pull=0.72,
#     push=0.24,
#     lambda0_hi=0.56,
#     lambda0_lo=0.44,
#     smin_bulk=0.80,
#     smax_bulk=1.55,
#     smin_bridge=0.92,
#     smax_bridge=1.14,
#     max_step_local=0.92,
#     step_lo=0.70,
#     step_hi=0.88,
#     q_start=0.78,
#     q_end=0.88,
#     overlap0_lo=0.64,
#     overlap0_hi=0.74,
#     w_overlap=0.14,
#     w_strain=1.1,
#     penalty_gamma=1.60,
#     verbose=True,
#     use_gpu=True,
#     gpu_device=0,
#     ot_backend="torch",            # {'torch','ot','pot'}
# )
# print(metrics)
