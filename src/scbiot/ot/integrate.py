# =================== integrate.py: unified pipeline (rare-protection + supervised option) ===================
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder

from ..utils.coral_helpers import _coral_prealign, _ot_prealign
from ..utils.ot_helpers import (
    _FAISS_AVAIL,
    _FAISS_GPU,
    _as_nd_f32_c,
    _class_means,
    _compute_prototypes,
    _compute_prototypes_union,
    _graph_strain,
    _knn_idx,
    _knn_graph,
    _knn_overlap,
    _neighbor_batch_entropy_per_cell,
    _nearest_other_class_index,
    _ot_barycentric_gpu,
    _ptp,
    _resolve_reference_query_masks,
    _trustworthiness_score,
)
from ..utils.ot_transport import compute_ot_alignment
from ._time import _standardize_time, _time_to_numeric
from .supbiot import _validate_transfer_mode, predict_pseudo_labels

# -------------------- Main integration (now supervised-ready) --------------------


@dataclass(frozen=True)
class PseudoLabelConfig:
    every_iters: int = 1
    min_conf: float = 0.80
    min_batches: int = 2
    min_count: int = 30
    ema: float = 0.30
    max_ref: int = 20000
    debatch_max: float = 0.80
    debatch_ramp_power: float = 2.0


@dataclass(frozen=True)
class BacktrackingConfig:
    factor: float = 0.5
    max_backtracks: int = 8
    eta_stop: float = 1.0 / 64.0


_PSEUDO_CFG = PseudoLabelConfig()
_BACKTRACK_CFG = BacktrackingConfig()

_DEFAULT_K_LOCAL = 15
_DEFAULT_K_EVAL = 30
_DEFAULT_EVAL_SUBSAMPLE = 5000
_DEFAULT_TRUST_SUBSAMPLE = 2500
_DEFAULT_MAX_ITER = 15
_DEFAULT_PATIENCE = 2
_DEFAULT_TOL = 1e-3
_AFFINE_FIT_MAX = 500_000
_PREALIGN_AUTO_K = 30
_PREALIGN_AUTO_MAX_POINTS = 5000
_PREALIGN_AUTO_THRESHOLD = 0.45
_PREALIGN_AUTO_STRENGTH_MAX = 0.80
_PREALIGN_AUTO_STRENGTH_SCALE = 1.80
_RARE_LOCAL_SCALE_Q = 0.85
# Reference classes whose share of labeled cells is below this fraction are
# treated as rare and shielded from batch-mixing displacement (label-aware
# complement to the density-based rare mask).
_SMALL_CLASS_FRAC = 0.01
_ALIGN_REFERENCE_REP_KEYS = {"x_pca", "x_spca"}
_ALIGN_REFERENCE_PCA_PREALIGN_STRENGTH = 0.35
_ALIGN_REFERENCE_PCA_RARE_SCALE = 0.25

DEFAULT_MAX_ITER = _DEFAULT_MAX_ITER

_MODALITY_DEFAULTS: Dict[str, Dict[str, float]] = {
    "rna": {
        "reg": 0.03,
        "reg_m": 0.40,
    },
    "atac": {
        "reg": 0.036,
        "reg_m": 0.30,
    },
}


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _clip_int(value: float, lo: int, hi: int) -> int:
    return int(np.clip(np.round(value), lo, hi))


def _batchwise_rare_mask(
    knn_mean_dist0: np.ndarray,
    b: np.ndarray,
    rare_quantile: float,
) -> np.ndarray:
    knn_mean_dist0 = np.asarray(knn_mean_dist0, dtype=np.float32)
    b = np.asarray(b)
    out = np.zeros(len(knn_mean_dist0), dtype=bool)

    for bi in np.unique(b):
        m = (b == bi)
        n = int(m.sum())
        if n == 0:
            continue
        if n < 32:
            # tiny batches are fragile: protect all of them from diffusion
            out[m] = True
            continue
        q = np.quantile(knn_mean_dist0[m], rare_quantile)
        out[m] = knn_mean_dist0[m] >= q
    return out


def _small_class_mask(
    y_eff: Optional[np.ndarray],
    n_labels: int,
    *,
    frac: float = _SMALL_CLASS_FRAC,
) -> Optional[np.ndarray]:
    """Flag cells whose labeled (reference) class is small.

    A class is "small" when its share of labeled cells is below ``frac``. These
    are exactly the rare cell types that drive macro-F1, so the density-based
    rare mask is augmented with them to shield their geometry from batch-mixing
    displacement. Unlabeled cells (``y_eff < 0``) are never flagged here.
    """
    if y_eff is None or n_labels <= 0:
        return None
    labeled = np.asarray(y_eff) >= 0
    n_labeled = int(labeled.sum())
    if n_labeled == 0:
        return None
    counts = np.bincount(np.asarray(y_eff)[labeled], minlength=int(n_labels))
    threshold = max(1.0, float(frac) * float(n_labeled))
    small_classes = np.flatnonzero((counts > 0) & (counts <= threshold))
    if small_classes.size == 0:
        return None
    out = np.zeros(len(y_eff), dtype=bool)
    out[labeled] = np.isin(np.asarray(y_eff)[labeled], small_classes)
    return out


@dataclass(frozen=True)
class GeometryPolicy:
    proto_ref: int
    proto_batch: int
    reg: float
    reg_m: float
    cost_clip_q: float
    geom_budget: float
    geom_budget_q90: float
    diffusion_alpha: float
    diffusion_steps: int
    rare_quantile: float
    lam_sup: float
    lam_repulse: float


def derive_geometry_policy(
    *,
    modality: str,
    n_obs: int,
    med_batch: int,
    knobs: OTKnobs,
) -> GeometryPolicy:
    defaults = _MODALITY_DEFAULTS[modality]
    n_obs = max(1, int(n_obs))
    med_batch = max(1, int(med_batch))
    conservation = _clip01(knobs.conservation)
    prototypes = _clip01(knobs.prototypes)
    supervision = _clip01(knobs.supervision)
    geom_budget = float(np.clip(0.14 - 0.08 * conservation, 0.05, 0.14))
    return GeometryPolicy(
        proto_ref=_clip_int(np.sqrt(n_obs) * (0.75 + 1.50 * prototypes), 64, 4096),
        proto_batch=_clip_int(np.sqrt(med_batch) * (0.75 + 1.25 * prototypes), 32, 2048),
        reg=float(defaults["reg"]),
        reg_m=float(defaults["reg_m"]),
        cost_clip_q=0.85,
        geom_budget=geom_budget,
        geom_budget_q90=float(np.clip(geom_budget * 1.8, 0.10, 0.25)),
        diffusion_alpha=float(np.clip(0.10 + 0.20 * conservation, 0.10, 0.30)),
        diffusion_steps=1,
        rare_quantile=float(_RARE_LOCAL_SCALE_Q),
        lam_sup=0.8 * supervision,
        lam_repulse=0.25 * supervision,
    )


def derive_policy(
    *,
    n_obs: int,
    med_batch: int,
    strength: float,
    conservation: float,
    prototypes: float,
) -> GeometryPolicy:
    # Backwards-compatible shim used by downstream tests and legacy callers.
    knobs = OTKnobs(
        strength=strength,
        conservation=conservation,
        prototypes=prototypes,
        supervision=0.0,
    )
    return derive_geometry_policy(modality="rna", n_obs=n_obs, med_batch=med_batch, knobs=knobs)


def build_diffusion_operator(
    X: np.ndarray,
    idx: np.ndarray,
    dist: np.ndarray,
) -> sp.csr_matrix:
    X = _as_nd_f32_c(X)
    idx = np.asarray(idx, dtype=np.int32)
    dist = np.asarray(dist, dtype=np.float32)
    n_obs = int(X.shape[0])
    if n_obs == 0 or idx.size == 0 or dist.size == 0:
        return sp.csr_matrix((n_obs, n_obs), dtype=np.float32)

    rows = np.repeat(np.arange(n_obs, dtype=np.int32), idx.shape[1])
    cols = idx.reshape(-1)
    dij = dist.reshape(-1)
    sigma = np.median(dist, axis=1)
    sigma = np.clip(sigma, 1e-6, None)
    sigma_ij = sigma[rows] * sigma[cols]
    weights = np.exp(-(dij ** 2) / (sigma_ij + 1e-6)).astype(np.float32, copy=False)

    W = sp.csr_matrix((weights, (rows, cols)), shape=(n_obs, n_obs), dtype=np.float32)
    W = 0.5 * (W + W.T)
    degree = np.asarray(W.sum(axis=1)).ravel()
    degree = np.clip(degree, 1e-6, None)
    return (sp.diags(1.0 / degree, dtype=np.float32) @ W).tocsr()


def diffusion_smooth(
    shift: np.ndarray,
    diffusion: sp.csr_matrix,
    *,
    alpha: float,
    steps: int,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    out = _as_nd_f32_c(shift).copy()
    if alpha <= 0.0 or steps <= 0 or diffusion.nnz == 0:
        return out

    allow = np.ones(len(out), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    for _ in range(int(steps)):
        mixed = np.asarray(diffusion @ out, dtype=np.float32)
        out[allow] = (1.0 - alpha) * out[allow] + alpha * mixed[allow]
    return out


def relative_local_distortion(
    X_new: np.ndarray,
    idx_ref: np.ndarray,
    d_ref: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[float, np.ndarray]:
    if idx_ref.size == 0 or d_ref.size == 0:
        per_cell = np.zeros(int(np.asarray(X_new).shape[0]), dtype=np.float32)
        return 0.0, per_cell
    X_new = _as_nd_f32_c(X_new)
    nbr = X_new[idx_ref]
    d_new = np.linalg.norm(X_new[:, None, :] - nbr, axis=2)
    rel = np.abs(d_new / (d_ref + eps) - 1.0)
    per_cell = rel.mean(axis=1).astype(np.float32, copy=False)
    return float(per_cell.mean()), per_cell


def build_supervised_edge_weights(
    idx_ref: np.ndarray,
    labels: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    idx_ref = np.asarray(idx_ref, dtype=np.int32)
    labels = np.asarray(labels, dtype=np.int32)
    n_obs, k = idx_ref.shape if idx_ref.ndim == 2 else (len(labels), 0)
    if n_obs == 0 or k == 0:
        return np.zeros((n_obs, k), dtype=np.float32)
    if weights is None:
        weights = np.ones(n_obs, dtype=np.float32)
    else:
        weights = np.asarray(weights, dtype=np.float32)

    src_labels = labels[:, None]
    nbr_labels = labels[idx_ref]
    src_w = weights[:, None]
    nbr_w = weights[idx_ref]

    same_known = (src_labels >= 0) & (nbr_labels >= 0) & (src_labels == nbr_labels)
    diff_known = (src_labels >= 0) & (nbr_labels >= 0) & (src_labels != nbr_labels)
    any_unknown = (src_labels < 0) | (nbr_labels < 0)

    edge_w = np.zeros((n_obs, k), dtype=np.float32)
    edge_w[same_known] = np.minimum(src_w, nbr_w)[same_known]
    edge_w[diff_known] = (0.05 * np.minimum(src_w, nbr_w))[diff_known]
    edge_w[any_unknown] = 0.35
    return np.clip(edge_w, 0.0, 1.0, out=edge_w)


def label_aware_local_distortion(
    X_new: np.ndarray,
    idx_ref: np.ndarray,
    d_ref: np.ndarray,
    labels: np.ndarray,
    weights: Optional[np.ndarray],
    eps: float = 1e-6,
) -> Tuple[float, np.ndarray]:
    base_mean, base_per_cell = relative_local_distortion(X_new, idx_ref, d_ref, eps=eps)
    if idx_ref.size == 0 or d_ref.size == 0:
        return base_mean, base_per_cell
    edge_w = build_supervised_edge_weights(idx_ref, labels, weights)
    if edge_w.size == 0:
        return base_mean, base_per_cell

    X_new = _as_nd_f32_c(X_new)
    nbr = X_new[idx_ref]
    d_new = np.linalg.norm(X_new[:, None, :] - nbr, axis=2)
    rel = np.abs(d_new / (d_ref + eps) - 1.0).astype(np.float32, copy=False)
    per_cell = (edge_w * rel).sum(axis=1) / (edge_w.sum(axis=1) + 1e-6)
    per_cell = per_cell.astype(np.float32, copy=False)
    return float(per_cell.mean()), per_cell


@dataclass
class SupervisionState:
    has_label_key: bool
    y: Optional[np.ndarray]
    y_eff: Optional[np.ndarray]
    w_sup: Optional[np.ndarray]
    n_labels: int
    do_pseudo: bool
    label_to_code: Dict[str, int]
    unknown_aliases: Tuple[str, ...]


@dataclass(frozen=True)
class LoopContext:
    idx0: np.ndarray
    d0: np.ndarray
    diffusion: sp.csr_matrix
    rare_mask: np.ndarray
    bridge_score: np.ndarray
    mix0: float
    strain0: float


def _get_geometry_distortion_fn(
    *,
    loop_context: LoopContext,
    supervision: SupervisionState,
    knobs: OTKnobs,
) -> Callable[[np.ndarray], Tuple[float, np.ndarray]]:
    if (
        supervision.has_label_key
        and supervision.y_eff is not None
        and supervision.w_sup is not None
        and np.any(supervision.y_eff >= 0)
        and knobs.supervision > 0.0
    ):
        return lambda X_new: label_aware_local_distortion(
            X_new,
            loop_context.idx0,
            loop_context.d0,
            supervision.y_eff,
            supervision.w_sup,
        )
    return lambda X_new: relative_local_distortion(X_new, loop_context.idx0, loop_context.d0)


def adaptive_trust_move(
    X: np.ndarray,
    shift: np.ndarray,
    idx_ref: np.ndarray,
    d_ref: np.ndarray,
    *,
    step_size: float,
) -> np.ndarray:
    shift = _as_nd_f32_c(shift)
    _, per_cell_error = relative_local_distortion(_as_nd_f32_c(X) + shift, idx_ref, d_ref)
    trust = 1.0 / (1.0 + per_cell_error)
    move = float(step_size) * trust[:, None] * shift
    radius = d_ref.mean(axis=1) if d_ref.size else np.zeros(len(move), dtype=np.float32)
    move_norm = np.linalg.norm(move, axis=1)
    cap = np.minimum(1.0, radius / (move_norm + 1e-6))
    return move * cap[:, None]


def backtracking_geometry_step(
    X: np.ndarray,
    shift: np.ndarray,
    idx_ref: np.ndarray,
    d_ref: np.ndarray,
    geom_budget: float,
    geom_budget_q90: float,
    factor: float = _BACKTRACK_CFG.factor,
    max_backtracks: int = _BACKTRACK_CFG.max_backtracks,
) -> Tuple[np.ndarray, float, float, float]:
    X = _as_nd_f32_c(X)
    shift = _as_nd_f32_c(shift)
    eta = 1.0
    best_X = X.copy()
    best_eta = 0.0
    best_mean = np.inf
    best_q90 = np.inf
    for _ in range(int(max_backtracks) + 1):
        Xcand = X + eta * shift
        mean_d, per_cell = relative_local_distortion(Xcand, idx_ref, d_ref)
        q90_d = float(np.quantile(per_cell, 0.90)) if per_cell.size else 0.0
        if mean_d <= geom_budget and q90_d <= geom_budget_q90:
            return Xcand, float(eta), float(mean_d), float(q90_d)
        if mean_d < best_mean or (np.isclose(mean_d, best_mean) and q90_d < best_q90):
            best_X = Xcand
            best_eta = float(eta)
            best_mean = float(mean_d)
            best_q90 = float(q90_d)
        eta *= float(factor)
    return best_X, best_eta, best_mean, best_q90


def apply_geometry_budgeted_ot_step(
    X: np.ndarray,
    shift: np.ndarray,
    idx_ref: np.ndarray,
    d_ref: np.ndarray,
    geom_budget: float,
    geom_budget_q90: float,
) -> Tuple[np.ndarray, Dict[str, float | bool]]:
    Xnext, eta, mean_d, q90_d = backtracking_geometry_step(
        X,
        shift,
        idx_ref,
        d_ref,
        geom_budget=geom_budget,
        geom_budget_q90=geom_budget_q90,
    )
    feasible = bool(mean_d <= geom_budget and q90_d <= geom_budget_q90)
    return Xnext, {
        "eta": float(eta),
        "local_distortion": float(mean_d),
        "local_distortion_q90": float(q90_d),
        "feasible": feasible,
        "constrained_step": not feasible,
    }


def _sample_eval_views(
    X_ref: np.ndarray,
    X_new: np.ndarray,
    *,
    eval_subsample: Optional[int],
    trust_subsample: Optional[int],
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n_obs = int(len(X_new))
    if trust_subsample is not None and n_obs > trust_subsample:
        rng = np.random.default_rng(random_state)
        sample = rng.choice(n_obs, size=trust_subsample, replace=False)
        return X_ref[sample], X_new[sample]
    if eval_subsample is not None and n_obs > eval_subsample:
        rng = np.random.default_rng(random_state + 17)
        sample = rng.choice(n_obs, size=eval_subsample, replace=False)
        return X_ref[sample], X_new[sample]
    return X_ref, X_new


def _format_iteration_log(
    *,
    it: int,
    eta: float,
    mix: float,
    local_distortion: float,
    local_distortion_q90: float,
    feasible: bool,
    constrained_step: bool,
    accepted: bool,
    overlap0: float,
    strain: float,
    best_it: int,
) -> str:
    return (
        f"[iter {it:02d}] mix={mix:.3f} overlap0={overlap0:.3f} "
        f"strain={strain:.5f} best_it={best_it}"
    )


def should_accept_step(
    mix_delta: float,
    step_info: Dict[str, float | bool],
    *,
    tol: float,
    eta_stop: float = _BACKTRACK_CFG.eta_stop,
) -> bool:
    if bool(step_info["constrained_step"]) and float(step_info["eta"]) <= eta_stop:
        return False
    return mix_delta > tol


def _embedding_diagnostics(
    X0: np.ndarray,
    X_new: np.ndarray,
    b: np.ndarray,
    idx0: np.ndarray,
    d0: np.ndarray,
    *,
    k_eval: int,
    eval_subsample: Optional[int],
    trust_subsample: Optional[int],
    random_state: int,
    use_gpu: bool,
    gpu_device: int,
    include_tw: bool,
) -> Dict[str, float]:
    mix = float(_neighbor_batch_entropy_per_cell(X_new, b, k=k_eval, use_gpu=use_gpu, device=gpu_device)[1])
    overlap0 = float(
        _knn_overlap(
            X0,
            X_new,
            k=k_eval,
            subsample=eval_subsample,
            rng=random_state,
            use_gpu=use_gpu,
            device=gpu_device,
        )
    )
    strain = float(_graph_strain(X_new, idx0, d0, use_gpu=use_gpu, device=gpu_device))
    local_distortion, per_cell = relative_local_distortion(X_new, idx0, d0)
    local_distortion_q90 = float(np.quantile(per_cell, 0.90)) if per_cell.size else 0.0
    diagnostics = {
        "mix": mix,
        "overlap0": overlap0,
        "strain": strain,
        "local_distortion": float(local_distortion),
        "local_distortion_q90": float(local_distortion_q90),
    }
    if include_tw:
        X0_eval, X_new_eval = _sample_eval_views(
            X0,
            X_new,
            eval_subsample=eval_subsample,
            trust_subsample=trust_subsample,
            random_state=random_state,
        )
        k_tw = min(k_eval, max(1, len(X_new_eval) - 1))
        diagnostics["tw"] = float(
            _trustworthiness_score(
                X0_eval,
                X_new_eval,
                n_neighbors=k_tw,
                use_gpu=use_gpu,
                gpu_device=gpu_device,
            )
        )
    return diagnostics


def _label_knn_purity(
    X: np.ndarray,
    labels: Optional[np.ndarray],
    weights: Optional[np.ndarray],
    *,
    k: int,
    use_gpu: bool,
    gpu_device: int,
) -> float:
    if labels is None:
        return 1.0
    labels = np.asarray(labels, dtype=np.int32)
    known = labels >= 0
    if weights is not None:
        weights_arr = np.asarray(weights, dtype=np.float32)
        known &= weights_arr > 0
    else:
        weights_arr = np.ones(len(labels), dtype=np.float32)
    if int(known.sum()) <= 1:
        return 1.0

    X_known = _as_nd_f32_c(X)[known]
    y_known = labels[known]
    w_known = weights_arr[known].astype(np.float32, copy=False)
    k_eff = int(min(max(1, k), X_known.shape[0] - 1))
    idx = _knn_idx(X_known, k_eff, use_gpu=use_gpu, device=gpu_device)
    if idx.size == 0:
        return 1.0

    same = y_known[idx] == y_known[:, None]
    nbr_w = w_known[idx]
    per_cell = (same * nbr_w).sum(axis=1) / (nbr_w.sum(axis=1) + 1e-12)
    return float((per_cell * w_known).sum() / (w_known.sum() + 1e-12))


def _label_batch_entropy(
    X: np.ndarray,
    b: np.ndarray,
    y_eff: Optional[np.ndarray],
    *,
    k: int = 30,
    min_cells: int = 30,
    use_gpu: bool = True,
    device: int = 0,
) -> Dict[int, float]:
    if y_eff is None:
        return {}
    X = _as_nd_f32_c(X)
    b = np.asarray(b, dtype=np.int32)
    y_eff = np.asarray(y_eff, dtype=np.int32)
    out: Dict[int, float] = {}
    for label in np.unique(y_eff[y_eff >= 0]):
        label_int = int(label)
        mask = y_eff == label_int
        n_label = int(mask.sum())
        if n_label < int(min_cells):
            continue
        batches = np.unique(b[mask])
        n_batches = int(batches.size)
        if n_batches <= 1:
            out[label_int] = 0.0
            continue
        k_eff = int(min(max(1, k), n_label - 1))
        if k_eff <= 0:
            continue
        X_label = X[mask]
        b_label = b[mask]
        idx = _knn_idx(X_label, k_eff, use_gpu=use_gpu, device=device)
        if idx.size == 0:
            continue
        nbr_batches = b_label[idx]
        entropy = np.zeros(n_label, dtype=np.float32)
        for batch in batches:
            p = (nbr_batches == batch).mean(axis=1).astype(np.float32, copy=False)
            entropy -= np.where(p > 0, p * np.log(p + 1e-12), 0.0)
        out[label_int] = float(np.clip(entropy.mean() / (np.log(n_batches) + 1e-12), 0.0, 1.0))
    return out


def _adaptive_label_debatch_strengths(
    label_entropy: Dict[int, float],
    *,
    max_strength: float,
    target_entropy: float = 0.90,
) -> Dict[int, float]:
    strengths: Dict[int, float] = {}
    target = max(float(target_entropy), 1e-6)
    for label, entropy in label_entropy.items():
        deficit = float(np.clip((target - float(entropy)) / target, 0.0, 1.0))
        if deficit <= 0.0:
            strengths[int(label)] = 0.0
        else:
            strengths[int(label)] = float(np.clip(0.20 + max_strength * np.sqrt(deficit), 0.0, 1.0))
    return strengths


def _selection_score(
    *,
    batch_mix: float,
    label_purity: float,
    overlap0: float,
    strain: float,
) -> float:
    return float(batch_mix + 0.25 * label_purity + 0.10 * overlap0 - 0.10 * strain)


def _fit_affine(src: np.ndarray, tgt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    src = np.asarray(src, dtype=np.float64)
    tgt = np.asarray(tgt, dtype=np.float64)
    ones = np.ones((src.shape[0], 1), dtype=src.dtype)
    src_aug = np.hstack([src, ones])
    coeff, _, _, _ = np.linalg.lstsq(src_aug, tgt, rcond=None)
    A = coeff[:-1, :].astype(np.float32, copy=False)
    t = coeff[-1, :].astype(np.float32, copy=False)
    return A, t


@dataclass
class OTKnobs:
    strength: float = 0.5
    conservation: float = 0.5
    prototypes: float = 0.5
    supervision: float = 0.5
    approximate: bool = False
    centroid: bool = False

    def __post_init__(self) -> None:
        self.strength = _clip01(self.strength)
        self.conservation = _clip01(self.conservation)
        self.prototypes = _clip01(self.prototypes)
        self.supervision = _clip01(self.supervision)
        self.approximate = bool(self.approximate)
        self.centroid = bool(self.centroid)


def _infer_modality(modality: str, obsm_key: str, X: np.ndarray) -> str:
    if modality != "auto":
        return modality
    key = str(obsm_key).lower()
    if "lsi" in key or "atac" in key:
        return "atac"
    # Heuristic: LSI embeddings are often row-L2 normalized ~1.
    rn = np.linalg.norm(X, axis=1)
    if np.isfinite(rn).all():
        if abs(float(rn.mean()) - 1.0) < 0.05 and float(rn.std()) < 0.10:
            return "atac"
    return "rna"


def _normalize_embedding(
    X: np.ndarray,
    modality: str,
    params: Optional[Dict[str, np.ndarray]] = None,
    mean_center: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    X = _as_nd_f32_c(X)
    if modality == "rna":
        return X, {} if params is None else params
    if params is None:
        params = {}
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    Xn = X / norms
    if mean_center:
        if "mean" not in params:
            params["mean"] = Xn.mean(0, keepdims=True)
        Xn = Xn - params["mean"]
    return Xn, params


def _auto_prealign_score(
    X: np.ndarray,
    batches: np.ndarray,
    k: int,
    max_points: int,
    rng: int,
    use_gpu: bool,
    device: int,
    apply_dims: Optional[slice],
) -> float:
    if X.size == 0 or batches.size == 0:
        return 0.0
    X_use = X[:, apply_dims] if apply_dims is not None else X
    n_obs = int(X_use.shape[0])
    if n_obs <= 1:
        return 0.0
    max_points = int(max_points)
    if max_points <= 0:
        max_points = n_obs
    if max_points < n_obs:
        rng_state = np.random.default_rng(int(rng))
        sub = rng_state.choice(n_obs, size=max_points, replace=False)
        X_sub = X_use[sub]
        batches_sub = batches[sub]
    else:
        X_sub = X_use
        batches_sub = batches
    k_eff = int(min(max(1, k), X_sub.shape[0] - 1))
    if k_eff <= 0:
        return 0.0
    idx = _knn_idx(X_sub, k_eff, use_gpu=use_gpu, device=device)
    if idx.size == 0:
        return 0.0
    neighbor_batches = batches_sub[idx]
    same_batch = neighbor_batches == batches_sub[:, None]
    mix_score = 1.0 - same_batch.mean()
    return float(mix_score)


def _auto_prealign_strength(
    mix_score: float,
    threshold: float,
    strength_max: float,
) -> float:
    if mix_score >= threshold:
        return 0.0
    u = (threshold - mix_score) / max(threshold, 1e-12)
    raw = u * strength_max
    scaled = raw * float(_PREALIGN_AUTO_STRENGTH_SCALE)
    upper_bound = float(np.clip(strength_max, 0.0, 1.0))
    return float(np.clip(scaled, 0.0, upper_bound))


@dataclass(frozen=True)
class PreparedEmbeddingState:
    X0: np.ndarray
    b_raw: np.ndarray
    b: np.ndarray
    le_batch: LabelEncoder
    modality: str
    apply_dims: Optional[slice]
    n_obs: int
    n_classes: int
    med_batch: int


@dataclass(frozen=True)
class PrealignState:
    X0: np.ndarray
    prealign_norm: str
    prealign_strength: float
    ref_mask: Optional[np.ndarray]
    query_mask: Optional[np.ndarray]


def _prepare_embedding_and_batches(
    adata: Any,
    *,
    obsm_key: str,
    batch_key: str,
    modality: str,
    spatial_key: Optional[str],
    spatial_weight: float,
    time_key: Optional[str] = None,
    time_weight: float = 0.5,
    time_mode: str = "auto",
    preserve_input_embedding: bool = False,
) -> PreparedEmbeddingState:
    if obsm_key not in adata.obsm:
        raise KeyError(f"Embedding '{obsm_key}' not found in adata.obsm.")

    X0_raw = _as_nd_f32_c(adata.obsm[obsm_key])
    if modality == "auto" and hasattr(adata, "uns") and isinstance(getattr(adata, "uns"), dict):
        for key in ("scbiot_modality", "modality"):
            value = str(adata.uns.get(key, "")).lower()
            if value in _MODALITY_DEFAULTS:
                modality = value
                break
    modality = _infer_modality(modality, obsm_key, X0_raw)
    if preserve_input_embedding:
        X0 = X0_raw.copy()
    else:
        X0, _ = _normalize_embedding(X0_raw, modality)

    d_embed = X0.shape[1]
    apply_dims = None
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
        apply_dims = slice(0, d_embed)

    if time_key is not None:
        if time_key not in adata.obs:
            raise KeyError(f"Time key '{time_key}' not found in adata.obs.")
        t_raw, _ = _time_to_numeric(adata.obs[time_key], time_mode)
        if t_raw.shape[0] != X0.shape[0]:
            raise ValueError(
                f"Time values '{time_key}' have {t_raw.shape[0]} rows; expected {X0.shape[0]}."
            )
        t_std, _, _ = _standardize_time(t_raw)
        X0 = np.hstack([X0, (t_std[:, None]).astype(np.float32) * float(time_weight)])
        apply_dims = slice(0, d_embed)

    b_raw = adata.obs[batch_key].to_numpy()
    le_batch = LabelEncoder()
    b = le_batch.fit_transform(b_raw).astype(np.int32, copy=False)
    n_classes = max(1, int(b.max()) + 1)
    n_obs = int(X0.shape[0])
    counts = np.bincount(b, minlength=n_classes)
    med_batch = int(np.median(counts)) if counts.size else n_obs

    return PreparedEmbeddingState(
        X0=X0,
        b_raw=b_raw,
        b=b,
        le_batch=le_batch,
        modality=modality,
        apply_dims=apply_dims,
        n_obs=n_obs,
        n_classes=n_classes,
        med_batch=med_batch,
    )


def _prepare_supervision(
    adata: Any,
    *,
    label_key: Optional[str],
    unlabeled_category: Any,
) -> SupervisionState:
    has_label_key = label_key is not None and label_key in adata.obs
    y: Optional[np.ndarray]
    le_labels: Optional[LabelEncoder] = None
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
        if known.any():
            le_labels = LabelEncoder().fit(labels_norm[known])
            y[known] = le_labels.transform(labels_norm[known])
            n_labels = int(y.max()) + 1
    else:
        y = None

    y_eff: Optional[np.ndarray] = y.copy() if y is not None else None
    w_sup: Optional[np.ndarray] = None
    if y is not None:
        w_sup = np.zeros(len(y), dtype=np.float32)
        w_sup[y >= 0] = 1.0
    do_pseudo = has_label_key and (y is not None) and np.any(y < 0) and n_labels > 0
    label_to_code = {label: i for i, label in enumerate(le_labels.classes_)} if le_labels else {}
    return SupervisionState(
        has_label_key=has_label_key,
        y=y,
        y_eff=y_eff,
        w_sup=w_sup,
        n_labels=n_labels,
        do_pseudo=do_pseudo,
        label_to_code=label_to_code,
        unknown_aliases=unknown_aliases,
    )


def _resolve_reference_mode(
    *,
    reference: str,
    batch_key: str,
    modality: str,
    supervision: SupervisionState,
    knobs: OTKnobs,
    align_reference: bool,
    b_raw: np.ndarray,
    le_batch: LabelEncoder,
) -> Tuple[str, int]:
    reference_norm = "auto" if reference is None else str(reference).lower()
    if reference_norm == "auto":
        reference_norm = "largest" if modality == "atac" else "union"
    is_semi = (
        supervision.y is not None
        and np.any(supervision.y >= 0)
        and np.any(supervision.y < 0)
    )
    force_union = (
        is_semi
        and knobs.supervision > 0
        and reference_norm != "union"
        and (not align_reference)
    )
    ref_mode = "union" if (reference_norm == "union" or force_union) else reference_norm

    if ref_mode == "largest":
        vals, counts = np.unique(b_raw, return_counts=True)
        ref_label_raw = vals[np.argmax(counts)]
        ref_label_enc = int(np.where(le_batch.classes_ == ref_label_raw)[0][0])
        return ref_mode, ref_label_enc
    if ref_mode == "union":
        return ref_mode, 0

    label_map = {str(label).lower(): label for label in le_batch.classes_}
    ref_label_raw = label_map.get(reference_norm)
    if ref_label_raw is not None:
        ref_label_enc = int(np.where(le_batch.classes_ == ref_label_raw)[0][0])
        return ref_mode, ref_label_enc
    try:
        return ref_mode, int(reference)
    except Exception as exc:
        available = ", ".join(str(lbl) for lbl in le_batch.classes_)
        raise ValueError(
            f"reference '{reference}' not found in batch_key '{batch_key}'. Available: {available}"
        ) from exc


def _prepare_alignment_masks(
    adata: Any,
    *,
    batch_key: str,
    label_key: Optional[str],
    unlabeled_category: Any,
    ref_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    return _resolve_reference_query_masks(
        adata,
        batch_key=batch_key,
        label_key=label_key,
        unlabeled_category=unlabeled_category,
        reference=ref_mode,
    )


def _run_prealign_if_needed(
    X0: np.ndarray,
    *,
    adata: Any,
    b: np.ndarray,
    align_reference: bool,
    batch_key: str,
    label_key: Optional[str],
    unlabeled_category: Any,
    ref_mode: str,
    ref_label_enc: int,
    prealign: Optional[str],
    prealign_strength: float,
    prealign_eps: float,
    prealign_max_points: int,
    random_state: int,
    apply_dims: Optional[slice],
    use_gpu: bool,
    gpu_device: int,
    verbose: bool,
    rare_mask: Optional[np.ndarray] = None,
    rare_scale: float = 0.25,
) -> PrealignState:
    prealign_norm = "none" if prealign is None else str(prealign).lower()
    prealign_strength_eff = float(prealign_strength)
    ref_mask: Optional[np.ndarray] = None
    query_mask: Optional[np.ndarray] = None
    if prealign_norm == "auto":
        auto_method = "ot"
        if align_reference:
            ref_mask, query_mask = _prepare_alignment_masks(
                adata,
                batch_key=batch_key,
                label_key=label_key,
                unlabeled_category=unlabeled_category,
                ref_mode=ref_mode,
            )
            b_auto = np.zeros(len(X0), dtype=np.int32)
            b_auto[query_mask] = 1
            batches_auto = b_auto
        else:
            batches_auto = b
        score = _auto_prealign_score(
            X0,
            batches_auto,
            k=_PREALIGN_AUTO_K,
            max_points=min(prealign_max_points, _PREALIGN_AUTO_MAX_POINTS),
            rng=random_state + 991,
            use_gpu=use_gpu,
            device=gpu_device,
            apply_dims=apply_dims,
        )
        prealign_strength_auto = _auto_prealign_strength(
            score,
            threshold=_PREALIGN_AUTO_THRESHOLD,
            strength_max=_PREALIGN_AUTO_STRENGTH_MAX,
        )
        if prealign_strength_auto <= 0.0:
            prealign_norm = "none"
            prealign_strength_eff = 0.0
        else:
            prealign_norm = auto_method
            prealign_strength_eff = prealign_strength_auto

    if prealign_norm in {"coral", "ot"}:
        if align_reference:
            if ref_mask is None or query_mask is None:
                ref_mask, query_mask = _prepare_alignment_masks(
                    adata,
                    batch_key=batch_key,
                    label_key=label_key,
                    unlabeled_category=unlabeled_category,
                    ref_mode=ref_mode,
                )
            b_align = np.zeros(len(X0), dtype=np.int32)
            b_align[query_mask] = 1
            fn = _coral_prealign if prealign_norm == "coral" else _ot_prealign
            X_in = X0
            X_pre = fn(
                X0,
                b_align,
                ref_label_enc=0,
                ref_mode="reference",
                strength=prealign_strength_eff,
                eps=prealign_eps,
                max_points=prealign_max_points,
                seed=random_state + 991,
                target="reference",
                apply_dims=apply_dims,
            )
            disp = X_pre - X_in
            if rare_mask is not None and np.any(rare_mask):
                disp[rare_mask] *= float(rare_scale)
                X0 = X_in + disp
            else:
                X0 = X_pre
            if verbose:
                name = "CORAL" if prealign_norm == "coral" else "OT-Gaussian"
                print(f"[prealign] {name} enabled target=reference strength={prealign_strength_eff}")
        else:
            fn = _coral_prealign if prealign_norm == "coral" else _ot_prealign
            X_in = X0
            X_pre = fn(
                X0,
                b,
                ref_label_enc=ref_label_enc,
                ref_mode=ref_mode,
                strength=prealign_strength_eff,
                eps=prealign_eps,
                max_points=prealign_max_points,
                seed=random_state + 991,
                target="auto",
                apply_dims=apply_dims,
            )
            disp = X_pre - X_in
            if rare_mask is not None and np.any(rare_mask):
                disp[rare_mask] *= float(rare_scale)
                X0 = X_in + disp
            else:
                X0 = X_pre
            if verbose:
                name = "CORAL" if prealign_norm == "coral" else "OT-Gaussian"
                print(f"[prealign] {name} enabled target=auto strength={prealign_strength_eff}")

    return PrealignState(
        X0=X0,
        prealign_norm=prealign_norm,
        prealign_strength=prealign_strength_eff,
        ref_mask=ref_mask,
        query_mask=query_mask,
    )


def _debatch_within_labels_shift(
    X: np.ndarray,
    b: np.ndarray,
    y_eff: np.ndarray,
    w_sup: Optional[np.ndarray],
    strength: float,
    min_cells: int = 30,
    label_strengths: Optional[Dict[int, float]] = None,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32, order="C")
    shift_out = np.zeros_like(X, dtype=np.float32)
    if strength <= 0 or y_eff is None:
        return shift_out
    labels = np.unique(y_eff[y_eff >= 0])
    if labels.size == 0:
        return shift_out
    for c in labels:
        c_int = int(c)
        strength_c = float(strength)
        if label_strengths is not None:
            strength_c = float(label_strengths.get(c_int, 0.0))
        if strength_c <= 0:
            continue
        label_mask = y_eff == c
        if int(label_mask.sum()) < int(min_cells):
            continue
        X_label = X[label_mask]
        if w_sup is None:
            w_label = np.ones(int(label_mask.sum()), dtype=np.float32)
        else:
            w_label = w_sup[label_mask].astype(np.float32, copy=False)
            if float(w_label.sum()) <= 0:
                w_label = np.ones_like(w_label, dtype=np.float32)
        w_sum = float(w_label.sum())
        w_norm = w_label / (w_sum + 1e-12)
        label_mean = (X_label * w_norm[:, None]).sum(axis=0)
        for batch in np.unique(b[label_mask]):
            batch_mask = label_mask & (b == batch)
            if not np.any(batch_mask):
                continue
            X_batch = X[batch_mask]
            if w_sup is None:
                w_batch = np.ones(int(batch_mask.sum()), dtype=np.float32)
            else:
                w_batch = w_sup[batch_mask].astype(np.float32, copy=False)
                if float(w_batch.sum()) <= 0:
                    w_batch = np.ones_like(w_batch, dtype=np.float32)
            w_batch_sum = float(w_batch.sum())
            w_batch_norm = w_batch / (w_batch_sum + 1e-12)
            batch_mean = (X_batch * w_batch_norm[:, None]).sum(axis=0)
            shift = strength_c * (label_mean - batch_mean)
            if w_sup is None:
                shift_out[batch_mask] = shift
            else:
                shift_out[batch_mask] = shift * w_batch[:, None]
    return shift_out.astype(X.dtype, copy=False)


def _debatch_within_labels_step(
    X: np.ndarray,
    b: np.ndarray,
    y_eff: np.ndarray,
    w_sup: Optional[np.ndarray],
    strength: float,
    min_cells: int = 30,
    label_strengths: Optional[Dict[int, float]] = None,
) -> np.ndarray:
    shift = _debatch_within_labels_shift(
        X,
        b,
        y_eff,
        w_sup,
        strength=strength,
        min_cells=min_cells,
        label_strengths=label_strengths,
    )
    return np.asarray(X, dtype=np.float32, order="C") + shift


def integrate_ot(
    adata: Any,
    obsm_key: str = "X_pca",
    batch_key: str = "batch",
    out_key: str = "scBIOT",
    strength: float = 0.5,
    conservation: float = 0.5,
    prototypes: float = 0.5,
    supervision: float = 0.5,
    approximate: bool = False,
    centroid: bool = False,
    reference: str = "auto",
    label_key: Optional[str] = None,
    unlabeled_category: Any = "unknown",
    use_gpu: bool = True,
    gpu_device: int = 0,
    ot_backend: str = "torch",
    random_state: int = 0,
    verbose: bool = True,
    modality: str = "auto",
    spatial_key: Optional[str] = None,
    spatial_weight: float = 0.5,
    time_key: Optional[str] = None,
    time_weight: float = 0.5,
    time_mode: str = "auto",
    prealign: Optional[str] = None,
    prealign_strength: float = 1.0,
    prealign_eps: float = 1e-3,
    prealign_max_points: int = 20000,
    max_iter: int = _DEFAULT_MAX_ITER,
    n_centroids_per_batch: int = 2048,
    max_samples_per_batch: int = 500_000,
    k_interp: int = 8,
    chunk_size: int = 500_000,
    tmp_path: Optional[str] = None,
    align_reference: bool = False,
    transfer_mode: str = "knn",
    _one_stage_supervised: bool = False,
) -> Tuple[Any, Dict[str, float | int]]:
    """
    **scBIOT** OT integration with semantic 0–1 knobs.

    Parameters
    ----------
    strength / conservation / prototypes / supervision
        Semantic 0–1 knobs controlling aggressiveness, structure preservation, capacity,
        and label supervision.
    approximate / centroid
        Switches for approximate Sinkhorn or centroid-level OT.
    obsm_key / batch_key / out_key / reference
        Keys and alignment semantics.
    align_reference
        When True, map query cells onto the reference subset (query→reference OT) and keep reference fixed.
        If `label_key` is provided (and exists in `adata.obs`), the reference/query split is inferred as:
        reference = labeled cells, query = `unlabeled_category` (plus NA). Otherwise, the split is inferred from
        `batch_key` and `reference`.
    max_iter
        Maximum number of outer optimization iterations.
    n_centroids_per_batch / max_samples_per_batch / k_interp / chunk_size / tmp_path
        Centroid-level OT controls forwarded to `integrate_centroids` when `centroid=True`.

    Examples
    --------
    RNA:
    >>> adata, metrics = integrate_ot(
    ...     adata,
    ...     obsm_key="X_pca",
    ...     batch_key="batch"
    ... )

    ATAC:
    >>> adata, metrics = integrate_ot(
    ...     adata,
    ...     obsm_key="X_lsi",
    ...     batch_key="batchname_all"
    ... )
    """
    knobs = OTKnobs(
        strength=strength,
        conservation=conservation,
        prototypes=prototypes,
        supervision=supervision,
        approximate=approximate,
        centroid=centroid,
    )
    transfer_mode = _validate_transfer_mode(transfer_mode)
    _one_stage_supervised = bool(_one_stage_supervised and label_key is not None)
    if knobs.centroid:
        if _one_stage_supervised:
            raise ValueError("Internal one-stage supervised OT is not supported for centroid=True.")
        if align_reference:
            raise ValueError("align_reference=True is not supported for centroid=True.")
        if spatial_key is not None or time_key is not None:
            # The memory-frugal centroid path re-normalizes the embedding internally,
            # and mixing absolute spatial coordinates into a cross-section/cross-stage
            # batch-correction embedding is not meaningful. Spatial/temporal geometry is
            # carried by ``tl.velocity_field_sb_centroids`` instead (per-stage local OT).
            import warnings

            warnings.warn(
                "spatial_key/time_key are ignored when centroid=True; the joint embedding "
                "uses expression geometry only. Apply spatial-temporal geometry via "
                "scbiot.velocity_field_sb_centroids on the integrated embedding.",
                RuntimeWarning,
                stacklevel=2,
            )
        from .integrate_centroids import integrate_centroids

        adata_out, metrics = integrate_centroids(
            adata,
            obsm_key=obsm_key,
            batch_key=batch_key,
            out_key=out_key,
            modality=modality,
            strength=knobs.strength,
            conservation=knobs.conservation,
            prototypes=knobs.prototypes,
            supervision=knobs.supervision,
            approximate=knobs.approximate,
            reference=reference,
            label_key=label_key,
            unlabeled_category=unlabeled_category,
            use_gpu=use_gpu,
            gpu_device=gpu_device,
            ot_backend=ot_backend,
            random_state=random_state,
            max_iter=max_iter,
            n_centroids_per_batch=n_centroids_per_batch,
            max_samples_per_batch=max_samples_per_batch,
            k_interp=k_interp,
            chunk_size=chunk_size,
            tmp_path=tmp_path,
            verbose=verbose,
        )
        return adata_out, metrics

    if align_reference and label_key is not None and label_key not in adata.obs:
        raise KeyError(f"label_key '{label_key}' not found in adata.obs.")
    if align_reference and modality == "auto" and str(obsm_key).lower() in {"x_pca", "x_spca"}:
        modality = "rna"

    prepared = _prepare_embedding_and_batches(
        adata,
        obsm_key=obsm_key,
        batch_key=batch_key,
        modality=modality,
        spatial_key=spatial_key,
        spatial_weight=spatial_weight,
        time_key=time_key,
        time_weight=time_weight,
        time_mode=time_mode,
        preserve_input_embedding=align_reference,
    )
    X0 = prepared.X0
    b_raw = prepared.b_raw
    b = prepared.b
    le_batch = prepared.le_batch
    modality = prepared.modality
    apply_dims = prepared.apply_dims
    n_classes = prepared.n_classes

    ot_iters = 200 if knobs.approximate else 1000
    ot_tol = 1e-4 if knobs.approximate else 1e-6

    policy = derive_geometry_policy(
        modality=modality,
        n_obs=prepared.n_obs,
        med_batch=prepared.med_batch,
        knobs=knobs,
    )
    K_ref = int(policy.proto_ref)
    K_batch = int(policy.proto_batch)
    reg = float(policy.reg)
    reg_m = float(policy.reg_m)
    cost_clip_q = float(policy.cost_clip_q)
    geom_budget = float(policy.geom_budget)
    geom_budget_q90 = float(policy.geom_budget_q90)
    diffusion_alpha = float(policy.diffusion_alpha)
    diffusion_steps = int(policy.diffusion_steps)
    rare_quantile = float(policy.rare_quantile)
    lam_sup = float(policy.lam_sup)
    lam_repulse = float(policy.lam_repulse)
    k_local = _DEFAULT_K_LOCAL
    k_eval = _DEFAULT_K_EVAL
    eval_subsample = _DEFAULT_EVAL_SUBSAMPLE
    trust_subsample = _DEFAULT_TRUST_SUBSAMPLE
    max_iter = int(max_iter)
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")
    patience = 4 if _one_stage_supervised else _DEFAULT_PATIENCE
    tol = _DEFAULT_TOL
    bridge_base = 0.35
    bridge_damp = float(np.clip(bridge_base * (1.15 - 0.3 * knobs.strength), 0.1, 0.6))

    supervision_state = _prepare_supervision(
        adata,
        label_key=label_key,
        unlabeled_category=unlabeled_category,
    )
    has_label_key = supervision_state.has_label_key
    y = supervision_state.y
    y_eff = supervision_state.y_eff
    w_sup = supervision_state.w_sup
    n_labels = supervision_state.n_labels
    do_pseudo = supervision_state.do_pseudo
    label_to_code = supervision_state.label_to_code
    unknown_aliases = supervision_state.unknown_aliases

    if not has_label_key:
        reg = min(reg, 0.018)
        reg_m = min(reg_m, 0.25)
        cost_clip_q = min(cost_clip_q, 0.80)
    clip_big = 50.0
    if not has_label_key:
        clip_big = min(clip_big, 20.0)

    ref_mode, ref_label_enc = _resolve_reference_mode(
        reference=reference,
        batch_key=batch_key,
        modality=modality,
        supervision=supervision_state,
        knobs=knobs,
        align_reference=align_reference,
        b_raw=b_raw,
        le_batch=le_batch,
    )

    d_pre, _ = _knn_graph(
        X0,
        k=max(15, k_local + 5),
        use_gpu=use_gpu,
        device=gpu_device,
    )
    knn_mean_dist_pre = d_pre.mean(axis=1) if d_pre.size else np.zeros(len(X0), dtype=X0.dtype)
    prealign_rare_mask = _batchwise_rare_mask(knn_mean_dist_pre, b, rare_quantile)

    align_reference_rep_key = str(obsm_key).lower()
    use_pca_query_mapping = bool(
        align_reference and align_reference_rep_key in _ALIGN_REFERENCE_REP_KEYS
    )
    prealign_for_run = prealign
    prealign_strength_for_run = float(prealign_strength)
    if use_pca_query_mapping:
        prealign_norm_requested = "none" if prealign is None else str(prealign).lower()
        if prealign_norm_requested == "auto":
            prealign_for_run = "ot"
            prealign_strength_for_run = _ALIGN_REFERENCE_PCA_PREALIGN_STRENGTH

    prealign_state = _run_prealign_if_needed(
        X0,
        adata=adata,
        b=b,
        align_reference=align_reference,
        batch_key=batch_key,
        label_key=label_key,
        unlabeled_category=unlabeled_category,
        ref_mode=ref_mode,
        ref_label_enc=ref_label_enc,
        prealign=prealign_for_run,
        prealign_strength=prealign_strength_for_run,
        prealign_eps=prealign_eps,
        prealign_max_points=prealign_max_points,
        random_state=random_state,
        apply_dims=apply_dims,
        use_gpu=use_gpu,
        gpu_device=gpu_device,
        verbose=verbose,
        rare_mask=prealign_rare_mask,
        rare_scale=_ALIGN_REFERENCE_PCA_RARE_SCALE if use_pca_query_mapping else 0.25,
    )
    X0 = prealign_state.X0
    prealign_norm = prealign_state.prealign_norm
    ref_mask = prealign_state.ref_mask
    query_mask = prealign_state.query_mask

    if align_reference:
        if prealign_norm not in {"coral", "ot"}:
            ref_mask, query_mask = _resolve_reference_query_masks(
                adata,
                batch_key=batch_key,
                label_key=label_key,
                unlabeled_category=unlabeled_category,
                reference=ref_mode,  # resolved (union/largest/specific)
            )
        ref_emb = np.asarray(X0[ref_mask], dtype=np.float32)
        query_emb = np.asarray(X0[query_mask], dtype=np.float32)
        if ref_emb.size == 0 or query_emb.size == 0:
            raise ValueError("Reference/query subset empty; check label_key/unlabeled_category/reference.")
        if ref_emb.shape[1] != query_emb.shape[1]:
            raise ValueError("Reference/query embedding dimensionality mismatch.")

        X_out = X0.copy()
        mapping_mode = "query_to_reference_prealign"
        align_topk = 0
        transport: Dict[str, Any] = {
            "indices": np.empty((int(query_mask.sum()), 0), dtype=np.int32),
            "weights": np.empty((int(query_mask.sum()), 0), dtype=np.float32),
            "residual": None,
        }
        if not use_pca_query_mapping:
            align_topk = int(min(64, max(1, ref_emb.shape[0])))
            align_chunk = 1024
            aligned, transport = compute_ot_alignment(
                query_emb,
                ref_emb,
                reg=reg,
                reg_m=reg_m,
                cost_clip_q=cost_clip_q,
                clip_big=clip_big,
                backend=ot_backend,
                iters=ot_iters,
                tol=ot_tol,
                use_gpu=use_gpu,
                gpu_device=gpu_device,
                transport_topk=align_topk,
                chunk_size=align_chunk,
            )
            X_out[query_mask] = aligned.astype(X_out.dtype, copy=False)
            mapping_mode = "query_to_reference_ot"

        X_out[ref_mask] = np.asarray(adata.obsm[obsm_key], dtype=np.float32)[ref_mask]
        input_shape = tuple(np.asarray(adata.obsm[obsm_key]).shape)
        if tuple(X_out.shape) != input_shape:
            raise ValueError(
                "align_reference=True requires output embedding shape to match "
                f"adata.obsm[{obsm_key!r}]; got {X_out.shape} and {input_shape}."
            )
        if not np.all(np.isfinite(X_out[query_mask])):
            raise ValueError("Query-to-reference OT mapping produced non-finite query coordinates.")
        adata.obsm[out_key] = X_out

        n_obs_all = int(X_out.shape[0])
        k_eval_eff = min(k_eval, max(1, n_obs_all - 1))
        d0, idx0 = _knn_graph(X0, k=max(15, k_local + 5), use_gpu=use_gpu, device=gpu_device)
        diag = _embedding_diagnostics(
            X0,
            X_out,
            b,
            idx0,
            d0,
            k_eval=k_eval_eff,
            eval_subsample=eval_subsample,
            trust_subsample=trust_subsample,
            random_state=random_state,
            use_gpu=use_gpu,
            gpu_device=gpu_device,
            include_tw=True,
        )

        metrics = dict(
            mix=float(diag["mix"]),
            overlap0=float(diag["overlap0"]),
            strain=float(diag["strain"]),
            tw=float(diag["tw"]),
            it=0,
        )
        adata.uns["_ot_alignment"] = {
            "indices": transport["indices"].astype(np.int32, copy=False),
            "weights": transport["weights"].astype(np.float32, copy=False),
            "residual": (
                transport.get("residual").astype(np.float32, copy=False)
                if transport.get("residual") is not None
                else None
            ),
            "rna_obs": np.asarray(adata.obs_names[ref_mask], dtype=object),
            "query_obs": np.asarray(adata.obs_names[query_mask], dtype=object),
            "input_rep_key": obsm_key,
            "output_rep_key": out_key,
            "reference_kept_fixed": True,
            "params": {
                "reg": float(reg),
                "reg_m": float(reg_m),
                "cost_clip_q": float(cost_clip_q),
                "clip_big": float(clip_big),
                "backend": ot_backend,
                "topk": align_topk,
                "mode": mapping_mode,
                "prealign": prealign_norm,
                "prealign_strength": float(prealign_state.prealign_strength),
            },
            "center": ref_emb.mean(axis=0, keepdims=True).astype(np.float32, copy=False),
            "scale": ref_emb.std(axis=0, keepdims=True).astype(np.float32, copy=False),
        }
        adata.uns["_supbiot"] = {
            "batch_key": batch_key,
            "rep_key": out_key,
            "input_rep_key": obsm_key,
            "modality": modality,
            "label_key": label_key,
            "unlabeled_category": unlabeled_category,
            "align_reference": True,
            "reference_kept_fixed": True,
        }
        if verbose:
            print(
                f"[align_reference] reference={int(ref_mask.sum())} query={int(query_mask.sum())} "
                f"input={obsm_key!r} output={out_key!r} reference_kept_fixed=True"
            )
            print(
                f"[align_reference] mode={mapping_mode} prealign={prealign_norm} "
                f"prealign_strength={float(prealign_state.prealign_strength):.3f}"
            )
            print(
                f"[align_reference] mix={metrics['mix']:.3f} overlap0={metrics['overlap0']:.3f} "
                f"strain={metrics['strain']:.5f} tw={metrics['tw']:.3f}"
            )
        return adata, metrics

    X = X0.copy()

    d0, idx0 = _knn_graph(X0, k=max(15, k_local + 5), use_gpu=use_gpu, device=gpu_device)
    knn_mean_dist0 = d0.mean(axis=1) if d0.size else np.zeros(len(X), dtype=X.dtype)
    dens0 = (knn_mean_dist0 - (knn_mean_dist0.min() if len(knn_mean_dist0) else 0.0)) / (
        _ptp(knn_mean_dist0) + 1e-12
    )
    diffusion = build_diffusion_operator(X0, idx0, d0)
    rare_mask = _batchwise_rare_mask(knn_mean_dist0, b, rare_quantile)
    small_class_mask = _small_class_mask(y_eff, n_labels)
    if small_class_mask is not None:
        rare_mask = rare_mask | small_class_mask

    H0_i, _ = _neighbor_batch_entropy_per_cell(
        X0, b, k=min(15, k_eval), use_gpu=use_gpu, device=gpu_device
    )
    H0_max = np.log(n_classes + 1e-12)
    H0_norm = np.clip(H0_i / (H0_max + 1e-12), 0.0, 1.0)
    bridge_score = 0.5 * H0_norm + 0.5 * dens0

    mix0 = _neighbor_batch_entropy_per_cell(
        X0, b, k=k_eval, use_gpu=use_gpu, device=gpu_device
    )[1]
    strain0 = _graph_strain(X0, idx0, d0, use_gpu=use_gpu, device=gpu_device)
    label_purity0 = (
        _label_knn_purity(
            X0,
            y_eff,
            w_sup,
            k=k_eval,
            use_gpu=use_gpu,
            gpu_device=gpu_device,
        )
        if _one_stage_supervised
        else 1.0
    )
    best = dict(
        X=X.copy(),
        mix=mix0,
        overlap0=1.0,
        strain=strain0,
        label_purity=label_purity0,
        score=_selection_score(
            batch_mix=float(mix0),
            label_purity=float(label_purity0),
            overlap0=1.0,
            strain=float(strain0),
        ),
        it=0,
        eta=0.0,
        local_distortion=0.0,
        local_distortion_q90=0.0,
        feasible=True,
        constrained_step=False,
    )
    mix_current = float(mix0)

    if verbose:
        backend = "FAISS-GPU" if (_FAISS_AVAIL and use_gpu and _FAISS_GPU) else (
            "FAISS-CPU" if _FAISS_AVAIL else "sklearn"
        )
        print(f"[baseline] KNN backend={backend} mix={mix0:.4f} strain={strain0:.5f}")

    no_imp = 0
    early_stop = False
    for it in range(1, max_iter + 1):
        t = (it - 1) / max(1, max_iter - 1)

        if do_pseudo and (it % _PSEUDO_CFG.every_iters == 0):
            pred_label, pred_conf = predict_pseudo_labels(
                adata,
                rep=X,
                label_key=label_key,
                unlabeled_category=unknown_aliases,
                pred_label_key="pred_cell_type",
                pred_conf_key="pred_confidence",
                min_conf=0.0,
                return_numpy=True,
                inplace=False,
                max_ref=_PSEUDO_CFG.max_ref,
                use_gpu=use_gpu,
                gpu_device=gpu_device,
                transfer_mode=transfer_mode,
            )

            pred_label_arr = np.asarray(pred_label, dtype=object)
            pred_conf_arr = np.asarray(pred_conf, dtype=np.float32)
            pred_norm = np.char.lower(np.char.strip(pred_label_arr.astype(str)))
            pred_codes = np.array([label_to_code.get(lbl, -1) for lbl in pred_norm], dtype=int)

            if y_eff is not None and w_sup is not None:
                prev_w_sup = w_sup.copy()
                y_eff = y.copy()
                w_sup = np.zeros(len(y), dtype=np.float32)
                w_sup[y >= 0] = 1.0

                conf_mask = pred_conf_arr >= _PSEUDO_CFG.min_conf
                pseudo_mask = (y < 0) & conf_mask & (pred_codes >= 0)
                y_eff[pseudo_mask] = pred_codes[pseudo_mask]
                if pseudo_mask.any():
                    w_sup[pseudo_mask] = (
                        (1.0 - _PSEUDO_CFG.ema) * prev_w_sup[pseudo_mask]
                        + _PSEUDO_CFG.ema * pred_conf_arr[pseudo_mask]
                    )
                w_sup = np.clip(w_sup, 0.0, 1.0, out=w_sup)
                w_sup[y >= 0] = 1.0

                pseudo_mask = (y < 0) & (y_eff >= 0)
                if pseudo_mask.any():
                    keep = np.zeros_like(pseudo_mask)
                    for cls in np.unique(y_eff[pseudo_mask]):
                        cls_mask = pseudo_mask & (y_eff == cls)
                        if int(cls_mask.sum()) < _PSEUDO_CFG.min_count:
                            continue
                        if np.unique(b[cls_mask]).size < _PSEUDO_CFG.min_batches:
                            continue
                        keep[cls_mask] = True
                    drop = pseudo_mask & ~keep
                    if drop.any():
                        y_eff[drop] = -1
                        w_sup[drop] = 0.0

        if ref_mode == "union":
            R, packs, _ = _compute_prototypes_union(
                X, b, K_ref, K_batch, random_state + it, use_gpu=use_gpu, device=gpu_device, y=y_eff
            )
        else:
            R, packs, _ = _compute_prototypes(
                X, b, ref_label_enc, K_ref, K_batch, random_state + it, use_gpu=use_gpu, device=gpu_device, y=y_eff
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

            if len(idx) == 0 or len(Bi) == 0:
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
                clip_big=clip_big,
                ot_backend=ot_backend,
                iters=ot_iters,
                tol=ot_tol,
                use_gpu=use_gpu,
                gpu_device=gpu_device,
            )
            disp_proto = Bi_to_R - Bi
            norm_move = np.linalg.norm(disp_proto, axis=1)
            s_dist = 1.0 / (1.0 + (norm_move / (norm_move.std() + 1e-8)))
            alpha_i = s_dist[nn_idx] * (1.0 - bridge_damp * bridge_score[idx])
            alpha[idx] = alpha_i.astype(X.dtype, copy=False)
            Xi = X[idx]
            if len(idx) > _AFFINE_FIT_MAX:
                rng_aff = np.random.default_rng(random_state + it * 31)
                sub_idx = rng_aff.choice(len(idx), size=_AFFINE_FIT_MAX, replace=False)
                Xi_fit = Xi[sub_idx]
                tgt_fit = Xi_fit + disp_proto[nn_idx[sub_idx]]
            else:
                Xi_fit = Xi
                tgt_fit = Xi + disp_proto[nn_idx]
            A, t_vec = _fit_affine(Xi_fit, tgt_fit)
            Xi_aff = Xi @ A + t_vec
            shift[idx] = (Xi_aff - Xi).astype(X.dtype, copy=False)

        shift = alpha[:, None] * shift

        rare_scale = 0.25 if has_label_key else 1.00
        shift[rare_mask] *= rare_scale

        # # ---- NEW: supervised pull/repulse boosts class compactness (optional) ----
        if (y_eff is not None) and np.any(y_eff >= 0) and (lam_sup > 0.0 or lam_repulse > 0.0):
            C = _class_means(X, y_eff, n_labels, use_gpu=use_gpu, device=gpu_device)
            known = (y_eff >= 0)
            if w_sup is not None:
                known &= w_sup > 0
            # only apply where class mean is valid
            known &= ~np.isnan(C[np.clip(y_eff, 0, n_labels - 1)]).any(axis=1)
            if known.any():
                if lam_sup > 0.0:
                    pull_vec = (C[y_eff[known]] - X[known]) * (1.0 - 0.20 * bridge_score[known])[:, None]
                    if w_sup is not None:
                        pull_vec = w_sup[known][:, None] * pull_vec
                    shift[known] += lam_sup * pull_vec
                if lam_repulse > 0.0 and n_labels > 1:
                    near_other = _nearest_other_class_index(
                        X,
                        y_eff,
                        C,
                        use_gpu=use_gpu,
                        device=gpu_device,
                    )
                    ok = known & (near_other >= 0)
                    if ok.any():
                        repulse_vec = (X[ok] - C[near_other[ok]]) * (1.0 - 0.20 * bridge_score[ok])[:, None]
                        if w_sup is not None:
                            repulse_vec = w_sup[ok][:, None] * repulse_vec
                        shift[ok] += lam_repulse * repulse_vec

        

        if diffusion_alpha > 0.0 and diffusion_steps > 0:
            shift = diffusion_smooth(
                shift,
                diffusion,
                alpha=diffusion_alpha,
                steps=diffusion_steps,
                mask=~rare_mask,
            )

        deb_strength = _PSEUDO_CFG.debatch_max * (t ** _PSEUDO_CFG.debatch_ramp_power)
        if _one_stage_supervised:
            deb_strength = _PSEUDO_CFG.debatch_max
        if (
            _one_stage_supervised
            and has_label_key
            and (y_eff is not None)
            and np.any(y_eff >= 0)
            and deb_strength > 0
        ):
            label_entropy = _label_batch_entropy(
                X,
                b,
                y_eff,
                k=k_eval,
                min_cells=_PSEUDO_CFG.min_count,
                use_gpu=use_gpu,
                device=gpu_device,
            )
            label_strengths = _adaptive_label_debatch_strengths(
                label_entropy,
                max_strength=float(_PSEUDO_CFG.debatch_max),
            )
            shift += _debatch_within_labels_shift(
                X,
                b,
                y_eff,
                w_sup,
                strength=deb_strength,
                min_cells=_PSEUDO_CFG.min_count,
                label_strengths=label_strengths,
            )

        if has_label_key:
            shift *= (1.0 - 0.12 * bridge_score)[:, None]
        shift = adaptive_trust_move(
            X,
            shift,
            idx0,
            d0,
            step_size=1.0,
        )
        Xcand, step_info = apply_geometry_budgeted_ot_step(
            X,
            shift,
            idx0,
            d0,
            geom_budget=geom_budget,
            geom_budget_q90=geom_budget_q90,
        )
        Xgeom = Xcand.copy()

        if (
            (not _one_stage_supervised)
            and has_label_key
            and (y_eff is not None)
            and np.any(y_eff >= 0)
            and deb_strength > 0
        ):
            Xcand = _debatch_within_labels_step(
                Xcand,
                b,
                y_eff,
                w_sup,
                strength=deb_strength,
                min_cells=_PSEUDO_CFG.min_count,
            )

        local_distortion_c, per_cell_distortion_c = relative_local_distortion(Xcand, idx0, d0)
        local_distortion_q90_c = float(np.quantile(per_cell_distortion_c, 0.90)) if per_cell_distortion_c.size else 0.0
        if local_distortion_c > geom_budget or local_distortion_q90_c > geom_budget_q90:
            Xcand = Xgeom
            local_distortion_c = float(step_info["local_distortion"])
            local_distortion_q90_c = float(step_info["local_distortion_q90"])
        step_info = {
            **step_info,
            "local_distortion": float(local_distortion_c),
            "local_distortion_q90": float(local_distortion_q90_c),
            "feasible": bool(local_distortion_c <= geom_budget and local_distortion_q90_c <= geom_budget_q90),
            "constrained_step": bool(local_distortion_c > geom_budget or local_distortion_q90_c > geom_budget_q90),
        }

        cand_diag = _embedding_diagnostics(
            X0,
            Xcand,
            b,
            idx0,
            d0,
            k_eval=k_eval,
            eval_subsample=eval_subsample,
            trust_subsample=trust_subsample,
            random_state=random_state + it,
            use_gpu=use_gpu,
            gpu_device=gpu_device,
            include_tw=False,
        )
        mix_c = float(cand_diag["mix"])
        overlap0 = float(cand_diag["overlap0"])
        strain_c = float(cand_diag["strain"])
        if _one_stage_supervised:
            label_purity_base = _label_knn_purity(
                X0,
                y_eff,
                w_sup,
                k=k_eval,
                use_gpu=use_gpu,
                gpu_device=gpu_device,
            )
            label_purity_c = _label_knn_purity(
                Xcand,
                y_eff,
                w_sup,
                k=k_eval,
                use_gpu=use_gpu,
                gpu_device=gpu_device,
            )
            label_guard_ok = bool(label_purity_c + 0.01 >= label_purity_base)
            score_c = _selection_score(
                batch_mix=mix_c,
                label_purity=label_purity_c,
                overlap0=overlap0,
                strain=strain_c,
            )
        else:
            label_purity_c = 1.0
            label_guard_ok = True
            score_c = float(mix_c)

        mix_delta = float(mix_c - mix_current)
        accepted = should_accept_step(mix_delta, step_info, tol=tol) and label_guard_ok
        if accepted:
            X = Xcand
            mix_current = float(mix_c)
            if _one_stage_supervised:
                improved = bool(
                    score_c > float(best["score"]) + tol
                    or (
                        np.isclose(score_c, float(best["score"]), atol=tol)
                        and mix_c > float(best["mix"]) + tol
                    )
                    or (
                        np.isclose(score_c, float(best["score"]), atol=tol)
                        and np.isclose(mix_c, float(best["mix"]), atol=tol)
                        and float(step_info["local_distortion"]) < float(best["local_distortion"])
                    )
                )
            else:
                improved = bool(
                    mix_c > best["mix"] + tol
                    or (
                        np.isclose(mix_c, best["mix"], atol=tol)
                        and float(step_info["local_distortion"]) < float(best["local_distortion"])
                    )
                )
            if improved:
                best.update(
                    X=X.copy(),
                    mix=float(mix_c),
                    overlap0=float(overlap0),
                    strain=float(strain_c),
                    label_purity=float(label_purity_c),
                    score=float(score_c),
                    it=it,
                    eta=float(step_info["eta"]),
                    local_distortion=float(step_info["local_distortion"]),
                    local_distortion_q90=float(step_info["local_distortion_q90"]),
                    feasible=bool(step_info["feasible"]),
                    constrained_step=bool(step_info["constrained_step"]),
                )
                no_imp = 0
            else:
                no_imp += 1
        else:
            no_imp += 1

        if verbose:
            print(
                _format_iteration_log(
                    it=it,
                    eta=float(step_info["eta"]),
                    mix=float(mix_c),
                    local_distortion=float(step_info["local_distortion"]),
                    local_distortion_q90=float(step_info["local_distortion_q90"]),
                    feasible=bool(step_info["feasible"]),
                    constrained_step=bool(step_info["constrained_step"]),
                    accepted=accepted,
                    overlap0=float(overlap0),
                    strain=float(strain_c),
                    best_it=int(best["it"]),
                )
            )

        if bool(step_info["constrained_step"]) and float(step_info["eta"]) <= _BACKTRACK_CFG.eta_stop:
            early_stop = True
            break

        if no_imp >= patience:
            early_stop = True
            break

    X_best = best["X"]
    final_diag = _embedding_diagnostics(
        X0,
        X_best,
        b,
        idx0,
        d0,
        k_eval=k_eval,
        eval_subsample=eval_subsample,
        trust_subsample=trust_subsample,
        random_state=random_state,
        use_gpu=use_gpu,
        gpu_device=gpu_device,
        include_tw=True,
    )
    tw = float(final_diag["tw"])
    adata.obsm[out_key] = X_best
    metrics = dict(
        mix=best["mix"],
        overlap0=best["overlap0"],
        strain=best["strain"],
        tw=tw,
        it=best["it"],
    )
    if _one_stage_supervised:
        metrics["label_purity"] = float(best["label_purity"])
        metrics["score"] = float(best["score"])
    if verbose and early_stop:
        print("\n[early stop] plateau reached.")
    if verbose:
        print(
            f"[final] it*={int(best['it'])} mix={float(best['mix']):.3f} "
            f"overlap0={float(best['overlap0']):.3f} strain={float(best['strain']):.5f} tw={tw:.3f}"
        )

    return adata, metrics
