# =================== integrate.py: unified pipeline (rare-protection + supervised option) ===================
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder

from ..utils.coral_helpers import _coral_prealign, _ot_prealign
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
    _resolve_reference_query_masks,
    _smooth_by_knn,
    _trustworthiness_score,
)
from ..utils.ot_projector import laplacian_from_knn, train_projector
from ..utils.torch_utils import _torch_device
from ..utils.ot_transport import compute_ot_alignment
from .supbiot import predict_pseudo_labels

# -------------------- Main integration (now supervised-ready) --------------------

_PSEUDO_EVERY_ITERS = 1
_PSEUDO_MIN_CONF = 0.80
_PSEUDO_MIN_BATCHES = 2
_PSEUDO_MIN_COUNT = 30
_PSEUDO_EMA = 0.30
_DEBATCH_MAX = 0.80
_DEBATCH_RAMP_POWER = 2.0
_PSEUDO_TOPK = 64
_PSEUDO_MAX_REF = 20000
_DEFAULT_K_LOCAL = 15
_DEFAULT_K_EVAL = 30
_DEFAULT_EVAL_SUBSAMPLE = 5000
_DEFAULT_TRUST_SUBSAMPLE = 2500
_DEFAULT_MAX_ITER = 15
_DEFAULT_PATIENCE = 3
_DEFAULT_TOL = 1e-3
_DEFAULT_POSTSCALE = True

DEFAULT_MAX_ITER = _DEFAULT_MAX_ITER

_ANCHOR: Dict[str, Dict[str, float]] = {
    "rna": {
        "anchor_n": 32000,
        "anchor_med_batch": 2000,
        "K_ref": 1024,
        "K_batch": 512,
        "reg": 0.03,
        "reg_m": 0.40,
        "sharpen": 0.15,
        "K_pseudo": 24,
        "pull": 0.75,
        "push": 0.30,
        "lambda0_hi": 0.50,
        "lambda0_lo": 0.35,
        "smin_bulk": 1.55,
        "smax_bulk": 1.65,
        "smin_bridge": 1.15,
        "smax_bridge": 1.25,
        "max_step_local": 1.0,
        "step_lo": 0.75,
        "step_hi": 0.95,
        "q_start": 0.80,
        "q_end": 0.90,
        "overlap0_lo": 0.60,
        "overlap0_hi": 0.70,
        "w_overlap": 0.20,
        "w_strain": 1.0,
        "penalty_gamma": 1.5,
        "projector_strength": 0.5,
    },
    "atac": {
        "anchor_n": 85000,
        "anchor_med_batch": 7700,
        "K_ref": 960,
        "K_batch": 360,
        "reg": 0.036,
        "reg_m": 0.30,
        "sharpen": 0.12,
        "K_pseudo": 20,
        "pull": 0.72,
        "push": 0.22,
        "lambda0_hi": 0.58,
        "lambda0_lo": 0.42,
        "smin_bulk": 0.78,
        "smax_bulk": 1.50,
        "smin_bridge": 0.90,
        "smax_bridge": 1.16,
        "max_step_local": 0.90,
        "step_lo": 0.70,
        "step_hi": 0.88,
        "q_start": 0.78,
        "q_end": 0.885,
        "overlap0_lo": 0.66,
        "overlap0_hi": 0.75,
        "w_overlap": 0.22,
        "w_strain": 1.0,
        "penalty_gamma": 1.30,
        "projector_strength": 0.20,
    },
}


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return float(np.log(p / (1.0 - p)))


def _sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-z)))


def _logit_affine(x: float, anchor: float, s: float) -> float:
    return _sigmoid(_logit(anchor) + float(s) * (_clip01(x) - 0.5))


def _exp2_affine(x: float, anchor: float, s: float) -> float:
    return float(anchor) * float(2.0 ** (float(s) * (_clip01(x) - 0.5)))


def _int_exp2_affine(x: float, anchor: float, s: float, lo: int, hi: int) -> int:
    value = int(np.round(_exp2_affine(x, anchor, s)))
    return int(np.clip(value, lo, hi))


def _ensure_gap(lo: float, hi: float, gap: float, lo_bound: float = 0.0, hi_bound: float = 1.0) -> Tuple[float, float]:
    if hi < lo + gap:
        hi = min(hi_bound, lo + gap)
        if hi <= lo:
            lo = max(lo_bound, hi - gap)
    return lo, hi


@dataclass
class OTKnobs:
    strength: float = 0.5
    conservation: float = 0.5
    prototypes: float = 0.5
    sharpen: float = 0.5
    supervision: float = 0.5
    projector: float = 0.5
    approximate: bool = False
    centroid: bool = False

    def __post_init__(self) -> None:
        self.strength = _clip01(self.strength)
        self.conservation = _clip01(self.conservation)
        self.prototypes = _clip01(self.prototypes)
        self.sharpen = _clip01(self.sharpen)
        self.supervision = _clip01(self.supervision)
        self.projector = _clip01(self.projector)
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


def _derive_params(modality: str, n_obs: int, med_batch: int, knobs: OTKnobs) -> Dict[str, float | int]:
    anchor = _ANCHOR[modality]
    n_obs = int(max(1, n_obs))
    med_batch = int(max(1, med_batch))
    sf_n = float(np.sqrt(n_obs / float(anchor["anchor_n"])))
    sf_b = float(np.sqrt(med_batch / float(anchor["anchor_med_batch"])))

    params: Dict[str, float | int] = {}
    params["reg"] = float(anchor["reg"])
    params["reg_m"] = float(anchor["reg_m"])

    k_ref_anchor = float(anchor["K_ref"]) * sf_n
    k_batch_anchor = float(anchor["K_batch"]) * sf_b
    params["K_ref"] = _int_exp2_affine(knobs.prototypes, k_ref_anchor, s=1.0, lo=64, hi=4096)
    params["K_batch"] = _int_exp2_affine(knobs.prototypes, k_batch_anchor, s=1.0, lo=32, hi=2048)

    k_pseudo_anchor = float(anchor["K_pseudo"]) * sf_n
    params["K_pseudo"] = int(
        np.clip(np.round(_exp2_affine(knobs.prototypes, k_pseudo_anchor, s=0.35)), 16, 64)
    )

    params["sharpen"] = _exp2_affine(knobs.sharpen, float(anchor["sharpen"]), s=0.8)
    pull = _logit_affine(knobs.sharpen, float(anchor["pull"]), s=1.0)
    ratio = float(anchor["push"]) / float(anchor["pull"])
    push = float(np.clip(pull * ratio, 0.05, 0.95))
    params["pull"] = pull
    params["push"] = push

    params["lambda0_hi"] = _logit_affine(knobs.conservation, float(anchor["lambda0_hi"]), s=1.0)
    params["lambda0_lo"] = _logit_affine(knobs.conservation, float(anchor["lambda0_lo"]), s=1.0)
    lo, hi = _ensure_gap(params["lambda0_lo"], params["lambda0_hi"], 0.01, 0.0, 1.0)
    params["lambda0_lo"] = lo
    params["lambda0_hi"] = hi

    params["overlap0_lo"] = _logit_affine(knobs.conservation, float(anchor["overlap0_lo"]), s=1.2)
    params["overlap0_hi"] = _logit_affine(knobs.conservation, float(anchor["overlap0_hi"]), s=1.2)
    lo, hi = _ensure_gap(params["overlap0_lo"], params["overlap0_hi"], 0.02, 0.0, 1.0)
    params["overlap0_lo"] = lo
    params["overlap0_hi"] = hi

    params["penalty_gamma"] = _exp2_affine(knobs.conservation, float(anchor["penalty_gamma"]), s=0.8)
    params["w_strain"] = _exp2_affine(knobs.conservation, float(anchor["w_strain"]), s=0.4)
    params["w_overlap"] = float(anchor["w_overlap"])

    params["step_lo"] = _logit_affine(knobs.strength, float(anchor["step_lo"]), s=1.0)
    params["step_hi"] = _logit_affine(knobs.strength, float(anchor["step_hi"]), s=1.0)
    lo, hi = _ensure_gap(params["step_lo"], params["step_hi"], 0.02, 0.0, 1.0)
    params["step_lo"] = lo
    params["step_hi"] = hi
    params["max_step_local"] = _exp2_affine(knobs.strength, float(anchor["max_step_local"]), s=0.6)

    for tag in ("bulk", "bridge"):
        smin = float(anchor[f"smin_{tag}"])
        smax = float(anchor[f"smax_{tag}"])
        mid = float(np.sqrt(smin * smax))
        ratio = float(smax / max(smin, 1e-8))
        mid = _exp2_affine(knobs.strength, mid, s=0.2)
        ratio = _exp2_affine(knobs.strength, ratio, s=0.4)
        ratio = max(ratio, 1.01)
        smin_new = mid / np.sqrt(ratio)
        smax_new = mid * np.sqrt(ratio)
        if smax_new < smin_new + 0.02:
            smax_new = smin_new + 0.02
        params[f"smin_{tag}"] = smin_new
        params[f"smax_{tag}"] = smax_new

    params["q_start"] = _logit_affine(knobs.conservation, float(anchor["q_start"]), s=-0.8)
    params["q_end"] = _logit_affine(knobs.conservation, float(anchor["q_end"]), s=-0.8)
    lo, hi = _ensure_gap(params["q_start"], params["q_end"], 0.01, 0.0, 1.0)
    params["q_start"] = lo
    params["q_end"] = hi

    params["lam_sup"] = 0.8 * _clip01(knobs.supervision)
    params["lam_repulse"] = 0.25 * _clip01(knobs.supervision)

    params["projector_strength"] = float(
        np.clip(_logit_affine(knobs.projector, float(anchor["projector_strength"]), s=1.2), 0.0, 1.0)
    )

    return params


def _debatch_within_labels_step(
    X: np.ndarray,
    b: np.ndarray,
    y_eff: np.ndarray,
    w_sup: Optional[np.ndarray],
    strength: float,
    min_cells: int = 30,
) -> np.ndarray:
    if strength <= 0 or y_eff is None:
        return X
    X_out = np.asarray(X, dtype=np.float32, order="C").copy()
    labels = np.unique(y_eff[y_eff >= 0])
    if labels.size == 0:
        return X_out
    for c in labels:
        label_mask = y_eff == c
        if int(label_mask.sum()) < int(min_cells):
            continue
        X_label = X_out[label_mask]
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
            X_batch = X_out[batch_mask]
            if w_sup is None:
                w_batch = np.ones(int(batch_mask.sum()), dtype=np.float32)
            else:
                w_batch = w_sup[batch_mask].astype(np.float32, copy=False)
                if float(w_batch.sum()) <= 0:
                    w_batch = np.ones_like(w_batch, dtype=np.float32)
            w_batch_sum = float(w_batch.sum())
            w_batch_norm = w_batch / (w_batch_sum + 1e-12)
            batch_mean = (X_batch * w_batch_norm[:, None]).sum(axis=0)
            shift = strength * (label_mean - batch_mean)
            if w_sup is None:
                X_out[batch_mask] = X_batch + shift
            else:
                X_out[batch_mask] = X_batch + shift * w_batch[:, None]
    return X_out

def integrate_ot(
    adata: Any,
    obsm_key: str = "X_pca",
    batch_key: str = "batch",
    out_key: str = "scBIOT",
    strength: float = 0.5,
    conservation: float = 0.5,
    prototypes: float = 0.5,
    sharpen: float = 0.5,
    supervision: float = 0.5,
    projector: float = 0.5,
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
) -> Tuple[Any, Dict[str, float | int]]:
    """
    **scBIOT** OT integration with semantic 0–1 knobs.

    Parameters
    ----------
    strength / conservation / prototypes / sharpen / supervision / projector
        Semantic 0–1 knobs controlling aggressiveness, structure preservation, capacity,
        sharpening, label supervision, and projector strength.
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
    ...     batch_key="batch",
    ...     reference="union",
    ... )

    ATAC:
    >>> adata, metrics = integrate_ot(
    ...     adata,
    ...     obsm_key="X_lsi",
    ...     batch_key="batchname_all",
    ...     reference="largest",
    ... )
    """
    knobs = OTKnobs(
        strength=strength,
        conservation=conservation,
        prototypes=prototypes,
        sharpen=sharpen,
        supervision=supervision,
        projector=projector,
        approximate=approximate,
        centroid=centroid,
    )
    if knobs.centroid:
        if align_reference:
            raise ValueError("align_reference=True is not supported for centroid=True.")
        from .integrate_centroids import integrate_centroids

        return integrate_centroids(
            adata,
            obsm_key=obsm_key,
            batch_key=batch_key,
            out_key=out_key,
            modality=modality,
            strength=knobs.strength,
            conservation=knobs.conservation,
            prototypes=knobs.prototypes,
            sharpen=knobs.sharpen,
            supervision=knobs.supervision,
            projector=knobs.projector,
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

    if obsm_key not in adata.obsm:
        raise KeyError(f"Embedding '{obsm_key}' not found in adata.obsm.")

    X0_raw = _as_nd_f32_c(adata.obsm[obsm_key])
    if modality == "auto" and hasattr(adata, "uns") and isinstance(getattr(adata, "uns"), dict):
        for key in ("scbiot_modality", "modality"):
            value = str(adata.uns.get(key, "")).lower()
            if value in _ANCHOR:
                modality = value
                break
    modality = _infer_modality(modality, obsm_key, X0_raw)
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

    ot_iters = 200 if knobs.approximate else 1000
    ot_tol = 1e-4 if knobs.approximate else 1e-6


    b_raw = adata.obs[batch_key].to_numpy()
    le_batch = LabelEncoder()
    b = le_batch.fit_transform(b_raw).astype(np.int32, copy=False)
    n_classes = max(1, int(b.max()) + 1)
    n_obs = int(X0.shape[0])
    counts = np.bincount(b, minlength=n_classes)
    med_batch = int(np.median(counts)) if counts.size else n_obs

    params = _derive_params(modality, n_obs, med_batch, knobs)
    K_ref = int(params["K_ref"])
    K_batch = int(params["K_batch"])
    reg = float(params["reg"])
    reg_m = float(params["reg_m"])
    sharpen = float(params["sharpen"])
    K_pseudo = int(params["K_pseudo"])
    pull = float(params["pull"])
    push = float(params["push"])
    lambda0_hi = float(params["lambda0_hi"])
    lambda0_lo = float(params["lambda0_lo"])
    smin_bulk = float(params["smin_bulk"])
    smax_bulk = float(params["smax_bulk"])
    smin_bridge = float(params["smin_bridge"])
    smax_bridge = float(params["smax_bridge"])
    max_step_local = float(params["max_step_local"])
    step_lo = float(params["step_lo"])
    step_hi = float(params["step_hi"])
    q_start = float(params["q_start"])
    q_end = float(params["q_end"])
    overlap0_lo = float(params["overlap0_lo"])
    overlap0_hi = float(params["overlap0_hi"])
    w_overlap = float(params["w_overlap"])
    penalty_gamma = float(params["penalty_gamma"])
    w_strain = float(params["w_strain"])
    lam_sup = float(params["lam_sup"])
    lam_repulse = float(params["lam_repulse"])
    projector_strength = float(params["projector_strength"])
    k_local = _DEFAULT_K_LOCAL
    k_eval = _DEFAULT_K_EVAL
    eval_subsample = _DEFAULT_EVAL_SUBSAMPLE
    trust_subsample = _DEFAULT_TRUST_SUBSAMPLE
    max_iter = int(max_iter)
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")
    patience = _DEFAULT_PATIENCE
    tol = _DEFAULT_TOL
    postscale = _DEFAULT_POSTSCALE
    bridge_base = 0.35
    bridge_damp = float(np.clip(bridge_base * (1.15 - 0.3 * knobs.strength), 0.1, 0.6))

    # optional supervised labels
    y: Optional[np.ndarray]
    le_labels: Optional[LabelEncoder] = None
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

    y_eff: Optional[np.ndarray] = y.copy() if y is not None else None
    w_sup: Optional[np.ndarray] = None
    if y is not None:
        w_sup = np.zeros(len(y), dtype=np.float32)
        w_sup[y >= 0] = 1.0

    do_pseudo = has_label_key and (y is not None) and np.any(y < 0) and n_labels > 0
    label_to_code = {label: i for i, label in enumerate(le_labels.classes_)} if le_labels else {}

    reference_norm = "auto" if reference is None else str(reference).lower()
    if reference_norm == "auto":
        reference_norm = "largest" if modality == "atac" else "union"
    is_semi = (y is not None) and np.any(y >= 0) and np.any(y < 0)   # has Unknowns
    force_union = (is_semi and knobs.supervision > 0 and reference_norm != "union" and (not align_reference))

    ref_mode = "union" if (reference_norm == "union" or force_union) else reference_norm

    if ref_mode == "largest":
        vals, counts = np.unique(b_raw, return_counts=True)
        ref_label_raw = vals[np.argmax(counts)]
        ref_label_enc = int(np.where(le_batch.classes_ == ref_label_raw)[0][0])
    elif ref_mode == "union":
        ref_label_enc = 0
    else:
        label_map = {str(label).lower(): label for label in le_batch.classes_}
        ref_label_raw = label_map.get(reference_norm)
        if ref_label_raw is not None:
            ref_label_enc = int(np.where(le_batch.classes_ == ref_label_raw)[0][0])
        else:
            # if integer was passed, use it; else fail fast
            try:
                ref_label_enc = int(reference)
            except Exception as exc:
                available = ", ".join(str(lbl) for lbl in le_batch.classes_)
                raise ValueError(
                    f"reference '{reference}' not found in batch_key '{batch_key}'. "
                    f"Available: {available}"
                ) from exc

        # --- CORAL pre-align: force overlap for disjoint batches ---

    prealign_norm = "none" if prealign is None else str(prealign).lower()
    if prealign_norm == "coral":
        if align_reference:
            ref_mask, query_mask = _resolve_reference_query_masks(
                adata,
                batch_key=batch_key,
                label_key=label_key,
                unlabeled_category=unlabeled_category,
                reference=ref_mode,  # resolved (union/largest/specific)
            )
            b_coral = np.zeros(len(X0), dtype=np.int32)
            b_coral[query_mask] = 1
            X0 = _coral_prealign(
                X0,
                b_coral,
                ref_label_enc=0,
                ref_mode="reference",
                strength=prealign_strength,
                eps=prealign_eps,
                max_points=prealign_max_points,
                seed=random_state + 991,
                target="reference",
                apply_dims=apply_dims,
            )
            if verbose:
                print(f"[prealign] CORAL enabled target=reference strength={prealign_strength}")
        else:
            X0 = _coral_prealign(
                X0,
                b,
                ref_label_enc=ref_label_enc,
                ref_mode=ref_mode,
                strength=prealign_strength,
                eps=prealign_eps,
                max_points=prealign_max_points,
                seed=random_state + 991,
                target="auto",
                apply_dims=apply_dims,
            )
            if verbose:
                print(f"[prealign] CORAL enabled target=auto strength={prealign_strength}")
    elif prealign_norm == "ot":
        if align_reference:
            ref_mask, query_mask = _resolve_reference_query_masks(
                adata,
                batch_key=batch_key,
                label_key=label_key,
                unlabeled_category=unlabeled_category,
                reference=ref_mode,  # resolved (union/largest/specific)
            )
            b_ot = np.zeros(len(X0), dtype=np.int32)
            b_ot[query_mask] = 1
            X0 = _ot_prealign(
                X0,
                b_ot,
                ref_label_enc=0,
                ref_mode="reference",
                strength=prealign_strength,
                eps=prealign_eps,
                max_points=prealign_max_points,
                seed=random_state + 991,
                target="reference",
                apply_dims=apply_dims,
            )
            if verbose:
                print(f"[prealign] OT-Gaussian enabled target=reference strength={prealign_strength}")
        else:
            X0 = _ot_prealign(
                X0,
                b,
                ref_label_enc=ref_label_enc,
                ref_mode=ref_mode,
                strength=prealign_strength,
                eps=prealign_eps,
                max_points=prealign_max_points,
                seed=random_state + 991,
                target="auto",
                apply_dims=apply_dims,
            )
            if verbose:
                print(f"[prealign] OT-Gaussian enabled target=auto strength={prealign_strength}")

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

        align_topk = int(min(64, max(1, ref_emb.shape[0])))
        align_chunk = 1024
        aligned, transport = compute_ot_alignment(
            query_emb,
            ref_emb,
            reg=reg,
            reg_m=reg_m,
            cost_clip_q=float(q_end),
            clip_big=50.0,
            backend=ot_backend,
            iters=ot_iters,
            tol=ot_tol,
            use_gpu=use_gpu,
            gpu_device=gpu_device,
            transport_topk=align_topk,
            chunk_size=align_chunk,
        )

        X_out = X0.copy()
        X_out[query_mask] = aligned.astype(X_out.dtype, copy=False)
        adata.obsm[out_key] = X_out

        n_obs_all = int(X_out.shape[0])
        k_eval_eff = min(k_eval, max(1, n_obs_all - 1))
        mix = _neighbor_batch_entropy_per_cell(
            X_out, b, k=k_eval_eff, use_gpu=use_gpu, device=gpu_device
        )[1]
        overlap0 = _knn_overlap(
            X0,
            X_out,
            k=k_eval_eff,
            subsample=eval_subsample,
            rng=random_state,
            use_gpu=use_gpu,
            device=gpu_device,
        )
        d0, idx0 = _knn_graph(X0, k=max(15, k_local + 5), use_gpu=use_gpu, device=gpu_device)
        strain = _graph_strain(X_out, idx0, d0, use_gpu=use_gpu, device=gpu_device)

        n_obs_eval = len(X_out)
        if trust_subsample is not None and n_obs_eval > trust_subsample:
            rng_tw = np.random.default_rng(random_state)
            sample_tw = rng_tw.choice(n_obs_eval, size=trust_subsample, replace=False)
            X0_eval = X0[sample_tw]
            X_out_eval = X_out[sample_tw]
        elif eval_subsample is not None and n_obs_eval > eval_subsample:
            rng_tw = np.random.default_rng(random_state + 17)
            sample_tw = rng_tw.choice(n_obs_eval, size=eval_subsample, replace=False)
            X0_eval = X0[sample_tw]
            X_out_eval = X_out[sample_tw]
        else:
            X0_eval = X0
            X_out_eval = X_out

        k_tw = min(k_eval, max(1, len(X_out_eval) - 1))
        tw = _trustworthiness_score(
            X0_eval,
            X_out_eval,
            n_neighbors=k_tw,
            use_gpu=use_gpu,
            gpu_device=gpu_device,
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
                "mode": "full_ot",
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

        metrics = dict(mix=mix, overlap0=overlap0, strain=strain, tw=tw, it=0)
        if verbose:
            print(f"[align_reference] mix={mix:.3f} overlap0={overlap0:.3f} strain={strain:.5f} tw={tw:.3f}")
        return adata, metrics

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
    strain0 = _graph_strain(X0, idx0, d0, use_gpu=use_gpu, device=gpu_device)
    best = dict(J=-np.inf, X=X.copy(), mix=mix0, overlap0=1.0, strain=strain0, it=0)

    if verbose:
        backend = "FAISS-GPU" if (_FAISS_AVAIL and use_gpu and _FAISS_GPU) else (
            "FAISS-CPU" if _FAISS_AVAIL else "sklearn"
        )
        print(f"[baseline] KNN backend={backend} mix={mix0:.4f} strain={strain0:.5f}")

    if force_union and verbose:
        if has_label_key and knobs.supervision > 0 and reference_norm != "union":
            print("[ot] supervision enabled -> using reference='union' for stability")

    proj_strength = float(projector_strength)
    proj_strength_eff = proj_strength * 10.0
    k_pair = int(np.clip(int(round(5 + 20 * proj_strength)), 5, 30))
    proj_max_pairs = 2_000_000
    k_lap = int(min(30, max(15, k_local)))

    no_imp = 0
    for it in range(1, max_iter + 1):
        t = (it - 1) / max(1, max_iter - 1)
        lambda_graph0 = _lerp(lambda0_hi, lambda0_lo, t)
        overlap0_floor = _lerp(overlap0_lo, overlap0_hi, t)
        step = _lerp(step_lo, step_hi, t)
        cost_clip_q = _lerp(q_start, q_end, t)

        if do_pseudo and (it % _PSEUDO_EVERY_ITERS == 0):
            pred_label, pred_conf = predict_pseudo_labels(
                adata,
                rep=X,
                label_key=label_key,
                unlabeled_category=unknown_aliases,
                min_conf=0.0,
                return_numpy=True,
                inplace=False,
                max_ref=_PSEUDO_MAX_REF,
                topk=_PSEUDO_TOPK,
                use_gpu=use_gpu,
                gpu_device=gpu_device,
                reg=reg,
                reg_m=reg_m,
                cost_clip_q=float(cost_clip_q),
                ot_backend=ot_backend,
                ot_iters=ot_iters,
                ot_tol=ot_tol,
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

                conf_mask = pred_conf_arr >= _PSEUDO_MIN_CONF
                pseudo_mask = (y < 0) & conf_mask & (pred_codes >= 0)
                y_eff[pseudo_mask] = pred_codes[pseudo_mask]
                if pseudo_mask.any():
                    w_sup[pseudo_mask] = (
                        (1.0 - _PSEUDO_EMA) * prev_w_sup[pseudo_mask] + _PSEUDO_EMA * pred_conf_arr[pseudo_mask]
                    )
                w_sup = np.clip(w_sup, 0.0, 1.0, out=w_sup)
                w_sup[y >= 0] = 1.0

                pseudo_mask = (y < 0) & (y_eff >= 0)
                if pseudo_mask.any():
                    keep = np.zeros_like(pseudo_mask)
                    for cls in np.unique(y_eff[pseudo_mask]):
                        cls_mask = pseudo_mask & (y_eff == cls)
                        if int(cls_mask.sum()) < _PSEUDO_MIN_COUNT:
                            continue
                        if np.unique(b[cls_mask]).size < _PSEUDO_MIN_BATCHES:
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

        do_projector = (proj_strength > 0.0) and (it >= 2 or best["overlap0"] >= overlap0_lo)

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

        

        # rare-friendly smoothing: avoid oversmoothing sparse islands
        dens0_q = np.quantile(dens0, 0.85) if len(dens0) else 1.0
        rare_mask = dens0 >= dens0_q
        if lambda_graph0 > 0 and idx0.size > 0:
            shift_sm = _smooth_by_knn(
                shift,
                idx0,
                lam=lambda_graph0,
                use_gpu=use_gpu,
                device=gpu_device,
            )
            shift[~rare_mask] = shift_sm[~rare_mask]

        shift *= (1.0 - 0.12 * bridge_score)[:, None]
        move = _cap_step_local(
            step * shift,
            knn_mean_dist0,
            max_step_local=max_step_local,
            use_gpu=use_gpu,
            device=gpu_device,
        )
        move = _guard_edge_stretch_weighted(
            X,
            move,
            idx0,
            d0,
            smin_i,
            smax_i,
            rounds=2,
            use_gpu=use_gpu,
            device=gpu_device,
        )

        Xcand = X + move
        if postscale:
            Xcand = (Xcand - Xcand.mean(0)) * (sd0 / (Xcand.std(0) + 1e-8)) + mu0

        deb_strength = _DEBATCH_MAX * (t ** _DEBATCH_RAMP_POWER)
        if has_label_key and (y_eff is not None) and np.any(y_eff >= 0) and deb_strength > 0:
            Xcand = _debatch_within_labels_step(
                Xcand,
                b,
                y_eff,
                w_sup,
                strength=deb_strength,
                min_cells=_PSEUDO_MIN_COUNT,
            )

        if do_projector and k_pair > 0:
            d_pair, idx_pair = _knn_graph(Xcand, k=k_pair, use_gpu=use_gpu, device=gpu_device)
            if idx_pair.size:
                src = np.repeat(np.arange(len(Xcand)), idx_pair.shape[1])
                dst = idx_pair.reshape(-1)
                dist = d_pair.reshape(-1)
                cross = b[src] != b[dst]
                if y_eff is not None and knobs.supervision > 0:
                    same_label = (y_eff[src] == y_eff[dst]) & (y_eff[src] >= 0)
                    cross &= same_label
                src = src[cross]
                dst = dst[cross]
                dist = dist[cross]

                if src.size:
                    tau = float(np.median(dist)) if dist.size else 1.0
                    w = np.exp(-dist / (tau + 1e-12)).astype(np.float32, copy=False)
                    w_sum = np.zeros(len(Xcand), dtype=np.float32)
                    np.add.at(w_sum, src, w)
                    w = w / (w_sum[src] + 1e-12)

                    src_sym = np.concatenate([src, dst]).astype(np.int64, copy=False)
                    dst_sym = np.concatenate([dst, src]).astype(np.int64, copy=False)
                    w_sym = np.concatenate([w, w]).astype(np.float32, copy=False)
                    src, dst, w = src_sym, dst_sym, w_sym

                    if src.size > proj_max_pairs:
                        rng = np.random.default_rng(random_state + it * 37)
                        keep = rng.choice(src.size, size=proj_max_pairs, replace=False)
                        src = src[keep]
                        dst = dst[keep]
                        w = w[keep]

                    rows = []
                    cols = []
                    data = []
                    for lbl in np.unique(b):
                        idx_b = np.where(b == lbl)[0]
                        if len(idx_b) <= 1:
                            continue
                        Lb = laplacian_from_knn(
                            Xcand[idx_b],
                            k=k_lap,
                            backend="faiss" if use_gpu else "sklearn",
                            sym=True,
                        )
                        if Lb.nnz == 0:
                            continue
                        coo = Lb.tocoo()
                        rows.append(idx_b[coo.row])
                        cols.append(idx_b[coo.col])
                        data.append(coo.data)

                    if rows:
                        row = np.concatenate(rows)
                        col = np.concatenate(cols)
                        dat = np.concatenate(data)
                        L = sp.csr_matrix((dat, (row, col)), shape=(len(Xcand), len(Xcand)), dtype=np.float32)
                    else:
                        L = sp.csr_matrix((len(Xcand), len(Xcand)), dtype=np.float32)

                    epochs = 1 if it < max(2, max_iter // 2) else 2
                    device = _torch_device(use_gpu, gpu_device)
                    Xcand = train_projector(
                        Xcand,
                        b,
                        (src, dst, w),
                        L,
                        strength=proj_strength_eff,
                        device=device,
                        epochs=epochs,
                        pair_batch=65536,
                        seed=random_state + it * 19,
                        label_codes=y_eff,
                        weights=w_sup,
                    )

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
        strain_c = _graph_strain(Xcand, idx0, d0, use_gpu=use_gpu, device=gpu_device)

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
    metrics = dict(mix=best["mix"], overlap0=best["overlap0"], strain=best["strain"], tw=tw, it=best["it"])
    if label_key is None or label_key not in adata.obs:
        if verbose:
            print("[label transfer] skipped; pass label_key to compute alignment metadata")
    else:
        ref_mask, query_mask = _resolve_reference_query_masks(
            adata,
            batch_key=batch_key,
            label_key=label_key,
            unlabeled_category=unlabeled_category,
            reference=ref_mode,  # keep consistent with reference semantics
        )
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
            iters=ot_iters,
            tol=ot_tol,
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

    return adata, metrics
