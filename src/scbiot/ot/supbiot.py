# =================== supbiot.py: label transfer ===================
from __future__ import annotations
from typing import Any, Optional, Sequence, Tuple

import difflib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

from ..utils.ot_helpers import _FAISS_AVAIL, _faiss_knn_search

LOGREG_DEFAULT_CLASS_WEIGHT = "balanced"
LOGREG_DEFAULT_C = 5.0
LOGREG_DEFAULT_SOLVER = "lbfgs"
LOGREG_DEFAULT_PENALTY = "l2"
LOGREG_DEFAULT_MULTI_CLASS = "multinomial"
LOGREG_DEFAULT_MAX_ITER = 3000
LOGREG_DEFAULT_PRIOR_ALPHA = 0.0
LOGREG_DEFAULT_STANDARDIZE = True
# Two-representation blend: the validated label-transfer workflow averages the primary
# rep's class scores with those from the supervised autoencoder embedding ``X_ae`` at a
# fixed weight ``(1 - w) * primary + w * X_ae``. This is the default; it gracefully
# no-ops when ``input_rep_key`` is absent from ``adata.obsm`` or equals the primary rep,
# so calls without an ``X_ae`` embedding behave as single-rep transfer.
SUPBIOT_DEFAULT_INPUT_REP_KEY = "X_ae"
SUPBIOT_DEFAULT_INPUT_REP_WEIGHT = 0.5
# Reference-prototype calibration: blend in the cosine similarity of each query cell to
# the labeled reference class centroids (softmax at a fixed temperature) to stabilize
# confusable class boundaries. On by default at a small fixed weight; disable with
# ``prototype_weight = 0``.
SUPBIOT_DEFAULT_PROTOTYPE_WEIGHT = 0.1
SUPBIOT_PROTOTYPE_TEMPERATURE = 0.1


def _prototype_scores(
    *,
    ref_emb: np.ndarray,
    ref_codes: np.ndarray,
    query_emb: np.ndarray,
    n_classes: int,
    temperature: float = SUPBIOT_PROTOTYPE_TEMPERATURE,
) -> np.ndarray:
    """Reference class-prototype probabilities for the query cells.

    Reference and query are standardized with reference statistics, class centroids
    are L2-normalized, and each (L2-normalized) query cell's cosine similarity to the
    centroids is turned into a probability via a fixed-temperature softmax.
    """
    if ref_emb.shape[0] == 0 or query_emb.shape[0] == 0:
        return np.zeros((query_emb.shape[0], n_classes), dtype=np.float32)
    ref_std, query_std = _standardize_ref_query(ref_emb, query_emb)
    centroids = np.zeros((n_classes, ref_std.shape[1]), dtype=np.float32)
    for c in range(n_classes):
        mask = ref_codes == c
        if np.any(mask):
            centroids[c] = ref_std[mask].mean(axis=0)
    centroids = _l2_normalize_rows(centroids)
    sims = _l2_normalize_rows(query_std) @ centroids.T
    logits = sims / max(float(temperature), 1e-6)
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits.astype(np.float64))
    return _normalize_rows(probs).astype(np.float32)


def _resolve_ref_query_masks(
    adata: Any,
    *,
    label_key: str,
    unlabeled_category: str | Sequence[str] | set[str] | None,
) -> Tuple[pd.Series, np.ndarray, np.ndarray]:
    if label_key not in adata.obs:
        raise KeyError(f"{label_key!r} not found in adata.obs")
    labels = adata.obs[label_key]
    if unlabeled_category is None:
        query_mask = labels.isna()
    elif isinstance(unlabeled_category, (list, tuple, set)):
        query_mask = labels.isna() | labels.isin(unlabeled_category)
    else:
        query_mask = labels.isna() | labels.eq(unlabeled_category)
    if int(query_mask.sum()) == 0:
        hint = ""
        available_labels = []
        if hasattr(labels, "cat"):
            available_labels = list(labels.cat.categories)
        else:
            available_labels = list(labels.dropna().unique())
        if isinstance(unlabeled_category, str) and available_labels:
            matches = difflib.get_close_matches(
                unlabeled_category,
                [str(label) for label in available_labels],
                n=1,
                cutoff=0.6,
            )
            if matches:
                hint = f" Did you mean {matches[0]!r}?"
        if available_labels:
            preview = ", ".join(repr(label) for label in available_labels[:10])
            suffix = "..." if len(available_labels) > 10 else ""
            hint = f"{hint} Available labels include: {preview}{suffix}"
        raise ValueError(f"Query subset empty; check unlabeled_category.{hint}")
    ref_mask = ~query_mask
    if int(ref_mask.sum()) == 0:
        raise ValueError("Reference subset empty; check unlabeled_category.")
    return labels, np.asarray(ref_mask), np.asarray(query_mask)


def _restrict_query_mask(
    adata: Any,
    query_mask: np.ndarray,
    *,
    query_group_key: Optional[str],
    query_group_value: str,
) -> np.ndarray:
    if query_group_key is None or query_group_key not in adata.obs:
        return np.asarray(query_mask, dtype=bool)
    group_mask = adata.obs[query_group_key].astype(str).eq(str(query_group_value)).to_numpy()
    filtered = np.asarray(query_mask, dtype=bool) & group_mask
    if int(filtered.sum()) == 0:
        return np.asarray(query_mask, dtype=bool)
    return filtered


def _normalize_rows(scores: np.ndarray) -> np.ndarray:
    scores = np.maximum(scores, 0.0)
    row_sum = scores.sum(axis=1, keepdims=True)
    return np.divide(
        scores,
        np.where(row_sum > 0, row_sum, 1.0),
        out=np.zeros_like(scores),
        where=row_sum > 0,
    )


def _standardize_ref_query(ref_emb: np.ndarray, query_emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature z-score using reference statistics (applied to ref and query)."""
    mu = ref_emb.mean(axis=0, keepdims=True)
    sd = ref_emb.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (
        ((ref_emb - mu) / sd).astype(np.float32, copy=False),
        ((query_emb - mu) / sd).astype(np.float32, copy=False),
    )


def _resolve_input_rep_key(
    adata: Any,
    *,
    primary_key: Optional[str],
    input_rep_key: Optional[str],
) -> list[str]:
    """Resolve the secondary embedding(s) to blend into the logreg scores.

    Returns ``[]`` (no blend) unless the caller explicitly names ``input_rep_key``.
    When it is given and present in ``adata.obsm``, the resolved list also includes
    its stability-pass siblings ``{input_rep_key}_e1``, ``{input_rep_key}_e2``, ...
    when they exist, so their label-transfer probabilities are averaged together.
    No embedding is searched for implicitly.
    """
    if input_rep_key is None:
        return []
    obsm = getattr(adata, "obsm", {})
    if input_rep_key not in obsm or input_rep_key == primary_key:
        return []
    keys = [input_rep_key]
    i = 1
    while f"{input_rep_key}_e{i}" in obsm and f"{input_rep_key}_e{i}" != primary_key:
        keys.append(f"{input_rep_key}_e{i}")
        i += 1
    return keys


def _encode_labels_for_transfer(labels_ref: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    labels_cat = labels_ref.astype("category")
    classes = labels_cat.cat.categories.to_numpy(dtype=object)
    codes = labels_cat.cat.codes.to_numpy(dtype=np.int32)
    valid = codes >= 0
    if not np.all(valid):
        raise ValueError("Reference labels contain invalid categories after encoding.")
    return classes, codes


def _select_stratified_reference_subset(
    labels_ref: pd.Series | Sequence[Any],
    *,
    max_ref: int | None,
    random_state: int = 0,
) -> np.ndarray:
    """
    Select a deterministic capped reference subset while preserving label coverage.

    Rare classes are retained first whenever the budget permits, and the remaining
    slots are distributed roughly in proportion to class size.
    """
    labels_s = pd.Series(labels_ref, copy=False).astype("string").fillna("__missing__")
    n_ref = int(labels_s.shape[0])
    if max_ref is None or n_ref <= int(max_ref):
        return np.arange(n_ref, dtype=np.int64)

    budget = max(0, int(max_ref))
    if budget == 0 or n_ref == 0:
        return np.empty(0, dtype=np.int64)

    classes, inverse = np.unique(labels_s.to_numpy(dtype=object), return_inverse=True)
    counts = np.bincount(inverse, minlength=classes.size).astype(np.int64, copy=False)
    class_indices = [np.flatnonzero(inverse == class_id) for class_id in range(classes.size)]
    allocations = np.zeros(classes.size, dtype=np.int64)

    if budget >= classes.size:
        allocations = np.minimum(counts, 1)
        remaining_budget = int(budget - allocations.sum())
    else:
        priority = np.lexsort((classes.astype(str), counts))
        allocations[priority[:budget]] = 1
        remaining_budget = 0

    if remaining_budget > 0:
        capacity = counts - allocations
        eligible = np.flatnonzero(capacity > 0)
        if eligible.size:
            eligible_capacity = capacity[eligible].astype(np.float64, copy=False)
            scaled = remaining_budget * eligible_capacity / eligible_capacity.sum()
            extra = np.minimum(np.floor(scaled).astype(np.int64), capacity[eligible])
            allocations[eligible] += extra
            remaining_budget -= int(extra.sum())

            if remaining_budget > 0:
                remainders = scaled - np.floor(scaled)
                refill_order = eligible[
                    np.lexsort(
                        (
                            classes[eligible].astype(str),
                            counts[eligible],
                            -remainders,
                        )
                    )
                ]
                for class_id in refill_order:
                    if remaining_budget == 0:
                        break
                    if allocations[class_id] >= counts[class_id]:
                        continue
                    allocations[class_id] += 1
                    remaining_budget -= 1

    rng = np.random.default_rng(int(random_state))
    selected_parts: list[np.ndarray] = []
    for class_id, class_count in enumerate(counts):
        take = int(min(allocations[class_id], class_count))
        if take <= 0:
            continue
        members = class_indices[class_id]
        if members.size <= take:
            selected_parts.append(members)
            continue
        chosen = members[np.sort(rng.permutation(members.size)[:take])]
        selected_parts.append(np.sort(chosen))

    if not selected_parts:
        return np.empty(0, dtype=np.int64)
    return np.sort(np.concatenate(selected_parts))
def _scores_from_knn(
    *,
    ref_emb: np.ndarray,
    ref_codes: np.ndarray,
    query_emb: np.ndarray,
    n_classes: int,
    k: int,
    metric: str,
    use_gpu: bool,
    gpu_device: int,
) -> np.ndarray:
    if ref_emb.shape[0] == 0:
        return np.zeros((query_emb.shape[0], n_classes), dtype=np.float32)
    dist, idx = _knn_search(
        query_emb,
        ref_emb,
        k=k,
        metric=metric,
        use_gpu=use_gpu,
        gpu_device=gpu_device,
    )
    k_eff = int(idx.shape[1])
    tau = float(np.median(dist)) if dist.size else 1.0
    if tau <= 0 or not np.isfinite(tau):
        tau = 1.0
    w = np.exp(-dist / tau).astype(np.float32, copy=False)
    cls = ref_codes[idx]

    out = np.zeros((query_emb.shape[0], n_classes), dtype=np.float32)
    rows = np.arange(query_emb.shape[0])
    for j in range(k_eff):
        np.add.at(out, (rows, cls[:, j]), w[:, j])
    return _normalize_rows(out)


def _as_f32_c(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32, order="C")


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    arr = _as_f32_c(x)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return np.asarray(arr / norms, dtype=np.float32, order="C")


def _knn_search(
    query_emb: np.ndarray,
    ref_emb: np.ndarray,
    *,
    k: int,
    metric: str,
    use_gpu: bool,
    gpu_device: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_ref = int(ref_emb.shape[0])
    n_query = int(query_emb.shape[0])
    if n_ref == 0 or n_query == 0:
        return (
            np.zeros((n_query, 0), dtype=np.float32),
            np.zeros((n_query, 0), dtype=np.int64),
        )

    k_eff = int(max(1, min(int(k), n_ref)))
    metric_name = str(metric).strip().lower()
    query_arr = _as_f32_c(query_emb)
    ref_arr = _as_f32_c(ref_emb)

    if _FAISS_AVAIL and metric_name in {"euclidean", "l2", "cosine"}:
        if metric_name == "cosine":
            search_query = _l2_normalize_rows(query_arr)
            search_ref = _l2_normalize_rows(ref_arr)
            dist_sq, idx = _faiss_knn_search(
                search_query,
                search_ref,
                k_eff,
                use_gpu=use_gpu,
                device=gpu_device,
            )
            dist = 0.5 * np.maximum(dist_sq, 0.0)
        else:
            dist_sq, idx = _faiss_knn_search(
                query_arr,
                ref_arr,
                k_eff,
                use_gpu=use_gpu,
                device=gpu_device,
            )
            dist = np.sqrt(np.maximum(dist_sq, 0.0))
        return (
            np.asarray(dist, dtype=np.float32, order="C"),
            np.asarray(idx, dtype=np.int64, order="C"),
        )

    nn = NearestNeighbors(n_neighbors=k_eff, metric=metric_name, n_jobs=-1).fit(ref_arr)
    dist, idx = nn.kneighbors(query_arr, n_neighbors=k_eff, return_distance=True)
    return (
        np.asarray(dist, dtype=np.float32, order="C"),
        np.asarray(idx, dtype=np.int64, order="C"),
    )
def _validate_transfer_mode(mode: str) -> str:
    normalized = (mode or "logreg").strip().lower()
    if normalized not in {"knn", "logreg"}:
        raise ValueError("transfer_mode must be one of ['knn', 'logreg'].")
    return normalized


_SUPPORTED_TRANSFER_KWARGS = {
    "max_ref",
    "topk",
    "use_gpu",
    "gpu_device",
    "knn_k",
    "knn_metric",
    "knn_tissue_key",
    "knn_weight_mode",
    "knn_prior_alpha",
    "knn_graph_k",
    "knn_diffuse_alpha",
    "knn_diffuse_steps",
    "knn_diffuse_group_key",
    "query_group_key",
    "query_group_value",
    "logreg_tissue_key",
    "logreg_min_cells",
    "logreg_min_classes",
    "logreg_class_weight",
    "logreg_C",
    "logreg_solver",
    "logreg_penalty",
    "logreg_multi_class",
    "logreg_max_iter",
    "logreg_prior_alpha",
    "logreg_standardize",
    "input_rep_key",
    "input_rep_weight",
    "prototype_weight",
    "random_state",
}


def _raise_on_unexpected_transfer_kwargs(kwargs: dict[str, Any]) -> None:
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword arguments: {unexpected}")


def _pop_predict_pseudo_label_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    predict_kwargs = {}
    for key in list(kwargs):
        if key in _SUPPORTED_TRANSFER_KWARGS:
            predict_kwargs[key] = kwargs.pop(key)
    _raise_on_unexpected_transfer_kwargs(kwargs)
    return predict_kwargs


def _decode_scores(
    *,
    class_scores: np.ndarray,
    classes: np.ndarray,
    min_conf: float,
    unknown_label: str,
) -> tuple[np.ndarray, np.ndarray]:
    scores = _normalize_rows(np.asarray(class_scores, dtype=np.float32, order="C"))
    pred_idx = scores.argmax(axis=1)
    pred = classes[pred_idx].astype(object)
    conf = scores.max(axis=1).astype(np.float32, copy=False)
    pred[conf < min_conf] = unknown_label
    return pred, conf


def _scores_from_tissue_aware_knn(
    *,
    ref_emb: np.ndarray,
    ref_codes: np.ndarray,
    query_emb: np.ndarray,
    n_classes: int,
    k: int,
    metric: str,
    ref_tissues: Optional[np.ndarray] = None,
    query_tissues: Optional[np.ndarray] = None,
    weight_mode: str = "exp",
    prior_alpha: float = 0.0,
    use_gpu: bool = True,
    gpu_device: int = 0,
) -> np.ndarray:
    if ref_emb.shape[0] == 0:
        return np.zeros((query_emb.shape[0], n_classes), dtype=np.float32)
    if ref_tissues is None or query_tissues is None:
        return _scores_from_knn(
            ref_emb=ref_emb,
            ref_codes=ref_codes,
            query_emb=query_emb,
            n_classes=n_classes,
            k=k,
            metric=metric,
            use_gpu=use_gpu,
            gpu_device=gpu_device,
        )
    counts = np.bincount(ref_codes, minlength=n_classes).astype(np.float64)
    class_prior = (1.0 / np.maximum(counts, 1.0)) ** float(prior_alpha)
    ref_tissue_s = pd.Series(np.asarray(ref_tissues, dtype=object), copy=False)
    query_tissue_s = pd.Series(np.asarray(query_tissues, dtype=object), copy=False)
    all_ref_idx = np.arange(ref_emb.shape[0], dtype=np.int64)
    out = np.zeros((query_emb.shape[0], n_classes), dtype=np.float32)
    by_tissue: dict[str, np.ndarray] = {}

    if ref_tissues is not None and query_tissues is not None:
        for tissue_value, group_index in ref_tissue_s.groupby(ref_tissue_s, dropna=False).groups.items():
            ref_idx = np.asarray(group_index, dtype=np.int64)
            if ref_idx.size == 0:
                continue
            by_tissue[str(tissue_value)] = ref_idx

    for tissue_value, group_index in query_tissue_s.groupby(query_tissue_s, dropna=False).groups.items():
        q_idx = np.asarray(group_index, dtype=np.int64)
        key = str(tissue_value)
        if key in by_tissue:
            ref_idx = by_tissue[key]
            dist, nbr_local = _knn_search(
                query_emb[q_idx],
                ref_emb[ref_idx],
                k=k,
                metric=metric,
                use_gpu=use_gpu,
                gpu_device=gpu_device,
            )
            nbr = ref_idx[nbr_local]
        else:
            ref_idx = all_ref_idx
            dist, nbr = _knn_search(
                query_emb[q_idx],
                ref_emb,
                k=k,
                metric=metric,
                use_gpu=use_gpu,
                gpu_device=gpu_device,
            )

        if weight_mode == "inv":
            weights = 1.0 / (dist + 1e-4)
        elif weight_mode == "exp":
            tau = float(np.median(dist)) if dist.size else 1.0
            if tau <= 0 or not np.isfinite(tau):
                tau = 1.0
            weights = np.exp(-dist / tau)
        else:
            raise ValueError("weight_mode must be one of {'inv','exp'}")

        cls = ref_codes[nbr]
        weights = weights * class_prior[cls]
        tissue_scores = np.zeros((q_idx.size, n_classes), dtype=np.float64)
        rows = np.arange(q_idx.size)
        for j in range(cls.shape[1]):
            np.add.at(tissue_scores, (rows, cls[:, j]), weights[:, j])
        out[q_idx] = _normalize_rows(tissue_scores).astype(np.float32, copy=False)
    return out


def _build_transition(
    x: np.ndarray,
    *,
    k: int,
    metric: str = "cosine",
    use_gpu: bool = True,
    gpu_device: int = 0,
) -> sp.csr_matrix:
    if x.shape[0] <= 1:
        return sp.eye(x.shape[0], dtype=np.float32, format="csr")
    k_eff = int(max(1, min(int(k), x.shape[0])))
    dist, idx = _knn_search(
        x,
        x,
        k=k_eff,
        metric=metric,
        use_gpu=use_gpu,
        gpu_device=gpu_device,
    )
    tau = float(np.median(dist)) if dist.size else 1.0
    if tau <= 0 or not np.isfinite(tau):
        tau = 1.0
    weights = np.exp(-dist / tau)
    rows = np.repeat(np.arange(x.shape[0]), idx.shape[1])
    graph = sp.csr_matrix(
        (weights.reshape(-1), (rows, idx.reshape(-1))),
        shape=(x.shape[0], x.shape[0]),
        dtype=np.float64,
    )
    graph = 0.5 * (graph + graph.T)
    deg = np.asarray(graph.sum(axis=1)).ravel()
    return (sp.diags(1.0 / np.clip(deg, 1e-8, None)) @ graph).tocsr()


def _diffuse_scores(scores: np.ndarray, trans: sp.csr_matrix, *, alpha: float, steps: int) -> np.ndarray:
    if alpha <= 0.0 or steps <= 0 or scores.shape[0] == 0:
        return _normalize_rows(scores)
    f = np.asarray(scores, dtype=np.float64, order="C").copy()
    f0 = f.copy()
    for _ in range(int(steps)):
        f = (1.0 - alpha) * f0 + alpha * (trans @ f)
    return _normalize_rows(f).astype(np.float32, copy=False)


def _diffuse_scores_by_group(
    scores: np.ndarray,
    query_emb: np.ndarray,
    *,
    alpha: float,
    steps: int,
    graph_k: int,
    metric: str,
    groups: Optional[np.ndarray] = None,
    use_gpu: bool = True,
    gpu_device: int = 0,
) -> np.ndarray:
    if groups is None:
        trans = _build_transition(
            query_emb,
            k=graph_k,
            metric=metric,
            use_gpu=use_gpu,
            gpu_device=gpu_device,
        )
        return _diffuse_scores(scores, trans, alpha=alpha, steps=steps)

    out = np.asarray(scores, dtype=np.float32, order="C").copy()
    group_s = pd.Series(np.asarray(groups, dtype=object), copy=False)
    for _, group_index in group_s.groupby(group_s, dropna=False).groups.items():
        idx = np.asarray(group_index, dtype=np.int64)
        if idx.size == 0:
            continue
        trans = _build_transition(
            query_emb[idx],
            k=graph_k,
            metric=metric,
            use_gpu=use_gpu,
            gpu_device=gpu_device,
        )
        out[idx] = _diffuse_scores(out[idx], trans, alpha=alpha, steps=steps)
    return out


def _fit_multinomial(
    x: np.ndarray,
    y: np.ndarray,
    *,
    c_value: float = LOGREG_DEFAULT_C,
    class_weight: Optional[str | dict[int, float]] = LOGREG_DEFAULT_CLASS_WEIGHT,
    solver: str = LOGREG_DEFAULT_SOLVER,
    penalty: str = LOGREG_DEFAULT_PENALTY,
    multi_class: str = LOGREG_DEFAULT_MULTI_CLASS,
    max_iter: int = LOGREG_DEFAULT_MAX_ITER,
    random_state: int = 0,
    sample_weight: Optional[np.ndarray] = None,
) -> LogisticRegression:
    solver_norm = str(solver).lower()
    penalty_norm = str(penalty).lower()
    multi_class_norm = str(multi_class).lower()
    if penalty_norm != "l2":
        raise ValueError("Only logreg_penalty='l2' is supported.")
    if solver_norm not in {"lbfgs", "saga"}:
        raise ValueError("logreg_solver must be one of {'lbfgs', 'saga'}.")
    if multi_class_norm not in {"auto", "ovr", "multinomial"}:
        raise ValueError("logreg_multi_class must be one of {'auto', 'ovr', 'multinomial'}.")
    base_kwargs = {
        "solver": solver_norm,
        "penalty": penalty_norm,
        "max_iter": int(max_iter),
        "C": float(c_value),
        "class_weight": class_weight,
        "random_state": int(random_state),
    }
    if multi_class_norm == "ovr":
        model = OneVsRestClassifier(LogisticRegression(**base_kwargs))
    else:
        model = LogisticRegression(**base_kwargs)
    if sample_weight is None:
        model.fit(x, y)
    else:
        model.fit(x, y, sample_weight=sample_weight)
    return model


def _align_proba(model: LogisticRegression, x: np.ndarray, n_classes: int) -> np.ndarray:
    proba_local = model.predict_proba(x)
    out = np.zeros((x.shape[0], n_classes), dtype=np.float32)
    out[:, model.classes_.astype(int)] = proba_local.astype(np.float32, copy=False)
    return out


def _apply_reference_prior_correction(
    scores: np.ndarray,
    y_ref_code: np.ndarray,
    *,
    n_classes: int,
    alpha: float,
) -> np.ndarray:
    alpha = float(alpha)
    if alpha <= 0.0:
        return _normalize_rows(scores)
    counts = np.bincount(np.asarray(y_ref_code, dtype=np.int32), minlength=int(n_classes)).astype(np.float64)
    inv_prior = (1.0 / np.maximum(counts, 1.0)) ** alpha
    corrected = np.asarray(scores, dtype=np.float64, order="C") * inv_prior[None, :]
    return _normalize_rows(corrected).astype(np.float32, copy=False)


def _scores_from_tissue_routed_logreg(
    *,
    ref_emb: np.ndarray,
    labels_ref: np.ndarray,
    query_emb: np.ndarray,
    ref_tissues: Optional[np.ndarray] = None,
    query_tissues: Optional[np.ndarray] = None,
    c_value: float = LOGREG_DEFAULT_C,
    min_cells: int = 80,
    min_classes: int = 2,
    classes: np.ndarray | None = None,
    y_ref_code: np.ndarray | None = None,
    class_weight: Optional[str | dict[int, float]] = LOGREG_DEFAULT_CLASS_WEIGHT,
    solver: str = LOGREG_DEFAULT_SOLVER,
    penalty: str = LOGREG_DEFAULT_PENALTY,
    multi_class: str = LOGREG_DEFAULT_MULTI_CLASS,
    max_iter: int = LOGREG_DEFAULT_MAX_ITER,
    prior_alpha: float = LOGREG_DEFAULT_PRIOR_ALPHA,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    if classes is None or y_ref_code is None:
        le = LabelEncoder()
        y_ref_code = le.fit_transform(labels_ref.astype(str))
        classes = le.classes_.astype(object)
    else:
        classes = np.asarray(classes, dtype=object)
        y_ref_code = np.asarray(y_ref_code, dtype=np.int32)
    n_classes = int(classes.size)

    global_model = _fit_multinomial(
        ref_emb,
        y_ref_code,
        c_value=c_value,
        class_weight=class_weight,
        solver=solver,
        penalty=penalty,
        multi_class=multi_class,
        max_iter=max_iter,
        random_state=random_state,
    )
    scores = _apply_reference_prior_correction(
        _align_proba(global_model, query_emb, n_classes),
        y_ref_code,
        n_classes=n_classes,
        alpha=prior_alpha,
    )

    if ref_tissues is None or query_tissues is None:
        return classes, _normalize_rows(scores)

    ref_tissue_s = pd.Series(np.asarray(ref_tissues, dtype=object), copy=False)
    query_tissue_s = pd.Series(np.asarray(query_tissues, dtype=object), copy=False)
    by_tissue_model: dict[str, LogisticRegression] = {}

    for tissue_value, group_index in ref_tissue_s.groupby(ref_tissue_s, dropna=False).groups.items():
        idx = np.asarray(group_index, dtype=np.int64)
        if idx.size < int(min_cells):
            continue
        y_loc = y_ref_code[idx]
        if np.unique(y_loc).size < int(min_classes):
            continue
        try:
            by_tissue_model[str(tissue_value)] = _fit_multinomial(
                ref_emb[idx],
                y_loc,
                c_value=c_value,
                class_weight=class_weight,
                solver=solver,
                penalty=penalty,
                multi_class=multi_class,
                max_iter=max_iter,
                random_state=random_state,
            )
        except Exception:
            continue

    for tissue_value, group_index in query_tissue_s.groupby(query_tissue_s, dropna=False).groups.items():
        key = str(tissue_value)
        if key not in by_tissue_model:
            continue
        q_idx = np.asarray(group_index, dtype=np.int64)
        scores[q_idx] = _apply_reference_prior_correction(
            _align_proba(by_tissue_model[key], query_emb[q_idx], n_classes),
            y_ref_code,
            n_classes=n_classes,
            alpha=prior_alpha,
        )

    return classes, _normalize_rows(scores)


def predict_pseudo_labels(
    adata: Any,
    *,
    rep: np.ndarray | None = None,
    rep_key: str | None = None,
    label_key: str,
    unlabeled_category: str | Sequence[str] | set[str] | None = "unknown",
    pred_label_key: str = "pred_cell_type",
    pred_conf_key: str = "pred_confidence",
    min_conf: float = 0.0,
    return_numpy: bool = True,
    inplace: bool = False,
    max_ref: int = 20000,
    use_gpu: bool = True,
    gpu_device: int = 0,
    transfer_mode: str = "logreg",
    knn_k: Optional[int] = None,
    knn_metric: str = "cosine",
    knn_tissue_key: Optional[str] = "tissue",
    knn_weight_mode: str = "exp",
    knn_prior_alpha: float = 0.0,
    knn_graph_k: int = 20,
    knn_diffuse_alpha: float = 0.2,
    knn_diffuse_steps: int = 1,
    knn_diffuse_group_key: Optional[str] = "modality",
    query_group_key: Optional[str] = "modality",
    query_group_value: str = "query",
    logreg_tissue_key: Optional[str] = "tissue",
    logreg_min_cells: int = 80,
    logreg_min_classes: int = 2,
    logreg_class_weight: Optional[str | dict[int, float]] = LOGREG_DEFAULT_CLASS_WEIGHT,
    logreg_C: float = LOGREG_DEFAULT_C,
    logreg_solver: str = LOGREG_DEFAULT_SOLVER,
    logreg_penalty: str = LOGREG_DEFAULT_PENALTY,
    logreg_multi_class: str = LOGREG_DEFAULT_MULTI_CLASS,
    logreg_max_iter: int = LOGREG_DEFAULT_MAX_ITER,
    logreg_prior_alpha: float = LOGREG_DEFAULT_PRIOR_ALPHA,
    logreg_standardize: bool = LOGREG_DEFAULT_STANDARDIZE,
    input_rep_key: Optional[str] = SUPBIOT_DEFAULT_INPUT_REP_KEY,
    input_rep_weight: float = SUPBIOT_DEFAULT_INPUT_REP_WEIGHT,
    prototype_weight: float = SUPBIOT_DEFAULT_PROTOTYPE_WEIGHT,
    random_state: int = 0,
) -> Tuple[np.ndarray | pd.Series, np.ndarray | pd.Series]:
    """
    Predict pseudo labels using either tissue-aware kNN or tissue-routed logistic regression.

    By default the primary representation's class probabilities are blended with those
    from the supervised autoencoder embedding ``input_rep_key`` (``X_ae``) as
    ``(1 - w) * primary + w * other``. The blend no-ops when that key is absent from
    ``adata.obsm`` or equals the primary rep, so single-rep transfer needs no extra
    arguments; pass ``input_rep_weight=0`` to disable it explicitly. If the named
    embedding has stability-pass siblings (``{key}_e1``, ``{key}_e2``, ... written by
    ``autoencoder(..., n_ensemble>1)``), their probabilities are averaged first, which
    stabilizes rare-label predictions.
    """

    n_obs = int(adata.n_obs)
    mode = _validate_transfer_mode(transfer_mode)
    unknown_label = unlabeled_category if isinstance(unlabeled_category, str) else "unknown"
    _, ref_mask, query_mask = _resolve_ref_query_masks(
        adata,
        label_key=label_key,
        unlabeled_category=unlabeled_category,
    )
    query_mask = _restrict_query_mask(
        adata,
        query_mask,
        query_group_key=query_group_key,
        query_group_value=query_group_value,
    )

    if rep is None:
        meta = adata.uns.get("_supbiot", {})
        rep_key = rep_key or meta.get("rep_key", "X_pca_shared_aligned")
        if rep_key not in adata.obsm:
            raise KeyError(f"{rep_key!r} not found in adata.obsm")
    rep_arr = np.asarray(rep if rep is not None else adata.obsm[rep_key], dtype=np.float32, order="C")
    if rep_arr.shape[0] != n_obs:
        raise ValueError("rep must have the same number of rows as adata.obs")

    ref_obs_idx = np.flatnonzero(np.asarray(ref_mask, dtype=bool))
    query_obs_idx = np.flatnonzero(np.asarray(query_mask, dtype=bool))
    labels_ref = adata.obs.iloc[ref_obs_idx][label_key].astype("string").astype(str)
    n_ref = int(labels_ref.shape[0])
    if max_ref is not None and n_ref > int(max_ref):
        keep = _select_stratified_reference_subset(
            labels_ref,
            max_ref=max_ref,
            random_state=random_state,
        )
        ref_obs_idx = ref_obs_idx[keep]
        labels_ref = labels_ref.iloc[keep]
    classes, ref_codes = _encode_labels_for_transfer(labels_ref)

    if rep_arr[ref_obs_idx].size == 0 or rep_arr[query_obs_idx].size == 0:
        raise ValueError("Reference/query subset empty; check label_key/unlabeled_category.")

    def _score_rep(rep_full: np.ndarray) -> np.ndarray:
        """Class probability scores for one embedding (shared ref/query split)."""
        ref_emb = rep_full[ref_obs_idx]
        query_emb = rep_full[query_obs_idx]
        if mode == "knn":
            ref_tissues: np.ndarray | None = None
            query_tissues: np.ndarray | None = None
            if knn_tissue_key is not None and knn_tissue_key in adata.obs:
                ref_tissues = adata.obs.iloc[ref_obs_idx][knn_tissue_key].to_numpy(copy=False)
                query_tissues = adata.obs.iloc[query_obs_idx][knn_tissue_key].to_numpy(copy=False)
            scores = _scores_from_tissue_aware_knn(
                ref_emb=ref_emb,
                ref_codes=ref_codes,
                query_emb=query_emb,
                n_classes=int(classes.size),
                k=int(knn_k if knn_k is not None else 8),
                metric=knn_metric,
                ref_tissues=ref_tissues,
                query_tissues=query_tissues,
                weight_mode=knn_weight_mode,
                prior_alpha=knn_prior_alpha,
                use_gpu=use_gpu,
                gpu_device=gpu_device,
            )
            graph_k_eff = int(min(max(2, int(knn_graph_k)), max(2, query_emb.shape[0] - 1)))
            query_groups: Optional[np.ndarray] = None
            if knn_diffuse_group_key is not None and knn_diffuse_group_key in adata.obs:
                query_groups = adata.obs.iloc[query_obs_idx][knn_diffuse_group_key].to_numpy(copy=False)
            return _diffuse_scores_by_group(
                scores,
                query_emb,
                alpha=float(knn_diffuse_alpha),
                steps=int(knn_diffuse_steps),
                graph_k=graph_k_eff,
                metric=knn_metric,
                groups=query_groups,
                use_gpu=use_gpu,
                gpu_device=gpu_device,
            )
        if mode == "logreg":
            ref_tissues = None
            query_tissues = None
            if logreg_tissue_key is not None and logreg_tissue_key in adata.obs:
                ref_tissues = adata.obs.iloc[ref_obs_idx][logreg_tissue_key].to_numpy(copy=False)
                query_tissues = adata.obs.iloc[query_obs_idx][logreg_tissue_key].to_numpy(copy=False)
            if bool(logreg_standardize):
                ref_emb, query_emb = _standardize_ref_query(ref_emb, query_emb)
            _, scores = _scores_from_tissue_routed_logreg(
                ref_emb=ref_emb,
                labels_ref=labels_ref.to_numpy(dtype=object),
                classes=classes,
                y_ref_code=ref_codes,
                query_emb=query_emb,
                ref_tissues=ref_tissues,
                query_tissues=query_tissues,
                c_value=logreg_C,
                min_cells=logreg_min_cells,
                min_classes=logreg_min_classes,
                class_weight=logreg_class_weight,
                solver=logreg_solver,
                penalty=logreg_penalty,
                multi_class=logreg_multi_class,
                max_iter=logreg_max_iter,
                prior_alpha=logreg_prior_alpha,
                random_state=random_state,
            )
            return scores
        raise ValueError(f"Unsupported transfer_mode {mode!r}.")

    class_scores = _score_rep(rep_arr)
    resolved_input_reps = _resolve_input_rep_key(
        adata, primary_key=rep_key, input_rep_key=input_rep_key
    )
    # Optional blend: average scores from the explicitly requested secondary
    # embedding(s), then mix with the primary rep's scores.
    input_scores = [
        _normalize_rows(_score_rep(np.asarray(adata.obsm[key], dtype=np.float32, order="C")))
        for key in resolved_input_reps
        if np.asarray(adata.obsm[key]).shape[0] == n_obs
    ]
    if input_scores and float(input_rep_weight) > 0.0:
        w = float(np.clip(input_rep_weight, 0.0, 1.0))
        input_mean = np.mean(input_scores, axis=0)
        class_scores = (1.0 - w) * _normalize_rows(class_scores) + w * input_mean

    # Optional reference-prototype calibration: blend cosine-to-centroid softmax scores
    # (computed on the primary rep) into the prediction to stabilize confusable class
    # boundaries. One shared weight across datasets.
    if float(prototype_weight) > 0.0:
        pw = float(np.clip(prototype_weight, 0.0, 1.0))
        proto = _prototype_scores(
            ref_emb=rep_arr[ref_obs_idx],
            ref_codes=ref_codes,
            query_emb=rep_arr[query_obs_idx],
            n_classes=int(classes.size),
        )
        class_scores = (1.0 - pw) * _normalize_rows(class_scores) + pw * proto

    pred_labels_query, pred_conf_query = _decode_scores(
        class_scores=class_scores,
        classes=classes,
        min_conf=min_conf,
        unknown_label=unknown_label,
    )

    pred_labels = np.full(n_obs, None, dtype=object)
    pred_conf = np.full(n_obs, np.nan, dtype=np.float32)
    pred_labels[query_mask] = pred_labels_query
    pred_conf[query_mask] = pred_conf_query

    if inplace:
        if pred_label_key in adata.obs and isinstance(adata.obs[pred_label_key].dtype, pd.CategoricalDtype):
            base_col = adata.obs[pred_label_key]
            extra = pd.unique(pd.Series(pred_labels_query).dropna())
            categories = list(base_col.cat.categories)
            for value in extra:
                if value not in categories:
                    categories.append(value)
            if categories != list(base_col.cat.categories):
                adata.obs[pred_label_key] = base_col.cat.set_categories(categories)
            adata.obs.loc[adata.obs_names[query_mask], pred_label_key] = pred_labels_query
        else:
            if pred_label_key in adata.obs:
                adata.obs.loc[adata.obs_names[query_mask], pred_label_key] = pred_labels_query
            else:
                adata.obs[pred_label_key] = pred_labels

        if pred_conf_key in adata.obs:
            adata.obs.loc[adata.obs_names[query_mask], pred_conf_key] = pred_conf_query
        else:
            adata.obs[pred_conf_key] = pred_conf

    if return_numpy:
        return pred_labels, pred_conf
    pred_labels_s = pd.Series(pred_labels, index=adata.obs_names, name=pred_label_key)
    pred_conf_s = pd.Series(pred_conf, index=adata.obs_names, name=pred_conf_key)
    return pred_labels_s, pred_conf_s


def transfer_labels(
    adata: Any,
    *,
    label_key: str,
    unlabeled_category: str | Sequence[str] | set[str] = "unknown",
    pred_label_key: str = "pred_cell_type",
    pred_conf_key: str = "pred_confidence",
    min_conf: float = 0.25,
    transfer_mode: str = "logreg",
    use_embedding_ref: bool = False,
    embedding_key: Optional[str] = None,
    embedding_k: Optional[int] = None,
    embedding_weight_power: float = 2.0,
    inplace: bool = True,
    out_key: Optional[str] = None,
    verbose: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Transfer labels from reference to query using OT alignment computed by integrate_ot.
    When ``use_embedding_ref=True``, projected query embeddings are stored under
    ``embedding_key``.
    """
    from ..utils.ot_transfer import (
        _knn_barycentric_project,
        _resolve_obsm_key,
        assemble_joint_embedding,
    )

    if "transfer_mode" in kwargs:
        transfer_mode_kw = kwargs.pop("transfer_mode")
        if transfer_mode != "logreg" and transfer_mode_kw != transfer_mode:
            raise ValueError("Conflicting transfer_mode specified in argument and kwargs.")
        transfer_mode = transfer_mode_kw
    predict_kwargs = _pop_predict_pseudo_label_kwargs(kwargs)

    meta = adata.uns.get("_supbiot", {})
    rep_key = out_key or meta.get("rep_key", "X_pca_shared_aligned")

    map_key = embedding_key or "X_umap"
    rep_key_map = rep_key
    if use_embedding_ref:
        if embedding_key is not None:
            map_key = embedding_key
    elif embedding_key is not None and verbose:
        print("[supbiot] embedding_key is ignored because use_embedding_ref=False.")

    _, ref_mask, query_mask = _resolve_ref_query_masks(
        adata,
        label_key=label_key,
        unlabeled_category=unlabeled_category,
    )
    query_mask = _restrict_query_mask(
        adata,
        query_mask,
        query_group_key=predict_kwargs.get("query_group_key", "modality"),
        query_group_value=predict_kwargs.get("query_group_value", "query"),
    )
    
    adata_ref = adata[ref_mask].copy()
    adata_query = adata[query_mask].copy()

    if rep_key not in adata.obsm:
        raise KeyError(f"{rep_key!r} not found in adata.obsm")

    pred_labels, pred_conf = predict_pseudo_labels(
        adata,
        rep=None,
        rep_key=rep_key,
        label_key=label_key,
        unlabeled_category=unlabeled_category,
        pred_label_key=pred_label_key,
        pred_conf_key=pred_conf_key,
        min_conf=min_conf,
        return_numpy=True,
        inplace=False,
        max_ref=predict_kwargs.get("max_ref", 20000),
        use_gpu=predict_kwargs.get("use_gpu", True),
        gpu_device=predict_kwargs.get("gpu_device", 0),
        transfer_mode=transfer_mode,
        knn_k=predict_kwargs.get("knn_k"),
        knn_metric=predict_kwargs.get("knn_metric", "cosine"),
        knn_tissue_key=predict_kwargs.get("knn_tissue_key", "tissue"),
        knn_weight_mode=predict_kwargs.get("knn_weight_mode", "exp"),
        knn_prior_alpha=predict_kwargs.get("knn_prior_alpha", 0.0),
        knn_graph_k=predict_kwargs.get("knn_graph_k", 20),
        knn_diffuse_alpha=predict_kwargs.get("knn_diffuse_alpha", 0.2),
        knn_diffuse_steps=predict_kwargs.get("knn_diffuse_steps", 1),
        knn_diffuse_group_key=predict_kwargs.get("knn_diffuse_group_key", "modality"),
        query_group_key=predict_kwargs.get("query_group_key", "modality"),
        query_group_value=predict_kwargs.get("query_group_value", "query"),
        logreg_tissue_key=predict_kwargs.get("logreg_tissue_key", "tissue"),
        logreg_min_cells=predict_kwargs.get("logreg_min_cells", 80),
        logreg_min_classes=predict_kwargs.get("logreg_min_classes", 2),
        logreg_class_weight=predict_kwargs.get("logreg_class_weight", LOGREG_DEFAULT_CLASS_WEIGHT),
        logreg_C=predict_kwargs.get("logreg_C", LOGREG_DEFAULT_C),
        logreg_solver=predict_kwargs.get("logreg_solver", LOGREG_DEFAULT_SOLVER),
        logreg_penalty=predict_kwargs.get("logreg_penalty", LOGREG_DEFAULT_PENALTY),
        logreg_multi_class=predict_kwargs.get("logreg_multi_class", LOGREG_DEFAULT_MULTI_CLASS),
        logreg_max_iter=predict_kwargs.get("logreg_max_iter", LOGREG_DEFAULT_MAX_ITER),
        logreg_prior_alpha=predict_kwargs.get("logreg_prior_alpha", LOGREG_DEFAULT_PRIOR_ALPHA),
        logreg_standardize=predict_kwargs.get("logreg_standardize", LOGREG_DEFAULT_STANDARDIZE),
        input_rep_key=predict_kwargs.get("input_rep_key", SUPBIOT_DEFAULT_INPUT_REP_KEY),
        input_rep_weight=predict_kwargs.get("input_rep_weight", SUPBIOT_DEFAULT_INPUT_REP_WEIGHT),
        prototype_weight=predict_kwargs.get("prototype_weight", SUPBIOT_DEFAULT_PROTOTYPE_WEIGHT),
        random_state=predict_kwargs.get("random_state", 0),
    )
    adata_query.obs["pred_cell_type"] = pred_labels[query_mask]
    adata_query.obs["pred_confidence"] = pred_conf[query_mask]
    if pred_label_key != "pred_cell_type":
        adata_query.obs[pred_label_key] = adata_query.obs["pred_cell_type"]
    if pred_conf_key != "pred_confidence":
        adata_query.obs[pred_conf_key] = adata_query.obs["pred_confidence"]

    ot_meta = adata.uns.get("_ot_alignment")

    mapped_key: Optional[str] = None
    ref_umap_key: Optional[str] = None
    do_map = use_embedding_ref
    if do_map:
        if ot_meta is None:
            raise RuntimeError("OT alignment metadata missing; run integrate_ot first.")
        topk = int(ot_meta.get("params", {}).get("topk", 64))
        k_default = min(max(topk, 1), 50)
        k_map = k_default if embedding_k is None else int(embedding_k)
        k_map = max(k_map, 1)

        rep_ref = adata_ref.obsm[rep_key_map]
        rep_query = adata_query.obsm[rep_key_map]

        ref_umap_key = _resolve_obsm_key(adata_ref, map_key)        
        ref_umap = np.asarray(adata_ref.obsm[ref_umap_key], dtype=np.float32)        
        mapped_key = ref_umap_key
        mapped_query = _knn_barycentric_project(
            rep_ref,
            rep_query,
            ref_umap,
            k=k_map,
            metric="euclidean",
            weight_power=embedding_weight_power,
        )
        adata_query.obsm[mapped_key] = mapped_query

    adata_joint = assemble_joint_embedding(
        rep_key,
        {"Reference": adata_ref, "Query": adata_query},
    )

    if mapped_key is not None:
        ref_umap = adata_ref.obsm[ref_umap_key]
        query_umap = adata_query.obsm[mapped_key]
        if ref_umap.shape[1] != query_umap.shape[1]:
            raise ValueError("UMAP dimensionality mismatch between reference and query.")
        joint_umap = np.full((adata_joint.n_obs, ref_umap.shape[1]), np.nan, dtype=np.float32)
        mask_ref = adata_joint.obs["modality"] == "Reference"
        mask_query = adata_joint.obs["modality"] == "Query"
        joint_umap[mask_ref] = ref_umap
        joint_umap[mask_query] = query_umap
        adata_joint.obsm[mapped_key] = joint_umap

    if inplace:
        for col in adata_query.obs.columns:
            if col in adata.obs and isinstance(adata.obs[col].dtype, pd.CategoricalDtype):
                base_col = adata.obs[col]
                query_col = adata_query.obs[col]
                base_categories = list(base_col.cat.categories)
                if isinstance(query_col.dtype, pd.CategoricalDtype):
                    extra_categories = list(query_col.cat.categories)
                else:
                    extra_categories = [v for v in pd.unique(query_col.dropna())]
                for value in extra_categories:
                    if value not in base_categories:
                        base_categories.append(value)
                if base_categories != list(base_col.cat.categories):
                    adata.obs[col] = base_col.cat.set_categories(base_categories)
                values = query_col.to_numpy()
            else:
                values = adata_query.obs[col].values
            adata.obs.loc[adata_query.obs_names, col] = values
        if mapped_key is not None:
            ref_umap = adata_ref.obsm[ref_umap_key]
            query_umap = adata_query.obsm[mapped_key]
            joint_umap = np.full((adata.n_obs, ref_umap.shape[1]), np.nan, dtype=np.float32)
            joint_umap[np.asarray(ref_mask)] = ref_umap
            joint_umap[np.asarray(query_mask)] = query_umap
            adata.obsm[mapped_key] = joint_umap
        return adata
    return adata_joint
