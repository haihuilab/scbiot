"""
Metadata helpers for storing OT plan mappings and masks.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import sparse


def _to_index_list(values: Any, n_obs: Optional[int] = None) -> list[int]:
    idx = np.asarray(values)
    if idx.dtype == bool:
        if n_obs is not None and idx.size != n_obs:
            raise ValueError("Boolean index mask length does not match number of observations.")
        idx = np.flatnonzero(idx)
    return idx.astype(int, copy=False).tolist()


def _normalize_condition_values(value: Any) -> np.ndarray:
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return np.asarray(list(value), dtype=object)
    return np.asarray([value], dtype=object)


def _infer_condition_key(adata: Any, cond_key: Optional[str], scb_key: str) -> str:
    if cond_key is not None:
        if cond_key not in adata.obs:
            raise KeyError(f"cond_key '{cond_key}' not found in adata.obs")
        return cond_key
    if scb_key in getattr(adata, "uns", {}):
        batch_key = adata.uns.get(scb_key, {}).get("batch_key")
        if batch_key and batch_key in adata.obs:
            return str(batch_key)
    for candidate in ("condition", "batch", "id"):
        if candidate in adata.obs:
            return candidate
    raise KeyError("cond_key not found; pass cond_key explicitly.")


def masks_from_conditions(
    adata: Any,
    cond1: Any,
    cond2: Any,
    *,
    cond_key: Optional[str] = None,
    scb_key: str = "scb_ot",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Infer source/target masks from condition labels stored in adata.obs[cond_key].
    If one of cond1/cond2 is None, the other is compared against the rest.
    """
    if cond1 is None and cond2 is None:
        info = getattr(adata, "uns", {}).get(scb_key, {})
        batches = info.get("batches")
        if batches is None or len(batches) != 2:
            raise ValueError("cond1 and cond2 are required unless exactly two batches are registered.")
        cond1, cond2 = batches[0], batches[1]

    key = _infer_condition_key(adata, cond_key, scb_key)
    values = np.asarray(adata.obs[key], dtype=object)
    if cond1 is None:
        cond2_vals = _normalize_condition_values(cond2)
        tgt_mask = np.isin(values, cond2_vals)
        if not np.any(tgt_mask):
            raise ValueError(f"No observations found for cond2 values in '{key}'.")
        src_mask = ~tgt_mask
        if not np.any(src_mask):
            raise ValueError(f"No observations found outside cond2 values in '{key}'.")
        return src_mask, tgt_mask
    if cond2 is None:
        cond1_vals = _normalize_condition_values(cond1)
        src_mask = np.isin(values, cond1_vals)
        if not np.any(src_mask):
            raise ValueError(f"No observations found for cond1 values in '{key}'.")
        tgt_mask = ~src_mask
        if not np.any(tgt_mask):
            raise ValueError(f"No observations found outside cond1 values in '{key}'.")
        return src_mask, tgt_mask

    cond1_vals = _normalize_condition_values(cond1)
    cond2_vals = _normalize_condition_values(cond2)
    src_mask = np.isin(values, cond1_vals)
    tgt_mask = np.isin(values, cond2_vals)
    if not np.any(src_mask):
        raise ValueError(f"No observations found for cond1 values in '{key}'.")
    if not np.any(tgt_mask):
        raise ValueError(f"No observations found for cond2 values in '{key}'.")
    return src_mask, tgt_mask


def _infer_rep_key(adata: Any, rep_key: Optional[str], scb_key: str) -> str:
    if rep_key is not None:
        if not hasattr(adata, "obsm") or rep_key not in adata.obsm:
            raise KeyError(f"rep_key '{rep_key}' not found in adata.obsm")
        return rep_key
    info = getattr(adata, "uns", {}).get(scb_key, {})
    for key in ("rep_key", "out_key", "obsm_key"):
        candidate = info.get(key)
        if candidate and candidate in getattr(adata, "obsm", {}):
            return str(candidate)
    for candidate in ("X_ot", "scBIOT", "X_pca", "X_lsi"):
        if candidate in getattr(adata, "obsm", {}):
            return candidate
    raise KeyError("rep_key not found; pass rep_key explicitly.")


def _format_condition_label(value: Any) -> str:
    if isinstance(value, (list, tuple, set, np.ndarray)):
        parts = [str(v) for v in list(value)]
        label = "__".join(parts)
    else:
        label = str(value)
    label = label.strip().replace(" ", "_").replace("/", "_").replace(":", "_")
    label = label.replace(";", "_").replace(",", "_")
    return label


def _format_gamma_key(cond1: Any, cond2: Any) -> str:
    return f"gamma_{_format_condition_label(cond1)}_to_{_format_condition_label(cond2)}"


def _rest_label(other: Any) -> str:
    return f"rest_not_{_format_condition_label(other)}"


def ensure_gamma_for_conditions(
    adata: Any,
    cond1: Any,
    cond2: Any,
    *,
    cond_key: Optional[str] = None,
    scb_key: str = "scb_ot",
    rep_key: Optional[str] = None,
    gamma_key: Optional[str] = None,
    compute_gamma: bool = True,
    transport_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, Any, Optional[str]]:
    """
    Ensure a gamma/transport plan exists for cond1->cond2; compute and store if missing.
    """
    cond1_use = cond1
    cond2_use = cond2
    if cond1_use is None and cond2_use is None:
        info = getattr(adata, "uns", {}).get(scb_key, {})
        batches = info.get("batches")
        if batches is not None and len(batches) == 2:
            cond1_use, cond2_use = batches[0], batches[1]

    src_mask, tgt_mask = masks_from_conditions(
        adata,
        cond1_use,
        cond2_use,
        cond_key=cond_key,
        scb_key=scb_key,
    )
    gamma_key_found = _find_gamma_key_for_masks(adata, src_mask, tgt_mask, scb_key=scb_key)
    if gamma_key_found and gamma_key_found in getattr(adata, "uns", {}):
        return src_mask, tgt_mask, adata.uns[gamma_key_found], gamma_key_found
    if not compute_gamma:
        return src_mask, tgt_mask, None, gamma_key_found

    rep_key_use = _infer_rep_key(adata, rep_key, scb_key)
    rep = adata.obsm[rep_key_use]
    if hasattr(rep, "toarray"):
        rep = rep.toarray()
    rep = np.asarray(rep, dtype=np.float32, order="C")
    src_emb = rep[src_mask]
    tgt_emb = rep[tgt_mask]
    if src_emb.size == 0 or tgt_emb.size == 0:
        raise ValueError("Source or target embedding subset is empty.")

    kwargs = dict(transport_kwargs or {})
    from .ot_transport import compute_ot_alignment

    _, transport = compute_ot_alignment(src_emb, tgt_emb, **kwargs)

    cond1_label = cond1_use
    cond2_label = cond2_use
    if cond1_use is None and cond2_use is not None:
        cond1_label = _rest_label(cond2_use)
    elif cond2_use is None and cond1_use is not None:
        cond2_label = _rest_label(cond1_use)
    gamma_key_use = gamma_key or _format_gamma_key(cond1_label, cond2_label)
    if gamma_key_use in getattr(adata, "uns", {}):
        i = 1
        candidate = f"{gamma_key_use}_{i}"
        while candidate in adata.uns:
            i += 1
            candidate = f"{gamma_key_use}_{i}"
        gamma_key_use = candidate

    info = getattr(adata, "uns", {}).get(scb_key, {})
    reference = info.get("reference", "union")
    batch_key = info.get("batch_key") or _infer_condition_key(adata, cond_key, scb_key)
    save_scbiot_metadata(
        adata,
        batch_key=batch_key,
        scb_key=scb_key,
        gamma_store={
            gamma_key_use: (
                transport,
                np.flatnonzero(src_mask),
                np.flatnonzero(tgt_mask),
            )
        },
        reference=reference,
    )
    info = dict(getattr(adata, "uns", {}).get(scb_key, {}))
    info["rep_key"] = rep_key_use
    adata.uns[scb_key] = info
    return src_mask, tgt_mask, transport, gamma_key_use


def transport_to_sparse(transport: Any, n_source: int, n_target: int) -> sparse.coo_matrix:
    if not isinstance(transport, dict):
        raise ValueError("transport must be a dict with 'indices' and 'weights'.")
    if "indices" not in transport or "weights" not in transport:
        raise ValueError("transport must contain 'indices' and 'weights'.")
    idx = np.asarray(transport["indices"])
    weights = np.asarray(transport["weights"])
    if idx.ndim == 1:
        idx = idx[:, None]
    if weights.ndim == 1:
        weights = weights[:, None]
    if idx.shape != weights.shape:
        raise ValueError("transport 'indices' and 'weights' must have the same shape.")
    if idx.shape[0] != n_source:
        raise ValueError("transport row count does not match source cell count.")
    rows = np.repeat(np.arange(n_source), idx.shape[1])
    cols = idx.reshape(-1).astype(np.int64, copy=False)
    data = weights.reshape(-1).astype(np.float32, copy=False)
    mask = np.isfinite(data)
    mask &= cols >= 0
    mask &= cols < n_target
    if not np.any(mask):
        return sparse.coo_matrix((n_source, n_target), dtype=np.float32)
    return sparse.coo_matrix((data[mask], (rows[mask], cols[mask])), shape=(n_source, n_target))


def save_scbiot_metadata(
    adata: Any,
    *,
    batch_key: str = "id",
    scb_key: str = "scb_ot",
    gamma_store: Optional[Dict[str, Tuple[Any, list, list]]] = None,
    reference: str = "union",
) -> Any:
    """
    Save scBIOT metadata into adata.uns[scb_key].

    gamma_store: optional dict mapping gamma_key -> (gamma_matrix, rows_list, cols_list)
      - rows_list/cols_list must be integer indices into adata.obs (current ordering)
      - function also stores obs_names for robustness
    """
    if not hasattr(adata, "uns") or adata.uns is None:
        adata.uns = {}
    if not hasattr(adata, "obs") or batch_key not in adata.obs:
        raise KeyError(f"batch_key '{batch_key}' not found in adata.obs")

    info: Dict[str, Any] = dict(adata.uns.get(scb_key, {}))
    info["reference"] = str(reference)
    info["batch_key"] = str(batch_key)

    series = adata.obs[batch_key]
    try:
        batches = list(series.cat.categories)  # type: ignore[attr-defined]
    except Exception:
        batches = sorted(series.unique().tolist())
    info["batches"] = batches

    values = np.asarray(series)
    batch_indices: Dict[Any, list[int]] = {}
    for batch in batches:
        idx = np.where(values == batch)[0].tolist()
        batch_indices[batch] = idx
    info["batch_indices"] = batch_indices

    gamma_index_map: Dict[str, Any] = dict(info.get("gamma_index_map", {}))
    if gamma_store:
        obs_names = np.asarray(getattr(adata, "obs_names", []), dtype=object)
        n_obs = int(getattr(adata, "n_obs", obs_names.size))
        for gamma_key, (gamma_mat, rows_list, cols_list) in gamma_store.items():
            if gamma_mat is not None:
                adata.uns[gamma_key] = gamma_mat
            rows_idx = _to_index_list(rows_list, n_obs)
            cols_idx = _to_index_list(cols_list, n_obs)
            rows_obs = obs_names[rows_idx].tolist() if rows_idx else []
            cols_obs = obs_names[cols_idx].tolist() if cols_idx else []
            gamma_index_map[gamma_key] = {
                "rows_idx": rows_idx,
                "cols_idx": cols_idx,
                "rows_obs_names": rows_obs,
                "cols_obs_names": cols_obs,
            }

    info["gamma_index_map"] = gamma_index_map
    adata.uns[scb_key] = info
    return adata


def _mask_indices_from_obs_names(adata: Any, names: Any) -> np.ndarray:
    """
    Convert a list of obs_names into a boolean mask for current adata ordering.
    """
    obs_names = np.asarray(getattr(adata, "obs_names", []), dtype=object)
    return np.isin(obs_names, np.asarray(names, dtype=object))


def _masks_from_gamma_key(
    adata: Any,
    gamma_key: str,
    *,
    scb_key: str = "scb_ot",
) -> Tuple[np.ndarray, np.ndarray, Any]:
    """
    Returns (src_mask, tgt_mask, gamma) inferred from adata.uns[scb_key].
    Expects adata.uns[scb_key]['gamma_index_map'][gamma_key] to exist.
    """
    if scb_key not in adata.uns:
        raise KeyError(f"{scb_key} not found in adata.uns")
    info = adata.uns[scb_key]
    if "gamma_index_map" not in info:
        raise KeyError(f"No 'gamma_index_map' found in adata.uns['{scb_key}']")
    if gamma_key not in info["gamma_index_map"]:
        raise KeyError(f"gamma_key '{gamma_key}' not present in adata.uns['{scb_key}']['gamma_index_map']")
    mapping = info["gamma_index_map"][gamma_key]

    rows_obs = mapping.get("rows_obs_names")
    cols_obs = mapping.get("cols_obs_names")
    n_obs = int(getattr(adata, "n_obs", len(getattr(adata, "obs_names", []))))
    if rows_obs is not None and cols_obs is not None:
        src_mask = _mask_indices_from_obs_names(adata, rows_obs)
        tgt_mask = _mask_indices_from_obs_names(adata, cols_obs)
    else:
        rows = np.asarray(mapping["rows_idx"], dtype=int)
        cols = np.asarray(mapping["cols_idx"], dtype=int)
        src_mask = np.zeros(n_obs, dtype=bool)
        tgt_mask = np.zeros(n_obs, dtype=bool)
        src_mask[rows] = True
        tgt_mask[cols] = True

    gamma = adata.uns.get(gamma_key, None)
    return src_mask, tgt_mask, gamma


def _find_gamma_key_for_masks(
    adata: Any,
    source_mask: Any,
    target_mask: Any,
    *,
    scb_key: str = "scb_ot",
) -> Optional[str]:
    info = getattr(adata, "uns", {}).get(scb_key)
    if not info:
        return None
    mapping = info.get("gamma_index_map", {})
    if not mapping:
        return None

    n_obs = int(getattr(adata, "n_obs", len(getattr(adata, "obs_names", []))))

    def _as_bool_mask(mask: Any) -> np.ndarray:
        arr = np.asarray(mask)
        if arr.dtype == bool:
            if arr.size != n_obs:
                raise ValueError("Boolean mask length does not match number of observations.")
            return arr
        idx = arr.astype(int, copy=False)
        out = np.zeros(n_obs, dtype=bool)
        out[idx] = True
        return out

    src_mask = _as_bool_mask(source_mask)
    tgt_mask = _as_bool_mask(target_mask)
    obs_names = np.asarray(getattr(adata, "obs_names", []), dtype=object)
    src_names = set(obs_names[src_mask].tolist())
    tgt_names = set(obs_names[tgt_mask].tolist())
    if not src_names or not tgt_names:
        return None

    for gamma_key, entry in mapping.items():
        rows_obs = entry.get("rows_obs_names")
        cols_obs = entry.get("cols_obs_names")
        if rows_obs is not None and cols_obs is not None:
            if set(rows_obs) == src_names and set(cols_obs) == tgt_names:
                return gamma_key
        else:
            rows_idx = entry.get("rows_idx")
            cols_idx = entry.get("cols_idx")
            if rows_idx is None or cols_idx is None:
                continue
            if np.array_equal(np.sort(rows_idx), np.flatnonzero(src_mask)) and np.array_equal(
                np.sort(cols_idx), np.flatnonzero(tgt_mask)
            ):
                return gamma_key
    return None


def gamma_block(
    gamma: Any,
    rows: Any,
    cols: Any,
    *,
    n_source: Optional[int] = None,
    n_target: Optional[int] = None,
    renormalize: Optional[str] = None,
) -> Any:
    """
    Extract a source/target submatrix from a dense/sparse gamma or transport dict.

    rows/cols are indices or boolean masks aligned to the gamma matrix.
    renormalize: "row", "col", or "both" to normalize mass in the block.
    """
    rows_idx = np.asarray(_to_index_list(rows, n_source), dtype=int)
    cols_idx = np.asarray(_to_index_list(cols, n_target), dtype=int)

    if isinstance(gamma, dict) and "indices" in gamma and "weights" in gamma:
        idx = np.asarray(gamma["indices"])
        n_source_eff = int(idx.shape[0])
        if n_target is None:
            n_target_eff = int(idx.max() + 1) if idx.size else 0
        else:
            n_target_eff = int(n_target)
        gamma = transport_to_sparse(gamma, n_source_eff, n_target_eff)

    if sparse.issparse(gamma):
        block = gamma.tocsr()[rows_idx][:, cols_idx]
    else:
        gamma_arr = np.asarray(gamma)
        block = gamma_arr[np.ix_(rows_idx, cols_idx)]

    if renormalize:
        mode = str(renormalize).lower()
        if mode == "true":
            mode = "row"
        if mode in {"row", "rows", "both"}:
            if sparse.issparse(block):
                row_sums = np.asarray(block.sum(axis=1)).ravel()
                row_sums = np.where(row_sums > 0, row_sums, 1.0)
                block = block.multiply(1.0 / row_sums[:, None])
            else:
                row_sums = block.sum(axis=1, keepdims=True)
                row_sums = np.where(row_sums > 0, row_sums, 1.0)
                block = block / row_sums
        if mode in {"col", "cols", "both"}:
            if sparse.issparse(block):
                col_sums = np.asarray(block.sum(axis=0)).ravel()
                col_sums = np.where(col_sums > 0, col_sums, 1.0)
                block = block.multiply(1.0 / col_sums[None, :])
            else:
                col_sums = block.sum(axis=0, keepdims=True)
                col_sums = np.where(col_sums > 0, col_sums, 1.0)
                block = block / col_sums

    return block
