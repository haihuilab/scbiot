"""Temporal helpers for spatial-temporal OT (Figure 7 / moscot-style data).

These three functions convert a ``time``/``timepoint`` annotation into a numeric
axis, standardize it, and bin it for forward (adjacent-stage) optimal transport.
They are shared by :func:`scbiot.ot.integrate.integrate_ot` (optional ``time_key``
geometry column) and :func:`scbiot.tl.trajectory_sb.velocity_field_sb_centroids`.

Discrete developmental stages (``E9.5``, ``E10.5`` ...) are handled as an ordered
ordinal axis; genuinely continuous pseudotime is handled by quantile binning.
When ``time_key`` is never supplied these helpers are not called, so non-spatial
RNA/ATAC behavior is unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


def _time_to_numeric(values: Any, mode: str = "auto") -> Tuple[np.ndarray, Dict[str, Any]]:
    mode_norm = "auto" if mode is None else str(mode).lower()
    if mode_norm not in {"auto", "continuous", "ordinal"}:
        raise ValueError(f"time_mode must be 'auto', 'continuous', or 'ordinal' (got {mode!r}).")

    try:
        import pandas as pd
        from pandas.api.types import is_categorical_dtype, is_datetime64_any_dtype, is_numeric_dtype
    except Exception:
        pd = None
        is_categorical_dtype = None
        is_datetime64_any_dtype = None
        is_numeric_dtype = None

    if pd is not None:
        series = values if isinstance(values, pd.Series) else pd.Series(values)
        if is_datetime64_any_dtype(series):
            t_raw = series.view("int64").to_numpy().astype(np.float32)
            return t_raw, {"mode": "continuous", "dtype": "datetime64"}
        is_num = is_numeric_dtype(series)
        if mode_norm == "continuous" or (mode_norm == "auto" and is_num):
            t_raw = np.asarray(series.to_numpy(), dtype=np.float32)
            return t_raw, {"mode": "continuous"}

        if is_categorical_dtype(series):
            cat = series.astype("category")
            if cat.cat.ordered:
                categories = list(cat.cat.categories)
            else:
                categories = list(pd.unique(cat.astype(object)))
                cat = pd.Categorical(cat.astype(object), categories=categories, ordered=True)
        else:
            if is_num and mode_norm == "ordinal":
                categories = sorted(pd.unique(series))
            else:
                categories = list(pd.unique(series.astype(object)))
            cat = pd.Categorical(series.astype(object), categories=categories, ordered=True)
        codes = cat.codes.astype(np.float32)
        codes[codes < 0] = np.nan
        return codes, {"mode": "ordinal", "categories": [str(c) for c in cat.categories]}

    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.datetime64):
        t_raw = arr.astype("datetime64[ns]").astype("int64").astype(np.float32)
        return t_raw, {"mode": "continuous", "dtype": "datetime64"}
    if mode_norm == "continuous" or (mode_norm == "auto" and np.issubdtype(arr.dtype, np.number)):
        return arr.astype(np.float32, copy=False), {"mode": "continuous"}

    uniq = []
    mapping: Dict[str, int] = {}
    out = np.empty(len(arr), dtype=np.float32)
    for i, v in enumerate(arr):
        key = str(v)
        if key not in mapping:
            mapping[key] = len(uniq)
            uniq.append(key)
        out[i] = mapping[key]
    return out, {"mode": "ordinal", "categories": uniq}


def _standardize_time(values: np.ndarray) -> Tuple[np.ndarray, float, float]:
    t = np.asarray(values, dtype=np.float32)
    mask = np.isfinite(t)
    if not mask.any():
        raise ValueError("time values are all missing.")
    mean = float(np.nanmean(t))
    std = float(np.nanstd(t))
    if std < 1e-6:
        std = 1.0
    t = (t - mean) / (std + 1e-6)
    t[~mask] = 0.0
    return t, mean, std


def _bin_time_vector(values: np.ndarray, time_bins: Optional[int], mode: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    meta: Dict[str, Any] = {}
    t = np.asarray(values, dtype=np.float32)
    mask = np.isfinite(t)
    if time_bins is not None:
        bins = int(time_bins)
        if bins < 1:
            raise ValueError(f"time_bins must be >= 1, got {time_bins}")
        if not mask.any():
            raise ValueError("time values are all missing.")
        edges = np.quantile(t[mask], np.linspace(0.0, 1.0, bins + 1))
        edges = np.unique(edges)
        if edges.size <= 1:
            time_bin = np.zeros_like(t, dtype=np.int32)
        else:
            time_bin = np.digitize(t, edges[1:-1], right=True).astype(np.int32)
        time_bin[~mask] = -1
        meta["bin_mode"] = "quantile"
        meta["bins"] = edges.astype(np.float32).tolist()
        return time_bin, meta

    mode_norm = str(mode).lower()
    if mode_norm == "ordinal":
        time_bin = np.where(mask, t, -1).astype(np.int32, copy=False)
        meta["bin_mode"] = "ordinal"
        return time_bin, meta

    if not mask.any():
        raise ValueError("time values are all missing.")
    time_bin = t.copy()
    if not mask.all():
        sentinel = float(np.nanmin(t[mask]) - 1.0)
        time_bin[~mask] = sentinel
        meta["missing_sentinel"] = sentinel
    meta["bin_mode"] = "continuous"
    return time_bin.astype(np.float32, copy=False), meta
