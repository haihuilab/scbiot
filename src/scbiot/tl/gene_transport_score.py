"""
Gene transport score utilities.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Literal

import numpy as np
from scipy import sparse, special

from ..utils.ot_metadata import (
    _infer_condition_key,
    _infer_rep_key,
    _format_condition_label,
    _masks_from_gamma_key,
    ensure_gamma_for_conditions,
    transport_to_sparse,
)

try:
    from statsmodels.stats.multitest import multipletests as _multipletests
except Exception:  # pragma: no cover - optional dependency
    _multipletests = None


def _as_array(X: Any) -> np.ndarray:
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X, order="C")


def _pooled_var_2sample(Xs: np.ndarray, Xt: np.ndarray, *, ddof: int = 1) -> np.ndarray:
    n1 = int(Xs.shape[0])
    n2 = int(Xt.shape[0])
    n_total = n1 + n2
    if n_total == 0 or n_total <= ddof:
        return np.zeros(Xs.shape[1], dtype=np.float64)
    sum1 = np.sum(Xs, axis=0, dtype=np.float64)
    sum2 = np.sum(Xt, axis=0, dtype=np.float64)
    sumsq1 = np.sum(Xs * Xs, axis=0, dtype=np.float64)
    sumsq2 = np.sum(Xt * Xt, axis=0, dtype=np.float64)
    sum_total = sum1 + sum2
    sumsq_total = sumsq1 + sumsq2
    mean_total = sum_total / float(n_total)
    var = (sumsq_total - float(n_total) * mean_total * mean_total) / float(n_total - ddof)
    return np.maximum(var, 0.0)


def _gamma_sums(gamma: Any) -> Tuple[Any, np.ndarray, np.ndarray]:
    if sparse.issparse(gamma):
        g = gamma.tocsr()
        row_sums = np.asarray(g.sum(axis=1)).ravel()
        col_sums = np.asarray(g.sum(axis=0)).ravel()
        return g, row_sums, col_sums
    g = np.asarray(gamma)
    row_sums = g.sum(axis=1)
    col_sums = g.sum(axis=0)
    return g, row_sums, col_sums


def _barycentric_project(gamma: Any, X_target: np.ndarray, row_sums: np.ndarray) -> np.ndarray:
    proj = gamma @ X_target
    denom = row_sums.reshape(-1, 1)
    denom_safe = np.where(denom > 0, denom, 1.0)
    proj = proj / denom_safe
    if np.any(denom <= 0):
        proj[denom.ravel() <= 0] = 0.0
    return proj


def _pct_expr(X: Any) -> np.ndarray:
    if sparse.issparse(X):
        X = X.tocsr()
        if X.shape[0] == 0:
            return np.zeros(X.shape[1], dtype=np.float64)
        if X.data.size and np.min(X.data) < 0:
            return np.asarray((X > 0).sum(axis=0)).ravel() / float(X.shape[0])
        return X.getnnz(axis=0) / float(X.shape[0])
    X_arr = np.asarray(X)
    if X_arr.size == 0:
        return np.zeros(X_arr.shape[1], dtype=np.float64)
    return (X_arr > 0).mean(axis=0)


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=np.float64)
    p = np.where(np.isfinite(p), p, 1.0)
    p = np.clip(p, 0.0, 1.0)
    n = p.size
    if n == 0:
        return p.astype(np.float32)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * float(n) / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return out.astype(np.float32, copy=False)


def _fdr_bh(pvals: np.ndarray) -> np.ndarray:
    if _multipletests is None:
        return _bh_fdr(pvals)
    return _multipletests(pvals, method="fdr_bh")[1]


def _comparison_label(cond1: Any, cond2: Any, *, cond_vs_rest: bool = False) -> str:
    if cond_vs_rest or (cond1 is None and cond2 is None):
        return "cond_vs_rest"
    if cond1 is None:
        return f"{_format_condition_label(cond2)}_vs_rest"
    if cond2 is None:
        return f"rest_vs_{_format_condition_label(cond1)}"
    return f"{_format_condition_label(cond2)}_vs_{_format_condition_label(cond1)}"


def _gene_names(adata: Any) -> list[str]:
    return [str(name) for name in adata.var_names]


def _ordered_time_labels(
    adata: Any,
    time_key: str,
    *,
    max_bins: int = 10,
) -> Tuple[np.ndarray, list[str]]:
    import pandas as pd

    if time_key not in adata.obs:
        raise KeyError(f"time_key '{time_key}' not found in adata.obs")

    series = adata.obs[time_key]
    labels = np.full(adata.n_obs, None, dtype=object)

    if isinstance(series.dtype, pd.CategoricalDtype):
        cat = series.astype("category")
        categories = [cat_val for cat_val in cat.cat.categories if pd.notna(cat_val)]
        if cat.cat.ordered:
            level_values = categories
        else:
            level_values = [val for val in pd.unique(series) if pd.notna(val)]
        mapping = {val: str(val) for val in level_values}
        values = series.to_numpy(dtype=object, copy=False)
        for idx, value in enumerate(values):
            if pd.notna(value) and value in mapping:
                labels[idx] = mapping[value]
        return labels, [mapping[val] for val in level_values]

    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        valid = numeric.notna().to_numpy()
        if not np.any(valid):
            raise ValueError(f"time_key '{time_key}' has no usable values.")
        unique_values = np.unique(numeric[valid].to_numpy())
        if unique_values.size <= max_bins:
            level_values = np.sort(unique_values)
            for value in level_values:
                labels[(numeric == value).to_numpy()] = str(value)
            return labels, [str(value) for value in level_values]

        n_bins = int(min(max_bins, unique_values.size))
        binned = pd.qcut(numeric, q=n_bins, duplicates="drop")
        if not isinstance(binned.dtype, pd.CategoricalDtype):
            raise ValueError(f"Unable to discretize numeric time_key '{time_key}'.")
        level_values = list(binned.cat.categories)
        values = binned.to_numpy(dtype=object, copy=False)
        for idx, value in enumerate(values):
            if pd.notna(value):
                labels[idx] = str(value)
        return labels, [str(value) for value in level_values]

    level_values = [val for val in pd.unique(series) if pd.notna(val)]
    values = series.to_numpy(dtype=object, copy=False)
    mapping = {val: str(val) for val in level_values}
    for idx, value in enumerate(values):
        if pd.notna(value) and value in mapping:
            labels[idx] = mapping[value]
    return labels, [mapping[val] for val in level_values]


def gene_transport_score(
    Xs: Any,
    Xt: Any,
    gamma: Any,
    *,
    eps: float = 1e-8,
    normalize: bool = True,
    signed: bool = False,
) -> np.ndarray:
    """
    Compute per-gene transport score.

    Tg = sum_{i,j} gamma_ij * (Xt[j,g] - Xs[i,g])^2

    Supports dense and scipy.sparse gamma.
    Inputs:
      Xs: (N, G) numpy array (source cells) or scipy.sparse matrix
      Xt: (M, G) numpy array (target cells) or scipy.sparse matrix
      gamma: (N, M) dense numpy array or scipy.sparse matrix
      eps: small float to avoid div-by-zero in normalization
      normalize: if True, divide by gene variance across pooled Xs/Xt
      signed: if True, compute signed transport: sum gamma_ij * (Xt - Xs)
    Returns:
      score: (G,) array, normalized if requested
    """
    Xs_arr = _as_array(Xs)
    Xt_arr = _as_array(Xt)

    N, G = Xs_arr.shape
    M = Xt_arr.shape[0]
    if Xt_arr.shape[1] != G:
        raise ValueError("Xs and Xt must have same gene columns")

    if isinstance(gamma, dict) and "indices" in gamma and "weights" in gamma:
        gamma = transport_to_sparse(gamma, N, M)

    gamma_eff, row_sums, col_sums = _gamma_sums(gamma)

    if signed:
        score = col_sums @ Xt_arr - row_sums @ Xs_arr
    else:
        term_xt = col_sums @ (Xt_arr * Xt_arr)
        term_xs = row_sums @ (Xs_arr * Xs_arr)
        if N <= M:
            gammaXt = gamma_eff @ Xt_arr
            cross = np.einsum("ij,ij->j", Xs_arr, gammaXt)
        else:
            gT_Xs = gamma_eff.T @ Xs_arr
            cross = np.einsum("ij,ij->j", Xt_arr, gT_Xs)
        score = term_xt + term_xs - 2.0 * cross

    if normalize:
        gene_var = _pooled_var_2sample(Xs_arr, Xt_arr, ddof=1)
        gene_var = np.where(np.isfinite(gene_var), gene_var, 0.0)
        score = score / (gene_var + eps)
    return score


def gene_transport_pvals(
    Xs: Any,
    Xt: Any,
    gamma: Any,
    *,
    n_perms: int = 200,
    normalize: bool = True,
    eps: float = 1e-8,
    signed: bool = False,
    pval_method: str = "perm",
    two_sided: bool = False,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute observed gene transport scores and permutation-based p-values and FDR q-values.

    Permutation: shuffle target cell indices (breaks alignment between gamma and gene expression).
    For pval_method="wald", use a fast normal approximation (signed=True required).
    Returns:
      obs: observed (G,)
      pvals: (G,)
      qvals: (G,)
    """
    rng = np.random.default_rng(random_state)
    Xs_arr = _as_array(Xs)
    Xt_arr = _as_array(Xt)

    gamma_eff = gamma
    if isinstance(gamma, dict) and "indices" in gamma and "weights" in gamma:
        gamma_eff = transport_to_sparse(gamma, Xs_arr.shape[0], Xt_arr.shape[0])

    method = str(pval_method).lower()
    if method not in {"perm", "wald"}:
        raise ValueError("pval_method must be 'perm' or 'wald'.")

    if method == "wald":
        if not signed:
            raise ValueError("pval_method='wald' requires signed=True.")
        gamma_eff, row_sums, col_sums = _gamma_sums(gamma_eff)
        obs_raw = col_sums @ Xt_arr - row_sums @ Xs_arr
        s2_s = Xs_arr.var(axis=0, ddof=1)
        s2_t = Xt_arr.var(axis=0, ddof=1)
        s2_s = np.where(np.isfinite(s2_s), s2_s, 0.0)
        s2_t = np.where(np.isfinite(s2_t), s2_t, 0.0)
        var_sum = s2_t * np.sum(col_sums * col_sums) + s2_s * np.sum(row_sums * row_sums)
        var_sum = np.maximum(var_sum, eps)
        z = obs_raw / np.sqrt(var_sum)
        if two_sided:
            pvals = special.erfc(np.abs(z) / np.sqrt(2.0))
        else:
            pvals = 0.5 * special.erfc(z / np.sqrt(2.0))
    else:
        n_perms = int(max(1, n_perms))
        if signed:
            gamma_eff, row_sums, col_sums = _gamma_sums(gamma_eff)
            base = row_sums @ Xs_arr
            obs_raw = col_sums @ Xt_arr - base
            obs_cmp = np.abs(obs_raw) if two_sided else obs_raw
            counts = np.zeros(obs_raw.shape[0], dtype=np.int64)
            for k in range(n_perms):
                perm_idx = rng.permutation(Xt_arr.shape[0])
                perm_inv = np.empty_like(perm_idx)
                perm_inv[perm_idx] = np.arange(perm_idx.size)
                w = col_sums[perm_inv]
                null = w @ Xt_arr - base
                if two_sided:
                    counts += np.abs(null) >= obs_cmp
                else:
                    counts += null >= obs_cmp
        else:
            obs_raw = gene_transport_score(Xs_arr, Xt_arr, gamma_eff, normalize=False, eps=eps, signed=signed)
            obs_cmp = np.abs(obs_raw) if two_sided else obs_raw
            counts = np.zeros(obs_raw.shape[0], dtype=np.int64)
            for k in range(n_perms):
                perm_idx = rng.permutation(Xt_arr.shape[0])
                Xt_perm = Xt_arr[perm_idx, :]
                null = gene_transport_score(
                    Xs_arr,
                    Xt_perm,
                    gamma_eff,
                    normalize=False,
                    eps=eps,
                    signed=signed,
                )
                if two_sided:
                    counts += np.abs(null) >= obs_cmp
                else:
                    counts += null >= obs_cmp
        pvals = (1 + counts) / (1 + n_perms)
    qvals = _fdr_bh(pvals)

    if normalize:
        gene_var = _pooled_var_2sample(Xs_arr, Xt_arr, ddof=1)
        gene_var = np.where(np.isfinite(gene_var), gene_var, 0.0)
        obs = obs_raw / (gene_var + eps)
    else:
        obs = obs_raw
    return obs, pvals, qvals


def _rank_single_trajectory_transport(
    adata: Any,
    *,
    time_key: str,
    rep_key: Optional[str],
    layer: Optional[str],
    n_perms: int,
    store_key: str,
    normalize: bool,
    eps: float,
    return_top: Optional[int],
    scb_key: str,
    compute_gamma: bool,
    transport_kwargs: Optional[dict[str, Any]],
    random_state: Optional[int] = None,
) -> Any:
    import pandas as pd

    if not compute_gamma:
        raise ValueError("Trajectory transport ranking requires compute_gamma=True.")

    rep_key_use = _infer_rep_key(adata, rep_key, scb_key)
    labels, time_levels = _ordered_time_labels(adata, time_key)
    valid_mask = np.asarray([label is not None for label in labels], dtype=bool)
    if np.count_nonzero(valid_mask) < 2:
        raise ValueError(f"time_key '{time_key}' must define at least two valid cells.")

    labels_valid = labels[valid_mask]
    observed_levels = [level for level in time_levels if np.any(labels_valid == level)]
    if len(observed_levels) < 2:
        raise ValueError(f"time_key '{time_key}' must contain at least two ordered steps.")

    X_all = adata.layers[layer] if layer is not None else adata.X
    rep_all = adata.obsm[rep_key_use]
    if hasattr(rep_all, "toarray"):
        rep_all = rep_all.toarray()
    rep_arr = np.asarray(rep_all, dtype=np.float32, order="C")[valid_mask]

    if sparse.issparse(X_all):
        X_valid = X_all[valid_mask, :]
    else:
        X_valid = _as_array(X_all)[valid_mask, :]

    from ..utils.ot_transport import compute_ot_alignment

    kwargs = dict(transport_kwargs or {})
    z_list: list[np.ndarray] = []
    effect_list: list[np.ndarray] = []
    agree_list: list[np.ndarray] = []
    pct1_list: list[np.ndarray] = []
    pct2_list: list[np.ndarray] = []
    step_labels: list[str] = []

    n_perms_use = int(max(1, n_perms))
    for idx in range(len(observed_levels) - 1):
        level_src = observed_levels[idx]
        level_tgt = observed_levels[idx + 1]
        step_label = f"{level_src}->{level_tgt}"
        source_mask = labels_valid == level_src
        target_mask = labels_valid == level_tgt
        if not np.any(source_mask) or not np.any(target_mask):
            continue

        if sparse.issparse(X_valid):
            Xs_fwd = X_valid[source_mask, :].toarray()
            Xt_fwd = X_valid[target_mask, :].toarray()
        else:
            Xs_fwd = X_valid[source_mask, :]
            Xt_fwd = X_valid[target_mask, :]

        src_emb = rep_arr[source_mask, :]
        tgt_emb = rep_arr[target_mask, :]
        _, gamma_fwd = compute_ot_alignment(src_emb, tgt_emb, **kwargs)
        _, gamma_rev = compute_ot_alignment(tgt_emb, src_emb, **kwargs)
        payload = _signed_score_arrays(
            Xs_fwd,
            Xt_fwd,
            gamma_fwd,
            Xt_fwd,
            Xs_fwd,
            gamma_rev,
            n_perms=n_perms_use,
            normalize=normalize,
            eps=eps,
            # distinct-but-reproducible permutation seed per trajectory step
            random_state=None if random_state is None else int(random_state) + idx,
        )
        z_list.append(payload["z"])
        effect_list.append(payload["score"])
        agree_list.append(payload["agree"])
        pct1_list.append(payload["pct1"])
        pct2_list.append(payload["pct2"])
        step_labels.append(step_label)

    if not z_list:
        raise ValueError(f"time_key '{time_key}' did not yield any adjacent trajectory steps.")

    z_stack = np.stack(z_list, axis=1)
    effect_stack = np.stack(effect_list, axis=1)
    agree_stack = np.stack(agree_list, axis=1)
    pct1_stack = np.stack(pct1_list, axis=1)
    pct2_stack = np.stack(pct2_list, axis=1)

    valid_steps = np.isfinite(z_stack)
    n_steps = z_stack.shape[1]
    z_filled = np.where(valid_steps, z_stack, 0.0)
    trajectory_dynamic_z = np.sqrt(np.nanmean(z_stack * z_stack, axis=1))
    trajectory_dynamic_z = np.where(np.isfinite(trajectory_dynamic_z), trajectory_dynamic_z, 0.0)
    trajectory_z = z_filled.sum(axis=1) / np.sqrt(float(n_steps))
    trajectory_score = np.nanmean(effect_stack, axis=1)
    trajectory_agree = np.nanmean(agree_stack, axis=1)
    trajectory_pct1 = np.nanmean(pct1_stack, axis=1)
    trajectory_pct2 = np.nanmean(pct2_stack, axis=1)

    overall_sign = np.sign(trajectory_z)
    step_sign = np.sign(z_stack)
    direction_match = valid_steps & (step_sign == overall_sign[:, None]) & (overall_sign[:, None] != 0)
    valid_counts = np.maximum(valid_steps.sum(axis=1), 1)
    step_consistency = direction_match.sum(axis=1) / valid_counts

    abs_z = np.where(valid_steps, np.abs(z_stack), -np.inf)
    peak_idx = abs_z.argmax(axis=1)
    peak_step = np.asarray([step_labels[idx] for idx in peak_idx], dtype=object)
    peak_step[~np.isfinite(abs_z.max(axis=1))] = None

    pvals = special.erfc(np.abs(trajectory_z) / np.sqrt(2.0))
    qvals = _fdr_bh(pvals)

    gene_names = _gene_names(adata)
    df = pd.DataFrame(
        {
            "gene": gene_names,
            f"{store_key}_dynamic_z": trajectory_dynamic_z,
            f"{store_key}_z": trajectory_z,
            f"{store_key}_score": trajectory_score,
            f"{store_key}_step_consistency": step_consistency,
            f"{store_key}_agree": trajectory_agree,
            f"{store_key}_peak_step": peak_step,
            f"{store_key}_pval": pvals,
            f"{store_key}_qval": qvals,
            f"{store_key}_pct1": trajectory_pct1,
            f"{store_key}_pct2": trajectory_pct2,
        }
    )
    df = df.sort_values(
        [f"{store_key}_dynamic_z", f"{store_key}_step_consistency", f"{store_key}_agree"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    results = {
        "mode": "trajectory",
        "time_key": time_key,
        "time_levels": observed_levels,
        "step_labels": step_labels,
        "n_steps": int(n_steps),
        "n_perms": n_perms_use,
        "rep_key": rep_key_use,
        "z_by_step": z_stack,
        "effect_by_step": effect_stack,
        "agree_by_step": agree_stack,
        "df": df,
    }
    if return_top is not None:
        top_genes = df["gene"].head(return_top).tolist()
        return results, top_genes
    return results


def _signed_null_stats(
    Xs: Any,
    Xt: Any,
    gamma: Any,
    *,
    n_perms: int,
    normalize: bool,
    eps: float,
    rng: np.random.Generator,
    align_sign: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xs_arr = _as_array(Xs)
    Xt_arr = _as_array(Xt)

    gamma_eff = gamma
    if isinstance(gamma_eff, dict) and "indices" in gamma_eff and "weights" in gamma_eff:
        gamma_eff = transport_to_sparse(gamma_eff, Xs_arr.shape[0], Xt_arr.shape[0])

    gamma_eff, row_sums, col_sums = _gamma_sums(gamma_eff)

    base = row_sums @ Xs_arr
    obs_raw = col_sums @ Xt_arr - base

    if normalize:
        gene_var = _pooled_var_2sample(Xs_arr, Xt_arr, ddof=1)
        gene_var = np.where(np.isfinite(gene_var), gene_var, 0.0)
        denom = gene_var + eps
        obs = obs_raw / denom
    else:
        denom = 1.0
        obs = obs_raw

    obs = obs * float(align_sign)

    n_perms = int(max(1, n_perms))
    mean = np.zeros_like(obs, dtype=np.float64)
    m2 = np.zeros_like(obs, dtype=np.float64)
    for k in range(n_perms):
        perm_idx = rng.permutation(Xt_arr.shape[0])
        perm_inv = np.empty_like(perm_idx)
        perm_inv[perm_idx] = np.arange(perm_idx.size)
        w = col_sums[perm_inv]
        null_raw = w @ Xt_arr - base
        null = null_raw / denom if normalize else null_raw
        null = null * float(align_sign)
        delta = null - mean
        mean += delta / float(k + 1)
        m2 += delta * (null - mean)

    if n_perms > 1:
        var = m2 / float(n_perms - 1)
    else:
        var = np.zeros_like(m2)
    std = np.sqrt(np.maximum(var, eps))
    return obs.astype(np.float64, copy=False), mean, std


def _score_arrays(
    Xs: np.ndarray,
    Xt: np.ndarray,
    gamma: Any,
    *,
    n_perms: int,
    normalize: bool,
    eps: float,
    signed: bool,
    pval_method: str,
    two_sided: bool,
) -> dict[str, np.ndarray]:
    obs, pvals, qvals = gene_transport_pvals(
        Xs,
        Xt,
        gamma,
        n_perms=n_perms,
        normalize=normalize,
        eps=eps,
        signed=signed,
        pval_method=pval_method,
        two_sided=two_sided,
    )
    return {
        "score": obs,
        "pct1": _pct_expr(Xs),
        "pct2": _pct_expr(Xt),
        "pval": pvals,
        "qval": qvals,
    }


def _signed_score_arrays(
    Xs_fwd: np.ndarray,
    Xt_fwd: np.ndarray,
    gamma_fwd: Any,
    Xs_rev: np.ndarray,
    Xt_rev: np.ndarray,
    gamma_rev: Any,
    *,
    n_perms: int,
    normalize: bool,
    eps: float,
    random_state: Optional[int] = None,
) -> dict[str, np.ndarray]:
    pct_1 = _pct_expr(Xs_fwd)
    pct_2 = _pct_expr(Xt_fwd)

    # Seed the permutation null so the signed transport z is reproducible. ``default_rng()`` with no
    # seed draws fresh OS entropy and ignores any global ``np.random.seed`` — pass random_state to fix it.
    rng = np.random.default_rng(random_state)
    obs_fwd, mean_fwd, std_fwd = _signed_null_stats(
        Xs_fwd,
        Xt_fwd,
        gamma_fwd,
        n_perms=n_perms,
        normalize=normalize,
        eps=eps,
        rng=rng,
        align_sign=1.0,
    )
    obs_rev_aligned, mean_rev, std_rev = _signed_null_stats(
        Xs_rev,
        Xt_rev,
        gamma_rev,
        n_perms=n_perms,
        normalize=normalize,
        eps=eps,
        rng=rng,
        align_sign=-1.0,
    )

    z_fwd = (obs_fwd - mean_fwd) / std_fwd
    z_rev = (obs_rev_aligned - mean_rev) / std_rev
    z = (z_fwd + z_rev) / 2.0
    pvals = special.erfc(np.abs(z) / np.sqrt(2.0))
    qvals = _fdr_bh(pvals)

    effect = 0.5 * (obs_fwd + obs_rev_aligned)
    sign_fwd = np.sign(obs_fwd)
    sign_rev = np.sign(obs_rev_aligned)
    min_abs_z = np.minimum(np.abs(z_fwd), np.abs(z_rev))
    agree = np.where(
        sign_fwd == sign_rev,
        1.0 - special.erfc(min_abs_z / np.sqrt(2.0)),
        0.0,
    )

    return {
        "score": effect,
        "pct1": pct_1,
        "pct2": pct_2,
        "z": z,
        "pval": pvals,
        "qval": qvals,
        "agree": agree,
    }


def _score_from_masks(
    adata: Any,
    *,
    source_mask: Any,
    target_mask: Any,
    gamma: Any,
    layer: Optional[str],
    n_perms: int,
    store_key: str,
    normalize: bool,
    eps: float,
    signed: bool,
    pval_method: str,
    two_sided: bool,
    return_top: Optional[int],
    gamma_key: Optional[str] = None,
    comparison: Optional[str] = None,
) -> Any:
    """
    Compute gene transport scores using explicit masks and a transport plan.
    """
    X_all = adata.layers[layer] if layer is not None else adata.X
    if sparse.issparse(X_all):
        Xs = X_all[source_mask, :].toarray() if source_mask is not None else None
        Xt = X_all[target_mask, :].toarray() if target_mask is not None else None
    else:
        X_all_arr = _as_array(X_all)
        Xs = X_all_arr[source_mask, :] if source_mask is not None else None
        Xt = X_all_arr[target_mask, :] if target_mask is not None else None
    if Xs is None or Xt is None:
        raise ValueError("Please pass source_mask and target_mask for transport scoring.")

    payload = _score_arrays(
        Xs,
        Xt,
        gamma,
        n_perms=n_perms,
        normalize=normalize,
        eps=eps,
        signed=signed,
        pval_method=pval_method,
        two_sided=two_sided,
    )
    obs = payload["score"]
    pct1 = payload["pct1"]
    pct2 = payload["pct2"]
    pvals = payload["pval"]
    qvals = payload["qval"]

    store_score = store_key != "transport_score"
    results = {
        "pct1_key": f"{store_key}_pct1",
        "pct2_key": f"{store_key}_pct2",
        "pval_key": f"{store_key}_pval",
        "qval_key": f"{store_key}_qval",
        "n_perms": n_perms,
    }
    if comparison is not None:
        results["transport_comparison"] = comparison
    if store_score:
        results["score_key"] = f"{store_key}_score"
    if gamma_key is not None:
        results["gamma_key"] = gamma_key
    try:
        import pandas as pd

        gene_names = _gene_names(adata)
        df_payload = {
            f"{store_key}_pct1": pct1,
            f"{store_key}_pct2": pct2,
            f"{store_key}_pval": pvals,
            f"{store_key}_qval": qvals,
        }
        if comparison is not None:
            df_payload["transport_comparison"] = comparison
        df = pd.DataFrame(df_payload, index=gene_names)
        df.index.name = "gene"
        df = df.reset_index()
        sort_col = f"{store_key}_z"
        if sort_col in df.columns:
            df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
        results["df"] = df
    except Exception:
        pass

    if return_top is not None:
        top_idx = np.argsort(-obs)[:return_top]
        try:
            top_genes = adata.var_names[top_idx].tolist()
        except Exception:
            top_genes = [str(i) for i in top_idx]
        return results, top_genes
    return results


def rank_transport_score(
    adata: Any,
    *,
    cond1: Optional[Any] = None,
    cond2: Optional[Any] = None,
    cond_key: Optional[str] = None,
    time_key: Optional[str] = None,
    lineage_key: Optional[str] = None,
    lineage_order: Optional[list[Any]] = None,
    rep_key: Optional[str] = None,
    layer: Optional[str] = None,
    n_perms: int = 200,
    store_key: str = "transport_score",
    normalize: bool = True,
    eps: float = 1e-8,
    signed: bool = True,
    pval_method: Optional[Literal["perm", "wald"]] = "wald",
    two_sided: bool = False,
    return_top: Optional[int] = None,
    scb_key: str = "scb_ot",
    compute_gamma: bool = True,
    transport_kwargs: Optional[dict[str, Any]] = None,
    random_state: Optional[int] = None,
) -> Any:
    """
    High-level helper to compute gene transport scores.
    Parameters:
      adata: AnnData object with .X (cells x genes) log-normalized
      cond1: source condition value(s) to select from adata.obs[cond_key]
      cond2: target condition value(s) to select from adata.obs[cond_key]
      cond_key: column in adata.obs for condition labels (defaults to scb_ot batch_key)
      rep_key: embedding key to compute transport plan if missing
      layer: if not None, use adata.layers[layer] as expression
      n_perms: number of permutations for p-values
      store_key: base key used for result column naming
      normalize: whether to normalize by pooled gene variance
      signed: if True, aggregate forward/reverse scores via Stouffer z combination (perm nulls)
      pval_method: 'perm' (default) or 'wald' (fast normal approximation; ignored when signed=True)
      two_sided: if True, compute two-sided p-values (ignored when signed=True; always two-sided)
      return_top: if int, return top-k genes by score
      compute_gamma: if True, compute and store a transport plan when missing
      transport_kwargs: optional kwargs forwarded to compute_ot_alignment
    Returns:
      pandas DataFrame, or (DataFrame, top_genes) if return_top is set
      Note: when signed=True, stores barycentric projections in adata.layers['transport_fwd'] and
            adata.layers['transport_rev'] for source cells in each direction.
      If cond1/cond2 are None and more than two batches are present, returns the
      consensus DataFrame from rank_batch_transport_genes.
    """
    if lineage_order is not None and lineage_key is None:
        raise ValueError("lineage_order requires lineage_key.")
    if lineage_key is not None and time_key is None:
        raise ValueError("lineage_key requires time_key.")
    if time_key is not None and not signed:
        raise ValueError("Trajectory transport ranking requires signed=True.")

    if time_key is not None:
        if time_key not in adata.obs:
            raise KeyError(f"time_key '{time_key}' not found in adata.obs")
        if lineage_key is not None and lineage_key not in adata.obs:
            raise KeyError(f"lineage_key '{lineage_key}' not found in adata.obs")

        import pandas as pd

        if lineage_key is None:
            result = _rank_single_trajectory_transport(
                adata,
                time_key=time_key,
                rep_key=rep_key,
                layer=layer,
                n_perms=n_perms,
                store_key=store_key,
                normalize=normalize,
                eps=eps,
                return_top=return_top,
                scb_key=scb_key,
                compute_gamma=compute_gamma,
                transport_kwargs=transport_kwargs,
                random_state=random_state,
            )
            payload = result[0] if return_top is not None else result
            top_genes = result[1] if return_top is not None else None
            adata.uns[store_key] = {
                "mode": "trajectory",
                "time_key": time_key,
                "lineage_key": None,
                "lineage_order": None,
                "df": payload["df"],
                "time_levels": payload["time_levels"],
                "step_labels": payload["step_labels"],
                "n_steps": payload["n_steps"],
                "n_perms": payload["n_perms"],
                "rep_key": payload["rep_key"],
                "z_by_step": payload["z_by_step"],
                "effect_by_step": payload["effect_by_step"],
                "agree_by_step": payload["agree_by_step"],
            }
            if return_top is not None:
                return payload["df"], top_genes
            return payload["df"]

        lineage_values = adata.obs[lineage_key]
        lineage_order_use = list(lineage_order) if lineage_order is not None else [
            val for val in pd.unique(lineage_values) if pd.notna(val)
        ]
        if not lineage_order_use:
            raise ValueError(f"lineage_key '{lineage_key}' did not yield any usable lineages.")
        per_lineage: dict[Any, Any] = {}
        returned: dict[Any, Any] = {}
        for lineage in lineage_order_use:
            lineage_mask = np.asarray(lineage_values == lineage, dtype=bool)
            if not np.any(lineage_mask):
                raise ValueError(f"No observations found for lineage '{lineage}'.")
            subset = adata[lineage_mask].copy()
            result = _rank_single_trajectory_transport(
                subset,
                time_key=time_key,
                rep_key=rep_key,
                layer=layer,
                n_perms=n_perms,
                store_key=store_key,
                normalize=normalize,
                eps=eps,
                return_top=return_top,
                scb_key=scb_key,
                compute_gamma=compute_gamma,
                transport_kwargs=transport_kwargs,
            )
            payload = result[0] if return_top is not None else result
            top_genes = result[1] if return_top is not None else None
            per_lineage[lineage] = {
                "df": payload["df"],
                "time_levels": payload["time_levels"],
                "step_labels": payload["step_labels"],
                "n_steps": payload["n_steps"],
                "n_perms": payload["n_perms"],
                "rep_key": payload["rep_key"],
                "z_by_step": payload["z_by_step"],
                "effect_by_step": payload["effect_by_step"],
                "agree_by_step": payload["agree_by_step"],
            }
            if return_top is not None:
                returned[lineage] = (payload["df"], top_genes)
            else:
                returned[lineage] = payload["df"]

        adata.uns[store_key] = {
            "mode": "trajectory",
            "time_key": time_key,
            "lineage_key": lineage_key,
            "lineage_order": lineage_order_use,
            "per_lineage": per_lineage,
        }
        return returned

    cond1_use = cond1
    cond2_use = cond2
    cond_key_use = cond_key
    if cond1_use is None and cond2_use is None:
        info = getattr(adata, "uns", {}).get(scb_key, {})
        batches = info.get("batches")
        if cond_key_use is None:
            cond_key_use = info.get("batch_key")
        if batches is None:
            if cond_key_use is None:
                try:
                    cond_key_use = _infer_condition_key(adata, None, scb_key)
                except KeyError:
                    cond_key_use = None
            if cond_key_use is not None and cond_key_use in adata.obs:
                import pandas as pd

                batches = list(pd.unique(adata.obs[cond_key_use].values))

        if batches is not None and len(batches) > 2:
            if cond_key_use is None:
                raise KeyError("cond_key not found; pass cond_key explicitly.")
            store_prefix = "transport_vs_rest_"
            if store_key != "transport_score":
                store_prefix = store_key if store_key.endswith("_") else f"{store_key}_"
            return rank_batch_transport_genes(
                adata,
                cond_key=cond_key_use,
                layer=layer,
                n_perms=n_perms,
                normalize=normalize,
                signed=signed,
                pval_method=pval_method or "wald",
                store_prefix=store_prefix,
                compute_gamma=compute_gamma,
                overwrite=False,
                verbose=False,
            )
        if batches is not None and len(batches) == 2:
            cond1_use, cond2_use = batches[0], batches[1]

    comparison = _comparison_label(cond1_use, cond2_use)
    source_mask, target_mask, gamma, gamma_key = ensure_gamma_for_conditions(
        adata,
        cond1_use,
        cond2_use,
        cond_key=cond_key_use,
        scb_key=scb_key,
        rep_key=rep_key,
        compute_gamma=compute_gamma,
        transport_kwargs=transport_kwargs,
    )
    if gamma is None:
        raise ValueError("No transport plan found; set compute_gamma=True or store gamma in adata.uns.")

    if not signed:
        res = _score_from_masks(
            adata,
            source_mask=source_mask,
            target_mask=target_mask,
            gamma=gamma,
            layer=layer,
            n_perms=n_perms,
            store_key=store_key,
            normalize=normalize,
            eps=eps,
            signed=signed,
            pval_method=pval_method,
            two_sided=two_sided,
            return_top=return_top,
            gamma_key=gamma_key,
            comparison=comparison,
        )
        if return_top is not None:
            results, top_genes = res
            df = results.get("df")
            if df is None:
                return results, top_genes
            return df, top_genes
        if isinstance(res, dict):
            df = res.get("df")
            if df is not None:
                return df
        return res

    cond1_rev = cond2_use
    cond2_rev = cond1_use
    if cond1_rev is None and cond2_rev is None:
        info = getattr(adata, "uns", {}).get(scb_key, {})
        batches = info.get("batches")
        if batches is None or len(batches) != 2:
            raise ValueError("cond1 and cond2 are required unless exactly two batches are registered.")
        cond1_rev, cond2_rev = batches[1], batches[0]

    source_mask_rev, target_mask_rev, gamma_rev, gamma_key_rev = ensure_gamma_for_conditions(
        adata,
        cond1_rev,
        cond2_rev,
        cond_key=cond_key_use,
        scb_key=scb_key,
        rep_key=rep_key,
        compute_gamma=compute_gamma,
        transport_kwargs=transport_kwargs,
    )
    if gamma_rev is None:
        raise ValueError("No reverse transport plan found; set compute_gamma=True or store gamma in adata.uns.")

    X_all = adata.layers[layer] if layer is not None else adata.X
    if sparse.issparse(X_all):
        Xs_fwd = X_all[source_mask, :].toarray()
        Xt_fwd = X_all[target_mask, :].toarray()
        Xs_rev = X_all[source_mask_rev, :].toarray()
        Xt_rev = X_all[target_mask_rev, :].toarray()
    else:
        X_all_arr = _as_array(X_all)
        Xs_fwd = X_all_arr[source_mask, :]
        Xt_fwd = X_all_arr[target_mask, :]
        Xs_rev = X_all_arr[source_mask_rev, :]
        Xt_rev = X_all_arr[target_mask_rev, :]

    gamma_fwd = gamma
    if isinstance(gamma_fwd, dict) and "indices" in gamma_fwd and "weights" in gamma_fwd:
        gamma_fwd = transport_to_sparse(gamma_fwd, Xs_fwd.shape[0], Xt_fwd.shape[0])
    gamma_rev_eff = gamma_rev
    if isinstance(gamma_rev_eff, dict) and "indices" in gamma_rev_eff and "weights" in gamma_rev_eff:
        gamma_rev_eff = transport_to_sparse(gamma_rev_eff, Xs_rev.shape[0], Xt_rev.shape[0])

    gamma_fwd_eff, row_sums_fwd, _ = _gamma_sums(gamma_fwd)
    gamma_rev_eff, row_sums_rev, _ = _gamma_sums(gamma_rev_eff)
    transport_fwd = _barycentric_project(gamma_fwd_eff, Xt_fwd, row_sums_fwd)
    transport_rev = _barycentric_project(gamma_rev_eff, Xt_rev, row_sums_rev)

    n_cells, n_genes = X_all.shape
    dtype_layers = np.result_type(transport_fwd.dtype, transport_rev.dtype, np.float32)
    transport_fwd_layer = np.zeros((n_cells, n_genes), dtype=dtype_layers)
    transport_rev_layer = np.zeros((n_cells, n_genes), dtype=dtype_layers)
    transport_fwd_layer[source_mask, :] = transport_fwd.astype(dtype_layers, copy=False)
    transport_rev_layer[source_mask_rev, :] = transport_rev.astype(dtype_layers, copy=False)
    adata.layers["transport_fwd"] = transport_fwd_layer
    adata.layers["transport_rev"] = transport_rev_layer

    n_perms_use = int(max(1, n_perms))
    eps_use = 1e-8
    payload = _signed_score_arrays(
        Xs_fwd,
        Xt_fwd,
        gamma,
        Xs_rev,
        Xt_rev,
        gamma_rev,
        n_perms=n_perms_use,
        normalize=normalize,
        eps=eps,
    )
    effect = payload["score"]
    pct_1 = payload["pct1"]
    pct_2 = payload["pct2"]
    z = payload["z"]
    pvals = payload["pval"]
    qvals = payload["qval"]
    agree = payload["agree"]
    if store_key == "transport_score":
        effect = -effect
        z = -z

    store_score = store_key != "transport_score"
    results = {
        "pct1_key": f"{store_key}_pct1",
        "pct2_key": f"{store_key}_pct2",
        "z_key": f"{store_key}_z",
        "pval_key": f"{store_key}_pval",
        "qval_key": f"{store_key}_qval",
        "agree_key": f"{store_key}_agree",
        "n_perms": n_perms_use,
    }
    if store_score:
        results["score_key"] = f"{store_key}_score"
    if gamma_key is not None:
        results["gamma_key"] = gamma_key
    if gamma_key_rev is not None:
        results["gamma_key_rev"] = gamma_key_rev
    results["transport_comparison"] = comparison
    try:
        import pandas as pd

        gene_names = _gene_names(adata)
        df_payload = {
            f"{store_key}_pct1": pct_1,
            f"{store_key}_pct2": pct_2,
            f"{store_key}_z": z,
            f"{store_key}_pval": pvals,
            f"{store_key}_qval": qvals,
            f"{store_key}_agree": agree,
            "transport_comparison": comparison,
        }
        df = pd.DataFrame(df_payload, index=gene_names)
        df.index.name = "gene"
        df = df.reset_index()
        sort_col = f"{store_key}_z"
        if sort_col in df.columns:
            df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
        results["df"] = df
    except Exception:
        pass

    if return_top is not None:
        scores = np.where(np.isfinite(effect), effect, -np.inf)
        top_idx = np.argsort(-scores)[:return_top]
        try:
            top_genes = adata.var_names[top_idx].tolist()
        except Exception:
            top_genes = [str(i) for i in top_idx]
        df = results.get("df")
        if df is None:
            return results, top_genes
        return df, top_genes
    df = results.get("df")
    if df is not None:
        return df
    return results


def rank_transport_from_gamma_key(
    adata: Any,
    gamma_key: str,
    *,
    scb_key: str = "scb_ot",
    **rank_kwargs: Any,
) -> Any:
    """
    Convenience wrapper: infer masks/gamma from metadata then compute transport scores.
    """
    source_mask, target_mask, gamma = _masks_from_gamma_key(adata, gamma_key, scb_key=scb_key)
    if gamma is None:
        raise KeyError(f"gamma matrix not found at adata.uns['{gamma_key}']; store it or pass gamma explicitly.")
    return _score_from_masks(
        adata,
        source_mask=source_mask,
        target_mask=target_mask,
        gamma=gamma,
        layer=rank_kwargs.get("layer"),
        n_perms=int(rank_kwargs.get("n_perms", 200)),
        store_key=rank_kwargs.get("store_key", "transport_score"),
        normalize=bool(rank_kwargs.get("normalize", True)),
        eps=float(rank_kwargs.get("eps", 1e-8)),
        signed=bool(rank_kwargs.get("signed", False)),
        pval_method=str(rank_kwargs.get("pval_method", "perm")),
        two_sided=bool(rank_kwargs.get("two_sided", False)),
        return_top=rank_kwargs.get("return_top"),
        gamma_key=gamma_key,
    )



# Batch correction scenario
def aggregate_transport_scores(
    adata: Any,
    *,
    prefix: str = "transport_vs_rest_",
    store_key: str = "transport_consensus",
    min_frac_agree: float = 0.6,
    top_n: Optional[int] = 200,
) -> Any:
    """
    Aggregate per-batch transport results into a consensus ranking.

    Expects per-batch results to have been computed with store_key pattern:
      f"{prefix}{batch}_z"       -> combined z (signed Stouffer) per gene
      f"{prefix}{batch}_agree"   -> agreement strength per gene in [0, 1] (0 = disagree)

    Parameters
    ----------
    adata : AnnData-like object
        Must have gene-level columns in adata.var produced by per-batch scoring.
    prefix : str
        Prefix used for per-batch store_key (default "transport_vs_rest_").
    store_key : str
        Where to save consensus results in adata.uns (default "transport_consensus").
    min_frac_agree : float
        Minimum fraction of batches with agree > 0 used to filter/report top genes.
    top_n : int or None
        If int, return top_n genes (after applying frac_agree threshold). If None, return full table.

    Returns
    -------
    consensus_sorted : pandas.DataFrame
        DataFrame indexed by gene with columns ['mean_z','median_z','frac_agree','n_batches'] sorted
        by (frac_agree desc, mean_z desc). Also stored in `adata.uns[store_key]['df']`.
    """
    # local import to avoid adding a module-wide dependency
    import pandas as pd

    # discover z and agree columns
    z_cols = [c for c in adata.var.columns if c.startswith(prefix) and c.endswith("_z")]
    agree_cols = [c.replace("_z", "_agree") for c in z_cols]

    if len(z_cols) == 0:
        raise KeyError(f"No per-batch z columns found with prefix '{prefix}' in adata.var")

    # nicer labels: remove prefix and suffix
    batch_names = [c[len(prefix) : -2] for c in z_cols]  # strip prefix and trailing "_z"
    # DataFrames of shape (n_genes, n_batches)
    z_df = adata.var[z_cols].copy()
    z_df.columns = batch_names

    # If agree columns don't exist, create a False DF of same shape (conservative)
    existing_agree = [c for c in agree_cols if c in adata.var.columns]
    if len(existing_agree) < len(agree_cols):
        # build agree_df from available columns and fill missing with False
        agree_df = pd.DataFrame(False, index=adata.var_names, columns=batch_names)
        for src_col, bn in zip(agree_cols, batch_names):
            if src_col in adata.var.columns:
                agree_df[bn] = adata.var[src_col] > 0
    else:
        agree_df = adata.var[existing_agree].copy()
        agree_df.columns = batch_names
        agree_df = agree_df > 0

    # compute consensus metrics
    # fill NaNs in z with 0 (no evidence) — matches earlier suggestion
    mean_z = z_df.fillna(0.0).mean(axis=1)
    median_z = z_df.fillna(0.0).median(axis=1)
    n_batches = z_df.shape[1]
    frac_agree = agree_df.fillna(False).astype(int).sum(axis=1) / max(1, n_batches)

    consensus = pd.DataFrame(
        {
            "mean_z": mean_z,
            "median_z": median_z,
            "frac_agree": frac_agree,
            "n_batches": n_batches,
        },
        index=adata.var_names,
    )

    # sort by (frac_agree desc, mean_z desc)
    consensus_sorted = consensus.sort_values(["frac_agree", "mean_z"], ascending=[False, False])

    # apply threshold for top list if requested
    if top_n is not None:
        filtered = consensus_sorted[consensus_sorted["frac_agree"] >= min_frac_agree]
        top_df = filtered.head(top_n)
    else:
        top_df = consensus_sorted[consensus_sorted["frac_agree"] >= min_frac_agree]

    # store results in adata.uns
    if "uns" not in adata.__dict__:
        adata.uns = {}
    adata.uns.setdefault(store_key, {})
    adata.uns[store_key].update(
        {
            "prefix": prefix,
            "min_frac_agree": float(min_frac_agree),
            "n_batches": int(n_batches),
            "df": consensus_sorted,  # store full pandas DataFrame
            "top_df": top_df,
        }
    )

    return consensus_sorted


def rank_batch_transport_genes(
    adata: Any,
    *,
    cond_key: str = "id",
    layer: Optional[str] = None,
    n_perms: int = 200,
    normalize: bool = True,
    signed: bool = True,
    pval_method: str = "wald",
    store_prefix: str = "transport_vs_rest_",
    min_frac_agree: float = 0.6,
    top_n: Optional[int] = 200,
    compute_gamma: bool = True,
    overwrite: bool = False,
    verbose: bool = True,
) -> "pandas.DataFrame":
    """
    Full pipeline: compute per-batch transport (batch vs rest) then aggregate into consensus.

    Example usage:
        df = rank_batch_transport_genes(adata, cond_key="id")
        top_genes = adata.uns["transport_consensus"]["top_df"].index.tolist()

    Per-batch results are kept in memory; only consensus outputs are stored in adata.var.

    Returns a sorted DataFrame with columns:
        gene, transport_score_z, transport_score_score, transport_score_pval, transport_score_qval,
        transport_score_pct1, transport_score_pct2
    """
    import pandas as pd

    if cond_key not in adata.obs.columns:
        raise KeyError(f"cond_key '{cond_key}' not in adata.obs")

    batches = list(pd.unique(adata.obs[cond_key].values))
    if len(batches) < 2:
        raise ValueError("rank_batch_transport_genes requires at least two batches.")

    if layer is not None and layer not in adata.layers:
        raise KeyError(f"layer '{layer}' not found in adata.layers")

    if not signed:
        raise ValueError("rank_batch_transport_genes requires signed=True to compute consensus.")

    X_all = adata.layers[layer] if layer is not None else adata.X
    X_all_arr = None if sparse.issparse(X_all) else _as_array(X_all)
    n_perms_use = int(max(1, n_perms))
    eps_use = 1e-8

    z_list = []
    agree_list = []
    score_list = []
    pct1_list = []
    pct2_list = []
    for b in batches:
        others = [x for x in batches if x != b]
        if not others:
            if verbose:
                print(f"[warning] only one batch present ({b}); skipping.")
            continue

        if verbose:
            print(f"[compute] transport for batch {b} -> rest")

        source_mask, target_mask, gamma, _ = ensure_gamma_for_conditions(
            adata,
            b,
            others,
            cond_key=cond_key,
            scb_key="scb_ot",
            rep_key=None,
            compute_gamma=compute_gamma,
            transport_kwargs=None,
        )
        if gamma is None:
            raise ValueError("No transport plan found; set compute_gamma=True or store gamma in adata.uns.")

        source_mask_rev, target_mask_rev, gamma_rev, _ = ensure_gamma_for_conditions(
            adata,
            others,
            b,
            cond_key=cond_key,
            scb_key="scb_ot",
            rep_key=None,
            compute_gamma=compute_gamma,
            transport_kwargs=None,
        )
        if gamma_rev is None:
            raise ValueError("No reverse transport plan found; set compute_gamma=True or store gamma in adata.uns.")

        if sparse.issparse(X_all):
            Xs_fwd = X_all[source_mask, :].toarray()
            Xt_fwd = X_all[target_mask, :].toarray()
            Xs_rev = X_all[source_mask_rev, :].toarray()
            Xt_rev = X_all[target_mask_rev, :].toarray()
        else:
            Xs_fwd = X_all_arr[source_mask, :]
            Xt_fwd = X_all_arr[target_mask, :]
            Xs_rev = X_all_arr[source_mask_rev, :]
            Xt_rev = X_all_arr[target_mask_rev, :]

        payload = _signed_score_arrays(
            Xs_fwd,
            Xt_fwd,
            gamma,
            Xs_rev,
            Xt_rev,
            gamma_rev,
            n_perms=n_perms_use,
            normalize=normalize,
            eps=eps_use,
        )
        z_list.append(payload["z"])
        agree_list.append(payload["agree"])
        score_list.append(payload["score"])
        pct1_list.append(payload["pct1"])
        pct2_list.append(payload["pct2"])

    if not z_list:
        raise ValueError("No batches available for transport scoring.")

    z_stack = np.stack(z_list, axis=1)
    agree_stack = np.stack(agree_list, axis=1)
    score_stack = np.stack(score_list, axis=1)
    pct1_stack = np.stack(pct1_list, axis=1)
    pct2_stack = np.stack(pct2_list, axis=1)

    z_filled = np.nan_to_num(z_stack, nan=0.0)
    mean_z = z_filled.mean(axis=1)
    median_z = np.median(z_filled, axis=1)
    n_batches = z_stack.shape[1]
    frac_agree = (agree_stack > 0).astype(int).sum(axis=1) / max(1, n_batches)
    consensus = pd.DataFrame(
        {
            "mean_z": mean_z,
            "median_z": median_z,
            "frac_agree": frac_agree,
            "n_batches": n_batches,
        },
        index=adata.var_names,
    )
    consensus_sorted = consensus.sort_values(["frac_agree", "mean_z"], ascending=[False, False])

    if top_n is not None:
        filtered = consensus_sorted[consensus_sorted["frac_agree"] >= min_frac_agree]
        top_df = filtered.head(top_n)
    else:
        top_df = consensus_sorted[consensus_sorted["frac_agree"] >= min_frac_agree]

    if "uns" not in adata.__dict__:
        adata.uns = {}
    adata.uns.setdefault("transport_consensus", {})
    adata.uns["transport_consensus"].update(
        {
            "prefix": store_prefix,
            "min_frac_agree": float(min_frac_agree),
            "n_batches": int(n_batches),
            "df": consensus_sorted,
            "top_df": top_df,
        }
    )

    z_series = -consensus_sorted["mean_z"]
    z_array = z_series.to_numpy(dtype=float, copy=False)
    pvals = special.erfc(np.abs(z_array) / np.sqrt(2.0))
    qvals = _fdr_bh(pvals)

    score_series = pd.Series(
        np.nan_to_num(score_stack, nan=0.0).mean(axis=1),
        index=consensus_sorted.index,
    )
    score_series = -score_series
    pct1_series = pd.Series(np.nanmean(pct1_stack, axis=1), index=consensus_sorted.index)
    pct2_series = pd.Series(np.nanmean(pct2_stack, axis=1), index=consensus_sorted.index)

    score_series = score_series.reindex(consensus_sorted.index)
    pct1_series = pct1_series.reindex(consensus_sorted.index)
    pct2_series = pct2_series.reindex(consensus_sorted.index)
    pval_series = pd.Series(pvals, index=consensus_sorted.index)
    qval_series = pd.Series(qvals, index=consensus_sorted.index)

    consensus_out = pd.DataFrame(
        {
            "gene": consensus_sorted.index.astype(str),
            "transport_score_z": z_series.to_numpy(copy=False),
            "transport_score_score": score_series.to_numpy(copy=False),
            "transport_score_pval": pval_series.to_numpy(copy=False),
            "transport_score_qval": qval_series.to_numpy(copy=False),
            "transport_score_pct1": pct1_series.to_numpy(copy=False),
            "transport_score_pct2": pct2_series.to_numpy(copy=False),
        }
    )
    consensus_out["transport_comparison"] = "cond_vs_rest"

    if overwrite or "transport_score_z" not in adata.var.columns:
        adata.var["transport_score_z"] = z_series
    if overwrite or "transport_score_pval" not in adata.var.columns:
        adata.var["transport_score_pval"] = pval_series
    if overwrite or "transport_score_qval" not in adata.var.columns:
        adata.var["transport_score_qval"] = qval_series
    if overwrite or "transport_score_pct1" not in adata.var.columns:
        adata.var["transport_score_pct1"] = pct1_series
    if overwrite or "transport_score_pct2" not in adata.var.columns:
        adata.var["transport_score_pct2"] = pct2_series

    if verbose:
        top_df = adata.uns["transport_consensus"]["top_df"]
        print(f"[done] consensus computed. top genes: {len(top_df)} (frac_agree >= {min_frac_agree})")
    return consensus_out
