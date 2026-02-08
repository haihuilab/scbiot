"""
Cell type mean-gene correlation plots and summaries.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import pearsonr, spearmanr


def _get_X_and_genes(adata, layer: Optional[str] = None, use_raw: bool = False):
    """
    Returns (X, var_names) where X is cells x genes.
    Priority:
      - if use_raw and adata.raw exists -> adata.raw.X / adata.raw.var_names
      - else if layer is not None -> adata.layers[layer] / adata.var_names
      - else -> adata.X / adata.var_names
    """
    if use_raw:
        if adata.raw is None:
            raise ValueError("use_raw=True but adata.raw is None.")
        return adata.raw.X, adata.raw.var_names

    if layer is not None:
        if layer not in adata.layers:
            raise ValueError(
                f"layer='{layer}' not found in adata.layers. Available: {list(adata.layers.keys())}"
            )
        return adata.layers[layer], adata.var_names

    return adata.X, adata.var_names


def _mean_over_cells(X, cell_idx: np.ndarray):
    """
    X: cells x genes (dense or sparse)
    cell_idx: 1D array-like indices (int) selecting rows
    returns: 1D np.array of gene means
    """
    if len(cell_idx) == 0:
        return None
    X_sub = X[cell_idx, :]
    if sp.issparse(X_sub):
        return np.asarray(X_sub.mean(axis=0)).ravel()
    return np.asarray(X_sub.mean(axis=0)).ravel()


def _safe_corr(x: np.ndarray, y: np.ndarray, method: str = "pearson") -> float:
    """
    Correlation on finite values only.
    Returns float (np.nan if insufficient data).
    """
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    if method == "pearson":
        return pearsonr(x[mask], y[mask])[0]
    if method == "spearman":
        return spearmanr(x[mask], y[mask]).correlation
    raise ValueError("method must be 'pearson' or 'spearman'")


def celltype_gene_mean_correlation(
    adata_ref,
    adata_query,
    ref_group_key: str = "cell_type",
    query_group_key: str = "pred_cell_type",
    layer: Optional[str] = None,
    use_raw: bool = False,
    cell_types: Optional[Iterable[str]] = None,
    min_cells: int = 20,
    eps: float = 1e-8,
    panel_source: Optional[Mapping[str, str] | pd.Series | pd.DataFrame] = None,
    ncols: int = 4,
    figsize_per_panel: Tuple[float, float] = (3.2, 3.2),
):
    """
    For each cell type label L:
      - mean_ref(gene | cell_type=L) vs mean_query(gene | pred_cell_type=L)
      - correlation across genes (log10 means)
      - log-log scatter plot with y=x and annotation

    Returns:
      fig, corr_df, means_ref_df, means_query_df

    means_*_df: genes x celltypes mean matrices
    corr_df: per-celltype summary table

    Examples
    --------
    Basic usage:

    >>> import scbiot as scb
    >>> fig, corr_df, means_ref, means_query = scb.pl.celltype_gene_mean_correlation(
    ...     adata_ref,
    ...     adata_query,
    ...     ref_group_key="cell_type",
    ...     query_group_key="pred_cell_type",
    ... )
    """
    # Import matplotlib lazily to avoid import-time backend issues for non-plot users.
    import matplotlib.pyplot as plt

    # --- get matrices and genes ---
    Xr, genes_r = _get_X_and_genes(adata_ref, layer=layer, use_raw=use_raw)
    Xq, genes_q = _get_X_and_genes(adata_query, layer=layer, use_raw=use_raw)

    # --- common genes, keep ref order ---
    genes_q_set = set(map(str, genes_q))
    common_genes = [g for g in map(str, genes_r) if g in genes_q_set]
    if len(common_genes) == 0:
        raise ValueError("No common genes found between adata_ref and adata_query.")

    # subset both to common genes, same order
    adata_ref_sub = adata_ref[:, common_genes]
    adata_query_sub = adata_query[:, common_genes]
    Xr, genes_r = _get_X_and_genes(adata_ref_sub, layer=layer, use_raw=use_raw)
    Xq, genes_q = _get_X_and_genes(adata_query_sub, layer=layer, use_raw=use_raw)
    genes = pd.Index(map(str, genes_r))

    # --- cell types to process ---
    ref_labels = pd.Series(adata_ref_sub.obs[ref_group_key].astype(str).values)
    qry_labels = pd.Series(adata_query_sub.obs[query_group_key].astype(str).values)

    if cell_types is None:
        cell_types = sorted(set(ref_labels) & set(qry_labels))
    else:
        cell_types = [str(x) for x in cell_types]

    if len(cell_types) == 0:
        raise ValueError("No overlapping cell type labels to compare.")

    # --- panel_source handling (optional) ---
    gene_to_panel = None
    if panel_source is not None:
        if isinstance(panel_source, (pd.Series, Mapping)):
            gene_to_panel = pd.Series(panel_source)
        elif isinstance(panel_source, pd.DataFrame):
            if not {"gene", "gene_panel"}.issubset(panel_source.columns):
                raise ValueError("panel_source DataFrame must contain columns: ['gene','gene_panel']")
            gene_to_panel = panel_source.set_index("gene")["gene_panel"]
        else:
            raise ValueError("panel_source must be a DataFrame, dict, or Series.")

        # align to current gene set
        gene_to_panel = gene_to_panel.reindex(genes)

    # --- precompute indices per label for speed ---
    ref_idx_by_label = {ct: np.where(ref_labels.values == ct)[0] for ct in cell_types}
    qry_idx_by_label = {ct: np.where(qry_labels.values == ct)[0] for ct in cell_types}

    # --- compute mean matrices: genes x cell_types ---
    means_ref = {}
    means_qry = {}
    keep_cell_types = []

    for ct in cell_types:
        idx_r = ref_idx_by_label[ct]
        idx_q = qry_idx_by_label[ct]
        if len(idx_r) < min_cells or len(idx_q) < min_cells:
            continue

        mr = _mean_over_cells(Xr, idx_r)
        mq = _mean_over_cells(Xq, idx_q)
        if mr is None or mq is None:
            continue

        means_ref[ct] = mr
        means_qry[ct] = mq
        keep_cell_types.append(ct)

    if len(keep_cell_types) == 0:
        raise ValueError(f"No cell types passed min_cells={min_cells} in both datasets.")

    means_ref_df = pd.DataFrame(means_ref, index=genes)
    means_qry_df = pd.DataFrame(means_qry, index=genes)

    # --- per-celltype correlation + plots ---
    n = len(keep_cell_types)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n / ncols))
    fig_w = figsize_per_panel[0] * ncols
    fig_h = figsize_per_panel[1] * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    records = []
    axes_flat = axes.ravel()

    for i, ct in enumerate(keep_cell_types):
        ax = axes_flat[i]
        x = means_qry_df[ct].values.astype(float)  # query on x-axis
        y = means_ref_df[ct].values.astype(float)  # ref on y-axis

        # log10 transform with masking for >0
        mask_pos = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
        x_log = np.log10(x[mask_pos] + eps)
        y_log = np.log10(y[mask_pos] + eps)

        r_p = _safe_corr(x_log, y_log, method="pearson")
        r_s = _safe_corr(x_log, y_log, method="spearman")

        # scatter (optionally colored by gene_panel)
        if gene_to_panel is not None:
            panels = gene_to_panel.values[mask_pos]
            uniq = pd.unique(pd.Series(panels).dropna())
            cmap = {p: j for j, p in enumerate(uniq)}
            c = np.array([cmap.get(p, -1) for p in panels])
            ax.scatter(x[mask_pos] + eps, y[mask_pos] + eps, s=8, c=c, alpha=0.7)
        else:
            ax.scatter(x[mask_pos] + eps, y[mask_pos] + eps, s=8, alpha=0.7)

        # identity line in data space on log axes
        if mask_pos.any():
            vmin = min((x[mask_pos] + eps).min(), (y[mask_pos] + eps).min())
            vmax = max((x[mask_pos] + eps).max(), (y[mask_pos] + eps).max())
            ax.plot([vmin, vmax], [vmin, vmax], linestyle="--", linewidth=1)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(ct, fontsize=10)
        ax.set_xlabel("Query (pred_cell_type) mean", fontsize=9)
        ax.set_ylabel("Ref (cell_type) mean", fontsize=9)

        ax.text(
            0.02,
            0.98,
            f"Pearson r={r_p:.3f}\nSpearman ρ={r_s:.3f}\n"
            f"n_ref={len(ref_idx_by_label[ct])}, n_query={len(qry_idx_by_label[ct])}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", alpha=0.15),
        )

        records.append(
            {
                "cell_type": ct,
                "n_ref": int(len(ref_idx_by_label[ct])),
                "n_query": int(len(qry_idx_by_label[ct])),
                "n_genes_used": int(mask_pos.sum()),
                "pearson_r_log10means": float(r_p) if np.isfinite(r_p) else np.nan,
                "spearman_rho_log10means": float(r_s) if np.isfinite(r_s) else np.nan,
            }
        )

    # blank unused axes
    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.tight_layout()

    corr_df = (
        pd.DataFrame(records)
        .sort_values("pearson_r_log10means", ascending=False)
        .reset_index(drop=True)
    )
    return fig, corr_df, means_ref_df, means_qry_df


def celltype_predtype_mean_corr_heatmap(
    adata_ref,
    adata_query,
    ref_group_key: str = "cell_type",
    query_group_key: str = "pred_cell_type",
    layer: Optional[str] = None,
    use_raw: bool = False,
    genes_use: Optional[Iterable[str]] = None,
    min_cells: int = 20,
    transform: str = "log1p",
    eps: float = 1e-8,
    method: str = "pearson",
    figsize: Tuple[float, float] = (10, 7),
    cmap: str = "magma",
    vmin: float = -1.0,
    vmax: float = 1.0,
):
    """
    Compute a cell-type (ref) x pred-type (query) correlation heatmap of per-gene mean expression.

    Returns:
      fig, ax, corr_df (rows=ref cell_type, cols=query pred_cell_type),
      means_ref_df, means_query_df (genes x types mean matrices)

    Examples
    --------
    Basic usage:

    >>> import scbiot as scb
    >>> fig, ax, corr_df, means_ref, means_query = (
    ...     scb.pl.celltype_predtype_mean_corr_heatmap(adata_ref, adata_query)
    ... )
    """
    # Import matplotlib lazily to avoid import-time backend issues for non-plot users.
    import matplotlib.pyplot as plt

    Xr, genes_r = _get_X_and_genes(adata_ref, layer=layer, use_raw=use_raw)
    Xq, genes_q = _get_X_and_genes(adata_query, layer=layer, use_raw=use_raw)

    # common genes (keep ref order)
    genes_q_set = set(map(str, genes_q))
    common = [g for g in map(str, genes_r) if g in genes_q_set]
    if genes_use is not None:
        genes_use = set(map(str, genes_use))
        common = [g for g in common if g in genes_use]
    if len(common) == 0:
        raise ValueError("No genes left after intersection / genes_use filtering.")

    adata_ref_sub = adata_ref[:, common]
    adata_query_sub = adata_query[:, common]
    Xr, genes_r = _get_X_and_genes(adata_ref_sub, layer=layer, use_raw=use_raw)
    Xq, genes_q = _get_X_and_genes(adata_query_sub, layer=layer, use_raw=use_raw)
    genes = pd.Index(map(str, genes_r))

    ref_labels = pd.Series(adata_ref_sub.obs[ref_group_key].astype(str).values)
    qry_labels = pd.Series(adata_query_sub.obs[query_group_key].astype(str).values)

    ref_types = sorted(ref_labels.unique())
    qry_types = sorted(qry_labels.unique())

    ref_idx = {t: np.where(ref_labels.values == t)[0] for t in ref_types}
    qry_idx = {t: np.where(qry_labels.values == t)[0] for t in qry_types}

    means_ref = {}
    for t in ref_types:
        if len(ref_idx[t]) >= min_cells:
            means_ref[t] = _mean_over_cells(Xr, ref_idx[t])
    means_qry = {}
    for t in qry_types:
        if len(qry_idx[t]) >= min_cells:
            means_qry[t] = _mean_over_cells(Xq, qry_idx[t])

    if len(means_ref) == 0 or len(means_qry) == 0:
        raise ValueError(
            f"No groups passed min_cells={min_cells} (ref={len(means_ref)}, query={len(means_qry)})."
        )

    means_ref_df = pd.DataFrame(means_ref, index=genes)
    means_qry_df = pd.DataFrame(means_qry, index=genes)

    # transform
    if transform == "log1p":
        A = np.log1p(means_ref_df.values.astype(float))
        B = np.log1p(means_qry_df.values.astype(float))
    elif transform == "log10":
        A = np.log10(means_ref_df.values.astype(float) + eps)
        B = np.log10(means_qry_df.values.astype(float) + eps)
    else:
        raise ValueError("transform must be 'log1p' or 'log10'.")

    # correlation matrix: ref_types x qry_types
    ref_cols = list(means_ref_df.columns)
    qry_cols = list(means_qry_df.columns)
    corr = np.full((len(ref_cols), len(qry_cols)), np.nan, dtype=float)

    for i, _ in enumerate(ref_cols):
        x = A[:, i]
        for j, _ in enumerate(qry_cols):
            y = B[:, j]
            corr[i, j] = _safe_corr(x, y, method=method)

    corr_df = pd.DataFrame(corr, index=ref_cols, columns=qry_cols)

    # plot heatmap (matplotlib only)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        corr_df.values,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation="nearest",
    )

    ax.set_xticks(np.arange(corr_df.shape[1]))
    ax.set_yticks(np.arange(corr_df.shape[0]))
    ax.set_xticklabels(corr_df.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr_df.index, fontsize=8)

    ax.set_title(
        f"{method.title()} correlation of per-gene mean expression\n"
        f"ref: {ref_group_key}  vs  query: {query_group_key}  "
        f"(genes={len(genes)}, min_cells={min_cells}, {transform})",
        fontsize=10,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Correlation", fontsize=9)

    fig.tight_layout()
    return fig, ax, corr_df, means_ref_df, means_qry_df
