"""
Confusion-style heatmap plots for annotations stored in adata.obs.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


def plot_anndata_confusion(
    adata,
    true_key: str,
    pred_key: str,
    *,
    dropna: bool = True,
    drop_unknown: bool = True,
    unknown_values: Sequence[str] = ("unknown",),
    order_key: Optional[str] = None,
    normalize: str = "pred",
    figsize: Tuple[float, float] = (10, 3.6),
    cmap: str = "Blues",
    linewidths: float = 0.5,
    linecolor: str = "lightgrey",
    annotate_mapping: bool = True,
    mapping_fontsize: int = 10,
    tick_fontsize: int = 14,
    title: Optional[str] = None,
    return_data: bool = False,
) -> Union[Tuple["plt.Figure", "plt.Axes"], Tuple["plt.Figure", "plt.Axes", pd.DataFrame, pd.DataFrame]]:
    """
    Plot a confusion-style heatmap comparing two categorical annotations in adata.obs.

    Parameters
    ----------
    adata : AnnData-like
        Must have `adata.obs` as a pandas DataFrame.
    true_key : str
        Column in `adata.obs` for ground-truth labels.
    pred_key : str
        Column in `adata.obs` for predicted labels.
    dropna : bool
        Drop rows where either label is NA.
    drop_unknown : bool
        Drop rows where either label matches values in `unknown_values` (case-insensitive).
    unknown_values : Sequence[str]
        Values treated as unknown.
    order_key : Optional[str]
        Which column to use for frequency-based ordering. Defaults to `true_key`.
    normalize : str
        "pred" => rows sum to 1 (fraction within predicted class).
        "true" => columns sum to 1 (fraction within true class).
        "all"  => all entries sum to 1.
        "none" => raw counts.
    figsize : (w, h)
        Figure size.
    cmap : str
        Colormap name.
    linewidths, linecolor : float, str
        Heatmap grid styling.
    annotate_mapping : bool
        Show numeric ticks with side mapping text.
    mapping_fontsize, tick_fontsize : int
        Font sizes.
    title : Optional[str]
        Title for the heatmap.
    return_data : bool
        If True, also return (counts_df, normalized_df).

    Returns
    -------
    (fig, ax) or (fig, ax, counts_df, norm_df)
    """
    if not hasattr(adata, "obs"):
        raise TypeError("`adata` must have an `.obs` attribute (pandas DataFrame).")
    if true_key not in adata.obs.columns:
        raise KeyError(f"true_key='{true_key}' not found in adata.obs")
    if pred_key not in adata.obs.columns:
        raise KeyError(f"pred_key='{pred_key}' not found in adata.obs")

    ok_order_key = order_key or true_key
    if ok_order_key not in adata.obs.columns:
        raise KeyError(f"order_key='{ok_order_key}' not found in adata.obs")

    # Import plotting libraries lazily to avoid backend issues for non-plot users.
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.gridspec import GridSpec

    # Extract and clean
    df = adata.obs[[true_key, pred_key, ok_order_key]].copy()
    if dropna:
        df = df.dropna(subset=[true_key, pred_key])

    df[true_key] = df[true_key].astype(str)
    df[pred_key] = df[pred_key].astype(str)

    if drop_unknown and unknown_values:
        unk = {str(v).lower() for v in unknown_values}
        mask = (~df[true_key].str.lower().isin(unk)) & (~df[pred_key].str.lower().isin(unk))
        df = df.loc[mask]

    if df.empty:
        raise ValueError("No rows left after filtering (dropna/drop_unknown).")

    # Frequency-based label ordering (from order_key)
    counts = df[ok_order_key].value_counts(dropna=True)
    base_order = counts.index.astype(str).tolist()

    # Counts table: rows=pred, cols=true
    counts_df = pd.crosstab(df[pred_key], df[true_key])

    # Apply stable ordering: base_order first, then any extra labels
    true_order = [ct for ct in base_order if ct in counts_df.columns] + [
        ct for ct in counts_df.columns if ct not in base_order
    ]
    pred_order = [ct for ct in base_order if ct in counts_df.index] + [
        ct for ct in counts_df.index if ct not in base_order
    ]
    counts_df = counts_df.reindex(index=pred_order, columns=true_order, fill_value=0)

    # Normalize
    if normalize == "pred":
        denom = counts_df.sum(axis=1).replace(0, np.nan)
        norm_df = counts_df.div(denom, axis=0).fillna(0)
        cbar_label = "Fraction within predicted class"
        vmin, vmax = 0, 1
    elif normalize == "true":
        denom = counts_df.sum(axis=0).replace(0, np.nan)
        norm_df = counts_df.div(denom, axis=1).fillna(0)
        cbar_label = "Fraction within true class"
        vmin, vmax = 0, 1
    elif normalize == "all":
        total = counts_df.to_numpy().sum()
        norm_df = counts_df / total if total > 0 else counts_df.astype(float)
        cbar_label = "Fraction of all cells"
        vmin, vmax = 0, float(norm_df.to_numpy().max()) if total > 0 else 1.0
    elif normalize == "none":
        norm_df = counts_df.copy()
        cbar_label = "Cell count"
        vmin, vmax = 0, float(counts_df.to_numpy().max())
    else:
        raise ValueError("normalize must be one of: 'pred', 'true', 'all', 'none'.")

    # Plot layout
    if annotate_mapping:
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = GridSpec(nrows=1, ncols=3, figure=fig, width_ratios=[1.0, 0.05, 1.05])
        ax = fig.add_subplot(gs[0])
        cax = fig.add_subplot(gs[1])
        lax = fig.add_subplot(gs[2])
        lax.axis("off")
    else:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        cax, lax = None, None

    hm = sns.heatmap(
        norm_df,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=linewidths,
        linecolor=linecolor,
        cbar=True,
        cbar_ax=cax,
        cbar_kws={"label": cbar_label},
    )

    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=tick_fontsize)
    cbar.set_label(cbar_label, fontsize=tick_fontsize)

    if title is None:
        title = f"{pred_key} vs {true_key}"
    ax.set_title(title, fontsize=tick_fontsize)

    if annotate_mapping:
        ax.set_yticklabels(list(range(len(norm_df.index))), rotation=0, fontsize=tick_fontsize)
        ax.set_xticklabels(list(range(len(norm_df.columns))), rotation=0, ha="center", fontsize=tick_fontsize)

        map_y = [f"{i} -> {name}" for i, name in enumerate(norm_df.index)]
        map_x = [f"{i} -> {name}" for i, name in enumerate(norm_df.columns)]
        legend_text = (
            "Y / Predicted cell types:\n" + "\n".join(map_y) + "\n\n"
            + "X / Ground truth cell types:\n" + "\n".join(map_x)
        )
        lax.text(
            0.0,
            1.0,
            legend_text,
            va="top",
            ha="left",
            fontsize=mapping_fontsize,
            family="monospace",
        )
    else:
        ax.set_yticklabels(norm_df.index, rotation=0, fontsize=tick_fontsize)
        ax.set_xticklabels(norm_df.columns, rotation=90, ha="center", fontsize=tick_fontsize)

    ax.set_ylabel("Predicted cell types", fontsize=tick_fontsize)
    ax.set_xlabel("Ground truth cell types", fontsize=tick_fontsize)

    # Keep squares when possible (works best when label counts are similar).
    try:
        ax.set_aspect("equal")
    except Exception:
        pass

    if return_data:
        return fig, ax, counts_df, norm_df
    return fig, ax
