"""
Barplot for differential transport energy by cell type.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns


def transport_energy_barplot(
    adata,
    *,
    cell_type_key,
    condition_key,
    cond1,
    cond2,
    energy_key: str = "transport_energy",
    ax=None,
):
    """
    Barplot showing differential transport energy between conditions.

    Parameters
    ----------
    adata
        AnnData object
    cell_type_key
        Column in adata.obs for cell types
    condition_key
        Column in adata.obs for condition
    cond1
        Baseline condition
    cond2
        Perturbed condition
    energy_key
        Column storing transport energy
    ax
        matplotlib axis
    """

    if energy_key not in adata.obs:
        raise KeyError(f"{energy_key} not found in adata.obs")

    obs_df = adata.obs[[cell_type_key, condition_key, energy_key]].copy()

    baseline = obs_df.loc[obs_df[condition_key] == cond1, energy_key].mean()

    pert_means = (
        obs_df.loc[obs_df[condition_key] == cond2]
        .groupby(cell_type_key)[energy_key]
        .mean()
    )

    delta = pert_means - baseline

    plot_df = delta.sort_values(ascending=False).reset_index(name="delta")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    order = plot_df[cell_type_key].tolist()

    sns.barplot(
        data=plot_df,
        x="delta",
        y=cell_type_key,
        order=order,
        errorbar=None,
        ax=ax,
    )

    ax.set_xlabel(f"Mean transport_energy ({cond2} - {cond1})")
    ax.set_ylabel("Cell type")
    ax.set_title("Cell-type transport impact")

    return ax
