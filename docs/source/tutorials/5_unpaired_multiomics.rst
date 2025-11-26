5. Unpaired Multiomics
===================

:download:`Open notebook <../../../examples/unpaired_Yao-2021_integration.ipynb>` to match modalities with optimal transport when cells are unpaired.

Load the Yao et al. 2021 RNA and ATAC AnnData objects, set reproducibility, and pull in the dependencies used in the notebook:

.. code-block:: python

    import warnings
    from pathlib import Path
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import scanpy as sc

    import numpy as np
    import pandas as pd
    from sklearn.metrics import normalized_mutual_info_score

    import scbiot as scb
    from scbiot.utils import set_seed

    warnings.filterwarnings("ignore")
    set_seed(42)

    dir = Path.cwd()
    verbose = True

    adata_gex = sc.read_h5ad(f"{dir}/inputs/Yao-2021-RNA.h5ad")
    adata_atac = sc.read_h5ad(f"{dir}/inputs/Yao-2021-ATAC.h5ad")

Preprocess ATAC into a smoothed gene activity matrix with the exact pipeline used in the example (remove promoter-proximal peaks, choose variable features, run iterative LSI, and annotate GA):

.. code-block:: python

    gtf_file = f"{dir}/inputs/gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz"

    def prep_unpaired_atac(atac, rna, *, gtf_file: str, verbose: bool = True, lsi_components: int = 51):
        top = scb.pp.remove_promoter_proximal_peaks(
            atac,
            gtf_file=gtf_file,
            promoter_up=2000,
            promoter_down=500,
        )

        top = scb.pp.find_variable_features(
            top,
            batch_key="batch",
            topN=30000,
            add_key="var_features",
            normalize_output=False,
        )

        counts = scb.pp.ensure_csr_f32(top.X)
        counts.data[:] = 1.0
        counts.eliminate_zeros()
        top.X = counts
        top.layers["counts"] = counts

        scb.pp.add_iterative_lsi(
            top,
            n_iter=2,
            per_cluster_union=False,
            sample_cells_pre=None,
            n_components=lsi_components,
            drop_first_component=True,
            add_key="X_lsi",
        )
        atac.obsm["X_lsi"] = top.obsm["X_lsi"]

        ga = scb.pp.annotate_gene_activity(
            atac,
            gtf_file=gtf_file,
            promoter_up=2000,
            promoter_down=500,
            include_gene_body=True,
            weight_by_distance=True,
            tss_decay_bp=2000,
            promoter_priority=True,
            verbose=verbose,
        )
        ga.layers["ga"] = scb.pp.ensure_csr_f32(ga.X)

        scb.pp.knn_smooth_ga_on_atac(
            atac,
            ga,
            n_neighbors=50,
        )

        scb.pp.harmonize_gene_names(rna, ga, rna_name_col="gene_name", verbose=True)
        return ga

    adata_ga = prep_unpaired_atac(adata_atac, adata_gex, gtf_file=gtf_file, verbose=verbose)

Build the aligned co-embedding, transfer labels into the ATAC-derived GA matrix, and assemble the joint object used for visualization:

.. code-block:: python

    coembed = scb.ot.build_aligned_coembedding(adata_gex, adata_ga, verbose=True)
    rep_key = coembed.embedding_key
    print(f"[HVG] Joint (shared) HVGs: {len(coembed.genes):,}")

    if "cell_type" in adata_atac.obs and "cell_type_original" not in adata_atac.obs:
        adata_atac.obs["cell_type_original"] = adata_atac.obs["cell_type"].astype(str)

    scb.ot.label_transfer_shared_pca(
        adata_gex,
        adata_ga,
        label_key="cell_type",
        use_rep="X_pca_shared_aligned",
        min_conf=0.55,
    )

    true_labels = adata_ga.obs.get("cell_type_original", adata_ga.obs.get("cell_type"))
    pred_labels = adata_ga.obs["pred_cell_type"]
    pred_labels_str = pred_labels.astype(str)
    mask = (~pd.isna(true_labels)) & (~pd.isna(pred_labels)) & (pred_labels_str != "unknown")
    if mask.any():
        nmi_score = normalized_mutual_info_score(
            true_labels[mask].astype(str),
            pred_labels_str[mask],
        )
        print(f"[Metrics] NMI(pred_cell_type vs cell_type) = {nmi_score:.4f}")

    joint_rep_key = rep_key
    adata = scb.pp.assemble_joint_embedding(joint_rep_key, {"RNA": adata_gex, "ATAC_GA": adata_ga})

Compute a shared neighborhood graph and UMAP layout, then plot modality and cell type overlays exactly as in the notebook:

.. code-block:: python

    sc.pp.neighbors(adata, use_rep="X_pca_shared_aligned", n_neighbors=50, metric="cosine")
    sc.tl.umap(adata, min_dist=0.3, spread=1.0, random_state=0)

    counts = adata_gex.obs["cell_type"].value_counts()
    celltype_order = counts.index.tolist()
    adata.obs["cell_type"] = adata.obs["cell_type"].astype(
        pd.CategoricalDtype(categories=celltype_order, ordered=True)
    )

    sc.settings._vector_friendly = True
    mpl.rcParams["axes.edgecolor"] = "black"
    mpl.rcParams["axes.linewidth"] = 1.0

    def force_border(ax):
        ax.set_axis_on()
        ax.set_frame_on(True)
        ax.patch.set_visible(True)
        for side in ax.spines.values():
            side.set_visible(True)
            side.set_color("black")
            side.set_linewidth(1.0)

    methods = ["X_pca_shared_aligned"]
    ncols = 2 * len(methods)
    fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 4.2), squeeze=False)
    axes = axes[0]

    for i, method in enumerate(methods):
        axL, axR = axes[2 * i], axes[2 * i + 1]

        sc.pl.embedding(
            adata,
            basis="umap",
            color="modality",
            frameon=True,
            ax=axL,
            show=False,
            legend_loc="right margin",
            legend_fontsize=16,
            title="Modality",
        )
        axL.set_box_aspect(1)
        axL.set_xlabel("UMAP1")
        axL.set_ylabel("UMAP2")
        force_border(axL)

        sc.pl.embedding(
            adata,
            basis="umap",
            color="cell_type",
            frameon=True,
            ax=axR,
            show=False,
            legend_loc="right margin",
            legend_fontsize=16,
            title="",
        )
        axR.set_box_aspect(1)
        axR.set_xlabel("UMAP1")
        axR.set_ylabel("UMAP2")
        force_border(axR)

    plt.tight_layout()

.. figure:: /_static/plots/unpaired_Yao-2021_integration_plot01.png
   :alt: UMAP coloured by modality for the unpaired RNA+ATAC coembedding.
   :width: 100%

Optionally, build the confusion matrix comparing predicted and truth labels (the script will skip plotting if the paired labels are not present):

.. code-block:: python

    import seaborn as sns
    from matplotlib.gridspec import GridSpec

    pred_col_candidates = ["pred_cell_type", "predicted_cell_type"]
    truth_col_candidates = ["cell_type_original", "cell_type", "final_cell_label"]
    obs_candidates = []

    if "adata" in locals():
        obs_view = adata.obs.copy()
        if "modality" in obs_view.columns:
            obs_view = obs_view[obs_view["modality"] == "ATAC"]
        obs_candidates.append(("assembled ATAC", obs_view))
    if "adata_ga" in locals():
        obs_candidates.append(("gene activity", adata_ga.obs.copy()))
    if "adata_atac" in locals():
        obs_candidates.append(("raw ATAC", adata_atac.obs.copy()))

    selected_obs = None
    selected_source = None
    pred_column = None
    truth_column = None
    for label, frame in obs_candidates:
        if frame is None or frame.empty:
            continue
        pred_column = next((col for col in pred_col_candidates if col in frame.columns), None)
        truth_column = next((col for col in truth_col_candidates if col in frame.columns), None)
        if pred_column and truth_column:
            selected_obs = frame[[pred_column, truth_column]].copy()
            selected_obs.columns = ["predicted", "truth"]
            selected_source = label
            break

    if selected_obs is not None:
        selected_obs = selected_obs.dropna().copy()
        selected_obs["predicted"] = selected_obs["predicted"].astype(str)
        selected_obs["truth"] = selected_obs["truth"].astype(str)
        mask = (selected_obs["predicted"].str.lower() != "unknown") & (selected_obs["truth"].str.lower() != "unknown")
        selected_obs = selected_obs.loc[mask]

    if selected_obs is not None and not selected_obs.empty:
        contingency = pd.crosstab(selected_obs["predicted"], selected_obs["truth"])

        truth_order = [ct for ct in celltype_order if ct in contingency.columns]
        truth_order += [ct for ct in contingency.columns if ct not in truth_order]
        pred_order = [ct for ct in celltype_order if ct in contingency.index]
        pred_order += [ct for ct in contingency.index if ct not in pred_order]

        contingency = contingency.reindex(index=pred_order, columns=truth_order, fill_value=0)
        row_sums = contingency.sum(axis=1).replace(0, np.nan)
        contingency_norm = contingency.div(row_sums, axis=0).fillna(0)

        fig = plt.figure(figsize=(10, 3.6), constrained_layout=True)
        gs = GridSpec(nrows=1, ncols=3, figure=fig, width_ratios=[1.0, 0.05, 1.05])
        ax = fig.add_subplot(gs[0])
        cax = fig.add_subplot(gs[1])
        lax = fig.add_subplot(gs[2])

        hm = sns.heatmap(
            contingency_norm,
            ax=ax,
            cmap="Blues",
            vmin=0,
            vmax=1,
            linewidths=0.5,
            linecolor="lightgrey",
            cbar=True,
            cbar_ax=cax,
            cbar_kws={"label": "Fraction of predicted class"},
        )
        cbar = hm.collections[0].colorbar
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label("Fraction of predicted class", fontsize=14)

        ax.set_yticklabels(list(range(len(contingency_norm.index))), rotation=0, fontsize=14)
        ax.set_xticklabels(list(range(len(contingency_norm.columns))), rotation=0, ha="center", fontsize=14)
        ax.set_ylabel("Predicted cell types", fontsize=14)
        ax.set_xlabel("Ground truth cell types", fontsize=14)
        ax.set_aspect("equal")

        lax.axis("off")
        map_y = [f"{i} -> {name}" for i, name in enumerate(contingency_norm.index)]
        map_x = [f"{i} -> {name}" for i, name in enumerate(contingency_norm.columns)]
        legend_text = (
            f"Source: {selected_source}\n"
            + "Y / Predicted cell types:\n" + "\n".join(map_y) + "\n\n"
            + "X / Ground truth cell types:\n" + "\n".join(map_x)
        )
        lax.text(0.0, 1.0, legend_text, va="top", ha="left", fontsize=10, family="monospace")

        plt.show()

.. figure:: /_static/plots/unpaired_Yao-2021_integration_plot02.png
   :alt: UMAP coloured by cell type for the shared embedding.
   :width: 100%
