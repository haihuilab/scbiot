SNARE-seq integration (Chen 2019)
==================================

:download:`Open notebook <../../../examples/paired_Chen-2019_integration.ipynb>` to align paired transcriptome and chromatin profiles from SNARE-seq.

Environment and inputs
----------------------

.. code-block:: python

    import warnings
    warnings.filterwarnings("ignore")

    import scanpy as sc
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scbiot as scb
    from scbiot.utils import set_seed

    from pathlib import Path
    dir = Path.cwd()
    set_seed(42)

    adata_gex = sc.read(
        f"{dir}/inputs/Chen-2021-RNA.h5ad",
        backup_url="https://figshare.com/ndownloader/files/59742638",
    )
    adata_atac = sc.read(
        f"{dir}/inputs/Chen-2021-ATAC.h5ad",
        backup_url="https://figshare.com/ndownloader/files/59742644",
    )

    # ensure identical cell ordering
    adata_atac = adata_atac[adata_gex.obs_names].copy()

    adata_gex.var["feature_types"] = "GEX"
    adata_atac.var["feature_types"] = "ATAC"
    adata_gex.var_names_make_unique()
    adata_atac.var_names_make_unique()

    def ensure_counts_layer(a):
        if "counts" not in a.layers:
            a.layers["counts"] = a.X.copy()
        a.layers["counts"] = scb.pp.ensure_csr_f32(a.layers["counts"])
        a.X = a.layers["counts"]

    ensure_counts_layer(adata_gex)
    ensure_counts_layer(adata_atac)

    adata = ad.concat(
        [adata_gex, adata_atac],
        axis=1,
        join="outer",
        merge="first",
    )

Prep peaks and LSI (matching the Chen 2019 paired dataset) before integration:

.. code-block:: python

    adata_top = scb.pp.remove_promoter_proximal_peaks(
        adata_atac,
        gtf_file=f"{dir}/inputs/gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz",
    )

    scb.pp.find_variable_features(adata_top, topN=30000, batch_key="batch")
    scb.pp.add_iterative_lsi(
        adata_top,
        n_components=31,
        drop_first_component=True,
        add_key="X_lsi",
    )

    adata_atac.obsm["X_lsi"] = adata_top.obsm["X_lsi"]
    adata.obsm["X_lsi"] = adata_top.obsm["X_lsi"]

Integrate paired RNA+ATAC views with OT, then train and export the latent space:

.. code-block:: python

    adata, metrics = scb.ot.integrate(
        adata,
        modality="paired",
        obsm_key="X_pca",          # base view for geometry/smoothing
        batch_key="batch",
        out_key="X_ot",
        mode="ufgw_barycenter",
        view_keys=("X_pca", "X_lsi"),
        K_pseudo=24,
    )
    print(metrics)

    sc.pp.neighbors(adata, use_rep='X_ot', n_neighbors=8)
    sc.tl.umap(adata, min_dist=0.10, spread=1.0)
    sc.tl.leiden(adata, key_added='leiden_X_ot', resolution=1.)

    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

    df = pd.DataFrame(adata.obsm["X_ot"], index=adata.obs.index)
    df["batch"] = adata.obs["batch"]

    df["target"] = adata.obs["cell_type"]
    counts = df["target"].value_counts()
    mapping = {cat: idx for idx, cat in enumerate(counts.index)}
    df["target"] = df["target"].map(mapping)

    df["pseudo"] = adata.obs["leiden_X_ot"]
    counts = df["pseudo"].value_counts()
    mapping = {cat: idx for idx, cat in enumerate(counts.index)}
    df["pseudo"] = df["pseudo"].map(mapping)

    true_labels = df["target"]
    pred_labels = df["pseudo"]
    nmi_score = normalized_mutual_info_score(true_labels, pred_labels)
    ari_score = adjusted_rand_score(true_labels, pred_labels)
    print(f"NMI: {nmi_score:.4f} | ARI: {ari_score:.4f}")

    import matplotlib.pyplot as plt
    import scanpy as sc

    methods = ["X_ot"]

    # 2 rows x len(methods) columns
    fig, axes = plt.subplots(
        2,
        len(methods),
        figsize=(4 * len(methods), 8),
        squeeze=False  # ensures axes is a 2D array
    )

    for col, method in enumerate(methods):
        # 1) Top row (row=0): color by "batch"
        sc.pl.embedding(
            adata,
            basis=f"X_umap",  # The coordinates stored in adata.obsm["X_umap_{method}"]
            color="cell_type",            # Assume adata.obs["batch"] exists
            frameon=False,
            ax=axes[0, col],
            show=False,
            legend_loc="on data",
            legend_fontsize=10,  # smaller font
            title=f"{method}"
        )

        # 2) Bottom row (row=1): color by the Leiden clusters for this method
        leiden_key = f"leiden_{method}"
        sc.pl.embedding(
            adata,
            basis=f"X_umap",
            color=leiden_key,         # Column in adata.obs
            frameon=False,
            ax=axes[1, col],
            show=False,
            legend_loc="on data",
            legend_fontsize=10,  # smaller font
            # title=f"{method}"
        )

    plt.tight_layout()
    # fig.savefig("batch_and_leiden_per_embedding.pdf", dpi=300)
    # plt.close(fig)


Train the latent VAE on OT embeddings and store ``scBIOT``:

.. code-block:: python

    scb.pp.setup_anndata(
        adata,
        var_key="X_ot",
        batch_key="batch",
        pseudo_key="leiden_X_ot",
        true_key=None,
    )

    model = scb.models.vae(adata, verbose=True)
    model.train()
    SCBIOT_LATENT_KEY = "scBIOT"
    adata.obsm[SCBIOT_LATENT_KEY] = model.get_latent_representation(
        n_compoents=30,
        svd_solver="arpack",
        random_state=42,
    )

.. figure:: /_static/plots/paired_Chen-2019_integration_plot01.png
   :alt: UMAPs of OT and scBIOT embeddings for SNARE-seq.
   :width: 100%

Visualise integration through UMAPs coloured by donor, cell type, and clusters:

.. code-block:: python

    sc.pp.neighbors(adata, use_rep="X_ot")
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.8, key_added="X_ot_leiden")

    methods = ["X_ot", "scBIOT"]
    for method in methods:
        sc.pp.neighbors(adata, use_rep=method)
        sc.tl.umap(adata)
        adata.obsm[f"X_umap_{method}"] = adata.obsm["X_umap"].copy()
        sc.tl.leiden(adata, key_added=f"{method}_leiden", resolution=0.8)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(methods), figsize=(4 * len(methods), 8), squeeze=False)
    for col, method in enumerate(methods):
        sc.pl.embedding(
            adata,
            basis=f"X_umap_{method}",
            color="cell_type",
            frameon=False,
            ax=axes[0, col],
            show=False,
            legend_loc="on data",
            legend_fontsize=10,
            title=f"{method}",
        )
        leiden_key = f"{method}_leiden"
        sc.pl.embedding(
            adata,
            basis=f"X_umap_{method}",
            color=leiden_key,
            frameon=False,
            ax=axes[1, col],
            show=False,
            legend_loc="on data",
            legend_fontsize=10,
        )
    plt.tight_layout()


.. figure:: /_static/plots/paired_Chen-2019_integration_plot02.png
   :alt: scib-metrics benchmarking table for SNARE-seq integration.
   :width: 100%
   

Assess integration and biological conservation with ``scib-metrics``:

.. code-block:: python

    bm = Benchmarker(
        adata,
        batch_key="batch",
        label_key="cell_type",
        bio_conservation_metrics=BioConservation(),
        batch_correction_metrics=BatchCorrection(),
        embedding_obsm_keys=["X_pca", "X_ot", "scBIOT"],
        n_jobs=32,
    )
    bm.benchmark()
    bm.plot_results_table(min_max_scale=False)

.. figure:: /_static/plots/paired_Chen-2019_integration_plot03.png
   :alt: Additional diagnostic table from the notebook.
   :width: 150%
