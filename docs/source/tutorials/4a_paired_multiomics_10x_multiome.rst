10x Multiome integration
========================

:download:`Open notebook <../../../examples/multiome_integration.ipynb>` for joint RNA+ATAC analysis on matched 10x Multiome data.

Environment and inputs
----------------------

.. code-block:: python

    import warnings
    warnings.filterwarnings("ignore")

    import scanpy as sc
    import pandas as pd
    import scbiot as scb
    from scbiot.utils import set_seed

    from pathlib import Path
    dir = Path.cwd()
    set_seed(42)

    adata = sc.read(
        f"{dir}/inputs/multiome.h5ad",
        backup_url="https://figshare.com/ndownloader/files/59742665",
    )

    gex_vars = adata.var["feature_types"] == "GEX"
    adata_gex = adata[:, gex_vars].copy()
    atac_vars = adata.var["feature_types"] == "ATAC"
    adata_atac = adata[:, atac_vars].copy()

    sc.pp.normalize_per_cell(adata_gex, counts_per_cell_after=1e4)
    sc.pp.log1p(adata_gex)
    sc.pp.highly_variable_genes(adata_gex, n_top_genes=2000, flavor="cell_ranger", batch_key="batch")
    sc.tl.pca(adata_gex, n_comps=30, use_highly_variable=True)
    adata.obsm["X_pca"] = adata_gex.obsm["X_pca"]

Prepare peaks (remove promoter-proximal, select top features, build LSI) and stash views:

.. code-block:: python

    adata_top = scb.pp.remove_promoter_proximal_peaks(
        adata_atac,
        gtf_file=f"{dir}/inputs/gencode.v48.chr_patch_hapl_scaff.basic.annotation.gtf.gz",
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

Integrate paired modalities with barycentric OT (RNA PCA + ATAC LSI), then train and store the latent:

.. code-block:: python

    adata, metrics = scb.ot.integrate(
        adata,
        modality="paired",
        obsm_key="X_pca",          # geometry / smoothing base view
        batch_key="batch",
        out_key="X_ot",
        mode="ufgw_barycenter",
        view_keys=("X_pca", "X_lsi"),
    )
    print(metrics)

Train the latent model on top of OT embeddings:

.. code-block:: python

    scb.pp.setup_anndata(
        adata,
        var_key="X_ot",
        batch_key="batch",
        pseudo_key="leiden_X_ot",
        true_key=None,
    )

    model = scb.models.vae(adata, prior_pcr=5, verbose=True)
    model.train()
    SCBIOT_LATENT_KEY = "scBIOT"
    adata.obsm[SCBIOT_LATENT_KEY] = model.get_latent_representation(
        n_compoents=30,
        svd_solver="arpack",
        random_state=42,
    )

.. figure:: /_static/plots/multiome_integration_plot01.png
   :alt: Initial diagnostic plot from the notebook.
   :width: 50%

Create UMAPs for OT and scBIOT embeddings to visualise batch mixing and clusters:

.. code-block:: python

    sc.pp.neighbors(adata, use_rep="X_ot", n_neighbors=8)
    sc.tl.umap(adata, min_dist=0.10, spread=1.0)
    sc.tl.leiden(adata, resolution=0.8, key_added="leiden_X_ot")

    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    import pandas as pd

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

    df = df.reset_index().set_index(["index", "batch", "target", "pseudo"])
    true_labels = df.reset_index()["target"]
    pred_labels = df.reset_index()["pseudo"]

    nmi_score = normalized_mutual_info_score(true_labels, pred_labels)
    ari_score = adjusted_rand_score(true_labels, pred_labels)
    print(f"NMI: {nmi_score:.4f} | ARI: {ari_score:.4f}")

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

.. figure:: /_static/plots/multiome_integration_plot02.png
   :alt: UMAPs of OT and scBIOT embeddings for 10x Multiome.
   :width: 50%

Benchmark embeddings with ``scib-metrics`` to balance batch correction and biology:

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

.. figure:: /_static/plots/multiome_integration_plot03.png
   :alt: scib-metrics benchmarking table for 10x Multiome integration.
   :width: 150%
