Lung atlas benchmarking (unsupervised)
======================================

:download:`Open notebook <../../../examples/unsupervised_lung_benchmarking.ipynb>` for an end-to-end workflow with unsupervised integration, evaluation metrics, and downstream analysis.

Environment and data
--------------------

.. code-block:: python

    import warnings
    warnings.filterwarnings("ignore")

    import scanpy as sc
    import scbiot as scb
    from scbiot.utils import set_seed

    from pathlib import Path
    dir = Path.cwd()
    set_seed(42)

    adata_path = f"{dir}/inputs/lung_atlas.h5ad"
    adata = sc.read(
        adata_path,
        backup_url="https://figshare.com/ndownloader/files/24539942",
    )

Preprocess with PCA
-------------------

.. code-block:: python

    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3", batch_key="batch")
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    sc.tl.pca(adata, n_comps=50, use_highly_variable=True)

Run optimal-transport integration on the PCA space and inspect alignment metrics:

.. code-block:: python

    adata, metrics = scb.ot.integrate(
        adata,
        modality="rna",
        obsm_key="X_pca",
        batch_key="batch",
        out_key="X_ot",
    )
    print(metrics)

Prepare AnnData for model training and store the latent representation:

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
        n_compoents=50,
        svd_solver="arpack",
        random_state=42,
    )

Build UMAPs and Leiden clusters to visualise batch mixing and local structure:

.. code-block:: python

    sc.pp.neighbors(adata, use_rep="X_ot")
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.8, key_added="leiden_X_ot")

Quantify clustering quality before training the VAE:

.. code-block:: python

    from sklearn.metrics import normalized_mutual_info_score
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
    print(f"NMI: {nmi_score:.4f}")

    methods = ["X_ot", "scBIOT"]
    for method in methods:
        sc.pp.neighbors(adata, use_rep=method)
        sc.tl.umap(adata)
        adata.obsm[f"X_umap_{method}"] = adata.obsm["X_umap"].copy()
        sc.tl.leiden(adata, key_added=f"{method}_leiden", resolution=0.8)


Plot UMAPs coloured by batch and Leiden clusters (mirrors the notebook figures):

.. code-block:: python

    import matplotlib.pyplot as plt
    import scanpy as sc

    fig, axes = plt.subplots(2, len(methods), figsize=(4 * len(methods), 8), squeeze=False)
    for col, method in enumerate(methods):
        sc.pl.embedding(
            adata,
            basis=f"X_umap_{method}",
            color="batch",
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

.. figure:: /_static/plots/unsupervised_lung_benchmarking_plot01.png
   :alt: UMAPs comparing X_ot and scBIOT embeddings and Leiden clusters.
   :width: 50%


Score integration quality with ``scib-metrics`` (batch removal + biology conservation):
# fix the bug in the scib-metrics
# change _graph_connectivity.py: <mask = labels == label> to <mask = (labels == label).to_numpy()> 

.. code-block:: python

    bm = Benchmarker(
        adata,
        batch_key="batch",
        label_key="cell_type",
        bio_conservation_metrics=BioConservation(),
        batch_correction_metrics=BatchCorrection(),
        embedding_obsm_keys=["X_pca", "X_ot", "scBIOT"],
        n_jobs=-1,
    )
    bm.benchmark()
    bm.plot_results_table(min_max_scale=False)

.. figure:: /_static/plots/unsupervised_lung_benchmarking_plot02.png
   :alt: scib-metrics table benchmarking embeddings.
   :width: 150%

