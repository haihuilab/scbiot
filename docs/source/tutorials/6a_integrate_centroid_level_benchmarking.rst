6a. Lung atlas benchmark (balanced OT)
======================================

:download:`Open the notebook <../../../examples/optimal_transport_centroid_level_benchmarking.ipynb>`
to reproduce the scIB lung atlas benchmark end-to-end.  The dataset already
contains raw counts, batch annotations, and cell-type labels, so the tutorial
focuses on preparation, centroid OT, and downstream evaluation.

.. contents::
   :local:
   :depth: 2

Environment and inputs
----------------------

Silence noisy warnings, seed randomness, and fall back to the Figshare mirror
if the local file is missing:

.. code-block:: python

    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd
    import scanpy as sc
    import scbiot as scb
    from scbiot.utils import set_seed
    from pathlib import Path

    from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection

    set_seed(42)

    dir = Path.cwd()
    adata_path = dir / "inputs" / "lung_atlas.h5ad"
    adata = sc.read(
        adata_path,
        backup_url="https://figshare.com/ndownloader/files/24539942",
    )

Preprocessing and PCA backbone
------------------------------

Match the notebook’s preprocessing recipe so that the OT module receives
balanced features:

.. code-block:: python

    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=2000,
        flavor="seurat_v3",
        batch_key="batch",
    )
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    sc.tl.pca(adata, n_comps=50, use_highly_variable=True)

Centroid-level OT
-----------------

Run OT on 512 centroids per batch, then interpolate the displacement field back
to every cell with 8-nearest-centroid weights:

.. code-block:: python

    adata, metrics = scb.ot.integrate_centroids(
        adata,
        obsm_key="X_pca",
        batch_key="batch",
        out_key="X_ot",
        ot_mode="balanced",
        n_centroids_per_batch=512,   # fewer = faster, more = better fidelity
        max_samples_per_batch=500_000,
        k_interp=8,
        chunk_size=500_000,
        modality="rna",
    )
    print(metrics)

Neighbors, Leiden clustering, and metrics
-----------------------------------------

Build the graph/UMAP on the OT embedding and compare pseudo labels against the
truth with a quick normalized mutual information (NMI) score:

.. code-block:: python

    sc.pp.neighbors(adata, use_rep="X_ot")
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.8, key_added="leiden_X_ot")

    from sklearn.metrics import normalized_mutual_info_score

    df = pd.DataFrame(adata.obsm["X_ot"], index=adata.obs.index)
    df["batch"] = adata.obs["batch"]
    df["target"] = adata.obs["cell_type"]
    df["pseudo"] = adata.obs["leiden_X_ot"]

    # Map categorical labels to integers ranked by frequency
    for col in ("target", "pseudo"):
        counts = df[col].value_counts()
        mapping = {name: idx for idx, name in enumerate(counts.index)}
        df[col] = df[col].map(mapping)

    labels = df.reset_index()
    nmi = normalized_mutual_info_score(labels["target"], labels["pseudo"])
    print(f"NMI(target, pseudo) = {nmi:.4f}")

.. figure:: /_static/plots/centroid_lung_benchmark_plot01.png
   :alt: Example OT UMAP highlighting batch mixing after centroid integration.
   :width: 60%

   Example OT UMAP highlighting synthetic batch mixing after centroid-level integration.

Denoising with the scBIOT VAE
-----------------------------

Train the variational autoencoder on top of the OT embedding to obtain a
denoised latent space and compare layouts:

.. code-block:: python

    scb.pp.setup_anndata(
        adata,
        var_key="X_ot",
        batch_key="batch",
        pseudo_key="leiden_X_ot",
    )
    model = scb.models.vae(adata, verbose=True)
    model.train()

    adata.obsm["scBIOT"] = model.get_latent_representation(
        n_components=50,
        svd_solver="arpack",
        random_state=42,
    )

    methods = ["X_ot", "scBIOT"]
    for method in methods:
        sc.pp.neighbors(adata, use_rep=method)
        sc.tl.umap(adata)
        adata.obsm[f"X_umap_{method}"] = adata.obsm["X_umap"].copy()
        sc.tl.leiden(adata, key_added=f"{method}_leiden", resolution=0.8)

scIB evaluation
---------------

Recreate the scorecard from the notebook.  Older ``scib-metrics`` releases may
need ``mask = (labels == label).to_numpy()`` inside
``_graph_connectivity.py`` to avoid pandas warnings.

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

.. figure:: /_static/plots/centroid_lung_benchmark_plot02.png
   :alt: Example benchmarking metrics after centroid integration.
   :width: 100%