7. Brain-1.3M dataset integration
===============================

:download:`Open the notebook <../../../examples/brain_1.3M_gpu_integration.ipynb>`
to reproduce the RAPIDS-accelerated workflow that preprocesses, integrates, and
denoises 1.3 million mouse brain cells.  The tutorial mirrors that notebook so
you can bring the same GPU-friendly recipe into your own workloads.

.. contents::
   :local:
   :depth: 2

Environment, dependencies, and inputs
-------------------------------------

Import Scanpy/AnnData alongside RAPIDS SingleCell, CuPy, and scBIOT.  The
dataset ships as a 10x Genomics HDF5 file, with a Figshare URL available as a
backup if the local copy is missing:

.. code-block:: python

    import scanpy as sc
    import anndata as ad
    import cupy as cp
    import numpy as np
    import rapids_singlecell as rsc
    import scbiot as scb

    import time
    from pathlib import Path
    import gc
    import warnings

    warnings.filterwarnings("ignore")

    dir = Path.cwd()
    sc.logging.print_header()

    h5_path = dir / "inputs" / "1M_neurons_filtered_gene_bc_matrices_h5.h5"
    adata = sc.read_10x_h5(
        h5_path,
        backup_url="https://s3-us-west-2.amazonaws.com/10x.files/samples/cell/1M_neurons/1M_neurons_filtered_gene_bc_matrices_h5.h5",
    )
    adata.var_names_make_unique()

Centralize the key hyperparameters that drive filtering, HVG selection, and
downstream neighbors/UMAP settings:

.. code-block:: python

    MITO_GENE_PREFIX = "mt-"
    markers = ["Stmn2", "Hes1", "Olig1"]

    min_genes_per_cell = 200
    max_genes_per_cell = 6_000
    min_cells_per_gene = 1
    n_top_genes = 4_000

    n_components = 100
    n_neighbors = 15
    knn_n_pcs = 50

    umap_min_dist = 0.3
    umap_spread = 1.0
    ranking_n_top_genes = 50

Batch parsing and quality control
---------------------------------

Each barcode encodes its plate ID after the last ``-``.  Convert that suffix
into a categorical ``batch`` column before the QC pass:

.. code-block:: python

    suffix = adata.obs_names.str.rsplit("-", n=1).str[1]
    adata.obs["batch"] = suffix.astype("category")

Filter extreme cells/genes, derive mitochondrial fractions, and take a quick
look at the QC distributions:

.. code-block:: python

    sc.pp.filter_cells(adata, min_genes=min_genes_per_cell)
    sc.pp.filter_cells(adata, max_genes=max_genes_per_cell)
    sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)

    mito_genes = adata.var_names.str.startswith(MITO_GENE_PREFIX)
    n_counts = np.array(adata.X.sum(axis=1))
    adata.obs["percent_mito"] = np.array(np.sum(adata[:, mito_genes].X, axis=1)) / n_counts
    adata.obs["n_counts"] = n_counts

    sc.pl.violin(adata, keys=["n_genes", "n_counts", "percent_mito"])

Normalization, HVGs, and PCA backbone
-------------------------------------

Follow the notebook’s preprocessing choices so the OT module receives a
balanced representation:

.. code-block:: python

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor="cell_ranger",
    )

    for marker in markers:
        adata.obs[f"{marker}_raw"] = adata.X[:, adata.var.index == marker].toarray().ravel()

    adata = adata[:, adata.var.highly_variable]
    gc.collect()

    sc.pp.regress_out(adata, ["n_counts", "percent_mito"])
    sc.pp.scale(adata, max_value=10)

    sc.tl.pca(adata, n_comps=n_components)

Optimal transport alignment
---------------------------

Compute the OT embedding directly on the PCA coordinates, persist it to disk,
and prepare a lightweight AnnData whose ``X`` matrix contains the dense OT
representation:

.. code-block:: python

    adata, metrics = scb.ot.integrate(
        adata,
        preset="rna",
        obsm_key="X_pca",
        batch_key="batch",
        out_key="X_ot",
    )
    print(metrics)

    adata.write(dir / "inputs" / "brain_1M.h5ad", compression="lzf")
    adata = sc.read(dir / "inputs" / "brain_1M.h5ad")

    X_ot = np.asarray(adata.obsm["X_ot"], dtype="float32")
    adata_ot = ad.AnnData(X=X_ot, obs=adata.obs.copy())
    adata_ot.var_names = [f"OT{i+1}" for i in range(X_ot.shape[1])]
    adata_ot.obsm["X_ot"] = X_ot

GPU neighbors, UMAP, and pseudo labels
--------------------------------------

Load the OT matrix onto the GPU, configure RMM’s allocator, then re-build the
graph/UMAP entirely with RAPIDS SingleCell:

.. code-block:: python

    import rmm
    from rmm.allocators.cupy import rmm_cupy_allocator

    rmm.reinitialize(managed_memory=False, pool_allocator=False, devices=0)
    cp.cuda.set_allocator(rmm_cupy_allocator)

    rsc.get.anndata_to_GPU(adata_ot)

    rsc.pp.neighbors(adata_ot, use_rep="X_ot")
    rsc.tl.umap(adata_ot)
    rsc.tl.leiden(adata_ot, resolution=0.8, key_added="leiden_X_ot")

    sc.pl.umap(adata_ot, color=["leiden_X_ot"], ncols=1)

.. figure:: /_static/plots/brain_gpu_integration_plot01.png
   :alt: Side-by-side UMAPs for the OT baseline and scBIOT VAE embedding.
   :width: 80%
   

Train the scBIOT Transformer-VAE
--------------------------------

With GPU-derived pseudo labels available, set up the scBIOT VAE to denoise the
embedding and learn a latent space that respects both batch structure and the
Leiden clusters:

.. code-block:: python

    scb.pp.setup_anndata(
        adata_ot,
        var_key="X_ot",
        batch_key="batch",
        pseudo_key="leiden_X_ot",
    )
    model = scb.models.vae(adata_ot, verbose=True, batch_size=1024)
    model.train()

    SCBIOT_LATENT_KEY = "scBIOT"
    adata_ot.obsm[SCBIOT_LATENT_KEY] = model.get_latent_representation(
        n_components=n_components,
        svd_solver="arpack",
        random_state=42,
    )
    adata.obsm[SCBIOT_LATENT_KEY] = model.get_latent_representation(
        n_components=n_components,
        svd_solver="arpack",
        random_state=42,
    )

Compare OT and VAE embeddings
-----------------------------

Loop over the OT baseline and the scBIOT latent space, recompute neighbors on
each, and store the resulting UMAPs for side-by-side visualizations of batch
mixing versus Leiden partitions:

.. code-block:: python

    methods = ["X_ot", "scBIOT"]
    leiden_methods = [f"{method}_leiden" for method in methods]

    for method, leiden_method in zip(methods, leiden_methods):
        rsc.pp.neighbors(adata_ot, use_rep=method)
        rsc.tl.umap(adata_ot)
        adata_ot.obsm[f"X_umap_{method}"] = adata.obsm["X_umap"].copy()
        rsc.tl.leiden(adata_ot, key_added=leiden_method, resolution=0.8)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(methods), figsize=(4 * len(methods), 8), squeeze=False)
    for col, method in enumerate(methods):
        sc.pl.embedding(
            adata_ot,
            basis=f"X_umap_{method}",
            color="batch",
            frameon=False,
            ax=axes[0, col],
            show=False,
            legend_loc="on data",
            legend_fontsize=10,
            title=method,
        )
        leiden_key = f"{method}_leiden"
        sc.pl.embedding(
            adata_ot,
            basis=f"X_umap_{method}",
            color=leiden_key,
            frameon=False,
            ax=axes[1, col],
            show=False,
            legend_loc="on data",
            legend_fontsize=10,
        )

    plt.tight_layout()





