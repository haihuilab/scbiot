6b. Tahoe-100M centroid pipeline
================================

:download:`Open the notebook <../../../examples/optimal_transport_centroid_level_Tahoe100M.ipynb>`
for the complete out-of-core workflow.  The steps below mirror that notebook so
you can run the same procedure on local hardware or an HPC cluster.

.. contents::
   :local:
   :depth: 2

Environment and chunked AnnData setup
-------------------------------------

Prepare the dependencies, detect optional RAPIDS acceleration, and configure
how AnnData should stream sparse chunks:

.. code-block:: python

    from pathlib import Path

    import anndata as ad
    import dask
    import dask.array as da
    import h5py
    import numpy as np
    import scanpy as sc
    import scbiot as scb

    from dask.distributed import Client, LocalCluster

    use_gpu = False  # set True when RAPIDS is installed

    if use_gpu:
        import rapids_singlecell as rsc
        from cupyx.scipy import spx
        import rmm
        import cupy as cp
        from rmm.allocators.cupy import rmm_cupy_allocator

        SPARSE_CHUNK_SIZE = 1_000_000

        def set_mem():
            rmm.reinitialize(managed_memory=True)
            cp.cuda.set_allocator(rmm_cupy_allocator)

        set_mem()
        dask.array.register_chunk_type(spx.csr_matrix)
        mod = rsc
    else:
        SPARSE_CHUNK_SIZE = 100_000
        mod = sc

Load the lazily concatenated plates, leaving ``adata.X`` as a chunked array,
and attach the precomputed PCA coordinates:

.. code-block:: python

    with h5py.File(f"/home/figo/software/python_libs/plate_merged_{'gpu' if use_gpu else 'cpu'}.h5ad", "r") as f:
        adata = ad.AnnData(
            obs=ad.io.read_elem(f["obs"]),
            var=ad.io.read_elem(f["var"]),
        )
        adata.X = ad.experimental.read_elem_lazy(
            f["X"],
            chunks=(SPARSE_CHUNK_SIZE, adata.shape[1]),
        )

    adata.obsm["X_pca"] = da.from_zarr(
        f"/home/figo/software/python_libs/plate_merged_pca_{'gpu' if use_gpu else 'cpu'}.zarr"
    )

Centroid OT across plates
-------------------------

With ``adata.obsm["X_pca"]`` available, fit OT on 2k centroids per plate using
the unbalanced mode that handles skewed batch sizes:

.. code-block:: python

    adata, metrics = scb.ot.integrate_centroids(
        adata,
        obsm_key="X_pca",
        batch_key="plate",
        out_key="X_ot",
        reference="union",
        ot_mode="unbalanced",
        n_centroids_per_batch=2048,
        max_samples_per_batch=500_000,
        chunk_size=1_000_000,
        k_interp=8,
        K_pseudo=64,
        modality="rna",
        use_gpu=True,
        verbose=True,
    )
    print(metrics)

Persist the OT embedding to disk chunk-by-chunk so downstream jobs can stream
it without loading the full dense matrix into memory:

.. code-block:: python

    adata.obsm["X_ot"] = da.from_array(
        adata.obsm["X_ot"],
        chunks=(SPARSE_CHUNK_SIZE, -1),
    )
    adata.obsm["X_ot"].to_zarr(
        f"/home/figo/software/python_libs/plate_merged_ot_{'gpu' if use_gpu else 'cpu'}.zarr"
    )

Million-cell visualization subset
---------------------------------

Sample one million cells for exploratory plots.  The notebook runs Scanpy on
CPU or RAPIDS SingleCell on GPU (selected via ``mod`` above):

.. code-block:: python

    adata = adata[np.random.randint(0, adata.shape[0], (1_000_000,))]

    mod.pp.neighbors(adata, use_rep="X_ot")
    mod.tl.umap(adata)
    if use_gpu:
        adata.obsm["X_umap"] = adata.obsm["X_umap"].get()

    sc.pl.umap(
        adata,
        color=["cell_line", "plate"],
        ncols=1,
    )

    mod.tl.leiden(
        adata,
        resolution=0.8,
        key_added="leiden",
    )

.. figure:: /_static/plots/tahoe_centroid_plot01.png
   :alt: Synthetic million-cell subset showing plate and cell-line overlays.
   :width: 80%

   Synthetic million-cell subset showing plate (left) and cell-line (right) overlays for Tahoe-style centroid OT.

This sampled visualization mirrors the figure in the notebook while the
full-resolution OT coordinates remain on disk for large-scale downstream jobs.
