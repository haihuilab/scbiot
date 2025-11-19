Tutorials
=========

The tutorials mirror the Scanpy/Read the Docs layout: short landing page, a
clear menu of notebooks, and runnable code snippets. All notebooks live in
``examples/`` and can be opened locally or in any Jupyter environment.

.. toctree::
   :maxdepth: 1
   :caption: Workflows

   tutorials/scrna_seq
   tutorials/scatac_seq
   tutorials/paired_multiomics
   tutorials/paired_multiomics_10x_multiome
   tutorials/paired_multiomics_snare_seq
   tutorials/unpaired_multiomics
   tutorials/scrna_lung_atlas_benchmarking
   tutorials/scrna_supervised_supbiot

Quick start
-----------

Install the optional extras and launch a notebook from the repository root:

.. code-block:: bash

    pip install "scbiot[analysis]"
    jupyter lab examples/unsupervised_lung_benchmarking.ipynb

Each notebook walks through preprocessing, model training, and evaluation with
annotated cells, mirroring the style of the ``scvi-tools`` harmonization guide.

scRNA-seq
---------

1a. :download:`Lung atlas benchmarking (unsupervised) <../../examples/unsupervised_lung_benchmarking.ipynb>`
    shows an end-to-end workflow with unsupervised integration and downstream
    metrics.

1b. :download:`Supervised scBIOT (SupBIOT) <../../examples/supervised_lung_benchmarking.ipynb>`
    demonstrates label transfer and supervised contrastive training.

scATAC-seq
----------

2. :download:`Brain integration with large windows <../../examples/atac_brain_large_window_integration.ipynb>`
   covers iterative LSI, peak selection, and integration across replicates.

Paired Multiomics
-----------------

3a. :download:`10x Multiome integration <../../examples/multiome_integration.ipynb>`
    runs joint RNA+ATAC analysis for matched cells.

3b. :download:`SNARE-seq integration (Chen 2019) <../../examples/paired_Chen-2019_integration.ipynb>`
    aligns paired transcriptome and chromatin profiles.

Unpaired Multiomics
-------------------

4. :download:`Yao 2021 unpaired integration <../../examples/unpaired_Yao-2021_integration.ipynb>`
   matches modalities with optimal transport when cells are unpaired.

Inline code
-----------

All tutorials build on the main :class:`scbiot` API. For example, the
scRNA-seq notebooks start with:

.. code-block:: python

    import scbiot as scb

    # counts: cells x genes matrix or DataFrame
    adata, metrics = scb.ot.integrate(adata, modality="rna", obsm_key="X_pca", batch_key="batch", out_key="X_ot")
    sc.pp.neighbors(adata, use_rep="X_ot")
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.8, key_added="leiden_X_ot")
    scb.pp.setup_anndata(adata, var_key="X_ot", batch_key="batch", pseudo_key="leiden_X_ot", true_key=None)
    model = scb.models.vae(adata, verbose=True)
    model.train()
    SCBIOT_LATENT_KEY = "scBIOT"
    adata.obsm[SCBIOT_LATENT_KEY] = model.get_latent_representation(n_components=50, svd_solver="arpack", random_state=42)

With minimal changes you can switch to multi-omic inputs by passing a mapping of
modalities, exactly as shown in the paired and unpaired tutorials.
