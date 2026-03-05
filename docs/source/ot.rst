Optimal transport: ``ot``
=========================

OT utilities for aligning batches and modalities. The functions below match what
you see in the tutorials; refer to the notebooks for full, runnable examples.

- ``integrate``: batch correction for single-modality or cross-modality data (RNA or ATAC).

For a basic scRNA-seq dataset integration:

.. code-block:: python

   adata, metrics = scb.ot.integrate(
       adata,
       preset="rna",
       obsm_key="X_pca",
       batch_key="batch",
       out_key="X_ot"       
   )

For stable tuning, use the meta-parameter interface:

.. code-block:: python

   adata, metrics = scb.ot.integrate(
       adata,
       preset="rna",
       epsilon=0.03,
       tau=0.40,
       knn_scale=1.0,
       batch_strength=1.0,
       gate_temperature=1.0,
       # optional supervision:
       label_key="semi_cell_type",
       unlabeled_category="Unknown",
       sup_strength=0.10,
   )
   

For unpaired RNA/ATAC workflows, compute a shared PCA with ``pp.coembed_pca`` and
then run ``ot.integrate(preset="anchor", obsm_key="X_pca_shared",
batch_key="modality", reference_category="reference")`` to align query cells to
the reference.

For paired RNA/ATAC workflows, use the ``paired`` preset so OT sees each cell's
matched views directly. Call:

.. code-block:: python

   adata, metrics = scb.ot.integrate(
       adata,
       preset="paired",
       obsm_key="X_pca",
       view_key="X_lsi",
       batch_key="batch",
       out_key="X_ot"    
   )

The ``view_keys`` tuple points to the RNA PCA and ATAC LSI embeddings so the
barycentric objective leverages the paired measurements directly.

Scaling options
---------------

For ultra-large datasets, use centroid-level OT:

.. code-block:: python

   adata, metrics = scb.ot.integrate(
       adata,
       preset="centroid",
       obsm_key="X_pca",
       batch_key="batch",
       out_key="scBIOT",
   )

If you want centroid OT while keeping another preset's OT hyperparameters,
enable the flag:

.. code-block:: python

   adata, metrics = scb.ot.integrate(
       adata,
       preset="anchor",
       obsm_key="X_pca",
       batch_key="batch",
       out_key="X_ot",
       centroid_ot=True,
   )

For a faster approximate OT run on large datasets, enable the approximate OT
solver while keeping your preset's data keys:

.. code-block:: python

   adata, metrics = scb.ot.integrate(
       adata,
       preset="atac",
       obsm_key="X_lsi",
       batch_key="batchname_all",
       out_key="X_ot",
       approximate_ot=True,
   )

OT backend controls
-------------------

All OT entry points share the ``use_gpu``/``gpu_device`` and ``ot_backend`` knobs.
