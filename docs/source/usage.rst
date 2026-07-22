Usage
=====

scBIOT 1.2.0 uses a small set of semantic controls instead of dataset presets.
Choose an existing representation in ``adata.obsm``, identify the batch column,
and write the corrected coordinates to a new key.

Basic integration
-----------------

.. code-block:: python

   import scanpy as sc
   import scbiot as scb

   adata = sc.read_h5ad("alldata.h5ad")
   adata, metrics = scb.ot.integrate(
       adata,
       obsm_key="X_pca",
       batch_key="batch",
       out_key="X_ot",
   )
   print(metrics)

   sc.pp.neighbors(adata, use_rep="X_ot")
   sc.tl.umap(adata)
   sc.tl.leiden(adata, resolution=0.8, key_added="leiden_X_ot")

The returned ``metrics`` dictionary reports integration and geometry diagnostics.
Use ``strength``, ``conservation``, ``prototypes``, and ``supervision``—each in
the range 0 to 1—to tune the corresponding behavior.

Label transfer
--------------

Provide a label column whose query cells use a sentinel such as ``"Unknown"``.
After supervised integration, ``supbiot`` writes predictions and confidence to
``adata.obs``.

.. code-block:: python

   adata, metrics = scb.ot.integrate(
       adata,
       obsm_key="X_pca",
       batch_key="batch",
       out_key="X_supbiot",
       label_key="cell_type",
       unlabeled_category="Unknown",
   )
   adata = scb.ot.supbiot(
       adata,
       use_rep="X_supbiot",
       label_key="cell_type",
       unlabeled_category="Unknown",
       pred_label_key="pred_cell_type",
       pred_conf_key="pred_confidence",
   )

Cross-modality coembedding
--------------------------

For disjoint reference and query datasets with shared gene names, use the v1.2.0
linear autoencoder mapper before OT integration.

.. code-block:: python

   adata = scb.pp.autoencoder_map(
       adata_reference,
       adata_query,
       reference_layer="counts",
       query_layer="ga_smooth",
       label_key="cell_type",
       unlabeled_category="Unknown",
       out_key="X_ae",
   )
   adata, metrics = scb.ot.integrate(
       adata,
       obsm_key="X_ae",
       batch_key="modality",
       out_key="X_supbiot",
       label_key="cell_type",
       unlabeled_category="Unknown",
       align_reference=True,
   )

See :doc:`tutorials` for complete notebook workflows.
