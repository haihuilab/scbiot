Usage
=====

scBIOT 1.2.0 uses a small set of semantic controls instead of dataset presets.
Create the v1.2 linear-autoencoder representation, identify the batch column,
and write the corrected coordinates to a new key.

Basic integration
-----------------

.. code-block:: python

   import scanpy as sc
   import scbiot as scb

   adata = sc.read_h5ad("alldata.h5ad")
   adata = scb.pp.autoencoder(
       adata,
       input_key="counts",
       out_key="X_ae",
       batch_key="batch",
       random_state=0,
   )
   adata, metrics = scb.ot.integrate(
       adata,
       obsm_key="X_ae",
       batch_key="batch",
       out_key="X_ot",
       random_state=0,
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

   query = adata.obs["batch"].astype(str).eq("query")
   adata.obs["semi_cell_type"] = adata.obs["cell_type"].astype(str)
   adata.obs.loc[query, "semi_cell_type"] = "Unknown"

   adata = scb.pp.autoencoder(
       adata,
       input_key="counts",
       out_key="X_ae",
       batch_key="batch",
       label_key="semi_cell_type",
       unlabeled_category="Unknown",
       random_state=0,
   )
   adata, metrics = scb.ot.integrate(
       adata,
       obsm_key="X_ae",
       batch_key="batch",
       out_key="X_supbiot",
       label_key="semi_cell_type",
       unlabeled_category="Unknown",
       random_state=0,
   )
   adata = scb.ot.supbiot(
       adata,
       use_rep="X_supbiot",
       input_rep_key="X_ae",
       label_key="semi_cell_type",
       unlabeled_category="Unknown",
       pred_label_key="pred_cell_type",
       pred_conf_key="pred_confidence",
       random_state=0,
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
       random_state=0,
   )
   adata, metrics = scb.ot.integrate(
       adata,
       obsm_key="X_ae",
       batch_key="modality",
       out_key="X_supbiot",
       label_key="cell_type",
       unlabeled_category="Unknown",
       align_reference=True,
       random_state=0,
   )

Continue with :doc:`tutorials/1_scrna_seq` for RNA workflows,
:doc:`tutorials/4_paired_multiomics` and
:doc:`tutorials/5_unpaired_multiomics` for multi-omics, or
:doc:`tutorials/9_spatiotemporal_dynamics` for spatial and temporal dynamics.
