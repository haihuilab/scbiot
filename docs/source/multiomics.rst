Multi-omics integration
=======================

scBIOT 1.2.0 supports paired measurements from the same cells and unpaired
reference/query datasets. Choose the workflow based on whether observations are
matched across modalities.

Unpaired reference-to-query mapping
-----------------------------------

For an RNA reference and an ATAC gene-activity query, first create matching gene
features. ``autoencoder_map`` learns a shared tied-linear-autoencoder embedding
and returns a joint AnnData object.

.. code-block:: python

   import scbiot as scb

   adata_ga = scb.pp.create_gene_activity(
       adata_atac,
       adata_rna,
       gtf_file="genes.gtf.gz",
   )
   adata = scb.pp.autoencoder_map(
       adata_rna,
       adata_ga,
       label="modality",
       keys=("reference", "query"),
       reference_layer="counts",
       query_layer="ga_smooth",
       label_key="cell_type",
       unlabeled_category="Unknown",
       out_key="X_ae",
       random_state=0,
   )

Run reference-aligned OT on that representation, then transfer labels. The same
``out_key`` is passed explicitly with ``use_rep`` so the data flow is clear.

.. code-block:: python

   adata, metrics = scb.ot.integrate(
       adata,
       obsm_key="X_ae",
       batch_key="modality",
       out_key="X_supbiot",
       align_reference=True,
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
       input_rep_key="X_ae",
       transfer_mode="logreg",
       random_state=0,
   )

Paired multiome data
--------------------

For paired RNA and ATAC measurements, create one RNA view and one gene-activity
view for each cell. ``autoencoder_map`` places both views in the same feature
space, and the standard integrator aligns them with modality as the batch key.

.. code-block:: python

   adata = scb.pp.autoencoder_map(
       adata_rna,
       adata_gene_activity,
       label="modality",
       keys=("RNA", "ATAC"),
       reference_layer="counts",
       query_layer="ga_smooth",
       out_key="X_ae",
       random_state=0,
   )
   adata, metrics = scb.ot.integrate(
       adata,
       obsm_key="X_ae",
       batch_key="modality",
       out_key="X_multiome",
       random_state=0,
   )

Because rows are duplicated into RNA and ATAC views, collapse the aligned
coordinates by the original cell identifier when a single same-cell
representation is needed. Run the complete
:doc:`tutorials/4_paired_multiomics` notebook for data download, gene activity,
autoencoder mapping, OT integration, and optional supBIOT label transfer.
