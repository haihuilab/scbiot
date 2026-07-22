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
       n_components=50,
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
       transfer_mode="logreg",
   )

Paired multiome data
--------------------

When RNA PCA and ATAC LSI rows describe the same cells in the same order, use the
paired-aware integrator. Its default Procrustes step makes the two component
spaces comparable before OT.

.. code-block:: python

   adata, metrics = scb.ot.integrate_paired(
       adata,
       obsm_key="X_pca",
       view_key="X_lsi",
       batch_key="batch",
       out_key="X_multiome",
       prealign="procrustes",
   )

For larger paired datasets, set ``approximate_ot=True`` or
``centroid_ot=True`` on ``integrate_paired``. See :doc:`tutorials` for complete
RNA+ATAC notebooks.
