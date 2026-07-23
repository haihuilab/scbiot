Preprocessing: ``pp``
=====================

Utilities for creating scBIOT input representations from AnnData objects. In
v1.2.0, ``autoencoder`` and ``autoencoder_map`` provide the supported linear
embedding and cross-modality mapping workflows.

.. currentmodule:: scbiot.pp

Embeddings
----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   autoencoder
   autoencoder_map

``autoencoder`` creates a PCA-like tied-linear-autoencoder embedding for one
AnnData object. ``autoencoder_map`` coembeds separate reference and query objects
that share gene names.

ATAC and gene activity
----------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   remove_promoter_proximal_peaks
   find_variable_features
   add_iterative_lsi
   create_gene_activity
   annotate_gene_activity
   harmonize_gene_names
   knn_smooth_ga_on_atac
