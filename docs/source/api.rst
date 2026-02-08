API
===

Core public entry points for OT integration and preparing AnnData objects for the
Transformer VAE.

OT integration and label transfer
--------------

.. currentmodule:: scbiot.ot

.. autosummary::
   :toctree: generated
   :nosignatures:

   integrate  
   supbiot
   


AnnData utilities
-----------------

.. currentmodule:: scbiot.pp

.. autosummary::
   :toctree: generated
   :nosignatures:
   

   coembed_pca 
   annotate_gene_activity
   remove_promoter_proximal_peaks   
   add_iterative_lsi


Model utilities
-----------------

.. currentmodule:: scbiot.models

.. autosummary::
   :toctree: generated
   :nosignatures:

   setup_anndata
   vae
   get_latent_representation


Plotting utilities
-----------------

.. currentmodule:: scbiot.pl

.. autosummary::
   :toctree: generated
   :nosignatures:

   plot_anndata_confusion
   celltype_gene_mean_correlation
   celltype_predtype_mean_corr_heatmap

   