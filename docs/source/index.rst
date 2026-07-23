.. |scbiot-logo| image:: _static/scbiot_logo.svg
   :height: 36px
   :alt: scBIOT logo

scBIOT documentation
==================================

**scBIOT** (Single-Cell Biological Insights via Optimal Transport) integrates
single-cell RNA, ATAC, spatial, temporal, and multi-omic data. The 1.2.0 API
combines optimal-transport alignment, linear-autoencoder embeddings, label
transfer, and transport-aware downstream analysis in an AnnData workflow.


Highlights
----------

* Fast optimal transport with GPU.
* A unified `scBIOT` framework that can embed RNA, ATAC, or multi-omics modalities.
* Supports scRNA-seq, snATAC-seq, and paired and unpaired multi-omics.
* Spatial/time-aware integration and lineage-specific spatiotemporal velocity fields.
* Supports label transfer across disjoint datasets, such as scRNA-seq to Xenium, scRNA-seq to snATAC-seq.
* Built-in preprocessing steps (iterative LSI, gene activity annotation from peaks, and linear-autoencoder coembedding).
* Transport-aware gene, trajectory, and visualization tools.
* CPU and GPU execution.

.. toctree::
   :maxdepth: 2
   :caption: Get started

   installation
   usage
   multiomics
   spatiotemporal

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials
   tutorials/1_scrna_seq
   tutorials/2_scrna_seq_r
   tutorials/3_scatac_seq
   tutorials/4_paired_multiomics
   tutorials/5_unpaired_multiomics
   tutorials/6_integrate_centroid_level
   tutorials/7_brain_1_3M_integration
   tutorials/8_label_transfer_with_supbiot
   tutorials/9_spatiotemporal_dynamics

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api
   ot
   preprocessing
   tools
   plotting
   data_source
