.. |scbiot-logo| image:: _static/scbiot_logo.svg
   :height: 36px
   :alt: scBIOT logo

scBIOT documentation
==================================

**scBIOT** (Single-Cell Biological Insights via Optimal Transport and Omics Transformers) unifies optimal-transport alignment with Transformer encoders for preprocessing and embedding single-cell RNA, ATAC, and multi-omic data. 
The library is designed for embeddings, reproducible benchmarking, and scalable inference across modalities.


Highlights
----------

* Fast optimal transport with GPU.
* A unified `scBIOT` models that can embed RNA, ATAC, or multi-omics modalities.
* Supports scRNA-seq, snATAC-seq, and paired and unpaired multi-omics.
* Supports label transfer across disjoint datasets, such as scRNA-seq to Xenium, scRNA-seq to snATAC-seq.
* Built-in preprocessing steps (iterative LSI, gene activity annotation from peaks, coembedding of PCA from multiomics).
* Support both CPU and GPU.

.. toctree::
   :maxdepth: 2
   :caption: Get started

   installation
   usage

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
   tutorials/7_brain_1.3M_integration
   tutorials/8_label_transfer_with_supbiot

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api
   ot
   models
   preprocessing   
   plotting
   data_source
