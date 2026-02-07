.. |scbiot-logo| image:: _static/scbiot_logo.svg
   :height: 36px
   :alt: scBIOT logo

scBIOT documentation
==================================

**scBIOT** (Single-Cell Biological Insights via Optimal Transport and Omics Transformers) unifies optimal-transport alignment with Transformer encoders for preprocessing and embedding single-cell RNA, ATAC, and multi-omic data. The library is designed for reproducible benchmarking and scalable inference across modalities.


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

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api
   preprocessing
   ot
