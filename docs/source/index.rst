scBIOT documentation
====================

**scBIOT** (Single-Cell Biological Insights via Optimal Transport and Omics Transformers) unifies optimal-transport alignment with Transformer encoders for preprocessing and embedding single-cell RNA, ATAC, and multi-omic data. The library is designed for reproducible benchmarking and scalable inference across modalities.

Highlights
----------

* Fast optimal transport with GPU.
* A unified `scBIOT` models that can embed RNA, ATAC, or paired multi-omics  modalities and reuse the fitted pipeline for inference on new batches.
* Supports scRNA-seq, scATAC-seq, paired and unpaired multi-omics.
* Built-in preprocessing steps (iterative LSI).
* Support both CPU and GPU.

.. toctree::
   :maxdepth: 1
   :caption: Get started

   installation
   usage

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials

.. toctree::
   :maxdepth: 1
   :caption: API reference

   api
   preprocessing
   ot
