scBIOT
---------------

**scBIOT** is a lightweight Python library for single-cell omics integration. 
It bundles the preprocessing, embedding, transfer label workflows we routinely apply to RNA, ATAC, 
and paired or unpaired multi-omics datasets. The library emphasizes reproducible data preparation, 
single-cell clustering using embeddings derived from optimal transport, 
and concise APIs that work out of the box on AnnData data.

Highlights
---------------

   - **Batteries-included preprocessing**: scATAC-seq peak processing, iterative LSI, and gene activity annotation.
   - **Accurate atlas integration**: high-fidelity alignment with rare cell-type protection.
   - **Unified scBIOT framework**: a single framework for embedding RNA, ATAC, transfer learning, and paired or unpaired multi-omics.
   - **Fast integration via Optimal Transport (OT)**: scalable alignment for large single-cell datasets.
   - **Scales to 100M cells locally**: memory-efficent scalable processing.
   - **Label transfer**: across multi-omics modalities and between spatial data and scRNA-seq references.


Getting started
---------------

Install the package (with optional extras) and launch Jupyter from the
repository root:

.. code-block:: bash

    pip install scbiot    

.. toctree::
   :maxdepth: 1
   :caption: Workflows

   tutorials/1_scrna_seq
   tutorials/2_scrna_seq_r
   tutorials/3_scatac_seq
   tutorials/4_paired_multiomics
   tutorials/5_unpaired_multiomics
   tutorials/6_integrate_centroid_level
   tutorials/7_brain_1_3M_integration
   tutorials/8_label_transfer_with_supbiot   
