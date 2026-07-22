scBIOT
---------------

These notebooks demonstrate the scBIOT 1.2.0 workflows for preprocessing,
optimal-transport integration, label transfer, and scalable analysis. Download
links on each notebook page provide the same ``.ipynb`` files tracked in the
repository's ``tutorials/`` directory.

Highlights
---------------

   - **Batteries-included preprocessing**: scATAC-seq peak processing, iterative LSI, and gene activity annotation.
   - **Accurate atlas integration**: high-fidelity alignment with rare cell-type protection.
   - **Unified scBIOT framework**: one interface for RNA, ATAC, spatial, temporal, and multi-omics integration.
   - **Fast integration via Optimal Transport (OT)**: scalable alignment for large single-cell datasets.
   - **Linear autoencoders**: PCA-like embeddings and reference-to-query coembedding.
   - **Scales to 100M cells locally**: memory-efficient scalable processing.
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
   tutorials/9_spatiotemporal_dynamics
