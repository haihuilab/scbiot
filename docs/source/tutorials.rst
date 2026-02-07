
The notebooks showcase end-to-end workflows for preprocessing, model training,
and evaluation across scRNA-seq, scATAC-seq, and multi-omics data. They mirror
the Scanpy/Read the Docs layout with short landing pages, clear menus, and
runnable code snippets. All notebooks live in ``examples/`` and can be opened
locally or in any Jupyter environment.

.. note::

   Each tutorial page now renders the corresponding ``examples/*.ipynb``
   directly via MyST-NB so the docs always show the latest outputs when you run
   ``make -C docs html`` (the build installs ``scbiot`` and executes every
   notebook, failing on any execution errors).

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
   tutorials/5_unpaired_multiomics
   tutorials/6_integrate_centroid_level
   tutorials/7_brain_1.3M_integration
   tutorials/8_label_transfer_with_supbiot
