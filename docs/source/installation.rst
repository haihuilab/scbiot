Installation
============

scBIOT follows the lightweight installation pattern: install the
stable PyPI release or use an editable checkout for development and notebooks.

Install from PyPI
-----------------

The published wheels include all core dependencies:

.. code-block:: bash

    pip install scbiot

The core installation provides the autoencoder and optimal-transport pipeline.
Benchmarking frameworks, Jupyter, and FAISS are optional.

Optional installations
----------------------

Install only the features needed for a workflow:

.. code-block:: bash

    # Jupyter kernel for the tutorial notebooks
    pip install "scbiot[notebooks]"

    # FAISS acceleration: choose one backend
    pip install "scbiot[cpu]"
    pip install "scbiot[gpu]"       # Linux with CUDA 12

    # scIB benchmarking utilities
    pip install "scbiot[analysis]"

    # Analysis, GPU FAISS, and notebooks together
    pip install "scbiot[full]"

    # Documentation toolchain
    pip install "scbiot[docs]"

Do not install the ``cpu`` and ``gpu`` FAISS extras together because both
provide the ``faiss`` Python module. Without a FAISS extra, scBIOT uses its
scikit-learn nearest-neighbor fallback where available. Centroid interpolation
requires either the CPU or GPU FAISS extra.


Editable install
----------------

Clone the repository if you want to run the tutorial notebooks or contribute:

.. code-block:: bash

    git clone https://github.com/haihuilab/scbiot.git
    cd scbiot
    pip install -e .

The notebooks are stored in the repository's ``tutorials/`` directory. Install
editable extras with the same syntax, for example:

.. code-block:: bash

    pip install -e ".[notebooks]"
    pip install -e ".[gpu,notebooks]"

Test your setup
---------------

Confirm that the package imports and report the installed version:

.. code-block:: python

    import scbiot
    print(scbiot.__version__)

If you see an ``ImportError`` after installation, upgrade ``pip``/``setuptools``
and retry the command. GPU users should confirm that the NVIDIA driver is
compatible with the CUDA runtime selected by PyTorch and
``faiss-gpu-cu12``.
