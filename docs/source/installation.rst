Installation
============

scBIOT follows the lightweight installation pattern: install the
stable PyPI release or use an editable checkout for development and notebooks.

Install from PyPI
-----------------

The published wheels include all core dependencies:

.. code-block:: bash

    pip install scbiot


Editable install
----------------

Clone the repository if you want to run the example notebooks or contribute:

.. code-block:: bash

    git clone https://github.com/haihuilab/scbiot.git
    cd scbiot
    pip install -e .

Test your setup
---------------

Confirm that the package imports and report the installed version:

.. code-block:: python

    import scbiot
    print(scbiot.__version__)

If you see an ``ImportError`` after installation, upgrade ``pip``/``setuptools``
and retry the command. GPU users should install a CUDA-matching build of
``torch``, ``faiss-gpu`` or ``jax[cuda12]`` *before* installing scBIOT to ensure
the correct binaries are picked up.
