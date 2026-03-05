Usage
=====


Quick start
-----------

The :class:`scbiot` class wraps the full preprocessing + embedding
pipeline. Pass anndata object.

.. code-block:: python

    import numpy as np
    import pandas as pd
    import scbiot import scb
    import scanpy as sc

    # Use optimal transport for prealignment
    adata = sc.read_h5ad('alldata.h5ad')
    )
    adata, metrics = scb.ot.integrate(
            adata,
            preset="rna",
            obsm_key="X_pca",
            batch_key="batch",
            out_key="X_ot",
        )
        print(metrics)

    # Use OT embeddings for downstream analysis
    sc.pp.neighbors(adata, use_rep='X_ot')
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.8, key_added='leiden_X_ot')
    adata
