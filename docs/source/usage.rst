Usage
=====


Quick start
-----------

The :class:`scbiot` class wraps the full preprocessing + embedding
pipeline. Pass any matrix-like object where rows denote cells and columns denote
genes (``pandas.DataFrame`` works best).

.. code-block:: python

    import numpy as np
import pandas as pd
import scbiot import scb
import scanpy as sc

# toy count matrix: cells x genes
adata = sc.read_h5ad('alldata.h5ad')
)
adata, metrics = scb.ot.integrate(
        adata,
        modality="rna",
        obsm_key="X_pca",
        batch_key="batch",
        out_key="X_ot",
    )
    print(metrics)

Prepare AnnData for model training and store the latent representation:

.. code-block:: python

    scb.pp.setup_anndata(
        adata,
        var_key="X_ot",
        batch_key="batch",
        pseudo_key="leiden_X_ot",
        true_key=None,
    )

    model = scb.models.vae(adata, verbose=True)
    model.train()
    SCBIOT_LATENT_KEY = "scBIOT"
    adata.obsm[SCBIOT_LATENT_KEY] = model.get_latent_representation(
        n_compoents=50,
        svd_solver="arpack",
        random_state=42,
    )

