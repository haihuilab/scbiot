# scBIOT

**scBIOT** is a lightweight Python library for single-cell omics integration. 
It bundles the preprocessing and embedding workflows we routinely apply to RNA, ATAC, 
and paired or unpaired multi-omics datasets. The library emphasizes reproducible data preparation, 
single-cell clustering using embeddings derived from optimal transport and Transformer-based VAEs, 
and concise APIs that work out of the box on AnnData data.

## Highlights

- Batteries-included preprocessing: preprocessing of ATAC peaks, iterative LSI, and annotation of gene activity.
- A unified `ScBIOT` class that can embed RNA, ATAC, paired or unpaired multi-omics
  modalities and reuse the fitted pipeline for inference on new batches.
- Fast single-cell integration with optimal transport.  
- further enhanced single-cell integration with Transformer based VAE.

Documentation is published on [scbiot.readthedocs.io](https://scbiot.readthedocs.io/en/stable/)
and mirrors the examples below.

## Installation

```bash
pip install scbiot
```

The package targets Python 3.9+ and only depends on NumPy, pandas, and
scikit-learn. For documentation builds install `pip install scbiot[docs]`.

### Optional extras

Depending on your workflow you can pull in heavier scientific stacks as extras:


- `pip install scbiot` installs the CUDA-enabled FAISS + PyTorch combo (CUDA 12) `faiss-gpu-cu12 scib_metrics==0.5.1 leidenalg jaxlib scikit-misc "jax[cuda12]" pyranges`.


For an exact replica of our Conda dev environment use `pip install -r requirements.txt`
inside a fresh virtual environment.

## Quick start

```python
import numpy as np
import pandas as pd
import scbiot as scb
import scanpy as sc


adata = sc.datasets.pbmc3k()

sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3", batch_key='batch')
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.scale(adata)
sc.tl.pca(adata, n_comps=50, use_highly_variable=True)

adata, metrics = scb.ot.integrate(adata, modality='rna', obsm_key='X_pca', batch_key='batch', out_key='X_ot')
print(metrics)

sc.pp.neighbors(adata, use_rep='X_ot')
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.8, key_added='leiden_X_ot')

scb.pp.setup_anndata(adata, var_key='X_ot', batch_key='batch', pseudo_key='leiden_X_ot', true_key=None)
model = scb.models.vae(adata, verbose=True)
model.train()

SCBIOT_LATENT_KEY = "scBIOT"
adata.obsm[SCBIOT_LATENT_KEY] = model.get_latent_representation(n_compoents=50, svd_solver='arpack', random_state=42)

sc.pp.neighbors(adata, use_rep=SCBIOT_LATENT_KEY)
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.8, key_added=f'leiden_{SCBIOT_LATENT_KEY}')

```

To process snATAC-seq dataset

```python

# Usage
adata_top = scb.pp.remove_promoter_proximal_peaks(
    adata_atac,
    f"{dir}/inputs/gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz"    
)

# Peak selection
scb.pp.find_variable_features(adata_top, batch_key="batchname_all")

# TF-IDF
scb.pp.add_iterative_lsi(adata_top, n_components=31, drop_first_component=True, add_key="X_lsi")

# Save back
adata.obsm["X_lsi"] = adata_top.obsm["X_lsi"]
adata.obsm["Unintegrated"] = adata_top.obsm["X_lsi"]

# Optimal transport
adata, metrics = scb.ot.integrate(
    adata,
    modality='atac',
    obsm_key="X_lsi",
    batch_key="batchname_all",
    out_key="X_ot",
    reference="largest",  
    
)
print(metrics)

# 1. Compute neighbors using Harmony-corrected PCA
sc.pp.neighbors(adata, use_rep='X_ot', metric='cosine')
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.02, key_added='leiden_X_ot')

# Model training
scb.pp.setup_anndata(adata, var_key='X_ot', batch_key='batchname_all', pseudo_key='leiden_X_ot', true_key=None)
model = scb.models.vae(adata, prior_pcr=5., verbose=True)
model.train()
SCBIOT_LATENT_KEY = "scBIOT"
adata.obsm[SCBIOT_LATENT_KEY] = model.get_latent_representation(n_compoents=30, svd_solver='arpack', random_state=42)

sc.pp.neighbors(adata, use_rep=SCBIOT_LATENT_KEY)
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.8, key_added=f'leiden_{SCBIOT_LATENT_KEY}')

```

## API surface

Refer to `examples/examples.py` for a runnable end-to-end notebook-friendly
script, and the `tests/` folder to see terse usage patterns.

## Development setup

```bash
git clone https://github.com/haihuilab/scbiot.git
cd scbiot
pip install -e .[dev,docs]
pytest
make -C docs html
```

We use Hatch for packaging; the version is stored in `src/scbiot/__about__.py`.
See `CONTRIBUTING.md` (coming soon) for coding standards and contribution tips.
