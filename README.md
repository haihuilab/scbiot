# scBIOT

**scBIOT** is a lightweight Python library for single-cell omics integration. 
It bundles the preprocessing, embedding, transfer label workflows we routinely apply to RNA, ATAC, 
and paired or unpaired multi-omics datasets. The library emphasizes reproducible data preparation, 
single-cell clustering using embeddings derived from optimal transport and Transformer-based VAEs, 
and concise APIs that work out of the box on AnnData data.

## Highlights

- **Batteries-included preprocessing**: scATAC-seq peak processing, iterative LSI, and gene activity annotation.
- **Accurate atlas integration**: high-fidelity alignment with rare cell-type protection.
- **Unified scBIOT framework**: a single framework for embedding RNA, ATAC, transfer learning, and paired or unpaired multi-omics.
- **Fast integration via Optimal Transport (OT)**: scalable alignment for large single-cell datasets.
- **Transformer-VAE**: further enhanced integration for stronger representation learning and improved robustness.
- **Scales to 100M cells locally**: memory-efficent scalable processing.
- **Label transfer**: across multi-omics modalities and between spatial data and scRNA-seq references.

## Installation

```bash
pip install scbiot
```

For documentation builds install `pip install scbiot[docs]`.

### Optional extras

Depending on your workflow you can pull in heavier scientific stacks as extras:


- `pip install scbiot` installs the CUDA-enabled FAISS + PyTorch combo (CUDA 12) `faiss-gpu-cu12 scib_metrics==0.5.1 leidenalg jaxlib scikit-misc "jax[cuda12]" pyranges`.


For an exact replica of our Conda dev environment use `pip install -r requirements.txt`
inside a fresh virtual environment.

## Quick start
- Detailed documentation is published on [scbiot.readthedocs.io](https://scbiot.readthedocs.io/en/stable/)
and mirrors the examples below.
- Refer to `examples/` folder for a runnable end-to-end notebook-friendly script.

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

adata, metrics = scb.ot.integrate(adata, preset='rna', obsm_key='X_pca', batch_key='batch', out_key='X_ot')
print(metrics)

sc.pp.neighbors(adata, use_rep='X_ot')
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.8, key_added='leiden_X_ot')

scb.models.setup_anndata(adata, var_key='X_ot', batch_key='batch', true_key=None)
model = scb.models.vae(adata, verbose=True)
model.train()

SCBIOT_LATENT_KEY = "scBIOT"
adata.obsm[SCBIOT_LATENT_KEY] = model.get_latent_representation(n_compoents=50, svd_solver='arpack', random_state=42)

sc.pp.neighbors(adata, use_rep=SCBIOT_LATENT_KEY)
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.8, key_added=f'leiden_{SCBIOT_LATENT_KEY}')

```

For stable tuning, use the meta-parameter interface:

```python
adata, metrics = scb.ot.integrate(
    adata,
    preset="rna",
    epsilon=0.03,
    tau=0.40,
    knn_scale=1.0,
    batch_strength=1.0,
    gate_temperature=1.0,
    # optional supervision:
    label_key="semi_cell_type",
    unlabeled_category="Unknown",
    sup_strength=0.10,
)
```

### Scaling options

For ultra-large datasets, use centroid-level OT:

```python
adata, metrics = scb.ot.integrate(
    adata,
    preset="centroid",
    obsm_key="X_pca",
    batch_key="batch",
    out_key="scBIOT",
)
```

You can also enable centroid OT while keeping another preset's OT hyperparameters via
`centroid_ot=True`.

For a faster approximate OT run on large datasets, enable the approximate OT solver
while keeping your preset's data keys:

```python
adata, metrics = scb.ot.integrate(
    adata,
    preset="atac",
    obsm_key="X_lsi",
    batch_key="batchname_all",
    out_key="X_ot",
    approximate_ot=True,
)
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
    preset='atac',
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
scb.models.setup_anndata(adata, var_key='X_ot', batch_key='batchname_all', true_key=None)
model = scb.models.vae(adata, prior_pcr=5., verbose=True)
model.train()
SCBIOT_LATENT_KEY = "scBIOT"
adata.obsm[SCBIOT_LATENT_KEY] = model.get_latent_representation(n_compoents=30, svd_solver='arpack', random_state=42)

sc.pp.neighbors(adata, use_rep=SCBIOT_LATENT_KEY)
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.8, key_added=f'leiden_{SCBIOT_LATENT_KEY}')

```
