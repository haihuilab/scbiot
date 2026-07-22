# scBIOT

**scBIOT** is a Python library for optimal-transport integration and analysis of
single-cell RNA, ATAC, spatial, and paired or unpaired multi-omics data. Version
1.2.0 provides a compact parameter interface, scalable centroid integration,
linear-autoencoder embeddings, label transfer, and transport-aware downstream
analysis for AnnData objects.

## Highlights

- **Batteries-included preprocessing**: scATAC-seq peak processing, iterative LSI, and gene activity annotation.
- **Accurate atlas integration**: high-fidelity alignment with rare cell-type protection.
- **Unified scBIOT framework**: one interface for RNA, ATAC, spatial, temporal, and multi-omics integration.
- **Fast integration via Optimal Transport (OT)**: scalable alignment for large single-cell datasets.
- **Linear autoencoders**: PCA-like single-dataset embeddings and reference-to-query coembedding.
- **Scales to 100M cells locally**: memory-efficient processing with centroid-level OT.
- **Label transfer**: across multi-omics modalities and between spatial data and scRNA-seq references.
- **Transport-aware analysis**: gene transport scores, trajectory diagnostics, and plotting utilities.
- **Spatiotemporal dynamics**: spatial/time-aware integration and lineage-specific velocity fields.

## Installation

```bash
pip install scbiot
```

For documentation builds install `pip install scbiot[docs]`.

### Optional extras

For an exact replica of the development environment, use
`pip install -r requirements.txt` inside a fresh virtual environment. The
default package dependencies target CUDA 12; see the
[installation guide](https://scbiot.readthedocs.io/en/stable/installation.html)
for CPU and GPU notes.

## Quick start

- Detailed documentation is published on [Read the Docs](https://scbiot.readthedocs.io/en/stable/).
- The [`tutorials/`](tutorials/) directory contains runnable end-to-end notebooks.

```python
import scanpy as sc
import scbiot as scb

adata = sc.read(
    "lung_atlas.h5ad",
    backup_url="https://figshare.com/ndownloader/files/24539942",
)

sc.pp.highly_variable_genes(
    adata, n_top_genes=2000, flavor="seurat_v3", batch_key="batch"
)
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.scale(adata)
sc.tl.pca(adata, n_comps=30, use_highly_variable=True)

adata, metrics = scb.ot.integrate(
    adata,
    obsm_key="X_pca",
    batch_key="batch",
    out_key="X_ot",
)
print(metrics)

sc.pp.neighbors(adata, use_rep="X_ot")
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.8, key_added="leiden_X_ot")
```



### Scaling options

For ultra-large datasets, use centroid-level OT:

```python
adata, metrics = scb.ot.integrate(
    adata,    
    obsm_key="X_pca",
    batch_key="batch",
    out_key="X_ot",
    centroid=True
)
```


For a faster approximate OT run on large datasets, enable the approximate solver:

```python
adata, metrics = scb.ot.integrate(
    adata,    
    obsm_key="X_lsi",
    batch_key="batchname_all",
    out_key="X_ot",
    approximate=True,
)
```

To process snATAC-seq dataset

```python

adata_top = scb.pp.remove_promoter_proximal_peaks(
    adata_atac,
    "gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz",
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
    obsm_key="X_lsi",
    batch_key="batchname_all",
    out_key="X_ot",
    reference="largest",
)
print(metrics)

# 1. Compute neighbors using Harmony-corrected PCA
sc.pp.neighbors(adata, use_rep="X_ot", metric="cosine")
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.2, key_added="leiden_X_ot")

```

For cross-modality reference mapping in v1.2.0, use
`scb.pp.autoencoder_map(reference, query, out_key="X_ae")`, then integrate
`X_ae` and transfer labels with `scb.ot.supbiot`.

See the dedicated documentation for
[multi-omics](https://scbiot.readthedocs.io/en/stable/multiomics.html) and
[spatiotemporal dynamics](https://scbiot.readthedocs.io/en/stable/spatiotemporal.html).
