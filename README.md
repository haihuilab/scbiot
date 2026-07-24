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

The default installation contains the AE and optimal-transport pipeline without
benchmarking frameworks, Jupyter, or FAISS. Add only the features you need:

```bash
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
```

Do not install the `cpu` and `gpu` FAISS extras together. Without either extra,
scBIOT uses its scikit-learn fallback where available; centroid interpolation
requires a FAISS backend. On Linux, the GPU extra uses PyTorch 2.10 with CUDA
12.8 and FAISS GPU 1.14; use an NVIDIA 570-series or newer driver. See the
[installation guide](https://scbiot.readthedocs.io/en/stable/installation.html)
for details.

## Data availability

Prepared tutorial inputs are archived in the
[scBIOT Figshare collection](https://figshare.com/articles/dataset/Anndata_for_scBIOT_analysis/30671669).
Inputs not present there are linked to their original providers in the
[data-source documentation](https://scbiot.readthedocs.io/en/stable/data_source.html).

## Quick start

- Detailed documentation is published on [Read the Docs](https://scbiot.readthedocs.io/en/stable/).
- The [`tutorials/`](tutorials/) directory contains runnable end-to-end notebooks.

The example below deterministically holds out batch `B4` as an unlabeled query
and retains the original `cell_type` column for evaluation:

```python
import scanpy as sc
import scbiot as scb

adata = sc.read(
    "lung_atlas.h5ad",
    backup_url="https://figshare.com/ndownloader/files/24539942",
)

LABEL_KEY = "semi_cell_type"
UNLABELED = "Unknown"
QUERY_BATCH = "B4"
RANDOM_STATE = 0

# The downloaded AnnData contains batch, cell_type, and a counts layer.
required_obs = {"batch", "cell_type"}
missing_obs = required_obs.difference(adata.obs.columns)
if missing_obs:
    raise KeyError(f"Missing required obs columns: {sorted(missing_obs)}")
if "counts" not in adata.layers:
    raise KeyError("Missing required adata.layers['counts']")

# Build the semi-supervised label column used by both AE and supBIOT.
batch = adata.obs["batch"].astype("string")
query_mask = batch.eq(QUERY_BATCH).fillna(False)
if not query_mask.any() or query_mask.all():
    raise ValueError(f"QUERY_BATCH={QUERY_BATCH!r} does not define a valid query")
adata.obs[LABEL_KEY] = adata.obs["cell_type"].astype("string")
adata.obs.loc[query_mask, LABEL_KEY] = UNLABELED

# 1. Learn the v1.2 linear-autoencoder representation.
adata = scb.pp.autoencoder(
    adata,
    input_key="counts",
    out_key="X_ae",
    batch_key="batch",
    random_state=RANDOM_STATE,
)

# 2. Align batches in the autoencoder representation with optimal transport.
adata, metrics = scb.ot.integrate(
    adata,
    obsm_key="X_ae",
    batch_key="batch",
    out_key="X_supbiot",
    label_key=LABEL_KEY,
    unlabeled_category=UNLABELED,
    random_state=RANDOM_STATE,
)
print(metrics)

# 3. Transfer reference labels to the unlabeled query cells.
adata = scb.ot.supbiot(
    adata,
    use_rep="X_supbiot",
    input_rep_key="X_ae",
    label_key=LABEL_KEY,
    unlabeled_category=UNLABELED,
    pred_label_key="pred_cell_type",
    pred_conf_key="pred_confidence",
    min_conf=0.0,
    random_state=RANDOM_STATE,
)

sc.pp.neighbors(adata, use_rep="X_supbiot")
sc.tl.umap(adata, random_state=RANDOM_STATE)

# Combine known reference labels and transferred query labels for visualization.
adata.obs["supbiot_cell_type"] = adata.obs[LABEL_KEY].astype("string")
adata.obs.loc[query_mask, "supbiot_cell_type"] = adata.obs.loc[
    query_mask, "pred_cell_type"
].astype("string")
sc.pl.umap(adata, color=["batch", "supbiot_cell_type", "pred_confidence"])
```



### Scaling options

Replace step 2 above with one of the following calls, then continue with step 3.
For a faster approximate OT run on large datasets, enable the approximate solver:

```python
adata, metrics = scb.ot.integrate(
    adata,
    obsm_key="X_ae",
    batch_key="batch",
    out_key="X_supbiot",
    label_key=LABEL_KEY,
    unlabeled_category=UNLABELED,
    approximate=True,
    random_state=RANDOM_STATE,
)
```

For ultra-large datasets, use centroid-level OT:

```python
adata, metrics = scb.ot.integrate(
    adata,
    obsm_key="X_ae",
    batch_key="batch",
    out_key="X_supbiot",
    label_key=LABEL_KEY,
    unlabeled_category=UNLABELED,
    centroid=True,
    random_state=RANDOM_STATE,
)
```

For cross-modality reference mapping in v1.2.0, use
`scb.pp.autoencoder_map(reference, query, out_key="X_ae")`, then integrate
`X_ae` and transfer labels with `scb.ot.supbiot`.

See the dedicated documentation for
[multi-omics](https://scbiot.readthedocs.io/en/stable/multiomics.html) and
[spatiotemporal dynamics](https://scbiot.readthedocs.io/en/stable/spatiotemporal.html).
