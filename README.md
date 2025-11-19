# scBIOT

**scBIOT** (Single-Cell Biological Insights via Optimal Transport and Omics Transformers) is a
light-weight Python library that packages the preprocessing and embedding
workflow we commonly apply to RNA, ATAC, and joint multi-omics datasets.
It focuses on repeatable data preparation, explainable latent embeddings, and
concise APIs that work out-of-the-box on real AnnData data.

## Highlights

- Batteries-included preprocessing: preprocessing of ATAC peaks, iterative LSI, and annotation of gene activity.
- A unified `ScBIOT` class that can embed RNA, ATAC, or paired multi-omics
  modalities and reuse the fitted pipeline for inference on new batches.
- Built-in clustering (`k`-Means) and feature-loading inspection to reason
  about latent components.
- Pure Python + NumPy/Pandas/scikit-learn dependency stack — no GPU required.

Documentation is published on [scbiot.readthedocs.io](https://scbiot.readthedocs.io/en/latest/)
and mirrors the examples below.

## Installation

```bash
pip install scbiot
```

The package targets Python 3.9+ and only depends on NumPy, pandas, and
scikit-learn. For documentation builds install `pip install scbiot[docs]`.

### Optional extras

Depending on your workflow you can pull in heavier scientific stacks as extras:


- `pip install scbiot` installs the CUDA-enabled FAISS + PyTorch combo (CUDA 12).


For an exact replica of our Conda dev environment use `pip install -r requirements.txt`
inside a fresh virtual environment.

## Quick start

```python
import numpy as np
import pandas as pd
from scbiot import ScBIOT
import scanpy as sc


adata = sc.datasets.pbmc3k()
model = ScBIOT(mode="RNA", latent_dim=16, n_top_genes=500)
embeddings = model.train(counts)

result = model.inference(counts, k=8)
print(result.embeddings.head())
print(result.clusters.value_counts())

loadings = model.get_feature_loadings(top_n=5)
print(loadings.head())
```

To process a paired multi-omic dataset simply pass a dictionary where each key
denotes a modality:

```python
rna = counts
atac = counts.sample(frac=1.0, replace=True)  # stand-in for demo
model = ScBIOT(mode="multi", latent_dim=24)
embeddings = model.train({"RNA": rna, "ATAC": atac})
```

## API surface

- `scbiot.ScBIOT`: orchestrates preprocessing, dimensionality reduction and
  clustering.
- `scbiot.ScBIOTResult`: simple dataclass returned by `inference`.

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
