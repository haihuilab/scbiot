from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from anndata import AnnData

from .autoencoder import AUTOENCODER_OBSM_KEY, AutoencoderConfig

def autoencoder(
    adata: Any,
    *,
    input_key: str = "counts",
    out_key: str = "X_ae",
    n_top_genes: int = 3000,
    hidden_dim: int = 512,
    latent_dim: int = 50,
    batch_size: int = 512,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    dropout: float = 0.1,
    batch_key: str = "batch",
    l2: bool = False,
    random_state: int = 0,
    max_epochs: int = 100,
    early_stop_patience: int = 10,
    validation_split: float = 0.1,
    batch_in_decoder: bool = False,
    orthogonality_weight: float = 1e-2,
) -> Any: ...
def autoencoder_map(
    adata_reference: AnnData,
    adata_query: AnnData,
    *,
    out_key: str = "X_ae",
    label: str = "modality",
    keys: Sequence[str] = ("reference", "query"),
    reference_layer: Optional[str] = None,
    query_layer: Optional[str] = None,
    label_key: Optional[str] = None,
    unlabeled_category: str = "Unknown",
    batch_key: Optional[str] = None,
    n_top_genes: int = 5000,
    latent_dim: int = 150,
    n_components: Optional[int] = None,
    genes: Optional[Sequence[str]] = None,
    hidden_dim: int = 512,
    batch_size: int = 512,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    dropout: float = 0.1,
    max_epochs: int = 100,
    early_stop_patience: int = 10,
    validation_split: float = 0.1,
    orthogonality_weight: float = 1e-2,
    supervised_weight: float = 2.0,
    batch_in_decoder: bool = False,
    l2: bool = False,
    flag_outlier: bool = False,
    outlier_k: int = 30,
    outlier_z: float = 3.0,
    diffuse_query: bool = True,
    diffuse_k: int = 30,
    diffuse_iters: int = 2,
    random_state: int = 0,
    verbose: bool = True,
) -> AnnData: ...
__all__ = [
    "autoencoder",
    "autoencoder_map",
    "AUTOENCODER_OBSM_KEY",
    "AutoencoderConfig",
]
