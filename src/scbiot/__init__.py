"""
Convenience shim so downstream code can access the public API from a single place.
"""

from __future__ import annotations

import os

from .__about__ import __version__, __version_info__


def _ensure_pandas_value_counts() -> None:
    try:
        import pandas as pd
    except Exception:
        return
    if hasattr(pd, "value_counts"):
        return

    def _value_counts(values, *args, **kwargs):
        return pd.Series(values).value_counts(*args, **kwargs)

    pd.value_counts = _value_counts  # type: ignore[attr-defined]

print(f"scbiot version {__version__}")
_ensure_pandas_value_counts()

# Keep OT helpers importable for docs without pulling heavy training dependencies.
DOCS_MODE = bool(os.environ.get("SCBIOT_DOCS"))

from . import models
from . import ot
from . import pl
from . import pp

if DOCS_MODE:
    VAE = Model_VAE = Encoder_model = Decoder_model = VAEModel = compute_loss = None  # type: ignore
    VAETrainer = None  # type: ignore
else:
    from .models.vae import (
        VAE,
        Model_VAE,
        Encoder_model,
        Decoder_model,
    )
    from .models.vae_train import (
        VAEModel,
        compute_loss,
    )
    VAETrainer = None  # placeholder for backwards compatibility

version_info = __version_info__

__all__ = [
    "__version__",
    "version_info",
    "VAE",
    "Model_VAE",
    "Encoder_model",
    "Decoder_model",
    "VAEModel",
    "VAETrainer",
    "compute_loss",
    "models",
    "ot",
    "pl",
    "pp",
    "spatial",
    "pp_spatial",
]
