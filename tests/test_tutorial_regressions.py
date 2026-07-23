from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

from scbiot.ot._time import _time_to_numeric
from scbiot.pp.autoencoder import _prepare_log1p_hvg_matrix
from scbiot.tl import compute_transport_energy


def test_ordered_categorical_time_uses_category_order() -> None:
    values = pd.Series(
        pd.Categorical(
            ["late", "early", "middle"],
            categories=["early", "middle", "late"],
            ordered=True,
        )
    )

    numeric, metadata = _time_to_numeric(values)

    np.testing.assert_array_equal(numeric, [2.0, 0.0, 1.0])
    assert metadata["categories"] == ["early", "middle", "late"]


def test_transport_energy_returns_mutated_anndata() -> None:
    adata = AnnData(X=sparse.csr_matrix([[1.0, 0.0], [0.0, 2.0]]))
    adata.layers["transport_fwd"] = sparse.csr_matrix(
        [[2.0, 0.0], [1.0, 2.0]]
    )

    result = compute_transport_energy(adata)

    assert result is adata
    np.testing.assert_array_equal(adata.obs["transport_energy"], [1.0, 1.0])


def test_autoencoder_hvg_has_raw_count_fallback(monkeypatch) -> None:
    import scanpy as sc

    adata = AnnData(
        X=sparse.csr_matrix(
            [
                [1, 0, 2, 0],
                [0, 4, 0, 1],
                [2, 0, 5, 0],
                [0, 3, 0, 2],
            ],
            dtype=np.float32,
        )
    )

    def fail_loess(*args, **kwargs):
        raise ValueError("singular LOESS fit")

    monkeypatch.setattr(sc.pp, "highly_variable_genes", fail_loess)
    matrix = _prepare_log1p_hvg_matrix(
        adata,
        input_key=None,
        batch_key=None,
        n_top_genes=2,
    )

    assert matrix.shape == (4, 2)
    assert np.isfinite(matrix.data).all()
