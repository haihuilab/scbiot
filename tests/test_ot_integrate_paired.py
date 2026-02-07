from __future__ import annotations

import os

import numpy as np
import pandas as pd

os.environ.setdefault("SCBIOT_DOCS", "1")

import scbiot.ot as ot


class DummyAdata:
    def __init__(self, X_base: np.ndarray, X_view: np.ndarray, batch: np.ndarray) -> None:
        self.obsm = {
            "X_pca": np.asarray(X_base, dtype=np.float32),
            "X_lsi": np.asarray(X_view, dtype=np.float32),
        }
        self.obs = pd.DataFrame({"batch": batch})

    @property
    def n_obs(self) -> int:
        return int(self.obsm["X_pca"].shape[0])


def _zscore(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    mean = X.mean(0, keepdims=True)
    std = X.std(0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (X - mean) / std


def test_paired_shape_and_key_presence() -> None:
    rng = np.random.default_rng(0)
    X_base = rng.standard_normal((10, 4))
    X_view = rng.standard_normal((10, 6))
    adata = DummyAdata(X_base, X_view, batch=np.array(["a"] * 10))

    adata, metrics = ot.integrate(
        adata,
        preset="paired",
        out_key="X_joint",
        approximate_ot=False,
        centroid_ot=False,
        use_gpu=False,
        ot_backend="torch",
    )

    assert "X_joint" in adata.obsm
    assert adata.obsm["X_joint"].shape == (10, 4)
    assert metrics["preset"] == "paired"


def test_paired_strong_diagonal_prior_identity() -> None:
    rng = np.random.default_rng(1)
    X_base = rng.standard_normal((8, 5))
    X_view = X_base + 0.01 * rng.standard_normal((8, 5))
    adata = DummyAdata(X_base, X_view, batch=np.array(["a"] * 8))

    adata, _ = ot.integrate(
        adata,
        preset="paired",
        out_key="X_joint",
        w_base=0.0,
        w_view=1.0,
        diag_mass=0.95,
        prior_strength=6.0,
        reg=0.02,
        reg_m=0.2,
        use_gpu=False,
        ot_backend="torch",
    )

    X_joint = adata.obsm["X_joint"]
    X_base_z = _zscore(X_base)
    mse = float(np.mean((X_joint - X_base_z) ** 2))
    assert mse < 0.05


def test_paired_nudge_reduces_shift() -> None:
    rng = np.random.default_rng(2)
    X_base = rng.standard_normal((12, 4))
    X_view = X_base + 0.05 * rng.standard_normal((12, 4))
    adata = DummyAdata(X_base, X_view, batch=np.array(["a"] * 12))

    adata, _ = ot.integrate(
        adata,
        preset="paired",
        out_key="X_joint",
        w_base=0.5,
        w_view=0.5,
        diag_mass=0.8,
        prior_strength=2.0,
        reg=0.02,
        reg_m=0.2,
        use_gpu=False,
        ot_backend="torch",
    )

    X_joint = adata.obsm["X_joint"]
    X_base_z = _zscore(X_base)
    X_view_z = _zscore(X_view)
    mse_joint = float(np.mean((X_joint - X_base_z) ** 2))
    mse_view = float(np.mean((X_view_z - X_base_z) ** 2))
    assert mse_joint <= mse_view


def test_paired_centroid_ot_path() -> None:
    rng = np.random.default_rng(3)
    X_base = rng.standard_normal((6, 3))
    X_view = rng.standard_normal((6, 3))
    batch = np.array(["a", "a", "a", "b", "b", "b"])
    adata = DummyAdata(X_base, X_view, batch=batch)

    adata, metrics = ot.integrate(
        adata,
        preset="paired",
        out_key="X_joint",
        centroid_ot=True,
        w_base=0.0,
        w_view=1.0,
        use_gpu=False,
        ot_backend="torch",
    )

    X_joint = adata.obsm["X_joint"]
    for label in np.unique(batch):
        rows = X_joint[batch == label]
        assert np.allclose(rows, rows[0])
    assert metrics["centroid_ot"] is True


def test_paired_approximate_ot_path() -> None:
    rng = np.random.default_rng(4)
    X_base = rng.standard_normal((15, 4))
    X_view = rng.standard_normal((15, 4))
    adata = DummyAdata(X_base, X_view, batch=np.array(["a"] * 15))

    adata, metrics = ot.integrate(
        adata,
        preset="paired",
        out_key="X_joint",
        approximate_ot=True,
        K_ref=6,
        K_batch=6,
        random_state=7,
        use_gpu=False,
        ot_backend="torch",
    )

    assert adata.obsm["X_joint"].shape == (15, 4)
    assert metrics["approximate_ot"] is True
