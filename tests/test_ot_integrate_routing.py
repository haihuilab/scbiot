from __future__ import annotations

import os

import pytest

os.environ.setdefault("SCBIOT_DOCS", "1")

import scbiot.ot as ot


def test_integrate_default_backward_compatibility(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def fake_integrate_ot(adata: object, modality: str | None = None, **kwargs: object):
        calls["adata"] = adata
        calls["modality"] = modality
        calls["kwargs"] = kwargs
        return adata, {"mix": 0.1}

    monkeypatch.setattr(ot, "integrate_ot", fake_integrate_ot)

    adata = object()
    out_adata, metrics = ot.integrate(adata, obsm_key="X_pca", batch_key="batch", out_key="X_ot")

    assert out_adata is adata
    assert metrics == {"mix": 0.1}
    assert calls["modality"] == "rna"
    assert "approximate_ot" not in calls["kwargs"]
    assert "centroid_ot" not in calls["kwargs"]


def test_integrate_centroid_preset_routes_to_centroids(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def fake_integrate_centroids(adata: object, modality: str | None = None, **kwargs: object):
        calls["adata"] = adata
        calls["modality"] = modality
        calls["kwargs"] = kwargs
        return adata, {"n_centroids": 12}

    monkeypatch.setattr(ot, "integrate_centroids", fake_integrate_centroids)
    monkeypatch.setattr(
        ot,
        "integrate_ot",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("integrate_ot should not be called")),
    )

    adata = object()
    out_adata, metrics = ot.integrate(adata, preset="centroid")

    assert out_adata is adata
    assert metrics["n_centroids"] == 12
    assert calls["modality"] is None


def test_integrate_centroid_flag_routes_to_centroids(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def fake_integrate_centroids(adata: object, modality: str | None = None, **kwargs: object):
        calls["modality"] = modality
        calls["kwargs"] = kwargs
        return adata, {"n_centroids": 4}

    monkeypatch.setattr(ot, "integrate_centroids", fake_integrate_centroids)

    adata = object()
    ot.integrate(adata, preset="anchor", centroid_ot=True)

    assert calls["modality"] == "anchor"
    assert calls["kwargs"]["n_centroids_per_batch"] == 2048
    assert calls["kwargs"]["max_samples_per_batch"] == 500_000


def test_integrate_approximate_ot_uses_approximate_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def fake_integrate_ot(adata: object, modality: str | None = None, **kwargs: object):
        calls["modality"] = modality
        calls["kwargs"] = kwargs
        return adata, {"mix": 0.2}

    monkeypatch.setattr(ot, "integrate_ot", fake_integrate_ot)

    adata = object()
    ot.integrate(adata, preset="atac", approximate_ot=True)

    assert calls["modality"] == "atac"
    assert calls["kwargs"]["obsm_key"] == "X_lsi"
    assert calls["kwargs"]["batch_key"] == "batchname_all"
    assert calls["kwargs"]["reference"] == "largest"
    assert calls["kwargs"]["approximate_ot"] is True


def test_integrate_mutual_exclusivity() -> None:
    adata = object()
    with pytest.raises(ValueError):
        ot.integrate(adata, approximate_ot=True, centroid_ot=True)


def test_anchor_reference_category_sets_reference_align(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def fake_integrate_ot(adata: object, modality: str | None = None, **kwargs: object):
        calls["kwargs"] = kwargs
        return adata, {}

    monkeypatch.setattr(ot, "integrate_ot", fake_integrate_ot)

    adata = object()
    ot.integrate(
        adata,
        preset="anchor",
        reference_category="reference",
        approximate_ot=False,
    )

    assert calls["kwargs"]["reference_align"] is True
    assert "approximate_ot" not in calls["kwargs"]


def test_anchor_reference_category_approximate_sets_reference_align(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    def fake_integrate_ot(adata: object, modality: str | None = None, **kwargs: object):
        calls["kwargs"] = kwargs
        return adata, {}

    monkeypatch.setattr(ot, "integrate_ot", fake_integrate_ot)

    adata = object()
    ot.integrate(
        adata,
        preset="anchor",
        reference_category="reference",
        approximate_ot=True,
    )

    assert calls["kwargs"]["reference_align"] is True
    assert calls["kwargs"]["approximate_ot"] is True
