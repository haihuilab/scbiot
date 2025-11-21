"""
Helpers for registering AnnData fields that the transformer VAE expects.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping, Optional

try:
    from anndata import AnnData as _AnnData  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    _AnnData = None  # type: ignore

SCBIOT_REGISTRY_KEY = "_scbiot_setup"


class AnnDataSetupError(RuntimeError):
    """Raised when AnnData metadata is missing or invalid."""


def _matrix_exists(adata: Any, key: Optional[str]) -> bool:
    if key in (None, "X"):
        return adata.X is not None
    if key in getattr(adata, "obsm", {}):
        return True
    if key in getattr(adata, "layers", {}):
        return True
    return False


def _validate_matrix(adata: Any, key: Optional[str]) -> None:
    if key is None:
        raise AnnDataSetupError("var_key cannot be None for transformer VAE training.")
    if not _matrix_exists(adata, key):
        raise KeyError(
            f"Could not find '{key}' in adata.obsm, adata.layers, or as the primary matrix."
        )


def _validate_obs(adata: Any, key: Optional[str], *, field_name: str) -> None:
    if key is None:
        raise AnnDataSetupError(f"{field_name} must be provided, got None.")
    if key not in adata.obs.columns:
        raise KeyError(f"'{key}' not found in adata.obs for {field_name}.")


def _registry(adata: Any) -> MutableMapping[str, Any]:
    if SCBIOT_REGISTRY_KEY not in adata.uns:
        adata.uns[SCBIOT_REGISTRY_KEY] = {}
    registry = adata.uns[SCBIOT_REGISTRY_KEY]
    assert isinstance(registry, MutableMapping)
    return registry


def get_anndata_setup(adata: Any) -> Dict[str, Any]:
    """
    Return a shallow copy of the registered AnnData configuration.
    """
    payload = adata.uns.get(SCBIOT_REGISTRY_KEY, {})
    assert isinstance(payload, Mapping)
    return dict(payload)


def ensure_anndata_setup(adata: Any) -> Dict[str, Any]:
    """
    Return the registered AnnData setup or raise if it is missing.
    """
    payload = get_anndata_setup(adata)
    if not payload:
        raise AnnDataSetupError(
            "Call `scbiot.pp.setup_anndata(...)` before instantiating the transformer VAE."
        )
    return payload


def setup_anndata(
    adata: Any,
    *,
    var_key: str = "scBIOT_OT",
    batch_key: str = "batch",
    pseudo_key: Optional[str] = "leiden_scBIOT_OT",
    true_key: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Register the AnnData fields the transformer VAE relies on.

    Parameters
    ----------
    adata:
        AnnData object that already contains the OT-aligned embedding.
    var_key:
        Key pointing to the numerical representation (``adata.obsm[var_key]`` or ``adata.X``).
    batch_key:
        ``adata.obs`` column that stores batch labels.
    pseudo_key / true_key:
        Optional ``adata.obs`` columns for pseudo and ground-truth labels. Set either to
        ``None`` to disable that label type (``true_key`` defaults to ``None``).
    overwrite:
        Replace an existing registration in ``adata.uns`` when True.
    """
    if _AnnData is not None and not isinstance(adata, _AnnData):
        raise TypeError("setup_anndata expects an AnnData object.")

    _validate_matrix(adata, var_key)
    _validate_obs(adata, batch_key, field_name="batch_key")
    if pseudo_key is not None and pseudo_key not in adata.obs.columns:
        raise KeyError(f"'{pseudo_key}' not found in adata.obs for pseudo_key.")
    if true_key is not None and true_key not in adata.obs.columns:
        raise KeyError(f"'{true_key}' not found in adata.obs for true_key.")

    registry = _registry(adata)
    if registry and not overwrite:
        # Avoid surprising silent overrides.
        raise AnnDataSetupError(
            "scBIOT AnnData setup already exists. "
            "Pass overwrite=True to replace the registered keys."
        )

    registry.clear()
    registry.update(
        {
            "var_key": var_key,
            "batch_key": batch_key,
            "pseudo_key": pseudo_key,
            "true_key": true_key,
        }
    )
    return dict(registry)
