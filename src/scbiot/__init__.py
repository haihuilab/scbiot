"""Convenience shim so downstream code can access the public API from a single place."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from .__about__ import __version__, __version_info__

if TYPE_CHECKING:
    from . import ot, pl, pp, tl


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

_ensure_pandas_value_counts()
_PUBLIC_SUBMODULES = {"ot", "pl", "pp", "tl"}

# Map top-level convenience names to ``(submodule, attribute)`` so the public
# spatial-temporal API (``sb.integrate``, ``sb.velocity_field_sb_centroids``) is
# importable directly from the package root without eagerly loading submodules.
_PUBLIC_FUNCTIONS = {
    "integrate": ("ot", "integrate"),
    "velocity_field_sb_centroids": ("tl", "velocity_field_sb_centroids"),
}


def __getattr__(name: str):
    if name in _PUBLIC_SUBMODULES:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    if name in _PUBLIC_FUNCTIONS:
        submodule, attr = _PUBLIC_FUNCTIONS[name]
        obj = getattr(import_module(f".{submodule}", __name__), attr)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

version_info = __version_info__

__all__ = [
    "__version__",
    "version_info",
    "ot",
    "pl",
    "pp",
    "tl",
    "integrate",
    "velocity_field_sb_centroids",
]
