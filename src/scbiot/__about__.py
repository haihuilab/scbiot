"""Package version metadata."""

from __future__ import annotations

from typing import Tuple

__all__ = ["__version__", "__version_info__"]

__version__ = "1.0.1"


def _parse_version(version: str) -> Tuple[int | str, ...]:
    parts: list[int | str] = []
    for piece in version.replace("-", ".").split("."):
        if not piece:
            continue
        parts.append(int(piece) if piece.isdigit() else piece)
    return tuple(parts)


__version_info__ = _parse_version(__version__)
