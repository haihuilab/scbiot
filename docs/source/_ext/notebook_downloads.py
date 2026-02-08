from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from sphinx.application import Sphinx
from sphinx.errors import SphinxError
from sphinx.util import logging

import shutil


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class NotebookInfo:
    source_path: Path
    output_dir: str
    notebook_relpath: str


def resolve_notebook_info(
    srcdir: Path,
    pagename: str,
    page_source_suffix: str | None = None,
    output_dir: str = "_notebooks",
) -> NotebookInfo | None:
    notebook_relpath = PurePosixPath(f"{pagename}.ipynb")
    source_path = srcdir / notebook_relpath
    static_path = srcdir / "_static" / "notebooks" / notebook_relpath

    def build_info(path: Path) -> NotebookInfo:
        return NotebookInfo(
            source_path=path,
            output_dir=output_dir,
            notebook_relpath=notebook_relpath.as_posix(),
        )

    if source_path.exists():
        return build_info(source_path)

    if static_path.exists():
        return build_info(static_path)

    if page_source_suffix == ".ipynb":
        LOGGER.warning("Notebook source missing for %s", source_path)

    return None


def _ensure_tracker(app: Sphinx) -> dict[str, NotebookInfo]:
    entries = getattr(app.env, "_scbiot_notebook_entries", None)
    if entries is None:
        entries = {}
        setattr(app.env, "_scbiot_notebook_entries", entries)
    return entries


def _set_notebook_context(
    app: Sphinx,
    pagename: str,
    templatename: str,
    context: dict[str, Any],
    doctree: Any,
) -> None:
    output_dir = getattr(app.config, "notebook_downloads_output_dir", "_notebooks")
    info = resolve_notebook_info(
        Path(app.srcdir),
        pagename,
        context.get("page_source_suffix"),
        output_dir=output_dir,
    )

    context["has_notebook"] = info is not None
    context["notebook_download_url"] = ""

    if not info:
        return

    pathto = context.get("pathto")
    base = pathto(info.output_dir, 1) if callable(pathto) else info.output_dir
    context["notebook_download_url"] = f"{base}/{info.notebook_relpath}"

    entries = _ensure_tracker(app)
    entries[pagename] = info


def _copy_notebook_files(app: Sphinx, entries: dict[str, NotebookInfo]) -> None:
    outdir = Path(app.builder.outdir)
    for info in entries.values():
        if not info.source_path.exists():
            LOGGER.warning("Notebook source missing for %s", info.source_path)
            continue
        target = outdir / Path(info.output_dir) / Path(info.notebook_relpath)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(info.source_path, target)


def verify_notebook_downloads(
    html_dir: Path,
    entries: dict[str, NotebookInfo],
    *,
    strict: bool = True,
) -> None:
    if not entries:
        return

    errors: list[str] = []
    for pagename, info in entries.items():
        html_path = html_dir / Path(pagename).with_suffix(".html")
        if not html_path.exists():
            errors.append(f"{pagename}: missing HTML output {html_path}")
            continue

        contents = html_path.read_text(encoding="utf-8", errors="ignore")
        if "btn-download-notebook-button" not in contents:
            errors.append(f"{pagename}: missing .ipynb download entry")

        expected_notebook = (
            html_dir / Path(info.output_dir) / Path(info.notebook_relpath)
        )
        if not expected_notebook.exists():
            errors.append(f"{pagename}: missing notebook file {expected_notebook}")

    if not errors:
        return

    message = "Notebook download verification failed:\n- " + "\n- ".join(errors)
    if strict:
        raise SphinxError(message)
    LOGGER.warning(message)


def _on_build_finished(app: Sphinx, exception: Exception | None) -> None:
    if exception or app.builder.name != "html":
        return
    entries = getattr(app.env, "_scbiot_notebook_entries", {})
    _copy_notebook_files(app, entries)
    strict = getattr(app.config, "notebook_downloads_strict", True)
    verify_notebook_downloads(
        Path(app.builder.outdir), entries, strict=strict
    )


def _reset_notebook_entries(app: Sphinx) -> None:
    setattr(app.env, "_scbiot_notebook_entries", {})


def setup(app: Sphinx) -> dict[str, Any]:
    app.connect("builder-inited", _reset_notebook_entries)
    app.connect("html-page-context", _set_notebook_context)
    app.connect("build-finished", _on_build_finished)
    app.add_config_value("notebook_downloads_output_dir", "_notebooks", "env")
    app.add_config_value("notebook_downloads_strict", True, "env")
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
