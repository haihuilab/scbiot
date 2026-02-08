#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


DOCS_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = DOCS_DIR / "source"
EXT_DIR = SRC_DIR / "_ext"
sys.path.insert(0, str(EXT_DIR))

try:
    from notebook_downloads import resolve_notebook_info
except Exception as exc:
    raise SystemExit(f"Failed to import notebook_downloads: {exc}") from exc


def _iter_html_pages(html_dir: Path) -> list[tuple[Path, Path]]:
    pages: list[tuple[Path, Path]] = []
    for path in html_dir.rglob("*.html"):
        rel = path.relative_to(html_dir)
        pages.append((path, rel))
    return pages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify notebook download entries in built HTML pages."
    )
    parser.add_argument("--source-dir", type=Path, default=SRC_DIR)
    parser.add_argument("--html-dir", type=Path, default=DOCS_DIR / "_build" / "html")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("_notebooks"),
        help="Relative directory under the HTML output where notebooks are copied",
    )
    args = parser.parse_args()

    source_dir = args.source_dir
    html_dir = args.html_dir

    if not html_dir.exists():
        raise SystemExit(f"HTML output directory not found: {html_dir}")

    errors: list[str] = []
    checked = 0
    for html_path, rel in _iter_html_pages(html_dir):
        pagename = rel.with_suffix("").as_posix()
        info = resolve_notebook_info(
            source_dir, pagename, output_dir=args.output_dir.as_posix()
        )
        if not info:
            continue

        checked += 1
        contents = html_path.read_text(encoding="utf-8", errors="ignore")
        if "btn-download-notebook-button" not in contents:
            errors.append(f"{pagename}: missing .ipynb download entry")

        expected_notebook = (
            html_dir / args.output_dir / Path(info.notebook_relpath)
        )
        if not expected_notebook.exists():
            errors.append(f"{pagename}: missing notebook file {expected_notebook}")

    if errors:
        raise SystemExit(
            "Notebook download verification failed:\n- " + "\n- ".join(errors)
        )

    print(f"Notebook download verification OK ({checked} pages).")


if __name__ == "__main__":
    main()
