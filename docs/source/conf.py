# Configuration file for the Sphinx documentation builder.

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("SCBIOT_DOCS", "1")

# Importing scbiot triggers heavy optional deps; read version directly instead.
ABOUT_PATH = ROOT / "src" / "scbiot" / "__about__.py"
about: dict[str, object] = {}
with ABOUT_PATH.open("r", encoding="utf-8") as fh:
    exec(fh.read(), about)
package_version = about["__version__"]  # type: ignore[index]

# -- Project information

project = 'scBIOT'
copyright = '2026, Haihui Zhang'
author = 'Haihui Zhang'

release = package_version
version = package_version

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'myst_parser',
    'sphinx_copybutton',
    'sphinx_design',
]

# Mock heavy optional dependencies so autodoc can import scbiot without installing them.
autodoc_mock_imports = [
    "anndata",
    "faiss",
    "numpy",
    "ot",
    "pandas",
    "pyranges",
    "scanpy",
    "scipy",
    "sklearn",
    "torch",
    "tqdm",
]

autodoc_typehints = "description"
autosummary_generate = True

# Ensure mocked imports are active before autosummary tries to import modules.
try:
    from sphinx.ext.autodoc.mock import MockFinder

    _mock_finder = MockFinder(autodoc_mock_imports)
    if _mock_finder not in sys.meta_path:
        sys.meta_path.insert(0, _mock_finder)
except Exception:
    # Fall back gracefully if Sphinx internals change.
    pass

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "generated/scbiot.ot.integrate_centroids.rst",
    "generated/scbiot.ot.integrate_ot.rst",
    "generated/scbiot.pp.ensure_anndata_setup.rst",
    "generated/scbiot.pp.get_anndata_setup.rst",
    "generated/scbiot.pp.setup_anndata.rst",
]

# Support both reStructuredText and Markdown sources
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for HTML output

html_theme = 'scanpydoc'
html_theme_options = {
    "repository_url": "https://github.com/haihuilab/scbiot",
    "repository_branch": "main",
    "path_to_docs": "docs/source",
    "use_repository_button": True,
    "use_source_button": True,
    "use_edit_page_button": True,
    "use_issues_button": True,
    "secondary_sidebar_items": {"**": ["page-toc", "sourcelink"]},
    "show_navbar_depth": 2,
    "home_page_in_toc": True,
}
html_static_path = ['_static']
html_favicon = '_static/scbiot_logo.svg'
html_logo = '_static/scbiot_logo.svg'
html_css_files = ['custom.css']
html_js_files = ['landing-search.js']

# -- Options for EPUB output
epub_show_urls = 'footnote'
