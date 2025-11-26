# Configuration file for the Sphinx documentation builder.

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("SCBIOT_DOCS", "1")

# -- Project information

project = 'scBIOT'
copyright = '2025, Haihui Zhang'
author = 'Haihui Zhang'

release = '1.0.0'
version = '1.0.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'myst_parser',
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

# Support both reStructuredText and Markdown sources
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_favicon = '_static/scbiot_logo.svg'
html_logo = '_static/scbiot_logo.svg'

# -- Options for EPUB output
epub_show_urls = 'footnote'
