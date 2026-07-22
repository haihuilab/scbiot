# Configuration file for the Sphinx documentation builder.

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS_SOURCE = Path(__file__).resolve().parent
DOCS_ROOT = DOCS_SOURCE.parent
TUTORIALS_PATH = ROOT / "tutorials"
JUPYTER_CACHE = DOCS_ROOT / "_jupyter_cache"
sys.path.insert(0, str(DOCS_SOURCE / "_ext"))
os.environ.setdefault("SCBIOT_DOCS", "1")
os.environ.setdefault("SCBIOT_TUTORIALS_PATH", str(TUTORIALS_PATH))

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
    # 'myst_parser',
    'myst_nb',
    'sphinx_copybutton',
    'sphinx_design',
    'notebook_downloads',
]

# Mock heavy optional dependencies so autodoc can import scbiot without installing them.
autodoc_mock_imports = [
    "anndata",
    "faiss",
    "matplotlib",
    "numpy",
    "ot",
    "pandas",
    "pyranges",
    "scanpy",
    "scipy",
    "seaborn",
    "sklearn",
    "torch",
    "tqdm",
]

autodoc_typehints = "description"
autosummary_generate = True
suppress_warnings = ["myst.header"]

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

notebook_downloads_output_dir = "_notebooks"
notebook_downloads_strict = os.environ.get("READTHEDOCS") != "True"

# Support both reStructuredText and Markdown sources
source_suffix = {
    '.rst': 'restructuredtext',    
    '.ipynb': 'myst-nb',
}

NB_EXECUTION_MODE = os.environ.get("SCBIOT_DOCS_NB_EXECUTION", "off").strip().lower()
if NB_EXECUTION_MODE not in {"auto", "cache", "force", "off"}:
    NB_EXECUTION_MODE = "off"
nb_execution_mode = NB_EXECUTION_MODE
nb_execution_timeout = 600
nb_execution_cache_path = str(JUPYTER_CACHE)
nb_execution_raise_on_error = True
nb_execution_show_tb = True

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
