PYTHON ?= python
SPHINXOPTS ?=
SPHINXBUILD ?= sphinx-build
SOURCEDIR = source
BUILDDIR = build
DOCREQ = requirements.txt
JUPYTER_CACHE = $(CURDIR)/_jupyter_cache

# Put it first so that "make" without argument is like "make help".
help:
\t@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

install:
\t@$(PYTHON) -m pip install -e ..
\t@$(PYTHON) -m pip install -r $(DOCREQ)

clean:
\t@rm -rf "$(BUILDDIR)" "$(BUILDDIR)/jupyter_execute" "$(JUPYTER_CACHE)"

.PHONY: help install clean Makefile

html: install
\t@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
\t@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
