# Codex Guideline: Refactor `scbiot` docs to `.ipynb` only (no PDF anywhere)

## Goal
Make the documentation build **reliable on Read the Docs** and ensure:
- Every tutorial page is sourced from a **committed `.ipynb`**.
- The built site offers **downloadable `.ipynb`** notebooks (**no `.pdf`** anywhere).
- The docs build **never fails** due to overly strict notebook download verification.
- No broken `.. include::` paths; use **toctrees** pointing directly to notebooks.
- Autosummary targets are importable (no bogus module paths).

---

## Scope & constraints
- Keep docs dependencies lightweight (docs must build without installing heavy runtime deps).
- Keep changes scoped to:
  - `docs/source/conf.py`
  - `docs/source/_ext/`
  - `docs/source/**/*.rst`
  - `docs/source/tutorials/*.ipynb` (plus renames)
- Must pass:
  ```bash
  python -m sphinx -T -b html -d docs/_build/doctrees docs/source docs/_build/html
  ```

---

## A) Remove PDF usage (repo-wide)
### A1. Delete all PDF links and generation paths
**Actions**
- Remove any `:download:` links pointing to `.pdf`.
- Remove any config, scripts, or extension code that creates or validates PDFs.
- Remove references to:
  - `latexpdf`, `rinoh`, `pdflatex`, `nbconvert` (PDF mode), `*.pdf`

**Acceptance**
- Searching under `docs/` for PDF strings returns nothing:
  ```bash
  rg -n "\.pdf\b|latexpdf\b|rinoh\b|pdflatex\b" docs/
  ```

---

## B) Make `.ipynb` the single source of truth for tutorials
### B1. Ensure all tutorials referenced in toctrees exist as `.ipynb`
**Actions**
- For every tutorial referenced from `index.rst`, `tutorials.rst`, etc.:
  - verify `docs/source/tutorials/<name>.ipynb` exists and is committed.
- Remove reliance on generated `.ipynb.rst` artifacts as “sources”.
  - If `*.ipynb.rst` exists, treat it as build output and stop referencing it as the canonical source.

**Acceptance**
- Every tutorial entry in a toctree resolves to a real `.ipynb`.

### B2. Normalize notebook filenames (no dots in basenames)
**Actions**
- Rename:
  - `docs/source/tutorials/7_brain_1.3M_integration.ipynb`
  - → `docs/source/tutorials/7_brain_1_3M_integration.ipynb`
- Update all references (toctrees, links).

**Acceptance**
- No tutorial `.ipynb` filename contains extra dots besides the `.ipynb` extension.

---

## C) Replace broken `include::` directives with toctrees
Your logs show `CRITICAL: Problems with "include" directive path` from missing files.

### C1. Remove `.. include:: tutorials/<missing>.rst`
**Actions**
- For each `docs/source/tutorials/*.rst` file that contains:
  - `.. include:: tutorials/<something>.rst`
- Replace with a toctree pointing to the actual notebook docnames:
  ```rst
  .. toctree::
     :maxdepth: 1

     4a_paired_multiomics_10x_multiome
     4b_paired_multiomics_snare_seq
  ```
  (Do **not** include the `.ipynb` extension in toctree entries.)

**Acceptance**
- Sphinx build shows **no CRITICAL include errors**.

---

## D) Configure Sphinx to treat `.ipynb` as first-class pages
### D1. Enable `myst_nb` and disable execution for RTD
**Actions**
- In `docs/source/conf.py`:
  - ensure `myst_nb` is in `extensions`
  - set notebook execution off:
    ```python
    nb_execution_mode = "off"
    ```
- Keep `docs/requirements.txt` docs-only.

**Acceptance**
- `.ipynb` tutorials render as HTML pages without executing cells.

---

## E) Provide `.ipynb` downloads (and only `.ipynb`)
### Policy
**All notebooks that appear as docs pages must be downloadable as `.ipynb`** from the built HTML.

### E1. Copy notebook sources into the built HTML output
**Actions**
- Update `docs/source/_ext/notebook_downloads.py` so it:
  - finds `.ipynb` sources among docs
  - copies each `.ipynb` into HTML output under `_notebooks/…` preserving relative paths.

**Implementation requirements**
- Add config values in the extension:
  - `notebook_downloads_output_dir` default `"_notebooks"`
  - `notebook_downloads_strict` default `True`
- During `build-finished`:
  - for each found `.ipynb` doc source:
    - copy `docs/source/tutorials/X.ipynb` → `<outdir>/_notebooks/tutorials/X.ipynb`
    - create dirs with `mkdir(parents=True, exist_ok=True)`
    - use `shutil.copy2`

**Acceptance**
- Built output contains:
  - `docs/_build/html/_notebooks/tutorials/*.ipynb`

### E2. Add a “Download this notebook (.ipynb)” link on each notebook page
**Actions**
- In the extension, connect to `html-page-context`:
  - if current page corresponds to a notebook doc:
    - inject `context["notebook_download_url"] = "/_notebooks/<relative>.ipynb"`
- Add a template override:
  - create `docs/source/_templates/page.html` (or appropriate theme override)
  - render a small link when `notebook_download_url` exists.
- In `conf.py`:
  ```python
  templates_path = ["_templates"]
  ```

**Acceptance**
- Notebook pages show a download link and it points to a real file.

### E3. Make verification `.ipynb`-only and RTD-safe
**Actions**
- Remove any PDF checks from verification.
- Verification checks only that each notebook page has a corresponding `_notebooks/.../*.ipynb` copied file.
- On RTD, **do not hard fail** builds for missing downloads:
  - log a warning instead.
- In `conf.py`:
  ```python
  import os
  notebook_downloads_strict = os.environ.get("READTHEDOCS") != "True"
  ```

**Acceptance**
- RTD build succeeds even if a notebook download entry is missing (warning-only).

---

## F) Fix autosummary import failures
Your logs show invalid autosummary targets like `scbiot.models.get_latent_representation`.

### F1. Use importable autosummary targets
**Actions**
- Update autosummary lists to reference:
  - `scbiot.models.VAEModel`
  - `scbiot.models.VAEModel.get_latent_representation`
- Remove or correct any target that is not importable.

**Acceptance**
- No `autosummary` “failed to import …” warnings for scbiot symbols.

---

## G) Final acceptance tests
Run locally:
```bash
python -m pip install -r docs/requirements.txt
python -m pip install --no-deps .
python -m sphinx -T -b html -d docs/_build/doctrees docs/source docs/_build/html
```

Pass criteria:
- No `CRITICAL: Problems with "include" directive path`.
- No `SphinxError` from `notebook_downloads`.
- Output includes `_notebooks/tutorials/*.ipynb`.
- No PDFs referenced in built output:
  ```bash
  rg -n "\.pdf\b" docs/_build/html
  ```

---

## One-shot Codex execution prompt (copy/paste)
> Refactor the `scbiot` documentation so tutorials are `.ipynb`-only (no PDFs). Remove all `.pdf` download links, PDF generation, and any PDF verification. Replace broken `.. include::` tutorial wrappers with `.. toctree::` entries that point directly to existing `.ipynb` tutorial docnames. Rename `docs/source/tutorials/7_brain_1.3M_integration.ipynb` to `7_brain_1_3M_integration.ipynb` and update references. Modify `docs/source/_ext/notebook_downloads.py` to copy notebook sources into the built HTML output under `_notebooks/` and inject a per-page “Download this notebook (.ipynb)” link using `html-page-context` plus a template override in `docs/source/_templates/`. Update the verifier to check only `.ipynb` copies and never mention PDFs; on RTD (`READTHEDOCS=True`) do not hard-fail for missing notebook downloads (warning-only), but keep strict mode for local builds. Fix autosummary targets so they reference importable objects (e.g. `VAEModel.get_latent_representation`). Ensure `python -m sphinx -T -b html ...` succeeds and built output contains `_notebooks/tutorials/*.ipynb` and no `.pdf` references.
