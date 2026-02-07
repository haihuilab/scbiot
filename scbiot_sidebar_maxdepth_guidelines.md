# Codex Guidelines: Show Secondary Tutorial Pages in the Left Sidebar (Sphinx `maxdepth`)

Target page (example): `tutorials/1_scrna_seq` (built HTML like `tutorials/1_scrna_seq.html`).

## Goal

Update the Sphinx docs so the **left sidebar immediately shows second-level tutorial pages** (children of a tutorial page) **without requiring a click on the parent tutorial title**.

In practice, under **Tutorials → 1. scRNA-seq**, users should see `a.` / `b.` pages listed in the sidebar by default.

---

## What’s causing the issue

This behavior is usually due to **one (or both)** of the following:

1. **The parent `toctree` is too shallow**  
   The landing page `.. toctree::` has `:maxdepth: 1` (or missing), so Sphinx does not include the second level in the navigation tree.

2. **The theme collapses navigation or renders only 1 level by default**  
   Even if Sphinx provides a deeper tree, the theme can hide deeper levels unless theme options are set.

---

## Codex tasks (do in order)

### Task 0 — Determine theme and entrypoints

1. Locate `conf.py` (usually `docs/source/conf.py`).
2. Identify:
   - `html_theme = ...`
   - `html_theme_options = {...}` (if present)
3. Locate the Tutorials landing page file:
   - Common patterns: `docs/source/tutorials.rst`, `docs/source/tutorials/index.rst`, or a `tutorials/` folder with an `index.rst`
4. Locate the main docs entrypoint:
   - Common patterns: `docs/source/index.rst`

**Constraint:** Do not restructure docs; only adjust `toctree` and theme options.

---

### Task 1 — Increase `toctree` depth on the Tutorials landing page

Edit the Tutorials landing page file and ensure the main tutorials `toctree` includes at least two levels.

**Required change:** set `:maxdepth:` to **2** (or higher, but 2 is the minimum for this request).

Example:

```rst
.. toctree::
   :caption: Tutorials
   :maxdepth: 2

   tutorials/1_scrna_seq
   tutorials/2_scrna_seq_in_r
   tutorials/3_scatac_seq
   tutorials/5_unpaired_multiomics
   tutorials/6_centroid_ot
   tutorials/7_brain_1_3m
```

**Acceptance check:** When building HTML, the generated navigation tree includes children of `tutorials/1_scrna_seq` (i.e., the `a.` / `b.` pages).

---

### Task 2 — Ensure each parent tutorial lists its children via a nested `toctree`

Open `docs/source/tutorials/1_scrna_seq.rst` (or similarly named source). Confirm it contains a `toctree` listing its child pages.

If missing, add:

```rst
.. toctree::
   :maxdepth: 1

   1_scrna_seq_a_lung_atlas
   1_scrna_seq_b_supbiot
```

**Notes:**
- Use **relative paths consistent with your repo**.
- Child pages must be included by filename (without `.rst`).
- Keep headings and content unchanged.

---

### Task 3 — Update theme options so the sidebar shows 2 levels by default

Modify `html_theme_options` in `conf.py` based on the theme detected in **Task 0**.

#### If `html_theme = "pydata_sphinx_theme"`

Add or update:

```python
html_theme_options = {
    # keep existing options
    "show_nav_level": 2,
}
```

#### If `html_theme = "sphinx_rtd_theme"`

Add or update:

```python
html_theme_options = {
    # keep existing options
    "collapse_navigation": False,
    "navigation_depth": 4,
}
```

#### If theme is neither of the above

1. Keep `toctree :maxdepth:` changes (Tasks 1–2).
2. Search the theme docs / config keys already present in `conf.py` for:
   - “navigation depth”
   - “collapse”
   - “show nav”
3. If the theme supports a “nav depth / collapse” option, set it so **two levels are visible by default**.

**Constraint:** Do not switch themes. Only adjust existing configuration.

---

## Verification checklist (must pass)

1. Build docs locally:
   - `python -m sphinx -b html docs/source docs/_build/html`
2. Open:
   - `docs/_build/html/tutorials/1_scrna_seq.html`
3. Confirm:
   - Left sidebar shows **Tutorials → 1. scRNA-seq → its child pages** without clicking the parent.
4. Confirm no Sphinx warnings:
   - No “document isn't included in any toctree”
   - No broken references

---

## Done criteria

- Sidebar shows second-level pages by default on tutorial pages.
- Changes limited to:
  - `toctree :maxdepth:` values and nested `toctree` blocks
  - `conf.py` theme options
- Docs build succeeds without warnings.
