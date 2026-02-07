# scBIOT Docs Search Fix — Codex Guidelines (Sphinx + Read the Docs)

> **Problem**: The documentation search box loads but returns no results (or cannot search anything) on the hosted site.  
> **Goal**: Update the docs build + templates so **both**:
> 1) **Sphinx static search** (client-side `searchindex.js`) is generated and works, and  
> 2) **Read the Docs (RTD) server-side search** can correctly index tutorial content.

---

## What Codex should do (high-level)

1. **Identify the docs toolchain**
   - Confirm Sphinx is used (check `docs/source/conf.py`).
   - Identify theme and template overrides (`docs/source/_templates/`).
   - Confirm RTD build configuration exists (`.readthedocs.yaml`).

2. **Make sure Sphinx builds search assets**
   - Ensure the build produces:
     - `docs/build/html/searchindex.js`
     - `docs/build/html/_static/searchtools.js`
     - `docs/build/html/_static/language_data.js`
   - Ensure the search UI loads these scripts and is not broken by custom templates.

3. **Make sure RTD can index tutorial content**
   - Ensure each built HTML page has **exactly one** main content node:
     - `<main>` **or** an element with `role="main"`
   - Ensure tutorial content is inside that main node.
   - Avoid multiple `<main>` elements per page (RTD indexing can fail).

4. **Add a “Search Sanity Check” test**
   - Build docs in CI.
   - Validate `searchindex.js` contains at least N known tutorial keywords.
   - Fail build if the search index is empty or missing.

5. **Pin stable versions for docs build**
   - Pin `sphinx`, theme, and search-related extensions in the RTD environment.
   - Prevent “works locally but breaks on RTD” due to drifting versions.

---

## Step 1 — Inventory & diagnostics

### 1.1 Confirm Sphinx is building HTML
Codex must locate:
- `docs/source/conf.py`
- `docs/source/index.rst` (or `docs/source/index.md`)
- `docs/requirements.txt` (or docs extras in `pyproject.toml`)
- `.readthedocs.yaml`

### 1.2 Check whether templates override the layout/search
Codex must check for:
- `docs/source/_templates/layout.html`
- `docs/source/_templates/search.html`
- `docs/source/_templates/**`

If present, these overrides are a top cause of “search box works visually but returns nothing”.

### 1.3 Capture RTD URL for reference
Use the user-provided docs homepage:

```text
https://scbiot.readthedocs.io/en/stable/index.html
```

---

## Step 2 — Ensure Sphinx static search assets exist and work

### 2.1 Build docs locally and assert search assets exist
Codex must add a reproducible command path, for example:

```bash
python -m pip install -r docs/requirements.txt
sphinx-build -b html docs/source docs/build/html
```

Then assert these files exist:

- `docs/build/html/searchindex.js`
- `docs/build/html/_static/searchtools.js`
- `docs/build/html/_static/language_data.js`

If any are missing:
- Check `exclude_patterns` in `docs/source/conf.py` for accidental patterns like:
  - `search*`, `search.html`, `_static/*search*`, `_build`, etc.
- Check custom template overrides for missing `<script>` includes.

### 2.2 Test locally **over HTTP** (not `file://`)
Codex must document this for dev testing:

```bash
cd docs/build/html
python -m http.server 8000
# open http://localhost:8000 and test search
```

Browsers can block JS asset loads under `file://`, making search appear broken even when it isn’t.

---

## Step 3 — Ensure RTD indexing works (critical for hosted search)

RTD extracts the **main content node** from your built HTML.  
Codex must ensure:

- Exactly one `<main>` **or** one node with `role="main"` per page  
- Tutorial content is inside that node  
- No nested or duplicated main nodes

### 3.1 How Codex should verify this
Codex should:
- Build docs (`sphinx-build -b html ...`)
- Inspect a built tutorial page HTML and confirm a single main node exists.

Example check pattern (pseudo):

- Count occurrences of `<main` in HTML
- If > 1, fix template to produce only one main

### 3.2 If you override templates, fix them
If `docs/source/_templates/layout.html` exists, Codex must ensure the template produces:

- One main content wrapper:
  - `<main>` … `{% block body %}` … `</main>`
- Avoid wrapping content in multiple “main” containers.

---

## Step 4 — Add a Search Sanity Check test (CI / pre-release)

### 4.1 Add a script: `scripts/docs_search_sanity_check.py`
Codex must implement a script that:
1. Builds docs
2. Reads `searchindex.js`
3. Confirms it contains known tutorial keywords

**Example keywords** (replace with your real tutorial terms):
- `integrate_ot`
- `integrate`
- `AnnData`
- `X_pca`
- `batch_key`
- `preset`

### 4.2 Example script (drop-in)
Create `scripts/docs_search_sanity_check.py`:

```python
from __future__ import annotations
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SOURCE = DOCS / "source"
OUT = DOCS / "build" / "html"
SEARCHINDEX = OUT / "searchindex.js"

KEYWORDS = [
    "integrate_ot",
    "integrate(",
    "AnnData",
    "X_pca",
    "batch_key",
    "preset",
]

def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))

def main() -> int:
    # Build docs
    run(["sphinx-build", "-b", "html", str(SOURCE), str(OUT)])

    if not SEARCHINDEX.exists():
        print(f"ERROR: missing {SEARCHINDEX}")
        return 2

    text = SEARCHINDEX.read_text(encoding="utf-8", errors="ignore")
    hits = [k for k in KEYWORDS if k in text]

    if not hits:
        print("ERROR: searchindex.js exists but contains none of the expected keywords.")
        print("Likely empty/failed indexing, template override issue, or excluded content.")
        return 3

    print("OK: search sanity check passed. Found:", ", ".join(hits))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

### 4.3 Add a CI entry (example)
Codex should add a CI job step like:

```bash
python -m pip install -r docs/requirements.txt
python scripts/docs_search_sanity_check.py
```

This prevents shipping docs with “search UI but empty index”.

---

## Step 5 — Pin docs build versions for stability (RTD + local)

Codex must ensure RTD uses the same pinned dependency versions as local build.

### 5.1 Preferred: `docs/requirements.txt`
Add pins such as:

```text
sphinx==7.4.7
# theme (example)
pydata-sphinx-theme==0.15.4
# any extensions you use
myst-parser==2.0.0
sphinx-design==0.6.1
```

Adjust versions to the ones already known-good in the repo.

### 5.2 Ensure `.readthedocs.yaml` installs docs requirements
Codex must ensure RTD config installs `docs/requirements.txt`, e.g.:

```yaml
python:
  install:
    - requirements: docs/requirements.txt
```

---

## Step 6 — Make tutorials more searchable (content guidelines)

Even with correct plumbing, search can feel “broken” if tutorials don’t contain searchable text.

Codex must ensure tutorial pages include:
- Clear **H1** and **H2** headings
- Plain-text explanations near code blocks
- Consistent terminology (don’t change function names in prose)

Recommended tutorial structure:

- Title
- “What you will learn”
- Minimal runnable example
- Section headings for each step (preprocess / integrate / evaluate)
- “Troubleshooting” section containing common error messages (these become searchable)

---

## Step 7 — Codex prompt to execute the full patch

Copy/paste this into Codex:

```text
You are a senior documentation engineer working inside this repo.

Goal: fix the hosted documentation search so the search box returns results for tutorial keywords.

Constraints:
- Minimal diffs.
- Do not rewrite the whole docs system; only change what is required.
- Ensure both Sphinx static search and RTD indexing work.

Tasks:
1) Identify docs toolchain:
   - locate docs/source/conf.py, theme, docs requirements, and .readthedocs.yaml.
   - list any template overrides in docs/source/_templates.

2) Ensure Sphinx search is built:
   - run sphinx-build -b html docs/source docs/build/html
   - verify these files exist:
     docs/build/html/searchindex.js
     docs/build/html/_static/searchtools.js
     docs/build/html/_static/language_data.js
   - if missing or empty, fix exclude_patterns, search template, or theme config.

3) Ensure RTD indexing works:
   - inspect built HTML for a tutorial page and confirm exactly one <main> (or role="main") exists.
   - if multiple, patch the template override to enforce a single main content container.

4) Add search sanity test:
   - add scripts/docs_search_sanity_check.py that builds docs and checks searchindex.js contains at least 5 known tutorial keywords.

5) Pin docs deps:
   - pin sphinx + theme + key extensions in docs/requirements.txt
   - ensure .readthedocs.yaml installs docs/requirements.txt

Deliverables:
- file-by-file diffs
- commands to verify locally (serve docs over HTTP) and confirm search works
- short patch summary explaining why search was broken and how it’s fixed
```

---

## Local verification checklist (post-fix)

1. Build docs:
```bash
sphinx-build -b html docs/source docs/build/html
```

2. Confirm search assets exist:
- `docs/build/html/searchindex.js`
- `docs/build/html/_static/searchtools.js`
- `docs/build/html/_static/language_data.js`

3. Serve docs:
```bash
cd docs/build/html
python -m http.server 8000
```

4. Verify search returns tutorial results for keywords:
- `integrate_ot`
- `X_pca`
- `batch_key`
- `preset`

5. Hosted RTD:
- Search should return at least one tutorial page for the same keywords.

---

## Notes on common root causes (so Codex targets the right fix)

- **Template overrides** that omit Sphinx search scripts or break main content markup.
- `exclude_patterns` that inadvertently drops `search` pages or tutorial sources.
- Docs tested via `file://` leading to false “search is broken”.
- Multiple `<main>` nodes causing RTD server-side indexing failures.

---

**End of document**
