# scbiot — Codex Pre‑Ship Audit Prompt (Concise)

Paste this into Codex to run a full‑repo audit of `scbiot` before shipping.

---

## 0) Use

1. Open Codex in the repo workspace.  
2. Paste Sections 1–3.  
3. (Optional) Paste Section 4.  
4. Ask for the report in the specified format.

---

## 1) Master prompt (paste into Codex)

> You are the release‑review maintainer for **scbiot**, a production Python library for optimal‑transport single‑cell integration built around **AnnData/Scanpy** with optional **PyTorch GPU** backends.  
>  
> **Task:** Perform a full pre‑ship audit of this repository and produce a release readiness report with actionable, minimal fixes.  
>  
> **Priority order:**  
> 1) Correctness & invariants (AnnData shapes/keys, view safety, sparse safety)  
> 2) Public API stability (exports, signatures, defaults, compatibility)  
> 3) Numerical stability (NaN/Inf guards, validation, seeding)  
> 4) Performance & memory (avoid O(N²), densification, huge copies, CPU↔GPU churn)  
> 5) Maintainability (clarity, types, errors)  
> 6) Tests & docs (NumPy docstrings, examples, Sphinx/RTD)  
> 7) Packaging/release hygiene (pyproject, versioning, extras, import‑time behavior)  
>  
> **Required approach:**  
> - Build a concise **repo map** (modules, public entry points, core algorithms).  
> - Run automated checks and report results.  
> - Review module‑by‑module with public API and core algorithms first.  
> - For each issue: **file path**, **symbol**, **severity**, **why**, **minimal patch**.  
> - Avoid large refactors unless correctness or blockers require them.  
>  
> **scbiot invariants:**  
> - Writes to `adata.obsm[...]` must be `(adata.n_obs, d)` aligned to `adata.obs_names`.  
> - Never silently densify sparse matrices; avoid `.A`/`.toarray()` unless explicit and documented.  
> - AnnData views: no unsafe in‑place writes; copy or use safe assignments.  
> - Determinism: `random_state` controls numpy + torch; document GPU nondeterminism.  
> - Optional deps remain optional (no import‑time failure).  
>  
> **Output format (must follow):**  
> ### Repo map  
> ### Automated checks summary (commands + outcomes)  
> ### Release blockers (must‑fix)  
> ### Major issues (should‑fix)  
> ### Minor issues (nice‑to‑have)  
> ### API surface audit (public functions + stability notes)  
> ### Perf/memory audit (hotspots + quick wins)  
> ### Test gaps (exact tests to add)  
> ### Docs gaps (exact pages/docstrings/examples)  
> ### Packaging/release checklist (pass/fail)  
> ### Final verdict: READY / NOT READY (and why)  
>  
> Use severities: **[BLOCKER] [MAJOR] [MINOR] [NIT]**. Prefer minimal diffs with patch snippets.

---

## 2) Automated gates (paste into Codex)

> Run these checks in order and record outcomes:  
> 1) `python -c "import scbiot; print(scbiot.__version__ if hasattr(scbiot,'__version__') else 'no version')"`  
> 2) `ruff check .` (or repo‑configured linter)  
> 3) `ruff format --check .` (or `black --check .`)  
> 4) `pytest -q`  
> 5) `python -m compileall -q .`  
> 6) Type check: `pyright` or `mypy` (whichever repo uses)  
> 7) Docs build: `sphinx-build -b html docs docs/_build/html` (or repo doc command)  
> 8) Packaging: `python -m build` and `twine check dist/*`  
>  
> Any failing gate is **[BLOCKER]**; propose the smallest fix.

---

## 3) Scope control (paste into Codex)

> Scope control:  
> - No architecture rewrites; fix blockers first.  
> - Algorithm changes require a regression test + brief doc note.  
> - Public API changes require back‑compat or deprecation.  
> - Prefer minimal diffs and localized edits.

---

## 4) Optional: review order

> 1) Public entry points: `scbiot/__init__.py`, exported APIs, `integrate()` wrappers  
> 2) Core OT solvers/backends (torch / pot)  
> 3) Pre‑alignment (e.g., CORAL) and numeric helpers  
> 4) Label/supervision paths (supbiot, label transfer metadata)  
> 5) Utilities (neighbors/metrics/subsampling)  
> 6) Docs + examples + CLI (if any)

---

## 5) Acceptance criteria (must include)

Include all of the following in the final report:

### 5.1 Public API inventory
- List every exported public symbol (`__all__`, package `__init__`, docs).  
- For each public function: signature stability, parameter naming consistency (`obsm_key`, `batch_key`, `out_key`, `label_key`, `random_state`, `verbose`, `use_gpu`), return types, and AnnData mutations (`obs`, `obsm`, `uns`).

### 5.2 AnnData mutation map
- Enumerate all writes to `adata.obs`, `adata.obsm`, `adata.uns`, and `adata.layers` (if any).  
- For each write: confirm shape alignment, view safety, and sparse safety.

### 5.3 Perf/memory hotspots
- Identify: O(N²) pairwise distances, repeated neighbor graphs, large copies/concats, CPU↔GPU transfers, sparse densification.  
- For each hotspot: quick win (minimal change) and “bigger refactor” only if required.

### 5.4 Numerical stability & determinism
- Identify NaN/Inf sources with minimal guards (eps, clamp, safe norms/logs).  
- Confirm `random_state` handles numpy + torch.  
- Note GPU nondeterminism and mitigation docs.

### 5.5 Test gaps (exact tests)
- For each major subsystem: file name + test name, assertion, minimal synthetic data strategy.

### 5.6 Docs gaps
- Missing/incorrect docstrings (NumPy style), param docs vs defaults, missing examples, Sphinx/RTD warnings.

### 5.7 Packaging/release checklist (pass/fail)
- `pyproject.toml` metadata, `__version__` single‑source, extras/optional deps, import‑time behavior, build + twine.

---

## 6) Optional: issue tracker table

If desired, request this table in addition to the narrative report:

| severity | file | symbol | issue | why it matters | minimal fix | test to add | doc to update |
|---|---|---|---|---|---|---|---|

---

## 7) Final ship‑gate rubric

### READY
- All automated gates pass.  
- No BLOCKER issues remain.  
- Major issues are mitigated or explicitly deferred with rationale.

### NOT READY
- Any failing gate (tests/docs/build/type), or  
- Any BLOCKER correctness/API/perf/memory issue.

---

## 8) One‑liner (optional)
If you want a single concise instruction:

> Do a full pre‑ship audit of this entire repo for scbiot with a release readiness report: repo map, automated checks, blockers/majors/minors, API inventory, AnnData mutation map, perf/memory hotspots, numerical stability/determinism, test gaps, docs gaps, packaging checklist, and final READY/NOT READY verdict. Use minimal patch suggestions and exact file+symbol references.
