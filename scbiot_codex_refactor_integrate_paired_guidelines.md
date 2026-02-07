# Codex Refactor Guidelines: `integrate_paired.py` (scBIOT OT)

## Goal
Refactor `integrate_paired.py` so it:
1. Implements the **paired multiome joint embedding** via a available ot helper function or **clean OT barycentric projection** or a:
   - Transport **view** modality (e.g., ATAC LSI) into the **base** modality space (e.g., RNA PCA) using OT.
   - Fuse **base** embedding with **transported view→base** embedding into a single `X_joint`.
2. Exposes the **same core parameters and defaults pattern** as `integrate.py` (especially:
   `approximate_ot`, `centroid_ot`, and other shared OT knobs),
   and is called consistently from `scb.ot.integrate(..., preset="paired", ...)`.

Target call:

```python
adata, metrics = scb.ot.integrate(
    adata,
    preset="paired",
    obsm_key="X_pca",      # base view for geometry/smoothing
    view_key="X_lsi",      # paired view to transport into base space
    batch_key="batch",
    out_key="X_joint",
    approximate_ot=False,
    centroid_ot=False,
)
print(metrics)
```

---

## Design Requirementsa
First to use available functions in the project


### 1) Single public API, preset dispatch
- `scb.ot.integrate(...)` remains the user-facing entry point.
- `preset="paired"` must dispatch into the paired implementation.
- `integrate_paired.py` must **not** define a separate public API with different parameter names.
- `integrate_paired` should be an internal implementation called by `integrate.py`/preset router.

### 2) Parameter parity with `integrate.py`
`integrate_paired(...)` must accept and forward the same “basic” OT parameters used in `integrate.py`, at minimum:

**Required parity parameters**
- `obsm_key`, `batch_key`, `out_key`
- `approximate_ot: bool = False`
- `centroid_ot: bool = False`
- OT regularization + solver knobs used by your core OT helper(s) (keep names consistent with integrate.py):
  - `reg` / `epsilon`
  - `n_iter` / `sinkhorn_iter`
  - `K_ref`, `K_batch` (if used in your approximate/anchor path)
  - `seed`, `dtype`, `use_faiss` (if applicable)

**Paired-specific additions**
- `view_key: str` (required when preset="paired")
- `w_base: float = 0.5`, `w_view: float = 0.5` (fusion weights)
- `prior_strength: float` and `diag_mass: float` (pairing prior; see algorithm)

**Rule**
- Do not invent new names for knobs that already exist in `integrate.py`.
- If `integrate.py` uses `reg`, do not call it `epsilon` here.
- If `integrate.py` uses `approximate_ot`, do not introduce `use_approx`.

### 3) Shared helpers, no duplicated OT code
- Reuse your existing OT utilities from `utils/ot_helpers.py` (or equivalent).
- If `integrate.py` already has a robust cost construction, Sinkhorn, barycentric projection, and FAISS-based KNN OT (approximate path), reuse them.
- `integrate_paired.py` should be “thin”: validate inputs → build cost → call OT helper → build joint embedding → compute metrics.

### 4) Metrics contract
Return `metrics: dict` consistent with other presets:
- Always include keys:
  - `preset`: `"paired"`
  - `n_obs`, `n_components`
  - `base_key`, `view_key`, `out_key`
  - `approximate_ot`, `centroid_ot`
  - `fusion_weights`: `{ "w_base": ..., "w_view": ... }`
- If OT coupling is computed:
  - `ot_cost_mean`, `ot_cost_p50`, `ot_cost_p90`
  - `transport_entropy` (or solver’s equivalent)
  - `diag_mass_used`, `prior_strength`
- If batch_key exists:
  - optional neighborhood metrics used elsewhere (e.g., `knn_overlap`, `batch_entropy_per_cell_mean`)

---

## Algorithm Spec: Paired Joint Embedding (OT)

### Inputs
- `adata.obsm[obsm_key]` (e.g., RNA PCA), shape `(n, d1)`
- `adata.obsm[view_key]` (e.g., ATAC LSI), shape `(n, d2)`
- Must be **paired**: same `adata.n_obs`, same row order (same cells).

### Preprocessing
- Trim to `n_components = min(d1, d2, user_n_components)` (default: follow integrate.py convention).
- Standardize each modality **feature-wise** (z-score) before distances so one modality cannot dominate:
  - `X_base_z = zscore(X_base)`
  - `X_view_z = zscore(X_view)`

### Cost
Construct cost matrix `C(i,j)` between view cell `i` and base cell `j`:

- Default: squared Euclidean in standardized space
  - `C = ||X_view_z[i] - X_base_z[j]||^2`

### Pairing prior (soft diagonal bias)
Because data are paired, add a diagonal preference without forcing identity mapping:

- Build a smooth prior `P0`:
  - `P0 = diag_mass * diag(1/n) + (1 - diag_mass) * uniform(1/n^2)`
- Modify cost:
  - `C' = C - prior_strength * log(P0 + 1e-12)`

This encourages mass near the diagonal but still allows small “nudges” that correct global shift/rotation/scale.

### OT solver
Compute entropic OT coupling `T` with uniform marginals (or your project’s standard marginal scheme).

### Barycentric projection (view → base)
Map each view cell into base space:

- `X_view_to_base[i] = (Σ_j T[i,j] * X_base_z[j]) / (Σ_j T[i,j])`

### Fusion
Build joint embedding in base-like coordinates:

- `X_joint = w_base * X_base_z + w_view * X_view_to_base`

Write to:
- `adata.obsm[out_key] = X_joint.astype(float32)`

---

## Scaling: `approximate_ot` and `centroid_ot`

### `centroid_ot=True` (recommended for large N)
Do OT on **group centroids** (metacells) then broadcast back to cells.

**Contract**
- `group_key` source: reuse integrate.py’s existing centroid grouping logic:
  - if integrate.py uses Leiden/metacell key, accept that key here too (do not invent a new grouping system).
- Compute centroid matrices:
  - `C_base[g] = mean(X_base_z[cells in g])`
  - `C_view[g] = mean(X_view_z[cells in g])`
- Run OT on `k×k` centroids, produce `C_view_to_base[g]`.
- For each cell, assign `X_view_to_base[cell] = C_view_to_base[group_of_cell]`.
- Fuse into `X_joint`.

### `approximate_ot=True`
Use the same approximate/anchor OT method you already ship (FAISS/KNN-capped OT):
- Instead of full `n×n` cost, compute KNN candidates per view cell against base cells using FAISS (or your existing kNN helper).
- Build sparse candidate set, run capped Sinkhorn/OT on that set.
- Apply barycentric projection on the sparse plan.

**Rule**
- Follow integrate.py’s defaults for:
  - `K_ref`, `K_batch` (or equivalent)
  - capping/striding
  - FAISS GPU/CPU toggles

---

## Refactor Plan (Codex Tasks)

### Task 0 — Inventory and align signatures
1. Open `integrate.py` and list its parameters and defaults.
2. Open `integrate_paired.py` and list its parameters and defaults.
3. Modify `integrate_paired.integrate_paired(...)` signature to:
   - include parity parameters (`approximate_ot`, `centroid_ot`, OT solver knobs)
   - include `view_key`
   - keep name alignment with integrate.py

### Task 1 — Centralize preset routing
- In preset registry (e.g., `_presets.py` / `get_modality_preset`), add `"paired"` preset.
- In `integrate.py`, route:
  - `if preset == "paired": return integrate_paired(...)`

### Task 2 — Implement the paired OT barycentric projection
- Implement a small internal function in `integrate_paired.py`:

**Suggested internal structure**
- `_validate_paired_inputs(adata, obsm_key, view_key, ...)`
- `_standardize_embeddings(X_base, X_view, n_components, ...)`
- `_paired_cost_with_prior(X_view_z, X_base_z, diag_mass, prior_strength, ...)`
- `_solve_ot(C, approximate_ot, centroid_ot, ...)` (delegates to shared OT helpers)
- `_barycentric_project(T, X_base_z)`
- `_fuse(X_base_z, X_view_to_base, w_base, w_view)`

Keep every “heavy” operation in shared OT helpers if it exists.

### Task 3 — Metrics
Return a dict with the required contract above.
- Ensure metrics keys match integrate.py style and naming.

### Task 4 — Tests (must add)
Add tests that pass on CPU with small synthetic data.

**Minimum tests**
1. **Shape + key presence**
   - `adata.obsm[out_key].shape == (n, n_components)`
2. **Identity sanity (strong diagonal prior)**
   - With `diag_mass=0.95`, `prior_strength` high, `w_view=1.0`, `w_base=0.0`:
     - transported embedding approximates `X_base_z` (low MSE)
3. **Small “nudge” behavior**
   - Construct `X_view = X_base + small_shift`
   - Confirm OT transport reduces the shift in the fused joint embedding.
4. **`centroid_ot=True` path**
   - Create fake groups, run, confirm output shape and deterministic mapping per group.
5. **`approximate_ot=True` path** (if supported in CI)
   - Use small `K_ref`, confirm it runs and returns deterministic output.

### Task 5 — Documentation and typing
- Add docstring for preset “paired” in `integrate` docs:
  - explain `view_key`, barycentric projection, fusion weights
- Ensure full type annotations like integrate.py (no `Any` leakage for public API).
- Add to API docs / docstrings so IDE signature shows full parameters.

---

## Input Validation Rules (must enforce)
- `obsm_key` and `view_key` must exist in `.obsm`
- Both must have `n_obs` rows equal to `adata.n_obs`
- `view_key` must not equal `out_key`
- If `batch_key` provided, it must exist in `.obs`
- If `centroid_ot=True`, grouping key must exist (reuse integrate.py behavior)

Raise `ValueError` with precise messages matching project style.

---

## Performance Rules
- Full OT must never allocate `n×n` cost when `n` is “large” according to integrate.py thresholds.
  - If integrate.py has a cutoff, reuse it.
  - Otherwise: if `n > 50_000`, require `centroid_ot` or `approximate_ot` (same policy as integrate.py).
- Always store embeddings as `float32`.

---

## Acceptance Checklist (Codex must satisfy)
- [ ] `preset="paired"` works via `scb.ot.integrate(...)`
- [ ] `integrate_paired.py` uses the same base parameters as `integrate.py`
- [ ] `approximate_ot` and `centroid_ot` are supported and wired
- [ ] Outputs `adata.obsm[out_key]` joint embedding from OT transport + fusion
- [ ] Returns `metrics` with the required keys
- [ ] Unit tests cover full, approximate, and centroid paths
- [ ] Docstrings + typing match project conventions
