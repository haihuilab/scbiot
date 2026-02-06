# Codex Refactor Guidelines: add `approximate_ot` + `centroid_ot` + `centroid` preset

You are working **inside the scBIOT repo**. Goal: extend the **existing public API** `scb.ot.integrate(...)` with two new flags and a new preset, while keeping backward compatibility and minimizing code churn.

---

## 0) Target behavior (spec)

### Old API (must keep working)
```python
adata, metrics = scb.ot.integrate(
    adata,
    preset="anchor",
    obsm_key="X_shared_pca",
    batch_key="modality",
    reference_category="reference",
    out_key="X_supbiot",
    label_key="cell_type",
    unlabeled_category="Unknown",
)
```

### New API (must be supported)
```python
adata, metrics = scb.ot.integrate(
    adata,
    preset="anchor",
    obsm_key="X_shared_pca",
    batch_key="modality",
    reference_category="reference",
    out_key="X_supbiot",
    label_key="cell_type",
    unlabeled_category="Unknown",
    approximate_ot=True,
    centroid_ot=False,
)
```

### New parameters
- `approximate_ot: bool = False`
- `centroid_ot: bool = False`

### Method selection rules (authoritative)
1. **Mutual exclusivity**
   - If `approximate_ot and centroid_ot` are both `True`: raise `ValueError` with clear message.

2. **Centroid OT path**
   - If `centroid_ot is True`: call **centroid OT** implementation from `integrate_centroids.py` (the helper currently named `integrate_centroids`).
   - `preset` still influences hyperparameters, but centroid defaults must be taken from the centroid preset (see below).

3. **Approximate OT path**
   - If `approximate_ot is True`: use the same configuration/behavior as **preset `"anchor"`** mode.
   - This means: reuse the exact anchor-preset code path and defaults (do not create a new solver path).

4. **Default path**
   - If both flags are False: preserve current behavior of `preset=...`.

### New preset
- Add preset `"centroid"`:
  - It reflects the **default settings in `integrate_centroids.py`**:
    - `n_centroids_per_batch`
    - `max_samples_per_batch`
    - `k_interp`
    - `chunk_size`
    - `use_gpu`, `gpu_device`
    - `tmp_path` (optional)
  - When user sets `preset="centroid"`, it should behave as if:
    - `centroid_ot=True` (even if user doesn’t pass it explicitly)
    - and default centroid settings are applied unless overridden by explicit keyword args

---

## 1) Codex operating principles

### Minimal-change strategy
- Do not redesign architecture.
- Add routing logic in one central place (the public `integrate()` function).
- Reuse existing preset and argument plumbing.
- Keep `integrate_ot(...)` and `integrate_centroids(...)` as “sources of truth” for their method.

### Backward compatibility
- Any existing call that did not pass the two new flags must behave identically.
- Existing presets (`rna`, `atac`, `supervised`, `anchor`, etc.) must remain valid.

### Type safety + docs
- Add type annotations for new params.
- Update docstrings and README/API docs wherever `integrate()` is documented.
- Add at least one minimal unit/integration test for each new branch.

---

## 2) Exact implementation steps (what to change)

### Step A — Locate the public integration entrypoint
- Find the public function:
  - `scb.ot.integrate(...)` (or equivalent module exporting `integrate`)
- Identify how `preset` is currently parsed and how kwargs flow into `integrate_ot(...)`.

**Deliverable:** list of files and functions involved (entrypoint + presets + wrappers).

---

### Step B — Extend signature
Add these kwargs to the public function signature:
- `approximate_ot: bool = False`
- `centroid_ot: bool = False`

Ensure they are documented in the docstring and are included in generated API docs.

---

### Step C — Add preset `"centroid"` (single source of defaults)
1. In preset registry (`._presets.py` or similar), add:
   - a `centroid` preset object/dict that includes centroid defaults that currently live in `integrate_centroids.py`.
2. Implement `get_modality_preset("centroid")` (or equivalent) returning those defaults.

**Rule:** centroid preset defaults must match `integrate_centroids.py` defaults exactly.

---

### Step D — Route calls in `integrate()` (central dispatcher)

Add a small dispatcher at the top of `integrate()` after presets are resolved:

1. Compute **effective mode**:
   - `effective_centroid = centroid_ot or (preset == "centroid")`
   - `effective_approx = approximate_ot`

2. Validate exclusivity:
   - if `effective_centroid and effective_approx`: raise `ValueError`

3. Route:

#### D1) Centroid route
- If `effective_centroid`:
  - Ensure centroid defaults are loaded:
    - If `preset == "centroid"`, load centroid preset defaults
    - If `preset != "centroid"` and `centroid_ot=True`, still allow user-chosen `preset` for OT hyperparams (reg/reg_m/sharpen/etc.), but centroid-specific args must come from centroid preset defaults unless explicitly overridden.
  - Call:
    - `integrate_centroids(adata, obsm_key=..., batch_key=..., out_key=..., modality=<modality/preset?>, **kwargs)`
  - Return its `(adata, metrics)`.

**Important:** keep a clean boundary:
- centroid-specific args are consumed by `integrate_centroids`
- OT args are forwarded via `**integrate_kwargs` to `integrate_ot` inside centroid code

#### D2) Approximate route
- If `effective_approx`:
  - Force the same behavior as `"anchor"` preset:
    - Preferred minimal approach: set `preset = "anchor"` internally for this call only (do not change user-visible `preset` string outside this call).
  - Call the existing `integrate_ot` path (not centroid).

#### D3) Default route
- Otherwise: existing behavior unchanged.

---

### Step E — Update `integrate_centroids.py` integration surface
- Ensure `integrate_centroids(...)` accepts the full set of relevant kwargs and:
  - consumes centroid-specific kwargs directly
  - forwards remaining kwargs to `integrate_ot(...)` (already present in your code via `**integrate_kwargs`)

**Add/confirm:**
- `modality` argument behavior:
  - allow `"centroid"` preset to populate centroid params
  - still allow `"rna"|"atac"|"supervised"` presets to initialize OT hyperparams, but centroid params remain centroid defaults

---

## 3) Parameter mapping (authoritative)

### New flags interpretation
- `approximate_ot=True`:
  - Equivalent to running with `"anchor"` preset behavior for OT (same defaults and same code path)
- `centroid_ot=True`:
  - Equivalent to using `integrate_centroids(...)` (centroid OT + FAISS interpolation)

### Preset `"centroid"`
- Should imply `centroid_ot=True` automatically.
- `approximate_ot` is ignored unless explicitly set True; if both end up True -> error.

---

## 4) Tests (must add)

Add minimal tests that verify:
1. **Backward compatibility**
   - Old call without new flags yields identical output keys and metrics schema as before.

2. **Centroid preset activates centroid path**
   - `preset="centroid"` results in calling centroid implementation and adds `metrics["n_centroids"]`.

3. **Explicit centroid_ot activates centroid path**
   - `preset="anchor", centroid_ot=True` calls centroid implementation.

4. **Approximate flag activates anchor behavior**
   - `approximate_ot=True` forces anchor behavior (verify by checking the OT config or a sentinel in metrics).

5. **Mutual exclusivity**
   - both flags True raises `ValueError`.

---

## 5) Documentation updates (must do)

Update:
- function docstring for `scb.ot.integrate`
- README / API page examples:
  - include new flags
  - add an example for `preset="centroid"`
- Mention scaling guidance:
  - centroid path for ultra-large datasets
  - approximate anchor path for faster approximate OT on large datasets

---

## 6) Codex “Do-Not” list (avoid regressions)
- Do not rename public functions unless explicitly requested.
- Do not change existing default values for non-centroid presets.
- Do not move large blocks of code across files.
- Do not introduce new dependencies beyond what centroid method already requires.
- Do not break import paths (`__all__`, module exports).

---

## 7) Acceptance checklist (PR-ready)
- [ ] New params added and typed: `approximate_ot`, `centroid_ot`
- [ ] New preset `"centroid"` added and matches centroid defaults
- [ ] Dispatcher routes correctly and raises on invalid combo
- [ ] All old calls still work unchanged
- [ ] Tests cover all new branches
- [ ] Docs updated with new usage examples
- [ ] Lint/type check passes

---

# Codex Task Prompt (copy/paste)

You are a senior Python library engineer working inside the scBIOT repo.

Task: Refactor `scb.ot.integrate(...)` to add two new boolean parameters:
- `approximate_ot: bool = False`
- `centroid_ot: bool = False`

Rules:
- If both flags True -> raise ValueError.
- If `centroid_ot=True` OR `preset="centroid"`, route to centroid implementation in `integrate_centroids.py` (call `integrate_centroids`).
- Add a new preset `"centroid"` whose defaults match the defaults currently defined in `integrate_centroids.py` (n_centroids_per_batch, max_samples_per_batch, k_interp, chunk_size, use_gpu, gpu_device, tmp_path).
- If `approximate_ot=True`, force the same behavior as the `"anchor"` preset path (reuse existing anchor preset logic, do not create a new solver).
- Keep backward compatibility: old calls without the new flags behave identically.

Implementation requirements:
- Minimal code changes; add routing logic only in the public integrate entrypoint.
- Update docstrings and examples.
- Add minimal unit tests covering: centroid preset, centroid_ot flag, approximate_ot flag, mutual exclusivity, and backward compatibility.

Output:
- Provide a patch touching only the necessary files.
- Provide a short summary of changed files and why.
