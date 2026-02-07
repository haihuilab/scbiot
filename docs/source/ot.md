# Optimal transport: `ot`

OT utilities for aligning batches and modalities. The public API entry point is
`integrate`, which matches the workflows shown in the tutorials.

For unpaired RNA/ATAC workflows, build a shared PCA embedding (see the
tutorials) and then run:

```python
adata, metrics = scb.ot.integrate(
    adata,
    preset="anchor",
    obsm_key="X_pca_shared",
    batch_key="modality",
    reference_category="reference",
)
```

## Scaling options

For ultra-large datasets, use centroid-level OT:

```python
adata, metrics = scb.ot.integrate(
    adata,
    preset="centroid",
    obsm_key="X_pca",
    batch_key="batch",
    out_key="scBIOT",
)
```

If you want centroid OT while keeping another preset's OT hyperparameters, enable the flag:

```python
adata, metrics = scb.ot.integrate(
    adata,
    preset="anchor",
    obsm_key="X_pca",
    batch_key="batch",
    out_key="X_ot",
    centroid_ot=True,
)
```

For a faster approximate OT run on large datasets, enable the approximate OT solver
while keeping your preset's data keys:

```python
adata, metrics = scb.ot.integrate(
    adata,
    preset="atac",
    obsm_key="X_lsi",
    batch_key="batchname_all",
    out_key="X_ot",
    approximate_ot=True,
)
```

## OT backend controls

`scbiot.ot.integrate` exposes the ``use_gpu``/``gpu_device`` and ``ot_backend``
knobs. It also supports ``ot_mode`` to select unbalanced OT (``"unbalanced"``,
the rare-aware behavior) or balanced OT (``"balanced"``) for stronger batch
mixing. When you request balanced OT while keeping ``reference="largest"``,
scBIOT automatically switches to the union reference so that every batch can
move symmetrically.
