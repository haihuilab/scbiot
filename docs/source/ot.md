# Optimal transport: `ot`

OT utilities for aligning batches and modalities. The functions below match what
you see in the tutorials; refer to the notebooks for full, runnable examples.

- `integrate`: batch correction for single-modality data (RNA or ATAC).
- `integrate_ot`: OT-only integration when you want to supply your own embedding.
- `pp.coembed_pca`: build a shared PCA embedding for unpaired modalities.
- `harmonize_gene_names`: ensure gene naming across RNA/ATAC inputs matches.

For unpaired RNA/ATAC workflows, compute a shared PCA with `pp.coembed_pca` and then run
`ot.integrate(preset="anchor", obsm_key="X_pca_shared", batch_key="modality",
reference_category="reference")` to align query cells to the reference.

For paired RNA/ATAC workflows, use the `paired` preset so OT sees each cell's matched views directly. Call
.. code-block:: python

    adata, metrics = scb.ot.integrate(
        adata,
        preset="paired",
        obsm_key="X_pca",
        batch_key="batch",
        out_key="X_ot",
        mode="ufgw_barycenter",
        view_keys=("X_pca", "X_lsi"),
    )

The `view_keys` tuple points to the RNA PCA and ATAC LSI embeddings so the barycentric objective leverages the paired measurements directly.

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

All OT entry points share the ``use_gpu``/``gpu_device`` and ``ot_backend`` knobs.
In addition, :func:`scbiot.ot.integrate` (and the modality presets that wrap it)
now expose an ``ot_mode`` parameter that selects between unbalanced OT
(``"unbalanced"``, the rare-aware behavior) and balanced OT (``"balanced"``) for
stronger batch mixing. When you request balanced OT while keeping
``reference="largest"``, scBIOT automatically switches to the union reference so
that every batch can move symmetrically.
