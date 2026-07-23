Optimal transport: ``ot``
=========================

The v1.2.0 OT API uses semantic 0–1 controls rather than named presets.
:func:`scbiot.ot.integrate` is the public integration entry point, and
:func:`scbiot.ot.supbiot` performs label transfer from its integration metadata.
Approximate and centroid-level execution are selected through ``integrate()``
parameters rather than separate public functions.

.. currentmodule:: scbiot.ot

.. autosummary::
   :toctree: generated
   :nosignatures:

   integrate
   supbiot

Basic integration
-----------------

``X_ae`` below is produced by :func:`scbiot.pp.autoencoder`, as shown in
:doc:`usage`.

.. code-block:: python

   adata, metrics = scb.ot.integrate(
       adata,
       obsm_key="X_ae",
       batch_key="batch",
       out_key="X_ot",
       strength=0.5,
       conservation=0.5,
       prototypes=0.5,
   )

``strength`` controls the amount of batch correction, ``conservation`` protects
local geometry, and ``prototypes`` controls prototype resolution. When
``label_key`` is supplied, ``supervision`` controls label guidance.

Reference mapping and label transfer
------------------------------------

Set ``align_reference=True`` when query cells should be mapped into the reference
space. ``prealign="auto"`` is the public default; ``"coral"``, ``"ot"``, or
``None`` may be selected explicitly.

.. code-block:: python

   adata, metrics = scb.ot.integrate(
       adata,
       obsm_key="X_ae",
       batch_key="modality",
       out_key="X_supbiot",
       reference="reference",
       align_reference=True,
       label_key="cell_type",
       unlabeled_category="Unknown",
   )
   adata = scb.ot.supbiot(
       adata,
       use_rep="X_supbiot",
       label_key="cell_type",
       unlabeled_category="Unknown",
       transfer_mode="logreg",
   )

Spatial and temporal structure
------------------------------

Use ``spatial_key`` to include spatial coordinates and ``time_key`` to preserve a
continuous or categorical trajectory. With a ``time_key``, automatic Gaussian
prealignment is disabled to avoid collapsing the trajectory; pass ``prealign``
explicitly only when that behavior is intended.

.. code-block:: python

   adata, metrics = scb.ot.integrate(
       adata,
       obsm_key="X_ae",
       batch_key="sample",
       out_key="X_scbiot_st",
       spatial_key="spatial",
       spatial_weight=0.5,
       time_key="timepoint",
       time_weight=0.5,
   )

See :doc:`spatiotemporal` for velocity fields, transport-gene ranking, and
dynamic visualizations.

Scaling options
---------------

For very large datasets, set ``centroid=True``. For a faster, lower-precision
Sinkhorn solve, set ``approximate=True``.

.. code-block:: python

   adata, metrics = scb.ot.integrate(
       adata,
       obsm_key="X_ae",
       batch_key="batch",
       out_key="X_ot",
       centroid=True,
       n_centroids_per_batch=2048,
       max_samples_per_batch=500_000,
       k_interp=8,
       chunk_size=500_000,
   )
