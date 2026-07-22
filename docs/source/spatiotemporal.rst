Spatiotemporal dynamics
=======================

scBIOT 1.2.0 can preserve spatial coordinates and ordered biological time during
integration, estimate lineage-specific velocity fields between time bins, and
rank genes associated with transport along a trajectory.

Spatial/time-aware integration
------------------------------

``spatial_key`` names coordinates in ``adata.obsm`` and ``time_key`` names a
continuous or categorical column in ``adata.obs``. Their weights are semantic
0–1 controls. Automatic Gaussian prealignment is disabled when ``time_key`` is
present because aligning every time point to one distribution can erase real
trajectory structure.

.. code-block:: python

   import scbiot as scb

   adata, metrics = scb.ot.integrate(
       adata,
       obsm_key="X_pca",
       batch_key="sample",
       out_key="X_scbiot_st",
       spatial_key="spatial",
       spatial_weight=0.5,
       time_key="timepoint",
       time_weight=0.5,
       time_mode="auto",
   )

Lineage-specific velocity fields
--------------------------------

The velocity routine builds centroids within time-bin/lineage groups, couples
adjacent bins with entropic OT, and interpolates the resulting field back to all
cells. ``lineage_key`` may identify a hard label in ``adata.obs`` or a soft
lineage-membership matrix in ``adata.obsm``.

.. code-block:: python

   adata = scb.tl.velocity_field_sb_centroids(
       adata,
       obsm_key="X_scbiot_st",
       spatial_key="spatial",
       time_key="timepoint",
       lineage_key="lineage",
       out_vel_key="velocity_sb",
       time_bins=20,
       n_centroids_per_bin=512,
   )

The velocity vectors are stored in ``adata.obsm["velocity_sb"]`` and run metadata
in ``adata.uns["velocity_sb_meta"]``.

Transport genes and energy
--------------------------

Use the high-level trajectory ranking helper to compare consecutive time points
within each lineage. It stores detailed results in ``adata.uns`` and returns a
table per lineage.

.. code-block:: python

   ranked = scb.tl.rank_transport_score(
       adata,
       time_key="timepoint",
       lineage_key="lineage",
       rep_key="X_scbiot_st",
       store_key="transport_score",
       n_perms=200,
   )

After a forward transported-expression layer has been created, per-cell energy
and gene dynamics can be visualized directly.

.. code-block:: python

   scb.tl.compute_transport_energy(
       adata,
       layer="transport_fwd",
       key_added="transport_energy",
       log1p=True,
   )
   scb.pl.umap_transport_energy(adata, key="transport_energy")
   scb.pl.transport_gene_dynamics(
       adata,
       gene="SOX9",
       pseudotime_key="timepoint",
       layer="transport_fwd",
   )

See :doc:`analysis` and :doc:`plotting` for the full downstream API.
