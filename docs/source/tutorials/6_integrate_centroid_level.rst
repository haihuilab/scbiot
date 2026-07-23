6. Centroid-Level Optimal Transport
===================================

scBIOT’s centroid workflow projects each batch down to a few thousand centroid
representatives, runs OT on the compressed problem, and then interpolates the
displacement field back to every cell.  This keeps memory bounded while scaling
from benchmarking-sized datasets to the 100-million-cell Tahoe experiment. The
lung notebook is fully reproducible from Figshare; the Tahoe notebook expects a
prepared out-of-core embedding because the source collection is intentionally
not duplicated.

Choose one of the recipes below to dive into the matching notebook.

.. toctree::
   :maxdepth: 1

   a. Lung atlas benchmark <6a_integrate_centroid_level_benchmarking>
   b. Tahoe-100M scaling <6b_integrate_centroid_level_tahoe>
