API
===

The public v1.2.0 API is organized into preprocessing (``pp``), optimal
transport and label transfer (``ot``), downstream analysis (``tl``), and
plotting (``pl``).

.. autosummary::
   :nosignatures:

   scbiot.ot.integrate
   scbiot.ot.supbiot
   scbiot.pp.autoencoder
   scbiot.pp.autoencoder_map
   scbiot.tl.gene_transport_score
   scbiot.tl.velocity_field_sb_centroids

See the module pages for complete function lists:

* :doc:`ot`
* :doc:`preprocessing`
* :doc:`analysis`
* :doc:`plotting`

Workflow guides are available for :doc:`multiomics` and
:doc:`spatiotemporal` analysis.
