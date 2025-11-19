scATAC-seq
==========

:download:`Open notebook <../../../examples/atac_brain_large_window_integration.ipynb>` covering iterative LSI, peak selection, and integration across scATAC-seq replicates.

Preprocess peaks (remove promoter-proximal, select HV peaks, build iterative LSI):

.. code-block:: python

    adata_top = scb.pp.remove_promoter_proximal_peaks(
        adata,
        f"{dir}/inputs/gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz",
    )

    scb.pp.find_variable_features(adata_top, batch_key="batchname_all")
    scb.pp.add_iterative_lsi(
        adata_top,
        n_components=31,
        drop_first_component=True,
        add_key="X_lsi",
    )
    adata.obsm["X_lsi"] = adata_top.obsm["X_lsi"]
    adata.obsm["Unintegrated"] = adata_top.obsm["X_lsi"]

Run OT integration on LSI space, then train the VAE and store the embedding:

.. code-block:: python

    adata, metrics = scb.ot.integrate(
        adata,
        modality="atac",
        obsm_key="X_lsi",
        batch_key="batchname_all",
        out_key="X_ot",
        reference="largest",
    )
    print(metrics)

    scb.pp.setup_anndata(
        adata,
        var_key="X_ot",
        batch_key="batchname_all",
        pseudo_key="leiden_X_ot",
        true_key=None,
    )

    model = scb.models.vae(adata, prior_pcr=5.0, verbose=True)
    model.train()
    SCBIOT_LATENT_KEY = "scBIOT"
    adata.obsm[SCBIOT_LATENT_KEY] = model.get_latent_representation(
        n_compoents=30,
        svd_solver="arpack",
        random_state=42,
    )

Project to UMAP and compare batch mixing / clusters across embeddings:

.. code-block:: python

    sc.pp.neighbors(adata, use_rep="X_ot")
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.8, key_added="leiden_X_ot")

    methods = ["X_ot", "scBIOT"]
    for method in methods:
        sc.pp.neighbors(adata, use_rep=method)
        sc.tl.umap(adata)
        adata.obsm[f"X_umap_{method}"] = adata.obsm["X_umap"].copy()
        sc.tl.leiden(adata, key_added=f"{method}_leiden", resolution=0.8)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(methods), figsize=(4 * len(methods), 8), squeeze=False)
    for col, method in enumerate(methods):
        sc.pl.embedding(
            adata,
            basis=f"X_umap_{method}",
            color="batchname_all",
            frameon=False,
            ax=axes[0, col],
            show=False,
            legend_loc="on data",
            legend_fontsize=10,
            title=f"{method}",
        )
        leiden_key = f"{method}_leiden"
        sc.pl.embedding(
            adata,
            basis=f"X_umap_{method}",
            color=leiden_key,
            frameon=False,
            ax=axes[1, col],
            show=False,
            legend_loc="on data",
            legend_fontsize=10,
        )
    plt.tight_layout()

Quantify integration with ``scib-metrics`` (batch correction + biology):

.. code-block:: python

    bm = Benchmarker(
        adata,
        batch_key="batchname_all",
        label_key="cell_type",
        bio_conservation_metrics=BioConservation(),
        batch_correction_metrics=BatchCorrection(),
        embedding_obsm_keys=["X_pca", "X_ot", "scBIOT"],
        n_jobs=32,
    )
    bm.benchmark()
    bm.plot_results_table(min_max_scale=False)

Figures from the notebook:

.. figure:: _static/plots/atac_brain_large_window_integration_plot01.png
   :alt: First diagnostic plot from the scATAC workflow.
   :width: 90%

.. figure:: _static/plots/atac_brain_large_window_integration_plot02.png
   :alt: Second diagnostic plot from the scATAC workflow.
   :width: 90%

.. figure:: _static/plots/atac_brain_large_window_integration_plot03.png
   :alt: Third diagnostic plot from the scATAC workflow.
   :width: 90%

.. figure:: _static/plots/atac_brain_large_window_integration_plot04.png
   :alt: UMAPs of OT and scBIOT embeddings for scATAC brain dataset.
   :width: 100%

.. figure:: _static/plots/atac_brain_large_window_integration_plot05.png
   :alt: scib-metrics benchmarking table for scATAC integration.
   :width: 100%
