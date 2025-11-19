Supervised scBIOT (SupBIOT)
===========================

:download:`Open notebook <../../../examples/supervised_lung_benchmarking.ipynb>` to see label transfer and supervised contrastive training for scRNA-seq data.

Integrate with supervised OT (label-aware barycentric mapping and tuned transport hyperparameters):

.. code-block:: python

    adata, metrics = scb.ot.integrate(
        adata,
        modality="supervised",
        obsm_key="X_pca",
        batch_key="batch",
        out_key="X_ot",
        true_label_key="semi_cell_type",
        # OT
        K_ref=1024,
        K_batch=448,
        reg=0.03,
        reg_m=0.40,
        # Connectivity (relaxed)
        sharpen=0.15,
        K_pseudo=24,
        pull=0.75,
        push=0.30,
        lambda0_hi=0.50,
        lambda0_lo=0.35,
        smin_bulk=0.75,
        smax_bulk=1.65,
        smin_bridge=0.85,
        smax_bridge=1.25,
        max_step_local=1.0,
        step_lo=0.75,
        step_hi=0.95,
        q_start=0.80,
        q_end=0.90,
        overlap0_lo=0.60,
        overlap0_hi=0.70,
        w_overlap=0.20,
        w_strain=1.0,
        penalty_gamma=1.5,
        # supervised
        lam_sup=0.60,
        lam_repulse=0.18,
        use_gpu=True,
        ot_backend="torch",
        verbose=True,
    )
    print(metrics)

Set up AnnData and train the model, keeping the supervised embedding:

.. code-block:: python

    scb.pp.setup_anndata(
        adata,
        var_key="X_ot",
        batch_key="batch",
        pseudo_key="leiden_X_ot",
        true_key=None,
    )

    model = scb.models.vae(adata, verbose=True)
    model.train()
    SCBIOT_LATENT_KEY = "supBIOT"
    adata.obsm[SCBIOT_LATENT_KEY] = model.get_latent_representation(
        n_compoents=50,
        svd_solver="arpack",
        random_state=42,
    )

Compute UMAPs/Leiden clusters for both OT and supervised embeddings:

.. code-block:: python

    sc.pp.neighbors(adata, use_rep="X_ot")
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.8, key_added="leiden_X_ot")

    methods = ["X_ot", "supBIOT"]
    for method in methods:
        sc.pp.neighbors(adata, use_rep=method)
        sc.tl.umap(adata)
        adata.obsm[f"X_umap_{method}"] = adata.obsm["X_umap"].copy()
        sc.tl.leiden(adata, key_added=f"{method}_leiden", resolution=0.8)

Plot UMAPs coloured by batch and Leiden clusters to inspect mixing and label structure:

.. code-block:: python

    import matplotlib.pyplot as plt
    import scanpy as sc

    fig, axes = plt.subplots(2, len(methods), figsize=(4 * len(methods), 8), squeeze=False)
    for col, method in enumerate(methods):
        sc.pl.embedding(
            adata,
            basis=f"X_umap_{method}",
            color="batch",
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

Benchmark embeddings with ``scib-metrics`` to balance batch correction and biology:

.. code-block:: python

    bm = Benchmarker(
        adata,
        batch_key="batch",
        label_key="cell_type",
        bio_conservation_metrics=BioConservation(),
        batch_correction_metrics=BatchCorrection(),
        embedding_obsm_keys=["X_pca", "X_ot", "supBIOT"],
        n_jobs=-1,
    )
    bm.benchmark()
    bm.plot_results_table(min_max_scale=False)

Figures from the notebook:

.. figure:: _static/plots/supervised_lung_benchmarking_plot01.png
   :alt: UMAPs comparing OT and supervised scBIOT embeddings.
   :width: 100%

.. figure:: _static/plots/supervised_lung_benchmarking_plot02.png
   :alt: scib-metrics table for supervised scBIOT benchmarking.
   :width: 100%
