Unpaired Multiomics
===================

:download:`Open notebook <../../../examples/unpaired_Yao-2021_integration.ipynb>` to match modalities with optimal transport when cells are unpaired.

Preprocess ATAC, harmonize gene names, and build an aligned co-embedding:

.. code-block:: python

    from scbiot.ot.coembedding import (
        build_aligned_coembedding,
        harmonize_gene_names,
        label_transfer_shared_pca,
        preprocess_atac,
    )

    atac_prep = preprocess_atac(
        adata_atac,
        gtf_file=f"{dir}/inputs/gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz",
        verbose=True,
    )
    adata_ga = atac_prep.ga

    harmonize_gene_names(
        adata_gex,
        adata_ga,
        rna_name_col="gene_name",
        verbose=True,
    )

    coembed = build_aligned_coembedding(adata_gex, adata_ga, verbose=True)
    rep_key = coembed.embedding_key
    print(f"[HVG] Joint (shared) HVGs: {len(coembed.genes):,}")

Transfer labels into ATAC via the shared PCA space and assemble the joint embedding:

.. code-block:: python

    label_transfer_shared_pca(
        adata_gex,
        adata_ga,
        label_key="cell_type",
        use_rep="X_pca_shared_aligned",
        min_conf=0.55,
    )

    true_labels = adata_ga.obs.get("cell_type_original", adata_ga.obs.get("cell_type"))
    pred_labels = adata_ga.obs["pred_cell_type"]
    pred_labels_str = pred_labels.astype(str)
    mask = (~pd.isna(true_labels)) & (~pd.isna(pred_labels)) & (pred_labels_str != "unknown")
    if mask.any():
        nmi_score = normalized_mutual_info_score(
            true_labels[mask].astype(str),
            pred_labels_str[mask],
        )
        print(f"[Metrics] NMI(pred_cell_type vs cell_type) = {nmi_score:.4f}")

    joint_rep_key = rep_key
    adata = atac_prep.assemble_joint_embedding(joint_rep_key, {"RNA": adata_gex})

Plot the aligned embedding to inspect label transfer quality and clustering:

.. code-block:: python

    sc.pp.neighbors(adata, use_rep=joint_rep_key)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.8, key_added=f"{joint_rep_key}_leiden")

    import matplotlib.pyplot as plt
    import scanpy as sc

    fig, axes = plt.subplots(2, 1, figsize=(6, 10), squeeze=False)
    sc.pl.embedding(
        adata,
        basis="X_umap",
        color="cell_type",
        frameon=False,
        ax=axes[0, 0],
        show=False,
        legend_loc="on data",
    )
    sc.pl.embedding(
        adata,
        basis="X_umap",
        color=f"{joint_rep_key}_leiden",
        frameon=False,
        ax=axes[1, 0],
        show=False,
        legend_loc="on data",
    )
    plt.tight_layout()

Figures from the notebook:

.. figure:: _static/plots/unpaired_Yao-2021_integration_plot01.png
   :alt: UMAP coloured by cell type for unpaired RNA+ATAC coembedding.
   :width: 100%

.. figure:: _static/plots/unpaired_Yao-2021_integration_plot02.png
   :alt: UMAP coloured by Leiden clusters for the shared embedding.
   :width: 100%
