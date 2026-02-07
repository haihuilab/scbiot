Label transfer from scRNA-seq to spatial Xenium
===============================================

:download:`Open notebook <../../../examples/xenium_10x_ovarian_label_transfer_with_supbiot.ipynb>`
to reproduce the ovarian Xenium label transfer workflow with supBIOT.

Inputs and quality control
--------------------------

Load the Xenium Zarr export (see the notebook for the one-time conversion step)
and compute basic QC metrics:

.. code-block:: python

    from pathlib import Path

    import numpy as np
    import pandas as pd
    import scanpy as sc
    import spatialdata as sd

    import scbiot as scb

    ROOT = Path.cwd()
    DATA_DIR = ROOT / "inputs" / "xenium"

    zarr_path = DATA_DIR / "Xenium_Prime_Ovarian_Cancer.zarr"
    rna_h5 = DATA_DIR / "reference" / "17k_Ovarian_Cancer_scFFPE_count_filtered_feature_bc_matrix.h5"
    cell_type_csv = DATA_DIR / "reference" / "FLEX_Ovarian_Barcode_Cluster_Annotation.csv"

    sdata = sd.read_zarr(zarr_path)
    adata_query = sdata.tables["table"]

    sc.pp.calculate_qc_metrics(adata_query, percent_top=(10, 20, 50, 150), inplace=True)
    adata_query.layers["counts"] = adata_query.X.copy()
    sc.pp.filter_cells(adata_query, min_counts=20)
    sc.pp.filter_cells(adata_query, max_counts=np.quantile(adata_query.obs["total_counts"], 0.98))
    sc.pp.filter_genes(adata_query, min_cells=100)

Load the scRNA-seq reference, map barcodes to cell types, and harmonize gene symbols:

.. code-block:: python

    ref = sc.read_10x_h5(rna_h5)
    df_celltype = pd.read_csv(cell_type_csv)

    mapping = (
        df_celltype.assign(_bc=df_celltype["Barcode"].astype(str))[["_bc", "Cell Annotation"]]
        .drop_duplicates("_bc")
        .set_index("_bc")["Cell Annotation"]
    )
    ref.obs["cell_type"] = pd.Series(ref.obs_names.astype(str), index=ref.obs_names).map(mapping)
    adata_ref = ref[ref.obs["cell_type"].notna()].copy()

    adata_ref.layers["counts"] = adata_ref.X.copy()
    adata_query.layers["counts"] = adata_query.X.copy()
    adata_ref.var["gene_symbol"] = adata_ref.var_names

    adata_ref = sc.get.aggregate(
        adata_ref,
        by="gene_symbol",
        axis="var",
        func="sum",
        layer="counts",
    )
    adata_ref.layers["counts"] = adata_ref.layers["sum"]
    adata_ref.X = adata_ref.layers["sum"]

supBIOT label transfer
----------------------

Compute a reference UMAP backbone, concatenate reference/query cells, and run
label-aware OT plus supBIOT prediction:

.. code-block:: python

    import anndata as ad

    adata_ref_umap = adata_ref.copy()
    sc.pp.highly_variable_genes(adata_ref_umap, n_top_genes=2000, flavor="seurat_v3")
    sc.pp.normalize_total(adata_ref_umap)
    sc.pp.log1p(adata_ref_umap)
    sc.pp.scale(adata_ref_umap)
    sc.tl.pca(adata_ref_umap, n_comps=50, use_highly_variable=True)
    sc.pp.neighbors(adata_ref_umap, use_rep="X_pca")
    sc.tl.umap(adata_ref_umap)

    adata_ref.obsm["X_pca"] = adata_ref_umap.obsm["X_pca"].copy()
    adata_ref.obsm["X_umap"] = adata_ref_umap.obsm["X_umap"].copy()

    adata_query.obs["cell_type"] = "Unknown"
    adata = ad.concat([adata_ref, adata_query], join="inner", label="batch", keys=["reference", "query"])
    adata.X = adata.layers["counts"].copy()

    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3")
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    sc.tl.pca(adata, n_comps=50, use_highly_variable=True)

    adata, metrics = scb.ot.integrate(
        adata,
        preset="anchor",
        obsm_key="X_pca",
        batch_key="batch",
        reference_category="reference",
        out_key="X_supbiot",
        label_key="cell_type",
        unlabeled_category="Unknown",
    )

    adata = scb.ot.supbiot(
        adata,
        label_key="cell_type",
        unlabeled_category="Unknown",
        pred_label_key="pred_cell_type",
        pred_conf_key="pred_confidence",
        min_conf=0.25,
    )

Inspect label assignments
-------------------------

Split back out the query cells, plot confidence scores, and visualize the
reference/query alignment in UMAP space:

.. code-block:: python

    adata_query = adata[adata.obs["batch"] == "query"].copy()
    adata_query.obs["pred_cell_type"].value_counts()

    sc.pl.violin(adata_query, keys="pred_confidence", groupby="pred_cell_type", rotation=90)

    sc.pp.neighbors(adata, use_rep="X_supbiot", n_neighbors=50, metric="cosine")
    sc.tl.umap(adata, min_dist=0.3, spread=1.0, random_state=0)

    sc.pl.embedding(
        adata,
        basis="X_umap",
        color="batch",
        legend_loc="right margin",
        title="Reference and query",
    )
    sc.pl.embedding(
        adata,
        basis="X_umap",
        color="pred_cell_type",
        legend_loc="right margin",
        title="",
    )

.. figure:: /_static/plots/xenium_label_transfer_pred_conf_violin.png
   :alt: supBIOT confidence by predicted cell type for query Xenium spots.
   :width: 70%

   Predicted labels carry assessment scores that help you filter low-confidence spots.

.. figure:: /_static/plots/xenium_label_transfer_query_ref_umap.png
   :alt: UMAP embedding of the Xenium query and scRNA-seq reference coloured by batch (left) and predicted labels (right).
   :width: 85%

   The supBIOT embedding keeps the reference structure intact while aligning query spots.

Marker-level validation
-----------------------

Define the marker list that best represents the predicted cell types and run a
`dotplot <https://scanpy.readthedocs.io/>`_ to ensure predicted labels express the
expected genes.  The resulting visualization highlights each marker set's
relative expression and (with the same dot sizes exported from the notebook) is
shown below.

.. figure:: /_static/plots/xenium_label_transfer_marker_dotplot.png
   :alt: Dotplot of curated marker genes for Xenium predictions.
   :width: 85%

   Dotplot colours report average expression, dot sizes report fraction of cells.

Correlation-based diagnostics
-----------------------------

Matching mean expression profiles between reference and predicted groups helps
spot systematic label mismatches.  Use ``scb.pl.celltype_gene_mean_correlation``
and ``scb.pl.celltype_predtype_mean_corr_heatmap`` (as in the notebook) to generate
gene-wise scatterplots and heatmaps for the top marker genes.

.. figure:: /_static/plots/xenium_label_transfer_gene_mean_corr.png
   :alt: Correlation between predicted and reference mean expression for candidate markers.
   :width: 100%

   Each point summarizes a gene's mean expression across matched cell types.

.. figure:: /_static/plots/xenium_label_transfer_predtype_heatmap.png
   :alt: Heatmap of Pearson correlations between reference and predicted classes.
   :width: 100%

   The darker diagonal indicates strong agreement for most cell types.
