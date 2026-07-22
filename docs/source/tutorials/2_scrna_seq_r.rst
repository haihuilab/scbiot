2. scRNA-seq in R (Seurat + reticulate)
====================================

:download:`Open the R Markdown workflow <../../../tutorials/R/scRNA-seq_scbiot_in_R.Rmd>`
for a complete, runnable script. The steps mirror the Python notebooks but stay
inside a Seurat workflow by hopping into ``scbiot`` through ``reticulate``.

Environment setup
-----------------

Load the required packages and (optionally) the demo PBMC dataset from
``SeuratData``:

.. code-block:: r

    library(Seurat)
    library(SeuratData)
    library(reticulate)
    library(tidyverse)
    library(sceasy)  # converts Seurat objects to AnnData

    data("ifnb")
    alldata <- UpdateSeuratObject(ifnb)
    alldata[["pct_mt"]] <- PercentageFeatureSet(alldata, pattern = "^MT-")
    alldata <- subset(
      alldata,
      subset = nFeature_RNA > 200 & nFeature_RNA < 2500 & pct_mt < 5
    )

Convert to AnnData
------------------

``scBIOT`` works with AnnData inputs. ``sceasy`` bridges Seurat and AnnData,
and ``reticulate`` exposes the Python API:

.. code-block:: r

    sc <- import("scanpy", convert = FALSE)
    scb <- import("scbiot", convert = FALSE)

    adata <- sceasy::convertFormat(
      alldata,
      from = "seurat",
      to = "anndata",
      main_layer = "counts",
      drop_single_values = FALSE
    )

    sceasy::convertFormat(
      alldata,
      from = "seurat",
      to = "anndata",
      main_layer = "counts",
      drop_single_values = FALSE,
      outFile = "ifnb.h5ad"
    )

Preprocess and run PCA with Scanpy
----------------------------------

Once the data sit in AnnData, call the same Scanpy preprocessing helpers we
use in the Python quick start:

.. code-block:: r

    sc$pp$highly_variable_genes(
      adata,
      n_top_genes = as.integer(2000),
      flavor = "seurat_v3",
      batch_key = "stim"
    )
    sc$pp$normalize_total(adata)
    sc$pp$log1p(adata)
    sc$pp$scale(adata)
    sc$tl$pca(adata, n_comps = as.integer(50), use_highly_variable = TRUE)

Run optimal transport + annotate neighbors
------------------------------------------

Call the OT integrator and reuse Scanpy for graph construction and Leiden
clustering:

.. code-block:: r

    res <- scb$ot$integrate(
      adata,
      preset = "rna",
      obsm_key = "X_pca",
      batch_key = "stim",  # point this at your batch/condition column
      out_key = "X_ot"
    )
    adata <- res[[0]]
    metrics <- res[[1]]

    sc$pp$neighbors(adata, use_rep = "X_ot")
    sc$tl$umap(adata)
    sc$tl$leiden(adata, resolution = 0.8, key_added = "leiden_X_ot")
    adata

Visualise OT embeddings inside Seurat
-------------------------------------

Mirror the OT UMAP coordinates inside the Seurat object for quick comparison:

.. code-block:: r

    ot <- as.matrix(adata$obsm["X_umap"])
    rownames(ot) <- colnames(alldata)

    alldata[["ot"]] <- CreateDimReducObject(
      embeddings = ot,
      key = "ot_umap_",
      assay = DefaultAssay(alldata)
    )

    DimPlot(alldata, reduction = "ot", group.by = "stim", label = TRUE) +
      ggtitle("Optimal Transport Integration")

Finishing up
------------

``sessionInfo()`` records package versions when sharing results:

.. code-block:: r

    info <- sessionInfo()
    info$loadedOnly <- NULL
    print(info, locale = FALSE)

The linked R Markdown contains all of the above cells so it can be rendered via
``rmarkdown::render()`` or opened interactively in RStudio / VS Code.
