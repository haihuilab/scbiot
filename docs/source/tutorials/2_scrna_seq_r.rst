2. scRNA-seq in R (Seurat + reticulate)
==========================================

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

Build the v1.2.0 autoencoder representation
---------------------------------------------

Use CTRL cells as the labeled reference and STIM cells as the query. The
``semi_cell_type`` column is defined explicitly before it is used:

.. code-block:: r

    np <- import("numpy", convert = FALSE)
    adata$obs[["cell_type"]] <- adata$obs[["seurat_annotations"]]
    adata$obs[["semi_cell_type"]] <- np$where(
      adata$obs[["stim"]]$astype("str")$eq("CTRL"),
      adata$obs[["cell_type"]],
      "Unknown"
    )
    adata$layers[["counts"]] <- adata$X$copy()

    adata <- scb$pp$autoencoder(
      adata,
      input_key = "counts",
      out_key = "X_ae",
      batch_key = "stim",
      random_state = as.integer(0)
    )

Run optimal transport + supBIOT
-------------------------------

Call the v1.2.0 supervised pipeline and reuse Scanpy for graph construction:

.. code-block:: r

    res <- scb$ot$integrate(
      adata,
      obsm_key = "X_ae",
      batch_key = "stim",
      out_key = "X_supbiot",
      label_key = "semi_cell_type",
      unlabeled_category = "Unknown",
      random_state = as.integer(0)
    )
    adata <- res[[0]]
    metrics <- res[[1]]

    adata <- scb$ot$supbiot(
      adata,
      use_rep = "X_supbiot",
      input_rep_key = "X_ae",
      label_key = "semi_cell_type",
      unlabeled_category = "Unknown",
      pred_label_key = "pred_cell_type",
      pred_conf_key = "pred_confidence",
      min_conf = 0,
      random_state = as.integer(0)
    )

    sc$pp$neighbors(adata, use_rep = "X_supbiot")
    sc$tl$umap(adata)
    sc$tl$leiden(
      adata,
      resolution = 0.8,
      key_added = "leiden_X_supbiot"
    )
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
