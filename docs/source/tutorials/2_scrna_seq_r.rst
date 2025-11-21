2. scRNA-seq in R (Seurat + reticulate)
====================================

:download:`Open the R Markdown workflow <../../../examples/R/scRNA-seq_scbiot_in_R.Rmd>`
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

Preprocess and run PCA
----------------------

Standard log-normalisation, variable feature selection, scaling, and PCA:

.. code-block:: r

    alldata <- NormalizeData(
        alldata,
        normalization.method = "LogNormalize",
        scale.factor = 10000
      ) %>%
      FindVariableFeatures(selection.method = "vst", nfeatures = 2000) %>%
      ScaleData(
        features = VariableFeatures(object = alldata),
        vars.to.regress = c("nCount_RNA", "pct_mt")
      ) %>%
      RunPCA(npcs = 50)

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

Run optimal transport + latent model
------------------------------------

Call the same helpers showcased in the notebooks, but through ``reticulate``:

.. code-block:: r

    res <- scb$ot$integrate(
      adata,
      modality = "rna",
      obsm_key = "X_pca",
      batch_key = "stim",  # point this at your batch/condition column
      out_key = "X_ot",
      ot_mode = "balanced"
    )
    adata <- res[[0]]
    metrics <- res[[1]]

    sc$pp$neighbors(adata, use_rep = "X_ot")
    sc$tl$umap(adata)
    sc$tl$leiden(adata, resolution = 0.8, key_added = "leiden_X_ot")

    scb$pp$setup_anndata(
      adata,
      var_key = "X_ot",
      batch_key = "stim",
      pseudo_key = "leiden_X_ot",
      true_key = NULL,
      overwrite = TRUE
    )

    model <- scb$models$vae(adata, verbose = TRUE)
    model$train()

Bring the latent space back into Seurat
---------------------------------------

.. code-block:: r

    latent <- model$get_latent_representation(
      n_compoents = as.integer(50),
      svd_solver = "arpack",
      random_state = as.integer(42)
    )
    latent <- as.matrix(latent)
    rownames(latent) <- colnames(alldata)

    alldata[["scbiot"]] <- CreateDimReducObject(
      embeddings = latent,
      key = "scbiot_",
      assay = DefaultAssay(alldata)
    )

Visualise embeddings and markers
--------------------------------

Use the OT/latent embeddings in the usual Seurat plotting helpers:

.. code-block:: r

    alldata <- RunUMAP(alldata, dims = 1:30, reduction = "scbiot")
    alldata <- FindNeighbors(alldata, dims = 1:30, reduction = "scbiot", k.param = 15)
    alldata <- FindClusters(
      alldata,
      resolution = 0.8,
      cluster.name = "scbiot.res.0.8"
    )

    DimPlot(alldata, reduction = "umap", group.by = "stim", label = TRUE) +
      ggtitle("scBIOT Integration")

    FeaturePlot(
      alldata,
      features = c("SELL", "CREM", "CD8A", "GNLY", "CD79A", "FCGR3A", "CCL2", "PPBP"),
      min.cutoff = "q9"
    )

    FeaturePlot(
      alldata,
      features = c("GNLY", "IFI6"),
      split.by = "stim",
      max.cutoff = 3,
      cols = c("grey", "darkred")
    )

Finishing up
------------

``sessionInfo()`` records package versions when sharing results:

.. code-block:: r

    info <- sessionInfo()
    info$loadedOnly <- NULL
    print(info, locale = FALSE)

The linked R Markdown contains all of the above cells so it can be rendered via
``rmarkdown::render()`` or opened interactively in RStudio / VS Code.
