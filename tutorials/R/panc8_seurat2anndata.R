rm(list = ls())
library(Seurat)
library(SeuratData)
library(SingleCellExperiment)
library(zellkonverter)
dir <- dirname(here::here())


InstallData("panc8")
data("panc8")
seurat_obj <- UpdateSeuratObject(panc8)

sce <- as.SingleCellExperiment(seurat_obj)

# Write AnnData
writeH5AD(sce, paste0(dir, "/inputs/panc8.h5ad"))
