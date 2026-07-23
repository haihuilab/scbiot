#!/usr/bin/env Rscript
# Seurat label transfer on a frozen benchmark dataset.
# Invoked by reproduce_fig2_label_transfer.py:
#   Rscript run_seurat.R <dataset> <datasets_dir> <results_dir>
# Writes predictions and its own runtime/peak memory; scoring happens in
# Python so every method passes through the same metric.
suppressPackageStartupMessages({
  library(Seurat); library(zellkonverter)
  library(SingleCellExperiment); library(glue)
})
options(Seurat.object.assay.version = "v5")
options(future.globals.maxSize = 300 * 1024^3)
set.seed(0)

t_start <- Sys.time()
args <- commandArgs(trailingOnly = TRUE)
dataset <- args[1]; data_dir <- args[2]; out_dir <- args[3]

sce <- readH5AD(glue("{data_dir}/{dataset}.h5ad"), X_name = "counts")

# zellkonverter drops the h5ad obs index, so R would otherwise invent
# Cell_1..Cell_N and nothing would join back to the frozen labels.
cells <- read.csv(glue("{out_dir}/{dataset}.cells.csv"))$cell
stopifnot(length(cells) == ncol(sce))
counts <- assay(sce, "counts"); colnames(counts) <- cells
meta <- as.data.frame(colData(sce)); rownames(meta) <- cells
obj <- CreateSeuratObject(counts = counts, meta.data = meta)

is_ref <- obj$bench_split == "reference"
ref <- subset(obj, cells = colnames(obj)[is_ref])
query <- subset(obj, cells = colnames(obj)[!is_ref])

ref <- NormalizeData(ref, verbose = FALSE)
ref <- FindVariableFeatures(ref, verbose = FALSE)
if (ncol(ref) > 50000) {
  ref <- SketchData(ref, ncells = 50000, method = "LeverageScore",
                    sketched.assay = "sketch")
  DefaultAssay(ref) <- "sketch"
}
ref <- ScaleData(ref, verbose = FALSE)
ref <- RunPCA(ref, verbose = FALSE, reduction.name = "pca")

query <- NormalizeData(query, verbose = FALSE)
query <- FindVariableFeatures(query, verbose = FALSE)
query <- ScaleData(query, verbose = FALSE)
query <- RunPCA(query, verbose = FALSE)

anchors <- FindTransferAnchors(reference = ref, query = query, dims = 1:20,
                               reference.reduction = "pca")
ref_labels <- FetchData(ref, vars = "semi_cell_type", cells = Cells(ref))[, 1]
transfer <- TransferData(anchorset = anchors, refdata = ref_labels, dims = 1:20)

write.csv(
  data.frame(cell = colnames(query),
             prediction = as.character(transfer$predicted.id),
             stringsAsFactors = FALSE),
  glue("{out_dir}/{dataset}_Seurat.pred.csv"), row.names = FALSE)

elapsed <- as.numeric(difftime(Sys.time(), t_start, units = "secs"))
peak_mb <- sum(gc()[, "max used"] * c(8, 8) / 1e6, na.rm = TRUE)
write.csv(data.frame(runtime_s = round(elapsed, 1),
                     peak_rss_gb = round(peak_mb / 1024, 2)),
          glue("{out_dir}/{dataset}_Seurat.resources.csv"), row.names = FALSE)
