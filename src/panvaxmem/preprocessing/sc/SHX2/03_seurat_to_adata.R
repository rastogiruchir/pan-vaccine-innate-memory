#!/usr/bin/env Rscript

# This script was contributed by Maike Nägele.
#
# Convert a Seurat object to an h5ad file
#
# Call it with:
#   Rscript 03_seurat_to_adata.R \
#       --input-path **path to Seurat object** \
#       --output-path **output path to AnnData object**

library(argparse)
library(Seurat)
library(SingleCellExperiment)
library(zellkonverter)

parse_args <- function() {
    parser <- ArgumentParser(description = "Convert a Seurat .rds object to .h5ad.")
    parser$add_argument("--input-path", help = "Seurat object (.rds) to convert")
    parser$add_argument("--output-path", help = "Path for the output .h5ad file")
    parser$add_argument("--assay", default = "RNA", help = "Seurat assay to export")
    parser$add_argument("--layer", default = "counts", help = "Assay layer to export")
    return(parser$parse_args())
}

seurat_to_sce <- function(seu, assay, layer) {
    DefaultAssay(seu) <- assay
    counts <- GetAssayData(seu, assay = assay, layer = layer)
    meta <- seu@meta.data[colnames(counts), , drop = FALSE]
    stopifnot(identical(colnames(counts), rownames(meta)))

    return(SingleCellExperiment(assays = list(counts = counts), colData = meta))
}

main <- function() {
    args <- parse_args()
    sce <- seurat_to_sce(readRDS(args$input_path), args$assay, args$layer)

    dir.create(dirname(args$output_path), showWarnings = FALSE, recursive = TRUE)
    writeH5AD(sce, file = args$output_path, X_name = "counts")
    message("Wrote ", ncol(sce), " cells x ", nrow(sce), " genes to ", args$output_path)
}

main()
