#!/usr/bin/env Rscript

# This script was contributed by Maike Nägele.
#
# Convert Parse scRNA-seq output to a Seurat object and attach plate-layout metadata
#
# Call it with:
#   Rscript 01_load_parse_healthy.R \
#       --dge-dir **path to folder with Parse DGE data** \
#       --layout-path **path to plate layout CSV** \
#       --sample-ids **SAMPLE_IDs** \
#       --output-path healthyDonors_SHX2.rds

library(argparse)
library(Seurat)

parse_args <- function() {
    parser <- ArgumentParser(description = "Convert Parse DGE output to a Seurat object.")
    parser$add_argument("--dge-dir", help = "Parse DGE folder containing count_matrix.mtx, cell_metadata.csv and all_genes.csv")
    parser$add_argument("--layout-path", help = "Plate layout CSV with a 'Sample_name' column; all other columns are added as cell metadata")
    parser$add_argument("--sample-ids", nargs = "+", default = NULL, help = "Sample IDs to keep, separated by spaces")
    parser$add_argument("--output-path", help = "Path for the output Seurat .rds file")
    return(parser$parse_args())
}

load_parse <- function(dge_dir) {
    cells <- read.csv(file.path(dge_dir, "cell_metadata.csv"), row.names = "bc_wells")
    counts <- ReadMtx(
        mtx = file.path(dge_dir, "count_matrix.mtx"),
        cells = file.path(dge_dir, "cell_metadata.csv"),
        features = file.path(dge_dir, "all_genes.csv"),
        feature.sep = ",", cell.sep = ",",
        skip.cell = 1, skip.feature = 1, mtx.transpose = TRUE
    )
    seurat_obj <- CreateSeuratObject(counts = counts, project = "SHX2", meta.data = cells)
    seurat_obj$sample <- gsub("[-–]", "_", seurat_obj$sample)
    return(seurat_obj)
}

add_layout <- function(seurat_obj, layout_path) {
    layout <- read.csv(layout_path)
    layout$Sample_name <- gsub("[-–]", "_", layout$Sample_name)

    layout_cols <- layout[match(seurat_obj$sample, layout$Sample_name), ]
    rownames(layout_cols) <- colnames(seurat_obj)
    return(AddMetaData(seurat_obj, layout_cols[, names(layout_cols) != "Sample_name"]))
}

main <- function() {
    args <- parse_args()
    seurat_obj <- load_parse(args$dge_dir)
    if (!is.null(args$sample_ids)) {
        seurat_obj <- subset(seurat_obj, subset = sample %in% gsub("[-–]", "_", args$sample_ids))
    }
    seurat_obj <- add_layout(seurat_obj, args$layout_path)
    saveRDS(seurat_obj, args$output_path)
}

main()
