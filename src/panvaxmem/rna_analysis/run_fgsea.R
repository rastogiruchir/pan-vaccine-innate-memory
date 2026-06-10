#!/usr/bin/env Rscript

# Run fgsea on every DEG CSV in a directory, including per-donor DEG CSVs.
# The statistic used for ranking is the LFC_median statistic from scVI's differential expression.

library(argparse)
library(fgsea)

parser <- ArgumentParser(
    description = "Run fgsea for all DEG CSVs in a directory."
)
parser$add_argument("--deg-dir", required = TRUE)
parser$add_argument("--output-dir", required = TRUE)
parser$add_argument("--gene-set-dir", default = "../../../resources/gene_sets")
args <- parser$parse_args()

GENE_SET_PATHS <- c(
    "BTM" = file.path(args$gene_set_dir, "btm_for_gsea_20131008.gmt"),
    "HALLMARK" = file.path(args$gene_set_dir, "hallmark_hs_v2023_2.gmt"),
    "REACTOME" = file.path(args$gene_set_dir, "reactome_pathways.gmt")
)

get_deg_paths <- function(deg_dir) {
    deg_paths <- list.files(deg_dir, pattern = "\\.csv$", full.names = TRUE)
    per_donor_dir <- file.path(deg_dir, "per_donor")
    if (dir.exists(per_donor_dir)) {
        deg_paths <- c(
            deg_paths,
            list.files(per_donor_dir, pattern = "\\.csv$", full.names = TRUE)
        )
    }
    return(deg_paths)
}

get_fold_changes <- function(dge_path) {
    dge <- read.table(dge_path, row.names = 1, header = TRUE, sep = ",")
    fold_changes <- setNames(dge$lfc_median, rownames(dge))
    return(fold_changes)
}

run_fgsea <- function(gene_set_path, fold_changes) {
    gene_sets <- gmtPathways(gene_set_path)
    fgsea_res <- fgseaMultilevel(
        gene_sets,
        fold_changes,
        minSize = 8,
        maxSize = 500,
        nPermSimple = 10000
    )
    fgsea_res$leadingEdge <- as.character(fgsea_res$leadingEdge)
    return(fgsea_res)
}

get_output_prefix <- function(deg_path, output_dir) {
    if (basename(dirname(deg_path)) == "per_donor") {
        output_dir <- file.path(output_dir, "per_donor")
    }
    dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
    return(file.path(output_dir, tools::file_path_sans_ext(basename(deg_path))))
}

dir.create(args$output_dir, showWarnings = FALSE, recursive = TRUE)
deg_paths <- get_deg_paths(args$deg_dir)

for (gene_set_path in GENE_SET_PATHS) {
    if (!file.exists(gene_set_path)) {
        stop(sprintf("Gene set file does not exist: %s", gene_set_path))
    }
}

for (deg_path in deg_paths) {
    output_prefix <- get_output_prefix(deg_path, args$output_dir)
    fold_changes <- get_fold_changes(deg_path)
    for (gene_set_name in names(GENE_SET_PATHS)) {
        gene_set_path <- GENE_SET_PATHS[[gene_set_name]]
        fgsea_res <- run_fgsea(gene_set_path, fold_changes)
        output_path <- paste0(output_prefix, ".", gene_set_name, ".tsv")
        write.table(fgsea_res, output_path, sep = "\t", quote = TRUE, row.names = FALSE)
    }
}
