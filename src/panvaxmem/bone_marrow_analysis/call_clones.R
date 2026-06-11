#!/usr/bin/env Rscript

# Call mitochondrial clones for a sample from mtscATAC-seq MGATK output.

library(argparse)
library(Seurat)
library(Signac)
library(ggplot2)
library(patchwork)
library(lsa)


parse_args <- function() {
    parser <- ArgumentParser(description = "Call mitochondrial clones for a sample from MGATK output.")
    parser$add_argument(
        "--mgatk-dir",
        default = "/data/yosef2/users/ruchir/pan_vaccine_immune_response/compile_sc_data/TIV_addl/bm_mtscATAC/clonal_analysis/mgatk/p24243-s005_A5_mgatk/final"
    )
    parser$add_argument("--output-tables-dir", default = "outputs/clonal_calls/A5")
    parser$add_argument("--output-heatmap-path", default = "plotting/figures/A5_clone_heatmap.pdf")
    return(parser$parse_args())
}


call_clones <- function(mgatk_dir) {
    mito.data <- ReadMGATK(dir = mgatk_dir)

    mito <- CreateSeuratObject(
        counts = mito.data$counts,
        meta.data = mito.data$depth,
        assay = "mito"
    )
    mito <- subset(mito, mito.depth >= 10)

    variants <- IdentifyVariants(mito, refallele = mito.data$refallele)
    high.conf <- subset(
        variants,
        subset = (n_cells_conf_detected >= 5 & strand_correlation >= 0.65 & vmr > 0.01)
    )

    mito <- AlleleFreq(object = mito, variants = high.conf$variant, assay = "mito")
    DefaultAssay(mito) <- "alleles"
    mito <- FindClonotypes(mito, resolution = 1.5)
    return(mito)
}


publish_tables <- function(mito, output_dir) {
    dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

    write.csv(
        mito@meta.data,
        file = file.path(output_dir, "clones.csv"),
        row.names = TRUE,
        quote = FALSE
    )
    write.table(
        x = as.data.frame(VariableFeatures(mito)),
        file = file.path(output_dir, "high_conf_variants.tsv"),
        sep = "\t",
        quote = FALSE,
        row.names = FALSE
    )
}


publish_heatmap <- function(mito, heatmap_path) {
    dir.create(dirname(heatmap_path), showWarnings = FALSE, recursive = TRUE)

    heatmap <- DoHeatmap(
        mito,
        features = VariableFeatures(mito),
        slot = "data",
        disp.max = 0.1,
        angle = 90,
        size = 2.5
    ) + scale_fill_viridis_c()

    ggsave(heatmap_path, plot = heatmap, width = 15, height = 8)
}


main <- function() {
    args <- parse_args()
    mito <- call_clones(args$mgatk_dir)
    publish_tables(mito, args$output_tables_dir)
    publish_heatmap(mito, args$output_heatmap_path)
}


main()
