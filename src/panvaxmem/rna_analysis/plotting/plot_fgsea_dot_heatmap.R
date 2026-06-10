#!/usr/bin/env Rscript

# Plot an fgsea dot heatmap for CM differential expression.

library(argparse)
library(ComplexHeatmap)
library(grid)
source("../../utils/dot_heatmap.R")

parser <- ArgumentParser(description = "Plot combined fgsea results in a dot heatmap.")
parser$add_argument("--nes-path", default = "../outputs/fgsea/combined_tables/late/nes.BTM.csv")
parser$add_argument("--padj-path", default = "../outputs/fgsea/combined_tables/late/padj.BTM.csv")
parser$add_argument("--output-path", default = "figures/btm_fgsea_late_dot_heatmap.pdf")
parser$add_argument("--max-size-value", type = "double", default = NULL)
args <- parser$parse_args()

dir.create(dirname(args$output_path), showWarnings = FALSE, recursive = TRUE)
pdf(args$output_path, width = 9, height = 11)
btm_res <- draw_dot_heatmap(
    args$nes_path,
    args$padj_path,
    legend_title = "NES",
    cluster_columns = TRUE,
    cluster_rows = TRUE,
    show_row_dend = FALSE,
    show_column_dend = TRUE,
    max_size_value = args$max_size_value,
    column_name_gap = unit(1, "line"),
    col_just = "left",
    width_scale = 1.25,
    height_scale = 1.25,
    col_rot = 90,
    dotsize = 0.3,
    header_textsize = 11,
    row_textsize = 10
)
draw(
    btm_res$heatmap,
    heatmap_legend_list = list(btm_res$legend),
    heatmap_legend_side = "right"
)
dev.off()
