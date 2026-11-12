#!/usr/bin/env Rscript

# This script was contributed by Maike Nägele.
#
# QC, Harmony integration and two-round extraction of classical monocytes from the Parse SHX2 Seurat object
#
# Call it with:
#   Rscript 02_extract_monocytes.R \
#       --input-path healthyDonors_SHX2.rds \
#       --output-dir **path to output folder** \
#       --cmono-clusters 'c("0", "4", "10")' \
#       --final-clusters 'c("0", "1")'

library(argparse)
library(Seurat)
library(harmony)
library(patchwork)
library(ggplot2)
library(scuttle)
library(presto)

celltype_marker <- list(
    t_cell              = c("CD3D", "CD3E", "CD3G", "TRAC", "CD4", "CD8A", "CD8B", "CCR7", "SELL", "IL7R", "TCF7", "MKI67"),
    b_cell              = c("CD19", "MS4A1", "CD79A", "CD79B"),
    nk_cell             = c("KLRD1", "NKG7", "GNLY", "NCAM1"),
    platelet            = c("PPBP", "PF4", "ITGA2B"),
    erythroid           = c("HBB", "HBA1", "HBA2"),
    progenitor          = c("CD34"),
    dc                  = c("FCER1A", "CLEC10A", "CD1C", "CLEC9A", "XCR1", "LILRA4", "IL3RA", "ID2"),
    neutrophil          = c("FCGR3B", "CSF3R", "CXCR2"),
    classical_monos     = c("CD14", "LYZ", "S100A8", "S100A9", "S100A12", "FCN1", "VCAN", "CSF1R", "ITGAM", "MNDA", "CTSS"),
    non_classical_monos = c("FCGR3A", "MS4A7", "CDKN1C", "LST1", "HES4", "IFITM3")
)

small_legend <- theme(
    legend.text = element_text(size = 6),
    legend.title = element_text(size = 7),
    legend.key.size = unit(3, "mm")
)

parse_args <- function() {
    parser <- ArgumentParser(description = "QC, integrate and extract classical monocytes.")
    parser$add_argument("--input-path", default = "healthyDonors_SHX2.rds")
    parser$add_argument("--output-dir", default = "outputs")
    parser$add_argument("--cmono-clusters", default = 'c("0", "4", "10")', help = "Round-1 clusters to keep (R vector)")
    parser$add_argument("--final-clusters", default = 'c("0", "1")', help = "Round-2 clusters to keep (R vector)")
    parser$add_argument("--n-hvg", type = "integer", default = 4000)
    parser$add_argument("--resolution-all", type = "double", default = 0.3)
    parser$add_argument("--resolution-mono", type = "double", default = 0.4)
    parser$add_argument("--seed", type = "integer", default = 1234)
    return(parser$parse_args())
}

qc_filter <- function(seu) {
    seu$sampleID <- paste(seu$ID, seu$Visit, sep = "_")
    seu$percent.mt <- PercentageFeatureSet(seu, pattern = "^MT-")

    # per-sample MAD outliers (>3 MADs) combined with hard cutoffs
    qc <- perCellQCFilters(
        seu@meta.data,
        sum.field = "nCount_RNA",
        detected.field = "nFeature_RNA",
        sub.fields = "percent.mt",
        batch = seu$sampleID,
        nmads = 3
    )
    fail_nCount <- as.logical(qc$low_lib_size)
    fail_nFeature <- as.logical(qc$low_n_features) | seu$nFeature_RNA < 200
    fail_mt <- as.logical(qc$high_percent.mt) | seu$percent.mt > 20
    keep <- !(fail_nCount | fail_nFeature | fail_mt)

    print(c(low_nCount = sum(fail_nCount), low_nFeature = sum(fail_nFeature),
            high_mt = sum(fail_mt), discarded = sum(!keep), kept = sum(keep)))
    return(seu[, keep])
}

run_pca_harmony <- function(seu, n_hvg, seed) {
    seu <- FindVariableFeatures(seu, selection.method = "vst", nfeatures = n_hvg)
    seu <- ScaleData(seu, verbose = FALSE)
    seu <- RunPCA(seu, features = VariableFeatures(seu), seed.use = seed, verbose = FALSE)
    set.seed(seed)
    seu <- RunHarmony(seu, group.by.vars = "ID", reduction.use = "pca", reduction.save = "harmony")
    return(seu)
}

# HBC quantitative elbow: https://hbctraining.github.io/scRNA-seq/lessons/elbow_plot_metric.html
select_pcs <- function(seu) {
    pct <- seu[["pca"]]@stdev / sum(seu[["pca"]]@stdev) * 100
    co1 <- which(cumsum(pct) > 90 & pct < 5)[1]
    co2 <- max(which(diff(pct) < -0.1)) + 1
    return(min(co1, co2))
}

embed_and_cluster <- function(seu, reduction, n_pcs, resolution, name, seed) {
    seu <- RunUMAP(seu, reduction = reduction, dims = 1:n_pcs,
                   reduction.name = paste0("umap_", name), seed.use = seed, verbose = FALSE)
    seu <- FindNeighbors(seu, reduction = reduction, dims = 1:n_pcs,
                         graph.name = paste0(c("nn_", "snn_"), name), verbose = FALSE)
    seu <- FindClusters(seu, graph.name = paste0("snn_", name), resolution = resolution,
                        cluster.name = paste0("clusters_", name), random.seed = seed, verbose = FALSE)
    return(seu)
}

plot_umaps <- function(seu, name, title) {
    umap <- paste0("umap_", name)
    p <- wrap_plots(DimPlot(seu, reduction = umap, group.by = c(paste0("clusters_", name), "ID", "Visit"),
                            combine = FALSE), ncol = 3) /
         wrap_plots(FeaturePlot(seu, reduction = umap, features = c("nCount_RNA", "nFeature_RNA", "percent.mt"),
                                combine = FALSE), ncol = 3)
    print(p + plot_annotation(title = title) & small_legend)
}

plot_markers <- function(seu, clust_col) {
    print(DotPlot(seu, features = unique(unlist(celltype_marker)), group.by = clust_col) + RotatedAxis())
    for (ct in names(celltype_marker)) {
        features <- intersect(celltype_marker[[ct]], rownames(seu))
        if (length(features) > 0) {
            print(VlnPlot(seu, features = features, group.by = clust_col, pt.size = 0, ncol = 2) +
                  plot_annotation(title = ct))
        }
    }
}

main <- function() {
    args <- parse_args()
    dir.create(args$output_dir, showWarnings = FALSE, recursive = TRUE)
    pdf(file.path(args$output_dir, "plots.pdf"), width = 15, height = 9)

    # Round 1: all cells
    seu <- qc_filter(readRDS(args$input_path))
    seu <- NormalizeData(seu, verbose = FALSE)
    seu <- run_pca_harmony(seu, args$n_hvg, args$seed)
    n_pcs <- select_pcs(seu)
    message("Selected PCs (all cells): ", n_pcs)

    seu <- embed_and_cluster(seu, "pca", n_pcs, args$resolution_all, "pca", args$seed)
    plot_umaps(seu, "pca", "Before batch correction")
    seu <- embed_and_cluster(seu, "harmony", n_pcs, args$resolution_all, "harmony", args$seed)
    plot_umaps(seu, "harmony", "After batch correction")

    Idents(seu) <- "clusters_harmony"
    markers <- FindAllMarkers(seu, assay = "RNA", only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25)
    write.csv(markers, file.path(args$output_dir, "markers_round1.csv"), row.names = FALSE)
    plot_markers(seu, "clusters_harmony")

    cmono_clusters <- eval(parse(text = args$cmono_clusters))
    mono <- subset(seu, subset = clusters_harmony %in% cmono_clusters)
    saveRDS(mono, file.path(args$output_dir, "candidate_classical_monocytes.rds"))
    rm(seu); gc()

    # Round 2: candidate monocytes only
    mono <- run_pca_harmony(mono, args$n_hvg, args$seed)
    n_pcs_mono <- select_pcs(mono)
    message("Selected PCs (monocytes): ", n_pcs_mono)

    mono <- embed_and_cluster(mono, "harmony", n_pcs_mono, args$resolution_mono, "mono", args$seed)
    plot_umaps(mono, "mono", "After monocyte re-clustering")
    print(DotPlot(mono, features = unique(unlist(celltype_marker)), group.by = "clusters_mono") + RotatedAxis())
    dev.off()

    final_clusters <- eval(parse(text = args$final_clusters))
    mono_final <- subset(mono, subset = clusters_mono %in% final_clusters)
    print(table(ID = mono_final$ID, Visit = mono_final$Visit))
    saveRDS(mono_final, file.path(args$output_dir, "classical_monocytes.rds"))
}

main()
