#!/usr/bin/env Rscript

library(argparse)
library(ArchR)
library(data.table)
library(Matrix)
library(SummarizedExperiment)


parse_args <- function() {
    parser <- ArgumentParser(description = "Run ArchR preprocessing and export peak matrices.")
    parser$add_argument("--samples-csv", required = TRUE)
    parser$add_argument("--working-dir", required = TRUE)
    parser$add_argument("--save-dir", required = TRUE)
    parser$add_argument("--cohort-label", required = TRUE)
    parser$add_argument("--gene-score-dir", default = "GeneScoreMatrix")
    parser$add_argument("--export-mode", choices = c("full", "per-arrow"), required = TRUE)
    parser$add_argument("--threads", type = "integer", default = 30)
    parser$add_argument(
        "--macs2-path",
        default = "/data/yosef3/scratch/ruchir/tools/mambaforge/envs/r-env/bin/macs2"
    )
    return(parser$parse_args())
}


run_archr_workflow <- function(samples, args) {
    addArchRThreads(threads = args$threads)
    addArchRGenome("hg38")

    arrow_files <- createArrowFiles(
        inputFiles = samples$fragments_path,
        sampleNames = samples$library_id,
        minTSS = 4,
        minFrags = 1000,
        addTileMat = TRUE,
        addGeneScoreMat = TRUE
    )

    proj <- ArchRProject(
        ArrowFiles = arrow_files,
        outputDirectory = args$working_dir,
        copyArrows = FALSE
    )

    proj <- addIterativeLSI(
        ArchRProj = proj,
        useMatrix = "TileMatrix",
        name = "IterativeLSI",
        varFeatures = 25000,
        dimsToUse = 1:30
    )

    proj <- addClusters(
        input = proj,
        reducedDims = "IterativeLSI",
        method = "Seurat",
        name = "Clusters"
    )

    proj <- addGroupCoverages(
        ArchRProj = proj,
        groupBy = "Clusters",
        force = FALSE
    )

    proj <- addReproduciblePeakSet(
        ArchRProj = proj,
        groupBy = "Clusters",
        pathToMacs2 = args$macs2_path,
        reproducibility = "2"
    )

    proj <- addPeakMatrix(proj)
    proj <- saveArchRProject(ArchRProj = proj, outputDirectory = args$save_dir)
    return(proj)
}


write_shared_outputs <- function(proj, cohort_label) {
    write.table(data.frame(getPeakSet(proj)), paste0(cohort_label, "_pmat_peaks.tsv"))
    write.table(getCellColData(proj), paste0(cohort_label, "_cell_metadata.tsv"))
}


export_full_peak_matrix <- function(proj, cohort_label) {
    pmat <- getMatrixFromProject(proj, useMatrix = "PeakMatrix")
    writeMM(assay(pmat), paste0(cohort_label, "_pmat.mtx"))
    write.table(colnames(pmat), paste0(cohort_label, "_pmat_cells.tsv"))
}


export_per_arrow_peak_matrices <- function(proj) {
    arrow_files <- getArrowFiles(proj)
    for (name in names(arrow_files)) {
        message("Exporting per-arrow peak matrix for: ", name)
        pmat <- getMatrixFromArrow(
            ArrowFile = arrow_files[[name]],
            useMatrix = "PeakMatrix",
            ArchRProj = proj,
            logFile = createLogFile("getMatrixFromArrow")
        )
        writeMM(assay(pmat), paste0(name, "_pmat.mtx"))
        write.table(colnames(pmat), paste0(name, "_cells.tsv"))
    }
}


write_gene_score_matrix <- function(gene_score_matrix, output_dir) {
    if (!dir.exists(output_dir)) {
        dir.create(output_dir, recursive = TRUE)
    }

    fwrite(
        summary(assay(gene_score_matrix)),
        file = file.path(output_dir, "gene_score_ijx.tsv"),
        sep = "\t",
        quote = FALSE,
        row.names = FALSE
    )

    fwrite(
        as.data.frame(rowData(gene_score_matrix)),
        file = file.path(output_dir, "row_data.tsv"),
        sep = "\t",
        quote = FALSE,
        row.names = FALSE
    )

    fwrite(
        as.data.frame(colnames(gene_score_matrix)),
        file = file.path(output_dir, "colnames.tsv"),
        sep = "\t",
        quote = FALSE,
        row.names = FALSE,
        col.names = FALSE
    )
}


export_full_gene_score_matrix <- function(proj, output_dir) {
    gene_score_matrix <- getMatrixFromProject(
        ArchRProj = proj,
        useMatrix = "GeneScoreMatrix"
    )
    write_gene_score_matrix(gene_score_matrix, output_dir)
}


export_per_arrow_gene_score_matrices <- function(proj, output_dir) {
    arrow_files <- getArrowFiles(proj)
    for (name in names(arrow_files)) {
        message("Exporting per-arrow gene score matrix for: ", name)
        gene_score_matrix <- getMatrixFromArrow(
            ArrowFile = arrow_files[[name]],
            useMatrix = "GeneScoreMatrix",
            ArchRProj = proj
        )
        write_gene_score_matrix(gene_score_matrix, file.path(output_dir, name))
    }
}


main <- function() {
    args <- parse_args()
    samples <- read.csv(args$samples_csv)

    proj <- run_archr_workflow(samples, args)
    write_shared_outputs(proj, args$cohort_label)

    if (args$export_mode == "full") {
        export_full_peak_matrix(proj, args$cohort_label)
        export_full_gene_score_matrix(proj, args$gene_score_dir)
    } else {
        export_per_arrow_peak_matrices(proj)
        export_per_arrow_gene_score_matrices(proj, args$gene_score_dir)
    }
}


main()
