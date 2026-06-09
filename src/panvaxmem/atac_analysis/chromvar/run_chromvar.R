#!/usr/bin/env Rscript
# Run chromVAR on an ATAC AnnData object and write z-scored motif deviations.

library(argparse)
library(zellkonverter)
library(SingleCellExperiment)

library(chromVAR)
library(motifmatchr)
library(Matrix)
library(BSgenome.Hsapiens.UCSC.hg38)
library(SummarizedExperiment)
library(BiocParallel)

library(TFBSTools)
library(JASPAR2016)


parse_args <- function() {
    parser <- ArgumentParser(description = "Run chromVAR and write z-scored deviations.")
    parser$add_argument("--input-h5ad", required = TRUE, help = "Input ATAC AnnData file.")
    parser$add_argument(
        "--output-csv",
        required = TRUE,
        help = "Output CSV path for chromVAR deviationScores."
    )
    parser$add_argument(
        "--threads",
        type = "integer",
        default = 20,
        help = "Number of threads for BiocParallel. Default: 20."
    )
    parser$add_argument(
        "--p-cutoff",
        type = "double",
        default = 5e-5,
        help = "P-value cutoff passed to motifmatchr::matchMotifs. Default: 5e-5."
    )
    return(parser$parse_args())
}


get_motifs <- function() {
    opts <- list()
    opts["species"] <- "Homo sapiens"
    opts["collection"] <- "CORE"
    out <- TFBSTools::getMatrixSet(JASPAR2016::JASPAR2016, opts)
    if (!isTRUE(all.equal(TFBSTools::name(out), names(out)))) {
        names(out) <- paste(names(out), TFBSTools::name(out), sep = "_")
    }
    return(out)
}


main <- function() {
    args <- parse_args()

    adata <- readH5AD(args$input_h5ad)
    counts <- adata@assays@data$X
    peaks <- rowData(adata)
    ranges <- GRanges(seqnames = peaks$seqnames, IRanges(start = peaks$start, end = peaks$end))

    sre <- SummarizedExperiment(assays = list("counts" = counts), rowRanges = ranges)
    sre <- filterPeaks(sre)
    sre <- addGCBias(sre, genome = BSgenome.Hsapiens.UCSC.hg38)

    motifs <- get_motifs()

    register(MulticoreParam(args$threads))

    motif_ix <- matchMotifs(
        motifs,
        sre,
        genome = BSgenome.Hsapiens.UCSC.hg38,
        p.cutoff = args$p_cutoff
    )
    dev <- computeDeviations(object = sre, annotations = motif_ix)
    write.table(deviationScores(dev), args$output_csv, sep = ",")
}


main()
