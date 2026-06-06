# Single-cell preprocessing

This directory contains preprocessing code for single-cell and single-nucleus
datasets used in the manuscript. The primary outputs are processed `.h5ad`
objects and per-cell metadata tables. The scVI models trained are also used for
downstream RNA-seq analyses.

## Dataset overview

| Cohort | Modality | Raw processing | Model | Annotation source | Primary outputs |
|---|---|---|---|---|---|
| H5N1 +/- AS03 | scRNA-seq | cellranger | scVI | scVI clusters annotated by majority vote from legacy RNA labels | `rna.h5ad`, `rna_meta.csv` |
| H5N1 +/- AS03 | scATAC-seq | cellranger-atac + ArchR | peakVI | peakVI clusters annotated by majority vote from legacy ATAC labels | `atac.h5ad`, `atac_meta.csv` |
| TIV | scRNA-seq | cellranger | scVI | scVI clusters annotated by majority vote from legacy RNA labels | `rna.h5ad`, `rna_meta.csv` |
| TIV | scATAC-seq | cellranger-atac + ArchR | peakVI | peakVI clusters annotated by majority vote from legacy ATAC labels | `atac.h5ad`, `atac_meta.csv` |
| BNT162b2 | scRNA-seq | cellranger | scVI | scVI clusters annotated by majority vote from legacy RNA labels | `rna.h5ad`, `rna_meta.csv` |
| BNT162b2 | scATAC-seq | cellranger-atac + ArchR | peakVI | peakVI clusters annotated by majority vote from legacy ATAC labels | `atac.h5ad`, `atac_meta.csv` |
| YFV | snRNA-seq (from multiome) | cellranger-arc | multiVI; scVI also trained but only for DEG computation | multiVI clusters manually annotated, copied to the RNA component | `rna.h5ad`, `rna_meta.csv` |
| YFV | snATAC-seq (from multiome) | cellranger-arc + ArchR | multiVI | multiVI clusters manually annotated, copied to the ATAC component | `atac.h5ad`, `atac_meta.csv` |
| SHX | snRNA-seq (from multiome) | cellranger-arc | multiVI; scVI also trained but only for DEG computation | multiVI clusters manually annotated, copied to the RNA component | `rna.h5ad`, `rna_meta.csv` |
| SHX | snATAC-seq (from multiome) | cellranger-arc + ArchR | multiVI | multiVI clusters manually annotated, copied to the ATAC component | `atac.h5ad`, `atac_meta.csv` |
| TIV2 | mtscATAC-seq on PBMCs | cellranger-atac with special genome build + ArchR | peakVI | peakVI clusters manually annotated | `pbmc_atac.h5ad` |
| TIV2 | mtscATAC-seq on BMMCs | cellranger-atac with special genome build + ArchR | peakVI | peakVI clusters manually annotated | `bm_atac.h5ad` |
