# JASPAR 2016 chromVAR deviation scores

These files are symlinks to CSVs containing z-scored chromVAR deviations used by
the downstream analyses. The symlink targets record the output provenance directly;
this table records the corresponding input h5ad files used to compute each output.

| Cohort / analysis | Local symlink | Input h5ad |
| --- | --- | --- |
| H5N1 +/- AS03 | `h5n1.csv` | `/data/yosef2/users/ruchir/pan_vaccine_immune_response/compile_sc_data/H5N1/scATAC/h5n1_atac.h5ad` |
| TIV | `tiv.csv` | `/data/yosef2/users/ruchir/pan_vaccine_immune_response/compile_sc_data/TIV/scATAC/tiv_atac.h5ad` |
| BNT162b2 | `pfzr.csv` | `/data/yosef2/users/ruchir/pan_vaccine_immune_response/compile_sc_data/PFZR/scATAC/pfzr_atac.h5ad` |
| YFV | `yfv.csv` | `/data/yosef2/users/ruchir/pan_vaccine_immune_response/compile_sc_data/YFV/scATAC/yfv_atac.h5ad` |
| SHX | `shx.0_01_f.csv` | `/data/yosef2/users/ruchir/pan_vaccine_immune_response/scATAC/chromVAR/SHX/shx_atac_0_01_f.h5ad` |
| Joint CM (combined over the above cohorts; excludes TIV2) | `cm_joint.0_002_f_per_vax.csv` | `/data/yosef2/users/ruchir/pan_vaccine_immune_response/compile_sc_data/ALL/scATAC/only_CM/cm_atac_0_002_f.per_vax.h5ad` |
| TIV2 PBMC | `tiv2_pbmc.csv` | `/data/yosef2/users/ruchir/pan_vaccine_immune_response/compile_sc_data/TIV_addl/pbmc_mtscATAC/tiv_pbmc_atac.with_doublet_removal.h5ad` |
| TIV2 BMMC | `tiv2_bmmc.csv` | `/data/yosef2/users/ruchir/pan_vaccine_immune_response/compile_sc_data/TIV_addl/bm_mtscATAC/tiv_bm_atac.with_doublet_removal.h5ad` |
