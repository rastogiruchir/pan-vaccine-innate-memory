> [!NOTE]
> Raw fgsea results not pushed to GitHub because of data size.

1. `combined_tables` contains combined fgsea summary statistics for classical monocytes generated from `../create_combined_fgsea_tables.py`.

2. Vaccine-specific directories (only symlinks) contain donor-corrected and per-donor fgsea results for all celltypes. Below, we note the target of the symlinks:
    - H5N1+AS03 (H5N1_adj): `/data/yosef2/users/ruchir/pan_vaccine_immune_response/scRNA/GSEA/H5N1_reprocessed/adj_non_lvmde_scvi_1.2_reimpl_test_mode_two_delta_0.2`
    - H5N1 (H5N1_nadj): `/data/yosef2/users/ruchir/pan_vaccine_immune_response/scRNA/GSEA/H5N1_reprocessed/nadj_non_lvmde_scvi_1.2_reimpl_test_mode_two_delta_0.2`
    - TIV: `/data/yosef2/users/ruchir/pan_vaccine_immune_response/scRNA/GSEA/TIV_reprocessed/non_lvmde_scvi_1.2_reimpl_test_mode_two_delta_0.2`
    - BNT162b2 (PFZR): `/data/yosef2/users/ruchir/pan_vaccine_immune_response/scRNA/GSEA/PFZR_reprocessed/non_lvmde_scvi_1.2_reimpl_test_mode_two_delta_0.2`
    - YFV: `/data/yosef2/users/ruchir/pan_vaccine_immune_response/scRNA/GSEA/YFV_reprocessed/non_lvmde_scvi_1.2_reimpl_test_mode_two_delta_0.2`
    - SHX: `/data/yosef2/users/ruchir/pan_vaccine_immune_response/scRNA/GSEA/SHX_reprocessed/non_lvmde_scvi_1.2_reimpl_test_mode_two_delta_0.2`
