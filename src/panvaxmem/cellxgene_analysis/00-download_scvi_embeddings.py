import cellxgene_census

CENSUS_VERSION = "2025-11-08"
OUTPUT_ADATA = "scvi_embeddings.h5ad"


def main():
    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        print("Downloading scVI embeddings...")
        full_adata = cellxgene_census.get_anndata(
            census,
            "homo_sapiens",
            "RNA",
            obs_value_filter="is_primary_data == True",
            var_value_filter="feature_id in ['ENSG00000161798']",  # dummy gene just for code to run
            obs_embeddings=["scvi"],
        )
        print(f"Downloaded embeddings for {full_adata.n_obs} cells")

        full_adata.write(OUTPUT_ADATA)
        print("Embeddings saved to file.")


if __name__ == "__main__":
    main()
