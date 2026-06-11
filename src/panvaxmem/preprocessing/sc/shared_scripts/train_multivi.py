"""Train a multiVI model."""

import os
from argparse import ArgumentParser

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import scanpy as sc
import scvi


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input-h5ad", required=True)
    parser.add_argument("--output-h5ad", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--history-csv", required=True)
    parser.add_argument("--min-frac-cells", type=float, default=0.01)
    return parser.parse_args()


def main():
    args = parse_args()

    adata = sc.read_h5ad(args.input_h5ad)
    assert "modality" in adata.var.columns
    assert "modality" in adata.obs.columns
    assert "donor" in adata.obs.columns
    adata.obs["donor"] = adata.obs["donor"].astype(str)

    adata = adata[:, adata.var["modality"].argsort()].copy()
    sc.pp.filter_genes(adata, min_cells=int(args.min_frac_cells * adata.shape[0]))

    scvi.model.MULTIVI.setup_anndata(
        adata,
        batch_key="modality",
        categorical_covariate_keys=["donor"],
    )
    print(f"Training model with input data shape: {adata.shape}")

    model = scvi.model.MULTIVI(
        adata,
        n_genes=(adata.var["modality"] == "Gene Expression").sum(),
        n_regions=(adata.var["modality"] == "Peaks").sum(),
    )
    model.train(early_stopping=True, check_val_every_n_epoch=1)
    model.save(args.model_dir)

    history = model.history["elbo_train"]
    history["elbo_validation"] = model.history["elbo_validation"]
    history.to_csv(args.history_csv)

    adata.obsm["X_mvi"] = model.get_latent_representation()
    sc.pp.neighbors(adata, use_rep="X_mvi")
    sc.tl.umap(adata, min_dist=0.2)
    adata.write(args.output_h5ad, compression="gzip")


if __name__ == "__main__":
    main()
