"""Train the mrVI model used for cross-vaccine single-cell RNA integration."""

import json
import os
from argparse import ArgumentParser, BooleanOptionalAction

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import anndata
import scanpy as sc
from scvi.external import MRVI

LOW_QUALITY_CELLTYPES = ("LQC", "unk", "dead", "Dead")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("input_h5ad_path")
    parser.add_argument("--output-dir", default="outputs/mrvi")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-epochs", type=int, default=400)
    parser.add_argument("--n-hvg", type=int, default=4000)
    parser.add_argument("--hvg-flavor", default="seurat_v3")
    parser.add_argument("--batch-aware-hvg", action=BooleanOptionalAction, default=True)
    parser.add_argument("--min-frac-cells-per-vax", type=float, default=1e-3)
    parser.add_argument("--n-latent", type=int, default=30)
    parser.add_argument("--n-latent-u", type=int, default=10)
    parser.add_argument("--u-prior-mixture-k", type=int, default=20)
    parser.add_argument("--sample-key", default="donor")
    parser.add_argument("--batch-key", default="vaccine")
    return parser.parse_args()


def remove_low_quality_cells(adata):
    return adata[~adata.obs["celltype"].isin(LOW_QUALITY_CELLTYPES)].copy()


def remove_mitochondrial_genes(adata):
    adata = adata[:, ~adata.var_names.str.startswith("MT-")].copy()
    adata = adata[:, ~adata.var_names.str.startswith("mt-")].copy()
    return adata


def filter_genes_per_vaccine(adata, min_frac_cells: float):
    adatas = []
    for vaccine in adata.obs["vaccine"].unique():
        vaccine_adata = adata[adata.obs["vaccine"] == vaccine].copy()
        min_cells = int(min_frac_cells * vaccine_adata.shape[0])
        sc.pp.filter_genes(vaccine_adata, min_cells=min_cells)
        adatas.append(vaccine_adata)
    return anndata.concat(adatas, join="inner", merge="same")


def select_hvgs(adata, n_hvg: int, hvg_flavor: str, batch_aware_hvg: bool, batch_key: str):
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_hvg,
        flavor=hvg_flavor,
        batch_key=batch_key if batch_aware_hvg else None,
    )
    return adata[:, adata.var["highly_variable"]].copy()


def write_config(args):
    config = vars(args).copy()
    config.update(
        {
            "labels_key": None,
            "scale_observations": False,
            "early_stopping": True,
            "low_quality_celltypes": list(LOW_QUALITY_CELLTYPES),
        }
    )
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    write_config(args)

    adata = sc.read_h5ad(args.input_h5ad_path)
    adata = remove_low_quality_cells(adata)
    adata = remove_mitochondrial_genes(adata)
    adata = filter_genes_per_vaccine(adata, args.min_frac_cells_per_vax)
    adata = select_hvgs(
        adata,
        args.n_hvg,
        args.hvg_flavor,
        args.batch_aware_hvg,
        args.batch_key,
    )

    adata.obs[args.sample_key] = adata.obs[args.sample_key].astype("category")
    adata.obs[args.batch_key] = adata.obs[args.batch_key].astype("category")
    adata.obs["sample"] = adata.obs["donor"].astype(str) + ":" + adata.obs["time"].astype(str)

    MRVI.setup_anndata(
        adata,
        sample_key=args.sample_key,
        batch_key=args.batch_key,
        labels_key=None,
    )
    print(f"Training model with input data shape: {adata.shape}")

    model = MRVI(
        adata,
        u_prior_mixture_k=args.u_prior_mixture_k,
        n_latent=args.n_latent,
        n_latent_u=args.n_latent_u,
        scale_observations=False,
    )
    model.train(
        devices=args.gpu,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        early_stopping=True,
    )
    model.save(os.path.join(args.output_dir, "model"))

    # Compute latent representations
    adata.obsm["u"] = model.get_latent_representation()
    adata.obsm["z"] = model.get_latent_representation(give_z=True)

    # Run UMAP on u
    sc.pp.neighbors(adata, use_rep="u", key_added="neighbors_u")
    sc.tl.umap(adata, min_dist=0.2, neighbors_key="neighbors_u")
    adata.obsm["X_umap_u"] = adata.obsm["X_umap"].copy()

    # Run UMAP on z
    sc.pp.neighbors(adata, use_rep="z", key_added="neighbors_z")
    sc.tl.umap(adata, min_dist=0.2, neighbors_key="neighbors_z")
    adata.obsm["X_umap_z"] = adata.obsm["X_umap"].copy()
    del adata.obsm["X_umap"]

    # Save adata
    adata.write(os.path.join(args.output_dir, "mrvi.h5ad"))

    # Save model history
    history = model.history["elbo_train"].copy()
    history["elbo_validation"] = model.history["elbo_validation"]
    history.to_csv(os.path.join(args.output_dir, "model_history.csv"))


if __name__ == "__main__":
    main()
