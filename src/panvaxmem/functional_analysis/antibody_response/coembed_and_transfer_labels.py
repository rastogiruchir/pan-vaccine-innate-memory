"""Coembed reference (our dataset) CD14+ cells with Mulè et al. cells using mrVI and then
transfer reference cluster labels to the query cells.

Integration is restricted to the CD14+ monocytes from the H1N1 cohort of the Mulè et al. dataset.
All timepoints are included, though later only D0 cells are used to distinguish between high and
low vaccine responders."""

import os
from argparse import ArgumentParser, BooleanOptionalAction

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import anndata
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import scvi
from scvi.external import MRVI
from sklearn.neighbors import NearestNeighbors

matplotlib.rcParams["pdf.fonttype"] = 42


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--ref-adata-path",
        required=True,
        help=(
            "Path to the reference AnnData, which must already be filtered to highly variable "
            "genes and contain a non-null 'subcluster' column in obs."
        ),
    )
    parser.add_argument(
        "--query-adata-path",
        required=True,
        help="Path to the Mulè et al. query AnnData containing raw RNA counts.",
    )
    parser.add_argument(
        "--query-cohort",
        default="H1N1",
        help="Cohort to retain from the query AnnData.",
    )
    parser.add_argument(
        "--query-celltype",
        default="CD14_Mono",
        help="Cell type to retain from the query AnnData.",
    )
    parser.add_argument(
        "--output-dir",
        default="../outputs/antibody_response/mrvi_coembed",
    )
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-epochs", type=int, default=400)
    parser.add_argument("--n-latent", type=int, default=30)
    parser.add_argument("--n-latent-u", type=int, default=10)
    parser.add_argument("--u-prior-mixture-k", type=int, default=20)
    parser.add_argument("--early-stopping", action=BooleanOptionalAction, default=True)
    parser.add_argument("--n-neighbors", type=int, default=20)
    return parser.parse_args()


def prepare_reference_adata(ref_adata_path: str) -> anndata.AnnData:
    ref_adata = sc.read_h5ad(ref_adata_path)
    assert "subcluster" in ref_adata.obs
    assert ref_adata.obs["subcluster"].notnull().all()
    ref_adata.obs["source"] = "reference"
    ref_adata.obs["adjmfc.group"] = np.nan
    return ref_adata


def prepare_query_adata(
    query_adata_path: str,
    reference_var_names,
    query_cohort: str,
    query_celltype: str,
) -> anndata.AnnData:
    query_adata = sc.read_h5ad(query_adata_path)
    query_adata = query_adata[
        (query_adata.obs["cohort"] == query_cohort)
        & (query_adata.obs["celltype_joint"] == query_celltype)
    ].copy()
    assert query_adata.n_obs > 0
    assert query_adata.obs["adjmfc.group"].notnull().all()

    # Filter genes to match reference
    scvi.model.base._archesmixin._pad_and_sort_query_anndata(
        query_adata,
        reference_var_names,
        inplace=True,
    )

    query_adata.obs["source"] = "query"
    query_adata.obs["donor"] = query_adata.obs["sampleid"].copy()
    query_adata.obs["vaccine"] = query_adata.obs["cohort"].copy()
    query_adata.obs["time"] = query_adata.obs["timepoint"].str.upper()
    query_adata.obs["subcluster"] = np.nan
    return query_adata


def prepare_combined_adata(
    ref_adata_path: str,
    query_adata_path: str,
    query_cohort: str,
    query_celltype: str,
) -> anndata.AnnData:
    ref_adata = prepare_reference_adata(ref_adata_path)
    query_adata = prepare_query_adata(
        query_adata_path,
        ref_adata.var_names,
        query_cohort,
        query_celltype,
    )
    combined_adata = anndata.concat([ref_adata, query_adata], axis="obs")

    expected_obs_cols = {
        "source",
        "donor",
        "vaccine",
        "time",
        "adjmfc.group",
        "subcluster",
    }
    assert expected_obs_cols.issubset(combined_adata.obs.columns)

    combined_adata.obs["donor"] = combined_adata.obs["donor"].astype("category")
    combined_adata.obs["vaccine"] = combined_adata.obs["vaccine"].astype("category")
    combined_adata.obs["sample"] = (
        combined_adata.obs["donor"].astype(str)
        + ":"
        + combined_adata.obs["time"].astype(str)
    )
    return combined_adata


def train_model(combined_adata: anndata.AnnData, args) -> MRVI:
    MRVI.setup_anndata(
        combined_adata,
        sample_key="donor",
        batch_key="vaccine",
        labels_key=None,
    )
    print(f"Training model with input data shape: {combined_adata.shape}")
    print(
        f"Number of reference cells: {(combined_adata.obs['source'] == 'reference').sum()}"
    )
    print(f"Number of query cells: {(combined_adata.obs['source'] == 'query').sum()}")

    model = MRVI(
        combined_adata,
        u_prior_mixture_k=args.u_prior_mixture_k,
        n_latent=args.n_latent,
        n_latent_u=args.n_latent_u,
    )
    model.train(
        devices=args.gpu,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        early_stopping=args.early_stopping,
    )
    return model


def compute_embeddings(combined_adata: anndata.AnnData, model: MRVI) -> None:
    combined_adata.obsm["u"] = model.get_latent_representation()
    combined_adata.obsm["z"] = model.get_latent_representation(give_z=True)

    sc.pp.neighbors(combined_adata, use_rep="u", key_added="neighbors_u")
    sc.tl.umap(combined_adata, min_dist=0.2, neighbors_key="neighbors_u")
    combined_adata.obsm["X_umap_u"] = combined_adata.obsm["X_umap"].copy()

    sc.pp.neighbors(combined_adata, use_rep="z", key_added="neighbors_z")
    sc.tl.umap(combined_adata, min_dist=0.2, neighbors_key="neighbors_z")
    combined_adata.obsm["X_umap_z"] = combined_adata.obsm["X_umap"].copy()
    del combined_adata.obsm["X_umap"]


def transfer_subcluster_labels(
    combined_adata: anndata.AnnData,
    n_neighbors: int,
) -> anndata.AnnData:
    ref_adata = combined_adata[combined_adata.obs["source"] == "reference"].copy()
    query_adata = combined_adata[combined_adata.obs["source"] == "query"].copy()

    # 1. Fit nearest neighbors on reference u embeddings
    nearest_neighbors = NearestNeighbors(n_neighbors=n_neighbors)
    nearest_neighbors.fit(ref_adata.obsm["u"])

    # 2. Find nearest reference neighbors for each query cell
    query_to_ref_neighbors = nearest_neighbors.kneighbors(
        query_adata.obsm["u"],
        return_distance=False,
    )
    assert query_to_ref_neighbors.shape == (query_adata.n_obs, n_neighbors)

    # 3. Extract reference labels
    ref_labels = ref_adata.obs["subcluster"].to_numpy()
    neighbor_labels = ref_labels[
        query_to_ref_neighbors
    ]  # shape: (query_adata.n_obs, n_neighbors)

    # 4. Assign majority label to each query cell
    majority_labels = []
    for labels in neighbor_labels:
        unique_labels, counts = np.unique(labels, return_counts=True)
        majority_labels.append(unique_labels[np.argmax(counts)])

    query_adata.obs["predicted_subcluster"] = majority_labels
    return query_adata


def plot_umap_panel(
    adata: anndata.AnnData,
    color: str,
    title: str,
    ax,
    size=None,
    legend_ncols: int = 1,
) -> None:
    sc.pl.embedding(
        adata,
        basis="X_umap_u",
        color=color,
        size=size,
        show=False,
        ax=ax,
    )
    ax.set(xlabel="UMAP1", ylabel="UMAP2", title=title)
    if ax.get_legend() is not None:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            frameon=False,
            ncols=legend_ncols,
        )
    for collection in ax.collections:
        collection.set_rasterized(True)


def plot_integrated_space(
    combined_adata: anndata.AnnData,
    labeled_query_adata: anndata.AnnData,
    output_dir: str,
) -> None:
    fig, axs = plt.subplots(ncols=3, figsize=(13, 5))
    plot_umap_panel(
        combined_adata,
        color="source",
        title="Integrated space",
        ax=axs[0],
        size=np.where(combined_adata.obs["source"] == "query", 5, 1),
    )
    plot_umap_panel(
        combined_adata[combined_adata.obs["source"] == "reference"],
        color="subcluster",
        title="Reference labels in integrated space",
        ax=axs[1],
        legend_ncols=2,
    )
    plot_umap_panel(
        labeled_query_adata,
        color="predicted_subcluster",
        title="Predicted query labels in integrated space",
        ax=axs[2],
        legend_ncols=2,
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "integrated_space.pdf"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


def write_outputs(
    combined_adata: anndata.AnnData,
    labeled_query_adata: anndata.AnnData,
    model: MRVI,
    output_dir: str,
) -> None:
    model.save(os.path.join(output_dir, "model"))
    combined_adata.write(os.path.join(output_dir, "joint_mrvi.h5ad"))
    labeled_query_adata.write(os.path.join(output_dir, "query_mrvi.labeled.h5ad"))

    history = model.history["elbo_train"].copy()
    history["elbo_validation"] = model.history["elbo_validation"]
    history.to_csv(os.path.join(output_dir, "model_history.csv"))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    combined_adata = prepare_combined_adata(
        args.ref_adata_path,
        args.query_adata_path,
        args.query_cohort,
        args.query_celltype,
    )
    model = train_model(combined_adata, args)
    compute_embeddings(combined_adata, model)
    labeled_query_adata = transfer_subcluster_labels(
        combined_adata,
        args.n_neighbors,
    )
    plot_integrated_space(combined_adata, labeled_query_adata, args.output_dir)
    write_outputs(combined_adata, labeled_query_adata, model, args.output_dir)


if __name__ == "__main__":
    main()
