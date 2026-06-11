"""Run Hotspot on the full combined CM scRNA-seq counts matrix.

The mrVI `u` embedding is used to create the kNN graph used by Hotspot.
"""

import os
from argparse import ArgumentParser

import hotspot
import matplotlib.pyplot as plt
import scanpy as sc
from anndata import AnnData


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("full_adata_path")
    parser.add_argument("--output-dir", default="outputs/hotspot")
    parser.add_argument("--mrvi-adata-path", default="outputs/mrvi/mrvi.h5ad")
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--top-n-genes", type=int, default=1000)
    parser.add_argument("--min-genes-per-module", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=40)
    return parser.parse_args()


def prepare_anndata(full_adata_path: str, mrvi_adata_path: str) -> AnnData:
    adata = sc.read_h5ad(full_adata_path)
    adata.layers["counts_csc"] = adata.X.tocsc()
    adata.obs["total_counts"] = adata.X.sum(axis=1).A1

    sc.pp.filter_genes(adata, min_cells=1)

    mrvi_adata = sc.read_h5ad(mrvi_adata_path)
    assert set(adata.obs_names) == set(mrvi_adata.obs_names)
    mrvi_adata = mrvi_adata[adata.obs_names].copy()
    adata.obsm["u"] = mrvi_adata.obsm["u"].copy()

    return adata


def run_hotspot(adata: AnnData, args):
    hs = hotspot.Hotspot(
        adata,
        layer_key="counts_csc",
        model="danb",
        latent_obsm_key="u",
    )

    hs.create_knn_graph(weighted_graph=False, n_neighbors=args.n_neighbors)
    results = hs.compute_autocorrelations(jobs=args.n_jobs)

    top_genes = (
        results.loc[results.FDR < 0.05]
        .sort_values("Z", ascending=False)
        .head(args.top_n_genes)
        .index
    )

    hs.compute_local_correlations(top_genes, jobs=args.n_jobs)
    hs.create_modules(
        min_gene_threshold=args.min_genes_per_module,
        core_only=True,
        fdr_threshold=0.05,
    )
    module_scores = hs.calculate_module_scores()
    hs.plot_local_correlations(vmin=-12, vmax=12)
    return hs, results, module_scores


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    adata = prepare_anndata(args.full_adata_path, args.mrvi_adata_path)
    print(f"Running hotspot on adata with shape: {adata.shape}")
    hs, results, module_scores = run_hotspot(adata, args)

    fig = plt.gcf()
    fig.savefig(os.path.join(args.output_dir, "local_correlations.png"))

    results = results.join(hs.modules)
    results.to_csv(os.path.join(args.output_dir, "gene_results.csv"))

    module_scores.to_csv(os.path.join(args.output_dir, "module_scores.csv"))


if __name__ == "__main__":
    main()
