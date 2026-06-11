"""Analyze clonal heterogeneity in TIV2 bone-marrow mtscATAC-seq cells.

This script creates two figures from A5 clone calls and z-scored chromVAR
deviations: a PCA plot colored by clone and a clone-label permutation test.
"""

import argparse
from pathlib import Path

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

mpl.rcParams["pdf.fonttype"] = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze A5 clonal chromVAR heterogeneity.")
    # fmt: off
    parser.add_argument("--input-h5ad", type=Path, default=Path("../../../data/processed-single-cell-h5ad/TIV2/bm_atac.h5ad"))
    parser.add_argument("--deviation-scores-csv", type=Path, default=Path("../atac_analysis/chromvar/outputs/deviation_scores_jaspar2016/tiv2_bmmc.csv"))
    parser.add_argument("--clones-csv", type=Path, default=Path("outputs/clonal_calls/A5/clones.csv"))
    parser.add_argument("--sample-id", default="p24243-s005_A5")
    parser.add_argument("--clone-col", default="alleles_snn_res.1.5")
    parser.add_argument("--n-variable-motifs", type=int, default=50)
    parser.add_argument("--min-cells-per-celltype", type=int, default=50)
    parser.add_argument("--min-cells-per-clone", type=int, default=25)
    parser.add_argument("--n-null-permutations", type=int, default=100)
    parser.add_argument("--output-pca-path", type=Path, default=Path("plotting/figures/A5_chromvar_pca_by_clones.pdf"))
    parser.add_argument("--output-permutation-path", type=Path, default=Path("plotting/figures/A5_clone_pvalues_vs_null.pdf"))
    # fmt: on
    return parser.parse_args()


def load_inputs(
    input_h5ad: Path,
    deviation_scores_csv: Path,
    clones_csv: Path,
    sample_id: str,
    clone_col: str,
) -> tuple[ad.AnnData, pd.DataFrame]:
    adata = ad.read_h5ad(input_h5ad)

    deviations_df = pd.read_csv(deviation_scores_csv, index_col=0).T
    assert adata.obs_names.equals(deviations_df.index)

    clones_df = pd.read_csv(clones_csv, index_col=0)
    clones_df.index = f"{sample_id}#" + clones_df.index.astype(str)
    clone_labels = "C" + clones_df[clone_col].astype(int).astype(str)
    adata.obs["clone"] = clone_labels.reindex(adata.obs_names)

    # Don't subset to only cells with clone calls, since we use the full set of cells to determine
    # highly variable motifs.
    return adata, deviations_df


def get_eligible_celltypes(adata, min_cells_per_celltype: int) -> list[str]:
    clone_obs = adata.obs[adata.obs["clone"].notna()]
    celltype_counts = clone_obs["celltype"].value_counts()
    return sorted(celltype_counts[celltype_counts >= min_cells_per_celltype].index)


def get_common_clones(obs: pd.DataFrame, celltype: str, min_cells_per_clone: int) -> list[str]:
    clone_counts = obs.loc[obs["celltype"] == celltype, "clone"].value_counts()
    clones = clone_counts[clone_counts >= min_cells_per_clone].index.tolist()
    return sorted(clones, key=lambda clone: int(clone[1:]))


def get_most_variable_motifs(
    obs: pd.DataFrame,
    deviations_df: pd.DataFrame,
    celltype: str,
    n_motifs: int,
) -> list[str]:
    celltype_obs = obs[obs["celltype"] == celltype]
    variances = deviations_df.loc[celltype_obs.index].var(axis=0)
    return variances.sort_values(ascending=False).index[:n_motifs].tolist()


def get_clone_palette(adata) -> dict[str, tuple]:
    clones = pd.unique(adata.obs.loc[adata.obs["clone"].notna(), "clone"])
    return dict(zip(clones, sns.color_palette("tab20", n_colors=len(clones))))


def plot_pca_panel(
    adata,
    deviations_df: pd.DataFrame,
    celltypes: list[str],
    n_variable_motifs: int,
    min_cells_per_clone: int,
    output_path: Path,
):
    clone_palette = get_clone_palette(adata)
    fig, axs = plt.subplots(ncols=len(celltypes), figsize=(3.5 * len(celltypes), 3))
    axs = np.atleast_1d(axs)

    for ax, celltype in zip(axs, celltypes):
        clones = get_common_clones(adata.obs, celltype, min_cells_per_clone)
        motifs = get_most_variable_motifs(adata.obs, deviations_df, celltype, n_variable_motifs)
        adata_subset = adata[
            (adata.obs["celltype"] == celltype) & (adata.obs["clone"].isin(clones))
        ].copy()
        X = deviations_df.loc[adata_subset.obs_names, motifs].values
        X_pca = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X))
        adata_subset.obsm["X_pca"] = X_pca
        sc.pl.pca(
            adata_subset,
            color=["clone"],
            frameon=False,
            title=celltype,
            palette=clone_palette,
            size=15,
            show=False,
            ax=ax,
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def compute_clone_pvalues(X: np.ndarray, clones: np.ndarray) -> np.ndarray:
    pvalues = []
    for clone in np.unique(clones):
        mask = clones == clone
        pvalues.extend(ttest_ind(X[mask], X[~mask], axis=0).pvalue)
    return np.array(pvalues)


def compare_clonal_deviations_to_null(
    obs: pd.DataFrame,
    deviations_df: pd.DataFrame,
    celltype: str,
    motifs: list[str],
    min_cells_per_clone: int,
    n_null_permutations: int,
) -> tuple[np.ndarray, np.ndarray]:
    clones = get_common_clones(obs, celltype, min_cells_per_clone)
    obs = obs[(obs["celltype"] == celltype) & (obs["clone"].isin(clones))]
    X = deviations_df.loc[obs.index, motifs].values
    clones = obs["clone"].values

    observed_pvalues = compute_clone_pvalues(X, clones)
    null_pvalues = []
    for seed in tqdm(range(n_null_permutations), desc=f"Null permutations: {celltype}"):
        shuffled_clones = obs["clone"].sample(frac=1, random_state=seed).values
        null_pvalues.append(compute_clone_pvalues(X, shuffled_clones))
    return observed_pvalues, np.concatenate(null_pvalues)


def plot_permutation_panel(
    obs: pd.DataFrame,
    deviations_df: pd.DataFrame,
    celltypes: list[str],
    n_variable_motifs: int,
    min_cells_per_clone: int,
    n_null_permutations: int,
    output_path: Path,
) -> None:
    fig, axs = plt.subplots(ncols=len(celltypes), figsize=(3.5 * len(celltypes), 3))
    axs = np.atleast_1d(axs)

    for i, (ax, celltype) in enumerate(zip(axs, celltypes)):
        motifs = get_most_variable_motifs(obs, deviations_df, celltype, n_variable_motifs)
        observed_pvalues, null_pvalues = compare_clonal_deviations_to_null(
            obs,
            deviations_df,
            celltype,
            motifs,
            min_cells_per_clone,
            n_null_permutations,
        )
        observed_y = -np.log10(np.sort(observed_pvalues)[::-1])
        null_y = -np.log10(np.sort(null_pvalues)[::-1])
        observed_x = np.arange(1, len(observed_y) + 1) / len(observed_y) * 100
        null_x = np.arange(1, len(null_y) + 1) / len(null_y) * 100

        ax.plot(observed_x, observed_y, label=r"$p$-values from observed clones", color="blue")
        ax.plot(null_x, null_y, label=r"$p$-values from permuted clones", color="orange")
        if i == 0:
            ax.set_ylabel(r"$-\log_{10}(p)$")
            ax.legend(frameon=False)
        else:
            ax.legend().remove()
        ax.set_title(celltype)

    fig.supxlabel(r"Percentile of $p$-value")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    adata, deviations_df = load_inputs(
        args.input_h5ad,
        args.deviation_scores_csv,
        args.clones_csv,
        args.sample_id,
        args.clone_col,
    )
    celltypes = get_eligible_celltypes(adata, args.min_cells_per_celltype)

    plot_pca_panel(
        adata,
        deviations_df,
        celltypes,
        n_variable_motifs=args.n_variable_motifs,
        min_cells_per_clone=args.min_cells_per_clone,
        output_path=args.output_pca_path,
    )
    plot_permutation_panel(
        adata.obs,
        deviations_df,
        celltypes,
        n_variable_motifs=args.n_variable_motifs,
        min_cells_per_clone=1,
        n_null_permutations=args.n_null_permutations,
        output_path=args.output_permutation_path,
    )


if __name__ == "__main__":
    main()
