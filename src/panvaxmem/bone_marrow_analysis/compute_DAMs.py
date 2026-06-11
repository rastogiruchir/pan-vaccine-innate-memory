"""Compute TIV2 bone-marrow mtscATAC-seq chromVAR DAMs.

The D0 and D28 samples are unpaired, so this script makes two complementary
figures from z-scored chromVAR deviations:
    (1) a conservative pairwise donor-comparison heatmap and
    (2) a less conservative pooled-cell t-test volcano plot.
"""

import argparse
import itertools
import json
from functools import partial
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import textalloc as ta
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm

mpl.rcParams["pdf.fonttype"] = 42


VOLCANO_CELLTYPES = [
    "CMP",
    "GMP (monocyte-bias)",
    "GMP (granulocyte-bias)",
    "MEP",
    "CLP",
    "CM",
]

ALL_CONCORDANT_ALL_SIGNIFICANT = "All concordant and all significant"
ALL_CONCORDANT_MAJORITY_SIGNIFICANT = "All concordant and majority significant"
OTHER_PAIRWISE_RESULT = "Other"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute TIV2 BM chromVAR DAM figures.")
    # fmt: off
    parser.add_argument("--input-h5ad", type=Path, default=Path("../../../data/processed-single-cell-h5ad/TIV2/bm_atac.h5ad"))
    parser.add_argument("--deviation-scores-csv", type=Path, default=Path("../atac_analysis/chromvar/outputs/deviation_scores_jaspar2016/tiv2_bmmc.csv"))
    parser.add_argument("--figures-dir", type=Path, default=Path("plotting/figures"))
    parser.add_argument("--tf-families-path", type=Path, default=Path("../../../resources/tf_families.json"))
    parser.add_argument("--tf-palette-path", type=Path, default=Path("../../../resources/tf_family_palette.json"))
    # fmt: on
    return parser.parse_args()


def clean_motif_names(motifs: list[str]) -> list[str]:
    motifs = [motif.rsplit("_", 1)[1] for motif in motifs]
    return ["SMAD2::3::4" if motif == "SMAD2::SMAD3::SMAD4" else motif for motif in motifs]


def load_transcription_factor_families(
    tf_families_path: Path,
    my_families: list[str] = ["AP-1", "IRF/STAT", "NFkB", "C/EBP", "ETS"],
) -> dict[str, list[str]]:
    all_families = json.load(open(tf_families_path, "r"))
    return {family: all_families[family] for family in my_families}


def get_pairwise_annotation(
    pair_betas: np.ndarray,
    pair_pvalues: np.ndarray,
    aggregate_beta: float,
) -> str:
    all_concordant = (np.sign(pair_betas) == np.sign(aggregate_beta)).all()
    all_significant = (pair_pvalues < 0.05).all()
    majority_significant = (pair_pvalues < 0.05).mean() > 0.5

    if not all_concordant:
        return OTHER_PAIRWISE_RESULT
    if all_significant:
        return ALL_CONCORDANT_ALL_SIGNIFICANT
    if majority_significant:
        return ALL_CONCORDANT_MAJORITY_SIGNIFICANT
    return OTHER_PAIRWISE_RESULT


def compute_pairwise_donor_effects(
    adata,
    deviations_df: pd.DataFrame,
    cluster_col: str,
    exclude_clusters: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert adata.obs_names.equals(deviations_df.index)

    adata = adata[adata.obs[cluster_col].notna()].copy()
    deviations_df = deviations_df.loc[adata.obs_names]

    clusters = [
        cluster
        for cluster in sorted(adata.obs[cluster_col].unique())
        if cluster not in exclude_clusters
    ]
    motifs = deviations_df.columns.tolist()

    betas = np.full((len(clusters), len(motifs)), np.nan, dtype=float)
    annotations = np.full(
        (len(clusters), len(motifs)),
        OTHER_PAIRWISE_RESULT,
        dtype=object,
    )

    for row_idx, cluster in tqdm(
        enumerate(clusters),
        total=len(clusters),
        desc="Pairwise donor comparisons",
    ):
        cluster_obs = adata.obs[adata.obs[cluster_col] == cluster].copy()
        cluster_deviations = deviations_df.loc[cluster_obs.index]
        d0_donors = cluster_obs.loc[cluster_obs["time"] == "D0", "donor"].unique()
        d28_donors = cluster_obs.loc[cluster_obs["time"] == "D28", "donor"].unique()
        donor_masks = {
            donor: cluster_obs["donor"] == donor for donor in set(d0_donors) | set(d28_donors)
        }

        for col_idx, motif in enumerate(motifs):
            motif_scores = cluster_deviations[motif]
            time_means = motif_scores.groupby(cluster_obs["time"], observed=True).mean()
            if "D0" not in time_means or "D28" not in time_means:
                continue
            aggregate_beta = time_means["D28"] - time_means["D0"]
            betas[row_idx, col_idx] = aggregate_beta

            pair_betas = []
            pair_pvalues = []
            for d0_donor, d28_donor in itertools.product(d0_donors, d28_donors):
                d0_mask = donor_masks[d0_donor]
                d28_mask = donor_masks[d28_donor]
                if not (d0_mask.any() and d28_mask.any()):
                    continue
                d0_X = motif_scores[d0_mask].values
                d28_X = motif_scores[d28_mask].values
                pair_betas.append(d28_X.mean() - d0_X.mean())
                pair_pvalues.append(ttest_ind(d0_X, d28_X).pvalue)

            annotations[row_idx, col_idx] = get_pairwise_annotation(
                np.array(pair_betas),
                np.array(pair_pvalues),
                aggregate_beta,
            )

    motifs = clean_motif_names(motifs)
    betas_df = pd.DataFrame(betas, index=clusters, columns=motifs)
    annotations_df = pd.DataFrame(annotations, index=clusters, columns=motifs)
    assert betas_df.isna().sum().sum() == 0

    motifs_to_keep = annotations_df.columns[
        (annotations_df == ALL_CONCORDANT_ALL_SIGNIFICANT).any(axis=0)
    ]
    return betas_df[motifs_to_keep], annotations_df[motifs_to_keep]


def plot_pairwise_donor_heatmap(
    betas_df: pd.DataFrame,
    annotations_df: pd.DataFrame,
    output_path: Path,
):
    annotation_to_marker = {
        ALL_CONCORDANT_ALL_SIGNIFICANT: "★",
        ALL_CONCORDANT_MAJORITY_SIGNIFICANT: "☆",
        OTHER_PAIRWISE_RESULT: "",
    }

    cg = sns.clustermap(
        betas_df,
        annot=annotations_df.map(lambda value: annotation_to_marker[value]),
        fmt="",
        row_cluster=True,
        col_cluster=True,
        cmap="coolwarm",
        center=0,
        vmin=-betas_df.abs().max().max(),
        vmax=betas_df.abs().max().max(),
        figsize=(15, 3.75),
        cbar_kws={"label": r"D0vD28 $\beta$"},
        cbar_pos=(0.1, 0.375, 0.025, 0.4),
        xticklabels=True,
        yticklabels=True,
    )
    cg.ax_row_dendrogram.set_visible(False)
    cg.ax_col_dendrogram.set_visible(False)
    cg.cax.set_ylabel(r"D0vD28 $\beta$", fontsize=14)
    cg.ax_heatmap.tick_params(axis="x", labelsize=12)
    cg.ax_heatmap.tick_params(axis="y", labelsize=14)

    handles = [
        plt.Line2D(
            [],
            [],
            marker="$★$",
            markersize=10,
            linestyle="None",
            color="black",
            label=ALL_CONCORDANT_ALL_SIGNIFICANT,
        ),
        plt.Line2D(
            [],
            [],
            marker="$☆$",
            markersize=10,
            linestyle="None",
            color="black",
            label=ALL_CONCORDANT_MAJORITY_SIGNIFICANT,
        ),
    ]
    legend = cg.figure.legend(
        handles=handles,
        title="Pairwise comparison of D0 and D28 samples",
        loc="upper center",
        bbox_to_anchor=(0.55, -0.15),
        frameon=False,
        ncols=2,
        fontsize=12,
        title_fontsize=14,
    )
    legend.get_title().set_fontstyle("italic")
    cg.figure.savefig(output_path, bbox_inches="tight")
    plt.close(cg.figure)


def compute_ttest_motif_effects(
    adata,
    deviations_df: pd.DataFrame,
    celltype: str,
) -> pd.DataFrame:
    assert adata.obs_names.equals(deviations_df.index)

    adata = adata[adata.obs["celltype"] == celltype].copy()
    deviations_df = deviations_df.loc[adata.obs_names].copy()
    d0_mask = adata.obs["time"] == "D0"
    d28_mask = adata.obs["time"] == "D28"

    records = []
    for motif in deviations_df.columns:
        d0_X = deviations_df.loc[d0_mask, motif].values
        d28_X = deviations_df.loc[d28_mask, motif].values
        records.append(
            {
                "motif": motif,
                "beta": d28_X.mean() - d0_X.mean(),
                "p": ttest_ind(d0_X, d28_X).pvalue,
            }
        )

    results_df = pd.DataFrame(records).set_index("motif")
    results_df["p"] = fdrcorrection(results_df["p"])[1]
    results_df["-log10p"] = -np.log10(results_df["p"])
    results_df["significant"] = results_df["p"] < 0.05
    results_df.index = clean_motif_names(results_df.index.tolist())
    return results_df.sort_values("p")


def plot_ttest_volcano(
    adata,
    deviations_df: pd.DataFrame,
    celltype: str,
    ax: plt.Axes,
    family_to_tfs: dict[str, list[str]],
    family_to_colors: dict[str, str],
    top_k_per: int = 3,
):
    results_df = compute_ttest_motif_effects(adata, deviations_df, celltype)

    results_df["family"] = "Other significant"
    for family, tfs in family_to_tfs.items():
        results_df.loc[results_df.index.isin(tfs), "family"] = family
    results_df.loc[~results_df["significant"], "family"] = "Insignificant"

    sns.scatterplot(
        data=results_df,
        x="beta",
        y="-log10p",
        s=75,
        hue="family",
        palette=family_to_colors,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(celltype, fontsize=28)
    ax.axvline(0, color="red", linestyle="--")
    ax.get_legend().remove()
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16)

    motifs_to_label = set()
    motifs_to_label.update(results_df[results_df["beta"] > 0].sort_values("p").index[:top_k_per])
    motifs_to_label.update(results_df[results_df["beta"] < 0].sort_values("p").index[:top_k_per])

    xs, ys, texts = [], [], []
    for motif in motifs_to_label:
        xs.append(results_df.loc[motif, "beta"])
        ys.append(results_df.loc[motif, "-log10p"])
        texts.append(motif)
    ta.allocate(ax, xs, ys, texts, x_scatter=xs, y_scatter=ys, textsize=18, linecolor="gray")


def plot_ttest_volcano_panel(
    adata,
    deviations_df: pd.DataFrame,
    tf_families_path: Path,
    tf_palette_path: Path,
    output_path: Path,
):
    family_to_tfs = load_transcription_factor_families(tf_families_path)
    family_to_tfs["Other significant"] = []
    family_to_tfs["Insignificant"] = []

    tf_family_palette = json.load(open(tf_palette_path, "r"))
    family_to_colors = {
        family: tf_family_palette[family]
        for family in ["AP-1", "IRF/STAT", "NFkB", "C/EBP", "ETS", "Other significant"]
    }
    family_to_colors["Insignificant"] = "lightgray"

    fig, axs = plt.subplots(ncols=6, nrows=1, figsize=(30, 5), dpi=300)
    plot_fn = partial(
        plot_ttest_volcano,
        adata,
        deviations_df,
        family_to_tfs=family_to_tfs,
        family_to_colors=family_to_colors,
    )
    for ax, celltype in zip(axs, tqdm(VOLCANO_CELLTYPES, desc="T-tests")):
        plot_fn(celltype, ax)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=family,
            markerfacecolor=color,
            markersize=14,
        )
        for family, color in family_to_colors.items()
    ]
    fig.supxlabel(r"D0vD28 $\beta$", fontsize=24, x=0.525)
    fig.supylabel(r"$-\log_{10}p_{\mathrm{adj}}$", fontsize=24, x=0.0)
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=7,
        fontsize=20,
        bbox_to_anchor=(0.5, -0.175),
        frameon=False,
    )
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(args.input_h5ad)
    deviations_df = pd.read_csv(args.deviation_scores_csv, index_col=0).T
    assert adata.obs_names.equals(deviations_df.index)

    # (1) Conservative pairwise test across all D0-vs-D28 donor combinations.
    betas_df, annotations_df = compute_pairwise_donor_effects(
        adata,
        deviations_df,
        cluster_col="celltype",
        exclude_clusters=["Other"],
    )
    plot_pairwise_donor_heatmap(
        betas_df,
        annotations_df,
        args.figures_dir / "DAMs_pairwise_comparison.pdf",
    )

    # (2) Less conservative two-sample t-test pooling cells within each timepoint, irrespective
    #     of donor identity.
    plot_ttest_volcano_panel(
        adata,
        deviations_df,
        args.tf_families_path,
        args.tf_palette_path,
        args.figures_dir / "DAMs_t_test.pdf",
    )


if __name__ == "__main__":
    main()
