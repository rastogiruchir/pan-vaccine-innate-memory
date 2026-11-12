"""This script was contributed by Maike Nägele.

Compare classical-monocyte subcluster abundances over time between SHX1
(reference) and SHX2 (query).

Call it with:
    python compute_cluster_abundance.py \
        --reference-path **path to reference AnnData object** \
        --query-path **path to labeled SHX2 query AnnData object** \
        --output-dir **output folder** \
        --plot-clusters "C1 (Refractory)" "C2 (Pro-inflammatory)"
"""

import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

matplotlib_rc = {"pdf.fonttype": 42}
plt.rcParams.update(matplotlib_rc)

clusters = ["C0 (ISG+)", "C1 (Refractory)", "C2 (Pro-inflammatory)", "C3 (Neutral)", "C4 (DC-like)"]
cluster_colors = dict(zip(clusters, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]))
dataset_colors = {"reference": "#c3b678", "query": "#f4a259"}
dataset_labels = {"reference": "SHX1", "query": "SHX2"}
timepoints = {"reference": ["D000", "D060", "D090", "D180"], "query": ["D000", "D090", "D180"]}
query_visit_to_day = {"V1": "D000", "V4": "D090", "V7": "D180"}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--reference-path",
        required=True,
        help=(
            "Path to the reference AnnData containing 'donor', 'time', 'vaccine', "
            "and annotated 'subcluster' columns in obs."
        ),
    )
    parser.add_argument(
        "--query-path",
        required=True,
        help="Path to the labeled SHX2 query AnnData containing 'predicted_subcluster' in obs.",
    )
    parser.add_argument("--reference-vaccine", default="SHX")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--plot-clusters", nargs="+", default=["C1 (Refractory)", "C2 (Pro-inflammatory)"])
    return parser.parse_args()


def load_obs(reference_path, query_path, reference_vaccine):
    ref_obs = sc.read_h5ad(reference_path, backed="r").obs
    ref_obs = ref_obs[ref_obs["vaccine"] == reference_vaccine].copy()

    query_obs = sc.read_h5ad(query_path, backed="r").obs.copy()
    query_obs["donor"] = query_obs["donor"].astype(str).str.replace(r"_V\d+$", "", regex=True)
    query_obs["time"] = query_obs["time"].astype(str).replace(query_visit_to_day)
    query_obs["subcluster"] = query_obs["predicted_subcluster"]
    return ref_obs, query_obs


def calculate_cluster_abundance(obs):
    x = obs[["donor", "time", "subcluster"]].dropna().astype(str)
    x = x.groupby(["donor", "time", "subcluster"]).size().reset_index(name="n_cells")
    x["relative_abundance"] = x["n_cells"] / x.groupby(["donor", "time"])["n_cells"].transform("sum")

    base = x[x["time"] == "D000"][["donor", "subcluster", "relative_abundance"]]
    x = x.merge(base.rename(columns={"relative_abundance": "baseline_abundance"}), on=["donor", "subcluster"], how="left")
    x["delta_abundance"] = x["relative_abundance"] - x["baseline_abundance"]
    return x


def paired_baseline_tests(abundance, timepoints):
    """Paired Wilcoxon test of each cluster vs. baseline per timepoint, BH-adjusted across all tests."""
    baseline, followups = timepoints[0], timepoints[1:]
    rows = []
    for cluster in clusters:
        wide = abundance[abundance["subcluster"] == cluster].pivot(index="donor", columns="time", values="relative_abundance")
        for timepoint in followups:
            paired = wide.reindex(columns=[baseline, timepoint]).dropna()
            try:
                stat, pval = wilcoxon(paired[baseline], paired[timepoint])
            except ValueError:  # too few donors or all differences zero
                stat, pval = np.nan, np.nan
            rows.append({"cluster": cluster, "timepoint": timepoint, "n_donors": len(paired),
                         "statistic": stat, "pvalue": pval})

    results = pd.DataFrame(rows)
    valid = results["pvalue"].notna()
    results.loc[valid, "pvalue_adj"] = multipletests(results.loc[valid, "pvalue"], method="fdr_bh")[1]
    return results


def pooled_d000_vs_d180(abundance_by_dataset):
    paired = (pd.concat(abundance_by_dataset, names=["dataset"]).reset_index("dataset")
              .pivot_table(index=["dataset", "donor", "subcluster"], columns="time", values="relative_abundance")
              [["D000", "D180"]].dropna().reset_index())
    pvalues = paired.groupby("subcluster").apply(lambda d: wilcoxon(d["D000"], d["D180"])[1])
    p_adj = pd.Series(multipletests(pvalues, method="fdr_bh")[1], index=pvalues.index)
    return paired, p_adj


def plot_baseline_composition(abundance_by_dataset, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)
    for ax, (dataset, abundance) in zip(axes, abundance_by_dataset.items()):
        comp = (abundance[abundance["time"] == "D000"]
                .pivot(index="donor", columns="subcluster", values="relative_abundance")
                .reindex(columns=clusters, fill_value=0).fillna(0))
        comp.plot(kind="bar", stacked=True, ax=ax, color=cluster_colors, width=0.8, edgecolor="white", linewidth=0.4, legend=False)
        ax.set(title=dataset_labels[dataset], xlabel="", ylim=(0, 1))
        ax.tick_params(axis="x", labelsize=7)
    axes[0].set_ylabel("Proportion of cells")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=7)
    sns.despine(fig=fig)
    fig.savefig(os.path.join(output_dir, "cluster_abundance_D000.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_delta_trajectories(abundance_by_dataset, results_by_dataset, output_dir):
    fig, axes = plt.subplots(len(abundance_by_dataset), len(clusters), figsize=(15, 5.2), sharey="col", squeeze=False)
    for row, (dataset, abundance) in enumerate(abundance_by_dataset.items()):
        tps = timepoints[dataset]
        x = np.arange(len(tps))
        results = results_by_dataset[dataset]

        for col, cluster in enumerate(clusters):
            ax = axes[row, col]
            wide = (abundance[abundance["subcluster"] == cluster]
                    .pivot(index="donor", columns="time", values="delta_abundance").reindex(columns=tps))
            for _, values in wide.iterrows():
                ax.plot(x, values, color=dataset_colors[dataset], lw=1.8, marker="o", ms=4, alpha=0.75)
            ax.axhline(0, color="0.4", lw=0.8, ls="--", zorder=0)

            p_adj = results[results["cluster"] == cluster].set_index("timepoint")["pvalue_adj"]
            for i, tp in enumerate(tps[1:], start=1):
                if pd.notna(p_adj.get(tp)):
                    ax.text(i, 1.01, f"p_adj = {p_adj[tp]:.2f}", transform=ax.get_xaxis_transform(), ha="center",
                            va="bottom", fontsize=5, fontweight="bold" if p_adj[tp] < 0.05 else "normal")

            ax.set_xticks(x, tps, fontsize=6)
            ax.tick_params(axis="y", labelsize=6)
            ax.set_title(cluster if row == 0 else "", fontsize=8, pad=14)
            ax.set_ylabel(dataset_labels[dataset] if col == 0 else "", fontsize=8)

    fig.supylabel("Δ Proportion of cells (vs. baseline)", fontsize=8)
    sns.despine(fig=fig)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "delta_abundance_trajectories.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_d000_vs_d180(paired, p_adj, plot_clusters, output_dir):
    fig, axes = plt.subplots(len(plot_clusters), 1, figsize=(3, 3 * len(plot_clusters)), squeeze=False)
    for ax, cluster in zip(axes[:, 0], plot_clusters):
        df = paired[paired["subcluster"] == cluster]
        for _, donor in df.iterrows():
            ax.plot([0, 1], [donor["D000"], donor["D180"]], color=dataset_colors[donor["dataset"]],
                    lw=1.8, marker="o", ms=4, alpha=0.75)
        ax.plot([0, 1], [df["D000"].median(), df["D180"].median()], color="black", lw=3, marker="o", ms=6)

        ax.text(0.5, 1.0, f"p_adj = {p_adj[cluster]:.3f}", transform=ax.transAxes, ha="center", va="bottom",
                fontsize=7, fontweight="bold" if p_adj[cluster] < 0.05 else "normal")
        ax.set_title(cluster, fontsize=9, pad=14)
        ax.set_xticks([0, 1], ["D000", "D180"])
        ax.set_xlim(-0.3, 1.3)
        ax.set_ylabel("Proportion of cells")
    sns.despine(fig=fig)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "abundance_D000_vs_D180.pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    ref_obs, query_obs = load_obs(args.reference_path, args.query_path, args.reference_vaccine)
    abundance_by_dataset = {
        "reference": calculate_cluster_abundance(ref_obs),
        "query": calculate_cluster_abundance(query_obs),
    }
    results_by_dataset = {
        dataset: paired_baseline_tests(abundance, timepoints[dataset])
        for dataset, abundance in abundance_by_dataset.items()
    }
    paired, pooled_p_adj = pooled_d000_vs_d180(abundance_by_dataset)

    # Save tables
    pd.concat(abundance_by_dataset, names=["dataset"]).reset_index("dataset").to_csv(
        os.path.join(args.output_dir, "cluster_abundance.csv"), index=False)
    pd.concat(results_by_dataset, names=["dataset"]).reset_index("dataset").to_csv(
        os.path.join(args.output_dir, "wilcoxon_vs_baseline.csv"), index=False)
    pooled_p_adj.rename("pvalue_adj").to_csv(os.path.join(args.output_dir, "wilcoxon_pooled_D000_vs_D180.csv"))

    # Plots
    plot_baseline_composition(abundance_by_dataset, args.output_dir)
    plot_delta_trajectories(abundance_by_dataset, results_by_dataset, args.output_dir)
    plot_d000_vs_d180(paired, pooled_p_adj, args.plot_clusters, args.output_dir)


if __name__ == "__main__":
    main()
