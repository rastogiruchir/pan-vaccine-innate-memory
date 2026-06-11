"""Compute differential accessibility over time from scATAC-seq read counts.

For each cell type and peak, read counts are converted to fragment counts and a
poisson GLM is fit across time points:
    n_fragments ~ C(time, Treatment(reference=D0)) + C(donor) + TSSEnrichment,
with total fragments used as the exposure.
"""

import argparse
import os
from collections import defaultdict

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm


def extract_time(time: str) -> int:
    if time.startswith("d") or time.startswith("D"):
        return int(time[1:])
    return int(time)


def convert_reads_to_fragments(adata: anndata.AnnData) -> anndata.AnnData:
    num_ones = (adata.X == 1).sum()
    num_twos = (adata.X == 2).sum()
    assert num_twos > num_ones, "Data should contain read counts instead of fragment counts"

    scvi.data.reads_to_fragments(adata)
    adata.X = adata.layers["fragments"]
    return adata


def compute_poisson_time_effects(
    adata: anndata.AnnData,
    metadata_df: pd.DataFrame,
    output_dir: str,
    min_cells_per_celltype: int = 100,
    max_cells_per_celltype: int = 50_000,
    min_frac_cells_per_peak: float = 0.01,
):
    assert adata.obs_names.equals(
        metadata_df.index
    ), "adata and metadata_df must have the same index"
    os.makedirs(output_dir, exist_ok=True)

    adata = convert_reads_to_fragments(adata)
    metadata_df = metadata_df.copy()
    metadata_df["total_fragments"] = adata.X.sum(axis=1)
    metadata_df["donor"] = metadata_df["donor"].astype("category")
    metadata_df["time"] = metadata_df["time"].astype("category")
    metadata_df["TSSEnrichment"] = metadata_df["TSSEnrichment"].astype(float)

    times = sorted(metadata_df["time"].unique(), key=extract_time)
    baseline_time = times[0]
    assert extract_time(baseline_time) == 0, "Baseline time should be 0"
    comparison_times = times[1:]

    for celltype in tqdm(metadata_df["celltype"].unique(), desc="Celltype"):
        ct_metadata_df = metadata_df[metadata_df["celltype"] == celltype].copy()
        if ct_metadata_df.shape[0] < min_cells_per_celltype:
            print(f"Skipping {celltype} with {ct_metadata_df.shape[0]} cells")
            continue
        if ct_metadata_df.shape[0] > max_cells_per_celltype:
            ct_metadata_df = ct_metadata_df.sample(max_cells_per_celltype, random_state=42, axis=0)
        ct_adata = adata[ct_metadata_df.index].copy()

        sc.pp.filter_genes(
            ct_adata,
            min_cells=int(min_frac_cells_per_peak * ct_adata.shape[0]),
        )
        sc.pp.filter_genes(
            ct_adata,
            max_cells=int((1 - min_frac_cells_per_peak) * ct_adata.shape[0]),
        )
        print(f"Computing DARs for {celltype} with {ct_adata.shape[1]} peaks")

        X = ct_adata.X.toarray().astype(int)
        mean_fragments = np.mean(X, axis=0)

        betas, pvals = defaultdict(list), defaultdict(list)
        formula = (
            "n_fragments ~ "
            f"C(time, Treatment(reference='{baseline_time}')) "
            "+ C(donor) + TSSEnrichment"
        )
        for peak_idx in tqdm(range(X.shape[1]), desc=f"Peaks: {celltype}"):
            try:
                ct_metadata_df["n_fragments"] = X[:, peak_idx]
                model = smf.glm(
                    formula=formula,
                    data=ct_metadata_df,
                    exposure=ct_metadata_df["total_fragments"],
                    family=sm.families.Poisson(),
                )
                results = model.fit(disp=0)
                for comparison_time in comparison_times:
                    param_name = (
                        f"C(time, Treatment(reference='{baseline_time}'))" f"[T.{comparison_time}]"
                    )
                    betas[comparison_time].append(results.params[param_name])
                    pvals[comparison_time].append(results.pvalues[param_name])
            except Exception as e:
                print(f"Error at peak {peak_idx}: {e}")
                for comparison_time in comparison_times:
                    betas[comparison_time].append(np.nan)
                    pvals[comparison_time].append(np.nan)

        for comparison_time in comparison_times:
            output_df = ct_adata.var.copy()
            output_df["region"] = (
                output_df["seqnames"].astype(str)
                + ":"
                + output_df["start"].astype(str)
                + "-"
                + output_df["end"].astype(str)
            )
            output_df = output_df.set_index("region")
            output_df["beta"] = betas[comparison_time]
            output_df["pval"] = pvals[comparison_time]
            output_df["mean_fragments"] = mean_fragments

            mask = output_df["pval"].notnull()
            output_df.loc[mask, "pval_adj"] = fdrcorrection(output_df.loc[mask, "pval"])[1]
            output_df.to_csv(
                os.path.join(output_dir, f"{celltype}_{baseline_time}_v_{comparison_time}.csv")
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run poisson differential accessibility.")
    parser.add_argument("--input-h5ad", required=True)
    parser.add_argument("--cell-metadata-csv", required=True)
    parser.add_argument("--archr-metadata-tsv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--groupby-col", required=False, default=None)
    return parser.parse_args()


def load_metadata(cell_metadata_csv: str, archr_metadata_tsv: str) -> pd.DataFrame:
    metadata_df = pd.read_csv(cell_metadata_csv, index_col=0)
    archr_metadata_df = pd.read_csv(archr_metadata_tsv, sep=r"\s+", index_col=0)
    metadata_df["TSSEnrichment"] = archr_metadata_df.loc[
        metadata_df.index,
        "TSSEnrichment",
    ]
    return metadata_df


def check_metadata(metadata_df: pd.DataFrame):
    for col in ["time", "celltype", "donor", "TSSEnrichment"]:
        if col not in metadata_df.columns:
            raise ValueError(f"metadata_df must have a '{col}' column")


def main() -> None:
    args = parse_args()

    adata = sc.read_h5ad(args.input_h5ad)
    metadata_df = load_metadata(args.cell_metadata_csv, args.archr_metadata_tsv)
    assert adata.obs_names.equals(
        metadata_df.index
    ), "adata and metadata_df must have the same index"

    if args.groupby_col:
        if args.groupby_col not in metadata_df.columns:
            raise ValueError(f"metadata_df must have a '{args.groupby_col}' column")
        for group, group_metadata_df in tqdm(
            metadata_df.groupby(args.groupby_col),
            desc=f"Group: {args.groupby_col}",
        ):
            group_adata = adata[group_metadata_df.index].copy()
            group_output_dir = os.path.join(args.output_dir, f"{args.groupby_col}_{group}")
            check_metadata(group_metadata_df)
            compute_poisson_time_effects(group_adata, group_metadata_df, group_output_dir)
    else:
        check_metadata(metadata_df)
        compute_poisson_time_effects(adata, metadata_df, args.output_dir)


if __name__ == "__main__":
    main()
