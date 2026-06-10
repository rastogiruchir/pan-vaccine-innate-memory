"""Run scVI differential expression between D0 and post-vaccination time points.

Differential expression is computed
    (1) per cell type across donors with batch correction and
    (2) per cell type and donor without batch correction.
"""

import argparse
import os

import anndata
import pandas as pd
import scanpy as sc
import scvi
from tqdm import tqdm


def _sort_times(times: list[str]):
    def extract_time(time: str):
        if time.startswith("d") or time.startswith("D"):
            return int(time[1:])
        else:
            return int(time)

    return sorted(times, key=extract_time)


def run_de_per_celltype(
    model: scvi.model.SCVI,
    adata: anndata.AnnData,
    metadata_df: pd.DataFrame,
    output_dir: str,
    fdr: float = 0.1,
    delta: float = 0.2,
    min_cells_per_comparison: int = 50,
    batch_correction: bool = True,
    overwrite: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)
    assert adata.obs_names.equals(
        metadata_df.index
    ), "adata and metadata_df must have the same index"
    adata.obs["celltype"] = metadata_df["celltype"]

    times = _sort_times(adata.obs["time"].unique())
    for celltype in tqdm(adata.obs["celltype"].unique(), desc="Celltype"):
        assert " " not in celltype, "Celltype must not contain spaces"
        for comparison_time in tqdm(times[1:], desc="Comparison time"):
            output_fname = f"{celltype}_{times[0]}v{comparison_time}.csv"
            if not overwrite and os.path.exists(os.path.join(output_dir, output_fname)):
                continue
            # Setting idx1 to the comparison time and idx2 to the first time point means that
            # log fold changes will be computed as comparison_time - first_time_point
            idx1 = (adata.obs["time"] == comparison_time) & (adata.obs["celltype"] == celltype)
            idx2 = (adata.obs["time"] == times[0]) & (adata.obs["celltype"] == celltype)
            if idx1.sum() + idx2.sum() < min_cells_per_comparison:
                continue
            if idx1.sum() == 0 or idx2.sum() == 0:
                continue

            deg_df = model.differential_expression(
                adata=adata,
                idx1=idx1,
                idx2=idx2,
                mode="change",
                weights="uniform",
                filter_outlier_cells=False,
                pseudocounts=None,
                delta=delta,
                fdr_target=fdr,
                batch_correction=batch_correction,
                test_mode="two",
            )
            deg_df.to_csv(os.path.join(output_dir, output_fname))


def run_de_per_celltype_and_donor(
    model: scvi.model.SCVI,
    adata: anndata.AnnData,
    metadata_df: pd.DataFrame,
    output_dir: str,
    fdr: float = 0.1,
    delta: float = 0.2,
    min_cells_per_comparison: int = 50,
    overwrite: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)
    assert adata.obs_names.equals(
        metadata_df.index
    ), "adata and metadata_df must have the same index"
    assert "donor" in adata.obs.columns, "adata must have a 'donor' column"
    adata.obs["celltype"] = metadata_df["celltype"]

    times = _sort_times(adata.obs["time"].unique())
    for celltype in tqdm(adata.obs["celltype"].unique(), desc="Celltype"):
        assert " " not in celltype, "Celltype must not contain spaces"
        for donor in tqdm(adata.obs["donor"].unique(), desc="Donor"):
            assert " " not in donor, "Donor must not contain spaces"
            for comparison_time in tqdm(times[1:], desc="Comparison time"):
                output_fname = f"{celltype}_{donor}_{times[0]}v{comparison_time}.csv"
                if not overwrite and os.path.exists(os.path.join(output_dir, output_fname)):
                    continue
                # Setting idx1 to the comparison time and idx2 to the first time point means that
                # log fold changes will be computed as comparison_time - first_time_point
                idx1 = (
                    (adata.obs["time"] == comparison_time)
                    & (adata.obs["celltype"] == celltype)
                    & (adata.obs["donor"] == donor)
                )
                idx2 = (
                    (adata.obs["time"] == times[0])
                    & (adata.obs["celltype"] == celltype)
                    & (adata.obs["donor"] == donor)
                )
                if idx1.sum() + idx2.sum() < min_cells_per_comparison:
                    continue
                if idx1.sum() == 0 or idx2.sum() == 0:
                    continue

                deg_df = model.differential_expression(
                    adata=adata,
                    idx1=idx1,
                    idx2=idx2,
                    mode="change",
                    weights="uniform",
                    filter_outlier_cells=False,
                    pseudocounts=None,
                    delta=delta,
                    fdr_target=fdr,
                    batch_correction=False,  # no batch correction for per donor DE
                    test_mode="two",
                )
                deg_df.to_csv(os.path.join(output_dir, output_fname))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scVI differential expression.")
    parser.add_argument("--input-h5ad", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--groupby-col", required=False, default=None)
    return parser.parse_args()


def run_de(
    model: scvi.model.SCVI,
    adata: anndata.AnnData,
    metadata_df: pd.DataFrame,
    output_dir: str,
):
    run_de_per_celltype(model, adata, metadata_df, output_dir)
    run_de_per_celltype_and_donor(
        model,
        adata,
        metadata_df,
        os.path.join(output_dir, "per_donor"),
    )


def main() -> None:
    args = parse_args()

    adata = sc.read_h5ad(args.input_h5ad)
    model = scvi.model.SCVI.load(args.model_dir, adata)
    metadata_df = pd.read_csv(args.metadata_csv, index_col=0)

    assert adata.obs_names.equals(
        metadata_df.index
    ), "adata and metadata_df must have the same index"
    for col in ["time", "celltype", "donor"]:
        if col not in metadata_df.columns:
            raise ValueError(f"metadata_df must have a '{col}' column")

    print("Running differential expression using scVI version:", scvi.__version__)

    if args.groupby_col:
        if args.groupby_col not in metadata_df.columns:
            raise ValueError(f"metadata_df must have a '{args.groupby_col}' column")
        for group, group_metadata_df in tqdm(
            metadata_df.groupby(args.groupby_col),
            desc=f"Group: {args.groupby_col}",
        ):
            group_adata = adata[group_metadata_df.index].copy()
            group_output_dir = os.path.join(args.output_dir, f"{args.groupby_col}_{group}")
            run_de(model, group_adata, group_metadata_df, group_output_dir)
    else:
        run_de(model, adata, metadata_df, args.output_dir)


if __name__ == "__main__":
    main()
