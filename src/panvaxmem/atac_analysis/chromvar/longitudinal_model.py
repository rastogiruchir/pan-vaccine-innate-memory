import argparse
import os
from collections import defaultdict

import anndata
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm


def __extract_time(time: str) -> int:
    return int(time[1:])


def compute_batch_corrected_effects(
    deviations_df: pd.DataFrame,  # [cells, motifs]
    metadata_df: pd.DataFrame,  # [cells, ...]
    output_dir: str,
    min_cells: int = 100,
):
    os.makedirs(output_dir, exist_ok=True)

    assert deviations_df.index.equals(metadata_df.index)

    times = sorted(metadata_df["time"].unique(), key=__extract_time)
    d0 = times[0]
    assert __extract_time(d0) == 0, f"Expected first time point to be D0, got {d0}"

    for celltype in tqdm(
        metadata_df["celltype"].unique(), desc="Computing batch-corrected effects"
    ):
        celltype_df = metadata_df[metadata_df["celltype"] == celltype].copy()
        if celltype_df.shape[0] < min_cells:
            print(f"Skipping {celltype} with {celltype_df.shape[0]} cells")
            continue

        celltype_deviations_df = deviations_df.loc[celltype_df.index]

        motifs = defaultdict(list)
        betas = defaultdict(list)
        pvalues = defaultdict(list)

        for motif in celltype_deviations_df.columns:
            celltype_df["X"] = celltype_deviations_df[motif]
            formula = f"X ~ C(time, Treatment(reference='{d0}')) + C(donor)"
            model = smf.ols(formula=formula, data=celltype_df)
            results = model.fit(disp=0)
            for dcomp in times[1:]:
                param_name = f"C(time, Treatment(reference='{d0}'))[T.{dcomp}]"
                motifs[dcomp].append(motif)
                betas[dcomp].append(results.params[param_name])
                pvalues[dcomp].append(results.pvalues[param_name])

        for dcomp in times[1:]:
            output_df = pd.DataFrame(
                dict(motif=motifs[dcomp], beta=betas[dcomp], pval=pvalues[dcomp])
            )
            output_df["pval_adj"] = fdrcorrection(output_df["pval"])[1]
            output_path = os.path.join(output_dir, f"{celltype}_{d0}_v_{dcomp}.csv")
            output_df.to_csv(output_path, index=False)


def compute_per_donor_effects(
    deviations_df: pd.DataFrame,  # [cells, motifs]
    metadata_df: pd.DataFrame,  # [cells, ...]
    output_dir: str,
    min_cells: int = 25,
):
    os.makedirs(output_dir, exist_ok=True)

    assert deviations_df.index.equals(metadata_df.index)

    times = sorted(metadata_df["time"].unique(), key=__extract_time)
    d0 = times[0]
    assert __extract_time(d0) == 0, f"Expected first time point to be d0, got {d0}"

    for dcomp in tqdm(times[1:], desc="Computing per-donor effects"):
        for celltype in metadata_df["celltype"].unique():
            for donor in metadata_df["donor"].unique():
                d0_df = deviations_df[
                    (metadata_df["donor"] == donor)
                    & (metadata_df["celltype"] == celltype)
                    & (metadata_df["time"] == d0)
                ]
                dcomp_df = deviations_df[
                    (metadata_df["donor"] == donor)
                    & (metadata_df["celltype"] == celltype)
                    & (metadata_df["time"] == dcomp)
                ]

                if len(d0_df) < min_cells or len(dcomp_df) < min_cells:
                    continue

                motifs, betas, pvalues = [], [], []
                for motif in d0_df.columns:
                    beta = dcomp_df[motif].mean() - d0_df[motif].mean()
                    _, p = ttest_ind(d0_df[motif], dcomp_df[motif])
                    motifs.append(motif)
                    betas.append(beta)
                    pvalues.append(p)

                output_df = pd.DataFrame(dict(motif=motifs, beta=betas, pval=pvalues))
                output_df["pval_adj"] = fdrcorrection(output_df["pval"])[1]
                output_path = os.path.join(output_dir, f"{donor}_{celltype}_{d0}_v_{dcomp}.csv")
                output_df.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute chromVAR batch-corrected and per-donor effects from deviationScores."
    )
    parser.add_argument("--deviation-scores", required=True)
    metadata_group = parser.add_mutually_exclusive_group(required=True)
    metadata_group.add_argument("--metadata-csv")
    metadata_group.add_argument("--metadata-h5ad")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--groupby-col",
        required=False,
        help="If specified, compute effects separately within groups defined by this metadata column.",
    )
    return parser.parse_args()


def read_metadata(args: argparse.Namespace) -> pd.DataFrame:
    if args.metadata_csv:
        return pd.read_csv(args.metadata_csv, index_col=0)
    else:
        adata = anndata.read_h5ad(args.metadata_h5ad, backed="r")
        return adata.obs.copy()


def validate_inputs(
    deviations_df: pd.DataFrame, metadata_df: pd.DataFrame, groupby_col: str | None
) -> None:
    required_cols = {"time", "donor", "celltype"}
    if groupby_col:
        required_cols.add(groupby_col)

    missing_cols = sorted(required_cols - set(metadata_df.columns))
    if missing_cols:
        raise ValueError(f"Metadata is missing required columns: {missing_cols}")

    missing_metadata = deviations_df.index.difference(metadata_df.index)
    if len(missing_metadata) > 0:
        raise ValueError(
            "Metadata is missing cells from deviation scores. "
            f"First missing cells: {missing_metadata[:5].tolist()}"
        )


def compute_effects(
    deviations_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    output_dir: str,
) -> None:
    assert deviations_df.index.equals(metadata_df.index)
    compute_batch_corrected_effects(deviations_df, metadata_df, output_dir)
    compute_per_donor_effects(
        deviations_df,
        metadata_df,
        os.path.join(output_dir, "per_donor"),
    )


def main() -> None:
    args = parse_args()

    metadata_df = read_metadata(args)
    deviations_df = pd.read_csv(args.deviation_scores, index_col=0).T

    validate_inputs(deviations_df, metadata_df, args.groupby_col)
    metadata_df = metadata_df.loc[deviations_df.index]

    if args.groupby_col:
        for group, group_metadata_df in tqdm(
            metadata_df.groupby(args.groupby_col),
            desc=f"Computing effects by {args.groupby_col}",
        ):
            group_deviations_df = deviations_df.loc[group_metadata_df.index]
            group_output_dir = os.path.join(args.output_dir, str(group))
            compute_effects(
                group_deviations_df,
                group_metadata_df,
                group_output_dir,
            )
    else:
        compute_effects(deviations_df, metadata_df, args.output_dir)


if __name__ == "__main__":
    main()
