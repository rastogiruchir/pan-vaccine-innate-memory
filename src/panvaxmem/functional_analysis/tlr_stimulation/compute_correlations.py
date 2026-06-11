"""Correlate TLR cytokine responses with mrVI differential abundance."""

import os
from argparse import ArgumentParser

import pandas as pd
import scanpy as sc

import panvaxmem.functional_analysis.tlr_stimulation.dataset_utils as dataset_utils
import panvaxmem.functional_analysis.utils.correlations as correlations


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--labeled-mrvi-h5ad", required=True)
    parser.add_argument("--log-probs-parquet", required=True)
    parser.add_argument(
        "--functional-data-dir",
        default="/data/yosef2/users/ruchir/pan_vaccine_immune_response/functional/TLR/processed",
    )
    parser.add_argument("--output-dir", default="../outputs/tlr_stimulation")
    parser.add_argument("--vaccines", nargs="+", default=["tiv", "shx"])
    parser.add_argument("--stimulations", nargs="+", default=["bac", "vir", "unstim"])
    parser.add_argument("--n-permutations", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    adata = sc.read_h5ad(args.labeled_mrvi_h5ad)
    adata.obs["donor"] = adata.obs["donor"].astype(str)
    log_probs_df = pd.read_parquet(args.log_probs_parquet)
    functional_dfs = dataset_utils.load_tlr_stimulation_responses(
        args.functional_data_dir,
        args.stimulations,
        args.vaccines,
    )
    os.makedirs(args.output_dir, exist_ok=True)

    for stim, functional_df in functional_dfs.items():
        stim_adata = correlations.subset_cells_to_measured_samples(adata, functional_df)
        _, _, results_df = correlations.compute_subcluster_correlations(
            stim_adata,
            functional_df,
            log_probs_df,
            n_permutations=args.n_permutations,
            feature_name="cytokine",
        )

        output_label = dataset_utils.STIM_TO_OUTPUT_LABEL[stim]
        results_df.to_csv(
            os.path.join(args.output_dir, f"correlation_results.{output_label}.csv"),
            index=False,
        )


if __name__ == "__main__":
    main()
