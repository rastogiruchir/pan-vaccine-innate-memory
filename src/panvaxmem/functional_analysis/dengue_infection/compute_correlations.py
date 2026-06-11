"""Correlate dengue infection responses with mrVI differential abundance."""

import os
from argparse import ArgumentParser

import pandas as pd
import scanpy as sc

import panvaxmem.functional_analysis.dengue_infection.dataset_utils as dataset_utils
import panvaxmem.functional_analysis.utils.correlations as correlations


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--labeled-mrvi-h5ad", required=True)
    parser.add_argument("--log-probs-parquet", required=True)
    parser.add_argument(
        "--functional-data-dir",
        default="/data/yosef2/users/ruchir/pan_vaccine_immune_response/functional/viral/processed",
    )
    parser.add_argument("--output-dir", default="../outputs/dengue_infection")
    parser.add_argument("--vaccines", nargs="+", default=["pfzr", "shx"])
    parser.add_argument("--infection-virus", default="dengue")
    parser.add_argument("--n-permutations", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()

    adata = sc.read_h5ad(args.labeled_mrvi_h5ad)
    adata.obs["donor"] = adata.obs["donor"].astype(str)
    functional_df = dataset_utils.load_dengue_infection_responses(
        args.functional_data_dir,
        args.vaccines,
        infection_virus=args.infection_virus,
    )
    adata = correlations.subset_cells_to_measured_samples(adata, functional_df)
    log_probs_df = pd.read_parquet(args.log_probs_parquet)

    _, _, results_df = correlations.compute_subcluster_correlations(
        adata,
        functional_df,
        log_probs_df,
        n_permutations=args.n_permutations,
        feature_name="timepoint",
    )

    os.makedirs(args.output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(args.output_dir, "correlation_results.csv"), index=False)


if __name__ == "__main__":
    main()
