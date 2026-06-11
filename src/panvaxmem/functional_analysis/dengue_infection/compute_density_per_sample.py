"""Compute mrVI sample-posterior densities for dengue infection samples."""

import os
from argparse import ArgumentParser
from pathlib import Path

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import numpy as np
import pandas as pd
import scanpy as sc
from scvi.external import MRVI
from tqdm import tqdm

import panvaxmem.functional_analysis.dengue_infection.dataset_utils as dataset_utils
import panvaxmem.functional_analysis.utils.density as density
import panvaxmem.functional_analysis.utils.sample_metadata as sample_metadata


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--mrvi-h5ad", default="../../integration/outputs/mrvi/mrvi.h5ad")
    parser.add_argument("--model-dir", default="../../integration/outputs/mrvi/model")
    parser.add_argument(
        "--functional-data-dir",
        default="/data/yosef2/users/ruchir/pan_vaccine_immune_response/functional/viral/processed",
    )
    parser.add_argument("--output-parquet", default="../outputs/dengue_infection/log_probs.parquet")
    parser.add_argument("--vaccines", nargs="+", default=["pfzr", "shx"])
    parser.add_argument("--infection-virus", default="dengue")
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def main():
    args = parse_args()

    adata = sc.read_h5ad(args.mrvi_h5ad)
    adata.obs["donor"] = adata.obs["donor"].astype(str)
    adata.obs["vaccine"] = adata.obs["vaccine"].astype(str)
    adata.obs["time_int"] = adata.obs["time"].apply(sample_metadata.extract_time)
    model = MRVI.load(args.model_dir, adata=adata)

    functional_df = dataset_utils.load_dengue_infection_responses(
        args.functional_data_dir,
        args.vaccines,
        infection_virus=args.infection_virus,
    )
    samples = sample_metadata.find_samples_with_functional_measurements(
        adata.obs,
        [functional_df],
    )
    print(f"Number of samples: {len(samples)}")

    # (1) Compute posterior mean and scales for all cells in the dataset.
    qu_locs, qu_scales = density.compute_u_mean_and_scales(adata, model)

    # (2) For each observed donor/time sample, compute the log probability of
    # every cell under that sample's posterior.
    all_results = {}
    log_probs_per_sample = density.compute_log_probs_under_all_sample_posteriors(
        qu_locs,
        qu_scales,
        adata.obs,
        samples,
        batch_size=args.batch_size,
    )
    for sample, log_probs in log_probs_per_sample.items():
        all_results[f"{sample.donor}:{sample.time}_true"] = log_probs

    # (3) Repeat after shuffling time labels within each donor. These columns
    # define the null used for downstream empirical p-values.
    for i in tqdm(range(args.n_permutations), desc="Computing donor-within shuffled densities"):
        obs_permuted = sample_metadata.shuffle_time_within_donor(adata.obs, random_state=i)
        log_probs_per_sample = density.compute_log_probs_under_all_sample_posteriors(
            qu_locs,
            qu_scales,
            obs_permuted,
            samples,
            batch_size=args.batch_size,
        )
        for sample, log_probs in log_probs_per_sample.items():
            all_results[f"{sample.donor}:{sample.time}_perm{i}"] = log_probs

    log_probs_df = pd.DataFrame(all_results, index=adata.obs_names).astype(np.float32)
    Path(args.output_parquet).parent.mkdir(parents=True, exist_ok=True)
    log_probs_df.to_parquet(args.output_parquet)


if __name__ == "__main__":
    main()
