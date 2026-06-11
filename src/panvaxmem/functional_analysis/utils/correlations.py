"""Utilities for correlating mrVI differential abundance with functional responses."""

import numpy as np
import pandas as pd
from anndata import AnnData
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from panvaxmem.functional_analysis.utils.sample_metadata import extract_time


def subset_cells_to_measured_samples(adata: AnnData, functional_df: pd.DataFrame) -> AnnData:
    adata.obs["time_int"] = adata.obs["time"].apply(extract_time)
    donor_time_pairs = {
        pair for donor, time, _ in functional_df.index for pair in [(donor, 0), (donor, time)]
    }

    valid = adata.obs.apply(
        lambda row: (row["donor"], row["time_int"]) in donor_time_pairs,
        axis=1,
    )
    return adata[valid].copy()


def collate_da_scores_and_responses(
    adata: AnnData,
    functional_df: pd.DataFrame,
    log_probs_df: pd.DataFrame,
    suffix: str = "true",
    center_diff_log_probs: bool = True,
    min_cells_per_time: int = 50,
    min_percentile: float = 1.0,
    max_percentile: float = 99.0,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    assert set(adata.obs_names).issubset(set(log_probs_df.index))
    log_probs_df = log_probs_df.loc[adata.obs_names]
    adata.obs["time_int"] = adata.obs["time"].apply(extract_time)

    features = list(functional_df.columns)
    X_by_feature = {feature: [] for feature in features}
    y_by_feature = {feature: [] for feature in features}

    for (donor, time, _), row in functional_df.iterrows():
        baseline_colname = f"{donor}:0_{suffix}"
        comp_colname = f"{donor}:{time}_{suffix}"

        if baseline_colname not in log_probs_df.columns or comp_colname not in log_probs_df.columns:
            continue

        cells_per_time = adata[adata.obs["donor"] == donor].obs["time_int"].value_counts()
        if (
            cells_per_time.get(0, 0) < min_cells_per_time
            or cells_per_time.get(time, 0) < min_cells_per_time
        ):
            continue

        diff = log_probs_df[comp_colname] - log_probs_df[baseline_colname]
        if center_diff_log_probs:
            diff = diff - diff.mean()

        lower = np.percentile(diff, min_percentile)
        upper = np.percentile(diff, max_percentile)
        diff = np.clip(diff, lower, upper)

        for feature, val in row.items():
            if np.isnan(val):
                continue
            X_by_feature[feature].append(diff.values)
            y_by_feature[feature].append(val)

    return (
        {feature: np.array(X) for feature, X in X_by_feature.items()},
        {feature: np.array(y) for feature, y in y_by_feature.items()},
    )


def fast_pearson(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Correlate each column of X with y.

    X has shape n_samples x n_cells. Each column is one cell's differential
    abundance scores across samples, and y is the functional response across
    the same samples.
    """
    X_centered = X - X.mean(axis=0)
    y_centered = y - y.mean()

    X_norms = np.linalg.norm(X_centered, axis=0)
    y_norm = np.linalg.norm(y_centered)
    X_norms[X_norms == 0] = 1
    if y_norm == 0:
        return np.zeros(X.shape[1])

    return (X_centered.T @ y_centered) / (X_norms * y_norm)


def two_sided_empirical_pvalue(observed: float, null_distribution: np.ndarray) -> float:
    n = len(null_distribution)
    p_right = (1 + np.sum(null_distribution >= observed)) / (1 + n)
    p_left = (1 + np.sum(null_distribution <= observed)) / (1 + n)
    return min(1.0, 2 * min(p_left, p_right))


def compute_subcluster_correlations(
    adata: AnnData,
    functional_df: pd.DataFrame,
    log_probs_df: pd.DataFrame,
    feature_name: str,
    n_permutations: int = 100,
    center_diff_log_probs: bool = True,
    min_cells_per_time: int = 50,
) -> tuple[dict, dict, pd.DataFrame]:
    # (1) Get indices belonging to each subcluster
    subcluster_idxs = {
        sc: np.where(adata.obs["subcluster"] == sc)[0]
        for sc in sorted(adata.obs["subcluster"].unique())
    }

    # (2) Gather the true differential abundance scores and functional responses for each feature.
    X_by_feature, y_by_feature = collate_da_scores_and_responses(
        adata,
        functional_df,
        log_probs_df,
        suffix="true",
        center_diff_log_probs=center_diff_log_probs,
        min_cells_per_time=min_cells_per_time,
    )
    features = [f for f in functional_df.columns if len(X_by_feature[f]) > 0]

    # (3) Compute observed median Pearson per subcluster
    true_pearsons = {}
    observed_medians = {}
    for f in features:
        corrs = fast_pearson(X_by_feature[f], y_by_feature[f])
        true_pearsons[f] = corrs
        observed_medians[f] = {sc: np.median(corrs[idxs]) for sc, idxs in subcluster_idxs.items()}

    # (4) Get null median Pearson per subcluster
    null_medians = {f: [] for f in features}
    for permutation_idx in tqdm(range(n_permutations), desc="Computing null correlations"):
        X_by_feature, y_by_feature = collate_da_scores_and_responses(
            adata,
            functional_df,
            log_probs_df,
            suffix=f"perm{permutation_idx}",
            center_diff_log_probs=center_diff_log_probs,
            min_cells_per_time=min_cells_per_time,
        )
        for f in features:
            corrs = fast_pearson(X_by_feature[f], y_by_feature[f])
            null_medians[f].extend(np.median(corrs[idxs]) for idxs in subcluster_idxs.values())
    null_medians = {f: np.array(values) for f, values in null_medians.items()}

    # (5) Compute p-values and return results dataframe
    pvalues, tests = [], []
    for f in sorted(features):
        for sc in sorted(adata.obs["subcluster"].unique()):
            p = two_sided_empirical_pvalue(observed_medians[f][sc], null_medians[f])
            pvalues.append(p)
            tests.append((sc, f))

    _, pvalues_adj, _, _ = multipletests(pvalues, method="fdr_bh")

    results_df = pd.DataFrame(
        {
            "subcluster": [sc for sc, _ in tests],
            feature_name: [f for _, f in tests],
            "median_correlation": [observed_medians[f][sc] for sc, f in tests],
            "pvalue": pvalues,
            "pvalue_adj": pvalues_adj,
        }
    )

    return (true_pearsons, null_medians, results_df)
