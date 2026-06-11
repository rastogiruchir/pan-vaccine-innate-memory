"""Utilities for donor/time sample metadata used in functional DA analyses."""

from collections import namedtuple

import pandas as pd

Sample = namedtuple("Sample", ["donor", "time"])


def extract_time(time: str) -> int:
    if isinstance(time, (int, float)):
        return int(time)
    if time.startswith("d") or time.startswith("D"):
        return int(time[1:])
    return int(time)


def find_samples_with_functional_measurements(
    adata_obs: pd.DataFrame,
    functional_dfs: list[pd.DataFrame],
) -> list[Sample]:
    samples = set()
    for functional_df in functional_dfs:
        for (donor, comp_time, _), row in functional_df.iterrows():
            sample_obs = adata_obs[
                (adata_obs["donor"] == donor) & (adata_obs["time_int"] == comp_time)
            ]
            if len(sample_obs) == 0:
                continue
            if not pd.isna(row).all():
                samples.add(Sample(donor=donor, time=0))
                samples.add(Sample(donor=donor, time=comp_time))
    return sorted(samples)


def shuffle_time_within_donor(obs: pd.DataFrame, random_state: int) -> pd.DataFrame:
    shuffled = obs.copy()
    for donor in shuffled["donor"].unique():
        mask = shuffled["donor"] == donor
        shuffled.loc[mask, ["time", "time_int"]] = (
            shuffled.loc[
                mask,
                ["time", "time_int"],
            ]
            .sample(frac=1, random_state=random_state, replace=False, ignore_index=True)
            .values
        )
    return shuffled
