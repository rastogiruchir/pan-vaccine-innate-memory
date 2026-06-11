"""Load processed dengue infection functional data."""

import os
import re

import numpy as np
import pandas as pd


def _load_single_vaccine_file(
    fpath: str,
    vaccine: str,
    infection_virus: str = "dengue",
) -> pd.DataFrame:
    df = pd.read_csv(fpath)
    df = df[df["infection_virus"] == infection_virus].copy()
    df["subject_code"] = df["subject_code"].astype(str)
    if vaccine == "SHX":
        df["subject_code"] = df["subject_code"].str.replace("SHX", "")

    comp_time_to_col = {}
    for col in df.columns:
        if not (match := re.match(r"D(\d+)vD(\d+)", col)):
            continue
        baseline_time, comp_time = int(match.group(1)), int(match.group(2))
        if baseline_time == 0 and comp_time != 0:
            comp_time_to_col[comp_time] = col

    samples = sorted(
        (donor, comp_time)
        for donor in df["subject_code"].unique()
        for comp_time in comp_time_to_col
    )
    infection_times = sorted(df["infection_time"].unique())

    sample_to_row = {sample: i for i, sample in enumerate(samples)}
    infection_time_to_col = {infection_time: j for j, infection_time in enumerate(infection_times)}
    lfc_mtx = np.full((len(samples), len(infection_times)), np.nan)

    for _, row in df.iterrows():
        donor = row["subject_code"]
        infection_time = row["infection_time"]
        for comp_time, colname in comp_time_to_col.items():
            sample = (donor, comp_time)
            lfc_mtx[
                sample_to_row[sample],
                infection_time_to_col[infection_time],
            ] = row[colname]

    index = pd.MultiIndex.from_tuples(
        [(*sample, vaccine) for sample in samples],
        names=["donor", "comparison_time", "vaccine"],
    )
    return pd.DataFrame(lfc_mtx, index=index, columns=infection_times)


def load_dengue_infection_responses(
    functional_data_dir: str,
    vaccines: list[str],
    infection_virus: str = "dengue",
) -> pd.DataFrame:
    dfs = []
    for vaccine in vaccines:
        vax_path = os.path.join(functional_data_dir, f"{vaccine}.csv")
        dfs.append(
            _load_single_vaccine_file(
                vax_path,
                vaccine.upper(),
                infection_virus=infection_virus,
            )
        )
    return pd.concat(dfs, axis=0, join="outer")
