"""Load processed TLR stimulation cytokine data."""

import itertools
import os
import re

import numpy as np
import pandas as pd

TOP_CYTOKINES = [
    "CCL2",
    "CCL3",
    "CCL7",
    "CD40L",
    "CXCL10",
    "EGF",
    "FGF2",
    "GCSF",
    "IFNG",
    "IL10",
    "IL12P40",
    "IL1A",
    "IL1B",
    "IL1RA",
    "IL2",
    "IL22",
    "IL7",
    "IL9",
    "TGFA",
    "TNFA",
]


STIM_TO_OUTPUT_LABEL = {
    "bac": "bacterial",
    "vir": "viral",
    "unstim": "unstimulated",
}


def _load_single_stimulation_and_vaccine_file(stim_fpath: str, vaccine: str) -> pd.DataFrame:
    input_df = pd.read_csv(stim_fpath)
    input_df["donor"] = input_df["donor"].astype(str)

    comp_time_to_col = {}
    for col in input_df.columns:
        if not (match := re.match(r"D(\d+)vD(\d+)", col)):
            continue
        baseline_time, comp_time = int(match.group(1)), int(match.group(2))
        if baseline_time == 0 and comp_time != 0:
            comp_time_to_col[comp_time] = col

    samples = sorted(itertools.product(input_df["donor"].unique(), comp_time_to_col.keys()))
    cytokines = sorted(input_df["cytokine"].unique())

    output_mtx = np.full((len(samples), len(cytokines)), np.nan)
    for i, (donor, comp_time) in enumerate(samples):
        donor_df = input_df[input_df["donor"] == donor].set_index("cytokine")
        assert set(cytokines) == set(donor_df.index)
        output_mtx[i] = donor_df.loc[cytokines, comp_time_to_col[comp_time]].values

    index = pd.MultiIndex.from_tuples(
        [(*sample, vaccine) for sample in samples],
        names=["donor", "comparison_time", "vaccine"],
    )
    return pd.DataFrame(output_mtx, index=index, columns=cytokines)


def _load_single_stimulation(
    functional_data_dir: str,
    stimulation: str,
    vaccines: list[str],
) -> pd.DataFrame:
    dfs = []
    for vaccine in vaccines:
        vax_path = os.path.join(functional_data_dir, vaccine, f"{vaccine}_{stimulation}.csv")
        dfs.append(_load_single_stimulation_and_vaccine_file(vax_path, vaccine.upper()))
    return pd.concat(dfs, axis=0, join="outer")[TOP_CYTOKINES]


def load_tlr_stimulation_responses(
    functional_data_dir: str,
    stimulations: list[str],
    vaccines: list[str],
) -> dict[str, pd.DataFrame]:
    return {
        stimulation: _load_single_stimulation(functional_data_dir, stimulation, vaccines)
        for stimulation in stimulations
    }
