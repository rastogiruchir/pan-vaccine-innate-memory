"""Combine CM fgsea results across vaccines into NES and -log10(padj) tables.

Each table combines fgsea outputs from vaccine-specific CM differential
expression comparisons. Rows are selected pathways, and columns are
vaccine-timepoint comparisons.
"""

import argparse
import os
import re

import numpy as np
import pandas as pd

GSEA_RESULT_SUBDIRS = {
    "H5N1+AS03": "h5n1_adj",
    "H5N1": "h5n1_nadj",
    "TIV": "tiv",
    "BNT162b2": "pfzr",
    "YFV": "yfv",
    "SHX": "shx",
}
GENE_SETS = ["BTM", "HALLMARK", "REACTOME"]
LATE_TIMEPOINTS = {
    "H5N1+AS03": [21, 42],
    "H5N1": [21, 42],
    "TIV": [30],
    "BNT162b2": [21, 42],
    "YFV": [28],
    "SHX": [30, 90],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create fgsea NES and -log10(padj) summary tables."
    )
    parser.add_argument("--fgsea-results-dir", default="outputs/fgsea")
    parser.add_argument("--output-dir", default="outputs/fgsea/combined_tables/late")
    parser.add_argument(
        "--late-timepoints-only", default=True, action=argparse.BooleanOptionalAction
    )
    parser.add_argument("--clip-max-log10-pvalue", type=float, default=10)
    return parser.parse_args()


def get_result_paths(fgsea_results_dir: str, gene_set: str) -> list[dict]:
    pattern = re.compile(rf"CM_D(\d+)vD(\d+)\.{gene_set}\.tsv$")
    result_paths = []
    for vax, subdir in GSEA_RESULT_SUBDIRS.items():
        dir_path = os.path.join(fgsea_results_dir, subdir)
        for fname in os.listdir(dir_path):
            match = pattern.match(fname)
            if match is None:
                continue
            time1, time2 = map(int, match.groups())
            assert time1 == 0
            result_paths.append(
                {
                    "vax": vax,
                    "time": time2,
                    "path": os.path.join(dir_path, fname),
                }
            )
    return result_paths


def prettify_term(term: str) -> str:
    if term.startswith("HALLMARK_"):
        term = term.replace("HALLMARK_", "")
    if len(term) > 75:
        whitespace_idx = term.find(" ", len(term) // 2)
        if whitespace_idx != -1:
            term = term[:whitespace_idx] + "\n" + term[whitespace_idx:]
    return term


def load_results_table(fpath: str, pval_col: str = "padj") -> pd.DataFrame:
    results_df = pd.read_csv(fpath, sep="\t", index_col="pathway").sort_values(pval_col)
    min_nonzero_pval = results_df[results_df[pval_col] > 0][pval_col].min()
    results_df[pval_col] = results_df[pval_col].replace(0, min_nonzero_pval)
    results_df.index = results_df.index.map(prettify_term)
    return results_df


def select_terms(
    result_paths: list[dict],
    max_padjust: float = 1e-4,
    min_nes: float = 2.0,
    top_k_per_time: int = 5,
) -> list[str]:
    top_k_terms = set()
    for result_path in result_paths:
        results_df = load_results_table(result_path["path"])
        results_df = results_df[
            (results_df["padj"] < max_padjust) & (results_df["NES"].abs() > min_nes)
        ].sort_values("padj")
        top_k_terms.update(results_df.index[:top_k_per_time])
    return sorted(top_k_terms)


def create_tables(
    gene_set: str,
    fgsea_results_dir: str,
    late_timepoints_only: bool,
    clip_max_log10_pvalue: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_paths = get_result_paths(fgsea_results_dir, gene_set)
    output_result_paths = result_paths
    if late_timepoints_only:
        output_result_paths = [
            result_path
            for result_path in output_result_paths
            if result_path["time"] in LATE_TIMEPOINTS[result_path["vax"]]
        ]
    output_result_paths = sorted(
        output_result_paths,
        key=lambda result_path: (
            list(GSEA_RESULT_SUBDIRS).index(result_path["vax"]),
            result_path["time"],
        ),
    )

    terms = select_terms(result_paths)
    columns = [
        f"{result_path['vax']} (D{result_path['time']})" for result_path in output_result_paths
    ]
    nes_df = pd.DataFrame(0.0, index=terms, columns=columns)
    padj_df = pd.DataFrame(0.0, index=terms, columns=columns)

    for result_path in output_result_paths:
        results_df = load_results_table(result_path["path"])
        col = f"{result_path['vax']} (D{result_path['time']})"
        for term in terms:
            if term in results_df.index and pd.notnull(results_df.loc[term, "NES"]):
                nes_df.loc[term, col] = results_df.loc[term, "NES"]
            if term in results_df.index and pd.notnull(results_df.loc[term, "padj"]):
                padj_df.loc[term, col] = -np.log10(results_df.loc[term, "padj"])
    return nes_df, padj_df.clip(0, clip_max_log10_pvalue)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    for gene_set in GENE_SETS:
        nes_df, padj_df = create_tables(
            gene_set,
            args.fgsea_results_dir,
            args.late_timepoints_only,
            args.clip_max_log10_pvalue,
        )
        print(f"Found {len(nes_df)} significant terms for {gene_set} gene set")
        nes_df.to_csv(os.path.join(args.output_dir, f"nes.{gene_set}.csv"))
        padj_df.to_csv(os.path.join(args.output_dir, f"padj.{gene_set}.csv"))


if __name__ == "__main__":
    main()
