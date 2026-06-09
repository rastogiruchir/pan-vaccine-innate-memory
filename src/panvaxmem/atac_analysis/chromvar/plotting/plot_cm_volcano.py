"""Plot per-vaccine CM chromVAR volcano plots."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.pyplot as mpl
import numpy as np
import pandas as pd
import seaborn as sns
import textalloc as ta
from matplotlib.ticker import MaxNLocator, MultipleLocator

mpl.rcParams["pdf.fonttype"] = 42


VACCINE_PANELS = {
    "H5N1+AS03 (00v42)": "h5n1_adj/CM_D000_v_D042.csv",
    "H5N1 (00v42)": "h5n1_nadj/CM_D000_v_D042.csv",
    "TIV (00v30)": "tiv/CM_D000_v_D030.csv",
    "BNT162b2 (00v42)": "pfzr/C_mono_D00_v_D42.csv",
    "YFV (00v28)": "yfv/CM_D000_v_D028.csv",
    "SHX (00v90)": "shx.0_01_f/CM_D000_v_D090.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot CM chromVAR volcano plots.")
    # fmt: off
    parser.add_argument("--dams-dir", type=Path, default=Path("../outputs/longitudinal_effects"))
    parser.add_argument("--output-path", type=Path, default=Path("figures/cm_chromvar_volcano.pdf"))
    parser.add_argument("--tf-families-path", type=Path, default=Path("../../../../../resources/tf_families.json"))
    parser.add_argument("--tf-palette-path", type=Path, default=Path("../../../../../resources/tf_family_palette.json"))
    # fmt: on
    return parser.parse_args()


def load_transcription_factor_families(
    tf_families_path: Path,
    my_families: list[str] = ["AP-1", "IRF/STAT", "NFkB", "C/EBP", "ETS"],
) -> dict[str, list[str]]:
    all_families = json.load(open(tf_families_path, "r"))
    return {family: all_families[family] for family in my_families}


def plot_chromvar_volcano(
    results_path: Path,
    family_to_tfs: dict[str, list],
    family_to_colors: dict,
    title: str,
    ax: plt.Axes,
    max_pval_adj: float = 0.05,
    min_beta: float = 0.4,
    n_top_to_label: int = 3,
    xtick_spacing: str = "auto",
):
    df = pd.read_csv(results_path, index_col=0)
    df = df.dropna()
    df.index = [motif_id.split("_")[1] for motif_id in df.index]
    df["significant"] = (df["pval_adj"] <= max_pval_adj) & (df["beta"].abs() >= min_beta)

    if family_to_tfs:
        for family in family_to_tfs:
            tfs = family_to_tfs[family]
            df.loc[tfs, "family"] = family
        df["family"] = df["family"].fillna("Other significant")
        df.loc[~df["significant"], "family"] = "Insignificant"

    df["pval_adj"] = df["pval_adj"].replace(0, np.finfo(float).tiny)
    df["-log10p"] = -np.log10(df["pval_adj"])

    sns.scatterplot(
        data=df,
        x="beta",
        y="-log10p",
        hue="family" if family_to_tfs else "significant",
        palette=family_to_colors if family_to_colors else None,
        s=75,
        ax=ax,
    )
    ax.axvline(0, color="red", linestyle="--", linewidth=0.75)
    for label in ax.get_xticklabels():
        label.set_fontsize(16)
    for label in ax.get_yticklabels():
        label.set_fontsize(16)

    if xtick_spacing == "auto":
        ax.xaxis.set_major_locator(MaxNLocator(nbins="auto", steps=[1, 5, 10]))
    elif type(xtick_spacing) is float or type(xtick_spacing) is int:
        ax.xaxis.set_major_locator(MultipleLocator(xtick_spacing))
    else:
        raise ValueError("Invalid xtick_spacing value")

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title, fontsize=28)
    ax.legend().remove()

    df_up = df[df["beta"] > 0].sort_values("pval_adj")
    df_down = df[df["beta"] < 0].sort_values("pval_adj")

    xs, ys, texts = [], [], []
    for df in (df_up, df_down):
        for _, row in df.iloc[:n_top_to_label].iterrows():
            xs.append(row["beta"])
            ys.append(row["-log10p"])
            texts.append(row.name)
    ta.allocate(ax, xs, ys, texts, x_scatter=xs, y_scatter=ys, textsize=20, linecolor="gray")


def main() -> None:
    args = parse_args()

    family_to_tfs = load_transcription_factor_families(args.tf_families_path)
    family_to_tfs["Other significant"] = []
    family_to_tfs["Insignificant"] = []

    family_to_colors = json.load(open(args.tf_palette_path, "r"))

    fig, axs = plt.subplots(nrows=1, ncols=6, figsize=(30, 5), dpi=300)
    for ax, (title, results_path) in zip(axs, VACCINE_PANELS.items()):
        plot_chromvar_volcano(
            args.dams_dir / results_path,
            family_to_tfs,
            family_to_colors,
            title,
            ax,
        )

    handles = []
    for family in [family for family in family_to_colors.keys() if family != "Other"]:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=family,
                markerfacecolor=family_to_colors[family],
                markersize=14,
            )
        )

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=7,
        fontsize=20,
        bbox_to_anchor=(0.5, -0.125),
        frameon=False,
    )
    fig.supxlabel(r"change in motif accessibility ($\beta$)", fontsize=20)
    fig.supylabel(r"$-\log_{10} p_{\mathrm{adj}}$", fontsize=20, x=0.005)
    plt.tight_layout()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output_path, bbox_inches="tight")


if __name__ == "__main__":
    main()
