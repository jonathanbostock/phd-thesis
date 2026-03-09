"""
Zeta potential analysis: Δζ (mV) vs DNA:Lipid ratio on log-x scale.
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.plotting import setup_plot_style, format_axes, save_plot
from utils import defaults


def main() -> None:
    df_raw = pd.read_table("2023-06-27 30bp Zeta Potential.tsv").set_index(
        "Sample Name"
    )

    data_to_store = [
        "Peak 1 Mean by Intensity ordered by area (nm)",
        "Zeta Potential (mV)",
    ]

    sample_dict: dict = {}
    for index, row in df_raw.iterrows():
        for col in data_to_store:
            if not math.isnan(float(row[col])):  # type: ignore[arg-type]
                if index not in sample_dict:
                    sample_dict[index] = {}
                sample_dict[index][col] = row[col]

    control = sample_dict["Ext 1 Ctrl 2"]

    keys = list(sample_dict.keys())
    for key in keys:
        if "Ctrl" in key:
            del sample_dict[key]
        else:
            for col in data_to_store:
                sample_dict[key][col] -= control[col]

    data_df = pd.DataFrame.from_dict(sample_dict, orient="index")
    data_df["DNA:Lipid Ratio"] = [1 / 20, 1 / 100, 1 / 500, 1 / 2000]
    data_df = data_df.sort_values("DNA:Lipid Ratio")

    setup_plot_style()
    color = sns.color_palette("colorblind")[0]

    fig, ax = plt.subplots(figsize=(defaults.fig_width, defaults.fig_height))

    ax.scatter(
        data_df["DNA:Lipid Ratio"],
        data_df["Zeta Potential (mV)"],
        color=color,
        edgecolors="black",
        linewidths=0.5,
        s=50,
        zorder=5,
    )
    ax.plot(
        data_df["DNA:Lipid Ratio"],
        data_df["Zeta Potential (mV)"],
        color=color,
        linewidth=1.5,
        alpha=0.7,
    )

    ax.set_xscale("log")
    ax.set_xlabel("DNA:Lipid Ratio")
    ax.set_ylabel(r"$\Delta\zeta$ / mV")
    format_axes(ax)

    plt.tight_layout()
    save_plot(fig, "zeta-potential")


if __name__ == "__main__":
    main()
