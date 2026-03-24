"""
ATP aptamer brush: ΔD (nm) vs ATP concentration, raw data scatter, symlog x-scale.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.plotting import setup_plot_style, format_axes, save_plot, plot_mean_sem_overlay
from utils import defaults


def main() -> None:
    raw_data = pd.read_csv("2023-08-07 ATP Aptamer Brush.tsv", delimiter="\t")

    name_key = "Sample Name"
    p1mi_key = "Peak 1 Mean by Intensity ordered by area (nm)"

    cond_dict = {
        "0": 0,
        "1uM": 1e-3,
        "10uM": 1e-2,
        "100uM": 1e-1,
        "1mM": 1e0,
        "3mM": 3e0,
    }

    control_dict: dict[str, float] = {}
    x_raw: list[float] = []
    y_raw: list[float] = []

    for _, row in raw_data.iterrows():
        name_list = str(row[name_key]).split()
        size = float(row[p1mi_key])  # type: ignore[arg-type]
        extrusion = name_list[1]
        condition = name_list[2]

        if condition == "Ctrl":
            control_dict[extrusion] = size
        else:
            delta = size - control_dict[extrusion]
            x_raw.append(cond_dict[condition])
            y_raw.append(delta)

    x_arr = np.array(x_raw)
    y_arr = np.array(y_raw)

    setup_plot_style()
    color = sns.color_palette("colorblind")[0]

    fig, ax = plt.subplots(figsize=(defaults.fig_width, defaults.fig_height))

    ax.scatter(
        x_arr,
        y_arr,
        color=color,
        marker="o",
        edgecolors="black",
        linewidths=0.5,
        s=25,
        zorder=5,
    )

    plot_mean_sem_overlay(ax, x_arr, y_arr, color=color)

    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlabel("ATP Concentration / mM")
    ax.set_ylabel(r"$\Delta D$ / nm")
    current_ylim = ax.get_ylim()
    ax.set_ylim(bottom=-1, top=current_ylim[1])
    format_axes(ax)

    plt.tight_layout()
    save_plot(fig, "atp-aptamer-brush")


if __name__ == "__main__":
    main()
