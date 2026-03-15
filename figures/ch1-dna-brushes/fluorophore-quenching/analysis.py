"""
Fluorophore quenching assay: brush attachment and MBCD detachment timecourses.
Two separate 1-panel figures: (1) attachment quenching timecourse, (2) MBCD detachment by concentration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns
from scipy import stats

from utils.plotting import setup_plot_style, format_axes, save_plot
from utils import defaults


def main() -> None:
    kwargs: dict = {
        "header": 1,
        "engine": "python",
        "usecols": ["Time (min)", "Intensity (a.u.)"]
        + [f"Intensity (a.u.).{i}" for i in range(1, 21)],
    }

    columns = ["Time"] + [
        f"{condition} Repeat {repeat}"
        for repeat in [1, 2, 3]
        for condition in [
            "+ve",
            "1 mM",
            "2 mM",
            "5 mM",
            "0 mM",
            "Blank (no vesicles)",
            "Blank (vesicles)",
        ]
    ]

    start_df = pd.read_csv("2024-11-05 Co-Brush Before Vesicles.csv", **kwargs)[
        :2
    ].astype(float)
    attachment_df = pd.read_csv(
        "2024-11-05 Co-Brush Attachment Timecourse.csv", **kwargs
    )[:61].astype(float)
    detachment_df = pd.read_csv("2024-11-05 Co-Brush MBCD Timecourse.csv", **kwargs)[
        :51
    ].astype(float)

    start_df.columns = columns
    attachment_df.columns = columns
    detachment_df.columns = columns

    mbcd_concs = [0, 1, 2, 5]

    def get_quenching(df: pd.DataFrame, mbcd_conc: int | str) -> pd.DataFrame:
        positives = df[[f"+ve Repeat {repeat}" for repeat in [1, 2, 3]]]
        blanks_no_vesicles = df[
            [f"Blank (no vesicles) Repeat {repeat}" for repeat in [1, 2, 3]]
        ]
        blanks_vesicles = df[
            [f"Blank (vesicles) Repeat {repeat}" for repeat in [1, 2, 3]]
        ]
        data = df[[f"{mbcd_conc} mM Repeat {repeat}" for repeat in [1, 2, 3]]]

        positives.columns = [1, 2, 3]
        blanks_no_vesicles.columns = [1, 2, 3]
        blanks_vesicles.columns = [1, 2, 3]
        data.columns = [1, 2, 3]

        quenching = 1 - ((data - blanks_vesicles) / (positives - blanks_no_vesicles))

        new_df = pd.DataFrame()
        new_df["Mean"] = quenching.apply(np.mean, axis=1)
        new_df["SEM"] = quenching.apply(stats.sem, axis=1)
        new_df["MBCD"] = mbcd_conc
        new_df["Time"] = df["Time"].values
        return new_df

    for repeat in [1, 2, 3]:
        attachment_df[f"Mean mM Repeat {repeat}"] = attachment_df.apply(
            lambda row: np.mean(
                [row[f"{mbcd_conc} mM Repeat {repeat}"] for mbcd_conc in mbcd_concs]
            ),
            axis=1,
        )

    attachment_data = get_quenching(attachment_df, "Mean")

    detachment_data = pd.concat(
        [get_quenching(detachment_df, mbcd_conc) for mbcd_conc in mbcd_concs]
    )

    setup_plot_style()

    # Figure 1: attachment timecourse (single line, colorblind color 0)
    fig1, ax1 = plt.subplots(figsize=(defaults.fig_width, defaults.fig_height))
    color = sns.color_palette("colorblind")[0]
    ax1.plot(
        attachment_data["Time"],
        attachment_data["Mean"],
        color=color,
        linewidth=2,
    )
    ax1.fill_between(
        attachment_data["Time"],
        attachment_data["Mean"] - attachment_data["SEM"],
        attachment_data["Mean"] + attachment_data["SEM"],
        color=color,
        alpha=0.3,
    )
    ax1.set_xlabel("Time / min")
    ax1.set_ylabel("Quenching")
    ax1.set_title("Brush Attachment")
    format_axes(ax1)
    save_plot(fig1, "fluorophore-quenching-attachment")

    # Figure 2: MBCD detachment by concentration (gradient colormap)
    fig2, ax2 = plt.subplots(figsize=(defaults.fig_width, defaults.fig_height))
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=min(mbcd_concs), vmax=max(mbcd_concs))

    for conc in mbcd_concs:
        subset = detachment_data[detachment_data["MBCD"] == conc]
        color = cmap(norm(conc))
        ax2.plot(
            subset["Time"],
            subset["Mean"],
            color=color,
            linewidth=2,
            label=f"{conc} mM",
        )
        ax2.fill_between(
            subset["Time"],
            subset["Mean"] - subset["SEM"],
            subset["Mean"] + subset["SEM"],
            color=color,
            alpha=0.3,
        )

    ax2.set_xlabel("Time / min")
    ax2.set_ylabel("Quenching")
    ax2.set_title("MBCD Detachment")
    ax2.legend(title="MβCD", frameon=False, bbox_to_anchor=(1.05, 1), loc="upper left")
    format_axes(ax2)
    save_plot(fig2, "fluorophore-quenching")


if __name__ == "__main__":
    main()
