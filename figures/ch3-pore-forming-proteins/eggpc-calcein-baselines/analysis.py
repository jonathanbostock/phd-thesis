"""
EggPC calcein release baselines: timecourse by condition (-ve / No Brush / Sparse / Dense).
Line plot with SEM bands.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from utils.plotting import setup_plot_style, format_axes, save_plot
from utils import defaults


def main() -> None:
    fluorescence_columns = ["Intensity (a.u.)"] + [
        f"Intensity (a.u.).{i}" for i in range(1, 15)
    ]

    csv_kwargs: dict = {
        "header": 1,
        "engine": "python",
        "usecols": ["Time (min)"] + fluorescence_columns,
    }

    start_df = pd.read_csv("2024-11-21 EggPC Brush Ahl Before.csv", **csv_kwargs)[
        :2
    ].astype(float)
    end_df = pd.read_csv("2024-11-21 EggPC Brush Ahl After.csv", **csv_kwargs)[
        :2
    ].astype(float)
    timecourse_df = pd.read_csv(
        "2024-11-21 EggPC Brush Ahl Timecourse.csv", **csv_kwargs
    )[:60].astype(float)

    conditions = ["-ve", "No Brush", "Sparse Brush", "Dense Brush"]

    start_means = start_df.apply(np.mean)
    end_means = end_df.apply(np.mean)

    def normalize_condition(column_names: list[str], condition: str) -> pd.DataFrame:
        new_df = pd.DataFrame()
        normalized = []
        for col in column_names:
            norm = (
                (timecourse_df[col] - start_means[col])
                / (end_means[col] - start_means[col])
                * 100
            )
            normalized.append(norm)
        norm_array = pd.concat(normalized, axis=1)
        new_df["Mean"] = norm_array.mean(axis=1)
        new_df["SEM"] = norm_array.apply(stats.sem, axis=1)
        new_df["Time"] = timecourse_df["Time (min)"].values
        new_df["Condition"] = condition
        return new_df

    release_df = pd.concat(
        [
            normalize_condition(fluorescence_columns[i::5], condition)
            for i, condition in enumerate(conditions)
        ]
    )

    setup_plot_style()
    colors = sns.color_palette("colorblind", n_colors=len(conditions))
    markers = ["o", "s", "^", "D"]

    fig, ax = plt.subplots(figsize=(defaults.fig_width, defaults.fig_height))

    for i, cond in enumerate(conditions):
        subset = release_df[release_df["Condition"] == cond]
        color = colors[i]
        ax.plot(
            subset["Time"],
            subset["Mean"],
            color=color,
            linewidth=2,
            label=cond,
        )
        ax.fill_between(
            subset["Time"],
            subset["Mean"] - subset["SEM"],
            subset["Mean"] + subset["SEM"],
            color=color,
            alpha=0.3,
        )

    ax.set_xlabel("Time / min")
    ax.set_ylabel("Calcein Release (%)")
    ax.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc="upper left")
    format_axes(ax)

    plt.tight_layout()
    save_plot(fig, "eggpc-calcein-baselines")


if __name__ == "__main__":
    main()
