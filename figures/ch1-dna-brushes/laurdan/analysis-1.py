"""
Laurdan GP analysis: bar chart of General Polarization values for None/Sparse/Dense brush conditions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from utils.plotting import setup_plot_style, format_axes, save_plot
from utils import defaults


def general_polarization(column: pd.Series) -> float:
    low = column[435]
    high = column[500]
    return float((low - high) / (low + high))


def main() -> None:
    file_name = "2024-05-21 Laurdan POPC Brush Ex360.csv"

    raw_data = (
        pd.read_csv(file_name, header=1, index_col="Unnamed: 0")
        .iloc[1:150]
        .astype(float)
    )
    raw_data.index = [int(float(i)) for i in raw_data.index]

    names = [
        "None 0",
        "None 1",
        "None 2",
        "Sparse 0",
        "Sparse 1",
        "Sparse 2",
        "Dense 0",
        "Dense 1",
        "Dense 2",
    ]
    conditions = ["None", "Sparse", "Dense"]

    raw_data.columns = names + ["Blank"]

    for name in names:
        raw_data[f"{name} Blanked"] = raw_data[name] - raw_data["Blank"]

    gp_values = {
        name: general_polarization(raw_data[f"{name} Blanked"]) for name in names
    }

    gp_means = []
    gp_sems = []
    for cond in conditions:
        vals = [gp_values[f"{cond} {i}"] for i in range(3)]
        gp_means.append(np.mean(vals))
        gp_sems.append(stats.sem(vals))

    setup_plot_style()
    colors = sns.color_palette("colorblind", n_colors=len(conditions))

    fig, ax = plt.subplots(figsize=(defaults.fig_width, defaults.fig_height))

    for i, (cond, mean, sem) in enumerate(zip(conditions, gp_means, gp_sems)):
        ax.bar(
            i,
            mean,
            yerr=sem,
            color=colors[i],
            edgecolor="black",
            linewidth=0.5,
            capsize=4,
            error_kw={"linewidth": 1.5},
        )
        raw_vals = [gp_values[f"{cond} {j}"] for j in range(3)]
        jitter = np.linspace(-0.1, 0.1, 3)
        ax.scatter(
            i + jitter,
            raw_vals,
            color=colors[i],
            edgecolor="black",
            linewidths=0.5,
            s=30,
            zorder=5,
        )

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions)
    ax.set_xlabel("Brush Condition")
    ax.set_ylabel("General Polarization / GP")
    format_axes(ax)

    plt.tight_layout()
    save_plot(fig, "laurdan-gp")


if __name__ == "__main__":
    main()
