"""
Laurdan GP analysis: brush density vs GP and spectrum for 6 DNA:lipid ratios.
Two-panel plot: (1) normalised Laurdan spectrum per condition,
               (2) GP value vs Lipid:Construct ratio.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats

from utils.plotting import setup_plot_style, format_axes, save_plot
from utils import defaults

CONDITIONS = ["None", "1:2000", "1:1000", "1:400", "1:200", "1:100"]
REPLICATES = {"A": 0, "B": 1, "C": 2}
# Lipid:Construct ratios (inverse of DNA:Lipid); "None" has no construct
LIPID_CONSTRUCT_X = [2000, 1000, 400, 200, 100]
# Conditions that have a defined Lipid:Construct ratio (exclude "None")
CONDITIONS_WITH_CONSTRUCT = CONDITIONS[1:]


def general_polarization(column: pd.Series) -> float:
    low = column[440]
    high = column[490]
    return float((low - high) / (low + high))


def normalize_column(column: pd.Series, blank_df: pd.DataFrame) -> pd.Series:
    letter = column.name[-2]
    number = int(column.name[-1])
    blank = blank_df[f"Sample {letter}7"]
    blanked = column - blank
    norm_factor = blanked[440] + blanked[490]
    normed = blanked / norm_factor
    normed.name = f"{CONDITIONS[number - 1]} {REPLICATES[letter]}"
    return normed


def main() -> None:
    file_name = "2024-05-22 Laurdan POPC Brush.csv"

    raw_data = (
        pd.read_csv(file_name, header=1, index_col="Unnamed: 0")
        .iloc[1:150]
        .astype(float)
    )
    raw_data.index = [int(float(i)) for i in raw_data.index]

    # Normalise each of the 18 sample columns (A1-A6, B1-B6, C1-C6)
    good_columns = [
        f"Sample {letter}{j+1}"
        for letter in ["A", "B", "C"]
        for j in range(6)
    ]
    normed_data = raw_data[good_columns].apply(
        lambda col: normalize_column(col, raw_data)
    )
    # Fix column names (apply may rename based on output Series name)
    normed_data.columns = [
        f"{CONDITIONS[j]} {REPLICATES[letter]}"
        for letter in ["A", "B", "C"]
        for j in range(6)
    ]

    # Compute per-replicate GP values
    gp_data: dict[str, list[float]] = {c: [] for c in CONDITIONS}
    for letter in ["A", "B", "C"]:
        for j, cond in enumerate(CONDITIONS):
            col_name = f"{cond} {REPLICATES[letter]}"
            gp_data[cond].append(general_polarization(normed_data[col_name]))

    gp_means = {c: np.mean(gp_data[c]) for c in CONDITIONS}
    gp_sems = {c: stats.sem(gp_data[c]) for c in CONDITIONS}

    # Compute mean and SEM spectra per condition
    wavelengths = normed_data.index.to_numpy(dtype=float)
    spec_means = []
    spec_sems = []
    for cond in CONDITIONS:
        reps = normed_data[[f"{cond} {i}" for i in range(3)]].to_numpy()
        spec_means.append(reps.mean(axis=1))
        spec_sems.append(stats.sem(reps, axis=1))

    # ── Plotting ──────────────────────────────────────────────────────────────
    setup_plot_style()

    # Gradient palette across 6 conditions
    cmap = matplotlib.colormaps["viridis"]
    colors = [cmap(i / 5) for i in range(6)]

    # Build figure: ax1 | spacer | ax2_left // ax2_right
    # Flat GridSpec avoids tight_layout incompatibility with nested GridSpec.
    # Column 1 is an empty spacer that provides the inter-panel gap; wspace=0
    # keeps ax2_left and ax2_right directly adjacent for the break effect.
    fig = plt.figure(figsize=(defaults.double_fig_width, defaults.double_fig_height))
    gs = GridSpec(1, 4, figure=fig, width_ratios=[2.5, 0.35, 0.5, 2.0], wspace=0.0)
    ax1 = fig.add_subplot(gs[0])
    ax2_left = fig.add_subplot(gs[2])
    ax2_right = fig.add_subplot(gs[3], sharey=ax2_left)

    # Display labels for legend: "None" → "No construct", others show Lipid:Construct value
    display_labels = ["No construct"] + [str(x) for x in LIPID_CONSTRUCT_X]

    # Panel 1 – normalised spectrum
    for i, cond in enumerate(CONDITIONS):
        color = colors[i]
        ax1.plot(
            wavelengths, spec_means[i], color=color, linewidth=1.5, label=display_labels[i]
        )
        ax1.fill_between(
            wavelengths,
            spec_means[i] - spec_sems[i],
            spec_means[i] + spec_sems[i],
            color=color,
            alpha=0.25,
        )

    ax1.set_xlabel(r"$\lambda$ / nm")
    ax1.set_ylabel("Normalised Intensity")
    ax1.legend(
        title="Lipid:Construct",
        frameon=False,
        fontsize=6,
        title_fontsize=6,
        loc="upper right",
    )
    format_axes(ax1)

    # Panel 2 left – "No construct" at a single x position
    ax2_left.scatter(
        [0] * 3,
        gp_data["None"],
        color=colors[0],
        edgecolors="black",
        linewidths=0.5,
        s=20,
        zorder=5,
    )
    ax2_left.set_xlim(-0.5, 0.5)
    ax2_left.set_xticks([0])
    ax2_left.set_xticklabels(["No\nconstruct"], fontsize=6)
    ax2_left.set_ylabel("GP (440 nm, 490 nm)")
    format_axes(ax2_left)

    # Panel 2 right – GP vs Lipid:Construct ratio on log scale
    for i, cond in enumerate(CONDITIONS_WITH_CONSTRUCT):
        ax2_right.scatter(
            [LIPID_CONSTRUCT_X[i]] * 3,
            gp_data[cond],
            color=colors[i + 1],
            edgecolors="black",
            linewidths=0.5,
            s=20,
            zorder=5,
        )

    ax2_right.set_xscale("log")
    ax2_right.invert_xaxis()
    ax2_right.set_xlabel("Lipid:Construct Ratio")
    format_axes(ax2_right)
    # Hide left spine and ticks – the y-axis is carried by ax2_left
    ax2_right.spines["left"].set_visible(False)
    ax2_right.tick_params(left=False, labelleft=False)

    # Draw axis-break diagonal lines at consistent physical angle.
    # Both marks are "/" (lower-left to upper-right). ax2_right is 4× wider
    # (width_ratio 2.0 vs 0.5), so its d_x in axes coords must be scaled down
    # by the same factor to keep the same physical slope.
    w_left = 0.5   # GridSpec width_ratio for ax2_left
    w_right = 2.0  # GridSpec width_ratio for ax2_right
    d_y = 0.08
    d_x_left = d_y
    d_x_right = d_y * (w_left / w_right)
    break_kw = dict(color="k", clip_on=False, linewidth=0.8)
    ax2_left.plot(
        [1 - d_x_left, 1 + d_x_left], [-d_y, +d_y],
        transform=ax2_left.transAxes, **break_kw,
    )
    ax2_right.plot(
        [-d_x_right, +d_x_right], [-d_y, +d_y],
        transform=ax2_right.transAxes, **break_kw,
    )

    # Custom save: target 40×30 mm per axis (slightly larger than default 30×24 mm)
    plt.tight_layout(pad=0.5)
    pos = ax1.get_position()
    ax_w_mm, ax_h_mm = 40.0, 30.0
    fig.set_size_inches((ax_w_mm / 25.4) / pos.width, (ax_h_mm / 25.4) / pos.height)
    plt.tight_layout(pad=0.5)
    fig.savefig("laurdan-brush-density.svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
