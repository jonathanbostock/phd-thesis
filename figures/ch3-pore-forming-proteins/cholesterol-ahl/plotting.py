"""Plotting script for cholesterol + alpha hemolysin calcein release experiment.

Analyse timecourses of calcein leakage with POPC and DOPC at 7:3, 6:4, and 5:5 ratios.
Used three different batches of Ahl to compare.

Original analysis by Jonathan Bostock 2024-10-31.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import os

import utils.plotting as plotting


def main():
    current_path = os.path.dirname(os.path.abspath(__file__))

    kwargs = {
        "header": 1,
        "engine": "python",
        "usecols": ["Time (min)", "Intensity (a.u.)"]
        + [f"Intensity (a.u.).{i}" for i in range(1, 24)],
    }

    start_df = pd.read_csv(
        os.path.join(
            current_path, "2024-10-31 Calcein Leak High Cholesterol Before 700V.csv"
        ),
        **kwargs,
    )[:2].astype(float)
    end_df = pd.read_csv(
        os.path.join(
            current_path, "2024-10-31 Calcein Leak High Cholesterol After 700V.csv"
        ),
        **kwargs,
    )[:2].astype(float)
    timecourse_df = pd.read_csv(
        os.path.join(
            current_path,
            "2024-10-31 Calcein Leak High Cholesterol Timecourse 700V.csv",
        ),
        **kwargs,
    )[:121].astype(float)

    cell_data = [
        {"Chol %": chol_percent, "Lipid": lipid, "Ahl Batch": ahl_batch}
        for chol_percent in [30, 40, 50]
        for lipid in ["POPC", "DOPC"]
        for ahl_batch in ["None", "January", "February", "August"]
    ]

    cell_data_lists = {
        key: value
        for key, value in zip(
            cell_data[0].keys(), map(list, zip(*map(lambda x: x.values(), cell_data)))
        )
    }

    start_means = start_df.apply(np.mean)
    end_means = end_df.apply(np.mean)

    final_release = timecourse_df.T[1:].apply(
        lambda row: (row[120] - start_means[row.name])
        / (end_means[row.name] - start_means[row.name])
        * 100,
        axis=1,
    )

    final_release_df = final_release.to_frame(name="Release %")

    for key, value in cell_data_lists.items():
        final_release_df[key] = value

    # Set up the plot style
    plotting.setup_plot_style()

    # Create the figure
    fig, ax = plt.subplots(figsize=plotting.default_figsize)
    # Reserve right margin so long legend labels (e.g., "February") stay within the canvas
    fig.subplots_adjust(right=0.78)

    # Get unique values for color and shape
    ahl_batches = final_release_df["Ahl Batch"].unique()
    lipids = final_release_df["Lipid"].unique()

    colors = sns.color_palette("colorblind", n_colors=len(ahl_batches))
    markers = ["o", "s"]  # Circle for POPC, square for DOPC

    # Create a mapping for colors and markers
    color_map = {batch: colors[i] for i, batch in enumerate(ahl_batches)}
    marker_map = {lipid: markers[i] for i, lipid in enumerate(lipids)}

    # Plot each combination
    for ahl_batch in ahl_batches:
        for lipid in lipids:
            subset = final_release_df[
                (final_release_df["Ahl Batch"] == ahl_batch)
                & (final_release_df["Lipid"] == lipid)
            ]

            ax.scatter(
                subset["Chol %"],
                subset["Release %"],
                color=color_map[ahl_batch],
                marker=marker_map[lipid],
                edgecolors="black",
                linewidths=0.5,
                s=50,
                label=f"{ahl_batch} ({lipid})",
            )

    # Connect points with lines (grouped by Ahl Batch and Lipid)
    for ahl_batch in ahl_batches:
        for lipid in lipids:
            subset = final_release_df[
                (final_release_df["Ahl Batch"] == ahl_batch)
                & (final_release_df["Lipid"] == lipid)
            ].sort_values(by="Chol %")

            ax.plot(
                subset["Chol %"],
                subset["Release %"],
                color=color_map[ahl_batch],
                linestyle="-" if lipid == "POPC" else "--",
                linewidth=1,
                alpha=0.7,
            )

    ax.set_xlabel("Chol %")
    ax.set_ylabel("Release % after 2h")
    ax.set_xticks([30, 40, 50])

    # Create a custom legend
    # First, legend for Ahl Batch (colors)
    batch_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_map[batch],
            markersize=8,
            markeredgecolor="black",
            label=batch,
        )
        for batch in ahl_batches
    ]

    # Second, legend for Lipid (markers)
    lipid_handles = [
        Line2D(
            [0],
            [0],
            marker=marker_map[lipid],
            color="w",
            markerfacecolor="gray",
            markersize=8,
            markeredgecolor="black",
            label=lipid,
        )
        for lipid in lipids
    ]

    # Combine legends
    first_legend = ax.legend(
        handles=batch_handles,
        title="Ahl Batch",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=False,
    )
    ax.add_artist(first_legend)
    ax.legend(
        handles=lipid_handles,
        title="Base Lipid",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )

    plotting.format_axes(ax)

    fig.savefig(
        os.path.join(current_path, "Release.svg"), bbox_inches="tight", pad_inches=0.1
    )


if __name__ == "__main__":
    main()
