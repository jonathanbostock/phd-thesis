"""
GUV MBCD detachment: ΔD (nm) vs time for each MBCD concentration.
Gradient colormap for numeric MBCD concentrations.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from utils.plotting import setup_plot_style, format_axes, save_plot
from utils import defaults


def main() -> None:
    raw_data = pd.read_csv("2023-10-12 Dextrins.csv", index_col="Sample Name").loc[
        :, "Peak 1 Mean by Intensity ordered by area (nm)"
    ]

    concs = [10, 5, 2]
    times = ["Before", "1min", "30min"]
    time_labels = {"Before": 0, "1min": 1, "30min": 30}

    rows = []
    for conc in concs:
        for time in times:
            ctrl_key = f"Ctrl {conc} Dextrin {time}"
            dna_key = f"DNA {conc} Dextrin {time}"
            delta_d = raw_data.loc[dna_key] - raw_data.loc[ctrl_key]
            rows.append(
                {
                    "Concentration (mM)": conc,
                    "Time (min)": time_labels[time],
                    "Delta D (nm)": delta_d,
                }
            )

    delta_d_df = pd.DataFrame(rows)

    setup_plot_style()

    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=min(concs), vmax=max(concs))
    markers = ["o", "s", "^"]

    fig, ax = plt.subplots(figsize=(defaults.fig_width, defaults.fig_height))

    for i, conc in enumerate(concs):
        subset = delta_d_df[delta_d_df["Concentration (mM)"] == conc].sort_values(  # type: ignore[call-overload]
            "Time (min)"
        )
        color = cmap(norm(conc))
        ax.scatter(
            subset["Time (min)"],
            subset["Delta D (nm)"],
            color=color,
            marker=markers[i],
            edgecolors="black",
            linewidths=0.5,
            s=50,
            label=f"{conc} mM MBCD",
            zorder=5,
        )
        ax.plot(
            subset["Time (min)"],
            subset["Delta D (nm)"],
            color=color,
            linewidth=1.5,
            alpha=0.7,
        )

    ax.set_xlabel("Time / min")
    ax.set_ylabel(r"$\Delta D$ / nm")
    ax.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc="upper left")
    format_axes(ax)

    plt.tight_layout()
    save_plot(fig, "guv-mbcd-detachment")


if __name__ == "__main__":
    main()
