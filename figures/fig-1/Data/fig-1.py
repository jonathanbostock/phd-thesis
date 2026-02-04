# Jonathan Bostock
# Figure 1: Combined Length vs Concentration and Effect of Shape analysis
# Four-panel plot with legends on the right of each row

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib import colormaps
import seaborn as sns
from pathlib import Path

from utils.data_processing import process_length_data, process_shape_data
from utils.plotting import (
    setup_plot_style,
    format_axes,
    plot_fit_curve,
    plot_error_ellipse,
    save_plot,
)
from utils import defaults


def main() -> None:
    setup_plot_style()

    data_dir = Path(__file__).parent

    # --- Load and process data ---
    df_length = process_length_data(pd.read_csv(data_dir / "static-brush-data.csv"))
    df_shape = process_shape_data(
        pd.read_csv(data_dir / "shape-brush-data.csv"),
        dna_concentration_uM=2.5,
        control_name="control",
    )

    # --- Figure setup: 2 rows x 3 cols (col 3 is narrow for legends) ---
    fig = plt.figure(figsize=(defaults.fig_width * 2.2, defaults.fig_height * 2.4))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.05], hspace=0.4, wspace=0.45)
    ax1 = fig.add_subplot(gs[0, 0])  # Top-left: length dose-response
    ax2 = fig.add_subplot(gs[0, 1])  # Top-right: ΔD_max vs brush length
    ax3 = fig.add_subplot(gs[1, 0])  # Bottom-left: shape dose-response
    ax4 = fig.add_subplot(gs[1, 1])  # Bottom-right: shape params

    # =====================================================================
    # TOP ROW: Length vs Concentration
    # =====================================================================
    brush_lengths = sorted(df_length["Brush Length / bp"].unique())
    cmap = colormaps["viridis"]
    norm = Normalize(vmin=min(brush_lengths), vmax=max(brush_lengths))
    markers = ["o", "s", "^", "D"][: len(brush_lengths)]

    # Set axis limits for curves to extend to full range
    all_ratios = df_length["Lipid:DNA Ratio"].values
    min_ratio = min(all_ratios) * 0.3
    max_ratio = max(all_ratios) * 3.0
    ax1.set_xlim(max_ratio, min_ratio)  # Reversed for log scale

    fitted_params = []

    for i, bl in enumerate(brush_lengths):
        subset = df_length[df_length["Brush Length / bp"] == bl]
        color = cmap(norm(bl))

        ax1.scatter(
            subset["Lipid:DNA Ratio"],
            subset["Delta D"],
            color=color,
            marker=markers[i],
            edgecolors="black",
            linewidths=0.5,
            label=f"{int(bl)} bp",
            s=50,
        )

        popt, pcov = plot_fit_curve(
            ax1,
            subset,
            "Concentration",
            "Delta D",
            "Lipid:DNA Ratio",
            df_length["Concentration"].values,
            color,
        )

        if popt is not None and pcov is not None:
            fitted_params.append(
                {
                    "brush_length": bl,
                    "delta_d_max": popt[0],
                    "delta_d_max_error": np.sqrt(pcov[0, 0]),
                    "color": color,
                    "marker": markers[i],
                }
            )

    ax1.set_xscale("log")
    ax1.set_xlabel("Lipid:DNA Ratio")
    ax1.set_ylabel(r"$\Delta D$")
    ax1.set_title(r"Static Brush $\Delta D$ vs Lipid:DNA Ratio")
    format_axes(ax1)

    # Legend for top row — placed off to the right
    ax1.legend(
        title="Brush Length",
        bbox_to_anchor=(2.25, 1),
        loc="upper left",
        frameon=False,
    )

    # Top-right panel: ΔD_max vs Brush Length
    if fitted_params:
        bl_fit = [p["brush_length"] for p in fitted_params]
        ddm_vals = [p["delta_d_max"] for p in fitted_params]
        ddm_errs = [p["delta_d_max_error"] for p in fitted_params]
        colors_fit = [p["color"] for p in fitted_params]
        markers_fit = [p["marker"] for p in fitted_params]

        for j, (bl, ddm, err, c, m) in enumerate(
            zip(bl_fit, ddm_vals, ddm_errs, colors_fit, markers_fit)
        ):
            ax2.errorbar(
                bl,
                ddm,
                yerr=err,
                marker=m,
                color=c,
                markeredgecolor="black",
                markeredgewidth=0.5,
                markersize=8,
                capsize=3,
                linewidth=0.5,
            )

        # Linear fit through origin
        x_arr = np.array(bl_fit)
        y_arr = np.array(ddm_vals)
        slope = np.sum(x_arr * y_arr) / np.sum(x_arr**2)

        x_fit = np.linspace(0, max(bl_fit) * 1.1, 100)
        ax2.plot(
            x_fit,
            slope * x_fit,
            "k--",
            linewidth=2,
            alpha=0.7,
            label=f"Slope = {slope:.3f}",
        )

        ax2.set_xlabel("Brush Length (bp)")
        ax2.set_ylabel(r"$\Delta D_{max}$")
        ax2.set_title(r"Fitted $\Delta D_{max}$ vs Brush Length")
        ax2.legend(frameon=False)
        ax2.set_xlim(0, max(bl_fit) * 1.1)
        ax2.set_ylim(0, max(ddm_vals) * 1.1)
        format_axes(ax2)

    # =====================================================================
    # BOTTOM ROW: Effect of Shape
    # =====================================================================
    shapes = sorted(df_shape["Shape"].unique())
    shape_colors = sns.color_palette("colorblind", n_colors=len(shapes))
    shape_markers = ["o", "s", "^", "D"][: len(shapes)]

    all_ratios_shape = df_shape["Lipid:DNA Ratio"].values
    min_ratio_s = min(all_ratios_shape) * 0.3
    max_ratio_s = max(all_ratios_shape) * 3.0
    ax3.set_xlim(min_ratio_s, max_ratio_s)

    for i, shape in enumerate(shapes):
        subset = df_shape[df_shape["Shape"] == shape]
        color = shape_colors[i]

        ax3.scatter(
            subset["Lipid:DNA Ratio"],
            subset["Delta D"],
            color=color,
            marker=shape_markers[i],
            edgecolors="black",
            linewidths=0.5,
            label=f"{shape}",
            s=50,
        )

        param_mean, param_cov = plot_fit_curve(
            ax3,
            subset,
            "Concentration",
            "Delta D",
            "Lipid:DNA Ratio",
            df_shape["Concentration"].values,
            color,
        )

        if param_mean is None or param_cov is None:
            print(f"Could not fit curve for shape: {shape}")
            continue

        # Flip x and y for the parameter plot (c_1/2 on x, ΔD_max on y)
        param_mean_flip = param_mean[::-1]
        param_cov_flip = param_cov[::-1, ::-1]

        ax4.scatter(
            [np.pow(10, param_mean_flip[0])],
            [param_mean_flip[1]],
            color=color,
            marker=shape_markers[i],
            edgecolors="black",
            linewidths=0.5,
            label=f"{shape}",
            s=50,
        )

        plot_error_ellipse(
            param_mean_flip, param_cov_flip, ax=ax4, color=color, x_exponent_base=10
        )

    ax3.set_xscale("log")
    xlim = ax3.get_xlim()
    ax3.set_xlim(xlim[1], xlim[0])  # Reverse x-axis
    ax3.set_xlabel("Lipid:DNA Ratio")
    ax3.set_ylabel(r"$\Delta D$")
    ax3.set_title(r"Shape Effect on Brush $\Delta D$")
    format_axes(ax3)

    # Legend for bottom row — placed off to the right
    ax3.legend(
        title="DNA Shape",
        bbox_to_anchor=(2.25, 1),
        loc="upper left",
        frameon=False,
    )

    ax4.set_xlabel("$c_{1/2}$")
    ax4.set_xscale("log")
    ax4.set_ylabel(r"$\Delta D_{max} / nm$")
    ax4.set_title(r"Shape Effect on $\Delta D_{max}$ and $c_{1/2}$")
    format_axes(ax4)

    save_plot(fig, "Delta D")
    plt.show()


if __name__ == "__main__":
    main()
