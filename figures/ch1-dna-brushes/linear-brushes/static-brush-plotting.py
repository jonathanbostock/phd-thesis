# Jonathan Bostock
# 26-Apr-2023
# Analysing the data for the static, dsDNA brushes

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from itertools import product

from utils.plotting import (
    fit_function,
    setup_plot_style,
    format_axes,
    plot_fit_curve,
    save_plot,
)
from utils import defaults


def load_data() -> pd.DataFrame:
    """Read the DLS data and subtract the per-batch control measurements."""
    df = pd.read_csv("static-brush-data.csv")

    controls = df["sample_type"] == "ctrl"

    def process_row(row, controls):
        control_data = controls[controls["batch"] == row["batch"]]
        subtract_columns = [
            c not in ["batch", "sample_type", "sample_number"]
            for c in control_data.columns
        ]
        control_row = control_data.loc[..., subtract_columns].mean()

        new_row = row.copy()
        new_row[subtract_columns] = new_row[subtract_columns] - control_row
        return new_row

    df_ctrl = df[controls]
    df_data = df[~controls].apply(lambda row: process_row(row, df_ctrl), axis=1)
    assert isinstance(df_data, pd.DataFrame)

    # 5 microliters of 0.5 mg/mL POPC (molar mass 760)
    lipid_moles = 5e-6 * 0.5 / 760
    # microliters of 5 micromolar DNA
    df_data["dna_moles"] = df_data["sample_number"] * 1e-6 * 5e-6
    df_data["Construct Length / bp"] = list(map(float, df_data["sample_type"]))
    df_data["Lipid:Construct Ratio"] = (lipid_moles / df_data["dna_moles"]).astype(int)
    df_data["Concentration"] = (
        1 / df_data["Lipid:Construct Ratio"]
    )  # Concentration is inverse of ratio
    df_data["Delta D"] = df_data["peak_1_mean_intensity"]
    return df_data


def plot_saturation_curves(ax, df_data, brush_lengths, cmap, norm, markers) -> list:
    """Delta D against lipid:construct ratio with one saturation fit per construct length.

    Returns the fitted parameters (one dict per construct length), which feed the
    Delta D_max vs length plot.
    """
    # Store fitted parameters for second plot
    fitted_params = []

    # Set axis limits first to ensure curves extend to full range
    all_ratios = np.array(df_data["Lipid:Construct Ratio"])
    min_ratio = min(all_ratios) * 0.3  # Extend further left
    max_ratio = max(all_ratios) * 3.0  # Extend further right
    ax.set_xlim(max_ratio, min_ratio)  # Reversed for log scale

    # Plot individual data points and fit curves
    for i, brush_length in enumerate(brush_lengths):
        data_subset = df_data[df_data["Construct Length / bp"] == brush_length]
        color = cmap(norm(brush_length))
        ax.scatter(
            data_subset["Lipid:Construct Ratio"],
            data_subset["Delta D"],
            color=color,
            marker=markers[i],
            edgecolors="black",
            linewidths=0.5,
            label=f"{int(brush_length)} bp",
            s=50,
        )

        # Use shared utility for fitting and plotting
        popt, param_errors = plot_fit_curve(
            ax,
            data_subset,
            "Concentration",
            "Delta D",
            "Lipid:Construct Ratio",
            np.array(df_data["Concentration"]),
            color,
        )

        # Store fitted parameters for second plot
        if popt is not None and param_errors is not None:
            fitted_params.append(
                {
                    "brush_length": brush_length,
                    "delta_d_max": popt[0],
                    "delta_d_max_error": np.sqrt(param_errors[0, 0]),
                    "log_c_half": popt[1],
                    "color": color,
                }
            )

    # Format first subplot (original plot)
    ax.set_xscale("log")
    ax.set_xlabel("Lipid:Construct Ratio")
    ax.set_ylabel(r"$\Delta D$ / nm")
    ax.legend(
        title="Construct Length",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        frameon=False,
    )
    ax.set_title(r"Static Brush $\Delta D$ vs Lipid:Construct Ratio")
    format_axes(ax)

    return fitted_params


def plot_dmax_vs_length(ax, fitted_params, markers) -> None:
    """Fitted Delta D_max against construct length, with a line through the origin."""
    if not fitted_params:
        return

    # Extract data for plotting
    brush_lengths_fit = [p["brush_length"] for p in fitted_params]
    delta_d_max_vals = [p["delta_d_max"] for p in fitted_params]
    delta_d_max_errors = [p["delta_d_max_error"] for p in fitted_params]

    print(delta_d_max_errors)

    colors = [p["color"] for p in fitted_params]

    # Plot with error bars
    for i, (bl, ddm, err, color) in enumerate(
        zip(brush_lengths_fit, delta_d_max_vals, delta_d_max_errors, colors)
    ):
        ax.errorbar(
            bl,
            ddm,
            yerr=err,
            marker=markers[i],
            color=color,
            markeredgecolor="black",
            markeredgewidth=0.5,
            markersize=8,
            capsize=3,
            linewidth=0.5,
        )

    # Fit a line through the origin
    # Force intercept to be 0 by fitting y = mx model
    x_data = np.array(brush_lengths_fit)
    y_data = np.array(delta_d_max_vals)

    # Fit slope (forcing through origin)
    slope = np.sum(x_data * y_data) / np.sum(x_data**2)

    # Plot fitted line
    x_fit = np.linspace(0, max(brush_lengths_fit) * 1.1, 100)
    y_fit = slope * x_fit
    ax.plot(x_fit, y_fit, "k--", linewidth=2, alpha=0.7, label=f"Slope = {slope:.3f}")

    # Format second subplot
    ax.set_xlabel("Construct Length / bp")
    ax.set_ylabel(r"$\Delta D_{max}$ / nm")
    ax.set_title(r"Fitted $\Delta D_{max}$ vs Construct Length")
    ax.legend(frameon=False)

    # Set origin at (0,0)
    ax.set_xlim(0, max(brush_lengths_fit) * 1.1)
    ax.set_ylim(0, max(delta_d_max_vals) * 1.1)
    format_axes(ax)


def main() -> None:
    setup_plot_style()

    df_data = load_data()

    # Get unique brush lengths and create gradient colormap
    brush_lengths = sorted(np.unique(df_data["Construct Length / bp"]))
    # Create a colormap for gradient colors based on brush length
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    from matplotlib import colormaps

    cmap = colormaps["viridis"]
    norm = Normalize(vmin=min(brush_lengths), vmax=max(brush_lengths))
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"][: len(brush_lengths)]

    # Combined two-panel figure (the plot used in the original composite figure)
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(defaults.fig_width * 2, defaults.fig_height)
    )
    fitted_params = plot_saturation_curves(
        ax1, df_data, brush_lengths, cmap, norm, markers
    )
    plot_dmax_vs_length(ax2, fitted_params, markers)
    save_plot(fig, "Static Brush Plot")

    # The same two panels rendered individually for the thesis figures
    # linear-constructs-b and linear-constructs-c (panel a is drawn in Inkscape;
    # see split-panels.py).
    fig_b, ax_b = plt.subplots(figsize=(defaults.fig_width, defaults.fig_height))
    fitted_params_b = plot_saturation_curves(
        ax_b, df_data, brush_lengths, cmap, norm, markers
    )
    save_plot(fig_b, "linear-constructs-b")

    fig_c, ax_c = plt.subplots(figsize=(defaults.fig_width, defaults.fig_height))
    plot_dmax_vs_length(ax_c, fitted_params_b, markers)
    save_plot(fig_c, "linear-constructs-c")

    plt.show()


if __name__ == "__main__":
    main()
