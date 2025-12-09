"""
Analysis of DNA brush attachment data
Plots delta D over time with exponential fits
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

from utils.plotting import setup_plot_style, format_axes, save_plot
from utils import defaults


def exponential_attachment(t, delta_d_max, k, t0):
    """
    Exponential attachment model: delta_D_max * (1 - exp(-k * (t - t0)))
    For t < t0, returns 0
    """
    result = np.zeros_like(t, dtype=float)
    mask = t >= t0
    result[mask] = delta_d_max * (1 - np.exp(-k * (t[mask] - t0)))
    return result


def main():
    """Main analysis function"""
    # Load data
    df = pd.read_csv("figures/fig-3/Brush Attachment/attachment-data.csv")

    # Parse the data
    # Convert 'time (min)' to numeric, keeping 'ctrl' as string
    df["time_numeric"] = pd.to_numeric(df["time (min)"], errors="coerce")

    # Set up plotting style
    setup_plot_style()

    # Get unique batches
    batches = df["batch"].unique()

    # Use single color from colorblind palette
    colors = sns.color_palette("colorblind")
    color = colors[0]

    # Create figure
    fig, ax = plt.subplots(figsize=(defaults.fig_width, defaults.fig_height))

    # Collect all data points
    all_times = []
    all_delta_d = []

    # Process each batch to calculate delta D
    for batch in batches:
        batch_data = df[df["batch"] == batch].copy()

        # Get control value for this batch
        ctrl_row = batch_data[batch_data["time (min)"] == "ctrl"]
        if len(ctrl_row) == 0:
            print(f"Warning: No control found for batch {batch}")
            continue

        ctrl_p1mi_series = ctrl_row["p1mi"]
        ctrl_p1mi = float(ctrl_p1mi_series.iloc[0])  # pyright: ignore[reportAttributeAccessIssue]

        # Get non-control data
        data_rows = batch_data[batch_data["time (min)"] != "ctrl"].copy()

        # Calculate delta D (change from control)
        data_rows["delta_d"] = data_rows["p1mi"] - ctrl_p1mi

        times = np.array(data_rows["time_numeric"])
        delta_d = np.array(data_rows["delta_d"])

        # Collect all data
        all_times.extend(times)
        all_delta_d.extend(delta_d)

        # Plot scatter data for this batch
        ax.scatter(
            times,
            delta_d,
            color=color,
            marker="o",
            edgecolors="black",
            linewidths=0.5,
            s=50,
            zorder=3,
        )

    # Convert to numpy arrays
    all_times = np.array(all_times)
    all_delta_d = np.array(all_delta_d)

    # Fit single exponential curve to all data
    try:
        # Initial guess: max delta_d from data, k=0.1, t0=0
        p0 = [max(all_delta_d), 0.1, 0]

        # Set bounds: delta_d_max > 0, k > 0, t0 >= 0
        bounds = ([0, 0, 0], [np.inf, np.inf, max(all_times)])

        popt, pcov = curve_fit(
            exponential_attachment,
            all_times,
            all_delta_d,
            p0=p0,
            bounds=bounds,
            maxfev=10000,
        )

        delta_d_max, k, t0 = popt

        # Generate smooth curve
        t_smooth = np.linspace(0, max(all_times) * 1.1, 200)
        delta_d_fit = exponential_attachment(t_smooth, *popt)

        # Plot fitted curve
        ax.plot(t_smooth, delta_d_fit, color=color, linewidth=2, alpha=0.8, zorder=2)

        # Calculate uncertainty using Monte Carlo
        n_samples = 1000
        param_samples = np.random.multivariate_normal(popt, pcov, n_samples)

        # Ensure positive parameters
        param_samples = np.abs(param_samples)

        delta_d_samples = []
        for params in param_samples:
            delta_d_sample = exponential_attachment(t_smooth, *params)
            delta_d_samples.append(delta_d_sample)

        delta_d_samples = np.array(delta_d_samples)
        delta_d_lower = np.percentile(delta_d_samples, 16, axis=0)
        delta_d_upper = np.percentile(delta_d_samples, 84, axis=0)

        # Plot error band
        ax.fill_between(
            t_smooth, delta_d_lower, delta_d_upper, color=color, alpha=0.3, zorder=1
        )

        print("\nFit parameters (all batches combined):")
        print(f"  delta_D_max = {delta_d_max:.2f} +/- {np.sqrt(pcov[0, 0]):.2f} nm")
        print(f"  k = {k:.4f} +/- {np.sqrt(pcov[1, 1]):.4f} min^-1")
        print(f"  t_0 = {t0:.2f} +/- {np.sqrt(pcov[2, 2]):.2f} min")

        # Add fit parameters as text on plot
        param_text = (
            f"$\\Delta D_{{max}}$ = {delta_d_max:.1f} ± {np.sqrt(pcov[0, 0]):.1f} nm\n"
            f"$k$ = {k:.3f} ± {np.sqrt(pcov[1, 1]):.3f} min$^{{-1}}$\n"
            f"$t_0$ = {t0:.1f} ± {np.sqrt(pcov[2, 2]):.1f} min"
        )
        ax.text(
            0.95,
            0.05,
            param_text,
            transform=ax.transAxes,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    except (RuntimeError, ValueError) as e:
        print(f"Could not fit curve: {e}")

    # Format plot
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(r"$\Delta D$ (nm)")
    ax.set_title("DNA Brush Attachment Kinetics")
    ax.set_ylim(bottom=0)
    format_axes(ax)

    plt.tight_layout()

    # Save plot
    save_plot(fig, "figures/fig-3/Brush Attachment/Attachment Kinetics")

    print("\nPlot saved as 'Attachment Kinetics.svg'")

    plt.show()


if __name__ == "__main__":
    main()
