"""
Combined analysis of DLS kinetics data from 2023 and 2026
Plots delta D over time with exponential fits using time subtraction method
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


def process_2023_data(filepath):
    """
    Process 2023 data with time subtraction method.

    The 2023 data has format: "Ext N T min" or "Ext N Ctrl 1"
    where N is the series number (1, 2, 3) and T is the time point.
    """
    df = pd.read_csv(filepath)

    # Parse datetime column
    df["datetime"] = pd.to_datetime(
        df["Measurement Start Date And Time"], format="%d/%m/%Y %H:%M"
    )

    # Filter only GoodData entries
    df = df[df["Quality Indicator"] == "GoodData"].copy()

    all_times = []
    all_delta_d = []

    # Process each series (Ext 1, Ext 2, Ext 3)
    for series_num in [1, 2, 3]:
        # Get baseline value (Ctrl)
        baseline_row = df[df["Sample Name"] == f"Ext {series_num} Ctrl 1"]
        if len(baseline_row) == 0:
            print(f"Warning: No baseline found for Ext {series_num}")
            continue

        baseline_peak1 = float(
            baseline_row["Peak 1 Mean by Intensity ordered by area (nm)"].iloc[  # pyright: ignore[reportAttributeAccessIssue]
                0
            ]
        )

        # Get time zero (0 min measurement)
        start_row = df.loc[df["Sample Name"] == f"Ext {series_num} 0 min"]
        if len(start_row) == 0:
            print(f"Warning: No start time found for Ext {series_num}")
            continue

        time_zero = start_row["datetime"].iloc[0]  # pyright: ignore[reportAttributeAccessIssue]

        # Get all kinetics data for this series (all time points)
        # Match pattern "Ext N T min" where T is a number
        sample_names = df["Sample Name"]
        series_data = df.loc[
            sample_names.str.match(f"Ext {series_num} \\d+ min")  # pyright: ignore[reportAttributeAccessIssue]
        ].copy()

        if len(series_data) == 0:
            print(f"Warning: No kinetics data found for Ext {series_num}")
            continue

        # Calculate time in minutes from timestamps
        series_data["time_min"] = (
            series_data["datetime"] - time_zero
        ).dt.total_seconds() / 60

        # Calculate delta D
        series_data["delta_d"] = (
            series_data["Peak 1 Mean by Intensity ordered by area (nm)"]
            - baseline_peak1
        )

        times = np.array(series_data["time_min"])
        delta_d = np.array(series_data["delta_d"])

        # Collect all data
        all_times.extend(times)
        all_delta_d.extend(delta_d)

    return np.array(all_times), np.array(all_delta_d)


def process_2026_data(filepath):
    """
    Process 2026 data with time subtraction method.

    The 2026 data has format: "ext-n", "ext-n-before", "ext-n-start"
    where n is the series number (1, 2, 3).
    """
    df = pd.read_csv(filepath)

    # Parse datetime column
    df["datetime"] = pd.to_datetime(
        df["Measurement Start Date And Time"], format="%d/%m/%Y %H:%M"
    )

    all_times = []
    all_delta_d = []

    # Process each sample series (ext-1, ext-2, ext-3)
    for series_num in [1, 2, 3]:
        # Get baseline value (ext-n-before)
        baseline_row = df[df["Sample Name"] == f"ext-{series_num}-before"]
        if len(baseline_row) == 0:
            print(f"Warning: No baseline found for ext-{series_num}")
            continue

        baseline_peak1 = float(
            baseline_row["Peak 1 Mean by Intensity ordered by area (nm)"].iloc[  # pyright: ignore[reportAttributeAccessIssue]
                0
            ]
        )

        # Get time zero (ext-n-start)
        start_row = df.loc[df["Sample Name"] == f"ext-{series_num}-start"]
        if len(start_row) == 0:
            print(f"Warning: No start time found for ext-{series_num}")
            continue

        time_zero = start_row["datetime"].iloc[0]  # pyright: ignore[reportAttributeAccessIssue]

        # Get kinetics data (ext-n)
        kinetics_data = df[df["Sample Name"] == f"ext-{series_num}"].copy()

        if len(kinetics_data) == 0:
            print(f"Warning: No kinetics data found for ext-{series_num}")
            continue

        # Calculate time in minutes
        kinetics_data["time_min"] = (
            kinetics_data["datetime"] - time_zero
        ).dt.total_seconds() / 60

        # Calculate delta D
        kinetics_data["delta_d"] = (
            kinetics_data["Peak 1 Mean by Intensity ordered by area (nm)"]
            - baseline_peak1
        )

        times = np.array(kinetics_data["time_min"])
        delta_d = np.array(kinetics_data["delta_d"])

        # Collect all data
        all_times.extend(times)
        all_delta_d.extend(delta_d)

    return np.array(all_times), np.array(all_delta_d)


def main():
    """Main analysis function"""
    # Set up plotting style
    setup_plot_style()

    # Use colorblind palette
    colors = sns.color_palette("colorblind")
    color = colors[0]  # Use single color for all data

    # Create figure
    fig, ax = plt.subplots(figsize=(defaults.fig_width, defaults.fig_height))

    # Process 2023 data
    print("Processing 2023 data...")
    times_2023, delta_d_2023 = process_2023_data(
        "figures/fig-3/Brush Attachment Combined/2023-06-07 Attachment Timecourse.csv"
    )

    # Plot 2023 data
    ax.scatter(
        times_2023,
        delta_d_2023,
        color=color,
        marker="o",
        edgecolors="black",
        linewidths=0.5,
        s=50,
        zorder=3,
    )

    # Process 2026 data
    print("Processing 2026 data...")
    times_2026, delta_d_2026 = process_2026_data(
        "figures/fig-3/Brush Attachment Combined/2026-01-22 DLS Kinetics.csv"
    )

    # Plot 2026 data
    ax.scatter(
        times_2026,
        delta_d_2026,
        color=color,
        marker="o",
        edgecolors="black",
        linewidths=0.5,
        s=50,
        zorder=3,
    )

    # Combine all data for fitting
    all_times = np.concatenate([times_2023, times_2026])
    all_delta_d = np.concatenate([delta_d_2023, delta_d_2026])

    print(f"\nTotal data points: {len(all_times)}")
    print(f"  2023: {len(times_2023)} points")
    print(f"  2026: {len(times_2026)} points")

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

        # Plot fitted curve (same color as data)
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

        # Plot error band (same color as data)
        ax.fill_between(
            t_smooth, delta_d_lower, delta_d_upper, color=color, alpha=0.3, zorder=1
        )

        # Calculate t_1/2 from k using error propagation
        # For first-order kinetics: t_1/2 = ln(2) / k
        t_half = np.log(2) / k
        # Error propagation: dt_1/2 = t_1/2 * dk / k
        t_half_err = t_half * np.sqrt(pcov[1, 1]) / k

        print("\nFit parameters (combined data):")
        print(f"  delta_D_max = {delta_d_max:.2f} +/- {np.sqrt(pcov[0, 0]):.2f} nm")
        print(f"  k = {k:.4f} +/- {np.sqrt(pcov[1, 1]):.4f} min^-1")
        print(f"  t_1/2 = {t_half:.2f} +/- {t_half_err:.2f} min")
        print(f"  t_0 = {t0:.2f} +/- {np.sqrt(pcov[2, 2]):.2f} min")

        # Add fit parameters as text on plot
        param_text = (
            f"$\\Delta D_{{max}}$ = {delta_d_max:.1f} ± {np.sqrt(pcov[0, 0]):.1f} nm\n"
            f"$t_{{1/2}}$ = {t_half:.1f} ± {t_half_err:.1f} min\n"
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
    ax.set_title("DNA Brush Attachment Kinetics (Combined)")
    ax.set_ylim(bottom=0)
    format_axes(ax)

    plt.tight_layout()

    # Save plot
    save_plot(fig, "figures/fig-3/Brush Attachment Combined/Combined Kinetics")

    print("\nPlot saved as 'Combined Kinetics.svg'")

    plt.show()


if __name__ == "__main__":
    main()
