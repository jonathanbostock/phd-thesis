"""
Combined analysis of DLS kinetics data from 2023 and 2026
Plots delta D over time with exponential fits using time subtraction method
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

from utils.plotting import setup_plot_style, format_axes, save_plot
from utils import defaults


def exponential_attachment(
    t: np.ndarray, delta_d_max: float, k: float, t0: float
) -> np.ndarray:
    """Exponential attachment model: delta_D_max * (1 - exp(-k * (t - t0))). Zero for t < t0."""
    result = np.zeros_like(t, dtype=float)
    mask = t >= t0
    result[mask] = delta_d_max * (1 - np.exp(-k * (t[mask] - t0)))
    return result


def process_2023_data(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Process 2023 data. Sample names: "Ext N Ctrl 1", "Ext N 0 min", "Ext N T min".
    """
    df = pd.read_csv(filepath)
    df["datetime"] = pd.to_datetime(
        df["Measurement Start Date And Time"], format="%d/%m/%Y %H:%M"
    )
    df = df[df["Quality Indicator"] == "GoodData"].copy()

    all_times: list[float] = []
    all_delta_d: list[float] = []

    for series_num in [1, 2, 3]:
        baseline_row = df[df["Sample Name"] == f"Ext {series_num} Ctrl 1"]
        if len(baseline_row) == 0:
            print(f"Warning: No baseline found for Ext {series_num}")
            continue
        baseline_peak1 = float(
            baseline_row["Peak 1 Mean by Intensity ordered by area (nm)"].iloc[0]  # pyright: ignore[reportAttributeAccessIssue]
        )

        start_row = df[df["Sample Name"] == f"Ext {series_num} 0 min"]
        if len(start_row) == 0:
            print(f"Warning: No start time found for Ext {series_num}")
            continue
        time_zero = start_row["datetime"].iloc[0]  # pyright: ignore[reportAttributeAccessIssue]

        sample_names = df["Sample Name"]
        series_data = df[
            sample_names.str.match(f"Ext {series_num} \\d+ min")  # pyright: ignore[reportAttributeAccessIssue]
        ].copy()
        if len(series_data) == 0:
            continue

        series_data["time_min"] = (
            series_data["datetime"] - time_zero
        ).dt.total_seconds() / 60
        series_data["delta_d"] = (
            series_data["Peak 1 Mean by Intensity ordered by area (nm)"]
            - baseline_peak1
        )
        all_times.extend(series_data["time_min"].tolist())
        all_delta_d.extend(series_data["delta_d"].tolist())

    return np.array(all_times), np.array(all_delta_d)


def process_2026_data(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Process 2026 data. Sample names: "ext-n-before", "ext-n-start", "ext-n".
    """
    df = pd.read_csv(filepath)
    df["datetime"] = pd.to_datetime(
        df["Measurement Start Date And Time"], format="%d/%m/%Y %H:%M"
    )

    all_times: list[float] = []
    all_delta_d: list[float] = []

    for series_num in [1, 2, 3]:
        baseline_row = df[df["Sample Name"] == f"ext-{series_num}-before"]
        if len(baseline_row) == 0:
            print(f"Warning: No baseline found for ext-{series_num}")
            continue
        baseline_peak1 = float(
            baseline_row["Peak 1 Mean by Intensity ordered by area (nm)"].iloc[0]  # pyright: ignore[reportAttributeAccessIssue]
        )

        start_row = df[df["Sample Name"] == f"ext-{series_num}-start"]
        if len(start_row) == 0:
            print(f"Warning: No start time found for ext-{series_num}")
            continue
        time_zero = start_row["datetime"].iloc[0]  # pyright: ignore[reportAttributeAccessIssue]

        kinetics_data = df[df["Sample Name"] == f"ext-{series_num}"].copy()
        if len(kinetics_data) == 0:
            continue

        kinetics_data["time_min"] = (
            kinetics_data["datetime"] - time_zero
        ).dt.total_seconds() / 60
        kinetics_data["delta_d"] = (
            kinetics_data["Peak 1 Mean by Intensity ordered by area (nm)"]
            - baseline_peak1
        )
        all_times.extend(kinetics_data["time_min"].tolist())
        all_delta_d.extend(kinetics_data["delta_d"].tolist())

    return np.array(all_times), np.array(all_delta_d)


def main() -> None:
    setup_plot_style()
    colors = sns.color_palette("colorblind")
    color = colors[0]

    fig, ax = plt.subplots(figsize=(defaults.fig_width, defaults.fig_height))

    print("Processing 2023 data...")
    times_2023, delta_d_2023 = process_2023_data("2023-06-07 Attachment Timecourse.csv")
    print("Processing 2026 data...")
    times_2026, delta_d_2026 = process_2026_data("2026-01-22 DLS Kinetics.csv")

    for times, delta_d in [(times_2023, delta_d_2023), (times_2026, delta_d_2026)]:
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

    all_times = np.concatenate([times_2023, times_2026])
    all_delta_d = np.concatenate([delta_d_2023, delta_d_2026])
    print(
        f"Total: {len(all_times)} points (2023: {len(times_2023)}, 2026: {len(times_2026)})"
    )

    try:
        popt, pcov = curve_fit(
            exponential_attachment,
            all_times,
            all_delta_d,
            p0=[max(all_delta_d), 0.1, 0],
            bounds=([0, 0, 0], [np.inf, np.inf, max(all_times)]),
            maxfev=10000,
        )
        delta_d_max, k, t0 = popt

        t_smooth = np.linspace(0, max(all_times) * 1.1, 200)
        ax.plot(
            t_smooth,
            exponential_attachment(t_smooth, *popt),
            color=color,
            linewidth=2,
            alpha=0.8,
            zorder=2,
        )

        param_samples = np.abs(np.random.multivariate_normal(popt, pcov, 1000))
        band = np.array([exponential_attachment(t_smooth, *p) for p in param_samples])
        ax.fill_between(
            t_smooth,
            np.percentile(band, 16, axis=0),
            np.percentile(band, 84, axis=0),
            color=color,
            alpha=0.3,
            zorder=1,
        )

        t_half = np.log(2) / k
        t_half_err = t_half * np.sqrt(pcov[1, 1]) / k
        print(
            f"delta_D_max = {delta_d_max:.2f} ± {np.sqrt(pcov[0, 0]):.2f} nm, "
            f"t_1/2 = {t_half:.2f} ± {t_half_err:.2f} min, "
            f"t_0 = {t0:.2f} ± {np.sqrt(pcov[2, 2]):.2f} min"
        )

        ax.text(
            0.95,
            0.05,
            (
                f"$\\Delta D_{{max}}$ = {delta_d_max:.1f} ± {np.sqrt(pcov[0, 0]):.1f} nm\n"
                f"$t_{{1/2}}$ = {t_half:.1f} ± {t_half_err:.1f} min\n"
                f"$t_0$ = {t0:.1f} ± {np.sqrt(pcov[2, 2]):.1f} min"
            ),
            transform=ax.transAxes,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    except (RuntimeError, ValueError) as e:
        print(f"Could not fit curve: {e}")

    ax.set_xlabel("Time / min")
    ax.set_ylabel(r"$\Delta D$ / nm")
    ax.set_ylim(bottom=0)
    format_axes(ax)

    plt.tight_layout()
    save_plot(fig, "Attachment Kinetics")


if __name__ == "__main__":
    main()
