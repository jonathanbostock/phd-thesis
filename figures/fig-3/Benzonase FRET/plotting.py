"""Benzonase FRET Digestion Analysis
Analyzes DNA brush digestion kinetics using FRET measurements
"""

import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from utils.plotting import setup_plot_style, format_axes, save_plot
from utils import defaults

# Experimental setup
CONDITIONS = ["Free", "Sparse", "Dense"]
CONDITIONS_DICT = {
    1: "Free",
    2: "Free",
    3: "Free",
    4: "Dense",
    5: "Dense",
    6: "Dense",
    7: "Sparse",
    8: "Sparse",
    9: "Sparse",
}
INVESTIGATIONS = {"A": "+ve", "B": "-ve", "C": "Exp"}
CONTROLS = ["Exp", "+ve", "-ve"]


def convert_column(column_name: str) -> str:
    """Convert plate reader column names to condition labels

    Parameters
    ----------
    column_name : str
        Format like "Sample A1"

    Returns
    -------
    str
        Format like "Free +ve 0"
    """
    column_name = column_name.split(" ")[1]
    letter, number = column_name[0], int(column_name[1:])

    batch = (number - 1) % 3

    condition_name = CONDITIONS_DICT[number]
    investigation_name = INVESTIGATIONS[letter]

    return f"{condition_name} {investigation_name} {batch}"


def get_digestion(row: pd.Series, c: str, b: int) -> float:
    """Calculate digestion fraction from FRET signals

    Parameters
    ----------
    row : pd.Series
        Data row with +ve, -ve, and Exp columns
    c : str
        Condition (Free, Sparse, Dense)
    b : int
        Batch number (0-2)

    Returns
    -------
    float
        Digestion fraction (Exp-neg)/(pos-neg)
    """
    pos = row[f"{c} +ve {b}"]
    neg = row[f"{c} -ve {b}"]
    exp = row[f"{c} Exp {b}"]

    return (exp - neg) / (pos - neg)


def load_and_process_data(file_name: str) -> pd.DataFrame:
    """Load and process benzonase timecourse data

    Parameters
    ----------
    file_name : str
        Path to CSV file

    Returns
    -------
    pd.DataFrame
        Processed data with digestion values and statistics
    """
    # Load intensity data
    raw_data = pd.read_csv(
        file_name, skiprows=1, usecols=lambda x: "Intensity" in x, nrows=150
    )

    # Load column names
    columns = pd.read_csv(file_name, usecols=lambda x: len(x) < 10).columns

    # Convert column names
    raw_data.columns = [convert_column(c) for c in columns]

    # Calculate digestion values for each condition
    for c in CONDITIONS:
        for b in range(3):
            raw_data[f"{c} Digestion {b}"] = raw_data.apply(
                lambda row: get_digestion(row, c, b), axis=1
            )

            raw_data[f"{c} LogRemainder {b}"] = raw_data.apply(
                lambda row: np.log(1 - row[f"{c} Digestion {b}"]), axis=1
            )

        # Calculate means and standard errors
        raw_data[f"{c} Digestion Mean"] = raw_data.apply(
            lambda row: np.mean([row[f"{c} Digestion {b}"] for b in range(3)]), axis=1
        )
        raw_data[f"{c} Digestion SEM"] = raw_data.apply(
            lambda row: stats.sem([row[f"{c} Digestion {b}"] for b in range(3)]),
            axis=1,
        )

        for ctrl in CONTROLS:
            raw_data[f"{c} {ctrl} Mean"] = raw_data.apply(
                lambda row: np.mean([row[f"{c} {ctrl} {b}"] for b in range(3)]),
                axis=1,
            )
            raw_data[f"{c} {ctrl} SEM"] = raw_data.apply(
                lambda row: stats.sem([row[f"{c} {ctrl} {b}"] for b in range(3)]),
                axis=1,
            )

        raw_data[f"{c} LogRemainder Mean"] = raw_data.apply(
            lambda row: np.mean([row[f"{c} LogRemainder {b}"] for b in range(3)]),
            axis=1,
        )

        raw_data[f"{c} LogRemainder SEM"] = raw_data.apply(
            lambda row: stats.sem([row[f"{c} LogRemainder {b}"] for b in range(3)]),
            axis=1,
        )

    raw_data["Time"] = range(150)

    return raw_data


def fit_digestion_rates(raw_data: pd.DataFrame, fit_points: int = 25) -> dict:
    """Fit exponential decay rates to log-remainder data

    Parameters
    ----------
    raw_data : pd.DataFrame
        Processed data
    fit_points : int
        Number of initial time points to fit

    Returns
    -------
    dict
        Fit results for each condition
    """
    fits = {}
    for c in CONDITIONS:
        fits[c] = stats.linregress(
            x=range(fit_points), y=raw_data[f"{c} LogRemainder Mean"][:fit_points]
        )

    return fits


def plot_digestion_timecourse(raw_data: pd.DataFrame) -> tuple:
    """Plot digestion fraction over time

    Parameters
    ----------
    raw_data : pd.DataFrame
        Processed data

    Returns
    -------
    tuple
        (fig, ax)
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(defaults.fig_width, defaults.fig_height))

    names = ["No Lipid", "Sparse Brush", "Dense Brush"]
    colors = sns.color_palette("colorblind", n_colors=3)

    for i, c in enumerate(CONDITIONS):
        # Plot mean as line
        ax.plot(
            raw_data["Time"],
            raw_data[f"{c} Digestion Mean"],
            color=colors[i],
            linewidth=2,
            label=names[i],
        )

        # Plot SEM as shaded area
        ax.fill_between(
            raw_data["Time"],
            raw_data[f"{c} Digestion Mean"] - raw_data[f"{c} Digestion SEM"],
            raw_data[f"{c} Digestion Mean"] + raw_data[f"{c} Digestion SEM"],
            color=colors[i],
            alpha=0.3,
        )

    ax.set_xlabel("Time / min")
    ax.set_ylabel("Digestion Fraction")
    ax.legend(frameon=False)
    format_axes(ax)

    return fig, ax


def plot_fluorescence_timecourse(raw_data: pd.DataFrame) -> tuple:
    """Plot raw fluorescence over time for all controls

    Parameters
    ----------
    raw_data : pd.DataFrame
        Processed data

    Returns
    -------
    tuple
        (fig, ax)
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(defaults.fig_width, defaults.fig_height))

    condition_colors = sns.color_palette("colorblind", n_colors=3)
    control_linestyles = ["-", "--", ":"]
    control_names = ["Experimental", "Positive Control", "Negative Control"]
    condition_names = ["No Lipid", "Sparse Brush", "Dense Brush"]

    # Plot each control type
    for ctrl_idx, ctrl in enumerate(CONTROLS):
        for cond_idx, c in enumerate(CONDITIONS):
            ax.plot(
                raw_data["Time"],
                raw_data[f"{c} {ctrl} Mean"],
                color=condition_colors[cond_idx],
                linestyle=control_linestyles[ctrl_idx],
                linewidth=2,
            )

            # Plot SEM as shaded area
            ax.fill_between(
                raw_data["Time"],
                raw_data[f"{c} {ctrl} Mean"] - raw_data[f"{c} {ctrl} SEM"],
                raw_data[f"{c} {ctrl} Mean"] + raw_data[f"{c} {ctrl} SEM"],
                color=condition_colors[cond_idx],
                alpha=0.2,
            )

    # Create custom legend with grey lines for line styles and colored lines for conditions
    grey = "dimgrey"
    legend_handles = [
        Line2D([0], [0], color=grey, linestyle="-", linewidth=2),
        Line2D([0], [0], color=grey, linestyle="--", linewidth=2),
        Line2D([0], [0], color=grey, linestyle=":", linewidth=2),
        Line2D([0], [0], color=condition_colors[0], linestyle="-", linewidth=2),
        Line2D([0], [0], color=condition_colors[1], linestyle="-", linewidth=2),
        Line2D([0], [0], color=condition_colors[2], linestyle="-", linewidth=2),
    ]
    legend_labels = control_names + condition_names

    ax.set_xlabel("Time / min")
    ax.set_ylabel("Fluorescence / a.u.")
    ax.legend(legend_handles, legend_labels, frameon=False)
    format_axes(ax)

    return fig, ax


def plot_rate_fits(raw_data: pd.DataFrame, fits: dict, fit_points: int = 25) -> tuple:
    """Plot log-remainder with linear fits

    Parameters
    ----------
    raw_data : pd.DataFrame
        Processed data
    fits : dict
        Fit results for each condition
    fit_points : int
        Number of points used for fitting

    Returns
    -------
    tuple
        (fig, ax)
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(defaults.fig_width, defaults.fig_height))

    names = ["No Lipid", "Sparse Brush", "Dense Brush"]
    colors = sns.color_palette("colorblind", n_colors=3)

    for i, c in enumerate(CONDITIONS):
        # Plot mean as line with shaded SEM
        ax.plot(
            raw_data["Time"][:fit_points],
            raw_data[f"{c} LogRemainder Mean"][:fit_points],
            color=colors[i],
            linewidth=1.5,
            alpha=0.7,
        )

        # Plot SEM as shaded area
        ax.fill_between(
            raw_data["Time"][:fit_points],
            raw_data[f"{c} LogRemainder Mean"][:fit_points]
            - raw_data[f"{c} LogRemainder SEM"][:fit_points],
            raw_data[f"{c} LogRemainder Mean"][:fit_points]
            + raw_data[f"{c} LogRemainder SEM"][:fit_points],
            color=colors[i],
            alpha=0.3,
        )

        # Plot fit line (dashed)
        x_fit = np.array(range(fit_points))
        y_fit = fits[c].slope * x_fit + fits[c].intercept
        ax.plot(x_fit, y_fit, color=colors[i], linewidth=2, linestyle="--")

    # Create custom legend with grey lines for data/fit and colored lines for conditions
    grey = "dimgrey"
    legend_handles = [
        Line2D([0], [0], color=grey, linestyle="-", linewidth=1.5, alpha=0.7),
        Line2D([0], [0], color=grey, linestyle="--", linewidth=2),
        Line2D([0], [0], color=colors[0], linestyle="-", linewidth=2),
        Line2D([0], [0], color=colors[1], linestyle="-", linewidth=2),
        Line2D([0], [0], color=colors[2], linestyle="-", linewidth=2),
    ]
    legend_labels = ["Data", "Fit"] + names

    ax.set_xlabel("Time / min")
    ax.set_ylabel("ln(1 - Digestion)")
    ax.legend(legend_handles, legend_labels, frameon=False)
    format_axes(ax)

    return fig, ax


def main():
    """Main analysis function"""
    file_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(file_path)

    file_name = "2024-05-16 Brush Benzonase Timecourse.csv"

    # Load and process data
    print("Loading and processing data...")
    raw_data = load_and_process_data(file_name)

    # Fit digestion rates
    print("Fitting digestion rates...")
    fits = fit_digestion_rates(raw_data, fit_points=25)

    # Print rate constants
    print("\nDigestion rate constants (min^-1):")
    for c in CONDITIONS:
        rate = -fits[c].slope  # Negative because it's a decay
        rate_err = fits[c].stderr
        print(f"  {c:8s}: {rate:.4f} ± {rate_err:.4f}")

    # Generate plots
    print("\nGenerating plots...")

    fig1, ax1 = plot_digestion_timecourse(raw_data)
    save_plot(fig1, os.path.join(file_path, "Digestion By FRET 1"))

    fig2, ax2 = plot_fluorescence_timecourse(raw_data)
    save_plot(fig2, os.path.join(file_path, "Fluorescence Over Time"))

    fig3, ax3 = plot_rate_fits(raw_data, fits)
    save_plot(fig3, os.path.join(file_path, "Rate Fits 1"))

    print("Analysis complete!")


if __name__ == "__main__":
    main()
