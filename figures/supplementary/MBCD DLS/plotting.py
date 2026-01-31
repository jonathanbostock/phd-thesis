### Analyse MBCD Data
### Jonathan Bostock
### Rewritten to use project utilities and consistent styling

import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.optimize import curve_fit
from utils.plotting import setup_plot_style, format_axes, save_plot
from utils import defaults
import seaborn as sns


# Column keys for parsing DLS data files
P1MI_KEY = "Peak 1 Mean by Intensity ordered by area (nm)"
NAME_KEY = "Sample Name"
TIME_KEY = "Measurement Start Date And Time"


def exp_decay(t, k):
    """
    Exponential decay model for brush removal kinetics

    Parameters:
    -----------
    t : array-like
        Time in seconds
    k : float
        Rate constant in s^-1

    Returns:
    --------
    array-like
        Predicted ΔD values
    """
    return 35 * np.exp(-k * (t + 60))


def power_fit(x, k, n):
    """
    Power law fit for rate constant vs concentration

    Parameters:
    -----------
    x : array-like
        MBCD concentration in M
    k : float
        Prefactor
    n : float
        Power law exponent

    Returns:
    --------
    array-like
        Predicted rate constant values
    """
    return k * np.power(x, n)


def meta_exp_decay(input_array, k, n):
    """
    Combined exponential decay model accounting for concentration dependence

    Parameters:
    -----------
    input_array : array-like, shape (N, 2)
        First column: time (s), Second column: concentration (mM)
    k : float
        Base rate constant
    n : float
        Concentration exponent

    Returns:
    --------
    array-like
        Predicted ΔD values
    """
    t = input_array[..., 0]
    c = input_array[..., 1] / 1000  # Convert mM to M

    return 35 * np.exp(-(t + 60) * k * np.power(c, n))


def flatten(array):
    """Flatten a list of lists into a single list"""
    return [item for sublist in array for item in sublist]


def load_dls_data(file_path):
    """
    Load DLS data from tab-separated file

    Parameters:
    -----------
    file_path : str
        Path to the DLS data file

    Returns:
    --------
    tuple
        (DNA array, no DNA array, delta D array)
    """
    # Read only the columns we need
    data_frame = pd.read_csv(file_path, sep="\t")  # type: ignore[call-overload]
    # Filter to only the columns we need
    data_frame = data_frame[[NAME_KEY, P1MI_KEY, TIME_KEY]]

    DNA_array = np.array(data_frame[P1MI_KEY][:5])
    no_DNA_array = np.array(data_frame[P1MI_KEY][5:])
    delta_D_array = DNA_array - no_DNA_array

    return DNA_array, no_DNA_array, delta_D_array


def process_mbcd_data():
    """
    Process all MBCD DLS data files in the current directory

    Returns:
    --------
    tuple
        (data_dict, time_array, conc_vect)
    """
    # Get all .txt files (excluding ones marked as BAD)
    file_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(file_path)
    all_files = [
        f for f in os.listdir() if (f.endswith(".txt") and not f.startswith("BAD"))
    ]

    data_dict = {}

    for file_name in all_files:
        # Parse filename: "0.5 mM MBCD 1.txt" -> conc=0.5, number=1
        conc, _, _, number = file_name[:-4].split(" ")
        conc = float(conc)
        number = int(number)

        _, _, delta_D_array = load_dls_data(file_name)

        if conc not in data_dict:
            data_dict[conc] = []

        data_dict[conc].append(delta_D_array)

    # Time points: measurements every 195 seconds for 5 time points
    time_array = [i * 195 for i in range(5)]
    conc_vect = [0.5, 1, 2, 5]

    return data_dict, time_array, conc_vect


def fit_decay_curves(data_dict, time_array, conc_vect):
    """
    Fit exponential decay curves to each concentration

    Parameters:
    -----------
    data_dict : dict
        Dictionary mapping concentrations to lists of delta D arrays
    time_array : list
        Time points in seconds
    conc_vect : list
        Concentration values in mM

    Returns:
    --------
    tuple
        (raw_data_dict, curve_data_vect, curve_errors_vect, all_data_vect, all_conditions_vect)
    """
    raw_data_dict = {}
    curve_data_vect = []
    curve_errors_vect = []
    all_data_vect = []
    all_conditions_vect = []

    for conc in conc_vect:
        data = data_dict[conc]

        # Store raw data for scatter plotting
        raw_data_dict[conc] = data

        # Flatten data for fitting
        flat_data = flatten(data)
        flat_conditions = [[conc, t] for t in time_array] * len(data)

        # Fit exponential decay to individual concentration
        popt, pcov = curve_fit(exp_decay, time_array * len(data), flat_data, p0=[0.01])
        curve_data_vect.append(popt[0])
        curve_errors_vect.append(np.sqrt(pcov[0, 0]))  # Standard error of k

        all_data_vect += flat_data
        all_conditions_vect += flat_conditions

    return (
        raw_data_dict,
        curve_data_vect,
        curve_errors_vect,
        all_data_vect,
        all_conditions_vect,
    )


def plot_combined_analysis(
    time_array, raw_data_dict, curve_data_vect, curve_errors_vect, conc_vect
):
    """
    Plot combined figure: timecourse data and rate constants vs concentration

    Parameters:
    -----------
    time_array : list
        Time points in seconds
    raw_data_dict : dict
        Dictionary mapping concentrations to lists of raw delta D arrays
    curve_data_vect : list
        Fitted rate constants for each concentration
    curve_errors_vect : list
        Standard errors of fitted rate constants
    conc_vect : list
        Concentration values in mM

    Returns:
    --------
    tuple
        (fig, (ax1, ax2), power_fit_k, power_fit_n)
    """
    setup_plot_style()

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(defaults.fig_width * 2, defaults.fig_height)
    )

    # Use gradient colors for different concentrations with log scale
    # Map concentrations logarithmically across the full viridis range
    log_conc = np.log(conc_vect)
    cmap = cm.get_cmap("viridis")
    norm = Normalize(vmin=log_conc.min(), vmax=log_conc.max())
    colors = [cmap(norm(lc)) for lc in log_conc]

    labels = ["0.5 mM", "1 mM", "2 mM", "5 mM"]
    markers = ["o", "s", "^", "D"]

    # ===== LEFT PLOT: Timecourse =====
    for i, conc in enumerate(conc_vect):
        color = colors[i]

        # Plot raw data as scatter points
        raw_data = raw_data_dict[conc]
        for replicate in raw_data:
            ax1.scatter(
                time_array,
                replicate,
                color=color,
                marker=markers[i],
                edgecolors="black",
                linewidths=0.5,
                s=30,
                alpha=0.6,
            )

        # Plot one invisible point for the legend
        ax1.scatter(
            [],
            [],
            color=color,
            marker=markers[i],
            edgecolors="black",
            linewidths=0.5,
            s=50,
            label=labels[i],
        )

        # Plot fitted curve
        t_smooth = np.linspace(0, 800, 100)
        y_fit = exp_decay(t_smooth, curve_data_vect[i])
        ax1.plot(t_smooth, y_fit, color=color, linewidth=2, alpha=0.8)

    ax1.set_xlabel("Time / s")
    ax1.set_ylabel(r"$\Delta D$ / nm")
    ax1.legend(frameon=False, title="[MBCD]")
    ax1.set_title("Brush Removal Timecourse")
    format_axes(ax1)

    # ===== RIGHT PLOT: Rate Constants =====
    # Fit power law to rate constants
    conc_M = np.array(conc_vect) / 1000  # Convert mM to M
    rate_constant_results = stats.linregress(np.log(conc_M), np.log(curve_data_vect))

    power_fit_n = float(rate_constant_results.slope)  # type: ignore[attr-defined]
    power_fit_k = float(np.exp(rate_constant_results.intercept))  # type: ignore[attr-defined]

    # Plot data points with error bars
    for i, (conc_m, k_val, k_err) in enumerate(
        zip(conc_M, curve_data_vect, curve_errors_vect)
    ):
        ax2.errorbar(
            conc_m,
            k_val,
            yerr=k_err,
            color=colors[i],
            marker=markers[i],
            markeredgecolor="black",
            markeredgewidth=0.5,
            markersize=8,
            capsize=3,
            linewidth=0.5,
        )

    # Plot fitted curve in black
    c_smooth = np.logspace(np.log10(0.4 / 1000), np.log10(6 / 1000), 100)
    k_fit = power_fit(c_smooth, power_fit_k, power_fit_n)
    ax2.plot(
        c_smooth,
        k_fit,
        color="black",
        linewidth=2,
        label=f"$k = {power_fit_k:.1f}$ [MBCD]$^{{{power_fit_n:.2f}}}$",
    )

    ax2.set_yscale("log")
    ax2.set_xscale("log")
    ax2.set_xlabel("[MBCD] / M")
    ax2.set_ylabel(r"$k$ / s$^{-1}$")
    ax2.legend(frameon=False)
    ax2.set_title("Rate Constant vs [MBCD]")

    format_axes(ax2)
    plt.tight_layout()

    return fig, (ax1, ax2), power_fit_k, power_fit_n


def main():
    """Main function to process and plot MBCD data"""
    # Process data
    data_dict, time_array, conc_vect = process_mbcd_data()

    # Fit decay curves
    (
        raw_data_dict,
        curve_data_vect,
        curve_errors_vect,
        all_data_vect,
        all_conditions_vect,
    ) = fit_decay_curves(data_dict, time_array, conc_vect)

    # Fit meta-curve (optional - not plotted but could be useful)
    meta_curve_data = curve_fit(
        meta_exp_decay,
        np.array(all_conditions_vect),
        np.array(all_data_vect),
        p0=[1, 2],
    )

    # Plot combined figure
    fig, (ax1, ax2), power_fit_k, power_fit_n = plot_combined_analysis(
        time_array, raw_data_dict, curve_data_vect, curve_errors_vect, conc_vect
    )
    file_path = os.path.dirname(os.path.abspath(__file__))
    save_plot(fig, os.path.join(file_path, "MBCD Analysis"))

    print(f"Analysis complete!")
    print(f"Power law fit: k = {power_fit_k:.1f} * [MBCD]^{power_fit_n:.2f}")
    print(f"Rate constants (s^-1): {curve_data_vect}")
    print(f"Rate constant errors (s^-1): {curve_errors_vect}")


if __name__ == "__main__":
    main()
