"""
BCA assay calibration curve with WT and K237C sample points.
Polynomial fit to calibration standards, samples marked separately.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.plotting import setup_plot_style, format_axes, save_plot
from utils import defaults


def polyfit_eval(x: float, coefficients: np.ndarray) -> np.ndarray:
    return sum(c * x**i for i, c in enumerate(coefficients[::-1]))  # type: ignore[return-value]


def main() -> None:
    raw_data = pd.read_csv("2024-02-06 BCA Assay Ahl 1.txt", header=1)

    raw_array = np.array(raw_data)
    average_array = np.mean(raw_array, axis=0)
    stdev_array = np.std(raw_array, axis=0)

    # Standard concentrations: columns A-I are calibration, K237C and wt are samples
    x_array = np.array([2, 1.5, 1, 0.75, 0.5, 0.25, 0.125, 0.025, 0])

    cal_avg = average_array[:-2]
    cal_std = stdev_array[:-2]
    mut_avg = average_array[-2]
    wt_avg = average_array[-1]

    # Polynomial fit: concentration as function of absorbance
    polynomial_fit = np.polyfit(cal_avg, x_array, 3)

    abs_range = np.linspace(0, max(cal_avg) + 0.1, 200)
    conc_fit = np.polyval(polynomial_fit, abs_range)

    mut_conc = float(np.polyval(polynomial_fit, mut_avg))
    wt_conc = float(np.polyval(polynomial_fit, wt_avg))

    setup_plot_style()
    colors = sns.color_palette("colorblind")

    fig, ax = plt.subplots(figsize=(defaults.fig_width, defaults.fig_height))

    # Calibration scatter
    ax.scatter(
        x_array,
        cal_avg,
        color=colors[0],
        marker="o",
        edgecolors="black",
        linewidths=0.5,
        s=50,
        label="Calibration",
        zorder=5,
    )
    ax.errorbar(
        x_array,
        cal_avg,
        yerr=cal_std,
        fmt="none",
        color=colors[0],
        linewidth=1.5,
        capsize=3,
    )

    # Polynomial fit line (x = conc_fit, y = abs_range — i.e. x=concentration, y=absorbance)
    ax.plot(conc_fit, abs_range, color=colors[0], linewidth=2, alpha=0.8)

    # Sample points
    ax.scatter(
        [mut_conc],
        [mut_avg],
        color=colors[1],
        marker="s",
        edgecolors="black",
        linewidths=0.5,
        s=60,
        label=f"K237C: {mut_conc:.2g} mg/mL",
        zorder=6,
    )
    ax.scatter(
        [wt_conc],
        [wt_avg],
        color=colors[2],
        marker="^",
        edgecolors="black",
        linewidths=0.5,
        s=60,
        label=f"WT: {wt_conc:.2g} mg/mL",
        zorder=6,
    )

    ax.set_xlabel("Concentration / mg/mL")
    ax.set_ylabel("Absorbance at 562 nm")
    ax.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc="upper left")
    format_axes(ax)

    plt.tight_layout()
    save_plot(fig, "ahl-expression-bca")


if __name__ == "__main__":
    main()
