"""
ATP aptamer brush: ΔD (nm) vs ATP concentration with linear fit + uncertainty band.
Symlog x-scale.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy import stats

from utils.plotting import setup_plot_style, format_axes, save_plot
from utils import defaults


def linear(x: np.ndarray, m: float, c: float) -> np.ndarray:
    return m * x + c


def main() -> None:
    raw_data = pd.read_csv("2023-08-07 ATP Aptamer Brush.tsv", delimiter="\t")

    name_key = "Sample Name"
    p1mi_key = "Peak 1 Mean by Intensity ordered by area (nm)"

    cond_dict = {
        "0": 0,
        "1uM": 1e-3,
        "10uM": 1e-2,
        "100uM": 1e-1,
        "1mM": 1e0,
        "3mM": 3e0,
    }

    processed_dict: dict[str, list[float]] = {cond: [] for cond in cond_dict}
    control_dict: dict[str, float] = {}

    x_fit: list[float] = []
    y_fit: list[float] = []

    for _, row in raw_data.iterrows():
        name_list = str(row[name_key]).split()
        size = float(row[p1mi_key])  # type: ignore[arg-type]
        extrusion = name_list[1]
        condition = name_list[2]

        if condition == "Ctrl":
            control_dict[extrusion] = size
        else:
            delta = size - control_dict[extrusion]
            processed_dict[condition].append(delta)
            x_fit.append(cond_dict[condition])
            y_fit.append(delta)

    processed_data = pd.DataFrame(
        {
            cond: {
                "ATP Conc": cond_dict[cond],
                "Delta D": float(np.mean(processed_dict[cond])),
                "SEM": float(stats.sem(processed_dict[cond])),
            }
            for cond in cond_dict
        }
    ).T.astype(float)

    x_arr = np.array(x_fit)
    y_arr = np.array(y_fit)

    popt, pcov = curve_fit(linear, x_arr, y_arr, p0=[10.0, 0.0])

    x_plot = np.linspace(0, max(x_arr) * 1.1, 300)
    y_plot = linear(x_plot, *popt)

    n_samples = 1000
    param_samples = np.random.multivariate_normal(popt, pcov, n_samples)
    y_band = np.array([linear(x_plot, *p) for p in param_samples])
    y_lower = np.percentile(y_band, 16, axis=0)
    y_upper = np.percentile(y_band, 84, axis=0)

    setup_plot_style()
    color = sns.color_palette("colorblind")[0]

    fig, ax = plt.subplots(figsize=(defaults.fig_width, defaults.fig_height))

    ax.scatter(
        processed_data["ATP Conc"],
        processed_data["Delta D"],
        color=color,
        marker="o",
        edgecolors="black",
        linewidths=0.5,
        s=50,
        zorder=5,
    )

    ax.plot(x_plot, y_plot, color=color, linewidth=2, alpha=0.8)
    ax.fill_between(x_plot, y_lower, y_upper, color=color, alpha=0.3)

    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlabel("ATP Concentration / mM")
    ax.set_ylabel(r"$\Delta D$ / nm")
    format_axes(ax)

    plt.tight_layout()
    save_plot(fig, "atp-aptamer-brush")


if __name__ == "__main__":
    main()
