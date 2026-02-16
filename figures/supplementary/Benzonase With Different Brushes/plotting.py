"""Plot Benzonase FRET timecourse with different brush types."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import defaults
from utils.plotting import format_axes, save_plot, setup_plot_style

# --- Load data ---
data_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(data_dir, "2026-02-13 Benzonase FRET Different Brushes.csv")

# Read CSV, skipping header row (sample names) and stopping before metadata
raw = pd.read_csv(csv_path, header=1)

# Drop rows where the first column isn't numeric (metadata at end)
raw = raw[pd.to_numeric(raw.iloc[:, 0], errors="coerce").notna()]  # type: ignore

# Extract time and intensity for each well (paired columns)
wells = {}
for i in range(6):
    time_col = raw.columns[i * 2]
    intensity_col = raw.columns[i * 2 + 1]
    wells[i + 1] = {
        "time": np.asarray(raw[time_col], dtype=float),
        "intensity": np.asarray(raw[intensity_col], dtype=float),
    }

# --- Normalize between 0% (well 1) and 100% (well 2) digestion ---
baseline_0_time = wells[1]["time"]
baseline_0_intensity = wells[1]["intensity"]
baseline_100_time = wells[2]["time"]
baseline_100_intensity = wells[2]["intensity"]

conditions = {
    3: "No Brush",
    4: "Short Brush",
    5: "Long Brush",
    6: "Star Brush",
}

# Colorblind palette: blue(0), orange(1), green(2), vermilion(3), purple(4),
# brown(5), pink(6), grey(7), yellow(8), teal(9)
palette = sns.color_palette("colorblind", 10)
colors = {
    3: palette[8],  # yellow
    4: palette[0],  # blue
    5: palette[2],  # teal/green (#029E73)
    6: palette[3],  # vermilion
}

# --- Plot ---
setup_plot_style()
fig, ax = plt.subplots(figsize=(defaults.fig_width, defaults.fig_height))

for well_num, label in conditions.items():
    t = wells[well_num]["time"]
    intensity = wells[well_num]["intensity"]

    # Interpolate both baselines onto this well's timepoints
    b0 = np.interp(t, baseline_0_time, baseline_0_intensity)
    b100 = np.interp(t, baseline_100_time, baseline_100_intensity)

    # Normalize: 0% = baseline_0, 100% = baseline_100
    digestion = 100.0 * (intensity - b0) / (b100 - b0)

    ax.plot(t, digestion, color=colors[well_num], linewidth=1.5, label=label)

ax.set_xlabel("Time (min)")
ax.set_ylabel("DNA Digestion (%)")
ax.legend(frameon=False)
format_axes(ax)

plt.tight_layout()
save_plot(fig, os.path.join(data_dir, "benzonase_fret_brushes"))
plt.show()
