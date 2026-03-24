"""
Benzonase digestion protection assay (replicate 3).

Plate layout (3 rows × 10 wells):
  Well 1: Full digestion control
  Well 2: No digestion control
  Wells 3-10: Sparse brush, Sparse brush ctrl, Short brush, Short brush ctrl,
              Long brush, Long brush ctrl, Star brush, Star brush ctrl

Normalisation: for each row (repeat) and each timepoint,
  normalised = (value − full_digestion) / (no_digestion − full_digestion)
so 0 = fully digested, 1 = completely protected.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from utils.plotting import format_axes, save_plot, setup_plot_style

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
here = os.path.dirname(__file__)
data_file = os.path.join(here, "2026-03-17 Brush Digestion Protection.csv")

df_raw = pd.read_csv(data_file, header=None)

# Row index 1 → time points (columns 2 onward)
timepoints = df_raw.iloc[1, 2:].astype(float).values  # seconds
time_min = timepoints / 60

# Rows index 2+ → well data
well_data: dict[str, np.ndarray] = {}
for i in range(2, len(df_raw)):
    row = df_raw.iloc[i]
    well_id = str(row.iloc[0]).strip()
    if not well_id or well_id.lower() == "nan":
        continue
    well_data[well_id] = row.iloc[2:].astype(float).values

# ---------------------------------------------------------------------------
# Build data array: shape (n_repeats=3, n_wells=10, n_timepoints)
# ---------------------------------------------------------------------------
plate_rows = ["A", "B", "C"]
n_timepoints = len(timepoints)
data_array = np.zeros((3, 10, n_timepoints))
for r_idx, row_letter in enumerate(plate_rows):
    for w_idx in range(10):
        well_id = f"{row_letter}{w_idx + 1:02d}"
        data_array[r_idx, w_idx, :] = well_data[well_id]

# ---------------------------------------------------------------------------
# Normalise wells 3-10 per row per timepoint
# ---------------------------------------------------------------------------
full_dig = data_array[:, 0, :]  # (3, n_timepoints)
no_dig = data_array[:, 1, :]    # (3, n_timepoints)

# experimental wells: indices 2-9
exp_data = data_array[:, 2:, :]  # (3, 8, n_timepoints)
normalised = (exp_data - full_dig[:, np.newaxis, :]) / (
    (no_dig - full_dig)[:, np.newaxis, :]
)  # (3, 8, n_timepoints)

mean_norm = normalised.mean(axis=0)  # (8, n_timepoints)
sem_norm = stats.sem(normalised, axis=0)  # (8, n_timepoints)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
setup_plot_style()

brush_labels = ["Sparse", "Short", "Long", "Star"]
colors = sns.color_palette("colorblind", n_colors=4)
markers = ["o", "s", "^", "D"]

fig, ax = plt.subplots()

for i, (label, color, marker) in enumerate(zip(brush_labels, colors, markers)):
    brush_idx = 2 * i       # 0, 2, 4, 6
    ctrl_idx = 2 * i + 1    # 1, 3, 5, 7

    # Brush line
    ax.plot(
        time_min,
        mean_norm[brush_idx] * 100,
        color=color,
        linestyle="-",
        linewidth=1.2,
        label=f"{label} brush",
    )
    ax.fill_between(
        time_min,
        (mean_norm[brush_idx] - sem_norm[brush_idx]) * 100,
        (mean_norm[brush_idx] + sem_norm[brush_idx]) * 100,
        color=color,
        alpha=0.25,
    )

    # Control line (dashed, same colour)
    ax.plot(
        time_min,
        mean_norm[ctrl_idx] * 100,
        color=color,
        linestyle="--",
        linewidth=1.0,
        label=f"{label} ctrl",
    )
    ax.fill_between(
        time_min,
        (mean_norm[ctrl_idx] - sem_norm[ctrl_idx]) * 100,
        (mean_norm[ctrl_idx] + sem_norm[ctrl_idx]) * 100,
        color=color,
        alpha=0.15,
    )

ax.set_xlabel("Time / min")
ax.set_ylabel("Brush Digestion / %")
ax.set_ylim(0, 100)
ax.legend(
    ncol=2,
    frameon=False,
    fontsize=6,
    title="— brush  – – ctrl",
    title_fontsize=6,
)

format_axes(ax)
save_plot(fig, os.path.join(here, "benzonase-digestion-3"))

# ---------------------------------------------------------------------------
# Second plot: first 20 minutes
# ---------------------------------------------------------------------------
mask = time_min <= 10

fig2, ax2 = plt.subplots()

for i, (label, color, marker) in enumerate(zip(brush_labels, colors, markers)):
    brush_idx = 2 * i
    ctrl_idx = 2 * i + 1

    ax2.plot(
        time_min[mask],
        mean_norm[brush_idx][mask] * 100,
        color=color,
        linestyle="-",
        linewidth=1.2,
        label=f"{label} brush",
    )
    ax2.fill_between(
        time_min[mask],
        (mean_norm[brush_idx] - sem_norm[brush_idx])[mask] * 100,
        (mean_norm[brush_idx] + sem_norm[brush_idx])[mask] * 100,
        color=color,
        alpha=0.25,
    )

    ax2.plot(
        time_min[mask],
        mean_norm[ctrl_idx][mask] * 100,
        color=color,
        linestyle="--",
        linewidth=1.0,
        label=f"{label} ctrl",
    )
    ax2.fill_between(
        time_min[mask],
        (mean_norm[ctrl_idx] - sem_norm[ctrl_idx])[mask] * 100,
        (mean_norm[ctrl_idx] + sem_norm[ctrl_idx])[mask] * 100,
        color=color,
        alpha=0.15,
    )

ax2.set_xlabel("Time / min")
ax2.set_ylabel("Brush Digestion / %")
ax2.set_ylim(0, 100)
ax2.legend(
    ncol=2,
    frameon=False,
    fontsize=6,
    title="— brush  – – ctrl",
    title_fontsize=6,
)

format_axes(ax2)
save_plot(fig2, os.path.join(here, "benzonase-digestion-3-early"))
