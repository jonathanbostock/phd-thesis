"""
Benzonase digestion protection assay (replicate 2).

Plate layout (3 rows × 7 wells):
  Well 1: Full digestion control
  Well 2: No digestion control
  Wells 3-7: Sparse brush, Short brush, Long brush, Star brush, No brush

CSV format: interleaved (Time, Intensity) column pairs for each sample,
with each sample on its own slightly-offset time axis.

Normalisation per row per timepoint:
  normalised = (value − full_digestion) / (no_digestion − full_digestion)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.interpolate import interp1d

from utils.plotting import format_axes, save_plot, setup_plot_style

# ---------------------------------------------------------------------------
# Load and parse interleaved CSV
# ---------------------------------------------------------------------------
here = os.path.dirname(__file__)
data_file = os.path.join(here, "2026-03-05 Brush Digestion Benzonase.csv")

df_raw = pd.read_csv(data_file, header=None)

# Row 0: "Sample A1,,Sample A2,,..." — sample name every 2 columns
# Row 1: "Time (min)", "Intensity (a.u.)" repeated
# Rows 2+: data (stop before trailing metadata rows)

# Find last numeric row
last_data_row = 0
for i in range(2, len(df_raw)):
    try:
        float(df_raw.iloc[i, 0])
        last_data_row = i
    except (ValueError, TypeError):
        break

samples: dict[str, tuple[np.ndarray, np.ndarray]] = {}
n_cols = df_raw.shape[1]
for col in range(0, n_cols - 1, 2):
    name = str(df_raw.iloc[0, col]).strip().replace("Sample ", "")
    if not name or name.lower() == "nan":
        continue
    t = pd.to_numeric(df_raw.iloc[2:last_data_row + 1, col], errors="coerce").values
    y = pd.to_numeric(df_raw.iloc[2:last_data_row + 1, col + 1], errors="coerce").values
    mask = np.isfinite(t) & np.isfinite(y)
    samples[name] = (t[mask], y[mask])

# ---------------------------------------------------------------------------
# Interpolate all samples to a common 1-minute time grid
# ---------------------------------------------------------------------------
t_common = np.arange(0, 61, 1, dtype=float)

interp_data: dict[str, np.ndarray] = {}
for name, (t, y) in samples.items():
    f = interp1d(t, y, kind="linear", bounds_error=False, fill_value=np.nan)
    interp_data[name] = f(t_common)

# ---------------------------------------------------------------------------
# Build array: shape (3 repeats, 7 wells, n_timepoints)
# ---------------------------------------------------------------------------
plate_rows = ["A", "B", "C"]
n_timepoints = len(t_common)
data_array = np.zeros((3, 7, n_timepoints))
for r_idx, row_letter in enumerate(plate_rows):
    for w_idx in range(7):
        key = f"{row_letter}{w_idx + 1}"
        data_array[r_idx, w_idx, :] = interp_data[key]

# ---------------------------------------------------------------------------
# Normalise wells 3-7 per row per timepoint
# ---------------------------------------------------------------------------
full_dig = data_array[:, 0, :]  # (3, n_timepoints)
no_dig = data_array[:, 1, :]    # (3, n_timepoints)

exp_data = data_array[:, 2:, :]  # (3, 5, n_timepoints)
normalised = (exp_data - full_dig[:, np.newaxis, :]) / (
    (no_dig - full_dig)[:, np.newaxis, :]
)  # (3, 5, n_timepoints)

mean_norm = normalised.mean(axis=0)  # (5, n_timepoints)
sem_norm = stats.sem(normalised, axis=0)  # (5, n_timepoints)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
condition_labels = ["Sparse brush", "Short brush", "Long brush", "Star brush", "No brush"]
colors = sns.color_palette("colorblind", n_colors=5)
markers = ["o", "s", "^", "D", "v"]

setup_plot_style()

fig, ax = plt.subplots()

for i, (label, color) in enumerate(zip(condition_labels, colors)):
    ax.plot(
        t_common,
        mean_norm[i] * 100,
        color=color,
        linewidth=1.2,
        label=label,
    )
    ax.fill_between(
        t_common,
        (mean_norm[i] - sem_norm[i]) * 100,
        (mean_norm[i] + sem_norm[i]) * 100,
        color=color,
        alpha=0.25,
    )

ax.set_xlabel("Time / min")
ax.set_ylabel("Brush Digestion / %")
ax.set_ylim(0, 100)
ax.legend(frameon=False, fontsize=6)

format_axes(ax)
save_plot(fig, os.path.join(here, "benzonase-digestion-2"))
