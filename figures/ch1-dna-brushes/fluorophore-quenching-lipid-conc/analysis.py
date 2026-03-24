"""
FAM-BHQ fluorescence quenching vs lipid (SUV) concentration.

Plot 1: Before γ-CD treatment only – filled markers + line through means.
Plot 2: Before, 5 min γ-CD, 25 min γ-CD – open markers (+/x/1) + line
        through means, three equally-spaced viridis colours.

Both plots use a broken x-axis: 0–80 µL on the left, 800 µL on the right.
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats

from utils.plotting import setup_plot_style, format_axes
from utils import defaults

# 40 % larger than the standard multi-axis target (30.375 × 23.625 mm)
_AX_W_MM = 30.375 * 1.4
_AX_H_MM = 23.625 * 1.4


def save_plot_large(fig: plt.Figure, filename_base: str) -> None:
    subplot_axes = [ax for ax in fig.get_axes() if ax.get_subplotspec() is not None]
    plt.tight_layout(pad=0.5)
    pos = subplot_axes[0].get_position()
    fig.set_size_inches((_AX_W_MM / 25.4) / pos.width, (_AX_H_MM / 25.4) / pos.height)
    plt.tight_layout(pad=0.5)
    fig.savefig(f"{filename_base}.svg", bbox_inches="tight")

# µL of SUVs added; col 1 = blank, cols 2-12 = these concentrations
CONCENTRATIONS = [0, 2.5, 5, 7.5, 10, 15, 20, 40, 60, 80, 800]


def process_raw(path: str) -> pd.DataFrame:
    """
    Read a plate-reader CSV (nrows=17), normalise each well by its row-blank
    and row-control, return a DataFrame indexed by well name with an 'Average'
    column containing the mean intensity over wavelengths 511-515 nm.

    Column layout: 'Sample X{n}' (wavelength) | 'Unnamed: {2n-1}' (intensity)
    Row B is stored in reverse order (B12→B1) in the CSV.
    """
    df = pd.read_csv(path, nrows=17)

    usecols = [f"Unnamed: {i}" for i in range(1, 73, 2)]
    col_names = [c[7:] for c in df.columns if "Sample" in c]

    working = df.loc[1:16, usecols].copy()
    working.columns = col_names
    working.index = [str(i) for i in range(510, 526)]
    working = working.astype(float)

    for letter in ["A", "B", "C"]:
        blank = working[f"{letter}1"].copy()
        control = working[f"{letter}2"].copy() - blank
        for n in range(1, 13):
            col = f"{letter}{n}"
            working[col] = (working[col] - blank) / control

    result = working.T  # wells × wavelengths
    result["Average"] = result.apply(lambda s: float(np.mean(list(s)[1:6])), axis=1)
    return result


def get_replicates(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, y) arrays of individual replicate values for all CONCENTRATIONS."""
    xs, ys = [], []
    for i, conc in enumerate(CONCENTRATIONS):
        col = i + 2  # col 1 = blank, col 2 = 0 µL, …, col 12 = 800 µL
        for letter in ["A", "B", "C"]:
            xs.append(conc)
            ys.append(float(data.loc[f"{letter}{col}", "Average"]))
    return np.array(xs), np.array(ys)


def means_line(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: tuple) -> None:
    """Connect per-concentration means with a line."""
    unique_x = np.unique(x)
    means = np.array([y[x == xi].mean() for xi in unique_x])
    ax.plot(unique_x, means, color=color, linewidth=1.5, zorder=3)


def make_broken_fig() -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    """
    Create a figure with a broken x-axis: main range 0–80 µL on the left,
    800 µL on the right.  Uses a flat 4-column GridSpec (main | spacer |
    right-strip) so tight_layout works correctly.

    Returns (fig, ax_left, ax_right).
    """
    # width_ratios: [main 0-80 | spacer | 800-point strip]
    w_left, w_right = 3.0, 0.6
    fig = plt.figure(figsize=(defaults.fig_width, defaults.fig_height))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[w_left, 0.3, w_right], wspace=0.0)
    ax_l = fig.add_subplot(gs[0])
    ax_r = fig.add_subplot(gs[2], sharey=ax_l)

    # Break diagonal lines – compensate d_x so both "/" marks have the same
    # physical slope regardless of the differing axis widths.
    d_y = 0.08
    d_x_l = d_y
    d_x_r = d_y * (w_left / w_right)
    bkw = dict(color="k", clip_on=False, linewidth=0.8)
    ax_l.plot([1 - d_x_l, 1 + d_x_l], [-d_y, +d_y], transform=ax_l.transAxes, **bkw)
    ax_r.plot([-d_x_r, +d_x_r], [-d_y, +d_y], transform=ax_r.transAxes, **bkw)

    # Right-side axis formatting
    format_axes(ax_r)
    ax_r.spines["left"].set_visible(False)
    ax_r.tick_params(left=False, labelleft=False)
    ax_r.set_xlim(780, 820)
    ax_r.set_xticks([800])
    ax_r.set_xticklabels(["800"])

    return fig, ax_l, ax_r


def plot_scatter_and_line(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: tuple,
    marker: str,
    filled: bool,
    label: str = "",
    s: int = 25,
) -> None:
    scatter_kw: dict = dict(
        color=color, marker=marker, s=s, zorder=5, label=label
    )
    if filled:
        scatter_kw.update(edgecolors="black", linewidths=0.5)
    else:
        scatter_kw.update(linewidths=0.8)
    ax.scatter(x, y, **scatter_kw)
    means_line(ax, x, y, color)


def main() -> None:
    before = process_raw("2024-03-19 FAM BHQ Vesicles Before gCD 800V.csv")
    gcd5 = process_raw("2024-03-19 FAM BHQ Vesicles gCD 5min 800V.csv")
    gcd25 = process_raw("2024-03-19 FAM BHQ Vesicles gCD 25min 800V.csv")

    before_x, before_y = get_replicates(before)
    gcd5_x, gcd5_y = get_replicates(gcd5)
    gcd25_x, gcd25_y = get_replicates(gcd25)

    setup_plot_style()
    blue = sns.color_palette("colorblind")[0]
    cmap = matplotlib.colormaps["viridis"]
    c0, c1, c2 = cmap(0.0), cmap(0.5), cmap(1.0)

    # ── Plot 1: Before γ-CD ───────────────────────────────────────────────────
    fig1, ax1l, ax1r = make_broken_fig()

    for ax in (ax1l, ax1r):
        mask = before_x <= 80 if ax is ax1l else before_x == 800
        plot_scatter_and_line(ax, before_x[mask], before_y[mask], blue, "o", filled=True)

    ax1l.set_xlabel("SUV Volume Added / µL")
    ax1l.set_ylabel("Batch-Normalised Fluorescence")
    format_axes(ax1l)
    save_plot_large(fig1, "fluorophore-quenching-before")

    # ── Plot 2: Before + 5min + 25min ─────────────────────────────────────────
    fig2, ax2l, ax2r = make_broken_fig()

    conditions = [
        (before_x, before_y, r"Before $\gamma$-CD", "+", c0),
        (gcd5_x, gcd5_y, r"5 min $\gamma$-CD", "x", c1),
        (gcd25_x, gcd25_y, r"25 min $\gamma$-CD", "1", c2),
    ]

    for x, y, label, marker, color in conditions:
        for ax in (ax2l, ax2r):
            mask = x <= 80 if ax is ax2l else x == 800
            lbl = label if ax is ax2l else ""
            plot_scatter_and_line(ax, x[mask], y[mask], color, marker, filled=False,
                                  label=lbl, s=35)

    ax2l.set_xlabel("SUV Volume Added / µL")
    ax2l.set_ylabel("Batch-Normalised Fluorescence")
    ax2l.legend(frameon=False, fontsize=6)
    format_axes(ax2l)
    save_plot_large(fig2, "fluorophore-quenching-all")


if __name__ == "__main__":
    main()
