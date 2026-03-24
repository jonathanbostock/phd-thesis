# Jonathan Bostock
# pH Sensitive Brush Analysis (2023 data)
# Plots pH calibration, i-motif vs standard brush pH response,
# and standard brush length vs pH response.

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize

from utils.plotting import setup_plot_style, format_axes, save_plot, plot_mean_sem_overlay
from utils import defaults

P1MI_KEY = "Peak 1 Mean by Intensity ordered by area (nm)"


def load_ph_calibration(filepath: str) -> tuple[pd.DataFrame, object]:
    """Load pH calibration TSV and return (dataframe, pH interpolation function)."""
    cal_df = pd.read_csv(filepath, sep="\t")
    cal_df["mean_pH"] = cal_df[["pH 1", "pH 2", "pH 3"]].mean(axis=1)

    def get_pH(hcl_vol: float) -> float:
        if float(hcl_vol) == int(hcl_vol) and 0 <= int(hcl_vol) < len(cal_df):
            return cal_df.iloc[int(hcl_vol)]["mean_pH"]
        hcl_floor = int(np.floor(hcl_vol))
        hcl_ceil = hcl_floor + 1
        frac = hcl_vol - hcl_floor
        return (
            cal_df.iloc[hcl_floor]["mean_pH"] * (1 - frac)
            + cal_df.iloc[hcl_ceil]["mean_pH"] * frac
        )

    return cal_df, get_pH


def process_popc_hcl_data(filepath: str, get_pH) -> pd.DataFrame:
    """Process POPC-only HCl DLS data. ΔD relative to 0-HCl baseline per batch."""
    df = pd.read_csv(filepath, sep="\t")
    df = df[df["Quality Indicator"] == "GoodData"].set_index("Sample Name")

    records = []
    for idx in df.index:
        parts = idx.split()
        batch = parts[1]
        hcl_vol = int(parts[2])
        p1mi = df.loc[idx, P1MI_KEY]

        zero_key = f"Ext {batch} 0 HCl"
        if zero_key not in df.index:
            continue

        p1mi_zero = df.loc[zero_key, P1MI_KEY]
        records.append(
            {
                "condition": "No DNA",
                "hcl_vol": hcl_vol,
                "pH": get_pH(hcl_vol),
                "delta_p1mi": p1mi - p1mi_zero,
            }
        )

    return pd.DataFrame(records)


def process_imotif_brush_data(filepath: str, get_pH) -> pd.DataFrame:
    """Process 30bp + I-motif brush HCl data. ΔD relative to POPC ctrl per batch."""
    df = pd.read_csv(filepath, sep="\t")
    df = df[df["Quality Indicator"] == "GoodData"].set_index("Sample Name")

    condition_map = {"30 bp": "30 bp", "I Motif": "I-motif"}
    records = []

    for idx in df.index:
        parts = idx.split()
        batch = parts[1]
        condition_str = " ".join(parts[2:4])  # "30 bp", "I Motif", or "Ctrl ..."

        if condition_str not in condition_map:
            continue

        hcl_vol = int(parts[4])
        p1mi = df.loc[idx, P1MI_KEY]

        ctrl_key = f"Ext {batch} Ctrl"
        if ctrl_key not in df.index:
            continue

        records.append(
            {
                "condition": condition_map[condition_str],
                "hcl_vol": hcl_vol,
                "pH": get_pH(hcl_vol),
                "delta_p1mi": p1mi - df.loc[ctrl_key, P1MI_KEY],
            }
        )

    return pd.DataFrame(records)


def process_brush_length_data(filepath: str, get_pH) -> pd.DataFrame:
    """Process standard brush pH response data across different brush lengths."""
    df = pd.read_csv(filepath, sep="\t")
    df = df[df["Quality Indicator"] == "GoodData"].set_index("Sample Name")

    length_dict = {"0bp": 0, "15bp": 15, "30bp": 30, "42bp": 42}

    # First pass: collect controls per batch
    ctrl_dict: dict[str, float] = {}
    for idx in df.index:
        parts = idx.split()
        batch = parts[1]
        if "Ctrl" in idx:
            ctrl_dict[batch] = df.loc[idx, P1MI_KEY]

    # Second pass: collect data
    records = []
    for idx in df.index:
        parts = idx.split()
        if len(parts) < 4:
            continue
        batch = parts[1]
        length_str = parts[2]
        hcl_str = parts[3]

        if length_str not in length_dict or batch not in ctrl_dict:
            continue

        hcl_vol = int(hcl_str.replace("HCl", ""))
        p1mi = df.loc[idx, P1MI_KEY]

        records.append(
            {
                "length_bp": length_dict[length_str],
                "hcl_vol": hcl_vol,
                "pH": get_pH(hcl_vol),
                "delta_p1mi": p1mi - ctrl_dict[batch],
            }
        )

    return pd.DataFrame(records)


def plot_ph_calibration(cal_df: pd.DataFrame) -> plt.Figure:
    """Plot pH calibration: raw triplicates (pH 1, 2, 3) vs HCl volume added.

    Horizontal reference lines mark the working pH of Tris (8.1), PIPES (6.8),
    and acetate (4.8) buffers, with Gaussian fuzzy bands spanning each buffer's
    effective buffering range (pKa ± 1 pH unit).
    """
    setup_plot_style()
    colors = sns.color_palette("colorblind")
    dark_grey = "#555555"

    fig, ax = plt.subplots(
        figsize=(
            defaults.fig_width * defaults.small_fig_scale,
            defaults.fig_height * defaults.small_fig_scale,
        )
    )

    # Buffer reference lines: (pH, colorblind index)
    buffers = [
        (8.1, colors[0]),   # Tris
        (6.5, colors[1]),   # Bis-Tris
        (4.8, colors[2]),   # Acetate
    ]

    x_min = float(cal_df["Vol HCl"].min())
    x_max = float(cal_df["Vol HCl"].max())
    buffering_sigma = 1.5   # half-width of buffering region in pH units
    n_shells = 40           # shells for smooth Gaussian gradient
    max_alpha = 0.5         # cumulative peak opacity at centre

    for ph_centre, col in buffers:
        # Gaussian fuzzy band: stack shells from outermost inward so alpha
        # accumulates towards the centre.
        for k in range(n_shells, 0, -1):
            frac = k / n_shells
            half_w = buffering_sigma * frac
            layer_alpha = max_alpha * np.exp(-0.5 * frac ** 2) / n_shells
            ax.axhspan(
                ph_centre - half_w,
                ph_centre + half_w,
                color=col,
                alpha=layer_alpha,
                linewidth=0,
                zorder=1,
            )
        ax.axhline(ph_centre, color=col, linewidth=0.8, linestyle=":", alpha=0.9, zorder=2)

    for col_name in ["pH 1", "pH 2", "pH 3"]:
        ax.scatter(
            cal_df["Vol HCl"],
            cal_df[col_name],
            color=dark_grey,
            marker="o",
            edgecolors="black",
            linewidths=0.5,
            s=30,
            alpha=0.9,
            zorder=5,
        )

    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("V/V% HCl added")
    ax.set_ylabel("pH")
    format_axes(ax)
    plt.tight_layout()
    return fig


def plot_brushes_ph_changes(
    no_dna_df: pd.DataFrame, brush_df: pd.DataFrame
) -> plt.Figure:
    """Plot ΔD vs pH for No DNA (POPC only), 30bp brush, and I-motif brush."""
    setup_plot_style()
    colors = sns.color_palette("colorblind")
    markers = ["+", "x", "1"]

    fig, ax = plt.subplots(
        figsize=(
            defaults.fig_width * defaults.small_fig_scale,
            defaults.fig_height * defaults.small_fig_scale,
        )
    )

    combined = pd.concat([no_dna_df, brush_df], ignore_index=True)
    conditions = ["No DNA", "30 bp", "I-motif"]

    for i, condition in enumerate(conditions):
        data = combined[combined["condition"] == condition]
        x = data["pH"].to_numpy()
        y = data["delta_p1mi"].to_numpy()
        ax.scatter(
            x,
            y,
            color=colors[i],
            marker=markers[i],
            linewidths=0.8,
            s=35,
            label=condition,
        )
        plot_mean_sem_overlay(ax, x, y, color=colors[i])

    ax.set_xlabel("pH")
    ax.set_ylabel(r"$\Delta D$ / nm")
    ax.set_xlim(8.1, 5.4)
    ax.legend(frameon=False)
    format_axes(ax)
    plt.tight_layout()
    return fig


def plot_ph_response_standard_brushes(data_df: pd.DataFrame) -> plt.Figure:
    """Plot ΔD vs pH for standard brushes of different duplex lengths (viridis gradient)."""
    setup_plot_style()

    fig, ax = plt.subplots(
        figsize=(
            defaults.fig_width * defaults.small_fig_scale,
            defaults.fig_height * defaults.small_fig_scale,
        )
    )

    lengths = sorted(data_df["length_bp"].unique())
    cmap = plt.colormaps["viridis"]
    norm = Normalize(vmin=min(lengths), vmax=max(lengths))
    markers = ["+", "x", "1", "2"]

    for i, length in enumerate(lengths):
        data = data_df[data_df["length_bp"] == length]
        color = cmap(norm(length))
        x = data["pH"].to_numpy()
        y = data["delta_p1mi"].to_numpy()
        ax.scatter(
            x,
            y,
            color=color,
            marker=markers[i],
            linewidths=0.8,
            s=35,
            label=f"{length} bp",
        )
        plot_mean_sem_overlay(ax, x, y, color=color)

    ax.set_xlabel("pH")
    ax.set_ylabel(r"$\Delta D$ / nm")
    ax.set_xlim(8.1, 5.4)
    ax.legend(frameon=False)
    format_axes(ax)
    plt.tight_layout()
    return fig


def main() -> None:
    cal_df, get_pH = load_ph_calibration("pH_calibration.tsv")

    no_dna_df = process_popc_hcl_data("2023-07-27 POPC HCl DLS Data.tsv", get_pH)
    brush_df = process_imotif_brush_data("2023-08-01 HCl 30bp + IM.tsv", get_pH)
    length_df = process_brush_length_data("2023-08-24 pH Response 1+2.tsv", get_pH)

    fig1 = plot_ph_calibration(cal_df)
    save_plot(fig1, "pH Calibration Curve")

    fig2 = plot_brushes_ph_changes(no_dna_df, brush_df)
    save_plot(fig2, "Brushes and pH Changes")

    fig3 = plot_ph_response_standard_brushes(length_df)
    save_plot(fig3, "pH Response of Standard Brushes")

    plt.show()


if __name__ == "__main__":
    main()
