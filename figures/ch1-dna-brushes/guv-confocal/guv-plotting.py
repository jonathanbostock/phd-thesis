"""
GUV Membrane Fluorescence Analysis

Reads manually annotated cross-sectional profiles from H5 files and computes
membrane fluorescence using a Gaussian kernel at the annotated boundary
positions, minus background fluorescence outside the markers.

Usage:
    uv run python guv-plotting.py
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib.axes
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import cast

from utils.plotting import format_axes, save_plot, setup_plot_style

GAUSSIAN_SIGMA_UM = 1.0  # Gaussian kernel width in micrometres

CONDITION_ORDER = ["After DNA brush addition", "After cyclodextrin addition"]

CONDITION_LABELS = {
    "After DNA brush addition": "+ DNA Construct",
    "After cyclodextrin addition": "+ Cyclodextrin",
}


def find_h5_files(sample_dir: Path) -> dict[str, Path]:
    """Find H5 profile files in a sample directory, keyed by condition name."""
    result: dict[str, Path] = {}
    for h5_path in sorted(sample_dir.glob("*.h5")):
        for condition in CONDITION_LABELS:
            if condition.lower() in h5_path.name.lower():
                result[condition] = h5_path
                break
    return result


def marker_fluorescence(
    fluorescence_profile: np.ndarray,
    positions_um: np.ndarray,
    boundary_left_um: float,
    boundary_right_um: float,
    sigma_um: float = GAUSSIAN_SIGMA_UM,
) -> float:
    """Gaussian-weighted mean at membrane markers minus mean outside markers.

    Builds a kernel that is the sum of two Gaussians centred at the left and
    right membrane boundary positions.  'At markers' is the kernel-weighted
    mean of the fluorescence profile.  'Outside markers' is the unweighted
    mean of profile points where the kernel is below 1 % of its peak, i.e.
    far from both membranes (exterior medium and GUV interior).
    """
    kernel = np.exp(
        -((positions_um - boundary_left_um) ** 2) / (2 * sigma_um**2)
    ) + np.exp(-((positions_um - boundary_right_um) ** 2) / (2 * sigma_um**2))

    at_markers = float(np.sum(fluorescence_profile * kernel) / np.sum(kernel))

    outside_mask = kernel < 0.01 * kernel.max()
    if not outside_mask.any():
        return np.nan
    outside = float(fluorescence_profile[outside_mask].mean())

    return at_markers - outside


def load_condition_data(h5_path: Path) -> pd.DataFrame:
    """Load all frames and segments from an H5 profile file into a DataFrame."""
    rows = []
    with h5py.File(h5_path, "r") as f:
        frames_group = cast(h5py.Group, f["frames"])
        for frame_name in sorted(
            frames_group.keys(), key=lambda k: int(k.split("_")[1])
        ):
            frame_idx = int(frame_name.split("_")[1])
            segs = cast(
                h5py.Group, cast(h5py.Group, frames_group[frame_name])["segments"]
            )
            for seg_name in segs.keys():
                seg = cast(h5py.Group, segs[seg_name])
                fl = np.asarray(
                    cast(h5py.Dataset, seg["fluorescence_profile"])[()],
                    dtype=np.float64,
                )
                pos = np.asarray(
                    cast(h5py.Dataset, seg["positions_um"])[()], dtype=np.float64
                )
                bl = float(seg.attrs["boundary_left_um"])  # type: ignore[arg-type]
                br = float(seg.attrs["boundary_right_um"])  # type: ignore[arg-type]
                seg_id = int(seg.attrs["segment_id"])  # type: ignore[arg-type]
                diff = marker_fluorescence(fl, pos, bl, br)
                rows.append(
                    {"frame": frame_idx, "guv_id": seg_id, "membrane_signal": diff}
                )
    return pd.DataFrame(rows)


def plot_combined_timecourse(
    data: dict[str, pd.DataFrame], ax: matplotlib.axes.Axes
) -> None:
    """Plot both conditions on a single axis with a gap between them.

    Individual GUVs are drawn as thin, low-alpha lines in the condition colour.
    The per-frame mean is drawn as a thick, high-alpha line.  X-axis is in
    minutes (1 frame = 1 minute), with tick labels showing actual minutes
    within each condition segment.
    """
    palette = sns.color_palette("colorblind")

    # Determine gap from DNA brush segment length
    dna_df = data.get(CONDITION_ORDER[0], pd.DataFrame())
    max_dna_t = float(dna_df["frame"].max()) if not dna_df.empty else 0.0
    gap = max_dna_t * 0.15
    cd_offset = max_dna_t + gap

    offsets = {CONDITION_ORDER[0]: 0.0, CONDITION_ORDER[1]: cd_offset}

    tick_positions: list[float] = []
    tick_labels: list[str] = []

    for i, condition in enumerate(CONDITION_ORDER):
        df = data.get(condition)
        if df is None or df.empty:
            continue
        color = palette[i]
        offset = offsets[condition]
        max_t = float(df["frame"].max())

        # Individual GUV lines
        for guv_id in df["guv_id"].unique():
            guv = df[df["guv_id"] == guv_id].sort_values(by="frame")  # type: ignore[call-overload]
            ax.plot(
                guv["frame"] + offset,
                guv["membrane_signal"],
                color=color,
                linewidth=0.5,
                alpha=0.35,
            )

        # Mean line
        mean_by_frame = df.groupby("frame")["membrane_signal"].mean()
        ax.plot(
            mean_by_frame.index + offset,
            mean_by_frame.values,
            color=color,
            linewidth=2.5,
            alpha=0.9,
            label=CONDITION_LABELS[condition],
        )

        # Ticks every 15 minutes — labels show actual minutes
        for t in range(0, int(max_t) + 1, 15):
            tick_positions.append(t + offset)
            tick_labels.append(str(t))

    # Divider between conditions
    if max_dna_t > 0:
        ax.axvline(
            max_dna_t + gap / 2,
            color="gray",
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
        )

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Time / min")
    ax.set_ylabel("Membrane Fluorescence")
    ax.set_yticks([])
    ax.legend(frameon=False, loc="best")
    format_axes(ax)


def process_sample(sample_dir: Path) -> None:
    """Process all H5 profile files in a sample folder and save an SVG."""
    h5_files = find_h5_files(sample_dir)
    if not h5_files:
        print(f"  No H5 profile files found in {sample_dir.name}, skipping.")
        return

    data: dict[str, pd.DataFrame] = {}
    for condition, h5_path in h5_files.items():
        print(f"  Loading {h5_path.name}...")
        df = load_condition_data(h5_path)
        if condition == "After cyclodextrin addition":
            df = df[df["frame"] <= 30].copy()
        assert isinstance(df, pd.DataFrame)
        n_guvs = df["guv_id"].nunique()
        n_frames = df["frame"].nunique()
        print(f"    {n_guvs} GUVs across {n_frames} frames")
        data[condition] = df

    setup_plot_style()
    fig, ax = plt.subplots(figsize=(120 / 25.4, 45 / 25.4))
    plot_combined_timecourse(data, ax)
    plt.tight_layout()
    out_path = str(sample_dir / "membrane_fluorescence_timecourse")
    ax.set_ylim(0, None)  # Start y-axis at zero
    save_plot(fig, out_path)
    plt.close(fig)
    print(f"  Saved {out_path}.svg")


def main() -> None:
    base_dir = Path(__file__).parent
    sample_folders = sorted(
        d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("Sample")
    )
    if not sample_folders:
        print("No sample folders found!")
        return

    print(f"Found {len(sample_folders)} sample folder(s)")
    for sample_dir in sample_folders:
        print(f"\nProcessing {sample_dir.name}...")
        process_sample(sample_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
