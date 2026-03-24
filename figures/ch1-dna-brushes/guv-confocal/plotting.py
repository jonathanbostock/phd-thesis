"""Analysis and plotting of GUV confocal line-scan time-series data.

Reads manually annotated cross-sectional profiles from H5 files, computes
membrane fluorescence using a Gaussian kernel at the annotated boundary
positions minus background, and plots all run-N/ samples stacked on a shared
x-axis.
"""

import re
from pathlib import Path
from typing import cast

import h5py
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils.plotting import format_axes, setup_plot_style

GAUSSIAN_SIGMA_UM = 1.0

CONDITION_ORDER = ["After DNA brush addition", "After cyclodextrin addition"]

CONDITION_LABELS = {
    "After DNA brush addition": "+ DNA Construct",
    "After cyclodextrin addition": "+ Cyclodextrin",
}


def find_h5_files(run_dir: Path) -> dict[str, Path]:
    """Find H5 profile files in a run directory, keyed by condition name."""
    result: dict[str, Path] = {}
    for h5_path in sorted(run_dir.glob("*.h5")):
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
    """Gaussian-weighted mean at membrane markers minus mean outside markers."""
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


def find_run_dirs(base_dir: Path) -> list[Path]:
    """Return sorted list of run-N/ subdirectory paths."""
    pattern = re.compile(r"^run-\d+$")
    runs = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and pattern.match(d.name)],
        key=lambda p: int(re.search(r"\d+", p.name).group()),  # type: ignore[union-attr]
    )
    return runs


def plot_run_on_axis(
    data: dict[str, pd.DataFrame],
    ax: matplotlib.axes.Axes,
    dna_x_range: float,
    cyclo_x_range: float,
    show_x_labels: bool,
) -> None:
    """Plot one run's two conditions onto ax with a shared x-axis layout.

    The DNA phase is mapped to 0..dna_x_range and the cyclodextrin phase to
    dna_x_range+gap..dna_x_range+gap+cyclo_x_range, matching the scale used
    by all other rows.
    """
    palette = sns.color_palette("colorblind")

    gap = dna_x_range * 0.1
    cd_offset = dna_x_range + gap

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

        # Individual GUV traces
        for guv_id in df["guv_id"].unique():
            guv = df[df["guv_id"] == guv_id].sort_values("frame")  # type: ignore[call-overload]
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
            linewidth=2.0,
            alpha=0.9,
            label=CONDITION_LABELS[condition],
        )

        if show_x_labels:
            for t in range(0, int(max_t) + 1, 15):
                tick_positions.append(t + offset)
                tick_labels.append(str(t))

    # Divider between conditions
    ax.axvline(
        dna_x_range + gap / 2,
        color="gray",
        linestyle="--",
        linewidth=0.8,
        alpha=0.5,
    )

    ax.set_xlim(-1, dna_x_range + gap + cyclo_x_range + 1)
    ax.set_ylim(0, None)

    if show_x_labels:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=7)
        ax.set_xlabel("Time / min")
    else:
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    ax.set_ylabel("Membrane fluorescence", fontsize=8)
    format_axes(ax)


def main() -> None:
    base_dir = Path(__file__).parent
    run_dirs = find_run_dirs(base_dir)

    if not run_dirs:
        print("No run-N/ directories found.")
        return

    print(f"Found {len(run_dirs)} run dir(s)")

    # Load all data first so we can determine global x-axis limits
    all_data: list[dict[str, pd.DataFrame]] = []
    for run_dir in run_dirs:
        h5_files = find_h5_files(run_dir)
        data: dict[str, pd.DataFrame] = {}
        for condition, h5_path in h5_files.items():
            print(f"  Loading {run_dir.name}/{h5_path.name}...")
            df = load_condition_data(h5_path)
            if condition == CONDITION_ORDER[1]:
                df = df[df["frame"] <= 30].copy()
            n_guvs = df["guv_id"].nunique()
            n_frames = df["frame"].nunique()
            print(f"    {n_guvs} GUVs across {n_frames} frames")
            data[condition] = df
        all_data.append(data)

    # Shared x-axis: scale to the longest DNA and longest cyclodextrin run
    max_dna_frames = max(
        float(d[CONDITION_ORDER[0]]["frame"].max())
        for d in all_data
        if CONDITION_ORDER[0] in d and not d[CONDITION_ORDER[0]].empty
    )
    max_cyclo_frames = max(
        float(d[CONDITION_ORDER[1]]["frame"].max())
        for d in all_data
        if CONDITION_ORDER[1] in d and not d[CONDITION_ORDER[1]].empty
    )

    n_runs = len(run_dirs)
    setup_plot_style()
    fig, axes = plt.subplots(
        n_runs,
        1,
        figsize=(100 / 25.4, 45 / 25.4 * n_runs),
        sharex=False,  # handled manually via set_xlim
    )
    if n_runs == 1:
        axes = [axes]

    for i, (ax, run_dir, data) in enumerate(zip(axes, run_dirs, all_data)):
        is_bottom = i == n_runs - 1
        plot_run_on_axis(
            data,
            ax,
            dna_x_range=max_dna_frames,
            cyclo_x_range=max_cyclo_frames,
            show_x_labels=is_bottom,
        )
        ax.text(
            0.01,
            0.97,
            run_dir.name,
            transform=ax.transAxes,
            fontsize=7,
            va="top",
        )
        if i == 0:
            ax.legend(frameon=False, fontsize=7, loc="upper right")

    plt.tight_layout()
    out_path = base_dir / "guv-confocal-all-runs.svg"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
