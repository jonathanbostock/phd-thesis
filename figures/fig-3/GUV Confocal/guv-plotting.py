"""
GUV Membrane Fluorescence Analysis

Analyzes confocal microscopy data to track GUV membrane fluorescence
after DNA brush addition and after cyclodextrin treatment.

Processes each sample folder separately, outputting results to each folder.

Usage:
    uv run python guv-plotting.py
"""

import math
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from PIL import Image, ImageDraw, ImageFont
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from utils import defaults
from utils.guv_tracking import (
    TrackingConfig,
    generate_debug_images_for_lif,
    load_lif_series,
    process_lif_file,
)
from utils.plotting import format_axes, save_plot, setup_plot_style

# Custom colormap: black (0,0,0) to green (0,1,0) for fluorescence snapshots
FLUORESCENCE_CMAP = LinearSegmentedColormap.from_list(
    "black_to_green", [(0, 0, 0), (0, 1, 0)]
)

# === Configuration ===
CONFIG = TrackingConfig(
    fluorescence_channel=0,  # Channel 0 = fluorescence
    brightfield_channel=1,  # Channel 1 = brightfield
    min_diameter_um=5.0,  # Lowered from 10 to detect smaller vesicles
    max_diameter_um=50.0,
    annulus_width_px=3,
    # Use fluorescence-based detection (dark region detection)
    detection_method="fluorescence",
)

# Files to process within each sample folder (only "after" files)
FILES_TO_PROCESS: Dict[str, str] = {
    "After DNA brush addition.lif": "after_dna",
    "After cyclodextrin addition.lif": "after_cd",
}


def find_lif_file(directory: Path, canonical_name: str) -> Optional[Path]:
    """Find a .lif file by name, case-insensitive."""
    canonical_lower = canonical_name.lower()
    for f in directory.iterdir():
        if f.name.lower() == canonical_lower:
            return f
    return None


CONDITION_LABELS = {
    "after_dna": "After DNA Brush",
    "after_cd": "After Cyclodextrin",
}

# Okabe-Ito colorblind-safe palette
OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "sky_blue": "#56B4E9",
    "yellow": "#F0E442",
    "purple": "#CC79A7",
}


def draw_scale_bar(image: np.ndarray, pixel_size_um: float) -> np.ndarray:
    """Draw a scale bar on the image with auto-calculated length.

    The scale bar length is automatically chosen to be the nearest value
    in the form (1, 2, 5) * 10^n that is closest to 2/5 the width of the image.

    Parameters
    ----------
    image : np.ndarray
        Input image array (8-bit, can be grayscale or RGB/BGR)
    pixel_size_um : float
        Size of each pixel in micrometers

    Returns
    -------
    np.ndarray
        Image with scale bar drawn in bottom-right corner (8-bit BGR)
    """
    img_8bit = image.copy()

    # Handle grayscale vs RGB
    if img_8bit.ndim == 2:
        img_8bit = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)

    height, width = img_8bit.shape[:2]

    # Calculate target scale bar length: 2/5 of image width in µm
    target_width_px = width * 2 / 5.0
    target_width_um = target_width_px * pixel_size_um

    # Find nearest (1, 2, 5) * 10^n value
    log_value = math.log10(target_width_um)
    exponent = math.floor(log_value)
    mantissa = target_width_um / (10**exponent)

    # Find closest value from [1, 2, 5, 10]
    candidates = [1, 2, 5, 10]
    closest_mantissa = min(candidates, key=lambda x: abs(x - mantissa))

    # If we picked 10, increment the exponent and use 1
    if closest_mantissa == 10:
        exponent += 1
        closest_mantissa = 1

    # Calculate final scale bar length in µm
    scale_bar_um = closest_mantissa * (10**exponent)
    scale_bar_length_px = int(scale_bar_um / pixel_size_um)

    # Define padding and bar properties
    padding = 60  # Increased to move bar inward from edge
    bar_thickness = 16
    text_offset = 10

    # Calculate positions (bottom-right corner)
    bar_x1 = width - scale_bar_length_px - padding
    bar_x2 = width - padding
    bar_y = height - padding

    # Draw white scale bar
    cv2.rectangle(
        img_8bit,
        (bar_x1, bar_y - bar_thickness),
        (bar_x2, bar_y),
        (255, 255, 255),
        -1,
    )

    # Add text label above the bar using PIL for proper Unicode support
    text = f"{int(scale_bar_um)} µm"

    # Convert to PIL Image for text rendering
    img_pil = Image.fromarray(cv2.cvtColor(img_8bit, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # Use default font at larger size (PIL handles Unicode properly)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 36)
    except OSError:
        font = ImageFont.load_default()

    # Get text bounding box to center it above the bar
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_x = bar_x1 + (scale_bar_length_px - text_width) // 2
    text_y = bar_y - bar_thickness - text_offset - text_height

    # Draw white text
    draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255))

    # Convert back to OpenCV format
    img_8bit = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    return img_8bit


def save_snapshots_at_timepoints(
    sample_dir: Path,
    condition: str,
    timepoints: List[float],
) -> None:
    """
    Save fluorescence and brightfield snapshots at specified timepoints.

    Images include a scale bar burned into the bottom-right corner.

    Parameters
    ----------
    sample_dir : Path
        Path to sample folder containing .lif files
    condition : str
        Condition name ("after_dna" or "after_cd")
    timepoints : List[float]
        Frame indices (minutes) to save snapshots for (will be converted to int)
    """
    # Map condition to filename and label
    condition_to_file = {
        "after_dna": "After DNA brush addition.lif",
        "after_cd": "After cyclodextrin addition.lif",
    }
    condition_to_label = {
        "after_dna": "DNA",
        "after_cd": "cyclodextrin",
    }

    lif_name = condition_to_file.get(condition)
    label = condition_to_label.get(condition)
    if not lif_name or not label:
        return

    lif_path = find_lif_file(sample_dir, lif_name)
    if lif_path is None:
        return

    # Create snapshots directory
    snapshots_dir = sample_dir / "snapshots"
    snapshots_dir.mkdir(exist_ok=True)

    # Load the LIF file
    fluorescence, brightfield, metadata = load_lif_series(lif_path, CONFIG)
    n_frames = metadata["n_frames"]
    pixel_size_um = metadata["pixel_size_um"]

    for t in timepoints:
        frame_idx = int(t)
        if frame_idx < 0 or frame_idx >= n_frames:
            continue

        fl_frame = fluorescence[frame_idx]
        bf_frame = brightfield[frame_idx]

        # Normalize frames to 0-1 range for colormap application
        fl_norm = (fl_frame - fl_frame.min()) / (fl_frame.max() - fl_frame.min() + 1e-8)
        bf_norm = (bf_frame - bf_frame.min()) / (bf_frame.max() - bf_frame.min() + 1e-8)

        # Apply fluorescence colormap (black to green) and convert to 8-bit BGR
        fl_colored = FLUORESCENCE_CMAP(fl_norm)[:, :, :3]  # Drop alpha channel
        fl_8bit = (fl_colored * 255).astype(np.uint8)
        fl_8bit = cv2.cvtColor(fl_8bit, cv2.COLOR_RGB2BGR)

        # Apply grayscale colormap and convert to 8-bit BGR
        bf_8bit = (bf_norm * 255).astype(np.uint8)
        bf_8bit = cv2.cvtColor(bf_8bit, cv2.COLOR_GRAY2BGR)

        # Add scale bars
        fl_with_bar = draw_scale_bar(fl_8bit, pixel_size_um)
        bf_with_bar = draw_scale_bar(bf_8bit, pixel_size_um)

        # Save images
        fl_path = snapshots_dir / f"after_{label}_{int(t)}_minutes_fluorescence.png"
        bf_path = snapshots_dir / f"after_{label}_{int(t)}_minutes_brightfield.png"
        cv2.imwrite(str(fl_path), fl_with_bar)
        cv2.imwrite(str(bf_path), bf_with_bar)

    print(f"  Snapshots saved to {snapshots_dir}")


def save_to_hdf5(
    h5_path: Path,
    condition: str,
    measurements: pd.DataFrame,
    summary: Dict,
) -> None:
    """
    Save measurements for one condition to HDF5.

    Structure:
    /condition/vesicle_id/{time_seconds, x_um, y_um, radius_um, ...}
    """
    with h5py.File(h5_path, "a") as f:
        # Remove existing condition group if present
        if condition in f:
            del f[condition]

        cond_group = f.create_group(condition)

        # Store metadata as attributes
        for key, value in summary.items():
            if value is not None:
                cond_group.attrs[key] = value

        # Store each vesicle's data
        for vesicle_id in measurements["vesicle_id"].unique():
            vesicle_data = measurements[measurements["vesicle_id"] == vesicle_id]
            vesicle_group = cond_group.create_group(f"vesicle_{vesicle_id}")

            # Store arrays for each measurement
            for col in [
                "time_seconds",
                "x_um",
                "y_um",
                "radius_um",
                "membrane_fluorescence_mean",
            ]:
                vesicle_group.create_dataset(col, data=np.array(vesicle_data[col]))


def load_from_hdf5(h5_path: Path) -> pd.DataFrame:
    """
    Load all measurements from HDF5 file into a DataFrame.
    """
    rows: List[Dict] = []

    with h5py.File(h5_path, "r") as f:
        for condition in f.keys():  # type: ignore[union-attr]
            if condition == "metadata":
                continue

            cond_group = f[condition]  # type: ignore[index]

            for vesicle_name in cond_group.keys():  # type: ignore[union-attr]
                if not vesicle_name.startswith("vesicle_"):
                    continue

                vesicle_id = int(vesicle_name.split("_")[1])
                vesicle_group = cond_group[vesicle_name]  # type: ignore[index]

                time_data = np.array(vesicle_group["time_seconds"])  # type: ignore[index]
                x_data = np.array(vesicle_group["x_um"])  # type: ignore[index]
                y_data = np.array(vesicle_group["y_um"])  # type: ignore[index]
                radius_data = np.array(vesicle_group["radius_um"])  # type: ignore[index]
                fl_data = np.array(vesicle_group["membrane_fluorescence_mean"])  # type: ignore[index]

                n_frames = len(time_data)
                for i in range(n_frames):
                    rows.append(
                        {
                            "condition": condition,
                            "vesicle_id": vesicle_id,
                            "frame": i,
                            "time_seconds": float(time_data[i]),
                            "x_um": float(x_data[i]),
                            "y_um": float(y_data[i]),
                            "radius_um": float(radius_data[i]),
                            "membrane_fluorescence_mean": float(fl_data[i]),
                        }
                    )

    return pd.DataFrame(rows)


def plot_membrane_fluorescence_timecourse(
    df: pd.DataFrame, output_dir: Path
) -> Dict[str, List[float]]:
    """
    Plot membrane fluorescence vs time for both conditions on one axis.

    Both conditions plotted sequentially with a breakpoint between them.
    X-axis resets for each segment. Time in minutes.

    Returns
    -------
    tick_timepoints : Dict[str, List[float]]
        Dictionary mapping condition to list of tick timepoints (in minutes/frames)
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(defaults.fig_width * 1.5, defaults.fig_height))

    conditions = ["after_dna", "after_cd"]
    colors = [OKABE_ITO["blue"], OKABE_ITO["orange"]]

    # Calculate time offset for second condition (with gap)
    # Note: time_seconds is actually frame number, where each frame = 1 minute
    dna_data = df[df["condition"] == "after_dna"]
    max_dna_time_min = dna_data["time_seconds"].max() if len(dna_data) > 0 else 0
    gap_min = max_dna_time_min * 0.15  # 15% gap between segments
    time_offset = max_dna_time_min + gap_min

    # Track tick positions and labels for custom x-axis
    tick_positions: List[float] = []
    tick_labels: List[str] = []

    # Track tick timepoints per condition for snapshot saving
    tick_timepoints: Dict[str, List[float]] = {}

    for idx, condition in enumerate(conditions):
        cond_data = df[df["condition"] == condition]
        color = colors[idx]
        offset = time_offset if condition == "after_cd" else 0

        if len(cond_data) == 0:
            tick_timepoints[condition] = []
            continue

        # time_seconds is frame number = minutes (1 frame per minute)
        time_min = cond_data["time_seconds"]

        # Plot individual vesicle traces
        grouped = cond_data.groupby("vesicle_id")
        unique_ids = grouped.ngroups
        for _, vesicle_data in grouped:
            vesicle_time = vesicle_data["time_seconds"] + offset
            ax.plot(
                vesicle_time,
                vesicle_data["membrane_fluorescence_mean"],
                color=color,
                alpha=0.3,
                linewidth=0.5,
            )

        # Plot mean trace (thick line)
        mean_trace = cond_data.groupby("time_seconds")[
            "membrane_fluorescence_mean"
        ].mean()
        mean_time = mean_trace.index + offset
        label = f"{CONDITION_LABELS[condition]} (n={unique_ids})"
        ax.plot(
            mean_time,
            mean_trace.values,
            color=color,
            linewidth=2,
            label=label,
        )

        # Add ticks for this segment (0, mid, max)
        segment_max = time_min.max()
        segment_ticks = [0, segment_max / 2, segment_max]
        tick_timepoints[condition] = segment_ticks

        for t in segment_ticks:
            tick_positions.append(t + offset)
            tick_labels.append(f"{t:.0f}")

    # Add breakpoint indicator (vertical dashed line)
    if max_dna_time_min > 0:
        breakpoint_x = max_dna_time_min + gap_min / 2
        ax.axvline(breakpoint_x, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    # Set custom x-axis ticks
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Membrane Fluorescence (a.u.)")
    ax.legend(frameon=False, loc="upper left")

    format_axes(ax)

    # Put ticks outside (must be after format_axes which sets direction="in")
    ax.tick_params(axis="both", direction="out")

    plt.tight_layout()
    save_plot(fig, str(output_dir / "membrane_fluorescence_timecourse"))
    plt.close(fig)

    return tick_timepoints


def plot_condition_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot comparison of membrane fluorescence across conditions.

    Shows final timepoint fluorescence as scatter plot with mean bars.
    """
    setup_plot_style()

    # Get final timepoint for each vesicle
    final_data = df.loc[df.groupby(["condition", "vesicle_id"])["frame"].idxmax()]

    fig, ax = plt.subplots(figsize=(defaults.fig_width, defaults.fig_height))
    colors = [OKABE_ITO["blue"], OKABE_ITO["orange"]]

    condition_order = ["after_dna", "after_cd"]
    condition_labels = ["After DNA\nBrush", "After\nCyclodextrin"]

    for i, condition in enumerate(condition_order):
        cond_data = final_data[final_data["condition"] == condition]
        values = cond_data["membrane_fluorescence_mean"].values

        if len(values) == 0:
            continue

        # Scatter with jitter
        x_jitter = np.random.uniform(-0.2, 0.2, len(values))
        ax.scatter(
            i + x_jitter,
            values,
            color=colors[i],
            marker="o",
            edgecolors="black",
            linewidths=0.5,
            s=30,
            alpha=0.7,
        )

        # Add mean bar
        mean_val = np.mean(values)
        ax.hlines(mean_val, i - 0.3, i + 0.3, colors="black", linewidths=2)

    ax.set_xticks(range(len(condition_order)))
    ax.set_xticklabels(condition_labels)
    ax.set_ylabel("Final Membrane Fluorescence (a.u.)")

    format_axes(ax)

    # Put ticks outside (must be after format_axes which sets direction="in")
    ax.tick_params(axis="both", direction="out")

    plt.tight_layout()
    save_plot(fig, str(output_dir / "condition_comparison"))
    plt.close(fig)


def print_summary(summaries: List[Dict], sample_name: str) -> None:
    """Print summary statistics to console."""
    print(f"\n=== {sample_name} Processing Summary ===")
    for summary in summaries:
        print(f"\n{summary.get('file', 'Unknown')}:")
        print(f"  Condition: {summary.get('condition', 'N/A')}")
        print(
            f"  Vesicles detected (avg/frame): {summary.get('n_vesicles_detected', 'N/A')}"
        )
        print(f"  Vesicles tracked: {summary.get('n_vesicles_tracked', 'N/A')}")
        print(f"  Excluded (incomplete): {summary.get('excluded_incomplete', 0)}")
        print(f"  Excluded (border): {summary.get('excluded_border', 0)}")
        print(f"  Excluded (jump): {summary.get('excluded_jump', 0)}")


def process_sample_folder(sample_dir: Path) -> None:
    """Process a single sample folder."""
    sample_name = sample_dir.name
    print(f"\n{'=' * 60}")
    print(f"Processing {sample_name}")
    print("=" * 60)

    results_dir = sample_dir / "results"
    results_dir.mkdir(exist_ok=True)

    h5_path = results_dir / "guv_tracking_results.h5"

    # Remove existing HDF5 file to start fresh
    if h5_path.exists():
        h5_path.unlink()

    all_summaries: List[Dict] = []

    # Process each .lif file in the folder
    for lif_name, condition in FILES_TO_PROCESS.items():
        lif_path = find_lif_file(sample_dir, lif_name)
        if lif_path is None:
            print(f"Warning: {lif_name} not found, skipping")
            continue

        print(f"Processing {lif_name}...")

        try:
            measurements, summary = process_lif_file(lif_path, CONFIG)
            print(f"  Tracked {summary['n_vesicles_tracked']} vesicles")

            # Generate debug images with Otsu threshold overlay
            debug_dir = results_dir / "debug"
            generate_debug_images_for_lif(lif_path, debug_dir, CONFIG)
        except Exception as e:
            print(f"  Error processing {lif_name}: {e}")
            import traceback

            traceback.print_exc()
            continue

        # Add file info to summary
        summary["file"] = lif_name
        summary["condition"] = condition

        # Save to HDF5
        if len(measurements) > 0:
            save_to_hdf5(h5_path, condition, measurements, summary)
            print(f"  Saved {len(measurements)} measurements to HDF5")
        else:
            print(f"  No vesicles tracked for {lif_name}")

        all_summaries.append(summary)

    # Save summary statistics CSV
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_df.to_csv(results_dir / "summary_stats.csv", index=False)
        print_summary(all_summaries, sample_name)

    # Generate plots if we have data
    if h5_path.exists():
        print("\nGenerating plots...")
        combined_df = load_from_hdf5(h5_path)

        if len(combined_df) > 0:
            tick_timepoints = plot_membrane_fluorescence_timecourse(
                combined_df, sample_dir
            )
            plot_condition_comparison(combined_df, sample_dir)
            print("Plots saved.")

            # Save snapshots at tick timepoints
            print("\nSaving snapshots at tick timepoints...")
            for condition, timepoints in tick_timepoints.items():
                if timepoints:
                    save_snapshots_at_timepoints(sample_dir, condition, timepoints)
        else:
            print("No data to plot.")
    else:
        print("No HDF5 file created - no data to plot.")


def main() -> None:
    """Main analysis function - processes all sample folders."""
    base_dir = Path(__file__).parent

    # Find all sample folders
    sample_folders = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("Sample")]
    )

    if not sample_folders:
        print("No sample folders found!")
        return

    print(
        f"Found {len(sample_folders)} sample folder(s): {[f.name for f in sample_folders]}"
    )

    for sample_dir in sample_folders:
        process_sample_folder(sample_dir)

    print("\n" + "=" * 60)
    print("All samples processed!")


if __name__ == "__main__":
    main()
