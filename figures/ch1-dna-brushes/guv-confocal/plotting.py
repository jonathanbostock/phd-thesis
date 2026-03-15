"""Analysis and plotting of GUV confocal micrograph time-series data."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import Normalize
from PIL import Image
from scipy.signal import find_peaks


def setup_plotting_style():
    """Configure plotting style according to project guidelines."""
    sns.set_palette("colorblind")
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["figure.facecolor"] = "white"


def load_tiff_data(filename):
    """Load and extract both channels from TIFF file."""
    img = Image.open(filename)

    img.seek(0)  # First channel (fluorescence)
    channel1 = np.array(img)
    img.seek(1)  # Second channel (standard image)
    channel2 = np.array(img)

    return channel1, channel2


def find_membrane_peaks(profile, min_distance=20):
    """
    Find the two membrane peaks in a fluorescence profile.

    The profile should show two peaks (membranes) with dark (low intensity) between them.
    """
    peaks, _ = find_peaks(profile, distance=min_distance, prominence=10)

    # Sort peaks by intensity and take the top 2
    if len(peaks) >= 2:
        peak_intensities = profile[peaks]
        top_peaks_idx = np.argsort(peak_intensities)[-2:]
        top_peaks = peaks[top_peaks_idx]
        # Sort by position (left to right)
        top_peaks = np.sort(top_peaks)
        return top_peaks
    elif len(peaks) == 1:
        return [peaks[0], peaks[0]]  # Fallback if only one peak found
    else:
        # If no peaks found, use max intensity
        max_pos = np.argmax(profile)
        return [max_pos, max_pos]


def main():
    """Main analysis and plotting function."""
    setup_plotting_style()

    # Load the TIFF file
    channel1, channel2 = load_tiff_data("experiment-1-slice.tif")

    print(f"Channel 1 shape: {channel1.shape}")
    print(f"Channel 2 shape: {channel2.shape}")
    print(f"Data type: {channel1.dtype}")

    # Cut off t=35 (last timepoint)
    n_timepoints_raw, n_positions = channel1.shape
    n_timepoints = n_timepoints_raw - 1  # Exclude last timepoint
    channel1 = channel1[:n_timepoints, :]

    print(f"Number of timepoints (excluding last): {n_timepoints}")
    print(f"Number of positions: {n_positions}")

    # Task 1: Plot fluorescence/distance lines for each time slice
    fig1, ax1 = plt.subplots(figsize=(8, 6))

    # Create a colormap for time progression
    colors = sns.color_palette("viridis", n_timepoints)

    for t in range(n_timepoints):
        ax1.plot(
            range(n_positions),
            channel1[t, :],
            color=colors[t],
            alpha=0.6,
            linewidth=1,
        )

    ax1.set_xlabel("Distance / pixels")
    ax1.set_ylabel("Fluorescence Intensity")
    ax1.set_title("Fluorescence Profile Over Time")

    # Add colorbar to show time progression
    sm = plt.cm.ScalarMappable(
        cmap="viridis",
        norm=Normalize(vmin=0, vmax=n_timepoints - 1),
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1)
    cbar.set_label("Time Point")

    plt.tight_layout()
    plt.savefig("fluorescence_profiles.svg")
    print("Saved fluorescence_profiles.svg")
    plt.close()

    # Task 2: Extract membrane intensity over time
    # The cross-section shows TWO membranes with dark interior between them
    membrane1_intensity = []
    membrane2_intensity = []
    membrane1_positions = []
    membrane2_positions = []

    for t in range(n_timepoints):
        profile = channel1[t, :].astype(float)  # Convert to float to avoid overflow
        # Find both membrane peaks
        peaks = find_membrane_peaks(profile)

        membrane1_positions.append(peaks[0])
        membrane2_positions.append(peaks[1])
        membrane1_intensity.append(profile[peaks[0]])
        membrane2_intensity.append(profile[peaks[1]])

    # Plot membrane intensity over time
    fig2, ax2 = plt.subplots(figsize=(8, 6))

    # Plot both membranes and their average
    palette = sns.color_palette("colorblind")

    ax2.plot(
        range(n_timepoints),
        membrane1_intensity,
        marker="o",
        markerfacecolor=palette[0],
        markeredgecolor="black",
        markeredgewidth=1,
        linewidth=2,
        color=palette[0],
        label="Membrane 1",
        markersize=4,
    )

    ax2.plot(
        range(n_timepoints),
        membrane2_intensity,
        marker="s",
        markerfacecolor=palette[1],
        markeredgecolor="black",
        markeredgewidth=1,
        linewidth=2,
        color=palette[1],
        label="Membrane 2",
        markersize=4,
    )

    # Add vertical dashed lines at t=1, t=15, and t=31
    snapshot_times = [1, 15, 31]
    for t in snapshot_times:
        if t < n_timepoints:
            ax2.axvline(x=t, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    ax2.set_xlabel("Time Point")
    ax2.set_ylabel("Membrane Fluorescence Intensity")
    ax2.set_title("Membrane Fluorescence Over Time")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("membrane_intensity_time.svg")
    print("Saved membrane_intensity_time.svg")
    plt.close()

    # Save PNG images of fluorescence profiles at specific timepoints
    snapshot_times = [1, 15, 31]
    for t in snapshot_times:
        if t < n_timepoints:
            fig_snap, ax_snap = plt.subplots(figsize=(8, 6))
            ax_snap.plot(
                range(n_positions),
                channel1[t, :],
                color=colors[t],
                linewidth=2,
            )
            ax_snap.set_xlabel("Distance / pixels")
            ax_snap.set_ylabel("Fluorescence Intensity")
            ax_snap.set_title(f"Fluorescence Profile at t={t}")
            plt.tight_layout()
            plt.savefig(f"fluorescence_profile_t{t}.png", dpi=300)
            print(f"Saved fluorescence_profile_t{t}.png")
            plt.close()

    # Diagnostic plot showing where both membranes were detected
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.plot(
        membrane1_positions,
        range(n_timepoints),
        marker="o",
        markersize=3,
        label="Membrane 1",
        color=palette[0],
    )
    ax3.plot(
        membrane2_positions,
        range(n_timepoints),
        marker="s",
        markersize=3,
        label="Membrane 2",
        color=palette[1],
    )
    ax3.set_xlabel("Membrane Position / pixels")
    ax3.set_ylabel("Time Point")
    ax3.set_title("Detected Membrane Positions Over Time")
    ax3.legend()
    ax3.invert_yaxis()
    plt.tight_layout()
    plt.savefig("membrane_position_diagnostic.svg")
    print("Saved membrane_position_diagnostic.svg")
    plt.close()

    print(
        f"\nMembrane 1 intensity range: {min(membrane1_intensity):.1f} - {max(membrane1_intensity):.1f}"
    )
    print(
        f"Membrane 2 intensity range: {min(membrane2_intensity):.1f} - {max(membrane2_intensity):.1f}"
    )

    # Create a combined figure with both analyses
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: fluorescence profiles
    for t in range(n_timepoints):
        ax4a.plot(
            range(n_positions),
            channel1[t, :],
            color=colors[t],
            alpha=0.6,
            linewidth=1,
        )

    ax4a.set_xlabel("Distance / pixels")
    ax4a.set_ylabel("Fluorescence Intensity")
    ax4a.set_title("Fluorescence Profile Over Time")

    # Right panel: membrane intensity over time (both membranes)
    ax4b.plot(
        range(n_timepoints),
        membrane1_intensity,
        marker="o",
        markerfacecolor=palette[0],
        markeredgecolor="black",
        markeredgewidth=1,
        linewidth=2,
        color=palette[0],
        label="Membrane 1",
        markersize=4,
    )

    ax4b.plot(
        range(n_timepoints),
        membrane2_intensity,
        marker="s",
        markerfacecolor=palette[1],
        markeredgecolor="black",
        markeredgewidth=1,
        linewidth=2,
        color=palette[1],
        label="Membrane 2",
        markersize=4,
    )

    ax4b.set_xlabel("Time Point")
    ax4b.set_ylabel("Membrane Fluorescence Intensity")
    ax4b.set_title("Membrane Fluorescence Over Time")
    ax4b.legend()

    plt.tight_layout()
    plt.savefig("combined_analysis.svg")
    print("Saved combined_analysis.svg")
    plt.close()


if __name__ == "__main__":
    main()
