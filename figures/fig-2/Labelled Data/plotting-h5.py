"""FFT power spectrum analysis of H5 rectified data with radial distance encoding."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import seaborn as sns
import h5py
from pathlib import Path
from scipy.interpolate import interp1d
from statsmodels.nonparametric.smoothers_lowess import lowess
from tqdm import tqdm
from typing import Any, cast
import numpy.typing as npt

# Set up plotting style
sns.set_palette("colorblind")
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["axes.grid"] = False

# Define dataset names (used for finding H5 files)
DATASET_NAMES = [
    "38bp-dense",
    "68bp-dense",
    "68bp-sparse",
    "star-dense",
]


def compute_power_spectrum(row_data, curve_length_px, pixel_scale_nm):
    """
    Compute power spectrum from intensity data along a curve.

    For real-valued signals, the FFT produces symmetric positive and negative
    frequencies. This function properly combines the power from both sides.

    Parameters
    ----------
    row_data : array-like
        Intensity values (angular samples)
    curve_length_px : float
        Total length of the curve in pixels
    pixel_scale_nm : float
        nm per pixel

    Returns
    -------
    spacing_nm : ndarray
        Spatial spacing in nm
    power : ndarray
        Power spectrum values (with negative frequency power incorporated)
    """
    # Convert to numpy array and ensure float type
    row_data = np.asarray(row_data, dtype=float)

    # Perform FFT and compute power spectrum
    fft_result = np.fft.fft(row_data)
    power_full = np.abs(fft_result)

    # Get frequencies
    n_samples = len(row_data)
    frequencies = np.fft.fftfreq(n_samples)

    # Convert to spacing in nm
    # Total curve length in nm
    curve_length_nm = curve_length_px * pixel_scale_nm
    # Spacing per sample
    spacing_per_sample = curve_length_nm / n_samples

    # Convert frequency to spacing
    # frequency = 1/spacing, so spacing = 1/frequency
    with np.errstate(divide="ignore", invalid="ignore"):
        spacing_nm_full = np.where(
            frequencies != 0, spacing_per_sample / np.abs(frequencies), np.inf
        )

    # For real-valued signals, combine power from positive and negative frequencies
    # Keep only positive frequencies for output
    positive_mask = frequencies > 0

    # Get positive frequency indices
    pos_indices = np.where(positive_mask)[0]

    # For each positive frequency, find the corresponding negative frequency
    # and sum their powers
    power_combined = power_full[pos_indices].copy()

    # Handle the symmetric negative frequencies
    # For frequency f[i] > 0, the corresponding negative frequency is at f[-i]
    # But we need to match by absolute frequency value
    for i, pos_idx in enumerate(pos_indices):
        pos_freq = frequencies[pos_idx]
        # Find the negative frequency with the same absolute value
        neg_idx = np.where(np.abs(frequencies + pos_freq) < 1e-10)[0]
        if len(neg_idx) > 0:
            power_combined[i] += power_full[neg_idx[0]]

    spacing_nm = spacing_nm_full[positive_mask]
    power = power_combined

    return spacing_nm, power


def find_latest_h5_file(data_dir, dataset_name):
    """
    Find the latest timestamped H5 file for a given dataset.

    Parameters
    ----------
    data_dir : Path
        Directory containing h5-files subdirectory
    dataset_name : str
        Name of the dataset (e.g., "38bp-dense")

    Returns
    -------
    h5_path : Path
        Path to the rectified_data.h5 file
    """
    h5_dir = data_dir / "h5-files"

    # Find all directories matching this dataset
    matching_dirs = sorted(h5_dir.glob(f"{dataset_name}_*"))

    if not matching_dirs:
        raise FileNotFoundError(f"No H5 directories found for {dataset_name}")

    # Get the latest (last when sorted)
    latest_dir = matching_dirs[-1]
    h5_path = latest_dir / "rectified_data.h5"

    if not h5_path.exists():
        raise FileNotFoundError(f"rectified_data.h5 not found in {latest_dir}")

    return h5_path


def process_h5_file(
    h5_path: Path, spacing_range: tuple[float, float] = (0.1, 20)
) -> tuple[Any, list[tuple[Any, Any]]]:
    """
    Process an H5 file and compute power spectra for each radial distance.

    Parameters
    ----------
    h5_path : Path
        Path to rectified_data.h5 file
    spacing_range : tuple
        (min, max) spacing range in nm to consider

    Returns
    -------
    radial_distances : ndarray
        Array of radial distances from membrane (nm)
    lowess_curves : list of tuples
        List of (spacing_fit, power_fit) for each radial distance
    """
    with h5py.File(h5_path, "r") as f:
        # Load data
        images: Any = f["images"][:]  # type: ignore[index]
        curve_lengths_px: Any = f["curve_lengths_px"][:]  # type: ignore[index]
        pixel_scale_nm = float(f.attrs["pixel_scale_nm"])  # type: ignore[arg-type]
        radial_min = float(f.attrs["radial_range_nm_min"])  # type: ignore[arg-type]
        radial_max = float(f.attrs["radial_range_nm_max"])  # type: ignore[arg-type]
        angular_samples = int(f.attrs["angular_samples"])  # type: ignore[arg-type]
        radial_samples = int(f.attrs["radial_samples"])  # type: ignore[arg-type]

        # Calculate radial distances
        radial_distances = np.linspace(radial_min, radial_max, radial_samples)

        print(f"  Images shape: {images.shape}")
        print(f"  Radial range: {radial_min:.2f} to {radial_max:.2f} nm")
        print(f"  Processing {radial_samples} radial distances...")

        # Process each radial distance
        lowess_curves = []

        for radial_idx in tqdm(
            range(radial_samples), desc="  Radial distances", leave=False
        ):
            # Extract angular profiles at this radial distance from all images
            # Shape: (N_images, angular_samples)
            angular_profiles = images[:, radial_idx, :]

            # Compute FFT for each image at this radial distance
            all_spacings = []
            all_powers = []

            for img_idx in range(images.shape[0]):
                angular_profile = angular_profiles[img_idx, :]

                # Normalize to mean=0, std=1
                angular_profile = (angular_profile - np.mean(angular_profile)) / np.std(
                    angular_profile
                )

                # Use the actual curve length for this image
                curve_length_px = curve_lengths_px[img_idx]

                # Compute power spectrum
                spacing_nm, power = compute_power_spectrum(
                    angular_profile, curve_length_px, pixel_scale_nm
                )

                all_spacings.append(spacing_nm)
                all_powers.append(power)

            # Combine all data points
            all_spacings_combined = np.concatenate(all_spacings)
            all_powers_combined = np.concatenate(all_powers)

            # Filter to spacing range
            mask = (all_spacings_combined >= spacing_range[0]) & (
                all_spacings_combined <= spacing_range[1]
            )
            spacings_filtered = all_spacings_combined[mask]
            powers_filtered = all_powers_combined[mask]

            # Sort by spacing for LOWESS
            sort_idx = np.argsort(spacings_filtered)
            spacings_sorted = spacings_filtered[sort_idx]
            powers_sorted = powers_filtered[sort_idx]

            # Apply LOWESS smoothing
            lowess_result = lowess(powers_sorted, spacings_sorted, frac=0.1)

            # Interpolate to a smooth grid
            spacing_fit = np.logspace(
                np.log10(spacing_range[0]), np.log10(spacing_range[1]), 200
            )
            lowess_interp = interp1d(
                lowess_result[:, 0],  # x values (spacings)
                lowess_result[:, 1],  # y values (powers)
                kind="linear",
                bounds_error=False,
                fill_value=cast(float, "extrapolate"),  # type: ignore[arg-type]
            )
            power_fit = lowess_interp(spacing_fit)

            lowess_curves.append((spacing_fit, power_fit))

    return radial_distances, lowess_curves


def main():
    """Main plotting function."""
    # Get data directory
    data_dir = Path(__file__).parent

    # Create figure with 4 subplots (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Use a colormap for radial distances
    cmap = cm.get_cmap("plasma")

    # Track y-axis limits to make them consistent
    all_y_max = 0

    # Process each dataset
    for idx, dataset_name in enumerate(DATASET_NAMES):
        print(f"\nProcessing {dataset_name}...")

        ax = axes[idx]

        # Find and load H5 file
        try:
            h5_path = find_latest_h5_file(data_dir, dataset_name)
            print(f"  Loading: {h5_path}")
        except FileNotFoundError as e:
            print(f"  Error: {e}")
            ax.text(
                0.5,
                0.5,
                f"No data\n{dataset_name}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        # Process the H5 file
        radial_distances, lowess_curves = process_h5_file(h5_path)

        # Plot each radial distance curve with color from colormap
        for radial_idx, (spacing_fit, power_fit) in enumerate(lowess_curves):
            # Normalize radial index to [0, 1] for colormap
            color_val = (
                radial_idx / (len(lowess_curves) - 1) if len(lowess_curves) > 1 else 0
            )
            color = cmap(color_val)

            # Plot LOWESS curve
            ax.plot(
                spacing_fit,
                power_fit,
                color=color,
                linewidth=1.5,
                alpha=0.8,
            )

            # Track max y value
            all_y_max = max(all_y_max, np.nanmax(power_fit))

        # Add colorbar for this subplot
        sm = cm.ScalarMappable(
            cmap=cmap,
            norm=Normalize(vmin=radial_distances[0], vmax=radial_distances[-1]),
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label="Distance from membrane (nm)")

        # Labels and formatting
        ax.set_xlabel("Spacing (nm)", fontsize=10)
        ax.set_ylabel("Amplitude", fontsize=10)
        ax.set_xlim(0.1, 20)
        ax.set_title(dataset_name, fontsize=11, fontweight="bold")

    # Set consistent y-axis limits across all subplots
    for ax in axes:
        ax.set_ylim(0, all_y_max * 1.05)

    plt.tight_layout()

    # Save figure
    output_path = data_dir / "power_spectrum_h5_analysis.svg"
    plt.savefig(output_path, format="svg", dpi=300, bbox_inches="tight")
    print(f"\nSaved plot to {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
