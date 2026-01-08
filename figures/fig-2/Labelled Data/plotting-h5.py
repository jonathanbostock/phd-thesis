"""FFT power spectrum analysis of H5 rectified data with radial distance encoding."""

import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from pathlib import Path
from scipy.interpolate import interp1d
from statsmodels.nonparametric.smoothers_lowess import lowess
from tqdm import tqdm
from typing import Any, cast
from joblib import Parallel, delayed

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

    power_full = power_full / np.mean(power_full)

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
) -> tuple[Any, list[tuple[Any, Any]], list[tuple[Any, Any]]]:
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
    raw_data_list : list of tuples
        List of (spacings_filtered, powers_filtered) raw data for each radial distance
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
        raw_data_list = []

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

            # Store raw data for scatter plotting
            raw_data_list.append((spacings_sorted, powers_sorted))

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

    return radial_distances, lowess_curves, raw_data_list


def process_h5_file_with_particles(
    h5_path: Path, spacing_range: tuple[float, float] = (0.1, 20)
) -> tuple[Any, list[tuple[Any, Any]], list[list[tuple[Any, Any]]]]:
    """
    Process an H5 file preserving per-particle data for bootstrap analysis.

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
    per_particle_data : list of lists of tuples
        Outer list: radial distances (length = radial_samples)
        Inner list: particles (length = N_particles)
        Tuple: (spacings_sorted, powers_sorted) for one particle at one radial distance
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

        n_particles = images.shape[0]

        # Initialize per_particle_data structure
        per_particle_data: list[list[tuple[Any, Any]]] = [
            [] for _ in range(radial_samples)
        ]

        # Process each radial distance
        lowess_curves = []

        for radial_idx in tqdm(
            range(radial_samples), desc="  Radial distances", leave=False
        ):
            # For combined LOWESS (same as before)
            all_spacings_combined = []
            all_powers_combined = []

            # Process each particle separately
            for particle_idx in range(n_particles):
                angular_profile = images[particle_idx, radial_idx, :]

                # Normalize to mean=0, std=1
                angular_profile = (angular_profile - np.mean(angular_profile)) / np.std(
                    angular_profile
                )

                # Use the actual curve length for this particle
                curve_length_px = curve_lengths_px[particle_idx]

                # Compute power spectrum
                spacing_nm, power = compute_power_spectrum(
                    angular_profile, curve_length_px, pixel_scale_nm
                )

                # Filter to spacing range
                mask = (spacing_nm >= spacing_range[0]) & (
                    spacing_nm <= spacing_range[1]
                )
                spacings_filtered = spacing_nm[mask]
                powers_filtered = power[mask]

                # Sort by spacing
                sort_idx = np.argsort(spacings_filtered)
                spacings_sorted = spacings_filtered[sort_idx]
                powers_sorted = powers_filtered[sort_idx]

                # Store per-particle data
                per_particle_data[radial_idx].append((spacings_sorted, powers_sorted))

                # Also add to combined data for LOWESS
                all_spacings_combined.append(spacings_sorted)
                all_powers_combined.append(powers_sorted)

            # Combine all particles for this radial distance
            combined_spacings = np.concatenate(all_spacings_combined)
            combined_powers = np.concatenate(all_powers_combined)

            # Sort by spacing
            sort_idx = np.argsort(combined_spacings)
            spacings_sorted_all = combined_spacings[sort_idx]
            powers_sorted_all = combined_powers[sort_idx]

            # Apply LOWESS smoothing
            lowess_result = lowess(powers_sorted_all, spacings_sorted_all, frac=0.1)

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

    return radial_distances, lowess_curves, per_particle_data


def _single_bootstrap_power_spectrum(
    boot_idx: int,
    per_particle_data: list[list[tuple[Any, Any]]],
    selected_radial_indices: Any,
    n_particles: int,
    spacing_fit: Any,
) -> Any:
    """Single bootstrap iteration for power spectrum (for parallel execution)."""
    # Set random seed for reproducibility
    np.random.seed(boot_idx)

    # Resample particles with replacement
    resampled_particle_indices = np.random.choice(
        n_particles, size=n_particles, replace=True
    )

    # Collect data from resampled particles across selected radial distances
    all_spacings = []
    all_powers = []
    for radial_idx in selected_radial_indices:
        for particle_idx in resampled_particle_indices:
            spacings, powers = per_particle_data[radial_idx][particle_idx]
            all_spacings.append(spacings)
            all_powers.append(powers)

    # Combine and sort
    combined_spacings = np.concatenate(all_spacings)
    combined_powers = np.concatenate(all_powers)
    sort_idx = np.argsort(combined_spacings)

    # Apply LOWESS
    lowess_result = lowess(
        combined_powers[sort_idx], combined_spacings[sort_idx], frac=0.1
    )

    # Interpolate to spacing_fit grid
    lowess_interp = interp1d(
        lowess_result[:, 0],
        lowess_result[:, 1],
        kind="linear",
        bounds_error=False,
        fill_value=cast(float, "extrapolate"),  # type: ignore[arg-type]
    )
    return lowess_interp(spacing_fit)


def bootstrap_lowess_power_spectrum(
    per_particle_data: list[list[tuple[Any, Any]]],
    radial_distances: Any,
    radial_band_min: float,
    radial_band_max: float,
    spacing_fit: Any,
    n_bootstrap: int = 100,
) -> tuple[Any, Any]:
    """
    Bootstrap over particles to estimate SE of LOWESS power spectrum (parallelized).

    Parameters
    ----------
    per_particle_data : list of lists of tuples
        Per-particle data structure from process_h5_file_with_particles
    radial_distances : ndarray
        Array of radial distances
    radial_band_min : float
        Minimum radial distance for band
    radial_band_max : float
        Maximum radial distance for band
    spacing_fit : ndarray
        Grid of spacing values for interpolation
    n_bootstrap : int
        Number of bootstrap iterations (default: 100)

    Returns
    -------
    mean_curve : ndarray
        Mean LOWESS curve over bootstrap samples
    se_curve : ndarray
        Standard error at each spacing point
    """
    # Find radial indices in band
    mask = (radial_distances >= radial_band_min) & (radial_distances <= radial_band_max)
    selected_radial_indices = np.where(mask)[0]

    # Get number of particles
    n_particles = len(per_particle_data[selected_radial_indices[0]])

    # Run bootstrap iterations in parallel
    print(f"    Running {n_bootstrap} bootstrap iterations in parallel...")
    bootstrap_curves = Parallel(n_jobs=-1)(
        delayed(_single_bootstrap_power_spectrum)(
            boot_idx,
            per_particle_data,
            selected_radial_indices,
            n_particles,
            spacing_fit,
        )
        for boot_idx in tqdm(range(n_bootstrap), desc="    Bootstrap", leave=False)
    )

    # Convert to array
    bootstrap_curves = np.array(bootstrap_curves)

    # Compute mean and SE across bootstrap samples
    mean_curve = np.mean(bootstrap_curves, axis=0)
    se_curve = np.std(bootstrap_curves, axis=0, ddof=1)

    return mean_curve, se_curve


def _single_bootstrap_at_spacing(
    boot_idx: int,
    per_particle_data: list[list[tuple[Any, Any]]],
    n_particles: int,
    n_radial: int,
    target_spacing: float,
) -> Any:
    """Single bootstrap iteration for spacing analysis (for parallel execution)."""
    # Set random seed for reproducibility
    np.random.seed(boot_idx)

    # Resample particles
    resampled_indices = np.random.choice(n_particles, size=n_particles, replace=True)

    # Storage for this iteration
    radial_values = np.zeros(n_radial)

    # For each radial distance
    for radial_idx in range(n_radial):
        # Collect data from resampled particles
        all_spacings = []
        all_powers = []
        for particle_idx in resampled_indices:
            spacings, powers = per_particle_data[radial_idx][particle_idx]
            all_spacings.append(spacings)
            all_powers.append(powers)

        # Combine, sort, LOWESS
        combined_spacings = np.concatenate(all_spacings)
        combined_powers = np.concatenate(all_powers)
        sort_idx = np.argsort(combined_spacings)
        lowess_result = lowess(
            combined_powers[sort_idx], combined_spacings[sort_idx], frac=0.1
        )

        # Interpolate and extract value at target_spacing
        lowess_interp = interp1d(
            lowess_result[:, 0],
            lowess_result[:, 1],
            kind="linear",
            bounds_error=False,
            fill_value=cast(float, "extrapolate"),  # type: ignore[arg-type]
        )
        radial_values[radial_idx] = lowess_interp(target_spacing)

    return radial_values


def bootstrap_lowess_at_spacing(
    per_particle_data: list[list[tuple[Any, Any]]],
    radial_distances: Any,
    target_spacing: float,
    n_bootstrap: int = 100,
) -> tuple[Any, Any]:
    """
    Bootstrap to estimate SE of LOWESS values at target spacing (parallelized).

    Parameters
    ----------
    per_particle_data : list of lists of tuples
        Per-particle data structure from process_h5_file_with_particles
    radial_distances : ndarray
        Array of radial distances
    target_spacing : float
        Spacing value to extract from LOWESS curves
    n_bootstrap : int
        Number of bootstrap iterations (default: 100)

    Returns
    -------
    mean_values : ndarray
        Mean LOWESS values at each x_coord (interpolated)
    se_values : ndarray
        Standard error at each x_coord (interpolated)
    """
    n_particles = len(per_particle_data[0])
    n_radial = len(radial_distances)

    # Run bootstrap iterations in parallel
    print(f"    Running {n_bootstrap} bootstrap iterations in parallel...")
    bootstrap_values = Parallel(n_jobs=-1)(
        delayed(_single_bootstrap_at_spacing)(
            boot_idx, per_particle_data, n_particles, n_radial, target_spacing
        )
        for boot_idx in tqdm(range(n_bootstrap), desc="    Bootstrap", leave=False)
    )

    # Convert to array: (n_bootstrap, n_radial)
    bootstrap_values = np.array(bootstrap_values)

    # Compute mean and SE at each radial distance
    mean_at_radial = np.mean(bootstrap_values, axis=0)
    se_at_radial = np.std(bootstrap_values, axis=0, ddof=1)

    # Interpolate to custom x_coords [0.5, 1.5, ..., 24.5]
    x_coords = np.arange(0.5, 25.0, 1.0)
    interp_mean = interp1d(
        radial_distances,
        mean_at_radial,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",  # type: ignore[arg-type]
    )
    interp_se = interp1d(
        radial_distances,
        se_at_radial,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",  # type: ignore[arg-type]
    )

    return interp_mean(x_coords), interp_se(x_coords)


def plot_combined_spectrum_and_amplitude(
    all_dataset_raw_data: dict[str, tuple[Any, list[tuple[Any, Any]]]],
    all_dataset_curves: dict[str, tuple[Any, list[tuple[Any, Any]]]],
    all_dataset_per_particle_data: dict[str, tuple[Any, list[list[tuple[Any, Any]]]]],
    output_path: Path,
    radial_band_min: float = 2.0,
    radial_band_max: float = 5.0,
    peak_search_min: float = 2.0,
    peak_search_max: float = 7.0,
    spacing_range: tuple[float, float] = (0.5, 10.0),
    n_bootstrap: int = 100,
) -> None:
    """
    Create combined plot with power spectrum (left) and amplitude at peak spacing (right).

    Parameters
    ----------
    all_dataset_raw_data : dict
        Dictionary mapping dataset names to (radial_distances, raw_data_list) tuples
    all_dataset_curves : dict
        Dictionary mapping dataset names to (radial_distances, lowess_curves) tuples
    all_dataset_per_particle_data : dict
        Dictionary mapping dataset names to (radial_distances, per_particle_data) tuples
    output_path : Path
        Path to save the output SVG file
    radial_band_min : float
        Minimum radial distance in nm (default: 2.0)
    radial_band_max : float
        Maximum radial distance in nm (default: 5.0)
    peak_search_min : float
        Minimum spacing to search for peak (default: 2.0)
    peak_search_max : float
        Maximum spacing to search for peak (default: 7.0)
    spacing_range : tuple
        (min, max) spacing range in nm for plotting (default: 0.5 to 10.0)
    n_bootstrap : int
        Number of bootstrap iterations (default: 100)
    """
    # Create figure with 1x2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    colors = sns.color_palette("colorblind", n_colors=4)

    # Store peak spacings for each dataset
    peak_spacings = {}

    # LEFT SUBPLOT: Power spectrum
    for i, (dataset_name, (radial_distances, raw_data_list)) in enumerate(
        all_dataset_raw_data.items()
    ):
        # Find indices of radial distances in the band
        mask = (radial_distances >= radial_band_min) & (
            radial_distances <= radial_band_max
        )
        selected_indices = np.where(mask)[0]

        # Collect all raw data from selected radial distances
        all_spacings = []
        all_powers = []

        for idx in selected_indices:
            spacings_sorted, powers_sorted = raw_data_list[idx]
            all_spacings.append(spacings_sorted)
            all_powers.append(powers_sorted)

        # Combine all data
        combined_spacings = np.concatenate(all_spacings)
        combined_powers = np.concatenate(all_powers)

        # Sort by spacing
        sort_idx = np.argsort(combined_spacings)
        combined_spacings = combined_spacings[sort_idx]
        combined_powers = combined_powers[sort_idx]

        # Apply LOWESS smoothing
        lowess_result = lowess(combined_powers, combined_spacings, frac=0.1)

        # Interpolate to smooth grid
        spacing_fit = np.logspace(
            np.log10(spacing_range[0]), np.log10(spacing_range[1]), 200
        )
        lowess_interp = interp1d(
            lowess_result[:, 0],
            lowess_result[:, 1],
            kind="linear",
            bounds_error=False,
            fill_value=cast(float, "extrapolate"),  # type: ignore[arg-type]
        )
        power_fit = lowess_interp(spacing_fit)

        # Find peak in specified range
        peak_mask = (spacing_fit >= peak_search_min) & (spacing_fit <= peak_search_max)
        peak_idx = np.argmax(power_fit[peak_mask])
        peak_spacing = spacing_fit[peak_mask][peak_idx]
        peak_power = power_fit[peak_mask][peak_idx]

        # Store peak spacing for right subplot
        peak_spacings[dataset_name] = peak_spacing

        # Get bootstrap estimates
        _, per_particle_data = all_dataset_per_particle_data[dataset_name]
        mean_curve, se_curve = bootstrap_lowess_power_spectrum(
            per_particle_data,
            radial_distances,
            radial_band_min,
            radial_band_max,
            spacing_fit,
            n_bootstrap,
        )

        # Plot error band first (so it's behind the line)
        ax1.fill_between(
            spacing_fit,
            mean_curve - se_curve,
            mean_curve + se_curve,
            alpha=0.3,
            color=colors[i],
            linewidth=0,
        )

        # Plot main curve
        ax1.plot(
            spacing_fit,
            power_fit,
            color=colors[i],
            linewidth=2,
            label=f"{dataset_name} ({peak_spacing:.1f} nm)",
            alpha=0.8,
        )

        # Add marker at peak
        ax1.plot(
            peak_spacing,
            peak_power,
            marker="*",
            markersize=10,
            color=colors[i],
            markeredgecolor="black",
            markeredgewidth=0.5,
        )

    # Format left subplot
    ax1.set_xlabel("Spacing (nm)", fontsize=12)
    ax1.set_ylabel("Amplitude", fontsize=12)
    ax1.set_xlim(spacing_range[0], spacing_range[1])
    ax1.set_title("Average Power Spectrum", fontsize=14)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(False)

    # RIGHT SUBPLOT: Amplitude at peak spacing
    x_coords = np.arange(0.5, 25.0, 1.0)  # [0.5, 1.5, 2.5, ..., 24.5]

    for i, (dataset_name, (radial_distances, lowess_curves)) in enumerate(
        all_dataset_curves.items()
    ):
        # Get the peak spacing for this dataset
        peak_spacing = peak_spacings[dataset_name]

        # Extract values at peak spacing for each radial distance
        values_at_spacing = []

        for spacing_fit, power_fit in lowess_curves:
            # Find the index closest to peak spacing
            idx = np.argmin(np.abs(spacing_fit - peak_spacing))
            values_at_spacing.append(power_fit[idx])

        values_at_spacing = np.array(values_at_spacing)

        # Interpolate to custom x-coordinates
        interp_func = interp1d(
            radial_distances,
            values_at_spacing,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",  # type: ignore
        )
        values_interpolated = interp_func(x_coords)

        # Get bootstrap estimates
        _, per_particle_data = all_dataset_per_particle_data[dataset_name]
        mean_values, se_values = bootstrap_lowess_at_spacing(
            per_particle_data, radial_distances, peak_spacing, n_bootstrap
        )

        # Plot error band first (so it's behind the line)
        ax2.fill_between(
            x_coords,
            mean_values - se_values,
            mean_values + se_values,
            alpha=0.3,
            color=colors[i],
            linewidth=0,
        )

        # Plot main curve (no markers)
        ax2.plot(
            x_coords,
            values_interpolated,
            color=colors[i],
            linewidth=1.5,
            label=f"{dataset_name} ({peak_spacing:.1f} nm)",
            alpha=0.8,
        )

    # Format right subplot
    ax2.set_xlabel("Radial distance from membrane (nm)", fontsize=12)
    ax2.set_xlim(math.floor(x_coords.min()), math.ceil(x_coords.max()))
    ax2.set_title("Amplitude At Peak Spacing", fontsize=14)
    ax2.legend(frameon=False, fontsize=10, loc="upper right")
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.grid(False)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")

    # Share y-axis between subplots
    ax1.get_shared_y_axes().joined(ax1, ax2)
    ax2.set_ylim(ax1.get_ylim())

    plt.tight_layout()
    plt.savefig(output_path, format="svg", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    """Main plotting function."""
    # Get data directory
    data_dir = Path(__file__).parent

    # Dictionaries to collect data for all datasets
    all_dataset_curves = {}
    all_dataset_raw_data = {}
    all_dataset_per_particle_data = {}

    # Process each dataset
    for dataset_name in DATASET_NAMES:
        print(f"\nProcessing {dataset_name}...")

        # Find and load H5 file
        try:
            h5_path = find_latest_h5_file(data_dir, dataset_name)
            print(f"  Loading: {h5_path}")
        except FileNotFoundError as e:
            print(f"  Error: {e}")
            continue

        # Process the H5 file with per-particle data
        (
            radial_distances,
            lowess_curves,
            per_particle_data,
        ) = process_h5_file_with_particles(h5_path)

        # Store for combined plots
        all_dataset_curves[dataset_name] = (radial_distances, lowess_curves)
        all_dataset_per_particle_data[dataset_name] = (
            radial_distances,
            per_particle_data,
        )

        # Also need raw_data for the old interface (flatten per-particle to combined)
        # This is just for backwards compatibility with plot_combined_power_spectra
        raw_data_list = []
        for radial_idx in range(len(per_particle_data)):
            # Combine all particles at this radial distance
            all_spacings = []
            all_powers = []
            for spacings, powers in per_particle_data[radial_idx]:
                all_spacings.append(spacings)
                all_powers.append(powers)
            combined_spacings = np.concatenate(all_spacings)
            combined_powers = np.concatenate(all_powers)
            sort_idx = np.argsort(combined_spacings)
            raw_data_list.append(
                (combined_spacings[sort_idx], combined_powers[sort_idx])
            )
        all_dataset_raw_data[dataset_name] = (radial_distances, raw_data_list)

    # Create combined spectrum and amplitude plot
    if all_dataset_raw_data and all_dataset_curves and all_dataset_per_particle_data:
        print("\nCreating combined spectrum and amplitude plot...")
        combined_output = data_dir / "combined_spectrum_amplitude.svg"
        plot_combined_spectrum_and_amplitude(
            all_dataset_raw_data,
            all_dataset_curves,
            all_dataset_per_particle_data,
            combined_output,
            radial_band_min=2.0,
            radial_band_max=5.0,
            peak_search_min=2.0,
            peak_search_max=7.0,
        )
        print(f"  Saved combined plot to {combined_output}")


if __name__ == "__main__":
    main()
