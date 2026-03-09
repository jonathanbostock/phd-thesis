"""FFT power spectrum analysis of labeled DNA brush data."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.interpolate import interp1d
from statsmodels.nonparametric.smoothers_lowess import lowess
from tqdm import tqdm
from typing import cast

# Set up plotting style
sns.set_palette("colorblind")
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["axes.grid"] = False

# Define markers for each dataset
MARKERS = ["o", "s", "^", "D"]
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
        Intensity values (512 samples)
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

    # Normalize
    power = power  # / np.mean(power, keepdims=True)

    return spacing_nm, power


def process_csv_file(csv_path):
    """
    Process a CSV file and compute power spectra for all rows.

    Parameters
    ----------
    csv_path : Path
        Path to CSV file

    Returns
    -------
    all_spacings : list of ndarray
        Spacing values for each row
    all_powers : list of ndarray
        Power values for each row
    """
    # Read CSV, skipping comment lines
    df = pd.read_csv(csv_path, comment="#")

    # Extract intensity columns (s0 through s511)
    intensity_cols = [f"s{i}" for i in range(512)]

    all_spacings = []
    all_powers = []

    for _, row in df.iterrows():
        curve_length_px = float(row["curve_length_px"])  # type: ignore[arg-type]
        pixel_scale_nm = float(row["pixel_scale_nm"])  # type: ignore[arg-type]
        intensities = np.array([row[col] for col in intensity_cols], dtype=float)

        spacing_nm, power = compute_power_spectrum(
            intensities, curve_length_px, pixel_scale_nm
        )

        all_spacings.append(spacing_nm)
        all_powers.append(power)

    return all_spacings, all_powers


def fit_loess_curve(
    spacings_list, powers_list, spacing_range=(0.1, 20), n_bootstrap=10
):
    """
    Fit a LOESS curve to the power spectrum data with bootstrap error estimation.

    Parameters
    ----------
    spacings_list : list of ndarray
        Spacing values from multiple rows
    powers_list : list of ndarray
        Power values from multiple rows
    spacing_range : tuple
        (min, max) spacing range in nm to consider
    n_bootstrap : int
        Number of bootstrap samples to use for error estimation

    Returns
    -------
    spacing_fit : ndarray
        Spacing values for the fitted curve
    power_fit : ndarray
        Fitted power values
    power_sem : ndarray
        Standard error of the mean for fitted power values
    """
    # Generate smooth curve points for plotting
    spacing_fit = np.logspace(
        np.log10(spacing_range[0]), np.log10(spacing_range[1]), 200
    )

    # Store bootstrap results
    bootstrap_fits = np.zeros((n_bootstrap, len(spacing_fit)))

    # Number of rows (samples)
    n_rows = len(spacings_list)

    # Perform bootstrap resampling
    rng = np.random.default_rng(42)  # Set seed for reproducibility

    for boot_idx in tqdm(range(n_bootstrap), desc="Bootstrap", leave=False):
        # Resample whole rows with replacement
        sample_indices = rng.choice(n_rows, size=n_rows, replace=True)
        boot_spacings = [spacings_list[i] for i in sample_indices]
        boot_powers = [powers_list[i] for i in sample_indices]

        # Combine all data points from this bootstrap sample
        all_spacings = np.concatenate(boot_spacings)
        all_powers = np.concatenate(boot_powers)

        # Filter to spacing range
        mask = (all_spacings >= spacing_range[0]) & (all_spacings <= spacing_range[1])
        spacings_filtered = all_spacings[mask]
        powers_filtered = all_powers[mask]

        # Sort by spacing for LOESS
        sort_idx = np.argsort(spacings_filtered)
        spacings_sorted = spacings_filtered[sort_idx]
        powers_sorted = powers_filtered[sort_idx]

        # Apply LOWESS smoothing
        lowess_result = lowess(powers_sorted, spacings_sorted, frac=0.1)

        # Interpolate LOWESS result to common spacing grid
        lowess_interp = interp1d(
            lowess_result[:, 0],  # x values (spacings)
            lowess_result[:, 1],  # y values (powers)
            kind="linear",
            bounds_error=False,
            fill_value=cast(float, "extrapolate"),  # type: ignore[arg-type]
        )

        bootstrap_fits[boot_idx, :] = lowess_interp(spacing_fit)

    # Calculate mean and SEM across bootstrap samples
    power_fit = np.mean(bootstrap_fits, axis=0)
    power_sem = np.std(bootstrap_fits, axis=0, ddof=1)

    return spacing_fit, power_fit, power_sem


def main():
    """Main plotting function."""
    # Get all CSV files in the current directory
    data_dir = Path(__file__).parent
    csv_files = sorted(data_dir.glob("*-labels.csv"))

    if len(csv_files) != 4:
        print(f"Warning: Expected 4 CSV files, found {len(csv_files)}")

    # Create figure with 4 subplots (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Process each CSV file
    colors = sns.color_palette("colorblind", len(csv_files))

    # Track y-axis limits to make them consistent
    all_y_max = 0

    for idx, csv_path in enumerate(csv_files):
        print(f"Processing {csv_path.name}...")

        ax = axes[idx]

        # Get dataset name from filename
        dataset_name = csv_path.stem.replace("-labels", "")

        # Process the CSV
        spacings_list, powers_list = process_csv_file(csv_path)

        # Filter and flatten data for plotting
        spacing_range = (0.1, 20)
        plot_spacings = []
        plot_powers = []

        for spacings, powers in zip(spacings_list, powers_list):
            mask = (spacings >= spacing_range[0]) & (spacings <= spacing_range[1])
            plot_spacings.extend(spacings[mask])
            plot_powers.extend(powers[mask])

        # Plot scatter of raw data
        ax.scatter(
            plot_spacings,
            plot_powers,
            alpha=0.3,
            s=10,
            color=colors[idx],
            marker=MARKERS[idx],
            edgecolors="black",
            linewidths=0.5,
            label="Raw data",
        )

        # Fit and plot LOESS curve with bootstrap error
        try:
            spacing_fit, power_fit, power_sem = fit_loess_curve(
                spacings_list, powers_list, spacing_range
            )

            # Plot LOESS fit
            ax.plot(
                spacing_fit,
                power_fit,
                color=colors[idx],
                linewidth=2,
                label="LOESS fit",
            )

            # Plot +/- SEM as shaded region
            ax.fill_between(
                spacing_fit,
                power_fit - power_sem,
                power_fit + power_sem,
                color=colors[idx],
                alpha=0.2,
                label="±SEM",
            )

            # Track max y value (including error bars)
            all_y_max = max(all_y_max, np.max(power_fit + power_sem))
        except Exception as e:
            print(f"Warning: Could not fit LOESS for {dataset_name}: {e}")

        # Track max y value from raw data
        if plot_powers:
            all_y_max = max(all_y_max, np.max(plot_powers))

        # Labels and formatting
        ax.set_xlabel("Spacing (nm)", fontsize=10)
        ax.set_ylabel("Amplitude", fontsize=10)
        ax.set_xlim(0.1, 20)
        ax.set_title(dataset_name, fontsize=11, fontweight="bold")
        ax.legend(frameon=False, fontsize=8)

    # Set consistent y-axis limits across all subplots
    for ax in axes:
        ax.set_ylim(0, all_y_max * 1.05)

    plt.tight_layout()

    # Save figure
    output_path = data_dir / "power_spectrum_analysis.svg"
    plt.savefig(output_path, format="svg", dpi=300, bbox_inches="tight")
    print(f"\nSaved plot to {output_path}")


if __name__ == "__main__":
    main()
