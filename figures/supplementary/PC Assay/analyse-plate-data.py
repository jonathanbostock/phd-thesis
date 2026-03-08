"""
PC Assay Analysis Script

This script processes phosphatidylcholine (PC) quantification assay data from 96-well
plate readers. It recursively finds assay.csv files, performs linear calibration using
known standards, predicts unknown concentrations, and generates publication-quality plots.

Usage:
    uv run python analyse-plate-data.py
"""

from pathlib import Path
from typing import List, Dict, NamedTuple
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from utils.plotting import setup_plot_style, format_axes, save_plot


# Known calibration concentrations (mM) for first 8 wells of each row
CALIBRATION_CONCENTRATIONS = np.array([0, 0.1, 0.15, 0.2, 0.3, 0.35, 0.4, 0.5])


class CalibrationResult(NamedTuple):
    """Results from linear calibration of a single row"""

    slope: float
    intercept: float
    stderr: float
    rvalue: float
    pvalue: float
    concentrations: np.ndarray
    y_values: np.ndarray


def find_all_assay_files(base_dir: Path) -> List[Path]:
    """
    Find all assay.csv files recursively in subdirectories.

    Parameters
    ----------
    base_dir : Path
        Base directory to search from

    Returns
    -------
    List[Path]
        Sorted list of paths to assay.csv files
    """
    return sorted(base_dir.rglob("**/assay.csv"))


def load_and_parse_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load and parse assay CSV file.

    Skips first 2 header rows, extracts row letter from well ID,
    and calculates absorbance difference (column 4 - column 3).

    Parameters
    ----------
    csv_path : Path
        Path to assay.csv file

    Returns
    -------
    pd.DataFrame
        Processed DataFrame with columns: Well, row_letter, difference
    """
    # Read CSV - use first row as header, then skip the second row
    df = pd.read_csv(csv_path, skiprows=[1])

    # Extract row letter (A, B, C, etc.) from well ID
    df["row_letter"] = df["Well"].str[0]

    # Calculate absorbance difference: column 4 - column 3
    # Use positional indexing to avoid column name dependencies
    col3 = pd.to_numeric(df.iloc[:, 3], errors="coerce")
    col2 = pd.to_numeric(df.iloc[:, 2], errors="coerce")
    df["difference"] = col3 - col2  # type: ignore[operator]

    result = df[["Well", "row_letter", "difference"]].dropna()
    return result  # type: ignore[return-value]


def calibrate_row(row_data: pd.DataFrame) -> CalibrationResult:
    """
    Perform inverse linear calibration using first 8 wells with known concentrations.

    Uses inverse regression where concentration is the dependent variable (Y)
    and absorbance difference is the independent variable (X).

    Parameters
    ----------
    row_data : pd.DataFrame
        Data for a single row (sorted by well number)

    Returns
    -------
    CalibrationResult
        Calibration parameters and data
    """
    # Extract first 8 wells for calibration
    calib_data = row_data.iloc[:8]
    y_values = calib_data["difference"].values

    # Perform inverse regression: concentration = slope * absorbance + intercept
    # This gives us direct prediction of concentration from absorbance with proper error propagation
    result = stats.linregress(y_values, CALIBRATION_CONCENTRATIONS)

    return CalibrationResult(
        slope=result.slope,  # type: ignore[attr-defined]
        intercept=result.intercept,  # type: ignore[attr-defined]
        stderr=result.stderr,  # type: ignore[attr-defined]
        rvalue=result.rvalue,  # type: ignore[attr-defined]
        pvalue=result.pvalue,  # type: ignore[attr-defined]
        concentrations=CALIBRATION_CONCENTRATIONS,
        y_values=y_values,
    )


def predict_samples(row_data: pd.DataFrame, calib: CalibrationResult) -> pd.DataFrame:
    """
    Predict concentrations for sample wells (wells 9+) using inverse calibration curve.

    Parameters
    ----------
    row_data : pd.DataFrame
        Data for a single row
    calib : CalibrationResult
        Calibration parameters from inverse regression (concentration vs absorbance)

    Returns
    -------
    pd.DataFrame
        Predictions with columns: Well, Concentration / mM, Uncertainty / mM
    """
    # Extract sample data (wells 9+)
    sample_data = row_data.iloc[8:]

    if len(sample_data) == 0:
        return pd.DataFrame(
            columns=["Well", "Concentration / mM", "Uncertainty / mM"]  # type: ignore[arg-type]
        )

    y_samples = sample_data["difference"].values

    # Direct prediction using inverse regression: concentration = slope * absorbance + intercept
    pred_concentrations = calib.slope * y_samples + calib.intercept

    # Calculate prediction uncertainty
    n_calib = len(calib.y_values)
    x_mean = np.mean(calib.y_values)  # Mean of absorbance values

    # Calculate mean squared error from calibration residuals
    # For inverse regression: concentration = slope * absorbance + intercept
    conc_pred_calib = calib.slope * calib.y_values + calib.intercept
    rss = np.sum((calib.concentrations - conc_pred_calib) ** 2)
    mse = rss / (n_calib - 2)

    # Calculate uncertainty for each sample point (standard error of prediction)
    uncertainties = []
    for y_sample in y_samples:
        se_pred = np.sqrt(
            mse
            * (
                1
                + 1 / n_calib
                + (y_sample - x_mean) ** 2 / np.sum((calib.y_values - x_mean) ** 2)
            )
        )
        uncertainties.append(se_pred)

    return pd.DataFrame(
        {
            "Well": sample_data["Well"].values,
            "Concentration / mM": pred_concentrations,
            "Uncertainty / mM": uncertainties,
            "Absorbance": y_samples,  # Store absorbance for plotting
        }
    )


def validate_calibration(calib: CalibrationResult, row_letter: str) -> List[str]:
    """
    Validate calibration quality and return warning messages.

    Parameters
    ----------
    calib : CalibrationResult
        Calibration results to validate
    row_letter : str
        Row identifier for warning messages

    Returns
    -------
    List[str]
        List of warning messages (empty if no issues)
    """
    warnings = []

    r_squared = calib.rvalue**2
    if r_squared < 0.95:
        warnings.append(f"  Row {row_letter}: Low R² = {r_squared:.3f}")

    if calib.pvalue > 0.01:
        warnings.append(f"  Row {row_letter}: Poor significance p = {calib.pvalue:.3e}")

    if calib.slope <= 0:
        warnings.append(f"  Row {row_letter}: Negative or zero slope - check data")

    return warnings


def plot_calibration_curves(results_by_row: Dict, output_path: Path):
    """
    Generate publication-quality plots showing calibration curves and sample points.

    Parameters
    ----------
    results_by_row : Dict
        Dictionary mapping row letters to calibration and sample data
    output_path : Path
        Directory where plot will be saved
    """
    setup_plot_style()

    n_rows = len(results_by_row)
    fig, axes = plt.subplots(1, n_rows, figsize=(4 * n_rows, 3))

    # Handle single subplot case
    if n_rows == 1:
        axes = [axes]

    colors = sns.color_palette("colorblind")
    markers = [
        "o",
        "s",
        "^",
        "D",
        "v",
        "<",
        ">",
        "p",
    ]  # Standard marker order for samples

    for idx, (row_letter, data) in enumerate(sorted(results_by_row.items())):
        ax = axes[idx]
        calib = data["calibration"]
        samples = data["samples"]

        # Calculate MSE for prediction bands (inverse regression)
        conc_pred_calib = calib.slope * calib.y_values + calib.intercept
        rss = np.sum((calib.concentrations - conc_pred_calib) ** 2)
        mse = rss / (len(calib.y_values) - 2)

        # Create calibration curve equation label with R²
        # For inverse regression: concentration = slope * absorbance + intercept
        r_squared = calib.rvalue**2
        equation_label = (
            f"C = {calib.slope:.3f}A + {calib.intercept:.3f}, R² = {r_squared:.3f}"
        )

        # Plot calibration points (dark grey X markers)
        # X-axis: absorbance difference, Y-axis: concentration
        ax.scatter(
            calib.y_values,
            calib.concentrations,
            color="#2b2b2b",
            marker="x",
            s=50,
            linewidths=1.5,
            label="Calibration",
            zorder=3,
        )

        # Plot fitted line (very dark grey)
        y_line = np.linspace(0, max(calib.y_values) * 1.2, 100)
        c_line = calib.slope * y_line + calib.intercept
        ax.plot(
            y_line,
            c_line,
            color="#2b2b2b",
            linewidth=2,
            alpha=0.8,
            label=equation_label,
            zorder=2,
        )

        # Plot prediction band (grey, semi-transparent)
        c_pred_band_upper = []
        c_pred_band_lower = []
        y_mean = np.mean(calib.y_values)

        for y_val in y_line:
            c_pred = calib.slope * y_val + calib.intercept
            n = len(calib.y_values)
            se_pred = np.sqrt(
                mse
                * (
                    1
                    + 1 / n
                    + (y_val - y_mean) ** 2 / np.sum((calib.y_values - y_mean) ** 2)
                )
            )
            c_pred_band_upper.append(c_pred + se_pred)
            c_pred_band_lower.append(c_pred - se_pred)

        ax.fill_between(
            y_line,
            c_pred_band_lower,
            c_pred_band_upper,
            color="grey",
            alpha=0.3,
            zorder=1,
        )

        # Plot sample points (colorblind palette with varied markers)
        if len(samples) > 0:
            sample_absorbance = samples["Absorbance"].values
            sample_concentrations = samples["Concentration / mM"].values

            for i, (well, y, c) in enumerate(
                zip(samples["Well"].values, sample_absorbance, sample_concentrations)
            ):
                color_idx = i % len(colors)
                marker_idx = i % len(markers)
                # Plot as (absorbance, concentration) to match inverse regression axes
                ax.scatter(
                    [y],
                    [c],
                    color=colors[color_idx],
                    marker=markers[marker_idx],
                    s=50,
                    edgecolors="black",
                    linewidths=0.5,
                    label=well,
                    zorder=4,
                )

        # Format subplot - axes swapped for inverse regression
        ax.set_xlabel("Absorbance Difference")
        ax.set_ylabel("Concentration / mM")
        ax.set_title(f"Row {row_letter}")
        ax.legend(frameon=False, loc="upper left")
        format_axes(ax)

        # Override tick direction to outside
        ax.tick_params(axis="both", which="major", direction="out")
        ax.tick_params(axis="both", which="minor", direction="out")

    plt.tight_layout()
    save_plot(fig, str(output_path / "curves"))
    plt.close(fig)


def save_results_csv(samples_df: pd.DataFrame, output_path: Path):
    """
    Save predicted concentrations and uncertainties to CSV file.

    Parameters
    ----------
    samples_df : pd.DataFrame
        Combined sample predictions from all rows
    output_path : Path
        Directory where CSV will be saved
    """
    results_file = output_path / "results.csv"
    samples_df[["Well", "Concentration / mM", "Uncertainty / mM"]].to_csv(
        results_file, index=False
    )


def process_single_assay(csv_path: Path) -> bool:
    """
    Process a single assay.csv file.

    Performs calibration, prediction, and generates outputs.

    Parameters
    ----------
    csv_path : Path
        Path to assay.csv file

    Returns
    -------
    bool
        True if processing succeeded, False otherwise
    """
    try:
        print(f"Processing: {csv_path.parent.name}/{csv_path.name}")

        # Load and parse data
        df = load_and_parse_csv(csv_path)

        if len(df) == 0:
            print(f"  Warning: No valid data found")
            return False

        # Group by row letter
        results_by_row = {}
        all_samples = []
        warnings = []

        for row_letter in sorted(df["row_letter"].unique()):
            row_data = df[df["row_letter"] == row_letter].copy()
            row_data = row_data.sort_values("Well")  # type: ignore[call-overload]  # Ensure proper order

            # Check minimum wells for calibration
            if len(row_data) < 8:
                warnings.append(f"  Row {row_letter}: < 8 wells, skipping")
                continue

            # Calibrate
            calib = calibrate_row(row_data)

            # Validate
            cal_warnings = validate_calibration(calib, row_letter)
            warnings.extend(cal_warnings)

            # Predict samples
            samples = predict_samples(row_data, calib)

            # Filter out negative concentrations (physically impossible)
            negative_mask = samples["Concentration / mM"] < 0
            if negative_mask.any():
                warnings.append(
                    f"  Row {row_letter}: {negative_mask.sum()} negative "
                    "concentration(s) filtered out"
                )
                samples = samples[~negative_mask]

            results_by_row[row_letter] = {"calibration": calib, "samples": samples}

            all_samples.append(samples)

            print(
                f"  Row {row_letter}: R² = {calib.rvalue**2:.3f}, "
                f"{len(samples)} sample(s) predicted"
            )

        if not results_by_row:
            print(f"  Warning: No valid rows to process")
            return False

        # Print warnings
        for warning in warnings:
            print(warning)

        # Save outputs to same directory as input
        output_path = csv_path.parent

        # Save results CSV
        if all_samples:
            combined_samples = pd.concat(all_samples, ignore_index=True)
            save_results_csv(combined_samples, output_path)
            print(f"  Saved: {output_path.name}/results.csv")

        # Generate and save plot
        plot_calibration_curves(results_by_row, output_path)
        print(f"  Saved: {output_path.name}/curves.svg")

        return True

    except Exception as e:
        print(f"  Error: {str(e)}")
        return False


def main():
    """Main entry point for PC assay analysis."""
    # Get base directory (where this script is located)
    base_dir = Path(__file__).parent

    print("Finding assay.csv files...")
    assay_files = find_all_assay_files(base_dir)

    if not assay_files:
        print("No assay.csv files found")
        return

    print(f"Found {len(assay_files)} file(s)\n")

    # Process each file
    successful = 0
    for csv_path in assay_files:
        if process_single_assay(csv_path):
            successful += 1
        print()  # Blank line between files

    # Summary
    print(f"Successfully processed {successful}/{len(assay_files)} files")


if __name__ == "__main__":
    main()
