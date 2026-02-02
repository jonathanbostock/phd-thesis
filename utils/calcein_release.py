"""
Support for calcein release assays.
"""

import pandas as pd
import numpy as np
from scipy import stats
import glob
import os

from typing import Tuple, Union


def calculate_release_from_path(data_dir: Union[str, os.PathLike]) -> pd.DataFrame:
    """
    Calculate the calcein release from the before, timecourse, and after dataframes.

    Args:
        data_dir: Path to the data directory.

    Returns:
        DataFrame containing the calcein release over time for each cell.
    """

    # Find the data files
    before_files = glob.glob(os.path.join(data_dir, "*[B|b]efore*.csv"))
    timecourse_files = glob.glob(os.path.join(data_dir, "*[T|t]imecourse*.csv"))
    triton_files = glob.glob(os.path.join(data_dir, "*[T|t]riton*.csv"))

    if not before_files or not timecourse_files or not triton_files:
        raise FileNotFoundError(f"Could not find required files in {data_dir}")

    # Match files by their date prefix (e.g., "2025-11-04")
    def get_date_prefix(filepath: str) -> str:
        """Extract date prefix from filename like '2025-11-04 Ahl Control Before.csv'"""
        basename = os.path.basename(filepath)
        # Return the date portion (first 10 characters typically)
        parts = basename.split(" ")
        if len(parts) > 0 and len(parts[0]) == 10:
            return parts[0]
        return basename

    # Group files by date prefix
    before_by_date = {get_date_prefix(f): f for f in before_files}
    timecourse_by_date = {get_date_prefix(f): f for f in timecourse_files}
    triton_by_date = {get_date_prefix(f): f for f in triton_files}

    # Find common dates across all file types
    common_dates = (
        set(before_by_date.keys())
        & set(timecourse_by_date.keys())
        & set(triton_by_date.keys())
    )

    if not common_dates:
        raise FileNotFoundError(
            f"Could not find matching file sets (before/timecourse/triton) in {data_dir}"
        )

    # Process each matched set and combine results
    all_release_dfs = []

    for date_prefix in sorted(common_dates):
        before_file = before_by_date[date_prefix]
        timecourse_file = timecourse_by_date[date_prefix]
        triton_file = triton_by_date[date_prefix]

        # Read the data files
        before_data = pd.read_csv(before_file, header=None)
        timecourse_data = pd.read_csv(timecourse_file, header=None)
        triton_data = pd.read_csv(triton_file, header=None)

        # Extract sample names from the first row
        sample_names = []
        for i in range(0, len(before_data.columns), 2):
            if i < len(before_data.columns) - 1:
                sample_name = str(before_data.iloc[0, i]).strip()
                if sample_name.startswith("Sample "):
                    sample_names.append(sample_name.replace("Sample ", ""))

        # Extract time and intensity data
        # Data starts from row 2 (index 1) and has time, intensity pairs for each sample
        def extract_fluorescence_data(data, sample_names) -> pd.DataFrame:
            """Extract fluorescence data for each sample"""
            fluorescence_data = {}

            data_cleaned = data.iloc[:, :-1]  # Skip the last column
            data_cleaned = data_cleaned.reset_index(drop=True)
            # Remove rows from the bottom that aren't square
            n_samples = len(sample_names)
            n_cols_expected = n_samples * 2
            # Find the last row where all columns are present
            valid_rows = data_cleaned.iloc[:, :n_cols_expected].dropna(how="any")
            data_cleaned = data_cleaned.iloc[: len(valid_rows) + 1]

            for i, sample in enumerate(sample_names):
                # Each sample has 2 columns: time and intensity
                time_col = 2 * i
                intensity_col = 2 * i + 1

                if time_col < len(data.columns) and intensity_col < len(data.columns):
                    # Skip the header rows and get the actual data
                    data_start = 2  # Start after the header rows

                    # Get time and intensity values
                    times = data_cleaned.iloc[data_start:, time_col].astype(float)
                    intensities = data_cleaned.iloc[data_start:, intensity_col].astype(
                        float
                    )

                    # Remove any NaN values
                    valid_mask = ~(times.isna() | intensities.isna())
                    fluorescence_data[sample] = intensities[valid_mask].values

            return pd.DataFrame(fluorescence_data)

        # Extract data from all three files
        before_fluorescence = extract_fluorescence_data(before_data, sample_names)
        timecourse_fluorescence = extract_fluorescence_data(
            timecourse_data, sample_names
        )
        triton_fluorescence = extract_fluorescence_data(triton_data, sample_names)

        sample_names = before_fluorescence.columns

        before_avg = before_fluorescence.mean(axis=0)
        triton_avg = triton_fluorescence.mean(axis=0)

        # Ensure we have Series objects
        if not isinstance(before_avg, pd.Series):
            before_avg = pd.Series(before_avg)
        if not isinstance(triton_avg, pd.Series):
            triton_avg = pd.Series(triton_avg)

        # Calculate calcein release percentage
        release_data = {}

        for sample in sample_names:
            if (
                sample in timecourse_fluorescence.columns
                and sample in before_avg.index
                and sample in triton_avg.index
            ):
                f_timecourse = timecourse_fluorescence[sample]
                f_before = float(before_avg[sample])
                f_triton = float(triton_avg[sample])

                # Calculate release percentage
                release_percentage = (
                    100 * (f_timecourse - f_before) / (f_triton - f_before)
                )

                release_data[sample] = release_percentage

        # Create DataFrame for this run and add to list
        run_df = pd.DataFrame(release_data)
        all_release_dfs.append(run_df)

    # Combine all runs by concatenating columns (each run has different well names)
    if len(all_release_dfs) == 1:
        result_df = all_release_dfs[0]
    else:
        result_df = pd.concat(all_release_dfs, axis=1)

    return result_df


def calculate_release_from_fluorescence(
    initial_df: pd.DataFrame, timecourse_df: pd.DataFrame, triton_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate calcein release from 96-well plate data.

    Args:
        initial_df: DataFrame with initial fluorescence values (single timepoint)
        timecourse_df: DataFrame with fluorescence values over time
        triton_df: DataFrame with triton-treated fluorescence values (maximum release)

    Returns:
        DataFrame with calcein release percentage over time for each well
    """
    # Get the common wells that exist in all three datasets
    initial_wells = set(initial_df.columns)
    timecourse_wells = set(timecourse_df.columns)
    triton_wells = set(triton_df.columns)

    common_wells = initial_wells.intersection(timecourse_wells).intersection(
        triton_wells
    )
    common_wells = sorted(list(common_wells))

    # Calculate average initial and triton values for each well
    initial_avg = {}
    triton_avg = {}

    for well in common_wells:
        # For initial data, use the single value
        initial_avg[well] = initial_df[well].iloc[0]

        # For triton data, use the average of all timepoints
        triton_values = triton_df[well].dropna()
        if len(triton_values) > 0:
            triton_avg[well] = triton_values.mean()
        else:
            triton_avg[well] = np.nan

    # Calculate calcein release percentage for each well over time
    release_data = {}
    release_data["time"] = timecourse_df.index.values

    for well in common_wells:
        if well in initial_avg and well in triton_avg:
            f_initial = initial_avg[well]
            f_triton = triton_avg[well]

            if (
                not np.isnan(f_initial)
                and not np.isnan(f_triton)
                and f_triton > f_initial
            ):
                # Get timecourse values for this well
                f_timecourse = timecourse_df[well].values

                # Calculate release percentage: (F_t - F_0) / (F_triton - F_0) * 100
                release_percentage = (
                    100 * (f_timecourse - f_initial) / (f_triton - f_initial)
                )
                release_data[well] = release_percentage
            else:
                # Skip wells with invalid data
                print(f"Skipping well {well}: invalid initial or triton values")

    # Create the final DataFrame
    result_df = pd.DataFrame(release_data)
    result_df = result_df.set_index("time")

    return result_df


def calculate_average_and_sem_release(
    release_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate the average and sem of the release data.
    """
    # Group wells by number (column position) and calculate statistics
    number_stats = {}

    for well in release_df.columns:
        number = well[1:]  # Extract the number part (e.g., "1" from "A1")
        if number not in number_stats:
            number_stats[number] = []
        number_stats[number].append(release_df[well])

    # Calculate mean and SEM for each number
    mean_data = {}
    sem_data = {}

    for number, wells in number_stats.items():
        if wells:
            # Concatenate all wells for this number
            number_df = pd.concat(wells, axis=1)

            # Calculate mean and SEM across rows (timepoints)
            mean_data[number] = number_df.mean(axis=1)
            sem_data[number] = number_df.sem(axis=1)

    # Create result DataFrames
    mean_df = pd.DataFrame(mean_data)
    sem_df = pd.DataFrame(sem_data)

    return mean_df, sem_df
