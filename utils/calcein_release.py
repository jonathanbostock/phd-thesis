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

    if len(before_files) != len(timecourse_files) or len(before_files) != len(
        triton_files
    ):
        raise ValueError(
            f"Mismatched number of Before/Timecourse/Triton files in {data_dir}: "
            f"{len(before_files)} / {len(timecourse_files)} / {len(triton_files)}"
        )

    def extract_fluorescence_data(
        data: pd.DataFrame, sample_names: list[str]
    ) -> pd.DataFrame:
        """Extract fluorescence data for each sample from a raw CSV DataFrame."""
        fluorescence_data = {}
        data_cleaned = data.iloc[:, :-1].reset_index(drop=True)
        n_cols_expected = len(sample_names) * 2
        valid_rows = data_cleaned.iloc[:, :n_cols_expected].dropna(how="any")
        data_cleaned = data_cleaned.iloc[: len(valid_rows) + 1]

        for i, sample in enumerate(sample_names):
            time_col = 2 * i
            intensity_col = 2 * i + 1
            if time_col < len(data.columns) and intensity_col < len(data.columns):
                times = data_cleaned.iloc[2:, time_col].astype(float)
                intensities = data_cleaned.iloc[2:, intensity_col].astype(float)
                valid_mask = ~(times.isna() | intensities.isna())
                fluorescence_data[sample] = intensities[valid_mask].values

        return pd.DataFrame(fluorescence_data)

    def process_run(
        before_file: str, timecourse_file: str, triton_file: str
    ) -> pd.DataFrame:
        """Process one set of Before/Timecourse/Triton CSVs into a release DataFrame."""
        before_data = pd.read_csv(before_file, header=None)
        timecourse_data = pd.read_csv(timecourse_file, header=None)
        triton_data = pd.read_csv(triton_file, header=None)

        sample_names = []
        for i in range(0, len(before_data.columns), 2):
            if i < len(before_data.columns) - 1:
                name = str(before_data.iloc[0, i]).strip()
                if name.startswith("Sample "):
                    sample_names.append(name.replace("Sample ", ""))

        before_fluorescence = extract_fluorescence_data(before_data, sample_names)
        timecourse_fluorescence = extract_fluorescence_data(
            timecourse_data, sample_names
        )
        triton_fluorescence = extract_fluorescence_data(triton_data, sample_names)

        cols = before_fluorescence.columns
        before_avg = before_fluorescence.mean(axis=0)
        triton_avg = triton_fluorescence.mean(axis=0)

        release_data = {}
        for sample in cols:
            if sample in timecourse_fluorescence.columns:
                f_before = float(before_avg[sample])  # type: ignore[arg-type]
                f_triton = float(triton_avg[sample])  # type: ignore[arg-type]
                release_data[sample] = (
                    100
                    * (timecourse_fluorescence[sample] - f_before)
                    / (f_triton - f_before)
                )

        return pd.DataFrame(release_data)

    # Process all runs and concatenate (runs use distinct row letters, so column
    # names like A1, B1 from run 1 and C1, D1 from run 2 don't collide)
    run_dfs = [
        process_run(b, t, r)
        for b, t, r in zip(
            sorted(before_files), sorted(timecourse_files), sorted(triton_files)
        )
    ]

    return pd.concat(run_dfs, axis=1)


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
