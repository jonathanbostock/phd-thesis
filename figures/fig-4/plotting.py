"""Unified plotting for fig-4: Ahl and MAC calcein release experiments."""

import os
import re
import pandas as pd
from typing import Union

import utils.plotting as plotting
import utils.calcein_release as calcein_release
from utils.calcein_release import (
    calculate_release_from_fluorescence,
    calculate_average_and_sem_release,
)
from utils.stats import t_test_unpaired
from utils import defaults


def convert_csv_to_df(csv_path: Union[str, os.PathLike]) -> pd.DataFrame:
    """
    Convert a csv file to a dataframe, given our csvs are weirdly formatted.

    The CSV files contain 96-well plate data in a grid format where:
    - Rows are labeled A-H
    - Columns are labeled 1-12
    - For timecourse files, multiple cycles represent different timepoints

    Returns:
        pd.DataFrame: DataFrame where each row is a timepoint and each column is a well position (A1, A2, etc.)
    """
    with open(csv_path, "r") as f:
        lines = f.readlines()

    data_sections = []
    current_section = None
    current_time = None

    for i, line in enumerate(lines):
        line = line.strip()

        cycle_match = re.match(r"Cycle (\d+) \((\d+) min\)", line)
        if cycle_match:
            if current_section:
                current_section["end_line"] = i
                data_sections.append(current_section)
                current_section = None

            current_time = int(cycle_match.group(2))
            current_section = {"time": current_time, "start_line": i + 1}
            continue

        if "Raw Data" in line and current_section is None:
            current_time = 0
            current_section = {"time": current_time, "start_line": i + 1}
            continue

        if (
            current_section
            and line
            and not line.startswith(("A,", "B,", "C,", "D,", "E,", "F,", "G,", "H,"))
            and not line.startswith(",")
        ):
            current_section["end_line"] = i
            data_sections.append(current_section)
            current_section = None

        if current_section and cycle_match:
            current_section["end_line"] = i
            data_sections.append(current_section)
            current_section = None

    if current_section:
        current_section["end_line"] = len(lines)
        data_sections.append(current_section)

    all_data = []

    for section in data_sections:
        time = section["time"]
        start_line = section["start_line"]
        end_line = section["end_line"]

        grid_data = {}

        for line_num in range(start_line, end_line):
            line = lines[line_num].strip()
            if line.startswith(("A,", "B,", "C,", "D,", "E,", "F,", "G,", "H,")):
                parts = line.split(",")
                row = parts[0]

                for col_idx in range(1, 13):
                    if col_idx < len(parts) and parts[col_idx].strip():
                        try:
                            value = float(parts[col_idx])
                            well_pos = f"{row}{col_idx}"
                            grid_data[well_pos] = value
                        except ValueError:
                            continue

        row_data = {"time": time}
        row_data.update(grid_data)
        all_data.append(row_data)

    df = pd.DataFrame(all_data)

    if len(df) > 1:
        df = df.set_index("time")

    return df


def plot_ahl(fig4_path: str) -> None:
    """Load and plot alpha hemolysin data."""
    ahl_path = os.path.join(fig4_path, "Ahl")
    release_df = calcein_release.calculate_release_from_path(ahl_path)

    mean_release, sem_release = calcein_release.calculate_average_and_sem_release(
        release_df
    )

    experiment_names = [
        "No Brush",
        "Sparse Brush",
        "Dense Brush",
        "No Brush",
        "Sparse Brush",
        "Dense Brush",
    ]

    group_size = 3
    mean_df = mean_release.iloc[:, :-1]
    sem_df = sem_release.iloc[:, :-1]

    # Full barchart and linechart (with controls)
    fig_bar, fig_line = plotting.plot_calcein_release(
        mean_df=mean_df,
        sem_df=sem_df,
        raw_release_df=release_df,
        experiment_names=experiment_names,
        group_size=group_size,
    )
    plotting.save_plot(fig_bar, os.path.join(ahl_path, "barchart"))
    plotting.save_plot(fig_line, os.path.join(ahl_path, "linechart"))

    # Barchart without controls (last group_size columns only, half width)
    mean_df_nc = mean_df.iloc[:, -group_size:]
    sem_df_nc = sem_df.iloc[:, -group_size:]
    experiment_names_nc = experiment_names[-group_size:]

    fig_bar_nc, _ = plotting.plot_calcein_release(
        mean_df=mean_df_nc,
        sem_df=sem_df_nc,
        raw_release_df=release_df,
        experiment_names=experiment_names_nc,
        group_size=group_size,
        bar_figsize=(
            defaults.fig_width / 2 * defaults.small_fig_scale,
            defaults.fig_height * defaults.small_fig_scale,
        ),
    )
    plotting.save_plot(fig_bar_nc, os.path.join(ahl_path, "barchart_no_controls"))

    # P-values
    final_release = release_df.iloc[-1]
    no_brush_data = final_release[["A4", "B4", "C4"]].tolist()
    sparse_brush_data = final_release[["A5", "B5", "C5"]].tolist()
    dense_brush_data = final_release[["A6", "B6", "C6"]].tolist()

    p_value_dataframe = pd.DataFrame(
        [
            t_test_unpaired(no_brush_data, sparse_brush_data),
            t_test_unpaired(no_brush_data, dense_brush_data),
        ]
    )
    p_value_dataframe.index = ["No Brush vs Sparse Brush", "No Brush vs Dense Brush"]
    p_value_dataframe.to_csv(os.path.join(ahl_path, "p_values.csv"))


def plot_mac(fig4_path: str) -> None:
    """Load and plot MAC (membrane attack complex) data."""
    mac_path = os.path.join(fig4_path, "MAC")

    columns = [f"{letter}{number}" for letter in "ABC" for number in range(1, 9)]

    initial_df_full = convert_csv_to_df(os.path.join(mac_path, "initial.csv"))
    timecourse_df_full = convert_csv_to_df(os.path.join(mac_path, "c9-timecourse.csv"))
    triton_df_full = convert_csv_to_df(os.path.join(mac_path, "triton.csv")).iloc[:1]

    initial_df = pd.DataFrame(initial_df_full[columns])
    timecourse_df = pd.DataFrame(timecourse_df_full[columns])
    triton_df = pd.DataFrame(triton_df_full[columns])

    release_df = calculate_release_from_fluorescence(
        initial_df, timecourse_df, triton_df
    )

    group_size = 4
    experiment_names = [
        "No Brush",
        "Sparse Brush",
        "Dense Brush",
        "Star Brush",
        "No Brush",
        "Sparse Brush",
        "Dense Brush",
        "Star Brush",
    ]

    mean_release, sem_release = calculate_average_and_sem_release(release_df)

    # Full barchart and linechart (with controls)
    fig_bar, fig_line = plotting.plot_calcein_release(
        experiment_names=experiment_names,
        mean_df=mean_release,
        sem_df=sem_release,
        raw_release_df=release_df,
        group_size=group_size,
    )
    plotting.save_plot(fig_bar, os.path.join(mac_path, "barchart"))
    plotting.save_plot(fig_line, os.path.join(mac_path, "linechart"))

    # Barchart without controls (last group_size columns only, half width)
    mean_df_nc = mean_release.iloc[:, -group_size:]
    sem_df_nc = sem_release.iloc[:, -group_size:]
    experiment_names_nc = experiment_names[-group_size:]

    fig_bar_nc, _ = plotting.plot_calcein_release(
        mean_df=mean_df_nc,
        sem_df=sem_df_nc,
        raw_release_df=release_df,
        experiment_names=experiment_names_nc,
        group_size=group_size,
        bar_figsize=(
            defaults.fig_width / 2 * defaults.small_fig_scale,
            defaults.fig_height * defaults.small_fig_scale,
        ),
    )
    plotting.save_plot(fig_bar_nc, os.path.join(mac_path, "barchart_no_controls"))

    # P-values
    final_release = release_df.iloc[-1]
    no_brush_data = final_release[["A5", "B5", "C5"]].tolist()
    sparse_brush_data = final_release[["A6", "B6", "C6"]].tolist()
    dense_brush_data = final_release[["A7", "B7", "C7"]].tolist()
    star_brush_data = final_release[["A8", "B8", "C8"]].tolist()

    p_value_dataframe = pd.DataFrame(
        [
            t_test_unpaired(no_brush_data, sparse_brush_data),
            t_test_unpaired(no_brush_data, dense_brush_data),
            t_test_unpaired(no_brush_data, star_brush_data),
            t_test_unpaired(dense_brush_data, star_brush_data),
        ]
    )
    p_value_dataframe.index = [
        "No Brush vs Sparse Brush",
        "No Brush vs Dense Brush",
        "No Brush vs Star Brush",
        "Dense Brush vs Star Brush",
    ]
    p_value_dataframe.to_csv(os.path.join(mac_path, "p_values.csv"))


def main() -> None:
    fig4_path = os.path.dirname(os.path.abspath(__file__))
    plot_ahl(fig4_path)
    plot_mac(fig4_path)


if __name__ == "__main__":
    main()
