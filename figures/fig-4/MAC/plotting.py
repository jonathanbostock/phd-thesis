### Plot the data for MAC experiments

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from utils.calcein_release import calculate_release_from_fluorescence, calculate_average_and_sem_release
from utils.plotting import plot_calcein_release, save_plot
from utils.stats import t_test_unpaired

def main():
    """
    Main function to plot the data for MAC experiments.
    """
    # Get our file path for this script
    file_path = os.path.dirname(os.path.abspath(__file__))

    # Get our data
    columns = [f"{letter}{number}" for letter in "ABC" for number in range(1, 9)]

    initial_df = convert_csv_to_df(os.path.join(file_path, "initial.csv"))[columns]
    timecourse_df = convert_csv_to_df(os.path.join(file_path, "c9-timecourse.csv"))[columns]
    triton_df = convert_csv_to_df(os.path.join(file_path, "triton.csv")).iloc[:1][columns]

    # Calculate the calcein release
    release_df = calculate_release_from_fluorescence(
        initial_df, timecourse_df, triton_df)

    final_release = release_df.iloc[-1]
    no_brush_data = final_release[["A5", "B5", "C5"]].tolist()
    sparse_brush_data = final_release[["A6", "B6", "C6"]].tolist()
    dense_brush_data = final_release[["A7", "B7", "C7"]].tolist()
    star_brush_data = final_release[["A8", "B8", "C8"]].tolist()

    p_value_dataframe = pd.DataFrame([
        t_test_unpaired(no_brush_data, sparse_brush_data),
        t_test_unpaired(no_brush_data, dense_brush_data),
        t_test_unpaired(no_brush_data, star_brush_data),
        t_test_unpaired(dense_brush_data, star_brush_data)
    ], index=["No Brush vs Sparse Brush", "No Brush vs Dense Brush", "No Brush vs Star Brush", "Dense Brush vs Star Brush"])

    p_value_dataframe.to_csv(os.path.join(file_path, "p_values.csv"))

    mean_release, sem_release = calculate_average_and_sem_release(release_df)

    experiment_names = [
        "-MAC No Brush",
        "-MAC Sparse Brush",
        "-MAC Dense Brush",
        "-MAC Star Brush",
        "+MAC No Brush",
        "+MAC Sparse Brush",
        "+MAC Dense Brush",
        "+MAC Star Brush",
    ]

    barchart, linechart = plot_calcein_release(mean_release, sem_release, experiment_names, group_size=4)

    save_plot(barchart, os.path.join(file_path, "barchart"))
    save_plot(linechart, os.path.join(file_path, "linechart"))

def convert_csv_to_df(csv_path: str | os.PathLike) -> pd.DataFrame:
    """
    Convert a csv file to a dataframe, given our csvs are weirdly formatted.
    
    The CSV files contain 96-well plate data in a grid format where:
    - Rows are labeled A-H
    - Columns are labeled 1-12
    - For timecourse files, multiple cycles represent different timepoints
    
    Returns:
        pd.DataFrame: DataFrame where each row is a timepoint and each column is a well position (A1, A2, etc.)
    """
    # Read the CSV file
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    
    # Find the data sections
    data_sections = []
    current_section = None
    current_time = None
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Check if this is a cycle header (for timecourse files)
        cycle_match = re.match(r'Cycle (\d+) \((\d+) min\)', line)
        if cycle_match:
            # If we have a current section, close it before starting a new one
            if current_section:
                current_section['end_line'] = i
                data_sections.append(current_section)
                current_section = None
            
            current_time = int(cycle_match.group(2))
            current_section = {'time': current_time, 'start_line': i + 1}
            continue
        
        # Check if this is a single timepoint file (no cycle header)
        if 'Raw Data' in line and current_section is None:
            current_time = 0  # Assume time 0 for single timepoint files
            current_section = {'time': current_time, 'start_line': i + 1}
            continue
        
        # If we have a current section and find a line that doesn't start with row labels
        # and is not empty, then we've reached the end of the grid
        if (current_section and 
            line and 
            not line.startswith(('A,', 'B,', 'C,', 'D,', 'E,', 'F,', 'G,', 'H,')) and
            not line.startswith(',')):  # Skip empty lines
            current_section['end_line'] = i
            data_sections.append(current_section)
            current_section = None
        
        # If we have a current section and find the next cycle, close the current section
        if current_section and cycle_match:
            current_section['end_line'] = i
            data_sections.append(current_section)
            current_section = None
    
    # If we have a current section but no end line, it means the file ends with data
    if current_section:
        current_section['end_line'] = len(lines)
        data_sections.append(current_section)
    
    # Extract data from each section
    all_data = []
    
    for section in data_sections:
        time = section['time']
        start_line = section['start_line']
        end_line = section['end_line']
        
        # Extract the grid data
        grid_data = {}
        
        for line_num in range(start_line, end_line):
            line = lines[line_num].strip()
            if line.startswith(('A,', 'B,', 'C,', 'D,', 'E,', 'F,', 'G,', 'H,')):
                parts = line.split(',')
                row = parts[0]  # A, B, C, etc.
                
                # Process each column (1-12)
                for col_idx in range(1, 13):
                    if col_idx < len(parts) and parts[col_idx].strip():
                        try:
                            value = float(parts[col_idx])
                            well_pos = f"{row}{col_idx}"
                            grid_data[well_pos] = value
                        except ValueError:
                            # Skip non-numeric values
                            continue
        
        # Add time and data to the list
        row_data = {'time': time}
        row_data.update(grid_data)
        all_data.append(row_data)
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Set time as index if there are multiple timepoints
    if len(df) > 1:
        df = df.set_index('time')
    
    return df

if __name__ == "__main__":
    main()