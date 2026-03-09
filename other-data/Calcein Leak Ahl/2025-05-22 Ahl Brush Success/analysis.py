"""Jonathan Bostock 2025-05-22

Analyse timecourses of calcein leakage with Ahl brush experiments.
Conditions: 1=No Ahl, 2=No Brush, 3=Sparse, 4=Dense, 5=Triton (excluded)
Replicates: A, B, C
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('../../Python-Modules')
import jbplot

from icecream import ic


def main():

    # Read CSV files - they have alternating Time/Intensity columns
    start_df = pd.read_csv("2025-05-22 Ahl Before.csv", header=1, engine="python")
    end_df = pd.read_csv("2025-05-22 Ahl Triton.csv", header=1, engine="python")  
    timecourse_df = pd.read_csv("2025-05-22 Ahl Timecourse.csv", header=1, engine="python")

    # Clean data - keep only numeric rows at the beginning
    start_df = start_df[:3].apply(pd.to_numeric, errors='coerce')
    end_df = end_df[:3].apply(pd.to_numeric, errors='coerce')
    
    # For timecourse, find the numeric data rows (before the metadata)
    # Convert first column to numeric to identify data rows
    first_col_numeric = pd.to_numeric(timecourse_df.iloc[:, 0], errors='coerce')
    data_rows = ~pd.isna(first_col_numeric)
    timecourse_df = timecourse_df[data_rows].apply(pd.to_numeric, errors='coerce')
    
    # Get final timepoint (last valid row)
    final_timepoint = len(timecourse_df) - 1
    
    # Define conditions (exclude condition 5 which is Triton)
    conditions = ["No Ahl", "No Brush", "Sparse", "Dense"]
    replicates = ["A", "B", "C"]
    
    # Extract intensity columns only
    intensity_cols = [col for col in start_df.columns if 'Intensity (a.u.)' in col]
    
    # Calculate means for start and end measurements
    start_means = start_df[intensity_cols].mean()
    end_means = end_df[intensity_cols].mean()
    
    # Calculate final release for relevant columns (exclude Triton - every 5th column)
    final_release_data = []
    
    for rep_idx, replicate in enumerate(replicates):
        for cond_idx, condition in enumerate(conditions):
            # Column index: replicate*5 + condition (0-indexed)
            col_idx = rep_idx * 5 + cond_idx
            
            if col_idx < len(intensity_cols):
                col_name = intensity_cols[col_idx]
                
                start_val = start_means[col_name]
                end_val = end_means[col_name]
                final_val = timecourse_df[col_name].iloc[final_timepoint]
                
                if pd.notna(start_val) and pd.notna(end_val) and pd.notna(final_val):
                    release_pct = (final_val - start_val) / (end_val - start_val) * 100
                    
                    final_release_data.append({
                        "Condition": condition,
                        "Replicate": replicate,
                        "Release %": release_pct
                    })

    release_df = pd.DataFrame(final_release_data)
    
    print(release_df)

    # Create release percentage plot
    fig, ax = jbplot.figax()
    jbplot.plotdf(ax, release_df,
                  x="Condition",
                  y="Release %",
                  color="Replicate",
                  plot_type="scatterline")
    
    ax.set_xlabel("Condition")
    ax.set_ylabel("Release % after treatment")
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    jbplot.nice_legend(ax)
    jbplot.save(fig, "Release")

    # Create timecourse plot using jbplot.plotdf with means and standard deviations
    # Build long-form DataFrame for timecourse data with means and std
    timecourse_summary_data = []
    
    # Extract time column (first column)
    time_col = [col for col in timecourse_df.columns if 'Time (min)' in col][0]
    time_data = timecourse_df[time_col]
    
    # For each condition, calculate means and standard deviations across replicates
    for cond_idx, condition in enumerate(conditions):
        # Collect data from all replicates for this condition
        condition_timecourses = []
        
        for rep_idx in range(3):  # A, B, C
            col_idx = rep_idx * 5 + cond_idx
            if col_idx < len(intensity_cols):
                col_name = intensity_cols[col_idx]
                
                # Calculate normalized release for each timepoint
                start_val = start_means[col_name]
                end_val = end_means[col_name]
                
                if pd.notna(start_val) and pd.notna(end_val):
                    timecourse_release = (timecourse_df[col_name] - start_val) / (end_val - start_val) * 100
                    condition_timecourses.append(timecourse_release)
        
        if condition_timecourses:
            # Convert to DataFrame for easier calculation
            condition_matrix = pd.DataFrame(condition_timecourses).T
            
            # Calculate mean and standard error for each timepoint
            mean_timecourse = condition_matrix.mean(axis=1)
            sem_timecourse = condition_matrix.sem(axis=1)
            
            # Add mean and SEM data to long-form data
            for time_val, mean_val, sem_val in zip(time_data, mean_timecourse, sem_timecourse):
                if pd.notna(time_val) and pd.notna(mean_val) and pd.notna(sem_val):
                    timecourse_summary_data.append({
                        "Time (min)": time_val,
                        "Release % Mean": mean_val,
                        "Release % SEM": sem_val,
                        "Condition": condition
                    })
    
    timecourse_summary_df = pd.DataFrame(timecourse_summary_data)
    
    # Create timecourse plot with error bars (50% wider)
    fig2, ax2 = plt.subplots(1, 1, figsize=[6, 4])
    jbplot.plotdf(ax2, timecourse_summary_df,
                  x="Time (min)",
                  y="Release % Mean",
                  y_sig="Release % SEM",
                  color="Condition",
                  plot_type="line")
    
    ax2.set_xlabel("Time (min)")
    ax2.set_ylabel("Release %")
    jbplot.nice_legend(ax2)
    jbplot.save(fig2, "Timecourse")


if __name__ == "__main__":
    main()