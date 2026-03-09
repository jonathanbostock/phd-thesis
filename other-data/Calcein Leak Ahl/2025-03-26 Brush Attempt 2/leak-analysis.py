from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import phdutils.plot as jbplot

def main():

    sample_names = ("control", "no_brush", "sparse_brush", "dense_brush", "triton")


    def load_data(file_name: str, num_rows: int = 3) -> pd.DataFrame:
        file_location = Path(__file__).parent / file_name

        data = pd.read_csv(file_location,
                           skiprows=1,
                           nrows= num_rows)

        data = data.drop(columns=[c for c in data.columns if "Unnamed" in c or ("Time" in c and c != "Time (min)")])

        data.columns = ["Time / min"] + [f"{sample_name} {repeat}"
                                          for repeat in range(1, 4)
                                          for sample_name in sample_names]

        return data

    timecourse_data = load_data("2025-03-26 Calcein Release w Brush Timecourse.csv", num_rows=91)
    before_data = load_data("2025-03-26 Calcein Release w Brush Before.csv", num_rows=3).mean(axis=0)
    after_data = load_data("2025-03-26 Calcein Release w Brush After.csv", num_rows=3).mean(axis=0)

    timecourse_data_adjusted = timecourse_data.apply(lambda row: (row - before_data) / (after_data - before_data) * 100, axis=1)
    timecourse_data_adjusted["Time / min"] = timecourse_data["Time / min"]

    # Calculate mean and standard error for each condition
    processed_data = []
    
    for sample_name in [s for s in sample_names if s != "triton"]:

        # Get columns for this sample type
        sample_columns = [col for col in timecourse_data_adjusted.columns if sample_name in col]
        
        # For each time point
        for _, row in timecourse_data_adjusted.iterrows():

            time = row["Time / min"]
            values = row[sample_columns].values
            
            processed_data.append({
                "Time / min": time,
                "Sample Type": sample_name,
                "Mean": np.mean(values),
                "SEM": stats.sem(values)
            })
    
    # Convert to DataFrame
    df_processed = pd.DataFrame(processed_data)
    
    # Create a plot of the data
    fig, ax = jbplot.figax()
    
    jbplot.plotdf(ax, df_processed,
                  x="Time / min",
                  y="Mean",
                  y_sig="SEM",
                  split="Sample Type",
                  plot_type="line",
                  name_map= lambda tuple: tuple[0].replace("-", " ").title())
    
    ax.set_ylabel("Calcein Leakage / %")
    ax.set_xlabel("Time / min")
    jbplot.nice_legend(ax)

    jbplot.save(fig, Path(__file__).parent / "Calcein Leakage Timecourse")

if __name__ == "__main__":
    main()