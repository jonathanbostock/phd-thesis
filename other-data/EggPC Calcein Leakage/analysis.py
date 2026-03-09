### Jonathan Bostock

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import jbplot

def main():

    fluorescence_columns = ["Intensity (a.u.)"] + [f"Intensity (a.u.).{i}" for i in range(1,15)]

    csv_kwargs = {
        "header":   1,
        "engine":   "python",
        "usecols":  ["Time (min)"] + fluorescence_columns
    }

    # Load the files
    start_df = pd.read_csv("2024-11-21 EggPC Brush Ahl Before.csv", **csv_kwargs)[:2].astype(float)
    end_df = pd.read_csv("2024-11-21 EggPC Brush Ahl After.csv", **csv_kwargs)[:2].astype(float)
    timecourse_df = pd.read_csv("2024-11-21 EggPC Brush Ahl Timecourse.csv", **csv_kwargs)[:60].astype(float)

    conditions = ["-ve", "No Brush", "Sparse Brush", "Dense Brush"]


    start_means = start_df.apply(np.mean)
    end_means = end_df.apply(np.mean)

    def normalize_column_and_return_dataframe(column_names: list[str], condition: str) -> pd.DataFrame:

        new_df = pd.DataFrame()

        for i, column_name in enumerate(column_names):
            new_df[f"Fluorescence {i}"] = timecourse_df[column_name]
            new_df[f"Normalized Release {i}"] = (
                (new_df[f"Fluorescence {i}"] - start_means[column_name]) /\
                (end_means[column_name] - start_means[column_name]) * 100)

        new_df["Release Mean"] = new_df.apply(
            lambda row: np.mean([row[f"Normalized Release {i}"] for i in range(3)]),
            axis=1)
        new_df["Release SEM"] = new_df.apply(
            lambda row: stats.sem([row[f"Normalized Release {i}"] for i in range(3)]),
            axis=1)
        new_df["Time / min"] = timecourse_df["Time (min)"]

        new_df["Condition"] = condition

        return new_df


    release_df = pd.concat(
        [normalize_column_and_return_dataframe(fluorescence_columns[i::5], condition)
         for i, condition in enumerate(conditions)])

    fig, ax = plt.subplots(figsize=[6,4])
    jbplot.plotdf(ax, release_df,
                  x="Time / min",
                  y="Release Mean",
                  y_sig="Release SEM",
                  color="Condition",
                  plot_type = "line")
    jbplot.nice_legend(ax)
    ax.set_xlabel("Time / min")
    ax.set_ylabel("Release %")
    jbplot.save(fig, "EggPC Release")

if __name__ == "__main__":
    main()
