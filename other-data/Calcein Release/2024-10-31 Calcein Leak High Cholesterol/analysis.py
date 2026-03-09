"""Jonathan Bostock 2024-10-31 (spooky)

Analyse timecourses of calcein leakage with POPC and DOPC at 7:3, 6:4, and 5:5 ratios.
Used three different batches of Ahl to compare.
"""
import numpy as np
import pandas as pd
import jbplot

from icecream import ic


def main():

    kwargs = {
        "header":       1,
        "engine":       "python",
        "usecols":      ["Time (min)", "Intensity (a.u.)"] + [f"Intensity (a.u.).{i}" for i in range(1,24)],
    }

    start_df = pd.read_csv("2024-10-31 Calcein Leak High Cholesterol Before 700V.csv", **kwargs)[:2].astype(float)
    end_df = pd.read_csv("2024-10-31 Calcein Leak High Cholesterol After 700V.csv", **kwargs)[:2].astype(float)
    timecourse_df = pd.read_csv("2024-10-31 Calcein Leak High Cholesterol Timecourse 700V.csv", **kwargs)[:121].astype(float)

    cell_data = [{"Chol %": chol_percent, "Lipid":  lipid, "Ahl Batch": ahl_batch}
                 for chol_percent in [30, 40, 50]
                 for lipid in ["POPC", "DOPC"]
                 for ahl_batch in ["None", "January", "February", "August"]]

    cell_data_lists = {key: value
                       for key, value in zip(
                           cell_data[0].keys(),
                           map(list, zip(*map(lambda x: x.values(), cell_data))))}

    start_means = start_df.apply(np.mean)
    end_means = end_df.apply(np.mean)

    final_release = timecourse_df.T[1:].apply(
        lambda row: (row[120] - start_means[row.name]) / (end_means[row.name] - start_means[row.name]) * 100,
        axis=1)

    final_release_df = pd.DataFrame(
        columns = ["Release %"],
        data = final_release)

    print(final_release_df)

    for key, value in cell_data_lists.items():
        final_release_df[key] = value

    fig, ax = jbplot.figax()
    jbplot.plotdf(ax, final_release_df,
                  x = "Chol %",
                  y = "Release %",
                  color = "Ahl Batch",
                  shape = "Lipid",
                  plot_type = "scatterline")
    ax.set_xlabel("Chol %")
    ax.set_ylabel("Release % after 2h")
    jbplot.nice_legend(ax)
    jbplot.save(fig, "Release")



if __name__ == "__main__":
    main()
