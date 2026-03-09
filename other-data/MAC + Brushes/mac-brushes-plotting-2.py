### Jonathan Bostock

import numpy as np
import scipy.stats as stats
import pandas as pd

import jbplot
from icecream import ic

def main():

    before_df = pd.read_csv("2024-07-10 Calcein Leakage Before.csv",
                            header=5).iloc[...,3:]
    c8_df = pd.read_csv("2024-07-10 Calcein C5B6.csv",
                          header=5).iloc[...,3:]
    timecourse_df = pd.read_csv("2024-07-10 Calcein Leakage Timecourse.csv",
                                header=5).iloc[...,3:]
    after_df = pd.read_csv("2024-07-10 Calcein Leakage Triton.csv",
                           header=5).iloc[...,3:]

    norm_curried = lambda row: norm(row, before_df, after_df)

    c8_leakage_df = c8_df.apply(
        norm_curried,
        axis=1)
    timecourse_leakage_df = timecourse_df.apply(
        norm_curried,
        axis=1)

    conditions = ["No Brush",
                  "Sparse Brush",
                  "Dense Brush",
                  "Benzonase",
                  "MBCD",
                  "PEG"]
    controls = ["-MAC",
                "+MAC"]

    timecourse_processed = [pd.DataFrame(
        {"Time (min)":  range(61)}).T]
    c8_processed = []
    names = []

    for i_cond, cond in enumerate(conditions):
        for i_cont, cont in enumerate(controls):
            name = f"{cond} {cont}"
            names.append(name)

            indices = [i_cond + 6*i_cont + 12*i for i in range(3)]

            c8_repeats = [np.mean(c8_leakage_df.iloc[i]) for i in indices]
            c8_mean = np.mean(c8_repeats)
            c8_sem = stats.sem(c8_repeats)
            c8_processed.append([c8_mean, c8_sem])

            timecourse_data = np.array(
                [list(timecourse_leakage_df.iloc[i]) for i in indices])

            timecourse_processed.append(pd.DataFrame(
                {f"{name} Mean":    np.mean(timecourse_data, axis=0),
                 f"{name} SEM":     stats.sem(timecourse_data, axis=0)}).T)


    """
    c8_processed_df = pd.DataFrame(
        c8_processed, columns=["Mean", "SEM"],
        index = names)

    fig, ax = jbplot.figax()

    jbplot.barchart(ax,
                    c8_processed_df["Mean"],
                    name_list = names,
                    sigma_list = c8_processed_df["SEM"],
                    groups = 2,
                    label_rotation = 45)

    jbplot.save(fig, "C8 Leakage")
    """

    timecourse_df = pd.concat(timecourse_processed).T

    fig, ax = jbplot.figax()
    jbplot.plotdf(ax, timecourse_df,
                  x = "Time (min)",
                  y = [f"{cond} +MAC Mean" for cond in conditions],
                  y_sig = [f"{cond} +MAC SEM" for cond in conditions],
                  plot_type = "line",
                  linecode_override=0)
    jbplot.plotdf(ax, timecourse_df,
                  x = "Time (min)",
                  y = [f"{cond} -MAC Mean" for cond in conditions],
                  y_sig = [f"{cond} -MAC SEM" for cond in conditions],
                  plot_type = "line",
                  linecode_override=1)

    jbplot.scatterset(ax, [[]], [[]]*6,
                      marktype="c",
                      name_list = conditions)

    jbplot.plotline(ax, [], [],
                    linecode=0,
                    color_override = "#808080",
                    label="+MAC")
    jbplot.plotline(ax, [], [],
                    linecode=1,
                    color_override = "#808080",
                    label="-MAC")

    jbplot.nice_legend(ax)
    ax.set_ylabel("Lysis")
    ax.set_xlabel("Time / min")
    ax.set_yscale("symlog", linthresh=0.01)
    jbplot.save(fig, "Timecourse")


def norm(row, before_df, after_df):

    before = np.mean(before_df.loc[row.name])
    after = np.mean(after_df.loc[row.name])

    return (row - before) / (after - before)

if __name__ == "__main__":
    main()
