### Jonathan bostock

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import jbplot

CONDITIONS = ["Free",
              "Sparse",
              "Dense"]
CONDITIONS_DICT = {1:   "Free",
                   2:   "Free",
                   3:   "Free",
                   4:   "Dense",
                   5:   "Dense",
                   6:   "Dense",
                   7:   "Sparse",
                   8:   "Sparse",
                   9:   "Sparse"}
INVESTIGATIONS = {"A": "+ve",
                  "B": "-ve",
                  "C": "Exp"}
CONTROLS = ["Exp", "+ve", "-ve"]

def convert_column(column_name):

    column_name = column_name.split(" ")[1]

    letter, number = column_name[0], int(column_name[1:])

    batch = (number - 1) % 3

    condition_name = CONDITIONS_DICT[number]
    investigation_name = INVESTIGATIONS[letter]

    return f"{condition_name} {investigation_name} {batch}"

def get_digestion(row, c, b):

    pos = row[f"{c} +ve {b}"]
    neg = row[f"{c} -ve {b}"]
    exp = row[f"{c} Exp {b}"]

    return (exp - neg)/(pos - neg)


file_name = "2024-05-16 Brush Benzonase Timecourse.csv"

raw_data = pd.read_csv(file_name,
                       skiprows=1, usecols=lambda x: "Intensity" in x,
                       nrows=150)

columns = pd.read_csv(file_name, usecols=lambda x: len(x) < 10).columns

raw_data.columns = [convert_column(c) for c in columns]
fits = {}
for c in CONDITIONS:
    for b in range(3):
        raw_data[f"{c} Digestion {b}"] = raw_data.apply(
            lambda row: get_digestion(row, c, b),
            axis=1)

        raw_data[f"{c} LogRemainder {b}"] = raw_data.apply(
            lambda row: np.log(1-row[f"{c} Digestion {b}"]),
            axis=1)

    raw_data[f"{c} Digestion Mean"] = raw_data.apply(
        lambda row: np.mean([row[f"{c} Digestion {b}"] for b in range(3)]),
        axis=1)
    raw_data[f"{c} Digestion SEM"] = raw_data.apply(
        lambda row: stats.sem([row[f"{c} Digestion {b}"] for b in range(3)]),
        axis=1)

    for ctrl in CONTROLS:
        raw_data[f"{c} {ctrl} Mean"] = raw_data.apply(
            lambda row: np.mean([row[f"{c} {ctrl} {b}"] for b in range(3)]),
            axis=1)
        raw_data[f"{c} {ctrl} SEM"] = raw_data.apply(
            lambda row: stats.sem([row[f"{c} {ctrl} {b}"] for b in range(3)]),
            axis=1)

    raw_data[f"{c} LogRemainder Mean"] = raw_data.apply(
        lambda row: np.mean([row[f"{c} LogRemainder {b}"] for b in range(3)]),
        axis=1)

    raw_data[f"{c} LogRemainder SEM"] = raw_data.apply(
        lambda row: stats.sem([row[f"{c} LogRemainder {b}"] for b in range(3)]),
        axis=1)

    fits[c] = stats.linregress(x=range(25), y=raw_data[f"{c} LogRemainder Mean"][:25])


raw_data["Time"] = range(150)

NAMES = ["No Lipid", "Sparse Brush", "Dense Brush"]

fig, ax = jbplot.figax()

jbplot.plotdf(ax, raw_data,
              x = "Time",
              y = [f"{c} Digestion Mean" for c in CONDITIONS],
              y_sig = [f"{c} Digestion SEM" for c in CONDITIONS],
              name_list = NAMES,
              plot_type = "line",
              linecode_override=0)

jbplot.nice_legend(ax)
ax.set_xlabel("Time")
ax.set_ylabel("Digestion")

jbplot.save(fig, "Digestion By FRET 1")

fig_2, ax_2 = jbplot.figax()
for i, ctrl in enumerate(CONTROLS):
    jbplot.plotdf(ax_2, raw_data,
                  x = "Time",
                  y = [f"{c} {ctrl} Mean" for c in CONDITIONS],
                  y_sig = [f"{c} {ctrl} SEM" for c in CONDITIONS],
                  linecode_override=i,
                  plot_type = "line")

jbplot.plotlineset(ax_2,
                   [[]], [[],[],[]],
                   linecode_override = 0,
                   name_list = CONDITIONS)
jbplot.plotlineset(ax_2,
                   [[]], [[],[],[]],
                   color_override="grey",
                   name_list=CONTROLS)

jbplot.nice_legend(ax_2)
ax_2.set_xlabel("Time")
ax_2.set_ylabel("Fluorescence")

jbplot.save(fig_2, "Fluorescence Over Time")

fig_3, ax_3 = jbplot.figax()

jbplot.plotdf(ax_3, raw_data[:25],
              x="Time",
              y=[f"{c} LogRemainder Mean" for c in CONDITIONS],
              y_sig = [f"{c} LogRemainder SEM" for c in CONDITIONS],
              name_list=NAMES,
              plot_type="scatter")

jbplot.nice_legend(ax_3)
ax_3.set_xlabel("Time")
ax_3.set_ylabel("Log(1-Digestion)")

jbplot.save(fig_3, "Rate Fits 1")
