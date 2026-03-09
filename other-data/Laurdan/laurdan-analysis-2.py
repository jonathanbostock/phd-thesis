### Jonathan bostock

import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import jbplot

def general_polarization(column):

    low = column[435]
    high = column[500]

    return (low-high)/(low+high)

def normalize_column(column, blank_df):

    column_letter = column.name[-2]
    column_number = int(column.name[-1])
    blank = blank_df[f"Sample {column_letter}7"]
    blanked_column = column - blank
    norm_factor = blanked_column[435] + blanked_column[500]
    normed_column = blanked_column/norm_factor
    normed_column.name = f"{CONDITIONS[column_number-1]} {REPLICATES[column_letter]}"

    return normed_column


file_name = "2024-05-22 Laurdan POPC Brush.csv"

raw_data = pd.read_csv(file_name,
                       header=1,
                       index_col="Unnamed: 0").iloc[1:150].astype(float)
raw_data.index = [int(float(i)) for i in raw_data.index]

CONDITIONS = ["None", "1:2000", "1:1000",
              "1:400", "1:200", "1:100"]

REPLICATES = {"A":  0,
              "B":  1,
              "C":  2,
              0:    "A",
              1:    "B",
              2:    "C"}

good_columns = [f"Sample {['A', 'B', 'C'][i]}{j+1}" for i in range(3) for j in range(6)]
normed_data = raw_data.loc[::,good_columns].apply(
    lambda col: normalize_column(col, raw_data))

normed_data.columns = [f"{CONDITIONS[int(col[-1])-1]} {REPLICATES[col[-2]]}"
                       for col in normed_data.columns]

gp_data = pd.DataFrame(normed_data.apply(
    lambda col: general_polarization(col))).transpose()

for df in [gp_data, normed_data]:
    for c in CONDITIONS:
        df[f"{c} Mean"] = df.apply(
            lambda row: np.mean([row[f"{c} {i}"] for i in range(3)]),
            axis=1)
        df[f"{c} SEM"] = df.apply(
            lambda row: stats.sem([row[f"{c} {i}"] for i in range(3)]),
            axis=1)

normed_data["Wavelength"] = normed_data.index
gp_data = gp_data.transpose()

fig, ax = jbplot.figax()
jbplot.plotdf(ax, normed_data,
              x = "Wavelength",
              y = [f"{c} Mean" for c in CONDITIONS],
              y_sig = [f"{c} SEM" for c in CONDITIONS],
              name_list = CONDITIONS,
              gradient=True,
              gradient_list = range(6),
              plot_type="line")
ax.set_xlabel(r"$\lambda$ / nm")
ax.set_ylabel("Intensity (normed)")
jbplot.nice_legend(ax)
jbplot.save(fig, "Brush Density vs Laurdan Spectrum")

fig_2, ax_2 = jbplot.figax()
jbplot.scatter(ax_2,
               x_vect = [0, 1/2000, 1/1000, 1/400, 1/200, 1/100],
               y_vect = gp_data.loc[[f"{c} Mean" for c in CONDITIONS], 0],
               y_sig_vect = gp_data.loc[[f"{c} SEM" for c in CONDITIONS], 0])
ax_2.set_xlabel("DNA:Lipid Ratio")
ax_2.set_xscale("symlog", linthresh=1/1000)
ax_2.set_ylabel("GP (435 nm, 500 nm)")
jbplot.save(fig_2, "Brush Density vs GP")
