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


file_name = "2024-05-21 Laurdan POPC Brush Ex360.csv"

raw_data = pd.read_csv(file_name,
                       header=1,
                       index_col="Unnamed: 0").iloc[1:150].astype(float)
raw_data.index = [int(float(i)) for i in raw_data.index]

NAMES = ["None 0", "None 1", "None 2",
         "Sparse 0", "Sparse 1", "Sparse 2",
         "Dense 0", "Dense 1", "Dense 2"]
CONDITIONS = ["None", "Sparse", "Dense"]

raw_data.columns = NAMES + ["Blank"]

for name in NAMES:
    raw_data[f"{name} Blanked"] = raw_data.apply(
        lambda row: row[name] - row["Blank"],
        axis=1
    )

proc_data = raw_data.loc[::,[f"{name} Blanked" for name in NAMES]].apply(
    lambda col: general_polarization(col)
).transpose()



for cond in CONDITIONS:
    proc_data[f"{cond} Mean"] = np.mean([proc_data[f"{cond} {i} Blanked"] for i in range(3)])
    proc_data[f"{cond} SEM"] = stats.sem([proc_data[f"{cond} {i} Blanked"] for i in range(3)])

barchart_dict = {cond:  (proc_data[f"{cond} Mean"],
                         proc_data[f"{cond} SEM"])
                 for cond in CONDITIONS}

fig, ax = jbplot.figax()
jbplot.barchart(ax,
                barchart_dict)
jbplot.save(fig, "Laurdan Results", keep_box=True)
