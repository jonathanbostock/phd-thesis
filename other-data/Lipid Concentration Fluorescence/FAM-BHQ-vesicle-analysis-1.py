### Jonathan Bostock
### 19-Mar-2024

import numpy as np
import scipy
import pandas as pd
import jbplot
import matplotlib.pyplot as plt

def average_wavelength_function(series):
    return np.average(list(series)[1:6])

def wells_stats_function(series, well_column):

    wells = [letter + str(well_column) for letter in ["A","B","C"]]

    if "Mean" in series.name:
        return np.average(series.loc[wells])
    else:
        return scipy.stats.sem(series.loc[wells])

def process_raw(df):

    working_df = df.loc[1:16, USECOLS]
    working_df.columns = COLUMN_NAMES

    working_df.index = [str(i) for i in range(510, 526)]

    working_df = working_df.astype(float)

    for letter in ["A", "B", "C"]:

        blank_values = working_df[letter + "1"].copy()
        control_values = working_df[letter + "2"].copy() - blank_values

        for number in range(1,13):
            c = letter + str(number)
            working_df[c] = working_df[c] - blank_values
            working_df[c] = working_df[c] / control_values

    return working_df.transpose()

CONCENTRATIONS = [0, 2.5, 5, 7.5, 10, 15, 20, 40, 60, 80, 800]
USECOLS = ["Unnamed: {}".format(i) for i in range(1,73,2)]

before_raw = pd.read_csv("2024-03-19 FAM BHQ Vesicles Before gCD 800V.csv",
                         nrows = 17)
after_raw = pd.read_csv("2024-03-19 FAM BHQ Vesicles gCD 25min 800V.csv",
                        nrows=17)

COLUMN_NAMES = [c[7:] for c in before_raw.columns if "Sample" in c]

before_data = process_raw(before_raw)
before_data["Average"] = before_data.apply(average_wavelength_function, axis=1)

after_data = process_raw(after_raw)
after_data["Average"] = after_data.apply(average_wavelength_function, axis=1)

average_data = pd.DataFrame()
average_data.index = before_data.index
average_data["Before for Mean"] = before_data["Average"]
average_data["Before for SEM"] = before_data["Average"]
average_data["After for Mean"] = after_data["Average"]
average_data["After for SEM"] = after_data["Average"]

for i in range(1,13):
    average_data.loc[str(i)] = average_data.apply(
        lambda x: wells_stats_function(x, i))

stats_data = average_data.iloc[-11:]
stats_data["x"] = CONCENTRATIONS.copy()

fig, ax = plt.subplots(1,1,figsize=[5,4])

jbplot.plotdf(ax, stats_data[:-1],
              marktype="c",
              type_start=2,
              y = ["Before for Mean"],
              y_sig = ["Before for SEM"],
              name_list = ["Before $\gamma$-CD"])

ax.set_xlabel("Volume of SUVs Added /  $\mu$L")
ax.set_ylabel("Batch-Normalized Fluorescence")

jbplot.save(fig, "Fluorescence Quenching on Vesicles")
