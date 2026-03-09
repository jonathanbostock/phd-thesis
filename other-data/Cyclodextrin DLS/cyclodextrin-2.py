# Jonathan Bostock
# 12-Oct-2023
# Cyclodextrins 2

import pandas as pd
import jbplot
import matplotlib.pyplot as plt

raw_data = pd.read_csv("2023-10-12 Dextrins.csv",
                       index_col="Sample Name").loc[:,"Peak 1 Mean by Intensity ordered by area (nm)"]

delta_D_data = []

for conc in [10, 5, 2]:
    for time in ["Before", "1min", "30min"]:

        time_numeric = {"Before":   "Before\nAddition",
                        "1min":     "1",
                        "30min":    "30"}[time]

        condition_key = "{} Dextrin {}".format(conc, time)
        ctrl_key = "Ctrl {}".format(condition_key)
        DNA_key = "DNA {}".format(condition_key)

        delta_D_data.append([conc, time_numeric, raw_data.loc[DNA_key] - raw_data.loc[ctrl_key]])

delta_D_df = pd.DataFrame(data = delta_D_data, columns = ["Dextrin Concentration", "Time / min", "Delta D"])

fig, ax = plt.subplots(1, 1, figsize= [4,4])

jbplot.plotdf(ax,
              delta_D_df,
              x="Time / min",
              y="Delta D",
              split = "Dextrin Concentration",
              gradient = True,
              name = "Dextrin Concentration",
              name_map = lambda x: "{} mM MBCD".format(x),
              plot_type = "scatter")

jbplot.plotdf(ax,
              delta_D_df,
              x="Time / min",
              y="Delta D",
              split = "Dextrin Concentration",
              gradient = True,
              plot_type = "line")

ax.set_xlabel("Time / min")
ax.set_ylabel("Delta D / nm")
ax.legend()

jbplot.nice_legend(ax)
jbplot.save(fig, "MBCD And Brushes")
