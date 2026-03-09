# Jonathan Bostock
# 24-Aug-2023
# pH responses of regular brushes
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import lmfit
import pandas as pd
import os
import sys

# Find this file's location
# This file will only run if in a folder which is in "PhD Work"
abspath=os.path.abspath(__file__)
dname=os.path.dirname(abspath)

# Get the grandparent directory
gp_dir_name = os.path.dirname(dname)
sys.path.insert(0, gp_dir_name + "/Python Modules")

# Now we can import my module
import jbplot
import jbbuffer

# Change to the directory this file is in
os.chdir(dname)

P1MI_key = "Peak 1 Mean by Intensity ordered by area (nm)"
name_key = "Sample Name"

pH_data = pd.read_csv("pH_calibration.tsv",sep="\t")
raw_data = pd.read_csv("2023-08-24 pH Response 1+2.tsv",sep="\t")

# Calculate our pH
pH_data["Mean"] = pH_data.apply(
    lambda row: np.average([row["pH 1"],
                            row["pH 2"],
                            row["pH 3"]]),
    axis=1)

pH_data["Standard Error"] = pH_data.apply(
    lambda row: stats.sem([row["pH 1"],
                           row["pH 2"],
                           row["pH 3"]]),
   axis=1)

def get_pH(hcl):

    if type(hcl) == int:
        return pH_data.iloc[hcl]["Mean"]
    else:
        hcl_floor = int(np.floor(hcl))
        hcl_ceiling = hcl_floor + 1
        hcl_lerp_factor = hcl - np.floor(hcl)

        pH_floor = get_pH(hcl_floor)
        pH_ceiling = get_pH(hcl_ceiling)

        return pH_ceiling*hcl_lerp_factor + pH_floor * (1-hcl_lerp_factor)

length_dict = {"0bp":    38,
               "15bp":   53,
               "30bp":   68,
               "42bp":   80}

condition_dict = {"{}HCl".format(i*3):  get_pH(i*3) for i in range(6)}

data_dict = {length:   {condition: []
                        for condition in condition_dict.values()}
             for length in length_dict.values()}

control_dict = {}

for key, row in raw_data.iterrows():

    name = row[name_key]
    P1MI = row[P1MI_key]

    extrusion = name.split()[1]

    if "Ctrl" in name:
        control_dict[extrusion] = P1MI

    else:
        length = length_dict[name.split()[2]]
        condition = condition_dict[name.split()[3]]

        data_dict[length][condition].append(P1MI - control_dict[extrusion])

proc_data = pd.DataFrame()

proc_data["pH"] = condition_dict.values()

for length_name, length in length_dict.items():

    proc_data[length_name + " Mean"] = [np.mean(data_dict[length][condition])
                                        for condition in condition_dict.values()]
    proc_data[length_name + " SE"] = [stats.sem(data_dict[length][condition])
                                      for condition in condition_dict.values()]

## Plot everything

fig, ax = plt.subplots(1, 1, figsize=[4,4])

jbplot.plotdf(ax,
              proc_data,
              x = "pH",
              y = [l + " Mean" for l in length_dict.keys()],
              name_list = ["{} bp".format(l) for l in length_dict.values()],
              y_sig = [l + " SE" for l in length_dict.keys()],
              gradient = True,
              gradient_code = 4,
              gradient_map = lambda x: length_dict[x.split()[0]],
              plot_type = "scatter")

jbplot.plotdf(ax,
              proc_data,
              x = "pH",
              y = [l + " Mean" for l in length_dict.keys()],
              #y_sig = [l + " SE" for l in length_dict.keys()],
              gradient = True,
              gradient_code=4,
              gradient_map = lambda x: length_dict[x.split()[0]],
              plot_type = "line")

ax.set_xlabel("pH")
ax.set_ylabel("$\Delta D$ / nm")
ax.set_xlim(8.1, 5.4)
ax.set_ylim(-1, 51)

jbplot.nice_legend(ax)

jbplot.save(fig,
            "pH Response of Standard Brushes")
