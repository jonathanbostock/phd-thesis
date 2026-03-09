#Jonathan Bostock
#15-Jun-2023
#Making some calibration curves for my lipid concentration

import numpy as np
import scipy.optimize as op
import scipy.stats as stats
import matplotlib.pyplot as plt
import lmfit
import copy
import pandas as pd
import os
import sys

#Find this file's location
#This file will only run if in a folder which is in "PhD Work"
abspath=os.path.abspath(__file__)
dname=os.path.dirname(abspath)

#Get the grandparent directory
gp_dir_name = os.path.dirname(dname)
sys.path.insert(0, gp_dir_name + "/Python Modules")

#Now we can import my module
import jbplot

#Change to the directory this file is in
os.chdir(dname)

file_list = ["2023-06-15 Calibration Rh-PE 500v 555ex 10nmslit.csv",
             "2023-06-15 Calibration Rh-PE 600v 555ex 10nmslit.csv",
             "2023-06-15 Calibration Rh-PE 700v 555ex 10nmslit.csv"]

voltage_list = [500, 600, 700]
file_data = {}
processed_samples = {}
conc_list = [0,0.5,
             0.2,0.1,0.05,
             0.02,0.01,0.005]

sample_names = [str(conc) + " " + str(n+1) for conc in conc_list for n in range(3)]


for file_name, voltage in zip(file_list, voltage_list):
    raw_data = pd.read_csv(file_name,
                           header=1,
                           index_col=0)[22:23].T

    blanks = {}
    samples = {}
    adjusted_samples = {}

    for index, row in raw_data.iterrows():

        emission = float(row[0])
        name = index.split()[1]
        sample_set = name[0]
        conc = conc_list[int(name[1])-1]

        if conc == 0:
            blanks[sample_set] = emission
        else:
            try:
                samples[sample_set] += [(conc, emission)]
            except:
                samples[sample_set] = [(conc, emission)]

    for key in samples.keys():

        sample_list = samples[key]
        blank = blanks[key]
        adjusted_sample_dict = {s[0]:   s[1] - blank for s in sample_list}

        adjusted_samples[key] = adjusted_sample_dict


    adjusted_samples_transposed = {conc:    [adj_s[1][conc] for adj_s in adjusted_samples.items()]
                                   for conc in conc_list[1:]}

    processed_samples[voltage] = pd.DataFrame.from_dict(
        {str(conc) + " " + str(voltage):    [conc,
                                             voltage,
                                             np.mean(adjusted_samples_transposed[conc]),
                                             stats.sem(adjusted_samples_transposed[conc])]
         for conc in conc_list[1:]},
        columns=["Conc",
                 "Voltage",
                 "Mean E597",
                 "SE E597"],
        orient="index")


all_samples = pd.concat([processed_samples[voltage] for voltage in voltage_list])

#Do some fitting

linear_params = {}

for voltage in voltage_list:

    voltage_data = all_samples[all_samples["Voltage"] == voltage]
    data = voltage_data[voltage_data["Mean E597"] < 900]

    y_data = data["Mean E597"]
    x_data = data["Conc"]

    linear_regression = stats.linregress(x_data, y=y_data)

    linear_params[voltage] = (linear_regression.slope, linear_regression.intercept)
    print(linear_params[voltage])

def linear_function(x, voltage):
    params = linear_params[voltage]
    return x * params[0] + params[1]

#Plot everything
fig_1, ax_1 = plt.subplots(figsize=[4,4])

jbplot.plotdf(ax_1,
              all_samples,
              x="Conc",
              y="Mean E597",
              split="Voltage",
              y_sig="SE E597",
              name="Voltage",
              name_map = lambda x: str(x) + " V",
              gradient=True)
jbplot.plot2dfun(ax_1,
                 linear_function,
                 n=1000,
                 x_min=0,
                 x_max=0.5,
                 second_var = voltage_list,
                 gradient=True)
ax_1.set_xlabel("Lipid conc / mg mL$^{-1}$")
ax_1.set_ylabel("Fluorescence at 597 nm")
ax_1.set_ylim(-1,1010)
ax_1.legend()
ax_1.set_xscale("log")

fig_1.savefig("Lipid conc calibration.png",
              format="png",
              dpi=300,
              bbox_inches="tight")
