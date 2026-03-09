#Jonathan Bostock
#08-August-2023
#Analysing the data for the ATP Aptamer brushes (bad)

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

name_key = "Sample Name"
p1mi_key = "Peak 1 Mean by Intensity ordered by area (nm)"

raw_data = pd.read_csv("2023-08-07 ATP Aptamer Brush.tsv",delimiter="\t")


cond_dict = {"0":       0,
             "1uM":     1e-3,
             "10uM":    1e-2,
             "100uM":   1e-1,
             "1mM":     1e-0,
             "3mM":     3e-0}

cond_list = cond_dict.keys()

processed_dict = {cond: [] for cond in cond_list}
control_dict = {}

#Build a dataframe of processed data minus the controls, which is converted to mean, SE
x_fit_list = []
y_fit_list = []
for key, row in raw_data.iterrows():

    name_list = row[name_key].split()
    size = row[p1mi_key]

    extrusion = name_list[1]
    condition = name_list[2]

    if condition == "Ctrl":
        control_dict[extrusion] = size
    else:
        processed_dict[condition].append(size - control_dict[extrusion])
        x_fit_list.append(cond_dict[condition])
        y_fit_list.append(size - control_dict[extrusion])

processed_data = pd.DataFrame.from_dict(
    {cond:  {"ATP Conc":    cond_dict[cond],
             "Delta D":     np.average(processed_dict[cond]),
             "SE":          stats.sem(processed_dict[cond])} for cond in cond_list}).T

# Fit everything
def linf(x, m, c):
    return m*x + c
model = lmfit.Model(linf)
p = model.make_params(m=0,c=0)
fit = model.fit(data=y_fit_list, x=x_fit_list, params=p)

#Plot everything
fig, ax = plt.subplots(1,1,figsize=[4,4])

jbplot.plotdf(ax,
              processed_data,
              x="ATP Conc",
              y="Delta D",
              y_sig="SE")
jbplot.plotfun(ax,
               lambda x: fit.eval(x=x),
               sig_function = lambda x: fit.eval_uncertainty(x=x)[0],
               x_max=3)
ax.set_xscale("symlog",linthresh = 1e-3)
ax.set_ylim(0,45)
ax.set_xlabel("ATP Concentration / mM")
ax.set_ylabel("$\Delta D$ / nm")

jbplot.save(fig,
            "ATP Aptamer Brush 2")
