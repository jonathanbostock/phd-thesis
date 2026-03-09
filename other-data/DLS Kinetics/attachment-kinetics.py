###Jonathan Bostock
##07-Jun-2023

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

P1MI_key = "Peak 1 Mean by Intensity ordered by area (nm)"
name_key = "Sample Name"

file_data = pd.read_csv("2023-06-07 Attachment Timecourse.csv")

raw_data_cleaned = (file_data.where(file_data["Quality Indicator"]=="GoodData")
                    .dropna())

#Empty dictionaries
extrusion_controls = {}
time_dict = {}

for index, row in raw_data_cleaned.iterrows():
    #Get the name
    name = row[name_key]
    extrusion = " ".join(name.split()[0:2])
    time = " ".join(name.split()[2:])

    if "Cont" in name or "Ctrl" in name:
        extrusion_controls[extrusion] = float(row[P1MI_key])
    else:
        delta_d = float(row[P1MI_key]) - extrusion_controls[extrusion]
        try:
            time_dict[time].append(delta_d)
        except:
            time_dict[time] = [delta_d]

time_keys = time_dict.keys()
time_processed_dict = {i:[float(i.split()[0]),
                          np.average(time_dict[i]),
                          stats.sem(time_dict[i])] for i in time_keys}

time_dataframe = (pd.DataFrame
                  .from_dict(time_processed_dict, orient="index",
                             columns= ["Time/min",
                                       "Mean",
                                       "Standard Error"]))

###Now we do some fun fitting

def exp_func(x, y_end, k):

    return (1-np.exp(x*k)) * (y_end)

exp_model = lmfit.Model(exp_func,
                        name="Exponential fit")
exp_params = exp_model.make_params(y_end=30,k=-0.01)

#Here we do the fitting
exp_fit = exp_model.fit(time_dataframe["Mean"],
                        exp_params,
                        x=time_dataframe["Time/min"],
                        weights=time_dataframe["Standard Error"].apply(
                            lambda x: 1/x))

#Summarize the data
exp_param_data = exp_fit.summary()
best_k = -exp_param_data["best_values"]["k"]
half_life = np.log(2)/best_k

fit_label_text = "$k$ = {k:.4f} / min".format(k=best_k)


###Plot everything
fig_1, ax_1 = plt.subplots(figsize=[4.0,4.0])

jbplot.plotdf(ax_1,
              time_dataframe,
              x="Time/min",
              y="Mean",
              y_sig="Standard Error",
              name_list=["Experimental Data"])
jbplot.plotfun(ax_1, lambda x: exp_fit.eval(x=x),
               x_min=0,
               x_max=92,
               sig_function = lambda x: exp_fit.eval_uncertainty(x=x)[0],
               label=fit_label_text)
ax_1.set_xlabel("Time / min")
ax_1.set_ylabel("$\Delta D$ / nm")
ax_1.set_xlim(-2,92)
ax_1.set_ylim(-1,41)

ax_1.legend(loc="best", frameon=False)

jbplot.save(fig_1,
            "Kinetics of Brush Attachment")
