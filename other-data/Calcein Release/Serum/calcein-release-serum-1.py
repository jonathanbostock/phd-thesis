#Jonathan Bostock
#21-Jun-2023
#Analysing leakage in serum with and without a dsDNA brush

import numpy as np
import scipy.optimize as op
import scipy.stats as stats
from scipy.interpolate import splrep, BSpline
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

#Set up some useful variables
file_1 = "2023-06-21 Calcein Leak In Serum 1 750v.csv"
file_2 = "2023-06-21 Calcein Leak In Serum +Triton 750v.csv"

column_names = ["-DNA, Buffer 1", "K1", "-DNA, Serum 1", "K2", "+DNA, Serum 1", "K3", "Triton 1",
                "K4", "-DNA, Buffer 2", "K5", "-DNA, Serum 2", "K6", "+DNA, Serum 2", "K7", "Triton 2",
                "K8", "-DNA, Buffer 3", "K9", "-DNA, Serum 3", "K10", "+DNA, Serum 3", "K11", "Triton 3", "K12"]

condition_names = ["-DNA, Buffer", "-DNA, Serum", "+DNA, Serum"]
well_names = []

for i in range(1,4):
    for condition in condition_names:
        well_names.append(condition + " " + str(i))

parameter_names = ["-DNA, Buffer Mean", "-DNA, Buffer SEM",
                   "-DNA, Serum Mean", "-DNA, Serum SEM",
                   "+DNA, Serum Mean", "+DNA, Serum SEM"]


triton_names = ["Triton " + str(n) for n in range(1,4)]

###Load our csvs
raw_df = (pd.read_csv(file_1,
                      header=1,
                      names=column_names)
          .drop(labels=["K" + str(n) for n in range(1,13)],
                axis=1)[:900]
          .astype(float))

triton_df = (pd.read_csv(file_2,
                         header=1,
                         names=column_names)
             .drop(labels=["K" + str(n) for n in range(1,13)],
                   axis=1)[:3]
             .astype(float))


proc_df = raw_df.drop(labels=triton_names,
                      axis=1)
proc_df["Time"]=range(0,900)
proc_df = proc_df.set_index("Time")

###Process our data

#Smooth a dataframe
smooth_df = copy.deepcopy(proc_df)
for name in well_names:
    tck = splrep(range(0,900),smooth_df[name],s=50)
    smooth_df[name] = [BSpline(*tck)(t) for t in range(0,900)]

#Calculate initial and final fluorescence values
triton_fluorescence_values = (triton_df.drop(labels=triton_names,
                                             axis=1)
                              .apply(lambda c: np.average(c[0:3]),
                                     axis = "rows"))
initial_fluorescence_values = (proc_df.apply(lambda c: c[0],
                                             axis = "rows"))

triton_fluorescence_dict = {name: value for name, value in zip(well_names,triton_fluorescence_values)}

initial_fluorescence_dict = {name: value for name, value in zip(well_names,initial_fluorescence_values)}

#Make a normalized dataframe
normalized_df = copy.deepcopy(proc_df)
for name in ["-DNA, Serum 1",
             "-DNA, Serum 2",
             "-DNA, Serum 3",
             "+DNA, Serum 1",
             "+DNA, Serum 2",
             "+DNA, Serum 3"]:
    data = normalized_df[name].tolist()
    norm_data = [np.divide(d - initial_fluorescence_dict[name],
                           triton_fluorescence_dict[name] - initial_fluorescence_dict[name]) for d in data]
    tck = splrep(range(0,900),norm_data,s=0)
    normalized_df[name] = [BSpline(*tck)(t) for t in range(0,900)]

normalized_df["Time"] = range(0,900)

#Calculate final leak values
final_leak_values = {}
for name in well_names:

    final_leak_values[name] = normalized_df[name].tolist()[-1]

def params_from_row(row):

    params = []

    row_dict = {well_name: value for well_name, value in zip(well_names, row)}

    for condition_name in condition_names:
        name_list = [condition_name + " " + str(i) for i in range(1,4)]
        #Get the leakage %age
        values = [np.divide(row_dict[name] - initial_fluorescence_dict[name],
                            triton_fluorescence_dict[name] - initial_fluorescence_dict[name]) for name in name_list]
        #Append these values to our parameters
        params.append(np.average(values))
        params.append(stats.sem(values))

    return params

processed_dict = {index:    params_from_row(row) for index, row in smooth_df.iterrows()}

parameter_df = pd.DataFrame.from_dict(processed_dict,
                                      orient="index",
                                      columns=parameter_names)
parameter_df["Time"] = range(900)

###Plot everything

fig_1, ax_1 = plt.subplots(figsize=[4,4])
fig_2, ax_2 = plt.subplots(figsize=[4,4])

jbplot.plotdf(ax_1,
              parameter_df[:180],
              x="Time",
              y=["-DNA, Buffer Mean",
                 "-DNA, Serum Mean",
                 "+DNA, Serum Mean"],
              y_sig = ["-DNA, Buffer SEM",
                       "-DNA, Serum SEM",
                       "+DNA, Serum SEM"],
              name_list=["-DNA, Buffer",
                     "-DNA, Serum",
                     "+DNA, Serum"],
              plot_type="line")
ax_1.legend()
ax_1.set_xlabel("Time / min")
ax_1.set_ylabel("Calcein Release Fraction")

jbplot.plotdf(ax_2,
              normalized_df,
              x="Time",
              y=["-DNA, Serum 1",
                 "-DNA, Serum 2",
                 "-DNA, Serum 3",
                 "+DNA, Serum 1",
                 "+DNA, Serum 2",
                 "+DNA, Serum 3"],
              name_list = ["-DNA, Serum 1",
                           "-DNA, Serum 2",
                           "-DNA, Serum 3",
                           "+DNA, Serum 1",
                           "+DNA, Serum 2",
                           "+DNA, Serum 3"],
              plot_type="line")
ax_2.set_xlabel("Time / min")
ax_2.set_ylabel("Calcein Release Fraction")
ax_2.legend()

fig_1.savefig("Calcein Release Serum 1.png",
              format="png",
              dpi=300,
              bbox_inches="tight")

fig_2.savefig("Calcein Release Serum 1 Raw Release.png",
              format="png",
              dpi=300,
              bbox_inches="tight")
