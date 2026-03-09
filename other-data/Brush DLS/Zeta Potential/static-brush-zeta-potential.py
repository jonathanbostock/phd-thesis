#Jonathan Bostock
#26-Apr-2023
#Analysing the data for the static, dsDNA brushes

import numpy as np
import scipy.optimize as op
import scipy.stats as stats
import matplotlib.pyplot as plt
import lmfit
import copy
import pandas as pd
import os
import sys
import math

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

##That was a lot of faff, but we're done with boilerplate stuff

df_30bp_raw_1 = pd.read_table("2023-06-27 30bp Zeta Potential.tsv").set_index("Sample Name")


dict_30bp_1 = {}

data_to_store = ["Peak 1 Mean by Intensity ordered by area (nm)","Zeta Potential (mV)"]

for index, row in df_30bp_raw_1.iterrows():
    for data_name in data_to_store:
        if math.isnan(row[data_name]) == False:
            #Try to shove the data into my horrible dictionary of dictionaries
            try:
                dict_30bp_1[index][data_name] = row[data_name]
            except:
                #If it doesn't work, do this instead
                dict_30bp_1[index] = {}
                dict_30bp_1[index][data_name] = row[data_name]

control_30bp_1 = dict_30bp_1["Ext 1 Ctrl 2"]

keys = [k for k in dict_30bp_1.keys()]

for key in keys:

    if "Ctrl" in key:
        del dict_30bp_1[key]
    else:
        inner_dict = dict_30bp_1[key]

        for inner_key in inner_dict.keys():
            inner_dict[inner_key] -= control_30bp_1[inner_key]

        dict_30bp_1[key] = inner_dict

data_30bp_clean_1 = pd.DataFrame.from_dict(dict_30bp_1,
                                           orient="index")

data_30bp_clean_1["DNA:Lipid Ratio"] = [1/20,1/100,1/500,1/2000]

fig_1, ax_1 = plt.subplots(figsize=[4,4])

jbplot.plotdf(ax_1,
              data_30bp_clean_1,
              x="DNA:Lipid Ratio",
              y="Zeta Potential (mV)",
              plot_type="scatterline")
ax_1.set_xlabel("DNA:Lipid Ratio")
ax_1.set_ylabel("$\Delta \zeta$ / mV")
ax_1.set_xscale("log")
ax_1.set_xlim(1/5000,1/10)
ax_1.set_ylim(1,-36)

jbplot.save(fig_1, "30bp Zeta Potential")
