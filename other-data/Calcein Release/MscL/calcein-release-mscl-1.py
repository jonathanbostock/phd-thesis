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
column_names = ["K1", "0 nM sPLA", "K2", "0.05 nM sPLA",
                "K3", "0.5 nM sPLA", "K4", "5 nM sPLA",
                "K5", "2.5 uM DNA", "K6", "10% Virkon", "K7"]

#Load our csvs
#Timecourse
raw_df = (pd.read_csv("2023-04-18 Calcein Leak Assay Timeline.csv",
                      header=1,
                      names=column_names)
          .drop(labels=["K" + str(n) for n in range(1,8)],
                axis=1)
          .drop(labels=range(91,391),
                axis=0)
          .astype(float))

proc_df = pd.DataFrame()
proc_df["Time / min"] = range(91)

cond_name_list =["0 nM sPLA", "0.05 nM sPLA", "0.5 nM sPLA", "5 nM sPLA", "2.5 uM DNA"]

#Get an estimate of release using r = (f = f0)/f0
for cond in cond_name_list:
    column = raw_df[cond]
    proc_df[cond] = column.apply(lambda x: x/column[0])
    #Also load up our processed dataframe

fig, ax = plt.subplots(figsize=[4,4])

jbplot.plotdf(ax, proc_df,
              x="Time / min",
              y=cond_name_list[0:4],
              plot_type = "line",
              gradient = True,
              name_list=cond_name_list)
jbplot.plotdf(ax, proc_df,
              x ="Time / min",
              y ="2.5 uM DNA",
              plot_type = "line",
              linecode_override=2,
              name_list = ["2.5 uM DNA"])
ax.legend(loc=(1.05,0.4))
ax.set_xlabel("Time / min")
ax.set_ylabel("$f / f_{t=0}$")
ax.set_title("a",fontweight="bold",
             loc="left")

fig.savefig("Calcein Release MscL 1.png",
            format="png",
            bbox_inches="tight",
            dpi=600)
