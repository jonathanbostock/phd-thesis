#Jonathan Bostock
#2023-May-02

import numpy as np
import scipy.optimize as op
import scipy.stats as stats
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
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
df_columns = ["Time / min",
              "- D", "K1", "- A", "K2",
              "+ D", "K3", "+ A", "K4"]

raw_df = (pd.read_csv("2023-04-28 Vesicle Fusion Timecourse 700v.csv",
                      header=1,
                      skiprows=lambda x: x > 1203,
                      names=df_columns)
          .drop(columns=["K" + str(n) for n in range(1,5)])
          .astype(float))

raw_df_2 = (pd.read_csv("2023-04-28 Vesicle Fusion Triton1000v.csv",
                        header=1,
                        names=["K1", "-", "K2", "+", "K3"],
                        skiprows=lambda x: x > 83)
            .drop(columns=["K" + str(n) for n in range(1,4)])
            .astype(float))

#Make a list of columns
cond_list = ["+", "-"]

#This will hold processed data
proc_df = pd.DataFrame(columns=["Time / min", "- F.E.", "+ F.E.", "- Mixing",
                                "+ Mixing"])

#Initialize a dataframe for our smoothed data
smooth_df = pd.DataFrame(columns=raw_df.columns)
smooth_df["Time / min"] = raw_df["Time / min"]

#This will hold a processed spectrum (from the tritoned stuff)
proc_df_2 = pd.DataFrame()
proc_df_2["Wavelength / nm"] = range(520,601)

#Define a couple of functions to make life easier
def f_e(d, a):
    return a/(a + d)

def norm(x, a, b):
    return (x-a)/(b-a)

fret_triton = {}
fret_init = {}

for c in cond_list:
    smooth_df[c+" D"] = gaussian_filter(raw_df[c+" D"],2)
    smooth_df[c+" A"] = gaussian_filter(raw_df[c+" A"],2)

    proc_df_2[c] = raw_df_2[c]
    fret_triton[c] = f_e(np.average(proc_df_2[c][0:221]),
                         np.average(proc_df_2[c][60:81]))
    fret_init[c] = f_e(smooth_df[c + " D"][0],
                       smooth_df[c + " A"][0])

    proc_df[c + " F.E."]= gaussian_filter(
        raw_df.apply(
            lambda row: f_e(row[c+" D"],
                            row[c+" A"]),
            axis=1), 2)

    proc_df[c + " Mixing"] = proc_df.apply(
        lambda row: norm(row[c + " F.E."],fret_init[c],fret_triton[c]),
        axis=1)

proc_df["Time / min"] = np.arange(0,600.5,0.5)

trunc_df = proc_df.where(proc_df["Time / min"] <= 120)
fig, ax = plt.subplots(figsize=[4,4])

jbplot.plotdf(ax, trunc_df,
              x="Time / min",
              y=["- Mixing","+ Mixing"],
              name_list=["- Tendrils", "+ Tendrils"],
              plot_type="line",
              linecode_override=0)
ax.set_xlabel("Time / min")
ax.set_ylabel("Lipid Mixing")
ax.legend()
ax.set_title("a",
             fontweight="bold",
             loc="left")

fig.savefig("Fusion Tendrils 1.png",
            format="png",
            bbox_inches="tight",
            dpi=600)
