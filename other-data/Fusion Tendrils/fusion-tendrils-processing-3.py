#Jonathan Bostock
#2023-Jul-10

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

file_after = "2023-07-10 IM Tendril Fusion After 700v.csv"
file_before = "2023-07-10 IM Tendril Fusion Before 700v.csv"
file_tc = "2023-07-10 IM Tendril Fusion Timecourse 700v.csv"

columns = ["K0", "1A 8.0 D", "K1", "1A 8.0 A",
           "K2", "2A 8.0 D", "K3", "2A 8.0 A",
           "K4", "3A 8.0 D", "K5", "3A 8.0 A",
           "K6", "1A 5.5 D", "K7", "1A 5.5 A",
           "K8", "2A 5.5 D", "K9", "2A 5.5 A",
           "K10", "3A 5.5 D", "K11", "3A 5.5 A", "K12"]

conditions = ["1A 8.0", "2A 8.0", "3A 8.0",
              "1A 5.5", "2A 5.5", "3A 5.5"]

extra_columns = ["Ctrl D", "K13", "Ctrl A","K14"]
kill_columns = ["K" + str(n) for n in range(13)]


df_tc = (pd.read_csv(file_tc,
                     header=1,
                     skiprows = lambda x: x>303,
                     names=columns)
         .drop(columns=kill_columns)
         .astype(float))

df_after = (pd.read_csv(file_after,
                        header=1,
                        skiprows = lambda x: x> 5,
                        names=columns)
            .drop(columns=kill_columns)
            .astype(float))

df_before = (pd.read_csv(file_before,
                        header=1,
                        skiprows = lambda x: x> 5,
                        names=columns+extra_columns)
            .drop(columns=kill_columns+["K13","K14"])
            .astype(float))

controls = {"D":    np.average(df_before["Ctrl D"]),
            "A":    np.average(df_before["Ctrl A"])}

def fret(row, name):
    D = row[name + " D"]
    A = row[name + " A"]
    D_adj = np.subtract(D, controls["D"])
    A_adj = np.subtract(A, controls["A"])

    return np.divide(A_adj,np.add(D_adj,A_adj))

def lerp(start,end,value):

    return np.divide(np.subtract(value, start), np.subtract(end, start))

for c in conditions:

    df_tc[c + " Fret"] = df_tc.apply(lambda x: fret(x, c),
                           axis=1)

    df_after[c + " Fret"] = df_after.apply(lambda x: fret(x,c),
                                 axis=1)

    fret_0 = df_tc[c + " Fret"][0]

    fret_f = np.average(df_after[c + " Fret"])

    df_tc[c] = df_tc.apply(lambda row: lerp(fret_0, fret_f, row[c + " Fret"]),
                                 axis=1)

df_tc["Time"] = range(301)

##Plot stuff
fig, ax = plt.subplots(figsize=[4,4])

jbplot.plotdf(ax,
              df_tc[0:181],
              x="Time",
              y=conditions,
              plot_type="line",
              name_list=conditions)
ax.set_ylabel("Fusion")
ax.set_xlabel("Time / min")
ax.legend()

jbplot.save(fig,
            "IM Tendrils 2")
