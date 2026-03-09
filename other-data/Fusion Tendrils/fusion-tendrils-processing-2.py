#Jonathan Bostock
#2023-Jul-03

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
              "-IM pH 8 D", "K0", "-IM pH 8 A", "K1",
              "-IM pH 7 D", "K2", "-IM pH 7 A", "K3",
              "-IM pH 6 D", "K4", "-IM pH 6 A", "K5",
              "+IM pH 8 D", "K6", "+IM pH 8 A", "K7",
              "+IM pH 7 D", "K8", "+IM pH 7 A", "K9",
              "+IM pH 6 D", "K10", "+IM pH 6 A", "K11"]
extra_columns = ["Blank D", "K12", "Blank A", "K13"]

conditions = ["-IM pH 8","-IM pH 7","-IM pH 6",
              "+IM pH 8","+IM pH 7","+IM pH 6"]

kill_columns = ["K" + str(n) for n in range(12)]

file_before = "2023-07-03 IM Tendril Fusion Before 800v.csv"
file_after = "2023-07-03 IM Tendril Fusion After 800v.csv"
file_tc = "2023-07-03 IM Tendril Fusion Timecourse 800v.csv"

before_df = (pd.read_csv(file_before,
                         header=1,
                         skiprows = lambda x: x > 5,
                         names=df_columns)
             .drop(columns= kill_columns)
             .astype(float))

tc_df = (pd.read_csv(file_tc,
                     header=1,
                     skiprows = lambda x: x>303,
                     names=df_columns)
         .drop(columns = kill_columns)
         .astype(float))

after_df = (pd.read_csv(file_after,
                        header=1,
                        skiprows = lambda x: x > 5,
                        names=df_columns+extra_columns)
            .drop(columns = kill_columns + ["K12","K13"])
            .astype(float))

blanks = {"D":  np.average(after_df["Blank D"]),
          "A":  np.average(after_df["Blank A"])}

def fret(d, a):
    return np.divide(a,np.add(d,a))

def lerp(s, e, v):
    return np.divide(np.subtract(v,s),np.subtract(e,s))

def fret_blanks(d,a,blanks):
    return fret(np.subtract(d,blanks["D"]),np.subtract(a,blanks["A"]))

def fret_row(row, name, banks):
    return fret_blanks(row[name + " D"],row[name + " A"], blanks)

for c in conditions:
    #Get the "Before" fret value
    before_fret = tc_df[0:1].apply(lambda row: fret_row(row, c, blanks),
                                   axis=1)
    #Get the "After" fret value
    after_df[c] = after_df.apply(lambda row: fret_row(row, c, blanks),
                                 axis=1)
    after_fret = np.average(after_df[c])
    #Lerp and stuff
    tc_df[c] = tc_df.apply(lambda row: lerp(before_fret,after_fret,fret_row(row, c, blanks)),
                           axis=1)


fig_1, ax_1 = plt.subplots(figsize=[4,4])

jbplot.plotdf(ax_1,
              tc_df[0:61],
              x="Time / min",
              y=conditions,
              name_list=conditions,
              plot_type="line")
ax_1.set_xlabel("Time / min")
ax_1.set_ylabel("Fusion Efficiency")
ax_1.legend()

jbplot.save(fig_1,
            "2023-07-03 IM Tendril Test 1")
