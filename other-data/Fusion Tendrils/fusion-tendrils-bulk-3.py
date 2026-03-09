# Jonathan Bostock
# 30 Aug 2023

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
##Define some useful functions

def get_fret(row, name, blank_dict):

    d = np.subtract(row[name + " D"],blank_dict["D"])
    a = np.subtract(row[name + " A"],blank_dict["A"])

    return np.divide(a,np.add(d,a))

# Gets the fret from a row
def get_adj_fret(row, name, input_dict=None):

    if input_dict == None:
        return np.average([row[name + " {}".format(i+1)] for i in range(3)])

    average = np.average([row[name + " {}".format(i+1)] for i in range(3)])

    initial = input_dict[name]
    final = input_dict["Ctrl"]

    return np.divide(np.subtract(average,initial),np.subtract(final,initial))

def get_fret_se(row,name,input_dict):

    initial = input_dict[name]
    final = input_dict["Ctrl"]

    item_list = [row[name + " {}".format(i+1)] for i in range(3)]

    proc_list = [np.divide(np.subtract(i,initial),np.subtract(final,initial))]

    return stats.sem(proc_list)


before_df = (pd.read_csv("2023-08-30 Bulk Tendrils Before 900V.csv",
                        header=1,
                        skiprows = lambda x: x>4)
             .drop(columns=["Time (min).{}".format(n) for n in range(1,36)]))

tc_df = (pd.read_csv("2023-08-30 Bulk Tendrils Timecourse 900V.csv",
                    header=1,
                    skiprows = lambda x: x>62)
         .drop(columns=["Time (min).{}".format(n) for n in range(1,36)]))

##Make some initial lists
condition_list = ["Standard +IM","Standard -IM","Truncated +IM","Truncated -IM"]

total_list = []
for c in condition_list + ["Ctrl"]:
    for i in range(3):
        total_list.append(c + " {}".format(i+1))

name_list = ["Blank", "Ctrl"] + condition_list

#Get a dict of blanks for later use
blanks = {}

for d_or_a in ["D","A"]:
    for i in range(3):

        name_string = "{} {}".format(1+i,d_or_a)
        blanks[name_string] = np.average(before_df["Blank "+name_string])

    blanks[d_or_a] = np.average([blanks["{} {}".format(1+i,d_or_a)] for i in range(3)])

#Get a dict of FRETs before
before_proc_df = pd.DataFrame()

for column in total_list:
    before_proc_df[column] = before_df.apply(lambda row: get_fret(row, column, blanks),
                                             axis=1)

for condition in condition_list + ["Ctrl"]:

    before_proc_df[condition] = before_proc_df.apply(lambda row: get_adj_fret(row, condition),
                                                     axis=1)

before_dict = {column:  np.average(before_proc_df[column]) for column in condition_list + ["Ctrl"]}

#Process our main dataframe
proc_df = pd.DataFrame()

for column in total_list:

    proc_df[column] = tc_df.apply(lambda row: get_fret(row, column, blanks),
                                  axis=1)

proc_df["Time"] = tc_df["Time (min)"]

for condition in condition_list:

    proc_df[condition] = proc_df.apply(lambda row: get_adj_fret(row, condition,
                                                                input_dict=before_dict),
                                       axis=1)
    proc_df[condition + " SE"] = proc_df.apply(lambda row: get_fret_se(row, condition,
                                                                       input_dict=before_dict),
                                               axis=1)

factor_dict = {}
factor_dict["Standard Tendrils"] = np.average(proc_df.loc[58:63,"Standard -IM"])/np.average(proc_df.loc[58:63,"Standard +IM"])
factor_dict["Truncated Tendrils"] = np.average(proc_df.loc[58:63,"Truncated -IM"])/np.average(proc_df.loc[58:63,"Truncated +IM"])



#Plot things
fig_1, ax_1 = plt.subplots(1, 1, figsize=[4,4])

jbplot.plotdf(ax_1,
              proc_df,
              x="Time",
              y=condition_list,
              y_sig = [c + " SE" for c in condition_list],
              name_list = ["Standard +IM", "Standard -IM", "Truncated +IM", "Truncated -IM"],
              plot_type = "line")

ax_1.legend(frameon=False)
ax_1.set_ylabel("FRET")
ax_1.set_xlabel("Time / min")

jbplot.save(fig_1,
            "Truncated Tendrils Test 2")


fig_2, ax_2 = plt.subplots(1,1,figsize=[4,4])

jbplot.barchart(ax_2,
                factor_dict)

jbplot.save(fig_2,
            "Truncated Tendril Slowdown 2")

