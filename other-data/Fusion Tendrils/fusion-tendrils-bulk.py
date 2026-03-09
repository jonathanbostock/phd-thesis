#Jonathan Bostock
#2023-August-08

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

column_list = ["-ve -ve", "+ve +ve",
               "-IM -Blocker", "-IM 2O", "-IM 3O", "-IM 5O",
               "+IM -Blocker","+IM 2O","+IM 3O","+IM 5O",
               "Blank Blank"]

fluorophore_dict = {0:  "Donor",
                    1:  "Acceptor"}

def process_column_name(name):

    try:
        n = int(name[-2:])
    except:
        try:
            n = int(name[-1:])
        except:
            n=0

    if "Time" in name:
        if n == 0:
            return "Time"
        else:
            return "K{}".format(n)
    else:
        row = str(int(np.floor(n/22)) + 1)
        column = column_list[int(np.floor(n/2) % 11)]
        fluorophore = fluorophore_dict[n % 2]
        return " ".join([column, fluorophore, row])


def get_fret(row, name):

    if name == "+IM 5O":

        donor = row["+IM 5O Donor 1"]
        acceptor = row["+IM 5O Acceptor 1"]

    else:
        donor = np.average([row[" ".join([name, "Donor", str(i)])] for i in range(1,4)])
        acceptor = np.average([row[" ".join([name, "Acceptor", str(i)])] for i in range(1,4)])


    donor_adj = donor - np.average([row["Blank Blank Donor " + str(i)] for i in range(1,4)])

    acceptor_adj = acceptor - np.average([row["Blank Blank Acceptor " + str(i)] for i in range(1,4)])

    return np.divide(acceptor_adj,np.add(donor_adj,acceptor_adj))

def get_normalization(row, name, n_average, p_average):

    return np.divide(np.subtract(row[name], p_average),np.subtract(n_average,p_average))

raw_df = (pd.read_csv("2023-08-08 Tendril Blocker Testing 1000V.csv",
                      header=1,
                      skiprows=lambda x: x>80)
          .drop(columns="Unnamed: 132")
          .rename(columns = lambda x: process_column_name(x))
          .drop(columns= ["K{}".format(n) for n in range(1,66)]))

for column in column_list[:-1]:

    raw_df[column] = raw_df.apply(lambda row: get_fret(row,column),axis=1)


proc_df = pd.DataFrame()

n_average = np.average(raw_df.loc[:,"-ve -ve"])
p_average = np.average(raw_df.loc[:,"+ve +ve"])

for column in column_list[2:-1]:

    proc_df[column] = raw_df.apply(lambda row: get_normalization(row, column, n_average, p_average),
                                   axis=1)

proc_df["Time"] = raw_df["Time"]

##Do some calculations of efficacy

blocker_data = {}

for blocker in ["-Blocker", "2O", "3O", "5O"]:

    with_IM = proc_df.loc[21," ".join(["+IM", blocker])]
    without_IM = proc_df.loc[21," ".join(["-IM", blocker])]

    factor = np.divide(without_IM, with_IM)

    blocker_data[blocker] = factor


fig_1, ax_1 = plt.subplots(1, 1, figsize=[4,4])

jbplot.plotdf(ax_1,
              proc_df,
              x = "Time",
              y = column_list[2:-1],
              name_list = column_list[2:-1],
              plot_type = "line")

ax_1.legend()

fig_2, ax_2 = plt.subplots(1,1, figsize=[4,4])

jbplot.barchart(ax_2,
                blocker_data)

jbplot.save(fig_1,
            "Tests of IM Tendrils in Bulk")
jbplot.save(fig_2,
            "Efficicency of Blockers")
