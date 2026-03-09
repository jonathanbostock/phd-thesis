#Jonathan Bostock
#2023-August-10

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

#Now we can import my module
import jbplot

column_list = ["-MAC -Brush","+MAC -Brush","-MAC +Brush","+MAC +Brush"]

def process_column_name(name):

    try:
        n = int(name[-2:])
    except:
        try:
            n = int(name[-1:])
        except:
            n = 0

    if "Time" in name:
        if n == 0:
            return "Time"
        else:
            return "K{}".format(n)

    row = np.floor(float(n)/4)
    column = n % 4
    return " ".join([column_list[column],str(int(row+1))])

#Unfortunately we have two different types of column so this stuff has to go on
def process_column_name_2(name):


    try:
        n = int(name[-2:])
    except:
        try:
            n = int(name[-1:])
        except:
            n = 0

    m = n - int(2*np.floor(n/6)+1)

    if m in [4,5,6,7]:
        m = 11 - m

    if "Wavelength" in name:
        if m == 0:
            return "Wavelength"
        else:
            return "K{}".format(m)

    row = np.floor(float(m)/4)
    column = m % 4
    return " ".join([column_list[column],str(int(row+1))])

def get_adjusted_f(row, name, initial_f, final_f):

    f = row[name]

    return (f - initial_f)/(final_f - initial_f)

drop_columns = ["K{}".format(n) for n in range(1,12)]

def get_sem(row, name):

    return stats.sem([row.loc[name + " " + str(i)] for i in range(1,4)])

def get_average(row, name):

    return np.average([row.loc[name + " " + str(i)] for i in range(1,4)])

columns_for_before_df = []

for i in range(3):
    columns_for_before_df += [12*i + j + 2 for j in range(8)]

##Get our data
before_raw_df = (pd.read_csv("2023-08-10 MAC Test Before 900V.csv",
                             header = 1,
                             usecols = columns_for_before_df)
                 .rename(columns = lambda x: process_column_name_2(x))
                 .drop(columns = drop_columns))
c8_raw_df = (pd.read_csv("2023-08-10 MAC Test C8 900V.csv",
                         header=1,
                         usecols = columns_for_before_df)
             .rename(columns=lambda x:process_column_name_2(x))
             .drop(columns = drop_columns))
timecourse_raw_df = (pd.read_csv("2023-08-10 MAC Timecourse 900V.csv",
                                 header = 1,
                                 skiprows = lambda x: x>62)
                     .drop(columns="Unnamed: 24")
                     .rename(columns = lambda x: process_column_name(x))
                     .drop(columns = drop_columns))
triton_raw_df = (pd.read_csv("2023-08-10 MAC Triton 900V.csv",
                             header = 1,
                             skiprows = lambda x: x>5)
                 .drop(columns="Unnamed: 24")
                 .rename(columns = lambda x: process_column_name(x))
                 .drop(columns = drop_columns))
triton_spectra_raw_df = pd.read_csv("2023-08-10 MAC Triton Spectra 900V.csv")

#Process our data
release_df = pd.DataFrame()

release_c8_dict = {}
release_60m_dict = {}

for column_name in column_list:

    c8_release = []

    for i in range(3):

        name = column_name + " " + str(i+1)
        initial_f = before_raw_df.loc[5,name]
        final_f = np.average(triton_raw_df.loc[:,name])
        release_df[name] = timecourse_raw_df.apply(
            lambda row: get_adjusted_f(row, name, initial_f, final_f),
            axis=1)

        c8_f = c8_raw_df.loc[5,name]
        c8_release.append(np.divide(np.subtract(c8_f,initial_f),np.subtract(final_f,initial_f)))


    release_df[column_name + " Mean"] = release_df.apply(
        lambda row: get_average(row, column_name),
        axis=1)

    release_df[column_name + " SE"] = release_df.apply(
        lambda row: get_sem(row, column_name),
        axis=1)

    release_c8_dict[column_name] = (np.average(c8_release),
                                         stats.sem(c8_release))

    release_60m_dict[column_name] = (release_df.loc[41,column_name + " Mean"],
                                                 release_df.loc[41,column_name + " SE"])


release_df["Time"] = timecourse_raw_df["Time"]

#Do a scipy ANOVA

with_brush = [release_df.loc[41,"+MAC +Brush {}".format(i+1)] for i in range(3)]
without_brush = [release_df.loc[41,"+MAC -Brush {}".format(i+1)] for i in range(3)]

f_value, p_value = stats.f_oneway(with_brush, without_brush)

#Plot things

fig, ax = plt.subplots(1,1,figsize=[4,4])

jbplot.plotdf(ax,
              release_df,
              x="Time",
              y=[c + " Mean" for c in column_list],
              y_sig=[c+ " SE" for c in column_list],
              name_list = column_list,
              plot_type ="line")
ax.legend()
ax.set_ylabel("Normalized Lysis")
ax.set_xlabel("Time / min")

fig_2, ax_2 = plt.subplots(1,1,figsize=[4,4])

jbplot.barchart(ax_2, release_c8_dict, rotate_labels=True)
ax_2.set_ylabel("Lysis after Adding C8")

fig_3, ax_3 = plt.subplots(1,1,figsize=[4,4])

jbplot.barchart(ax_3, release_60m_dict, rotate_labels=True)
ax_3.set_ylabel("Lysis after 60 minutes")

jbplot.save(fig,
            "MAC + Brushes Timecourse")
jbplot.save(fig_2,
            "C8 Lysis")
jbplot.save(fig_3,
            "60 Minute Lysis", keep_box=True)
