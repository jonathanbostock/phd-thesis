### Analyse MBCD Data
### Jonathan Bostock

import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import jbplot
import os

P1MI_KEY = "Peak 1 Mean by Intensity ordered by area (nm)"
NAME_KEY = "Sample Name"
TIME_KEY = "Measurement Start Date And Time"

def power_fit(x, k, n):

    return np.multiply(np.power(x,n),k)

def exp_decay(t, k):

    return np.multiply(35, np.exp(- np.multiply(t + 60,k)))

def meta_exp_decay(input_array, k, n):

    t = input_array[...,0]
    c = np.divide(input_array[...,1],1000)

    # A exp(-(t-t_lag) * k * c)

    return np.multiply(35, np.exp(- np.multiply(np.add(t,60), np.multiply(k, np.power(c,n)))))

def exp_decay_fitted(t, i, param_array):

    A, k = param_array[i]

    return exp_decay(t, A, k)

def flatten(array):

    list_to_return = []

    for item in array:
        for i in item:
            list_to_return.append(i)

    return list_to_return

all_files = [f for f in os.listdir() if (f[-4:] == ".txt" and f[:3] != "BAD")]
data_dict = {}

for file_name in all_files:

    conc, _, _, number = file_name[:-4].split(" ")
    conc = float(conc)
    number = int(number)

    data_frame = (pd.read_csv(file_name,
                              sep="\t",
                              usecols=[NAME_KEY,
                                       P1MI_KEY,
                                       TIME_KEY]))

    DNA_array = np.array(data_frame[P1MI_KEY][:5])
    no_DNA_array = np.array(data_frame[P1MI_KEY][5:])

    delta_D_array = DNA_array - no_DNA_array

    if conc not in data_dict.keys():
        data_dict[conc] = []

    data_dict[conc].append(delta_D_array)

time_array = [i * 195 for i in range(5)]

average_vect = []
sterr_vect = []
conc_vect = [0.5, 1, 2, 5]
curve_data_vect = []
all_data_vect = []
all_conditions_vect = []

## Calculate standard error and fit individual curves
for conc in conc_vect:

    data = data_dict[conc]

    average_vect.append(np.average(data, axis=0))
    sterr_vect.append(sp.stats.sem(data, axis=0))

    flat_data = flatten(data)
    flat_conditions = [[conc, t] for t in time_array] * len(data)

    curve_data_vect.append(sp.optimize.curve_fit(
        exp_decay,
        time_array * len(data),
        flat_data,
        p0 = [0.01])[0][0])

    all_data_vect += flat_data
    all_conditions_vect += flat_conditions


## Fit the big meta-curve
meta_curve_data = sp.optimize.curve_fit(
    meta_exp_decay,
    np.array(all_conditions_vect),
    np.array(all_data_vect),
    p0 = [1, 2])

# Fit the constants
rate_constant_results = sp.stats.linregress(
    np.log(np.divide(conc_vect,1000)),
    y=np.log(curve_data_vect))

power_fit_n = rate_constant_results.slope
power_fit_k = np.exp(rate_constant_results.intercept)

fig, ax = plt.subplots(1,1,figsize=[4,4])
jbplot.scatterset(ax, [time_array], average_vect, y_sig_vect_set = sterr_vect,
                  name_list = ["0.5 mM", "1 mM", "2 mM", "5 mM"],
                  gradient=True,
                  gradient_vals = range(4),
                  gradient_code = 4)
jbplot.plot2dfun(ax, lambda x, y: exp_decay(x, curve_data_vect[y]),
                 second_var = range(4),
                 x_min = 0,
                 x_max = 800,
                 n = 100,
                 gradient=True,
                 gradient_code=4)

ax.set_xlabel("Time / s")
ax.set_ylabel("$\Delta D$ / nm")

jbplot.nice_legend(fig)
jbplot.save(fig, "MBCD Brush Removal Plot")

fig_2, ax_2 = plt.subplots(1,1,figsize=[4,4])
jbplot.scatter(ax_2, np.divide(conc_vect,1000), curve_data_vect, colorcode=1)
jbplot.plotfun(ax_2, lambda x: power_fit(x, power_fit_k, power_fit_n),
               x_min = 0.4/1000,
               x_max = 6/1000,
               n=100,
               log=True,
               colorcode = 1,
               label="y = ${k:.3g} x ^{{{n:.3g}}}$".format(k = power_fit_k, n = power_fit_n))
ax_2.set_yscale("log")
ax_2.set_xscale("log")
ax_2.set_xlabel("MBCD Concentration / M")
ax_2.set_ylabel("$k$ for DNA Brush Removal / s$^{-1}$")

jbplot.nice_legend(ax_2)
jbplot.save(fig_2, "MBCD Rate Constants")
