### BCA Assay for Ahl

import jbplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

raw_data = pd.read_csv("2024-02-06 BCA Assay Ahl 1.txt",
                       header=1)

raw_data_array = np.array(raw_data)
average_array = np.average(raw_data_array, axis=0)
stdev_array = np.std(raw_data_array, axis=0)

x_array = [2, 1.5, 1, 0.75, 0.5, 0.25, 0.125, 0.025, 0]

raw_fit_data = raw_data_array[...,:-2].flatten()

polynomial_fit = np.polyfit(average_array[:-2], x_array, 3)


def interpolate(y_point, x_data, y_data):

    for x_less, x_more, y_less, y_more in zip(x_data[:-1], x_data[1:], y_data[:-1], y_data[1:]):

        if y_less <= y_point:

            delta_x = x_more - x_less
            delta_y = y_more - y_less

            return x_less + delta_x * (y_point - y_less) / delta_y

    raise Exception("Interpolation failed")

def polyfit(x, coefficients):

    terms = [c * x ** i for i, c in enumerate(coefficients[::-1])]

    return np.sum(terms)

abs_range = np.arange(0, max(average_array+0.1),0.1)

conc_fit = np.array([polyfit(a, polynomial_fit) for a in abs_range])

mut_conc = polyfit(average_array[-2], polynomial_fit)
wt_conc = polyfit(average_array[-1], polynomial_fit)

fig, ax = plt.subplots(1,1,figsize=[4,4])

jbplot.scatter(ax, x_array, average_array[:-2], stdev_array[:-2],label = "Calibration")
jbplot.plotline(ax, conc_fit, abs_range)
jbplot.scatter(ax, [mut_conc], average_array[-2:-1], colorcode=2, markcode=2, label=f"K237C: {mut_conc:.3g} mg/mL")
jbplot.scatter(ax, [wt_conc], average_array[-1:], colorcode=3, markcode=3, label=f"wt: {wt_conc:.3g} mg/mL")

ax.set_xlabel("Concentration / mg/mL")
ax.set_ylabel("Absorbance at 562 nm")
ax.legend()
jbplot.nice_legend(ax)

jbplot.save(fig, "2024-02-06 BCA Calibration Curve")
