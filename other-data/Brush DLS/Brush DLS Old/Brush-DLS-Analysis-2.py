#24-Jan-2023

#Import and analyse data from DLS of brushes

import numpy as np
import scipy
import matplotlib.pyplot as plt
import jbplot
import os

##First experiment, 4 hr timecourse of decoration

#Import the first set of data
abspath = os.path.abspath(__file__)
path = os.path.dirname(abspath)
os.chdir(path)

raw_data_1 = np.genfromtxt("2023-01-24-Brush-DLS-Timecourse.tsv",
                         delimiter="\t").transpose()

control_tc = [raw_data_1[0,1:4],
                      raw_data_1[4,1:4],
                      raw_data_1[5,1:4]]
thirty_bp_tc = [raw_data_1[0,5:10],
                        raw_data_1[4,5:10],
                        raw_data_1[5,5:10]]

#Do Some Fancy Fitting

def expfit(x, final_y, y_change, x_coefficient):

    return final_y + y_change * np.exp(x*x_coefficient)

fit_data = scipy.optimize.curve_fit(expfit, thirty_bp_tc[0], thirty_bp_tc[1],
                         sigma=thirty_bp_tc[2],p0=[200,-50,-0.1])
params = fit_data[0]

#Make a fitted curve
x_range = np.arange(0,240,0.1)
thirty_bp_fit = expfit(x_range, params[0], params[1], params[2])

#Plot Figure 1
figure_1, ax_1 = plt.subplots()

figure_1.suptitle("4 Hour Timecourse of Decoration, with 68 bp dsDNA")

jbplot.scatterset(ax_1, [control_tc[0],thirty_bp_tc[0]],
                  [control_tc[1],thirty_bp_tc[1]],
                  y_sig_vect_set = [control_tc[2],thirty_bp_tc[2]],
                  name_set = ["400:1 Lipid:DNA",
                              "Control, no DNA"])
jbplot.plotline(ax_1, x_range,thirty_bp_fit,colorcode=1,linecode=1)

ax_1.set_xlabel("Time / min")
ax_1.set_ylabel("Z-average / nm")
ax_1.legend()

figure_1.savefig("2023-01-24 DLS Time Course.png",format="png",dpi=300)


##Second Experiment, comparison between 1 hr and 5 days in fridge

#Import the second set of data
raw_data_2 = np.genfromtxt("2023-01-24-Brush-DLS-5-days.tsv",
                           delimiter = "\t")

length_list = ["80 bp", "68 bp",
               "53 bp", "38 bp"]

high_conc_5_days = raw_data_2[7:0:-2,1:3]
low_conc_5_days = raw_data_2[8:1:-2,1:3]

figure_2, (ax_2, ax_3) = plt.subplots(1,2,sharey=True)

figure_2.suptitle("Effects of 5 days at 5 " + chr(176) + "C")

jbplot.scatterset(ax_2, [["1 Hr", "5 Days"]], high_conc_5_days,
                  name_set = length_list, line=True,
                  linecode_override=0)
jbplot.scatterset(ax_3, [["1 Hr", "5 Days"]], low_conc_5_days,
                  name_set = length_list, line=True,
                  linecode_override=1,marktype="l")


ax_2.legend()
ax_3.legend()

ax_2.set_title("200:1 Lipid:DNA Ratio")
ax_3.set_title("800:1 Lipid:DNA Ratio")

ax_2.set_ylabel("Z-average / nm")

figure_2.savefig("2023-01-24 DLS 5 days.png",format="png",dpi=300)
