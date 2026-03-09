##29 Jan 2023
#Jonathan Bostock

import jbplot
import numpy as np
import scipy.optimize as op
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

#Import data
abspath=os.path.abspath(__file__)
dname=os.path.dirname(abspath)
os.chdir(dname)

raw_data = np.genfromtxt("2023-01-27-Brush-DLS-4.tsv", delimiter = "\t")

raw_data_t = raw_data.transpose()

conc_vector = [200, 1000, 4000]
log_conc_space_vector = np.arange(2,4,0.001)
conc_space_vector = np.vectorize(lambda x: 10**x)(log_conc_space_vector)
inv_conc_vector = [1/200, 1/1000, 1/4000]
length_vector = [38, 53, 68, 80]
length_name_vector = ["38 bp", "53 bp",
                      "68 bp", "80 bp"]
length_space_vector = np.arange(0,80,1)


delta_avgs = [raw_data_t[8][4:7],
              raw_data_t[8][7:10],
              raw_data_t[8][10:13],
              raw_data_t[8][13:16]]

delta_stes = [raw_data_t[9][4:7],
                    raw_data_t[9][7:10],
                    raw_data_t[9][10:13],
                    raw_data_t[9][13:16]]

#Define a function to fit
def fit_function(x , y_max, x_half):

    return y_max * x / (x + x_half)

def linear_fit(x, coefficient):

    return x * coefficient


fit_38_bp = op.curve_fit(fit_function, inv_conc_vector, delta_avgs[0],
                         sigma=delta_stes[0])
fit_53_bp = op.curve_fit(fit_function, inv_conc_vector, delta_avgs[1],
                         sigma=delta_stes[1])
fit_68_bp = op.curve_fit(fit_function, inv_conc_vector, delta_avgs[2],
                         sigma=delta_stes[2])
fit_80_bp = op.curve_fit(fit_function, inv_conc_vector, delta_avgs[3],
                         sigma=delta_stes[3])

#Make fitted curves
curve_set = []
curve_set.append(np.vectorize(lambda x: fit_function(1/x,
                                                     fit_38_bp[0][0],
                                                     fit_38_bp[0][1])
                             )(conc_space_vector))
curve_set.append(np.vectorize(lambda x: fit_function(1/x,
                                                     fit_53_bp[0][0],
                                                     fit_53_bp[0][1])
                             )(conc_space_vector))
curve_set.append(np.vectorize(lambda x: fit_function(1/x,
                                                     fit_68_bp[0][0],
                                                     fit_68_bp[0][1])
                             )(conc_space_vector))
curve_set.append(np.vectorize(lambda x: fit_function(1/x,
                                                     fit_80_bp[0][0],
                                                     fit_80_bp[0][1])
                             )(conc_space_vector))



y_max_vector = [fit_38_bp[0][0], fit_53_bp[0][0],
                fit_68_bp[0][0], fit_80_bp[0][0]]
y_max_ste_vector = [fit_38_bp[1][0,0], fit_53_bp[1][0,0],
                fit_68_bp[1][0,0], fit_80_bp[1][0,0]]

x_half_vector = [fit_38_bp[0][1], fit_53_bp[0][1],
                fit_68_bp[0][1], fit_80_bp[0][1]]
x_half_ste_vector = [fit_38_bp[1][1,1], fit_53_bp[1][1,1],
                fit_68_bp[1][1,1], fit_80_bp[1][1,1]]

y_max_fit = op.curve_fit(linear_fit, length_vector, y_max_vector)

y_max_curve = np.vectorize(lambda x: x * y_max_fit[0][0])(length_space_vector)

#Plotting
#Figure 1

figure_1, (ax_1, ax_2) = plt.subplots(1,2,width_ratios=[3,2],figsize=(8,5))

jbplot.scatterset(ax_1, [conc_vector], delta_avgs, y_sig_vect_set=delta_stes,
                  name_set = length_name_vector)
jbplot.plotlineset(ax_1, [conc_space_vector], curve_set)

ax_1.set_xlabel("DNA:Lipid Ratio")
ax_1.set_xscale("log")
ax_1.invert_xaxis()
ax_1.set_ylabel("Change in Z-average / nm")
ax_1.legend()

#Axes 2, where we plot some fitted parameters
ax_2.plot(length_space_vector,y_max_curve,
          marker="none",color="hotpink")
ax_2.scatter(length_vector,y_max_vector,
             marker=">",linewidths=1,color="hotpink",
             edgecolors="black",zorder=2)

ax_2.set_xlabel("DNA Brush Length")
ax_2.set_ylabel("Maximum Delta Z-Average / nm")

figure_1.suptitle("DLS Analysis of DNA Brushes")

figure_1.savefig("2023-01-27 Brush DLS 4.png",format="png",dpi=600)
