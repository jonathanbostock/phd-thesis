##29 Jan 2023
#Jonathan Bostock


import numpy as np
import scipy.optimize as op
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
import sys

#Find out the file's location
abspath=os.path.abspath(__file__)
dname=os.path.dirname(abspath)

#Get the grandparent directory
gp_dir_name = os.path.dirname(dname)
sys.path.insert(0, gp_dir_name + "/Python Modules")

#Now we can import my module
import jbplot

#Change to the directory this file is in
os.chdir(dname)

#That was a lot of faff, but we're done with boilerplate stuff
#First off load our data:


raw_data = np.genfromtxt("2023-01-27-Brush-DLS-5.tsv", delimiter = "\t")
raw_data_2 = np.genfromtxt("2023-02-08-Brush-DLS-Others.tsv", delimiter =
                          "\t")
raw_data_3 = np.genfromtxt("2023-02-22-Brush-DLS-Charged-Liposomes.tsv",
                           delimiter="\t")

raw_data_t = raw_data.transpose()
raw_data_2_t = raw_data_2.transpose()
raw_data_3_t = raw_data_3.transpose()


conc_vector = [20,100, 500, 2000]
log_conc_space_vector = np.arange(1,4,0.001)
inv_conc_space_vector = np.vectorize(lambda x: 10**(-x))(log_conc_space_vector)
inv_conc_vector = [1/conc for conc in conc_vector]
length_vector = [38, 53, 68, 80]
length_name_vector = ["38 bp", "53 bp",
                      "68 bp", "80 bp"]
length_space_vector = np.arange(0,82,1)
length_name_vector_2 = ["68 bp, Junction", "20bp, ssDNA", "38bp, ssDNA"]
lipid_proportion_vector = [0, 0.1, 0.25, 0.50]

delta_avgs = [raw_data_t[9][7:11],
              raw_data_t[9][11:15],
              raw_data_t[9][15:19],
              raw_data_t[9][19:23]]

delta_stes = [raw_data_t[10][7:11],
              raw_data_t[10][11:15],
              raw_data_t[10][15:19],
              raw_data_t[10][19:23]]

delta_avgs_2 = [raw_data_2_t[8][8:12],
                raw_data_2_t[8][12:16],
                raw_data_2_t[8][16:20]]
delta_stes_2 = [raw_data_2_t[9][8:12],
                raw_data_2_t[9][12:16],
                raw_data_2_t[9][16:20]]

delta_avgs_3 = [raw_data_3_t[8][5:9]]
delta_stes_3 = [raw_data_3_t[9][5:9]]

#Define a function to fit
def fit_function(x , y_max, x_half):

    return y_max * x / (x + x_half)

def linear_fit(x, coefficient):

    return x * coefficient

def quad_fit(x, a, b, c):

    return a + (x * b) + (x**2 * c)

def flat_fit(x, a):

    return a

#Actually do the fitting
fit_38_bp = op.curve_fit(fit_function, inv_conc_vector, delta_avgs[0],
                         sigma=delta_stes[0], p0=[40,0.01])
fit_53_bp = op.curve_fit(fit_function, inv_conc_vector, delta_avgs[1],
                         sigma=delta_stes[1], p0=[40,0.01])
fit_68_bp = op.curve_fit(fit_function, inv_conc_vector, delta_avgs[2],
                         sigma=delta_stes[2], p0=[40,0.01])
fit_80_bp = op.curve_fit(fit_function, inv_conc_vector, delta_avgs[3],
                         sigma=delta_stes[3], p0=[40,0.01])

fit_junct = op.curve_fit(fit_function, inv_conc_vector[1:4],
                         delta_avgs_2[0][1:4],
                         sigma=delta_stes_2[0][1:4], p0 =[40,0.001])
fit_20_bp_ss = op.curve_fit(fit_function, inv_conc_vector,
                            delta_avgs_2[1],
                            sigma=delta_stes_2[1], p0 =[40, 0.001])
fit_38_bp_ss = op.curve_fit(fit_function, inv_conc_vector,
                            delta_avgs_2[2],
                            sigma=delta_stes_2[2],
                             p0 =[40, 0.001])

fit_charge = op.curve_fit(quad_fit, lipid_proportion_vector,
                          delta_avgs_3[0],
                          sigma=delta_stes_3[0],
                          p0 = [1, 1, 1])
fit_charge_2 = op.curve_fit(flat_fit, lipid_proportion_vector,
                            delta_avgs_3[0],
                            sigma=delta_stes_3[0],
                            p0 = [1])


#Make fitted curves
curve_set = []
curve_set.append(np.vectorize(lambda x: fit_function(x,
                                                     fit_38_bp[0][0],
                                                     fit_38_bp[0][1])
                             )(inv_conc_space_vector))
curve_set.append(np.vectorize(lambda x: fit_function(x,
                                                     fit_53_bp[0][0],
                                                     fit_53_bp[0][1])
                             )(inv_conc_space_vector))
curve_set.append(np.vectorize(lambda x: fit_function(x,
                                                     fit_68_bp[0][0],
                                                     fit_68_bp[0][1])
                             )(inv_conc_space_vector))
curve_set.append(np.vectorize(lambda x: fit_function(x,
                                                     fit_80_bp[0][0],
                                                     fit_80_bp[0][1])
                             )(inv_conc_space_vector))

curve_set_2=[]
curve_set_2.append(np.vectorize(lambda x: fit_function(x,
                                                       fit_junct[0][0],
                                                       fit_junct[0][1])
                               )(inv_conc_space_vector[1000:4000]))
curve_set_2.append(np.vectorize(lambda x: fit_function(x,
                                                       fit_20_bp_ss[0][0],
                                                       fit_20_bp_ss[0][1])
                               )(inv_conc_space_vector))
curve_set_2.append(np.vectorize(lambda x: fit_function(x,
                                                       fit_38_bp_ss[0][0],
                                                       fit_38_bp_ss[0][1])
                               )(inv_conc_space_vector))

curve_set_3=[]
curve_set_3.append(np.vectorize(lambda x: quad_fit(x,
                                                   fit_charge[0][0],
                                                   fit_charge[0][1],
                                                   fit_charge[0][2])
                               )(np.arange(0,0.5,0.01)))
curve_set_3.append(np.vectorize(lambda x: flat_fit(x,
                                                   fit_charge_2[0][0])
                               )(np.arange(0,0.5,0.01)))

#Put together a vector of y_max values
y_max_vector = [fit_38_bp[0][0], fit_53_bp[0][0],
                fit_68_bp[0][0], fit_80_bp[0][0]]
y_max_ste_vector = [fit_38_bp[1][0,0], fit_53_bp[1][0,0],
                fit_68_bp[1][0,0], fit_80_bp[1][0,0]]

y_max_vector_2 = [fit_junct[0][0],
                  fit_20_bp_ss[0][0],
                  fit_38_bp_ss[0][0]]
y_max_ste_vector_2 = [fit_junct[1][0,0],
                      fit_20_bp_ss[1][0,0],
                      fit_38_bp_ss[1][0,0]]

#Put this together for completeness, it probably won't be used
x_half_vector = [fit_38_bp[0][1], fit_53_bp[0][1],
                fit_68_bp[0][1], fit_80_bp[0][1]]
x_half_ste_vector = [fit_38_bp[1][1,1], fit_53_bp[1][1,1],
                fit_68_bp[1][1,1], fit_80_bp[1][1,1]]

x_half_vector_2 = [fit_junct[0][1],
                   fit_20_bp_ss[0][1],
                   fit_38_bp_ss[0][1]]
x_half_ste_vector_2 = [fit_junct[1][1,1],
                       fit_20_bp_ss[1][1,1],
                       fit_20_bp_ss[1][1,1]]

DNA_distance_vector = np.vectorize(lambda x: 1.07457 * np.sqrt(0.7/x))(x_half_vector)
DNA_distance_vector_2 = np.vectorize(lambda x: 1.07457 * np.sqrt(0.7/x))(
                                    x_half_vector_2)

#Fit a curve of y_max values
y_max_fit = op.curve_fit(linear_fit, length_vector, y_max_vector)

y_max_curve = np.vectorize(lambda x: x * y_max_fit[0][0])(length_space_vector)


#Plotting
#Figure 1
figure_1, ((ax_1, ax_2), (ax_3, ax_4)) = plt.subplots(2,2,
                                                      figsize=(8,7.5))
plt.subplots_adjust(hspace=0.25)

#Ax 1, where we plot our main data
jbplot.scatterset(ax_1, [inv_conc_vector], delta_avgs, y_sig_vect_set=delta_stes,
                  name_set = length_name_vector, gradient=True)
jbplot.plotfun(ax_1,
               lambda x: fit_function(x,
                                      fit_38_bp[0][0],
                                      fit_38_bp[0][1]),
               1/10000, 1, log=True)


ax_1.set_xlabel("DNA:Lipid Ratio")
ax_1.set_xscale("log")
ax_1.set_ylabel("$\Delta D$ / nm")
ax_1.legend()
ax_1.set_title("a",loc="left",fontweight="bold")

#Ax 2, where we plot some fitted parameters
ax_2.plot(length_space_vector,y_max_curve,
          marker="none",color=jbplot.colors[1],
          label="$y$ = {coeff:.2f}$x$".format(coeff=y_max_fit[0][0]))
ax_2.scatter(length_vector,y_max_vector,
             marker=jbplot.marks["f"][0],
             s=jbplot.marksizes[jbplot.marks["f"][0]],
             linewidths=1,color=jbplot.colors[1],
             label="Fitted $\Delta D_{max}$\nfrom plot a",
             edgecolors="black",zorder=2)
ax_2.legend()
ax_2.set_ylim(-1,51)
ax_2.set_xlim(-2,82)
ax_2.set_xlabel("DNA Brush Length / bp")
ax_2.set_ylabel("$\Delta D_{max}$ / nm")
ax_2.set_title("b", loc="left",fontweight="bold")

#Ax 3, where we include data about other types of brush

jbplot.scatterset(ax_3, [inv_conc_vector], delta_avgs_2,
                  y_sig_vect_set=delta_stes_2,
                  name_set=length_name_vector_2,
                  type_start=2)
jbplot.plotlineset(ax_3, [inv_conc_space_vector[1000:4000],
                          inv_conc_space_vector],
                   curve_set_2[0:2],
                   type_start=2)
ax_3.annotate("ex.", (inv_conc_vector[0],delta_avgs_2[0][0]+1),
              fontsize="small")

ax_3.legend()
ax_3.set_xscale("log")
ax_3.set_ylim(-1, 51)
ax_3.set_xlabel("DNA:Lipid Ratio")
ax_3.set_ylabel("$\Delta D$ / nm")
ax_3.set_title("c",loc="left",fontweight="bold")

#Ax 4, where we plot the effect of charged lipids
jbplot.scatterset(ax_4, [lipid_proportion_vector], delta_avgs_3,
                  y_sig_vect_set=delta_stes_3,
                  type_start=5,
                  name_set=["68 bp, 100:1 Lipid:DNA"])
curve_name_1  = "$y$ = {c:.0f}$x^2$ + {b:.1f}$x$ + {a:.1f}".format(a = fit_charge[0][0],
                                                        b = fit_charge[0][1],
                                                        c = fit_charge[0][2])
#curve_name_2 = "$y$ = {a:.1f}".format(a = fit_charge_2[0][0])
jbplot.plotline(ax_4, np.arange(0,0.5,0.01), curve_set_3[0],
                label = curve_name_1,
                colorcode=5,linecode=1)
#jbplot.plotline(ax_4, np.arange(0,50,0.01), curve_set_3[1],
#                label = curve_name_2,
#                colorcode=2, linecode=1)

ax_4.set_xlabel("Fraction of POPG in POPC")
ax_4.set_ylabel("$\Delta D$ / nm")
ax_4.set_ylim(-1,51)
ax_4.set_title("d",loc="left",fontweight="bold")
ax_4.legend()

figure_1.savefig("2023-01-27 Brush DLS Final Graph.png",
                 format="png",dpi=600,bbox_inches="tight")
