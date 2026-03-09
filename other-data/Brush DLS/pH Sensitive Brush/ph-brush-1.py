#Jonathan Bostock
#09-May-2023
#Calculating a pH calibration curve for my buffer

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import lmfit
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
import jbbuffer

#Change to the directory this file is in
os.chdir(dname)

P1MI_key = "Peak 1 Mean by Intensity ordered by area (nm)"
name_key = "Sample Name"

#Import our data
pH_data = pd.read_csv("pH_calibration.tsv",sep="\t")
neg_control_data = pd.read_csv("2023-07-27 POPC HCl DLS Data.tsv", delimiter = "\t")
im_1_data = pd.read_csv("2023-05-11 I Motif Brush DLS Data.csv")
im_2_data = pd.read_csv("2023-06-08 I Motif 2 Brush DLS Data.csv")
main_data = pd.read_csv("2023-08-01 HCl 30bp + IM.tsv", delimiter = "\t")


neg_control_data = (neg_control_data[neg_control_data["Quality Indicator"]=="GoodData"]
                    .set_index(name_key))
im_1_data = (im_1_data[im_1_data["Quality Indicator"]=="GoodData"]
             .set_index(name_key))
im_2_data = (im_2_data[im_2_data["Quality Indicator"]=="GoodData"]
             .set_index(name_key))
main_data = (main_data[main_data["Quality Indicator"]=="GoodData"]
             .set_index(name_key))


pH_data["Mean"] = pH_data.apply(
    lambda row: np.average([row["pH 1"],
                            row["pH 2"],
                            row["pH 3"]]),
    axis=1)

pH_data["Standard Error"] = pH_data.apply(
    lambda row: stats.sem([row["pH 1"],
                           row["pH 2"],
                           row["pH 3"]]),
   axis=1)

def get_pH(hcl):

    if type(hcl) == int:
        return pH_data.iloc[hcl]["Mean"]
    else:
        hcl_floor = int(np.floor(hcl))
        hcl_ceiling = hcl_floor + 1
        hcl_lerp_factor = hcl - np.floor(hcl)

        pH_floor = get_pH(hcl_floor)
        pH_ceiling = get_pH(hcl_ceiling)

        return pH_ceiling*hcl_lerp_factor + pH_floor * (1-hcl_lerp_factor)


neg_control_dict = {3*n:    [] for n in range(6)}

for index, row in neg_control_data.iterrows():
    index_list = index.split()
    extrusion = " ".join(index_list[0:2])
    hcl = " ".join(index_list[3:4])
    diameter = row[P1MI_key]

    delta_D = diameter - neg_control_data[neg_control_data.index == extrusion + " 0 HCl"][P1MI_key][0]
    #Pandas for some godforsaken reason makes you do a third index bit. I hate pandas.

    neg_control_dict[int(index_list[2])].append(delta_D)

neg_control_processed = pd.DataFrame.from_dict({3*n:  [get_pH(3*n),
                                                       "No DNA",
                                                       np.average(neg_control_dict[3*n]),
                                                       stats.sem(neg_control_dict[3*n])] for n in range(6)},
                                               orient="index",
                                               columns = ["pH", "Condition","Mean", "Standard Error"])

no_im_dict = {3*n:    [] for n in range(6)}
im_dict = {3*n:       [] for n in range(6)}

for index, row in main_data.iterrows():
    index_list = index.split()
    extrusion = " ".join(index_list[0:2])
    condition = " ".join(index_list[2:4])

    hcl = " ".join(index_list[4:6])
    diameter = row[P1MI_key]

    delta_D = diameter - main_data[main_data.index == extrusion + " Ctrl"][P1MI_key][0]

    #Pandas for some godforsaken reason makes you do a third index bit. I hate pandas.

    if condition != "Ctrl":
        if condition == "30 bp":
            no_im_dict[int(index_list[4])].append(delta_D)
        else:
            im_dict[int(index_list[4])].append(delta_D)

no_im_list = []
im_list = []
neg_control_list = []

for n in range(6):
    for item in no_im_dict[3*n]:
        no_im_list.append((get_pH(3*n), item))

    for item in im_dict[3*n]:
        im_list.append((get_pH(3*n), item))

    for item in neg_control_dict[3*n]:
        neg_control_list.append((get_pH(3*n), item))

no_im_processed = pd.DataFrame.from_dict({3*n:  [get_pH(3*n),
                                                 "30 bp",
                                                 np.average(no_im_dict[3*n]),
                                                 stats.sem(no_im_dict[3*n])] for n in range(6)},
                                         orient="index",
                                         columns = ["pH","Condition", "Mean", "Standard Error"])

im_processed = pd.DataFrame.from_dict({3*n:  [get_pH(3*n),
                                              "I-motif",
                                              np.average(im_dict[3*n]),
                                              stats.sem(im_dict[3*n])] for n in range(6)},
                                      orient="index",
                                      columns = ["pH", "Condition", "Mean", "Standard Error"])


main_data_processed = pd.concat([neg_control_processed,no_im_processed,im_processed])

##Do some linear regression

linear_regression = stats.linregress([item[0] for item in neg_control_list], y=[item[1] for item in neg_control_list])

linear_regression_string = "Linear Regression, {s}".format(
    s = (lambda p: "n.s." if p > 0.05 else "s.")(linear_regression.pvalue))

#Initialize some useful data structures
#cond_indices_1 = ["IM 1 " + str(n) + " HCl" for n in range(16)]
#cond_indices_2 = ["IM 2 " + str(n) + " HCl" for n in range(16)] + ["NoIM 0 HCl", "NoIM 12 HCl"]
#cond_dict_1 = {ci:float(ci.split()[-2]) for ci in cond_indices_1}
#cond_dict_2 = {ci:float(ci.split()[-2]) for ci in cond_indices_2}
#data_by_cond_1 = {ci:[] for ci in cond_indices_1}
#data_by_cond_2 = {ci:[] for ci in cond_indices_2}
#extrusion_names= set([name[0:5] for name in im_1_data.index])
#extrusion_controls = {name:[] for name in extrusion_names}
#
#samples_1 = []
#samples_2 = []
#
##Process our first dataframe
#for index, row in im_1_data.iterrows():
#
#    extrusion_name = index[0:5]
#    if "IM" in index:
#        sample_name = index[6:8] + " 1" + index[8:]
#    else:
#        sample_name = index[6:]
#
#    if "Ctrl" in sample_name or "Control" in sample_name:
#        extrusion_controls[extrusion_name].append(row[P1MI_key])
#    else:
#        samples_1.append([extrusion_name + " " + sample_name, row[P1MI_key]])
#
#for name, diameter in samples_1:
#
#    extrusion_name = name[0:5]
#    condition = name[6:]
#
#    adj_diameter = diameter - np.average(extrusion_controls[extrusion_name])
#
#    data_by_cond_1[condition].append(adj_diameter)
#
#data_params_by_cond_1 = {i:[cond_dict_1[i],
#                            np.average(data_by_cond_1[i]),
#                            stats.sem(data_by_cond_1[i])] for i in cond_indices_1}
#
#proc_data_1 = pd.DataFrame.from_dict(data_params_by_cond_1,
#                                     orient="index",
#                                     columns=["Vol HCl",
#                                              "Mean",
#                                              "Standard Error"])
##Process our second dataframe
#for index, row in im_2_data.iterrows():
#
#    extrusion_name = index[0:5]
#    if "IM" in index and not "NoIM" in index:
#        sample_name = index[6:8] + " 2" + index[8:]
#    else:
#        sample_name = index[6:]
#
#    if "Ctrl" in sample_name or "Control" in sample_name:
#        extrusion_controls[extrusion_name].append(row[P1MI_key])
#    else:
#        samples_2.append([extrusion_name + " " + sample_name, row[P1MI_key]])
#
#for name, diameter in samples_2:
#
#    extrusion_name = name[0:5]
#    condition = name[6:]
#
#    adj_diameter = diameter - np.average(extrusion_controls[extrusion_name])
#
#    data_by_cond_2[condition].append(adj_diameter)
#
#data_params_by_cond_2 = {i:[cond_dict_2[i],
#                            np.average(data_by_cond_2[i]),
#                            stats.sem(data_by_cond_2[i])] for i in cond_indices_2}
#
#proc_data_2 = pd.DataFrame.from_dict(data_params_by_cond_2,
#                                     orient="index",
#                                     columns=["Vol HCl",
#                                              "Mean",
#                                              "Standard Error"])
#
#proc_data = pd.concat([proc_data_1,proc_data_2])
#
#
#pH_dict = {n:pH_data["Mean"][n] for n in range(16)}
#
#proc_data["pH"] = proc_data.apply(lambda row: pH_dict[row["Vol HCl"]],
#                                  axis = 1)
#
#proc_data["IM"] = proc_data.apply(lambda row: " ".join(row.name.split()[0:-2]),
#                                  axis=1)

#Calculate starting acidity
buffer_list = [(10.32, 8.07),
               (9.8, 6.46),
               (1.032, 0),
               (1.032, 1.5),
               (1.032, 2),
               (1.032, 2.7),
               (1.032, 6.5),
               (1.032, 10.2)]

starting_acid = jbbuffer.get_acid_conc(
    buffer_list, pH_data["Mean"][0])

pH_data["Predicted HCl Conc"] = pH_data.apply(
    lambda row: jbbuffer.get_acid_conc(
        buffer_list,
        row["Mean"]) - starting_acid,
    axis=1)

###Do some basic fitting
#find estimated HCl concentration
pH_data["Est HCl Conc"] = pH_data.apply(
    lambda row: row["Predicted HCl Conc"]/row["Vol HCl"]/10 if row["Vol HCl"]
    != 0 else float("nan"),
    axis=1)

est_hcl_conc = np.nanmean(pH_data["Est HCl Conc"])

def fit_function(vvp_hcl):

    return jbbuffer.solve_buffer_system(
        buffer_list,
        vvp_hcl * est_hcl_conc*10 + starting_acid)


#Do a linear regression with errors
def lin_function(x, m, c):
    return m*x + c


lin_model = lmfit.Model(lin_function,
                        param_names=["m", "c"],
                        name="Linear Model")
lin_params = lin_model.make_params(m=1,c=1)
lin_fit = lin_model.fit([item[1] for item in neg_control_list],
                        lin_params,
                        x=[item[0] for item in neg_control_list])

#Try a sigmoid fit
#def sigmoid_function(pH, y_acid, y_base, pKa):
#
#    protonation_ratio = 10**(pKa-pH)
#    #Higher at low pH when more is protontaed
#
#    frac_protonated = protonation_ratio/(1+protonation_ratio)
#    #Also higher at low pH
#
#    y_output = y_base + (y_acid - y_base) * frac_protonated
#    #Higher frac_protonated means closer to y_acid
#
#    return y_output
#
#sigmoid_model = lmfit.Model(sigmoid_function,
#                            param_names=["y_acid","y_base","pKa"],
#                            name="Sigmoid Model")
#sigmoid_params = sigmoid_model.make_params(y_acid=10,y_base=30,pKa=6)
#
#sigmoid_fits = {}
#for name in ["IM 1", "IM 2"]:
#    proc_data_needed = proc_data.loc[proc_data["IM"]==name]
#    y_data = [y for y in proc_data_needed["Mean"]]
#    ph_data = [x for x in proc_data_needed["pH"]]
#    weight_data = [1/sig for sig in proc_data_needed["Standard Error"]]
#    sigmoid_fits[name] = sigmoid_model.fit(y_data,
#                                           sigmoid_params,
#                                           pH=ph_data,
#                                           weights=weight_data)
#
#def sigmoid_fit_function(x, name):
#    return sigmoid_fits[name].eval(pH=x)
#
#def sigmoid_fit_uncertainty_function(x, name):
#    return sigmoid_fits[name].eval_uncertainty(pH=x)

###Plot Everything
fig_1, ax_1 = plt.subplots(1, 1, figsize=[4,4])

jbplot.plotdf(ax_1, pH_data,
              x="Vol HCl",
              y="Mean",
              y_sig="Standard Error",
              plot_type="scatter",
              name_list=["Experimental"])
jbplot.plotfun(ax_1, fit_function,
               x_min=0, x_max=15,
               label="Theoretical fit")
ax_1.set_xlabel("V/V% {conc:.0f} mM HCl added".format
             (conc = est_hcl_conc*1000))
ax_1.set_ylabel("pH")
ax_1.legend()

jbplot.remove_box(ax_1)

fig_2, ax_2 = plt.subplots(1, 1, figsize=[4,4])

jbplot.plotdf(ax_2, main_data_processed,
              x="pH",
              y="Mean",
              y_sig="Standard Error",
              split = "Condition",
              name_list = ["No DNA", "30 bp", "I-Motif"])
jbplot.plotfun(ax_2, lambda x: lin_fit.eval(x=x),
               sig_function = lambda x: lin_fit.eval_uncertainty(x=x,sigma=1)[0],
               label=linear_regression_string,
               x_min = 5.45,
               x_max = 8.05,
               log=True)
ax_2.set_xlabel("pH")
ax_2.set_ylabel("$\Delta D$ / nm")
ax_2.set_xlim(8.1,5.4)

ax_2.legend()

jbplot.remove_box(ax_2)

jbplot.save(fig_1,"pH Calibration Curve")
jbplot.save(fig_2,"Brushes + pH Changes")

#jbplot.plotdf(ax_2, proc_data,
#              x="pH",
#              y="Mean",
#              y_sig="Standard Error",
#              split="IM",
#              plot_type="scatter",
#              name="IM")
#jbplot.plot2dfun(ax_2, sigmoid_fit_function,
#                 second_var=["IM 1", "IM 2"],
#                 sig_fun=sigmoid_fit_uncertainty_function,
#                 x_min = 5.5,x_max=8.0,
#                 log=True)
#
#ax_2.scatter([8.0], [37.74],marker="*",color="#808080",edgecolors="black",s=60)
#ax_2.legend()
#ax_2.set_ylim(-1,51)
#ax_2.set_xlabel("pH")
#ax_2.set_ylabel("$\Delta D / nm$")
#ax_2.set_title("b", loc="left",
#               fontweight="bold")
#
#fig.savefig("DLS Data from I-motif Brush.png",
#            format="png",
#            bbox_inches="tight",

