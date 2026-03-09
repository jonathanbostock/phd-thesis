#20-Jan-2023
#Jonathan Bostock

#Import and analyse data from DLS of brushes

import numpy as np
import scipy
import matplotlib.pyplot as plt

#Import the first set of data

raw_data = np.genfromtxt("2023-01-18-Brush-DLS-3.tsv", delimiter="\t")

before_after_data_1 = raw_data[1:,2:4]

differences_1 = []
final_size_1 = []


for pair in before_after_data_1:
    differences_1.append(pair[1]-pair[0])
    final_size_1.append(pair[1])

ratio_list = [40,200,800,4000]
length_list = [38, 53, 68, 80]

stdev = 3.2

data_grid_1 = []

data_grid_1.append(differences_1[1:5])
data_grid_1.append(differences_1[5:9])
data_grid_1.append(differences_1[9:13])
data_grid_1.append(differences_1[13:17])

data_grid_1t = np.transpose(data_grid_1)


##Plotting is done here
#Set up subplots

figure_1, (delta_vs_conc, delta_vs_length) = plt.subplots(1, 2)
figure_1.set_size_inches(9,4.5)

figure_1.suptitle("Change in Z-average after 60 minutes")


#Plot delta vs conc
delta_vs_conc.plot(ratio_list,data_grid_1[3],
                   label="80 bp",marker="s",color="darkorange",linestyle="solid")
delta_vs_conc.plot(ratio_list,data_grid_1[2],
                   label="68 bp",marker="+",color="dodgerblue",linestyle="solid")
delta_vs_conc.plot(ratio_list,data_grid_1[1],
                   label="53 bp",marker="^",color="gold",linestyle="solid")
delta_vs_conc.plot(ratio_list,data_grid_1[0],
                  label="38 bp",marker="x",color="darkcyan",linestyle="solid")

delta_vs_conc.set_xscale("log")
delta_vs_conc.invert_xaxis()
delta_vs_conc.set_xlabel("Lipid:DNA Ratio")
delta_vs_conc.set_ylabel("Change in Z-average / nm")

delta_vs_conc.legend()

#Plot delta vs length
delta_vs_length.plot(length_list,data_grid_1t[0],label="40:1",marker="s",color="darkorange",linestyle="dashed")
delta_vs_length.plot(length_list,data_grid_1t[1],label="200:1",marker="+",color="dodgerblue",linestyle="dashed")
delta_vs_length.plot(length_list,data_grid_1t[2],label="800:1",marker="^",color="gold",linestyle="dashed")
delta_vs_length.plot(length_list,data_grid_1t[3],label="4000:1",marker="x",color="darkcyan",linestyle="dashed")

delta_vs_length.set_xlabel("Length of DNA Brush")
delta_vs_length.set_ylabel("Change in Z-average / nm")

delta_vs_length.legend()


#Make a second figure to compare data when we don't take difference
figure_2, delta_vs_conc_2 = plt.subplots(1,1)

figure_2.set_size_inches(4.5,4.5)
figure_2.suptitle("Final Z-average")

#Plot delta vs conc
delta_vs_conc_2.plot(ratio_list,final_size_1[1:5],
                     label="38 bp",marker="s",color="darkorange",linestyle="solid")
delta_vs_conc_2.plot(ratio_list,final_size_1[5:9],
                     label="53 bp",marker="+",color="dodgerblue",linestyle="solid")
delta_vs_conc_2.plot(ratio_list,final_size_1[9:13],
                     label="68 bp",marker="^",color="gold",linestyle="solid")
delta_vs_conc_2.plot(ratio_list,final_size_1[13:17],
                     label="80 bp",marker="x",color="darkcyan",linestyle="solid")

delta_vs_conc_2.set_xscale("log")
delta_vs_conc_2.invert_xaxis()
delta_vs_conc_2.set_xlabel("Lipid:DNA Ratio")
delta_vs_conc_2.set_ylabel("Final Z-average / nm")
delta_vs_conc_2.legend

#Save the plots
figure_1.savefig("2023-01-18 DLS.png",format="png",dpi=300)
figure_2.savefig("2023-01-18 DLS 2.png",format="png",dpi=300)
