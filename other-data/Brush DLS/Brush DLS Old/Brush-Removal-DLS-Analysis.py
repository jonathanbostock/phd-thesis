##29 Jan 2023
#Jonathan Bostock


import numpy as np
import scipy.optimize as op
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
import jbplot

#Import data
abspath=os.path.abspath(__file__)
dname=os.path.dirname(abspath)
os.chdir(dname)

raw_data = np.genfromtxt("2023-03-09-Brush-DLS-TMSD-1.tsv", delimiter = "\t")

raw_data_t = raw_data.transpose()

delta_vector_set = raw_data_t[2:7,13:19]
time_vector = [0, 5, 15, 30, 45, 90]
conc_name_vector = ["0.2 eq", "0.5 eq", "1 eq", "2 eq", "No Linker"]

#Plotting
#Figure 1
figure_1, ax_1 = plt.subplots(figsize=(4,4))

#Ax 1, where we plot our main data
jbplot.scatterset(ax_1, [time_vector], delta_vector_set,
                  name_set = conc_name_vector,
                  line=True)

ax_1.set_xlabel("Time / min")
ax_1.set_ylabel("Brush Diamter / nm")
ax_1.legend()
ax_1.set_title("a",loc="left",fontweight="bold")

figure_1.savefig("2023-03-09 Brush Removal DLS.png",
                 format="png",dpi=600,bbox_inches="tight")
