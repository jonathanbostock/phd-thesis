##29 Jan 2023
#Jonathan Bostock


import numpy as np
import scipy.optimize as op
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import jbplot

#Import data
abspath=os.path.abspath(__file__)
dname=os.path.dirname(abspath)
os.chdir(dname)

test_list = range(100)
test_matrix = [test_list] * 10

circle_res = 90

test_circle = []

for i in range(circle_res):

    test_row = []
    for j in range(circle_res):

        i_adj = float(i) - (circle_res/2 - 0.5)
        j_adj = float(j) - (circle_res/2 - 0.5)

        angle_raw = np.arctan(i_adj/j_adj)

        if j_adj > 0:

            angle_raw -= np.pi * np.sign(i_adj)

        test_row.append(angle_raw/np.pi)

    test_circle.append(test_row)


fig_1, ((ax_1),(ax_2),(ax_3),(ax_4), (ax_5)) = plt.subplots(5, 1,
                                             figsize=(5,8),height_ratios=[1,1,1,1,4])
color_image = [mcolors.to_rgb(jbplot.colors[i]) for i in range(7)]

ax_1.imshow([color_image])
jbplot.heatmap(ax_2, test_matrix, colormap="lin", legend=False)
jbplot.heatmap(ax_3, test_matrix, colormap="lin2", legend=False)
jbplot.heatmap(ax_4, test_matrix, colormap="bilin", legend=False)
jbplot.heatmap(ax_5, test_circle, colormap="cyc", val_range=[-1,1])

ax_1.axis("off")
ax_2.axis("off")
ax_3.axis("off")
ax_4.axis("off")
ax_5.axis("off")

ax_1.set_title("Okabe Ito Colors",loc="left",fontweight="bold",fontsize=10)
ax_2.set_title("Linear Colormap",loc="left",fontweight="bold",fontsize=10)
ax_3.set_title("Alternate Linear Colormap",
               loc="left",fontweight="bold",fontsize=10)
ax_4.set_title("Bilinear Colormap",loc="left",fontweight="bold",fontsize=10)
ax_5.set_title("Cyclic Colormap",loc="left",fontweight="bold",fontsize=10)

fig_1.suptitle("Colors from JBPlot",fontweight="bold")

fig_1.savefig("JBPlot Colormap Test.png",format="png")
