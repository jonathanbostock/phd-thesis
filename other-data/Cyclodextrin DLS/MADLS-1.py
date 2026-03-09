### Jonathan Bostock
### 6 Oct 2023
### MADLS of Cyclodextrin stuff

import numpy as np
import matplotlib.pyplot as plt
import copy
import pandas as pd
import math
import jbplot

raw_data = (pd.read_csv("2023-10-06 Cyclodextrin MADLS 1.tsv",
                        sep="\t",
                        names = ["Size / nm", "DNA Before Dextrin", "K0",
                                 "DNA Dextrin 1 min", "K1", "DNA Dextrin 10 min", "K2",
                                 "Control Before Dextrin", "K3", "Control Dextrin 1 min", "K4",
                                 "DNA Dextrin 30 min", "K5", "Control Dextrin 10 min", "K6",
                                 "Control Dextrin 30 min"],
                        header=1)
            .drop(columns=["K{}".format(n) for n in range(7)]))


fig = plt.figure(figsize=[4,4])

jbplot.ridgelinedf(fig,
                   raw_data,
                   x = ["Size / nm"],
                   y = ["Control Before Dextrin","DNA Before Dextrin",
                        "Control Dextrin 1 min", "DNA Dextrin 1 min",
                        "Control Dextrin 10 min", "DNA Dextrin 10 min",
                        "Control Dextrin 30 min", "DNA Dextrin 30 min"],
                   x_scale = "log",
                   plots_per_axis = 2,
                   colorcodes = [i % 2 for i in range(8)],
                   linecodes = [0, 2] * 4,
                   overlap = 0,
                   axis_names = ["Before\nDextrin", "Dextrin\n1 min",
                                 "Dextrin\n10 min", "Dextrin\n30 min"],
                   legend_names =["Control", "DNA"],
                   x_axis_name = "Size / nm")



jbplot.save(fig, "MADLS")
