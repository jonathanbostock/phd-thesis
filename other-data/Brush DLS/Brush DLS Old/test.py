## 29 Jan 2023
# Jonathan Bostock


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

x_vals = [np.random.normal(loc=1, scale=2) for i in range(100)]

jbplot.violinplot()
