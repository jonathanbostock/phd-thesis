### Jonathan Bostock

import numpy as np
import pandas as pd
<<<<<<< HEAD
import scipy.stats as stats
import matplotlib.pyplot as plt
=======
>>>>>>> a8fc83bf5dba58cdd5edeedeb35dfe8d43cafb2a
import jbplot

def main():
    raw_data = pd.read_csv("2025-01-09 pH Response 1.csv")

    raw_data["pH"] = raw_data.apply(
<<<<<<< HEAD
        lambda row: float(row["Sample Name"].split(" ")[2]),
        axis=1)
    raw_data["DNA Present"] = raw_data.apply(
        lambda row: row["Sample Name"].split(" ")[0] != "Ctrl",
        axis=1)
    raw_data["Repeat"] = raw_data.apply(
        lambda row: float(row["Repeat"]),
        axis=1)
    
    data_dict = {ph: [] for ph in [6.0, 6.5, 7.0, 7.5, 8.0]}
    
    for repeat in range(1, 4):
        for ph in [6.0, 6.5, 7.0, 7.5, 8.0]:
            dna_data = raw_data[(raw_data["pH"] == ph) & (raw_data["Repeat"] == repeat) & (raw_data["DNA Present"] == True)]
            ctrl_data = raw_data[(raw_data["pH"] == ph) & (raw_data["Repeat"] == repeat) & (raw_data["DNA Present"] == False)]
            
            if len(dna_data) > 0 and len(ctrl_data) > 0:
                delta_d = dna_data["Peak 1 Mean by Intensity ordered by area (nm)"].iloc[0] - ctrl_data["Peak 1 Mean by Intensity ordered by area (nm)"].iloc[0]
                data_dict[ph].append(delta_d)
    
    proc_data = pd.DataFrame()
    proc_data["pH"] = list(data_dict.keys())
    proc_data["Mean"] = [np.mean(data_dict[ph]) if len(data_dict[ph]) > 0 else np.nan for ph in data_dict.keys()]
    proc_data["SE"] = [stats.sem(data_dict[ph]) if len(data_dict[ph]) > 1 else 0 for ph in data_dict.keys()]
    
    fig, ax = plt.subplots(1, 1, figsize=[4,4])
    
    jbplot.plotdf(ax,
                  proc_data,
                  x = "pH",
                  y = ["Mean"],
                  name_list = ["30bp"],
                  y_sig = ["SE"],
                  plot_type = "scatter")
    
    jbplot.plotdf(ax,
                  proc_data,
                  x = "pH",
                  y = ["Mean"],
                  plot_type = "line")
    
    ax.set_xlabel("pH")
    ax.set_ylabel("$\Delta D$ / nm")
    ax.set_xlim(8.1, 5.9)
    
    jbplot.nice_legend(ax)
    
    jbplot.save(fig, "pH Response Brush 2")
=======
	lambda row: float(row["Sample Name"].split(" ")[2]),
	axis=1)
    raw_data["DNA Present"] = raw_data.apply(
	lambda row: row["Sample Name"].split(" ")[0] != "Ctrl",
	axis=1)
    raw_data["Repeat"] = raw_data.apply(
	lambda row: float(row["Repeat"]),
	axis=1)
    
    """
    processed_data = pd.DataFrame()
    processed_data[["ph", "batch", "delta_d"]] = [
	[ph,
	 batch,
	 raw_data.where[raw_data["pH"] == ph && raw_data["Repeat"] == repeat && raw_data["DNA Present"] == True] \ 	 - raw_data.where[raw_data["pH"] == ph && raw_data["Repeat"] == repeat && raw_data["DNA Present"] == False]]
	for ph in [6.0, 6.5, 7.0, 7.5, 8.0]
	for repeat in range(1,4)]
    """

    processed_data_list = []
    for repeat in range(1,4):
        for ph in [6.0, 6.5, 7.0, 7.5, 8.0]:
            processed_data_list.append([
		ph,
		repeat,
		raw_data.where[raw_data["pH"] == ph & raw_data["Repeat"] == repeat & raw_data["DNA Present"] == True] - raw_data.where[raw_data["pH"] == ph & raw_data["Repeat"] == repeat & raw_data["DNA Present"] == False]])
>>>>>>> a8fc83bf5dba58cdd5edeedeb35dfe8d43cafb2a

if __name__ == "__main__":
    main()