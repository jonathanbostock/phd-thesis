#Jonathan bostock

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product
from scipy import stats

import phdutils.plot as jbplot


def main():
    # Load the data
    data = pd.read_csv(Path(__file__).parent / 'shape-effects-data.csv')
    data["extrusion"] = data.apply(lambda row: row["sample_name"].split(" ")[0][-1], axis=1)
    data["sample_type"] = data.apply(lambda row: row["sample_name"].split(" ")[1].lower(), axis=1)
    data["volume"] = data.apply(lambda row: float(row["sample_name"].split(" ")[2]) if len(row["sample_name"].split(" ")) > 2 else None, axis=1)
    data["p1mi_adjusted"] = data.apply(lambda row: row["p1mi"] - data.where(
        (data["sample_type"] == "control") & (data["extrusion"] == row["extrusion"])
    )["p1mi"].mean(), axis=1)
    data["z_average_adjusted"] = data.apply(lambda row: row["z_average"] - data.where(
        (data["sample_type"] == "control") & (data["extrusion"] == row["extrusion"])
    )["z_average"].mean(), axis=1)

    # volume per volume * lipid g/L / lipid mass
    lipid_conc = (5/500) * 0.5 / 760.091
    dna_conc = 2.5 * 10**-6

    data["lipid_dna_ratio"] =  lipid_conc / (dna_conc * data["volume"]/ 500)

    processed_data = []
    for sample_type, lipid_dna_ratio in product(
        set(data["sample_type"]), 
        set(data["lipid_dna_ratio"])):

        if sample_type == "control":
            continue

        data_subset = data[(data["sample_type"] == sample_type) & \
                           (data["lipid_dna_ratio"] == lipid_dna_ratio)]["z_average_adjusted"]

        processed_data.append({
            "Sample Type":          sample_type,
            "Lipid:DNA Ratio":      lipid_dna_ratio,
            "Mean Delta D":         np.mean(data_subset),
            "SEM Delta D":          stats.sem(data_subset)})

    df_processed_data = pd.DataFrame(processed_data).sort_values(by="Lipid:DNA Ratio").sort_values(
        by="Sample Type", key=lambda x: x.map(lambda y: ["30bp-linear", "30bp-branched", "4-way-star"].index(y)))

    fig, ax = jbplot.figax()

    jbplot.plotdf(ax, df_processed_data,
                  x="Lipid:DNA Ratio",
                  y="Mean Delta D",
                  y_sig="SEM Delta D",
                  split="Sample Type",
                  plot_type="scatter",
                  name_map = lambda tuple: tuple[0].replace("-", " ").title())
    ax.set_xscale("log")
    ax.set_xlim(ax.get_xlim()[::-1])
    jbplot.nice_legend(ax)
    jbplot.save(fig, Path(__file__).parent / "Shape Effect")




if __name__ == '__main__':
    main()