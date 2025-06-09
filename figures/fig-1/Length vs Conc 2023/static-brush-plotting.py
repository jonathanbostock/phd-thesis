# Jonathan Bostock
# 26-Apr-2023
# Analysing the data for the static, dsDNA brushes

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import phdutils.plot as jbplot
from itertools import product

def main() -> None:
    df = pd.read_csv("static-brush-data.csv")

    controls = (df["sample_type"] == "ctrl")

    def process_row(row, controls):

        control_data = controls[controls["batch"] == row["batch"]]
        subtract_columns = [c not in ["batch", "sample_type", "sample_number"]
                            for c in control_data.columns]
        control_row = control_data.loc[..., subtract_columns].mean()

        new_row = row.copy()

        new_row[subtract_columns] = new_row[subtract_columns] - control_row

        return new_row


    df_ctrl = df[controls]
    df_data = df[~controls].apply(
        lambda row: process_row(row, df_ctrl),
        axis=1)

    # 5 microliters of 0.5 mg/mL POPC (molar mass 760)
    lipid_moles = 5e-6 * 0.5 / 760
    # microliters of 5 micromolar DNA
    df_data["dna_moles"] = df_data["sample_number"] * 1e-6 * 5e-6
    df_data["Brush Length / bp"] = list(map(float, df_data["sample_type"]))

    df_data["Lipid:DNA Ratio"] = (lipid_moles/df_data["dna_moles"]).astype(int)

    processed_data = []

    for brush_length, lipid_dna_ratio in product(
        set(df_data["Brush Length / bp"]), 
        set(df_data["Lipid:DNA Ratio"])):

        data = df_data[(df_data["Brush Length / bp"] == brush_length) & \
                       (df_data["Lipid:DNA Ratio"] == lipid_dna_ratio)]["peak_1_mean_intensity"]

        processed_data.append({
            "Brush Length / bp":    brush_length,
            "Lipid:DNA Ratio":      lipid_dna_ratio,
            "Mean Delta D":         np.mean(data),
            "SEM Delta D":          stats.sem(data)})

    df_processed_data = pd.DataFrame(processed_data)

    fig, ax = jbplot.figax()

    jbplot.plotdf(ax, df_processed_data,
                  x="Lipid:DNA Ratio",
                  y="Mean Delta D",
                  y_sig="SEM Delta D",
                  split="Brush Length / bp",
                  plot_type="scatter",
                  gradient=True)
    ax.set_xscale("log")
    ax.set_xlim(ax.get_xlim()[::-1])
    jbplot.save(fig, "Static Brush Plot")


if __name__ == "__main__":
    main()
