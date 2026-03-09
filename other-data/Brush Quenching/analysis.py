### Jonathan bostock

import numpy as np
import pandas as pd
from scipy import stats
import jbplot
import matplotlib.pyplot as plt

def main():

    kwargs = {
        "header":       1,
        "engine":       "python",
        "usecols":      ["Time (min)", "Intensity (a.u.)"] + [f"Intensity (a.u.).{i}" for i in range(1,21)],
    }

    columns = ["Time"] + [
        f"{condition} Repeat {repeat}"
        for repeat in [1,2,3]
        for condition in ["+ve", "1 mM", "2 mM", "5 mM", "0 mM", "Blank (no vesicles)", "Blank (vesicles)"]]

    start_df = pd.read_csv("2024-11-05 Co-Brush Before Vesicles.csv", **kwargs)[:2].astype(float)
    attachment_df = pd.read_csv("2024-11-05 Co-Brush Attachment Timecourse.csv", **kwargs)[:61].astype(float)
    detachment_df = pd.read_csv("2024-11-05 Co-Brush MBCD Timecourse.csv", **kwargs)[:51].astype(float)

    start_df.columns = columns
    attachment_df.columns = columns
    detachment_df.columns = columns

    mbcd_concs = [0,1,2,5]

    def get_quenching(df: pd.DataFrame, mbcd_conc: int|str):
        positives = df[[f"+ve Repeat {repeat}" for repeat in [1,2,3]]]
        blanks_no_vesicles = df[[f"Blank (no vesicles) Repeat {repeat}" for repeat in [1,2,3]]]
        blanks_vesicles = df[[f"Blank (vesicles) Repeat {repeat}" for repeat in [1,2,3]]]
        data = df[[f"{mbcd_conc} mM Repeat {repeat}" for repeat in [1,2,3]]]

        positives.columns = [1,2,3]
        blanks_no_vesicles.columns = [1,2,3]
        blanks_vesicles.columns = [1,2,3]
        data.columns = [1,2,3]

        quenching = 1 - ((data - blanks_vesicles) / (positives - blanks_no_vesicles))

        new_df = pd.DataFrame()
        new_df["Quenching Mean"] = quenching.apply(np.mean, axis=1)
        new_df["Quenching SEM"] = quenching.apply(stats.sem, axis=1)
        new_df["MBCD"] = mbcd_conc
        new_df["Time"] = df["Time"]

        return new_df

    for repeat in [1,2,3]:
        attachment_df[f"Mean mM Repeat {repeat}"] = attachment_df.apply(
            lambda row: np.mean(row[[f"{mbcd_conc} mM Repeat {repeat}" for mbcd_conc in [0,1,2,5]]]),
            axis=1)

    attachment_data = get_quenching(attachment_df, "Mean")

    detachment_data = pd.concat([
        get_quenching(detachment_df, mbcd_conc)
        for mbcd_conc in [0,1,2,5]])

    mbcd_conc_map = lambda x: {0: 1, 1: 2, 2: 3, 5: 4}[x]


    fig, ax = plt.subplots(figsize=[6,4])

    jbplot.plotdf(
        ax, attachment_data,
        x = "Time",
        y = "Quenching Mean",
        y_sig = "Quenching SEM",
        plot_type = "line")

    ax.set_xlabel("Time / min")
    ax.set_ylabel("Quenching")
    jbplot.nice_legend(ax)

    jbplot.save(fig, "attachment")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=[6,4])



    jbplot.plotdf(
        ax, detachment_data,
        x="Time",
        y="Quenching Mean",
        y_sig="Quenching SEM",
        color="MBCD",
        gradient_map = mbcd_conc_map,
        gradient=True,
        color_name_map = lambda x: f"{x} mM MBCD",
        plot_type="line")

    ax.set_xlabel("Time / min")
    ax.set_ylabel("Quenching")
    jbplot.nice_legend(ax)

    jbplot.save(fig, "detachment")

if __name__ == "__main__":
    main()
