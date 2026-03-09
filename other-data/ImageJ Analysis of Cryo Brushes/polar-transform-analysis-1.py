# 06 November 2023
# Jonathan Bostock
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import jbplot
from scipy.stats import linregress
from scipy.signal import argrelextrema
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess

from icecream import ic

import os

def main():

    save_data_to_csvs()

    for smoothed in [True, False]:
        # final_plot(smoothed)
        pass

    plot_fft_averages()

    # plot_different_lines(data_df.loc[data_df["Condition"] == 1])


def save_data_to_csvs():
    files = [f for f in os.listdir("Polar Transforms")]

    data_dict = {
        "DNA Around Equator":               [],
        "DNA Around Equator (smoothed)":     [],
        "Vesicle Circumference":            [],
        "Batch":                            [],
        "Condition":                        [],
        "File Name":                        []
    }

    image_scale_dict = {
        "30k":  1.47 / 4096 * 1000,
        "42k":  1.05 / 4096 * 1000,
        "52k":  0.84 / 4096 * 1000
    }

    fft_dfs = []

    for file_name in [f for f in files]:

        file_name_split = file_name[:-4].split(" ")
        condition = file_name_split[1]
        batch = file_name_split[3]
        transform = file_name_split[5]
        image_scale = file_name_split[6]

        # NM Per Pixel
        nm_per_pixel = image_scale_dict[image_scale]

        file_dict = analyse_file(file_name, nm_per_pixel)

        vesicle_diameter = file_dict["membrane_center"] * nm_per_pixel * 2 + 2

        data_dict["DNA Around Equator"].append(file_dict["fft_max"])
        data_dict["DNA Around Equator (smoothed)"].append(file_dict["fft_max_smoothed"])
        data_dict["Vesicle Circumference"].append(vesicle_diameter * np.pi)
        data_dict["Batch"].append(batch)
        data_dict["Condition"].append(condition)
        data_dict["File Name"].append(file_name)

        this_file_df = pd.DataFrame(columns = ["Batch", "Condition", "FFT Average", "FFT Average (smoothed)", "Spacing / nm"])
        this_file_df["FFT Average"] = file_dict["fft_average"][15:]
        this_file_df["FFT Average (smoothed)"] = file_dict["fft_average_smoothed"][10:]
        this_file_df["Batch"] = batch
        this_file_df["Condition"] = condition
        this_file_df["Transform"] = transform
        this_file_df["Spacing / nm"] = (vesicle_diameter * np.pi) * 1/\
                (np.arange(len(file_dict["fft_average_smoothed"]) - 10) + 15)

        fft_dfs.append(this_file_df)


    data_df = pd.DataFrame(data_dict,
                           columns = data_dict.keys())
    fft_df = pd.concat(fft_dfs)

    # I have no clue why I hvae to do this, but I do for some reason
    # Also I guess it helps with repeatability
    data_df.to_csv("processed_data.csv")
    fft_df.to_csv("fft_data.csv")




def final_plot(smoothed = False):


    data_df = pd.read_csv("processed_data.csv",
                          index_col = 0)

    suffix = " (smoothed)" if smoothed else ""

    # Perform SYMMETRIC REGRESSION
    # Not the dumbass asymmetric regression you normally get
    linear_regressions = {}
    for condition in set(data_df["Condition"]):

        # Get the data
        df_split = data_df.loc[data_df["Condition"] == condition]
        circ = np.array(df_split["Vesicle Circumference"])
        dna = np.array(df_split[f"DNA Around Equator{suffix}"])

        # Normalize the data to mean = 0 and var = 1
        circ_norm = (circ - np.mean(circ))/np.std(circ)
        dna_norm = (dna - np.mean(dna))/np.std(dna)

        ## Everything here is in normalized coordinates
        # Calculate the covariance matrix wtih numpy
        c_d_cov = np.cov(circ_norm, dna_norm)

        # Get the eigenvalues and eigenvectors of the covariance matrix
        c_d_eig = np.linalg.eig(c_d_cov)

        # Find the principal component
        c_d_pc = c_d_eig[1][np.argmax(c_d_eig[0])]

        # Now we have a vector [c_component, d_component]
        # which represents the direction of the principal component
        # d_circ / d_dna === dna_spacing = -c_component/d_component
        slope_norm = -c_d_pc[0]/c_d_pc[1]

        ## Transfer basck to original coordinates
        slope = slope_norm * np.std(circ) / np.std(dna)
        intercept = np.mean(circ) - (slope * np.mean(dna))

        # Calculate the r
        r = c_d_cov[1,0] / np.sqrt(c_d_cov[0,0] * c_d_cov[1,1])

        linear_regressions[condition] = {
            "r":            r,
            "slope":        slope,
            "intercept":    intercept
        }


    condition_descriptions = {
        1:      "68bp Dense",
        2:      "68bp Sparse",
        3:      "38bp Dense",
        4:      "Star Brush Dense"}

    ## Plot the final plot
    fig, ax = plt.subplots(figsize=(6,4))
    jbplot.plotdf(ax, data_df,
                  x = f"DNA Around Equator{suffix}",
                  y = "Vesicle Circumference",
                  color="Condition",
                  color_name_map=lambda c: condition_descriptions[c],
                  shape="Batch",
                  shape_name_map=lambda b: f"Batch {b}",
                  assemble_legend = True)

    ax.set_xlim(-1, None)
    ax.set_ylim(-1, None)

    for i, kv in enumerate(linear_regressions.items()):
        k = kv[0]
        v = kv[1]
        fun = lambda x: v["intercept"] + x * v["slope"]
        label = f"$y = {{{v['slope']:.2g}}}x {{{v['intercept']:+,.2g}}}$"

        jbplot.plotfun(ax, fun,
                       colorcode=i,
                       x_min = 0,
                       x_max = max(data_df[f"DNA Around Equator{suffix}"] * 1.1),
                       label = label)
    """
    for i, kv in enumerate(c_to_d_linear_regressions.items()):
        k = kv[0]
        v = kv[1]

        fun = lambda x: (x - v.intercept) / v.slope
        label = f"$x = {{{v.slope:+,.2g}}}y {{{v.intercept:+,.2g}}}$"

        jbplot.plotfun(ax, fun,
                       colorcode=i,
                       linecode=1,
                       x_min = 0,
                       x_max = max(data_df[f"DNA Around Equator{suffix}"] * 1.1),
                       label = label)
    """

    ax.legend()
    ax.set_ylabel("Vesicle Circumference / nm")
    ax.set_xlabel("DNA Around Equator (estimated)")
    jbplot.nice_legend(ax)
    file_name = "Final Plot (smoothed)" if smoothed else "Final Plot"
    jbplot.save(fig, file_name)
    plt.close(fig)

def plot_fft_averages():

    fft_df = pd.read_csv("fft_data.csv", index_col=0)

    fig, ax = plt.subplots(figsize=[6,4])

    def get_loess(df, condition, batch):
        subset_df = df.loc[(df["Condition"] == condition),
                           (df["Batch"] == batch)]
        x = subset_df["Spacing / nm"]
        y = subset_df["FFT Average (smoothed)"]
        smoothed = lowess(y, x, frac=0.1)

        new_df = pd.DataFrame(
            data = smoothed,
            columns = ["DNA Spacing / nm", "FFT Intensity"])

        new_df["Condition"] = condition
        new_df["Batch"] = batch

        return new_df

    loess_df_list = [
        get_loess(fft_df, condition, batch)
        for condition in set(fft_df["Condition"])
        for batch in set(fft_df["Batch"])]

    loess_df = pd.concat(loess_df_list)

    jbplot.plotdf(ax, loess_df,
                  x = "DNA Spacing / nm",
                  y = "FFT Intensity",
                  color = "Condition",
                  color_name_map = lambda x: f"Condition {x}",
                  shape = "Batch",
                  shape_name_map = lambda x: f"Batch {x}",
                  plot_type = "line",
                  invisible_split = "Transform",
                  assemble_legend = True)

    jbplot.nice_legend(ax)

    ax.set_xlabel("DNA Spacing / nm")
    ax.set_ylabel("FFT Intensity (Averaged and Smoothed)")
    jbplot.save(fig, "FFT Intensity vs DNA Spacing (LOESS)")


def analyse_file(file_name, nm_per_pixel, radial_average_smoothing = 5, fft_smoothing = 10):
    data = np.array(Image.open(f"Polar Transforms/{file_name}"))[:,:500]

    data = (data - np.mean(data)) / np.std(data)

    radial_average = np.mean(data, axis=0)
    radial_fft = np.fft.fft(data, axis=0)
    radial_fft_max = np.argmax(radial_fft, axis=0)

    radial_average_smoothed = smooth(radial_average, radial_average_smoothing)
    # Complicated code to find the membrane center
    try:
        median_lightness = np.median(radial_average_smoothed)
        membrane_local_minima = set(argrelextrema(radial_average_smoothed, np.less)[0])
        membrane_dark = set(np.argwhere((radial_average_smoothed <  - 0.3))[...,0])
        membrane_center_candidates = membrane_local_minima.intersection(membrane_dark)
        membrane_center = max(membrane_center_candidates)
        membrane_edge = np.argmax(radial_average_smoothed[membrane_center:membrane_center+50]) + membrane_center
    except ValueError:
        print(radial_average)

        raise Exception(f"Failed to find minima of the data in file {file_name}")

    # Average the fft over the brush region
    fft_average_start = int(membrane_edge + 15)
    fft_average_end = fft_average_start + 20

    fft_average = np.average(np.abs(radial_fft[:180,fft_average_start:fft_average_end]), axis=1)
    fft_average_smoothed = smooth(fft_average, fft_smoothing)

    # Adding the +1 compensates for the truncation
    fft_max_val = np.argmax(fft_average[5:]) + 5
    fft_max_val_smoothed = np.argmax(
        fft_average_smoothed[fft_smoothing + 5:]) + fft_smoothing + 5

    ### Do all the plotting
    fig_1, ((ax_1, ax_2), (ax_3, ax_4)) = plt.subplots(2,2, figsize=[9,8], width_ratios = [4,5])

    # Heatmaps
    jbplot.heatmap(ax_1, data, legend=False, colormap = "bw")
    ax_1.set_axis_off()
    ax_1.set_title("Polar-transformed Image")

    jbplot.complex_heatmap(ax_2, radial_fft[1:], intensity_transform = lambda x: np.sqrt(x))
    ax_2.set_axis_off()
    ax_2.set_title("Angular Fourier Transform")

    # Plot the radial average
    jbplot.plotline(ax_3, range(len(radial_average)), radial_average, colorcode=0, linecode=0)
    jbplot.plotline(ax_3, range(len(radial_average_smoothed)), radial_average_smoothed, colorcode=1, linecode=2)
    ax_3.set_xlabel("Radius / pixels")
    ax_3.set_ylabel("Average Pixel Intensity")
    ax_3.set_title("Radial Average of Pixels")

    # Plot the average-marking lines
    ax_3.vlines([membrane_center, fft_average_start, fft_average_end],
                min(radial_average_smoothed), max(radial_average_smoothed),
                linestyle=["-",":",":"], color="black")
    ax_2.vlines([fft_average_start, fft_average_end],
                0, 360, linestyles=":", color="white")
    ax_1.vlines([membrane_center, fft_average_start, fft_average_end],
                0, 360, linestyle=["-", ":",":"], color="white")

    # Plot the averaged fft
    # jbplot.plotline(ax_4, range(len(fft_average)), fft_average, colorcode=3, linecode=0)
    jbplot.scatter(ax_4, range(1,len(fft_average)),
                   fft_average[1:],
                   linewidth=0.5)
    jbplot.plotline(ax_4, range(fft_smoothing,len(fft_average_smoothed)),
                    fft_average_smoothed[fft_smoothing:],
                    colorcode=-2, linecode=0)

    X = np.linspace(0, 180, 181)
    ax_4.set_title("Angluarly-Averaged FFT Values")
    ax_4.set_xlabel("$f / circle$")
    ax_4.set_ylabel("Average absolute value of FFT")

    jbplot.save(fig_1, f"Diagnostic Image Plots/FFT Figure {file_name[:-4]}", file_types=["png"])

    plt.close()

    dict_ = {"fft_max":                 fft_max_val,
             "fft_max_smoothed":        fft_max_val_smoothed,
             "fft_average":             fft_average,
             "fft_average_smoothed":    fft_average_smoothed,
             "membrane_center":         membrane_center}

    return dict_

def plot_different_lines(df):

    ## Line 1: regress Y onto X

    x = np.array(df["DNA Around Equator (smoothed)"])
    y = np.array(df["Vesicle Circumference"])


    covariance_matrix = np.cov(x,y)
    eig = np.linalg.eig(covariance_matrix)
    principal_component_vector = eig[1][np.argmax(np.abs(eig[0]))]

    x_norm = (x - np.mean(x)) / np.std(x)
    y_norm = (y - np.mean(y)) / np.std(y)

    covariance_matrix_norm = np.cov(x_norm, y_norm)

    beta = covariance_matrix_norm[1,0]

    eig_norm = np.linalg.eig(covariance_matrix_norm)
    principal_component_vector_norm = eig_norm[1][np.argmax(np.abs(eig_norm[0]))]

    regressions = {
        "x on y":   {},
        "y on x":   {},
        "pca":      {},
        "pca (normed)": {},
        "least sq":   {},
        "least sq (normed)": {}}

    std_ratio = np.std(y) / np.std(x)

    eigvals = sorted(eig[0])
    eigvals_norm = sorted(eig_norm[0])

    regressions["y on x"]["slope"] = std_ratio * beta

    regressions["x on y"]["slope"] = std_ratio / beta

    regressions["pca (normed)"]["slope"] = std_ratio * \
        (- principal_component_vector_norm[1]/principal_component_vector_norm[0])

    regressions["pca"]["slope"] = -principal_component_vector[1]/principal_component_vector[0]

    regressions["least sq"]["slope"] =  (eigvals[1] - eigvals[0]) / (2*covariance_matrix[0,1])
    regressions["least sq (normed)"]["slope"] = std_ratio * (eigvals_norm[1]-eigvals_norm[0]) / \
            (2*covariance_matrix_norm[0,1])

    for v in regressions.values():
        v["intercept"] = np.mean(y) - (np.mean(x) * v["slope"])


    fig, ax = jbplot.figax()
    jbplot.plotdf(ax, df,
                  x="DNA Around Equator (smoothed)",
                  y="Vesicle Circumference")

    colorcode = 1
    for k, v in regressions.items():

        label = f"{k}\n$y = {{{v['slope']:.3g}}}x{{{v['intercept']:+,.3g}}}$"

        jbplot.plotfun(ax,
                       lambda x: x * v["slope"] + v["intercept"],
                       x_min = np.min(x) - 10,
                       x_max = np.max(x) + 10,
                       colorcode = colorcode,
                       linecode = colorcode,
                       label = label)
        colorcode += 1

    jbplot.nice_legend(ax)
    ax.set_xlabel("DNA Around Equator (smoothed)")
    ax.set_ylabel("Vesicle Circumference / nm")


    jbplot.save(fig, "Effect of Regression Type")

def smooth(array, box):

    return np.concatenate([np.array([array[0]] * (box//2 + 1)),
                           np.average(np.array([array[i:-(box+1-i)] for i in range(box+1)]), axis=0)])
"""
def curve_function(x, offset, height, mean, sd):

    return offset + height * np.exp(-0.5*((x-mean)/sd)**2)

def prop_function(x, k):

    return x*k

def lin_function(x, m, c):

    return m*x + c
"""

if __name__ == "__main__":
    main()
