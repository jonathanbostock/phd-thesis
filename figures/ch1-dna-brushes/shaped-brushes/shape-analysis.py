# Jonathan Bostock
# DNA Brush Shape Analysis
# Analysing the data for different DNA shapes

import pandas as pd
import matplotlib.pyplot as plt
from utils.data_processing import process_shape_data
from utils.plotting import plot_brush_data_categorical, save_plot
from utils import defaults


def main() -> None:
    # Read the data
    df = pd.read_csv("shape-brush-data.csv")

    # Process the data with half concentration (2.5 μM instead of 5 μM)
    df_data = process_shape_data(df, dna_concentration_uM=2.5, control_name="control")
    df_data["Shape"] = df_data["Shape"].str.title()
    # The thesis (and the hand-labelled composite figure) call the plain
    # double-stranded construct "linear", so label it that way in the legend too
    df_data["Shape"] = df_data["Shape"].replace({"Duplex": "Linear"})

    plot_kwargs = dict(
        df_data=df_data,
        category_col="Shape",
        value_col="Concentration",
        x_col="Lipid:Construct Ratio",
        y_col="Delta D",
        ax_1_title=r"Shape Effect on Brush $\Delta D$",
        ax_2_title=r"Shape Effect on $\Delta D_{max}$ and $c_{1/2}$",
        xlabel="Lipid:Construct Ratio",
        ylabel=r"$\Delta D$ / nm",
        legend_title="Construct Shape",
    )

    # Combined two-panel figure (the plot used in the original composite figure)
    fig = plot_brush_data_categorical(
        figsize=(defaults.fig_width * 2, defaults.fig_height), **plot_kwargs
    )
    save_plot(fig, "Shape Brush Plot")

    # The same two panels rendered individually for the thesis figures
    # shaped-constructs-b and shaped-constructs-c (panel a is drawn in Inkscape;
    # see split-panels.py).
    fig_b, ax_b = plt.subplots(figsize=(defaults.fig_width, defaults.fig_height))
    fig_c, ax_c = plt.subplots(figsize=(defaults.fig_width, defaults.fig_height))
    plot_brush_data_categorical(axes=(ax_b, ax_c), **plot_kwargs)
    save_plot(fig_b, "shaped-constructs-b")
    save_plot(fig_c, "shaped-constructs-c")

    plt.show()


def plot_parameter_data(ax, parameter_data) -> None:
    pass


if __name__ == "__main__":
    main()
