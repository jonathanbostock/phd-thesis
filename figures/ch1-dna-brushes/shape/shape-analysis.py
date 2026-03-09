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

    # Create the plot
    fig = plot_brush_data_categorical(
        df_data=df_data,
        category_col="Shape",
        value_col="Concentration",
        x_col="Lipid:DNA Ratio",
        y_col="Delta D",
        ax_1_title=r"Shape Effect on Brush $\Delta D$",
        ax_2_title=r"Shape Effect on $\Delta D_{max}$ and $c_{1/2}$",
        xlabel="Lipid:DNA Ratio",
        ylabel=r"$\Delta D$",
        legend_title="DNA Shape",
        figsize=(defaults.fig_width * 2, defaults.fig_height),
    )

    # Save and show
    save_plot(fig, "Shape Brush Plot")
    plt.show()


def plot_parameter_data(ax, parameter_data) -> None:
    pass


if __name__ == "__main__":
    main()
