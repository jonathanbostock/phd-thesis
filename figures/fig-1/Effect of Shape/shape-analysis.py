# Jonathan Bostock
# DNA Brush Shape Analysis
# Analysing the data for different DNA shapes

import pandas as pd
import matplotlib.pyplot as plt
from utils.data_processing import process_shape_data
from utils.plotting import plot_brush_data_categorical, save_plot


def main() -> None:
    # Read the data
    df = pd.read_csv("shape-brush-data.csv")
    
    # Process the data with half concentration (2.5 μM instead of 5 μM)
    df_data = process_shape_data(df, dna_concentration_uM=2.5, control_name="control")
    
    # Create the plot
    fig, ax = plot_brush_data_categorical(
        df_data=df_data,
        category_col="Shape",
        value_col="Concentration", 
        x_col="Lipid:DNA Ratio",
        y_col="Delta D",
        title="Shape Effect on Brush Delta D vs Lipid:DNA Ratio",
        xlabel="Lipid:DNA Ratio",
        ylabel="Delta D",
        legend_title="DNA Shape",
        figsize=(6, 4)
    )
    
    # Save and show
    save_plot(fig, "Shape Brush Plot")
    plt.show()


if __name__ == "__main__":
    main()