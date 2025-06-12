# Jonathan Bostock
# pH Sensitive Brush Analysis
# Analysing the effect of pH on brush formation

import pandas as pd
import matplotlib.pyplot as plt
from utils.data_processing import process_ph_data
from utils.plotting import plot_linear_relationship, save_plot
from utils import defaults


def main() -> None:
    # Read the data with error handling for malformed lines
    df = pd.read_csv("ph-response-data.csv", on_bad_lines='skip')
    
    # Process the data
    df_data = process_ph_data(df)
    
    # Create the plot using delta_p1mi (equivalent to Delta D)
    fig, ax, slope, intercept, r_value, p_value, std_err = plot_linear_relationship(
        df_data=df_data,
        x_col="pH",
        y_col="delta_p1mi",
        title="Effect of pH on Brush",
        xlabel="pH",
        ylabel="$\Delta D$",
        color_index=3,
        figsize=(defaults.fig_width*defaults.small_fig_scale, defaults.fig_height*defaults.small_fig_scale)
    )

    # Save and show
    save_plot(fig, "pH Response Plot")
    plt.show()


if __name__ == "__main__":
    main()
