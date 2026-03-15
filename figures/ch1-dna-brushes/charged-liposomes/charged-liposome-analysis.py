# Jonathan Bostock
# Charged Liposome Analysis
# Analysing the effect of DOPG percentage on brush formation

import pandas as pd
import matplotlib.pyplot as plt
from utils.data_processing import process_charged_liposome_data
from utils.plotting import plot_linear_relationship, save_plot
from utils import defaults


def main() -> None:
    # Read the data
    df = pd.read_csv("charged-liposome-data.csv")

    # Process the data
    df_data = process_charged_liposome_data(df)

    # Create the plot using delta_p1mi (equivalent to Delta D)
    fig, ax, slope, intercept, r_value, p_value, std_err = plot_linear_relationship(
        df_data=df_data,
        x_col="DOPG_percentage",
        y_col="delta_p1mi",
        title="Effect of Charge on Brush Formation",
        xlabel="DOPG Percentage / %",
        ylabel=r"$\Delta D$",
        color_index=4,
        figsize=(
            defaults.fig_width * defaults.small_fig_scale,
            defaults.fig_height * defaults.small_fig_scale,
        ),
    )

    # Save and show
    save_plot(fig, "Charged Liposome Plot")
    plt.show()


if __name__ == "__main__":
    main()
