### Plotting our alpha hemolysin data
import utils.plotting as plotting
import utils.calcein_release as calcein_release
from utils.stats import t_test_unpaired
import os
import pandas as pd

def main():
    # Load the data
    current_path = os.path.dirname(os.path.abspath(__file__))
    release_df = calcein_release.calculate_release_from_path(current_path)

    mean_release, sem_release = calcein_release.calculate_average_and_sem_release(release_df)

    experiment_names = ["No Brush",
                        "Sparse Brush",
                        "Dense Brush",
                        "No Brush",
                        "Sparse Brush",
                        "Dense Brush"]

    # Plot the data
    fig1, fig2 = plotting.plot_calcein_release(
        mean_df = mean_release.iloc[:, :-1],
        sem_df = sem_release.iloc[:, :-1],
        raw_release_df = release_df,
        experiment_names = experiment_names, group_size=3)
    plotting.save_plot(fig1, os.path.join(current_path, 'barchart'))
    plotting.save_plot(fig2, os.path.join(current_path, 'linechart'))


    # P-values
    final_release = release_df.iloc[-1]
    no_brush_data = final_release[["A4", "B4", "C4"]].tolist()
    sparse_brush_data = final_release[["A5", "B5", "C5"]].tolist()
    dense_brush_data = final_release[["A6", "B6", "C6"]].tolist()

    p_value_dataframe = pd.DataFrame([
        t_test_unpaired(no_brush_data, sparse_brush_data),
        t_test_unpaired(no_brush_data, dense_brush_data),
    ], index=["No Brush vs Sparse Brush", "No Brush vs Dense Brush"])

    p_value_dataframe.to_csv(os.path.join(current_path, "p_values.csv"))

if __name__ == "__main__":
    main()