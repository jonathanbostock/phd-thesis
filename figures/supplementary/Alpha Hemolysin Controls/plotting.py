### Plotting Alpha Hemolysin Control data
import utils.plotting as plotting
import utils.calcein_release as calcein_release
import os


def main():
    # Load the data
    current_path = os.path.dirname(os.path.abspath(__file__))
    release_df = calcein_release.calculate_release_from_path(current_path)

    mean_release, sem_release = calcein_release.calculate_average_and_sem_release(
        release_df
    )

    experiment_names = [
        "Control",
        "No Brush",
        "Dense Brush",
        "No Brush + POPC",
        "Dense Brush + POPC",
        "PEG",
        "PEG Control",
    ]

    # Plot the data (N=1, no replicates yet)
    fig1, fig2 = plotting.plot_calcein_release(
        mean_df=mean_release,
        sem_df=sem_release,
        raw_release_df=release_df,
        experiment_names=experiment_names,
        group_size=7,
    )
    plotting.save_plot(fig1, os.path.join(current_path, "barchart"))
    plotting.save_plot(fig2, os.path.join(current_path, "linechart"))


if __name__ == "__main__":
    main()
