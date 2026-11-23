"""
Raw calcein fluorescence of two early runs in which \\gls{ahl} caused no leakage.

Both panels show fluorescence as recorded rather than a release percentage.
Neither run can be normalised: too little Triton X-100 was added at the end of
each to fully lyse the vesicles, so the final read does not mark 100 % release.
Both were recorded at 495/515 nm with the PMT at 600 V (medium), so the two
panels are on the same instrument scale, though the vesicle preparations
differ and the absolute levels are not meant to be compared.

\\textbf{a}: 2024-08-07, 100 \\% \\gls{popc} membranes.  One well with \\gls{ahl}
(A1) and its no-\\gls{ahl} control (A6); the run had no replicates.  The final
reading is dropped, because Triton was added to A1 while the timecourse was
still running and the last point of that well catches it (fluorescence jumps
from 21 to 185 in the closing 0.2 min).  The other wells of this run held
\\gls{mbcd} (A5, A8) or other conditions, and A12 was a free-calcein reference.
The Before and Triton reads live alongside the timecourse in
archived-raw-data.

\\textbf{b}: 2024-11-21, EggPC membranes.  The undecorated (no brush) vesicles
with \\gls{ahl} and the no-\\gls{ahl} control, three replicates each (plate rows
A--C), drawn individually rather than as a mean since they disagree.  Wells
are from the plate map in analysis.py, which normalises the same run across
all four of its conditions: plate columns are 1 = no-\\gls{ahl} control,
2 = no brush, 3 = sparse brush, 4 = dense brush, 5 = a Triton reference.

Usage:  uv run python plot-raw-fluorescence.py
"""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils.plotting import format_axes, save_plot, setup_plot_style
from utils.svg_panels import export, pdf_size_mm

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]

POPC = (
    REPO_ROOT
    / "archived-raw-data"
    / "Cary All Raw Data"
    / "2024-08-07 Calcein Leakage Ahl Timecourse 600V.csv"
)
POPC_WELLS = 9
POPC_TIMEPOINTS = 120
# Triton reached A1 during the final reading, so the last point is dropped
POPC_TRITON_CONTAMINATED_POINTS = 1
# (label, index of the well among this run's wells, linestyle)
POPC_CONDITIONS = [("+ Ahl", 0, "-"), ("Control (no Ahl)", 5, "--")]

EGGPC = HERE / "2024-11-21 EggPC Brush Ahl Timecourse.csv"
EGGPC_WELLS = 15
EGGPC_TIMEPOINTS = 60
EGGPC_PLATE_COLUMNS = 5  # the stride between the wells of one condition
# (label, plate column, linestyle)
EGGPC_CONDITIONS = [("+ Ahl, no brush", 2, "-"), ("Control (no Ahl)", 1, "--")]

# Each panel at the thesis single-plot axis size: these are two independent
# plots sharing a figure, not a pair of small multiples.
AXIS_WIDTH_MM = 50
AXIS_HEIGHT_MM = 40
DECORATION_WIDTH_IN = 1.5
DECORATION_HEIGHT_IN = 1.0


def load_timecourse(path: Path, n_wells: int, n_timepoints: int) -> pd.DataFrame:
    """Recorded fluorescence: "Time (min)" then one column per well, in plate order."""
    intensity_columns = ["Intensity (a.u.)"] + [
        f"Intensity (a.u.).{i}" for i in range(1, n_wells)
    ]
    raw = pd.read_csv(
        path,
        header=1,
        engine="python",
        usecols=["Time (min)"] + intensity_columns,
    )
    return raw.iloc[:n_timepoints].astype(float)


def well_columns(timecourse: pd.DataFrame) -> List[str]:
    return [column for column in timecourse.columns if column != "Time (min)"]


def plot_popc(ax) -> None:
    """Panel a: the single Ahl well and its control, on 100 % POPC."""
    timecourse = load_timecourse(POPC, POPC_WELLS, POPC_TIMEPOINTS).iloc[
        : POPC_TIMEPOINTS - POPC_TRITON_CONTAMINATED_POINTS
    ]
    wells = well_columns(timecourse)
    colour = sns.color_palette("colorblind")[0]

    for label, well_index, linestyle in POPC_CONDITIONS:
        ax.plot(
            timecourse["Time (min)"],
            timecourse[wells[well_index]],
            color=colour,
            linestyle=linestyle,
            linewidth=1,
            label=label,
        )

    ax.set_title("100 % POPC, 2024-08-07")


def plot_eggpc(ax) -> None:
    """Panel b: the no-brush wells and their controls, on EggPC."""
    timecourse = load_timecourse(EGGPC, EGGPC_WELLS, EGGPC_TIMEPOINTS)
    wells = well_columns(timecourse)
    colour = sns.color_palette("colorblind")[0]

    for label, plate_column, linestyle in EGGPC_CONDITIONS:
        replicates = wells[plate_column - 1 :: EGGPC_PLATE_COLUMNS]
        for index, well in enumerate(replicates):
            ax.plot(
                timecourse["Time (min)"],
                timecourse[well],
                color=colour,
                linestyle=linestyle,
                linewidth=1,
                label=label if index == 0 else None,
            )

    ax.set_title("EggPC, 2024-11-21")


def main() -> None:
    setup_plot_style()
    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(
            2 * AXIS_WIDTH_MM / 25.4 + DECORATION_WIDTH_IN,
            AXIS_HEIGHT_MM / 25.4 + DECORATION_HEIGHT_IN,
        ),
    )
    plot_popc(ax_a)
    plot_eggpc(ax_b)

    for ax, letter in ((ax_a, "a"), (ax_b, "b")):
        ax.set_xlabel("Time / min")
        ax.set_ylabel("Fluorescence / a.u.")
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        ax.legend(frameon=False, loc="lower left", handlelength=1.8)
        format_axes(ax)
        ax.text(
            -0.3,
            1.08,
            letter,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

    fig.tight_layout()
    save_plot(fig, str(HERE / "ahl-no-leakage"), resize=False)

    pdf = HERE / "ahl-no-leakage.pdf"
    export(HERE / "ahl-no-leakage.svg", pdf=pdf)
    width, height = pdf_size_mm(pdf)
    print(f"{pdf.name}: {width:.1f} x {height:.1f} mm")


if __name__ == "__main__":
    main()
