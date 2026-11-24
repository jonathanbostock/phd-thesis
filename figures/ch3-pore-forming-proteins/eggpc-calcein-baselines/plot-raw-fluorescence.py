"""
Raw calcein fluorescence of two early runs in which \\gls{ahl} caused no leakage.

Both panels show fluorescence as recorded rather than a release percentage.
Neither run can be normalised: too little Triton X-100 was added at the end of
each to fully lyse the vesicles, so the final read does not mark 100 % release.
Both were recorded at 495/515 nm with the PMT at 600 V, so the two panels are
on the same instrument scale, though the vesicle preparations differ and the
absolute levels are not meant to be compared.

Colour distinguishes vesicles incubated with \\gls{ahl} from the matched
no-\\gls{ahl} controls.  Lines are the mean and the shaded regions span one
standard error either side, as in the other calcein figures of this chapter.
Panel a has a single well per condition, so it carries a line with no band.
Neither panel has a title, and the legends do not give the number of
repeats: both belong in the figure caption.

Panel a: 2024-08-07, 100 % \\gls{popc} membranes, one well with \\gls{ahl} (A1)
and its control (A6); the run had no repeats.  The final reading is dropped,
because Triton was added to A1 while the timecourse was still running and the
last point of that well catches it (fluorescence jumps from 21 to 185 in the
closing 0.2 min).  The other wells of this run held \\gls{mbcd} (A5, A8) or
other conditions, and A12 was a free-calcein reference.  The Before and
Triton reads live alongside the timecourse in archived-raw-data.

Panel b: 2024-11-21, EggPC membranes, the undecorated (no brush) vesicles
with \\gls{ahl} against the no-\\gls{ahl} controls, three repeats each.  Wells
are from the plate map in analysis.py, which normalises the same run across
all four of its conditions: the fifteen intensity columns are in plate order
(A1-A5, B1-B5, C1-C5), plate columns are 1 = no-\\gls{ahl} control,
2 = no brush, 3 = sparse brush, 4 = dense brush, 5 = a Triton reference, and
plate rows A-C are the repeats.

Usage:  uv run python plot-raw-fluorescence.py
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils.plotting import format_axes, save_plot, setup_plot_style
from utils.svg_panels import export, pdf_size_mm

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]

# Matching the other calcein figures of this chapter
LINE_WIDTH = 2
BAND_ALPHA = 0.3

# Each panel at the thesis single-plot axis size: these are two independent
# plots sharing a figure, not a pair of small multiples.
AXIS_WIDTH_MM = 50
AXIS_HEIGHT_MM = 40
DECORATION_WIDTH_IN = 1.5
DECORATION_HEIGHT_IN = 0.9


@dataclass(frozen=True)
class Panel:
    """One run: where to read it, how much of it to use, and what to draw."""

    path: Path
    n_wells: int
    n_timepoints: int
    # Readings to drop from the end, where Triton reached a well mid-run
    drop_from_end: int
    # (legend label, indices of that condition's wells among this run's wells)
    series: Sequence[Tuple[str, Sequence[int]]]


PANELS = [
    Panel(
        path=REPO_ROOT
        / "archived-raw-data"
        / "Cary All Raw Data"
        / "2024-08-07 Calcein Leakage Ahl Timecourse 600V.csv",
        n_wells=9,  # A1-A8 and A12
        n_timepoints=120,
        drop_from_end=1,  # Triton reached A1 during the final reading
        series=[("+ Ahl", [0]), ("− Ahl", [5])],  # A1; A6
    ),
    Panel(
        path=HERE / "2024-11-21 EggPC Brush Ahl Timecourse.csv",
        n_wells=15,  # A1-A5, B1-B5, C1-C5
        n_timepoints=60,
        drop_from_end=0,
        series=[
            ("+ Ahl", [1, 6, 11]),  # A2, B2, C2: undecorated vesicles
            ("− Ahl", [0, 5, 10]),  # A1, B1, C1: the matched controls
        ],
    ),
]


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


def plot_panel(ax, panel: Panel) -> None:
    """Draw one run: a mean line per condition, with a standard-error band."""
    timecourse = load_timecourse(panel.path, panel.n_wells, panel.n_timepoints)
    timecourse = timecourse.iloc[: len(timecourse) - panel.drop_from_end]
    wells = well_columns(timecourse)
    time = timecourse["Time (min)"]
    colours = sns.color_palette("colorblind", n_colors=len(panel.series))

    for (label, indices), colour in zip(panel.series, colours):
        repeats = timecourse[[wells[index] for index in indices]]
        mean = repeats.mean(axis=1)
        ax.plot(time, mean, color=colour, linewidth=LINE_WIDTH, label=label)
        if len(indices) > 1:  # a single well has no standard error to show
            error = repeats.sem(axis=1)
            ax.fill_between(
                time, mean - error, mean + error, color=colour, alpha=BAND_ALPHA
            )

    ax.set_xlabel("Time / min")
    ax.set_ylabel("Fluorescence / a.u.")
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.legend(frameon=False, loc="lower left")
    format_axes(ax)


def main() -> None:
    setup_plot_style()
    fig, axes = plt.subplots(
        1,
        len(PANELS),
        figsize=(
            len(PANELS) * AXIS_WIDTH_MM / 25.4 + DECORATION_WIDTH_IN,
            AXIS_HEIGHT_MM / 25.4 + DECORATION_HEIGHT_IN,
        ),
    )

    for ax, panel, letter in zip(axes, PANELS, "abcdefg"):
        plot_panel(ax, panel)
        ax.text(
            -0.3,
            1.02,
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
