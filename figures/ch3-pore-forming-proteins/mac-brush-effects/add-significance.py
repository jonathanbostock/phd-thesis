"""Paired t-tests for the MAC leak assay, drawn as brackets on mac-leakage.svg.

The thesis figure (bar chart of the final release next to the timecourses) is
an Inkscape composite: its panels were drawn by a version of the plotting code
that is not in this repository (plotting.py now styles them differently), and
the "Control" bracket and panel letters were added by hand.  Its source was
never committed, so mac-leakage-no-significance.svg was recovered from the PDF
used in the thesis up to now:

    inkscape mac-leakage.pdf --pdf-font-strategy=keep --export-type=svg

The figure is therefore not re-plotted.  This script

1. compares the calcein release at the end of the timecourse (60 min) of the
   brush-free vesicles with MAC against each brush-coated sample with MAC,
   using two-sided paired t-tests.  The three replicates of a condition are
   paired by plate row (A, B, C), and the p-values are not corrected for
   multiple comparisons.  The results are written to paired_t_tests.csv.
   (p_values.csv, written by plotting.py, holds the unpaired tests.)
2. draws a bracket for each comparison above the bar chart (n.s. for
   p >= 0.05, * for p < 0.05, ** for p < 0.01, *** for p < 0.001), growing the
   page upwards to make room, and saves the result as mac-leakage.svg / .pdf.

Usage:  uv run python add-significance.py
"""

from pathlib import Path

from plotting import load_release  # the plotting script next to this one

from utils.stats import paired_t_tests_against
from utils.svg_panels import export, pdf_size_mm
from utils.svg_significance import (
    add_significance_brackets,
    check_points_match_values,
)

HERE = Path(__file__).resolve().parent
WITHOUT_BRACKETS = HERE / "mac-leakage-no-significance.svg"
FIGURE = HERE / "mac-leakage.svg"

REFERENCE = "No Brush"
# The conditions with MAC: the wells of their three replicates, and the ids of
# the scatter points drawn for those wells in the figure (left to right)
CONDITIONS = {
    "No Brush": (["A5", "B5", "C5"], ["path143", "path144", "path145"]),
    "Sparse Brush": (["A6", "B6", "C6"], ["path146", "path147", "path148"]),
    "Dense Brush": (["A7", "B7", "C7"], ["path149", "path150", "path151"]),
    "Star Brush": (["A8", "B8", "C8"], ["path152", "path153", "path154"]),
}


def main() -> None:
    final_release = load_release().iloc[-1]
    values = {
        name: [float(final_release[well]) for well in wells]
        for name, (wells, _) in CONDITIONS.items()
    }
    points = {name: point_ids for name, (_, point_ids) in CONDITIONS.items()}

    # The figure was not drawn by this script: make sure it shows these wells
    check_points_match_values(
        WITHOUT_BRACKETS,
        [point_id for point_ids in points.values() for point_id in point_ids],
        [value for group in values.values() for value in group],
    )

    tests = paired_t_tests_against(REFERENCE, values)
    tests.rename(index=lambda name: f"{REFERENCE} vs {name}").to_csv(
        HERE / "paired_t_tests.csv"
    )
    print(tests.to_string())

    add_significance_brackets(
        WITHOUT_BRACKETS,
        FIGURE,
        points,
        [
            (REFERENCE, str(name), str(label))
            for name, label in tests["significance"].items()
        ],
    )
    pdf = FIGURE.with_suffix(".pdf")
    export(FIGURE, pdf=pdf)
    width, height = pdf_size_mm(pdf)
    print(f"{pdf.name}: {width:.1f} x {height:.1f} mm")


if __name__ == "__main__":
    main()
