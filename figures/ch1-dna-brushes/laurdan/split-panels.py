"""Split the composite figure laurdan-gp.svg into its panels.

Panel a (Laurdan structure and the two emission-mode diagrams, drawn in
Inkscape) is everything in the composite except the bar chart and the panel
letters.  Panel b (general polarization bar chart) is rendered by
analysis-1.py as laurdan-gp-b.svg.  Both are exported to PDF.

Usage:  uv run python split-panels.py   (after analysis-1.py)
"""

from pathlib import Path

from utils.svg_panels import export, extract_panel, fit_page_to_drawing, pdf_size_mm

HERE = Path(__file__).parent
COMPOSITE = HERE / "laurdan-gp.svg"
NOT_PANEL_A = ["g2518", "text784", "text788"]  # bar chart, letter "a", letter "b"


def main() -> None:
    panel_a = HERE / "laurdan-gp-a.svg"
    extract_panel(COMPOSITE, panel_a, drop=NOT_PANEL_A)
    fit_page_to_drawing(panel_a)

    for stem in ["laurdan-gp-a", "laurdan-gp-b"]:
        pdf = HERE / f"{stem}.pdf"
        export(HERE / f"{stem}.svg", pdf=pdf)
        width, height = pdf_size_mm(pdf)
        print(f"{pdf.name}: {width:.1f} x {height:.1f} mm")


if __name__ == "__main__":
    main()
