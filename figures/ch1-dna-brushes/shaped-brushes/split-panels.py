"""Split the composite figure shaped-constructs.svg into its panels.

Panel a (scale schematics of the linear, junction and star constructs, drawn
in Inkscape, with the plot legend entries moved next to the constructs as
labels) is extracted from the composite.  Panels b and c are rendered
individually by shape-analysis.py.  All three are exported to PDF.

Usage:  uv run python split-panels.py   (after shape-analysis.py)
"""

from pathlib import Path

from utils.svg_panels import export, extract_panel, fit_page_to_drawing, pdf_size_mm

HERE = Path(__file__).parent
COMPOSITE = HERE / "shaped-constructs.svg"
# Root-level elements of the composite that make up panel a
PANEL_A = ["text_11", "g11226", "g11237", "g11209", "g11008"]


def main() -> None:
    panel_a = HERE / "shaped-constructs-a.svg"
    extract_panel(COMPOSITE, panel_a, keep=PANEL_A)
    fit_page_to_drawing(panel_a)

    for stem in ["shaped-constructs-a", "shaped-constructs-b", "shaped-constructs-c"]:
        pdf = HERE / f"{stem}.pdf"
        export(HERE / f"{stem}.svg", pdf=pdf)
        width, height = pdf_size_mm(pdf)
        print(f"{pdf.name}: {width:.1f} x {height:.1f} mm")


if __name__ == "__main__":
    main()
