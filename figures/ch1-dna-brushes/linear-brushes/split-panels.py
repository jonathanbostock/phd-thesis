"""Split the composite figure linear-constructs.svg into its panels.

Panel a (scale schematics of the four linear constructs, drawn in Inkscape,
with the plot legend entries moved next to the constructs as labels) is
extracted from the composite.  Panels b and c are rendered individually by
static-brush-plotting.py.  All three are exported to PDF for the thesis.

Usage:  uv run python split-panels.py   (after static-brush-plotting.py)
"""

from pathlib import Path

from utils.svg_panels import export, extract_panel, fit_page_to_drawing, pdf_size_mm

HERE = Path(__file__).parent
COMPOSITE = HERE / "linear-constructs.svg"
# Root-level elements of the composite that make up panel a
PANEL_A = ["layer1", "text_12", "g15405", "g15390", "g15376", "g15362"]


def main() -> None:
    panel_a = HERE / "linear-constructs-a.svg"
    extract_panel(COMPOSITE, panel_a, keep=PANEL_A)
    fit_page_to_drawing(panel_a)

    for stem in ["linear-constructs-a", "linear-constructs-b", "linear-constructs-c"]:
        pdf = HERE / f"{stem}.pdf"
        export(HERE / f"{stem}.svg", pdf=pdf)
        width, height = pdf_size_mm(pdf)
        print(f"{pdf.name}: {width:.1f} x {height:.1f} mm")


if __name__ == "__main__":
    main()
