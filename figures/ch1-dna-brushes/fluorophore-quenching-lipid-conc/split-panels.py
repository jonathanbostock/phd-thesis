"""Split the composite figure static-fluorophore-quenching.svg into its panels.

Both panels are taken from the composite, where everything sits inside the
pasted matplotlib group "figure_1".  Panel a is the Inkscape cartoon of the two
constructs far apart / close together.  Panel b is the fluorescence against SUV
volume plot produced by analysis.py ("before" cyclodextrin), as annotated in
Inkscape with micelle / vesicle cartoons and a colour-graded trend line -- so it
is extracted here rather than re-rendered, to keep those annotations.  Both
panels are exported to PDF.

Usage:  uv run python split-panels.py
"""

from pathlib import Path

from utils.svg_panels import export, extract_panel, fit_page_to_drawing, pdf_size_mm

HERE = Path(__file__).parent
COMPOSITE = HERE / "static-fluorophore-quenching.svg"
CONTAINER = "figure_1"
PANELS = {
    "a": ["g30286"],  # construct cartoon with its two captions
    "b": ["g29106"],  # annotated fluorescence plot
}


def main() -> None:
    for letter, element_ids in PANELS.items():
        svg = HERE / f"static-fluorophore-quenching-{letter}.svg"
        pdf = svg.with_suffix(".pdf")
        extract_panel(COMPOSITE, svg, keep=element_ids, container=CONTAINER)
        fit_page_to_drawing(svg)
        export(svg, pdf=pdf)
        width, height = pdf_size_mm(pdf)
        print(f"{pdf.name}: {width:.1f} x {height:.1f} mm")


if __name__ == "__main__":
    main()
