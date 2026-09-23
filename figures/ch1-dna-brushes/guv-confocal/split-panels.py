"""Split the composite figure guv-confocal-analysis.svg into its panels.

Both panels are taken from the composite, where everything sits inside the
pasted matplotlib group "figure_1".  Panel a is the membrane fluorescence time
course for runs 1 and 3 produced by plotting.py, as tidied in Inkscape (run
labels, legend placement, tick labels) -- so it is extracted rather than
re-rendered, to keep that tidying.  Panel b is the Inkscape schematic of brush
attachment ("+ DNA") and detachment ("+ CD"); its two diagrams are stacked
vertically in the composite, and here the lower pair (label + drawing) is moved
to the right of the upper pair so they sit side by side.  Both panels are
exported to PDF.

Usage:  uv run python split-panels.py
"""

from pathlib import Path

from utils.svg_panels import (
    export,
    extract_panel,
    fit_page_to_drawing,
    pdf_size_mm,
    query_bboxes,
    translate_element,
    union_bbox,
)

HERE = Path(__file__).parent
COMPOSITE = HERE / "guv-confocal-analysis.svg"
CONTAINER = "figure_1"
PANEL_A = ["axes_1", "axes_2"]  # run 1 and run 3 time courses
SCHEMATICS = "g22381"  # group holding both diagrams and their labels
TOP_PAIR = ["g16830", "g17120"]  # "+ DNA" label, attachment diagram
BOTTOM_PAIR = ["g16729", "g16975"]  # "+ CD" label, detachment diagram
GAP_MM = 4.0


def export_panel(svg: Path) -> None:
    pdf = svg.with_suffix(".pdf")
    export(svg, pdf=pdf)
    width, height = pdf_size_mm(pdf)
    print(f"{pdf.name}: {width:.1f} x {height:.1f} mm")


def main() -> None:
    panel_a = HERE / "guv-confocal-analysis-a.svg"
    extract_panel(COMPOSITE, panel_a, keep=PANEL_A, container=CONTAINER)
    fit_page_to_drawing(panel_a)
    export_panel(panel_a)

    panel_b = HERE / "guv-confocal-analysis-b.svg"
    extract_panel(COMPOSITE, panel_b, keep=[SCHEMATICS], container=CONTAINER)

    # Place the lower diagram to the right of the upper one, centred vertically
    boxes = query_bboxes(panel_b)
    top = union_bbox(boxes[i] for i in TOP_PAIR)
    bottom = union_bbox(boxes[i] for i in BOTTOM_PAIR)
    dx = (top[0] + top[2] + GAP_MM) - bottom[0]
    dy = (top[1] + top[3] / 2) - (bottom[1] + bottom[3] / 2)
    for element_id in BOTTOM_PAIR:
        translate_element(panel_b, element_id, dx, dy)

    boxes = query_bboxes(panel_b)
    moved = union_bbox(boxes[i] for i in BOTTOM_PAIR)
    print(
        f"upper pair x=[{top[0]:.1f}, {top[0] + top[2]:.1f}] mm; "
        f"lower pair now x=[{moved[0]:.1f}, {moved[0] + moved[2]:.1f}] mm "
        f"(gap {moved[0] - top[0] - top[2]:.1f} mm)"
    )
    fit_page_to_drawing(panel_b)
    export_panel(panel_b)


if __name__ == "__main__":
    main()
