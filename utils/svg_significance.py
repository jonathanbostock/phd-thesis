"""
Draw significance brackets on a bar chart inside an existing figure SVG.

Some thesis figures are Inkscape composites of several plot panels, so the
brackets are added to the SVG itself instead of re-plotting the panel: the
hand-made layout stays exactly as it is, and the page grows upwards to make
room for the brackets.

A bracket joins two *groups* of elements (the scatter points and/or the bar of
one condition).  Its two ends sit over the centres of the groups, it clears
everything drawn between them, and it carries a centred label ("n.s.", "*",
...).  Brackets are stacked narrowest first at a constant pitch, so a wider
bracket passes over the narrower ones.

All lengths are in points as rendered at the figure's natural size.  The style
follows the hand-drawn "Control" bracket of the calcein-release figures (1 pt
black line with round caps) and the thesis standard of 7 pt DejaVu Sans for
annotations.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

from utils.svg_panels import (
    SVG_NS,
    BBox,
    _page_geometry,
    _write,
    query_bboxes,
    union_bbox,
)

INKSCAPE_NS = "http://www.inkscape.org/namespaces/inkscape"
PT_PER_MM = 72 / 25.4

LINE_WIDTH_PT = 1.0
LEG_PT = 3.0  # length of the two vertical ends of a bracket
CLEARANCE_PT = 2.0  # between a bracket's ends and whatever is drawn below them
LABEL_GAP_PT = 1.5  # between a bracket's line and the ink of its label
FONT_SIZE_PT = 7.0
FONT_FAMILY = "DejaVu Sans"
COLOUR = "#000000"
PAGE_MARGIN_MM = 0.5  # above the topmost label, as for the other thesis figures

GROUP_ID = "significance-brackets"


def _boxes_pt(svg_path: Path) -> Dict[str, BBox]:
    """Visual bounding boxes (x, y, width, height) in points, page coordinates."""
    boxes: Dict[str, BBox] = {}
    for element_id, (x, y, w, h) in query_bboxes(svg_path).items():
        boxes[element_id] = (x * PT_PER_MM, y * PT_PER_MM, w * PT_PER_MM, h * PT_PER_MM)
    return boxes


def _label_ink(label: str) -> Tuple[float, float]:
    """Bottom and top of a label's ink above its baseline, in points.

    Labels are placed by their ink rather than their baseline because the
    asterisk of DejaVu Sans floats well above the baseline.
    """
    extents = TextPath(
        (0, 0), label, size=FONT_SIZE_PT, prop=FontProperties(family=FONT_FAMILY)
    ).get_extents()
    return float(extents.y0), float(extents.y1)


def check_points_match_values(
    svg_path: Path,
    point_ids: Sequence[str],
    values: Sequence[float],
    tolerance_pt: float = 0.05,
) -> None:
    """Check that scatter points drawn in a figure are the given data values.

    A plot maps values linearly onto the page, so the vertical positions of
    the points' centres must be a linear function of the values.  Raises
    ValueError if any point is more than ``tolerance_pt`` away from the
    best-fit line, i.e. if the figure was drawn from different data (or the
    ids are in another order than the values).
    """
    boxes = _boxes_pt(svg_path)
    centres_y = np.array([boxes[i][1] + boxes[i][3] / 2 for i in point_ids])
    data = np.asarray(values, dtype=float)
    slope, intercept = np.polyfit(data, centres_y, 1)
    worst = float(np.abs(centres_y - (slope * data + intercept)).max())
    if worst > tolerance_pt:
        raise ValueError(
            f"points in {svg_path.name} do not match the data: "
            f"largest deviation {worst:.3f} pt (tolerance {tolerance_pt} pt)"
        )


def add_significance_brackets(
    src: Path,
    dst: Path,
    groups: Mapping[str, Sequence[str]],
    comparisons: Sequence[Tuple[str, str, str]],
) -> None:
    """Copy ``src`` to ``dst`` with one labelled bracket per comparison.

    The brackets are stacked above the groups they span, and the page grows
    upwards (its other edges stay put) until the topmost label fits.

    Args:
        src: Figure without brackets
        dst: Where to write the figure with brackets
        groups: Ids of the elements drawn for each condition (its scatter
            points and/or bar), by condition name.  Brackets keep clear of
            every group listed here, compared or not.
        comparisons: (condition, other condition, label) for each bracket
    """
    boxes = _boxes_pt(src)
    group_boxes = {
        name: union_bbox(boxes[element_id] for element_id in element_ids)
        for name, element_ids in groups.items()
    }
    centre = {name: x + w / 2 for name, (x, _, w, _) in group_boxes.items()}

    ink = {label: _label_ink(label) for _, _, label in comparisons}
    label_height = max(top - bottom for bottom, top in ink.values())

    # Narrowest first, so that wider brackets are stacked over narrower ones
    spans = sorted(
        (
            (min(centre[a], centre[b]), max(centre[a], centre[b]), label)
            for a, b, label in comparisons
        ),
        key=lambda span: span[1] - span[0],
    )

    # (left, right, top) of everything a bracket has to clear; y points down
    obstacles: List[Tuple[float, float, float]] = [
        (x, x + w, y) for x, y, w, _ in group_boxes.values()
    ]
    tree = ET.parse(src)
    root = tree.getroot()
    width_mm, height_mm, view_box = _page_geometry(root)
    user_units_per_pt = view_box[2] / (width_mm * PT_PER_MM)

    # Inside this group one unit is one point, measured from the corner of the
    # page as it is in src
    group = ET.SubElement(
        root,
        f"{{{SVG_NS}}}g",
        {
            "id": GROUP_ID,
            "transform": f"translate({view_box[0]:.8g},{view_box[1]:.8g}) "
            f"scale({user_units_per_pt:.8g})",
        },
    )
    label_tops: List[float] = []
    for number, (left, right, label) in enumerate(spans, start=1):
        below = [top for x0, x1, top in obstacles if x1 >= left and x0 <= right]
        leg_end = min(below) - CLEARANCE_PT
        line = leg_end - LEG_PT
        ink_bottom = line - LINE_WIDTH_PT / 2 - LABEL_GAP_PT
        baseline = ink_bottom + ink[label][0]
        obstacles.append((left, right, ink_bottom - label_height))
        label_tops.append(baseline - ink[label][1])

        ET.SubElement(
            group,
            f"{{{SVG_NS}}}path",
            {
                "id": f"{GROUP_ID}-line-{number}",
                "d": f"M {left:.3f},{leg_end:.3f} V {line:.3f} H {right:.3f} "
                f"V {leg_end:.3f}",
                "style": f"fill:none;stroke:{COLOUR};stroke-width:{LINE_WIDTH_PT:g};"
                "stroke-linecap:round;stroke-linejoin:round",
            },
        )
        text = ET.SubElement(
            group,
            f"{{{SVG_NS}}}text",
            {
                "id": f"{GROUP_ID}-label-{number}",
                "x": f"{(left + right) / 2:.3f}",
                "y": f"{baseline:.3f}",
                "style": f"font-size:{FONT_SIZE_PT:g}px;font-family:'{FONT_FAMILY}';"
                f"text-anchor:middle;fill:{COLOUR};stroke:none",
            },
        )
        text.text = label

    # Grow the page upwards (only) until the topmost label fits with a margin
    extra_mm = PAGE_MARGIN_MM - min(label_tops) / PT_PER_MM
    if extra_mm > 0:
        user_units_per_mm = view_box[2] / width_mm
        root.set(
            "viewBox",
            f"{view_box[0]:.8g} {view_box[1] - extra_mm * user_units_per_mm:.8g} "
            f"{view_box[2]:.8g} {view_box[3] + extra_mm * user_units_per_mm:.8g}",
        )
        root.set("width", f"{width_mm:.4f}mm")
        root.set("height", f"{height_mm + extra_mm:.4f}mm")

    # The page is defined by the root element alone: Inkscape's own page
    # element would keep the old size, and makes the PDF export round the page
    # up to whole points
    for parent in list(root.iter()):
        for child in list(parent):
            if child.tag == f"{{{INKSCAPE_NS}}}page":
                parent.remove(child)

    _write(tree, dst)
