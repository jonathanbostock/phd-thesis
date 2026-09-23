"""
Split multi-panel Inkscape figures into single-panel SVG and PDF files.

Several thesis figures are composites: matplotlib plots pasted into Inkscape
alongside hand-drawn panels, with panel letters added on top.  The
``split-panels.py`` script in each figure folder uses this module to pull the
hand-drawn panels back out as standalone files (the plot panels are re-rendered
individually by the folder's plotting script instead):

1. :func:`extract_panel` copies the composite SVG keeping only the elements
   that make up one panel.  ``<defs>``/metadata are always kept so glyphs,
   markers and clip paths still resolve.
2. :func:`translate_element` nudges an element on the page (e.g. to put two
   diagrams side by side instead of stacked).
3. :func:`fit_page_to_drawing` shrinks the page to the remaining drawing.
4. :func:`export` renders PDF/PNG with Inkscape.

Panel letters ("a", "b", ...) are deliberately dropped: each panel becomes its
own numbered figure in the thesis.

All geometry is handled in millimetres.  Inkscape's ``--query-all`` reports
bounding boxes in CSS px (96 per inch) in page coordinates, so those are
converted here.
"""

from __future__ import annotations

import math
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

SVG_NS = "http://www.w3.org/2000/svg"
_NAMESPACES = {
    "": SVG_NS,
    "svg": SVG_NS,
    "inkscape": "http://www.inkscape.org/namespaces/inkscape",
    "sodipodi": "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd",
    "xlink": "http://www.w3.org/1999/xlink",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "cc": "http://creativecommons.org/ns#",
    "dc": "http://purl.org/dc/elements/1.1/",
}
for _prefix, _uri in _NAMESPACES.items():
    ET.register_namespace(_prefix, _uri)

# Elements that are never treated as drawing content
STRUCTURAL_TAGS = {"defs", "metadata", "namedview", "style", "title", "desc"}

INKSCAPE = "inkscape"
PX_PER_MM = 96 / 25.4

BBox = Tuple[float, float, float, float]  # x, y, width, height (mm)
Matrix = Tuple[float, float, float, float, float, float]  # SVG a b c d e f


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _local(tag: str) -> str:
    return tag.split("}")[-1]


def _find_by_id(root: ET.Element, element_id: str) -> ET.Element:
    for element in root.iter():
        if element.get("id") == element_id:
            return element
    raise KeyError(f"no element with id={element_id!r}")


def _length_to_mm(value: str) -> float:
    """Convert an SVG length ('150mm', '425pt', '566.9', ...) to millimetres."""
    match = re.fullmatch(r"\s*([0-9.eE+-]+)\s*([a-zA-Z%]*)\s*", value)
    if match is None:
        raise ValueError(f"cannot parse SVG length {value!r}")
    number, unit = float(match.group(1)), match.group(2).lower()
    per_mm = {
        "": PX_PER_MM,
        "px": PX_PER_MM,
        "mm": 1.0,
        "cm": 0.1,
        "in": 1 / 25.4,
        "pt": 72 / 25.4,
        "pc": 6 / 25.4,
    }
    if unit not in per_mm:
        raise ValueError(f"unsupported SVG unit in {value!r}")
    return number / per_mm[unit]


def _page_geometry(root: ET.Element) -> Tuple[float, float, Sequence[float]]:
    """Return (width_mm, height_mm, viewBox) of the document."""
    width_mm = _length_to_mm(root.get("width", "0"))
    height_mm = _length_to_mm(root.get("height", "0"))
    view_box_attr = root.get("viewBox")
    if view_box_attr is None:
        view_box = [0.0, 0.0, width_mm * PX_PER_MM, height_mm * PX_PER_MM]
    else:
        view_box = [float(v) for v in view_box_attr.replace(",", " ").split()]
    return width_mm, height_mm, view_box


def _user_units_per_mm(root: ET.Element) -> float:
    width_mm, _, view_box = _page_geometry(root)
    return view_box[2] / width_mm


def _matmul(m1: Matrix, m2: Matrix) -> Matrix:
    """Compose SVG matrices: result applies m2 first, then m1."""
    a1, b1, c1, d1, e1, f1 = m1
    a2, b2, c2, d2, e2, f2 = m2
    return (
        a1 * a2 + c1 * b2,
        b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2,
        b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1,
        b1 * e2 + d1 * f2 + f1,
    )


def _parse_transform(transform: Optional[str]) -> Matrix:
    """Parse an SVG transform attribute (translate / scale / matrix / rotate)."""
    result: Matrix = (1, 0, 0, 1, 0, 0)
    if not transform:
        return result
    for name, args in re.findall(r"(\w+)\s*\(([^)]*)\)", transform):
        values = [float(v) for v in re.split(r"[\s,]+", args.strip()) if v]
        if name == "translate":
            tx, ty = values[0], (values[1] if len(values) > 1 else 0.0)
            m: Matrix = (1, 0, 0, 1, tx, ty)
        elif name == "scale":
            sx, sy = values[0], (values[1] if len(values) > 1 else values[0])
            m = (sx, 0, 0, sy, 0, 0)
        elif name == "matrix":
            m = tuple(values)  # type: ignore[assignment]
        elif name == "rotate" and len(values) == 1:
            angle = math.radians(values[0])
            m = (
                math.cos(angle),
                math.sin(angle),
                -math.sin(angle),
                math.cos(angle),
                0,
                0,
            )
        else:
            raise NotImplementedError(f"unsupported transform {name}({args})")
        result = _matmul(result, m)
    return result


def _write(tree, path: Path) -> None:
    tree.write(path, xml_declaration=True, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def query_bboxes(svg_path: Path) -> Dict[str, BBox]:
    """Visual bounding boxes (mm, page coordinates) of every element with an id.

    The first row Inkscape returns is the root ``<svg>`` element, i.e. the bounding
    box of the whole drawing.
    """
    output = subprocess.run(
        [INKSCAPE, "--query-all", str(svg_path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    boxes: Dict[str, BBox] = {}
    for line in output.splitlines():
        parts = line.split(",")
        if len(parts) < 5:
            continue
        try:
            x, y, w, h = (float(v) / PX_PER_MM for v in parts[1:5])
        except ValueError:
            continue
        boxes[parts[0]] = (x, y, w, h)
    return boxes


def union_bbox(boxes: Iterable[BBox]) -> BBox:
    boxes = list(boxes)
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return (x0, y0, x1 - x0, y1 - y0)


def extract_panel(
    src: Path,
    dst: Path,
    *,
    keep: Optional[Iterable[str]] = None,
    drop: Optional[Iterable[str]] = None,
    container: Optional[str] = None,
) -> list:
    """Copy ``src`` to ``dst`` keeping only one panel's elements.

    Exactly one of ``keep`` / ``drop`` must be given: the ids of the children of
    ``container`` (the root ``<svg>`` by default) to keep, or to remove.
    Structural elements (defs, metadata, ...) are always kept.  When a
    ``container`` is given, any other drawing content outside it is removed too.
    Returns the ids that were removed.
    """
    if (keep is None) == (drop is None):
        raise ValueError("pass exactly one of keep= or drop=")
    tree = ET.parse(src)
    root = tree.getroot()
    parent_map = {child: parent for parent in root.iter() for child in parent}

    parent = root if container is None else _find_by_id(root, container)
    child_ids = {child.get("id") for child in parent}
    wanted = set(keep or []) | set(drop or [])
    missing = wanted - child_ids
    if missing:
        raise KeyError(
            f"not direct children of {container or '<svg>'}: {sorted(missing)}"
        )

    removed = []
    for child in list(parent):
        if _local(child.tag) in STRUCTURAL_TAGS:
            continue
        child_id = child.get("id")
        remove = (child_id not in wanted) if keep is not None else (child_id in wanted)
        if remove:
            parent.remove(child)
            removed.append(child_id)

    if container is not None:
        # Remove everything outside the container (except its ancestors)
        ancestors = set()
        node = parent
        while node in parent_map:
            ancestors.add(node)
            node = parent_map[node]
        for element in list(root):
            if _local(element.tag) in STRUCTURAL_TAGS or element in ancestors:
                continue
            root.remove(element)
            removed.append(element.get("id"))

    _write(tree, dst)
    return removed


def translate_element(
    svg_path: Path, element_id: str, dx_mm: float, dy_mm: float
) -> None:
    """Move an element by (dx, dy) millimetres on the page.

    The shift is expressed in the element's parent coordinate system (so it is
    correct inside scaled groups) and prepended to any existing transform.
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()
    parent_map = {child: parent for parent in root.iter() for child in parent}
    element = _find_by_id(root, element_id)

    # Cumulative transform of all ancestors, outermost first
    chain = []
    node = parent_map.get(element)
    while node is not None:
        chain.append(node)
        node = parent_map.get(node)
    ctm: Matrix = (1, 0, 0, 1, 0, 0)
    for ancestor in reversed(chain):
        ctm = _matmul(ctm, _parse_transform(ancestor.get("transform")))

    uu_per_mm = _user_units_per_mm(root)
    dx, dy = dx_mm * uu_per_mm, dy_mm * uu_per_mm
    a, b, c, d, _, _ = ctm
    det = a * d - b * c
    local_dx = (d * dx - c * dy) / det
    local_dy = (-b * dx + a * dy) / det

    old = (element.get("transform") or "").strip()
    element.set(
        "transform",
        f"translate({local_dx:.6g},{local_dy:.6g})" + (f" {old}" if old else ""),
    )
    _write(tree, svg_path)


def fit_page_to_drawing(svg_path: Path, margin_mm: float = 0.5) -> BBox:
    """Resize the page (width/height/viewBox) to the drawing plus a margin.

    Returns the new page bounding box in the old page coordinates (mm).
    """
    boxes = query_bboxes(svg_path)
    tree = ET.parse(svg_path)
    root = tree.getroot()
    drawing = boxes.get(root.get("id") or "") or next(iter(boxes.values()))
    x, y, w, h = drawing
    x, y, w, h = x - margin_mm, y - margin_mm, w + 2 * margin_mm, h + 2 * margin_mm

    _, _, view_box = _page_geometry(root)
    uu_per_mm = _user_units_per_mm(root)
    root.set(
        "viewBox",
        f"{view_box[0] + x * uu_per_mm:.6g} {view_box[1] + y * uu_per_mm:.6g} "
        f"{w * uu_per_mm:.6g} {h * uu_per_mm:.6g}",
    )
    root.set("width", f"{w:.4f}mm")
    root.set("height", f"{h:.4f}mm")
    _write(tree, svg_path)
    return (x, y, w, h)


def export(
    svg_path: Path,
    *,
    pdf: Optional[Path] = None,
    png: Optional[Path] = None,
    png_dpi: int = 150,
) -> None:
    """Export the page area of ``svg_path`` to PDF and/or PNG with Inkscape."""
    actions = []
    if pdf is not None:
        actions += [
            "export-type:pdf",
            f"export-filename:{pdf}",
            "export-area-page",
            "export-do",
        ]
    if png is not None:
        actions += [
            "export-type:png",
            f"export-filename:{png}",
            f"export-dpi:{png_dpi}",
            "export-area-page",
            "export-do",
        ]
    if not actions:
        return
    subprocess.run(
        [INKSCAPE, str(svg_path), f"--actions={';'.join(actions)}"],
        capture_output=True,
        text=True,
        check=True,
    )


def pdf_size_mm(pdf_path: Path) -> Tuple[float, float]:
    """Page size of a PDF in millimetres (via pdfinfo)."""
    info = subprocess.run(
        ["pdfinfo", str(pdf_path)], capture_output=True, text=True, check=True
    ).stdout
    match = re.search(r"Page size:\s+([0-9.]+) x ([0-9.]+) pts", info)
    if match is None:
        raise RuntimeError(f"could not read page size of {pdf_path}")
    return float(match.group(1)) / 72 * 25.4, float(match.group(2)) / 72 * 25.4
