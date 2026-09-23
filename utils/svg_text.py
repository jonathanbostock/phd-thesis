"""
Text-style helpers for figure SVGs: measure, resize and re-font text.

Two kinds of text occur in the thesis figures:

* real ``<text>``/``<tspan>`` elements typed in Inkscape, whose size is the
  ``font-size`` in their style attribute; and
* matplotlib text rendered as glyph outlines: a ``<g>`` whose transform's scale
  encodes the font size (matplotlib draws glyphs at 100 units per em) and whose
  children are ``<use>`` references to glyph paths in ``<defs>``.

All sizes here are in points *as rendered when the SVG is exported at its
natural size*, i.e. every ancestor transform and the page's unit scale are
taken into account.  The thesis standard is 7 pt for tick labels, legends and
annotations, 8 pt for axis labels, 12 pt bold for panel letters, all DejaVu Sans.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from utils.svg_panels import _matmul, _page_geometry, _parse_transform, _write

XLINK = "{http://www.w3.org/1999/xlink}"
GLYPH_REF = re.compile(
    r"#([A-Za-z]+(?:-Oblique|-Bold|-BoldOblique)?)-([0-9a-fA-F]+)(?:-\d+)?$"
)
FONT_SIZE = re.compile(r"font-size:\s*([0-9.]+)([a-z%]*)")
FONT_FAMILY = re.compile(r"font-family:\s*'?([^;'\"]+?)'?\s*(?=;|$)")


def _local(tag: str) -> str:
    return tag.split("}")[-1]


def _scale(m) -> float:
    a, b, c, d, _, _ = m
    return math.sqrt(abs(a * d - b * c))


class _Document:
    """Parsed SVG with cached cumulative transforms and unit conversion."""

    def __init__(self, path):
        self.path = Path(path)
        self.tree = ET.parse(self.path)
        self.root = self.tree.getroot()
        width_mm, _, view_box = _page_geometry(self.root)
        self.uu_per_mm = view_box[2] / width_mm
        self.pt_per_uu = 72 / 25.4 / self.uu_per_mm
        self.parent = {child: parent for parent in self.root.iter() for child in parent}
        self._ctm: dict = {}

    def ctm(self, element):
        if element in self._ctm:
            return self._ctm[element]
        if element is self.root:
            m = (1, 0, 0, 1, 0, 0)
        else:
            m = _matmul(
                self.ctm(self.parent[element]),
                _parse_transform(element.get("transform")),
            )
        self._ctm[element] = m
        return m

    def font_size_uu(self, style: Optional[str]) -> Optional[float]:
        """font-size from a style string, in user units (None if absent or relative)."""
        m = FONT_SIZE.search(style or "")
        if not m or m.group(2) == "%":
            return None
        value, unit = float(m.group(1)), m.group(2) or "px"
        per_uu = {
            "px": 1.0,
            "pt": (25.4 / 72) * self.uu_per_mm,
            "mm": self.uu_per_mm,
        }.get(unit)
        return None if per_uu is None else value * per_uu

    def glyph_runs(self) -> List[Tuple[ET.Element, float, str]]:
        """(group, size_pt, text) for every matplotlib glyph run."""
        runs = []
        for element in self.root.iter():
            if _local(element.tag) != "g" or not element.get("transform"):
                continue
            glyphs = [
                GLYPH_REF.search(child.get(XLINK + "href") or "")
                for child in element
                if _local(child.tag) == "use"
            ]
            matches = [m for m in glyphs if m is not None]
            if not matches:
                continue
            size_pt = 100 * _scale(self.ctm(element)) * self.pt_per_uu
            text = "".join(chr(int(m.group(2), 16)) for m in matches)
            runs.append((element, size_pt, text))
        return runs

    def text_nodes(self) -> List[Tuple[ET.Element, ET.Element, float, str]]:
        """(node carrying the size, owning <text>, size_pt, text) for real text.

        A <tspan> with its own font-size is reported on its own; otherwise the
        <text> element is reported once with its inherited size.
        """
        nodes = []
        for text_el in self.root.iter():
            if _local(text_el.tag) != "text":
                continue
            spans = [
                k
                for k in text_el.iter()
                if _local(k.tag) == "tspan" and self.font_size_uu(k.get("style"))
            ]
            for node in spans or [text_el]:
                size_uu = self.font_size_uu(node.get("style")) or self.font_size_uu(
                    text_el.get("style")
                )
                content = "".join(node.itertext()).strip()
                if size_uu is None or not content:
                    continue
                nodes.append(
                    (
                        node,
                        text_el,
                        size_uu * _scale(self.ctm(text_el)) * self.pt_per_uu,
                        content,
                    )
                )
        return nodes

    def save(self) -> None:
        _write(self.tree, self.path)


def _half_pt(size: float) -> float:
    return round(size * 2) / 2


def text_sizes(svg_path) -> Tuple[Counter, Counter]:
    """(real text, glyph runs): Counter of size (pt, to 0.5) -> number of characters."""
    doc = _Document(svg_path)
    real, glyph = Counter(), Counter()
    for _, _, size_pt, content in doc.text_nodes():
        real[_half_pt(size_pt)] += len(content)
    for _, size_pt, content in doc.glyph_runs():
        glyph[_half_pt(size_pt)] += len(content)
    return real, glyph


def texts_at(svg_path, size_pt: float, tol: float = 0.3) -> Tuple[List[str], List[str]]:
    """The strings of (real text, glyph runs) currently rendering at about size_pt."""
    doc = _Document(svg_path)
    real = [c for _, _, s, c in doc.text_nodes() if abs(s - size_pt) <= tol]
    glyph = [c for _, s, c in doc.glyph_runs() if abs(s - size_pt) <= tol]
    return real, glyph


def font_families(svg_path) -> Counter:
    """font-family -> number of characters, for real text."""
    doc = _Document(svg_path)
    families: Counter = Counter()
    for node, owner, _, content in doc.text_nodes():
        m = FONT_FAMILY.search(node.get("style") or "") or FONT_FAMILY.search(
            owner.get("style") or ""
        )
        families[m.group(1).strip() if m else "(inherited)"] += len(content)
    return families


def retarget_text_sizes(
    svg_path,
    from_pt: float,
    to_pt: float,
    *,
    tol: float = 0.3,
    only_ids: Optional[Iterable[str]] = None,
) -> List[str]:
    """Change every real text node rendering at about from_pt so it renders at to_pt.

    Percentage-sized tspans (super/subscripts) stay relative to their parent.
    Returns the strings that were changed.
    """
    doc = _Document(svg_path)
    wanted = set(only_ids) if only_ids is not None else None
    changed = []
    for node, owner, size_pt, content in doc.text_nodes():
        if abs(size_pt - from_pt) > tol:
            continue
        if (
            wanted is not None
            and node.get("id") not in wanted
            and owner.get("id") not in wanted
        ):
            continue
        target = node if doc.font_size_uu(node.get("style")) else owner
        factor = to_pt / size_pt
        target.set(
            "style",
            FONT_SIZE.sub(
                lambda m: f"font-size:{float(m.group(1)) * factor:.5g}{m.group(2)}",
                target.get("style") or "",
                count=1,
            ),
        )
        changed.append(content)
    if changed:
        doc.save()
    return changed


def rescale_glyph_runs(
    svg_path, from_pt: float, to_pt: float, *, tol: float = 0.3
) -> List[str]:
    """Resize matplotlib glyph runs rendering at about from_pt to to_pt.

    Each run is scaled about its own anchor (the text origin), so labels keep
    their position and grow along their baseline.  Returns the strings changed.
    """
    doc = _Document(svg_path)
    factor = to_pt / from_pt
    changed = []
    for group, size_pt, content in doc.glyph_runs():
        if abs(size_pt - from_pt) > tol:
            continue
        a, b, c, d, e, f = _parse_transform(group.get("transform"))
        group.set(
            "transform",
            f"matrix({a * factor:.6g},{b * factor:.6g},{c * factor:.6g},{d * factor:.6g},{e:.6g},{f:.6g})",
        )
        changed.append(content)
    if changed:
        doc.save()
    return changed


def replace_font_families(svg_path, mapping: Dict[str, str]) -> int:
    """Rewrite font-family (and Inkscape's font specification) for the given families.

    Works on the raw file text so the rest of the SVG is untouched.  Returns the
    number of replacements.
    """
    path = Path(svg_path)
    text = path.read_text()
    count = 0
    for old, new in mapping.items():
        family = re.compile(
            r"(font-family:\s*)'?" + re.escape(old) + r"'?(?=\s*[;,\"'])"
        )
        text, n = family.subn(lambda m: f"{m.group(1)}'{new}'", text)
        count += n
        spec = re.compile(
            r"(-inkscape-font-specification:\s*)'?"
            + re.escape(old)
            + r"([^;\"']*?)'?(?=\s*[;\"'])"
        )
        text, n = spec.subn(lambda m: f"{m.group(1)}'{new}{m.group(2)}'", text)
        count += n
    if count:
        path.write_text(text)
    return count
