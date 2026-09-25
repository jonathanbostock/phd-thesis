"""Re-render the GUV confocal frames in dna-frames.svg and cyclodextrin-frames.svg.

The two figures were laid out by plot_frames.py and then tidied by hand in
Inkscape (panels removed, rows tightened, one shared row of "t = N" labels), so
they are not regenerated from scratch.  Instead each embedded frame is
re-rendered here from the raw .lif movie with the display transform used in
the paper -- intensities clipped to the 2nd and 98th percentiles of the *whole
movie* (all frames pooled; see utils/lif_movies.py) and shown green on black --
and the new PNG is swapped into the existing <image> element, leaving the
layout untouched.  The originals, whose frames were gamma corrected
(gamma = 0.5) by an earlier version of the movie viewer in the
guv-membrane-fluorescence-labelling repo, are kept as dna-frames-gamma.svg and
cyclodextrin-frames-gamma.svg; those are the inputs of this script, so it can
be re-run safely.

Which movie frame sits behind each <image> was recovered by matching the old
embedded images against the movies (Spearman rank correlation > 0.87 for every
frame, with the next-best frame far behind) and agrees with the "t = N"
headers, the movies having been acquired at one frame per minute.  Run N of
the figure is sample-N of the data folder.

The last frame of each row used to carry a 50 um scale bar baked into the
pixels; these are replaced by vector scale bars (white bar plus a "50 µm"
label in 7 pt DejaVu Sans, like the other thesis figures) whose length is set
from the pixel scale stored in the .lif file.

The old PDFs had been exported with inconsistent areas (the DNA one carried
about 9 mm of blank space below the last row, the cyclodextrin one was cropped
to the drawing), so before export the page is fitted to the drawing with a
0.5 mm margin, as for the other thesis figures.

The rendered frames are also written to run-N/<condition>-frame<NNN>.png
(git-ignored), which is the input layout plot_frames.py expects.

Usage:  uv run python export_frames.py [--data-dir DIR]

DIR holds the sample-N folders with the .lif files and defaults to the
sibling checkout of guv-membrane-fluorescence-labelling.
"""

from __future__ import annotations

import argparse
import base64
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
from PIL import Image

from utils.lif_movies import (
    HIGH_PERCENTILE,
    LOW_PERCENTILE,
    blank_frames,
    load_lif_movie,
    percentile_bounds,
    render_green,
)
from utils.svg_panels import export, pdf_size_mm, query_bboxes

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_DATA_DIR = REPO_ROOT.parent / "guv-membrane-fluorescence-labelling" / "data"

# Embedded frames are DISPLAY_PX square: the 512 px movies are used as they
# are and the 1024 px movies (run 3) are 2 x 2 block-averaged.
DISPLAY_PX = 512

SCALEBAR_UM = 50.0
SCALEBAR_THICKNESS_PT = 1.0
SCALEBAR_FONT_PT = 7.0
SCALEBAR_MARGIN_FRACTION = 0.04  # of the frame's displayed size, as before
SCALEBAR_LABEL_GAP_PT = 1.2

PAGE_MARGIN_MM = 0.5


@dataclass(frozen=True)
class MovieSpec:
    run: int
    sample: str
    lif_name: str
    series: str
    frames: Sequence[Tuple[int, str]]
    """(frame index, id of the <image> element showing it), left to right."""


# fmt: off
FIGURES: Dict[str, Tuple[str, Sequence[MovieSpec]]] = {
    "dna-frames": ("dna", (
        MovieSpec(1, "sample-1", "After DNA brush addition.lif", "Series003",
                  ((0, "imageac72ac8361"), (30, "imageeca0517664"))),
        MovieSpec(2, "sample-2", "After DNA brush addition.lif", "Series002",
                  ((0, "image0de86944e5"), (30, "imaged11cc97abb"),
                   (40, "imaged72df2427a"))),
        MovieSpec(3, "sample-3", "After DNA Brush Addition.lif", "Series006",
                  ((0, "image8c2c529b93"), (30, "imageb02734a179"),
                   (40, "image1adcb2eac6"), (60, "image7dda177101"),
                   (105, "imagef886b4d71c"))),
    )),
    "cyclodextrin-frames": ("cyclodextrin", (
        MovieSpec(1, "sample-1", "After cyclodextrin addition.lif", "Series002",
                  ((0, "imagec0e78c7c15"), (15, "imagee4f4050967"),
                   (30, "image5dd52a86a0"))),
        MovieSpec(2, "sample-2", "After cyclodextrin addition.lif", "Series002",
                  ((0, "imagef1acbc96ad"), (15, "image1b404fdaf3"),
                   (30, "image6683fc2053"))),
        MovieSpec(3, "sample-3", "After Cyclodextrin Addition.lif", "Series002",
                  ((0, "image7b1dbd7ffc"), (15, "imagee17ecef793"),
                   (30, "imagefb0c1cbcf0"))),
    )),
}
# fmt: on

# '>' cannot occur inside base64 data, so a lazy match up to '/>' is safe.
IMAGE_TAG = re.compile(r"<image\b.*?/>", re.S)
ATTRIBUTE = re.compile(r'\b([\w:-]+)="([^"]*)"')
PNG_HREF = re.compile(r'xlink:href="data:image/png;base64,[^"]*"', re.S)
MATRIX = re.compile(r"matrix\(([^)]*)\)")


def find_image_tag(svg: str, image_id: str) -> "re.Match[str]":
    for match in IMAGE_TAG.finditer(svg):
        attrs = dict(ATTRIBUTE.findall(match.group(0)))
        if attrs.get("id") == image_id:
            return match
    raise KeyError(f"No <image id={image_id!r}> in SVG")


def image_geometry(tag: str) -> Tuple[float, float, float, bool]:
    """Return ``(x0, y0, size, flipped)`` of the image's box in its parent group.

    matplotlib writes images with ``transform="matrix(1,0,0,-1,0,H)"`` and the
    pixel data upside down; ``flipped`` reports whether that is the case so the
    replacement PNG can be flipped the same way.
    """
    attrs = dict(ATTRIBUTE.findall(tag))
    x, y, width, height = (float(attrs[k]) for k in ("x", "y", "width", "height"))
    if abs(width - height) > 1e-6:
        raise ValueError("Expected square frames")
    transform = attrs.get("transform", "").strip()
    if not transform:
        return x, y, width, False
    match = MATRIX.fullmatch(transform)
    if match is None:
        raise ValueError(f"Unsupported image transform {transform!r}")
    a, b, c, d, e, f = (float(v) for v in re.split(r"[ ,]+", match.group(1).strip()))
    if b or c or abs(abs(a) - 1) > 1e-9 or abs(abs(d) - 1) > 1e-9:
        raise ValueError(f"Unsupported image transform {transform!r}")
    corners = [
        (a * px + c * py + e, b * px + d * py + f)
        for px, py in ((x, y), (x + width, y + height))
    ]
    xs, ys = zip(*corners)
    return min(xs), min(ys), width, d < 0


def root_tag(svg: str) -> "re.Match[str]":
    root = re.search(r"<svg\b[^>]*>", svg, re.S)
    if root is None:
        raise ValueError("No <svg> root element")
    return root


def user_units_per_mm(svg: str) -> float:
    attrs = dict(ATTRIBUTE.findall(root_tag(svg).group(0)))
    width_mm = re.fullmatch(r"([\d.]+)mm", attrs["width"])
    if width_mm is None:
        raise ValueError(f"Expected the SVG width in mm, got {attrs['width']!r}")
    view_box_width = float(attrs["viewBox"].split()[2])
    return view_box_width / float(width_mm.group(1))


def user_units_per_pt(svg: str) -> float:
    return user_units_per_mm(svg) * 25.4 / 72.0


def fit_page_to_drawing(svg: str, path: Path, margin_mm: float = PAGE_MARGIN_MM) -> str:
    """Set the page (width/height/viewBox) to the drawing plus a margin.

    Text-level counterpart of utils.svg_panels.fit_page_to_drawing, so that the
    rest of the file stays byte-for-byte identical to the original.  ``path``
    must hold the current ``svg`` text, as Inkscape is asked for the bounding
    boxes.
    """
    root = root_tag(svg)
    attrs = dict(ATTRIBUTE.findall(root.group(0)))
    x, y, width, height = query_bboxes(path)[attrs["id"]]  # mm, page coordinates
    x, y = x - margin_mm, y - margin_mm
    width, height = width + 2 * margin_mm, height + 2 * margin_mm

    uu_per_mm = user_units_per_mm(svg)
    view_x, view_y = (float(v) for v in attrs["viewBox"].split()[:2])
    view_box = (
        f"{view_x + x * uu_per_mm:.6g} {view_y + y * uu_per_mm:.6g} "
        f"{width * uu_per_mm:.6g} {height * uu_per_mm:.6g}"
    )
    tag = root.group(0)
    for key, value in (
        ("width", f"{width:.4f}mm"),
        ("height", f"{height:.4f}mm"),
        ("viewBox", view_box),
    ):
        tag, count = re.subn(rf'\b{key}="[^"]*"', f'{key}="{value}"', tag, count=1)
        if count != 1:
            raise ValueError(f"Root <svg> has no {key} attribute")
    return svg[: root.start()] + tag + svg[root.end() :]


def png_bytes(rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def scalebar_snippet(
    image_id: str,
    x0: float,
    y0: float,
    size: float,
    bar_length: float,
    uu_per_pt: float,
    indent: str,
) -> str:
    """SVG for a white scale bar with its label, bottom right of the frame."""
    margin = SCALEBAR_MARGIN_FRACTION * size
    thickness = SCALEBAR_THICKNESS_PT * uu_per_pt
    x1 = x0 + size - margin
    bar_top = y0 + size - margin - thickness
    baseline = bar_top - SCALEBAR_LABEL_GAP_PT * uu_per_pt
    font_size = SCALEBAR_FONT_PT * uu_per_pt
    style = (
        f"font-size:{font_size:g}px;font-family:'DejaVu Sans';"
        "text-anchor:end;fill:#ffffff;stroke:none"
    )
    lines = [
        f"{indent}<g",
        f'{indent}   id="scalebar_{image_id}">',
        f"{indent}  <rect",
        f'{indent}     style="fill:#ffffff;stroke:none"',
        f'{indent}     id="scalebar_{image_id}_bar"',
        f'{indent}     x="{x1 - bar_length:.4f}"',
        f'{indent}     y="{bar_top:.4f}"',
        f'{indent}     width="{bar_length:.4f}"',
        f'{indent}     height="{thickness:.4f}" />',
        f"{indent}  <text",
        f'{indent}     style="{style}"',
        f'{indent}     id="scalebar_{image_id}_label"',
        f'{indent}     x="{x1:.4f}"',
        f'{indent}     y="{baseline:.4f}">{SCALEBAR_UM:g} µm</text>',
        f"{indent}</g>",
    ]
    return "\n".join(lines) + "\n"


def axes_group_end(svg: str, tag_end: int) -> Tuple[int, str]:
    """Locate the ``</g>`` closing the axes group around an image.

    matplotlib nests each image as ``<g id="axes_N"><g clip-path=...><image/>
    </g></g>``.  Returns the offset of the start of the line holding the axes
    group's ``</g>`` and the indentation to use for a new child of that group.
    """
    clip_end = svg.index("</g>", tag_end)
    axes_end = svg.index("</g>", clip_end + 4)
    if svg[tag_end:clip_end].strip() or svg[clip_end + 4 : axes_end].strip():
        raise ValueError("Unexpected content around <image>; layout has changed")
    line_start = svg.rfind("\n", 0, axes_end) + 1
    indent = svg[line_start:axes_end]
    if indent.strip():
        raise ValueError("Expected the axes group's </g> on its own line")
    return line_start, indent + "  "


def process_figure(
    name: str, condition: str, specs: Sequence[MovieSpec], data_dir: Path
) -> None:
    source = HERE / f"{name}-gamma.svg"
    target = HERE / f"{name}.svg"
    svg = source.read_text(encoding="utf-8")
    uu_per_pt = user_units_per_pt(svg)

    print(f"\n{name}: {source.name} -> {target.name}")
    for spec in specs:
        movie = load_lif_movie(
            data_dir / spec.sample / spec.lif_name, series=spec.series
        )
        lo, hi = percentile_bounds(movie.fluorescence)
        height, width = movie.shape
        downsample = width // DISPLAY_PX
        interval = (
            f" every {movie.frame_interval_s:g} s" if movie.frame_interval_s else ""
        )
        print(
            f"  run {spec.run}: {spec.sample}/{spec.lif_name} [{movie.series_name}] "
            f"{movie.n_frames} frames of {width}x{height} px{interval}, "
            f"{movie.px_per_um:.3f} px/um, blank frames "
            f"{blank_frames(movie.fluorescence)}, "
            f"p{LOW_PERCENTILE:g}/p{HIGH_PERCENTILE:g} = {lo:g}/{hi:g}"
        )

        run_dir = HERE / f"run-{spec.run}"
        run_dir.mkdir(exist_ok=True)
        for column, (frame_index, image_id) in enumerate(spec.frames):
            rgb = render_green(
                movie.fluorescence[frame_index], lo, hi, downsample=downsample
            )
            Image.fromarray(rgb, mode="RGB").save(
                run_dir / f"{condition}-frame{frame_index:03d}.png"
            )

            match = find_image_tag(svg, image_id)
            tag = match.group(0)
            x0, y0, size, flipped = image_geometry(tag)
            payload = base64.b64encode(
                png_bytes(np.flipud(rgb) if flipped else rgb)
            ).decode("ascii")
            new_tag, count = PNG_HREF.subn(
                lambda _: f'xlink:href="data:image/png;base64,{payload}"', tag, count=1
            )
            if count != 1:
                raise ValueError(f"<image id={image_id!r}> has no embedded PNG")
            svg = svg[: match.start()] + new_tag + svg[match.end() :]
            tag_end = match.start() + len(new_tag)

            message = (
                f"    frame {frame_index:3d} -> {image_id} "
                f"({rgb.shape[1]}x{rgb.shape[0]} px)"
            )
            if column == len(spec.frames) - 1:
                bar_length = SCALEBAR_UM * movie.px_per_um * size / width
                insert_at, indent = axes_group_end(svg, tag_end)
                snippet = scalebar_snippet(
                    image_id, x0, y0, size, bar_length, uu_per_pt, indent
                )
                svg = svg[:insert_at] + snippet + svg[insert_at:]
                message += (
                    f", {SCALEBAR_UM:g} um scale bar = {bar_length / uu_per_pt:.2f} pt"
                )
            print(message)

    target.write_text(svg, encoding="utf-8")
    svg = fit_page_to_drawing(svg, target)
    target.write_text(svg, encoding="utf-8")
    pdf = HERE / f"{name}.pdf"
    export(target, pdf=pdf)
    width_mm, height_mm = pdf_size_mm(pdf)
    print(
        f"  wrote {target.name} ({target.stat().st_size / 1e6:.2f} MB) and "
        f"{pdf.name} ({width_mm:.1f} x {height_mm:.1f} mm)"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="folder holding sample-N/*.lif",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data_dir.is_dir():
        raise SystemExit(f"Data directory not found: {args.data_dir}")
    for name, (condition, specs) in FIGURES.items():
        process_figure(name, condition, specs, args.data_dir)


if __name__ == "__main__":
    main()
