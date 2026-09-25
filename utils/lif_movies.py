"""Read Leica ``.lif`` confocal movies and render frames for figures.

The GUV confocal experiments were recorded as ``.lif`` files (one per
condition, e.g. ``After DNA brush addition.lif``), each holding a few single
snapshots plus one long time series.  :func:`load_lif_movie` returns the raw
fluorescence stack of such a series; the remaining helpers implement the
display transform used for the thesis and paper figures: intensities are
clipped to the 2nd and 98th percentiles of the *whole movie* (all frames
pooled, see :func:`percentile_bounds`) and shown as green on black
(:func:`render_green`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
from readlif.reader import LifFile

LOW_PERCENTILE = 2.0
HIGH_PERCENTILE = 98.0


@dataclass(frozen=True)
class LifMovie:
    """One time series from a ``.lif`` file."""

    path: Path
    series_name: str
    fluorescence: np.ndarray
    """Raw integer intensities, shape (frames, height, width)."""
    px_per_um: float
    frame_interval_s: Optional[float]

    @property
    def n_frames(self) -> int:
        return int(self.fluorescence.shape[0])

    @property
    def shape(self) -> Tuple[int, int]:
        """(height, width) of a single frame in pixels."""
        return int(self.fluorescence.shape[1]), int(self.fluorescence.shape[2])


def list_series(path: Path) -> List[Tuple[str, int, Tuple[int, int]]]:
    """Return ``(name, n_frames, (height, width))`` for every series in a file."""
    lif = LifFile(str(path))
    return [
        (str(image.name), int(image.dims.t), (int(image.dims.y), int(image.dims.x)))
        for image in lif.get_iter_image()
    ]


def load_lif_movie(
    path: Path, *, series: Optional[str] = None, channel: int = 0
) -> LifMovie:
    """Load one channel of one time series from ``path``.

    ``series`` is the series name inside the file (e.g. ``"Series003"``); when
    omitted, the series with the most frames is used.  ``channel`` 0 is the
    fluorescence channel and channel 1 the transmitted-light channel in the
    GUV acquisitions.
    """
    lif = LifFile(str(path))
    images: List[Any] = list(lif.get_iter_image())
    if not images:
        raise ValueError(f"No series found in {path}")

    if series is None:
        image = max(images, key=lambda im: int(im.dims.t))
    else:
        matches = [im for im in images if str(im.name) == series]
        if not matches:
            names = ", ".join(str(im.name) for im in images)
            raise ValueError(f"Series {series!r} not in {path.name} (have: {names})")
        image = matches[0]

    n_frames = int(image.dims.t)
    stack = np.stack(
        [np.asarray(image.get_frame(z=0, t=t, c=channel)) for t in range(n_frames)]
    )

    # readlif's ``scale`` is (x, y, z, t): pixels per micron for x/y/z and
    # frames per second for t; z and t are None when not applicable.
    scale = tuple(image.scale)
    px_per_um = float(scale[0])
    frames_per_s = scale[3] if len(scale) > 3 else None
    frame_interval_s = 1.0 / float(frames_per_s) if frames_per_s else None

    return LifMovie(
        path=path,
        series_name=str(image.name),
        fluorescence=stack,
        px_per_um=px_per_um,
        frame_interval_s=frame_interval_s,
    )


def blank_frames(stack: np.ndarray) -> List[int]:
    """Indices of frames with constant intensity.

    Stopping an acquisition part-way through a frame leaves an empty (all
    zero) final frame in the file; such frames carry no data and are left out
    of the percentile calculation.
    """
    flat = stack.reshape(stack.shape[0], -1)
    constant = flat.max(axis=1) == flat.min(axis=1)
    return [int(i) for i in np.flatnonzero(constant)]


def percentile_bounds(
    stack: np.ndarray,
    low: float = LOW_PERCENTILE,
    high: float = HIGH_PERCENTILE,
    *,
    exclude_blank: bool = True,
) -> Tuple[float, float]:
    """Intensity values at the ``low`` and ``high`` percentiles of a movie.

    The percentiles are taken over every pixel of every frame of ``stack``
    pooled together (minus blank frames, see :func:`blank_frames`), so that a
    single mapping applies to all frames of the movie and intensities remain
    comparable between frames.
    """
    keep = np.ones(stack.shape[0], dtype=bool)
    if exclude_blank:
        keep[blank_frames(stack)] = False
    if not keep.any():
        raise ValueError("Every frame of the movie is blank")
    lo, hi = np.percentile(stack[keep], [low, high])
    if hi <= lo:
        raise ValueError(f"Degenerate intensity bounds: p{low}={lo}, p{high}={hi}")
    return float(lo), float(hi)


def block_mean(image: np.ndarray, factor: int) -> np.ndarray:
    """Downsample a 2-D image by averaging ``factor`` x ``factor`` blocks."""
    if factor < 1:
        raise ValueError("factor must be >= 1")
    array = np.asarray(image, dtype=np.float32)
    if factor == 1:
        return array
    height, width = array.shape
    height -= height % factor
    width -= width % factor
    blocks = array[:height, :width].reshape(
        height // factor, factor, width // factor, factor
    )
    return blocks.mean(axis=(1, 3))


def render_green(
    frame: np.ndarray, lo: float, hi: float, *, downsample: int = 1
) -> np.ndarray:
    """Render one frame as an 8-bit RGB image, green on black.

    Intensities are mapped linearly so that ``lo`` becomes black and ``hi``
    full green, and values outside that range are clipped -- i.e. percentile
    clipping when ``lo``/``hi`` come from :func:`percentile_bounds`.  With
    ``downsample`` > 1 the frame is block-averaged first (the bounds still
    refer to the raw movie intensities).
    """
    image = block_mean(frame, downsample)
    scaled = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    rgb = np.zeros((*scaled.shape, 3), dtype=np.uint8)
    rgb[..., 1] = np.round(scaled * 255.0).astype(np.uint8)
    return rgb
