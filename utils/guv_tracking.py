"""
GUV (Giant Unilamellar Vesicle) membrane fluorescence tracking utilities.

This module provides functions for:
- Loading .lif microscopy files
- Detecting vesicles in brightfield images
- Tracking vesicles across frames
- Measuring membrane fluorescence over time
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trackpy as tp
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Circle
from readlif.reader import LifFile
from skimage import measure
from skimage.draw import disk
from skimage.feature import canny
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.transform import hough_circle, hough_circle_peaks


@dataclass
class TrackingConfig:
    """Configuration parameters for GUV tracking."""

    fluorescence_channel: int = 0  # Channel 0 = fluorescence
    brightfield_channel: int = 1  # Channel 1 = brightfield
    min_diameter_um: float = 5.0  # Lowered from 10 to detect smaller vesicles
    max_diameter_um: float = 50.0
    annulus_width_px: int = 3
    dark_threshold_factor: float = 0.7  # Threshold multiplier for dark region detection
    max_eccentricity: float = 0.7  # Max eccentricity for valid vesicles (0=circle)
    search_range_factor: float = 0.5  # Fraction of diameter for linking
    max_jump_factor: float = 0.5  # Max jump as fraction of diameter
    min_track_fraction: float = 0.25  # Require presence in at least 25% of frames
    link_memory: int = 5  # Allow gaps of up to 5 frames in tracking

    # Detection method: "brightfield" (Hough circles) or "fluorescence" (dark regions)
    detection_method: str = "brightfield"

    # Brightfield circle detection (Hough transform) parameters
    canny_sigma: float = 2.0  # Gaussian smoothing for edge detection
    hough_min_distance: int = 20  # Min distance between circle centers (pixels)
    hough_threshold: float = 0.3  # Accumulator threshold (fraction of max)
    hough_num_peaks: int = 100  # Max circles to detect per frame

    # Fluorescence validation (used with brightfield detection)
    interior_darkness_threshold: float = 0.8  # Interior must be < threshold * Otsu


@dataclass
class GUVDetection:
    """Single GUV detection in one frame."""

    x_px: float  # Center x in pixels
    y_px: float  # Center y in pixels
    radius_px: float  # Radius in pixels
    frame: int  # Frame index
    interior_mean: float = 0.0  # Mean fluorescence inside
    exterior_mean: float = 0.0  # Mean fluorescence outside (annulus)


@dataclass
class GUVTrack:
    """A tracked GUV across multiple frames."""

    track_id: int
    detections: List[GUVDetection] = field(default_factory=list)


def load_lif_series(
    lif_path: Path,
    config: TrackingConfig,
    series_index: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load a .lif file and extract the image series with most timepoints.

    Parameters
    ----------
    lif_path : Path
        Path to .lif file
    config : TrackingConfig
        Configuration with channel indices
    series_index : Optional[int]
        Specific series to load. If None, selects series with most timepoints.

    Returns
    -------
    fluorescence : np.ndarray
        Shape (n_frames, height, width) - fluorescence channel
    brightfield : np.ndarray
        Shape (n_frames, height, width) - brightfield channel
    metadata : dict
        Contains 'pixel_size_um', 'time_interval_s', 'n_frames', 'dimensions'
    """
    lif = LifFile(str(lif_path))
    images = list(lif.get_iter_image())

    if series_index is None:
        # Find series with most timepoints
        max_frames = 0
        best_idx = 0
        for i, img in enumerate(images):
            n_frames = img.dims.t if img.dims.t else 1
            if n_frames > max_frames:
                max_frames = n_frames
                best_idx = i
        series_index = best_idx

    img = images[series_index]

    # Extract metadata
    scale = img.scale
    pixel_size_um = 1.0 / scale[0] if scale and scale[0] else 1.0
    time_interval_s = None  # readlif doesn't always provide this

    n_frames = img.dims.t if img.dims.t else 1
    n_channels = img.channels
    height = img.dims.y
    width = img.dims.x

    # Load all frames for both channels
    brightfield_frames = []
    fluorescence_frames = []

    for t in range(n_frames):
        # Get brightfield channel
        bf_frame = img.get_frame(z=0, t=t, c=config.brightfield_channel)
        brightfield_frames.append(np.array(bf_frame))

        # Get fluorescence channel
        fl_frame = img.get_frame(z=0, t=t, c=config.fluorescence_channel)
        fluorescence_frames.append(np.array(fl_frame))

    brightfield = np.stack(brightfield_frames)
    fluorescence = np.stack(fluorescence_frames)

    metadata = {
        "pixel_size_um": pixel_size_um,
        "time_interval_s": time_interval_s,
        "n_frames": n_frames,
        "n_channels": n_channels,
        "dimensions": (height, width),
        "series_index": series_index,
        "series_name": img.name if hasattr(img, "name") else f"Series {series_index}",
    }

    return fluorescence, brightfield, metadata


def _create_disk_mask(
    center: Tuple[float, float],
    radius: float,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """Create boolean mask for disk region."""
    rr, cc = disk((center[1], center[0]), radius, shape=image_shape)
    mask = np.zeros(image_shape, dtype=bool)
    mask[rr, cc] = True
    return mask


def _create_annulus_mask(
    center: Tuple[float, float],
    inner_radius: float,
    outer_radius: float,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """Create boolean mask for annulus region around vesicle membrane."""
    outer_mask = _create_disk_mask(center, outer_radius, image_shape)
    inner_mask = _create_disk_mask(center, inner_radius, image_shape)
    return outer_mask & ~inner_mask


def _sample_interior_exterior(
    fluorescence_frame: np.ndarray,
    detection: GUVDetection,
    margin_px: int = 5,
) -> Tuple[float, float]:
    """
    Sample mean fluorescence inside and outside a detected circle.

    Parameters
    ----------
    fluorescence_frame : np.ndarray
        Single fluorescence image
    detection : GUVDetection
        Detection with center and radius
    margin_px : int
        Margin to leave between interior/exterior regions

    Returns
    -------
    interior_mean : float
        Mean intensity inside the vesicle
    exterior_mean : float
        Mean intensity in annulus outside the vesicle
    """
    image_shape = fluorescence_frame.shape
    center = (detection.x_px, detection.y_px)
    radius = detection.radius_px

    # Interior: disk with margin from edge
    interior_radius = max(radius - margin_px, 1)
    interior_mask = _create_disk_mask(center, interior_radius, image_shape)

    # Exterior: annulus just outside the vesicle
    exterior_inner = radius + margin_px
    exterior_outer = radius + margin_px + 10  # 10 pixel wide annulus
    exterior_mask = _create_annulus_mask(
        center, exterior_inner, exterior_outer, image_shape
    )

    interior_mean = (
        np.mean(fluorescence_frame[interior_mask]) if np.any(interior_mask) else 0.0
    )
    exterior_mean = (
        np.mean(fluorescence_frame[exterior_mask]) if np.any(exterior_mask) else 0.0
    )

    return float(interior_mean), float(exterior_mean)


def detect_vesicles_dark_regions(
    fluorescence_frame: np.ndarray,
    brightfield_frame: np.ndarray,
    pixel_size_um: float,
    config: TrackingConfig,
    frame_idx: int = 0,
) -> List[GUVDetection]:
    """
    Detect vesicles by finding dark regions in fluorescence, then refine with brightfield.

    This approach works because:
    1. Vesicle interiors exclude the fluorescent dye, appearing dark
    2. The dark regions give us the center and approximate size
    3. We can optionally refine the circle using brightfield edges

    Parameters
    ----------
    fluorescence_frame : np.ndarray
        Fluorescence image (dark interiors = vesicles)
    brightfield_frame : np.ndarray
        Brightfield image (for optional circle refinement)
    pixel_size_um : float
        Pixel size in micrometers
    config : TrackingConfig
        Detection configuration
    frame_idx : int
        Frame index for the detection

    Returns
    -------
    detections : List[GUVDetection]
        Detected vesicles
    """
    # Convert diameter range to pixel areas
    min_radius_px = (config.min_diameter_um / 2) / pixel_size_um
    max_radius_px = (config.max_diameter_um / 2) / pixel_size_um
    min_area = np.pi * min_radius_px**2
    max_area = np.pi * max_radius_px**2

    # Find dark regions in fluorescence
    threshold = threshold_otsu(fluorescence_frame)
    dark_mask = fluorescence_frame < threshold * config.dark_threshold_factor

    # Clean up the mask
    dark_mask = remove_small_objects(dark_mask, max_size=int(min_area * 0.5))
    dark_mask = remove_small_holes(dark_mask, max_size=int(min_area * 0.3))

    # Label connected components and get properties
    labels = measure.label(dark_mask)
    regions = measure.regionprops(labels)

    # Filter by size and shape
    detections = []
    for region in regions:
        # Size filter
        if not (min_area < region.area < max_area):
            continue

        # Shape filter (reject elongated objects)
        if region.eccentricity > config.max_eccentricity:
            continue

        # Get centroid and equivalent radius
        y, x = region.centroid
        radius = np.sqrt(region.area / np.pi)

        # TODO: Could refine circle fit using brightfield edges here
        # For now, use the fluorescence-derived values

        det = GUVDetection(
            x_px=float(x),
            y_px=float(y),
            radius_px=float(radius),
            frame=frame_idx,
        )
        detections.append(det)

    return detections


def detect_circles_brightfield(
    brightfield_frame: np.ndarray,
    pixel_size_um: float,
    config: TrackingConfig,
    frame_idx: int = 0,
) -> List[GUVDetection]:
    """
    Detect circles in brightfield using Hough Circle Transform.

    GUV membranes appear as rings in brightfield, which can be detected
    even when vesicles are clustered together (unlike dark region detection).

    Parameters
    ----------
    brightfield_frame : np.ndarray
        Brightfield image
    pixel_size_um : float
        Pixel size in micrometers
    config : TrackingConfig
        Detection configuration
    frame_idx : int
        Frame index for the detection

    Returns
    -------
    detections : List[GUVDetection]
        Candidate circles (not yet validated with fluorescence)
    """
    # Convert diameter range to pixel radii
    min_radius_px = int((config.min_diameter_um / 2) / pixel_size_um)
    max_radius_px = int((config.max_diameter_um / 2) / pixel_size_um)

    # Normalize brightfield to 0-1 range for edge detection
    bf_min = brightfield_frame.min()
    bf_max = brightfield_frame.max()
    if bf_max > bf_min:
        bf_norm = (brightfield_frame - bf_min) / (bf_max - bf_min)
    else:
        return []  # Empty image

    # Apply Canny edge detection
    edges = canny(bf_norm, sigma=config.canny_sigma)

    # Define radius range to search (step by 1 pixel)
    hough_radii = np.arange(min_radius_px, max_radius_px + 1, 1)

    if len(hough_radii) == 0:
        return []

    # Run Hough Circle Transform
    hough_res = hough_circle(edges, hough_radii)

    # Find circle peaks
    # total_num_peaks limits how many circles we detect
    accums, cx, cy, radii = hough_circle_peaks(
        hough_res,
        hough_radii,
        min_xdistance=config.hough_min_distance,
        min_ydistance=config.hough_min_distance,
        threshold=config.hough_threshold * hough_res.max(),
        num_peaks=config.hough_num_peaks,
        total_num_peaks=config.hough_num_peaks,
    )

    # Convert to GUVDetection objects
    detections = []
    for x, y, radius in zip(cx, cy, radii):
        det = GUVDetection(
            x_px=float(x),
            y_px=float(y),
            radius_px=float(radius),
            frame=frame_idx,
        )
        detections.append(det)

    return detections


def validate_with_fluorescence(
    candidates: List[GUVDetection],
    fluorescence_frame: np.ndarray,
    config: TrackingConfig,
) -> List[GUVDetection]:
    """
    Validate candidate circles by checking if interior is dark in fluorescence.

    GUV interiors exclude the fluorescent dye, so valid vesicles should have
    dark interiors relative to the background.

    Parameters
    ----------
    candidates : List[GUVDetection]
        Candidate circles from brightfield detection
    fluorescence_frame : np.ndarray
        Fluorescence image (dark interiors = vesicles)
    config : TrackingConfig
        Configuration with validation threshold

    Returns
    -------
    validated : List[GUVDetection]
        Circles that passed fluorescence validation
    """
    if not candidates:
        return []

    # Calculate Otsu threshold for the fluorescence frame
    threshold = threshold_otsu(fluorescence_frame)
    darkness_threshold = threshold * config.interior_darkness_threshold

    image_shape = fluorescence_frame.shape
    validated = []

    for det in candidates:
        # Create disk mask for interior (with margin from edge to avoid membrane)
        margin = max(3, int(det.radius_px * 0.2))  # At least 3 pixels or 20% of radius
        interior_radius = det.radius_px - margin

        if interior_radius < 2:
            # Circle too small to measure interior reliably
            continue

        interior_mask = _create_disk_mask(
            (det.x_px, det.y_px), interior_radius, image_shape
        )

        if not np.any(interior_mask):
            continue

        # Calculate mean fluorescence inside
        interior_mean = np.mean(fluorescence_frame[interior_mask])

        # Accept if interior is sufficiently dark
        if interior_mean < darkness_threshold:
            det.interior_mean = float(interior_mean)
            validated.append(det)

    return validated


def detect_vesicles_brightfield_validated(
    fluorescence_frame: np.ndarray,
    brightfield_frame: np.ndarray,
    pixel_size_um: float,
    config: TrackingConfig,
    frame_idx: int = 0,
) -> List[GUVDetection]:
    """
    Detect vesicles using brightfield circles validated by fluorescence.

    Two-step process:
    1. Detect circles in brightfield (Hough transform)
    2. Validate by checking interior is dark in fluorescence

    This approach works better for clustered vesicles because brightfield
    shows membrane boundaries even when vesicles are touching.

    Parameters
    ----------
    fluorescence_frame : np.ndarray
        Fluorescence image (for validation)
    brightfield_frame : np.ndarray
        Brightfield image (for circle detection)
    pixel_size_um : float
        Pixel size in micrometers
    config : TrackingConfig
        Detection configuration
    frame_idx : int
        Frame index for the detection

    Returns
    -------
    detections : List[GUVDetection]
        Validated vesicle detections
    """
    # Step 1: Detect circles in brightfield
    candidates = detect_circles_brightfield(
        brightfield_frame, pixel_size_um, config, frame_idx
    )

    # Step 2: Validate with fluorescence (dark interior check)
    validated = validate_with_fluorescence(candidates, fluorescence_frame, config)

    return validated


def generate_debug_image(
    fluorescence_frame: np.ndarray,
    detections: List[GUVDetection],
    output_path: Path,
    config: TrackingConfig,
    title: str = "Detection Debug",
    brightfield_frame: Optional[np.ndarray] = None,
    candidate_detections: Optional[List[GUVDetection]] = None,
) -> None:
    """
    Generate a debug image showing detection results.

    For fluorescence detection method:
    - Left: Fluorescence with detected circles
    - Right: Otsu threshold overlay (cyan=dark, magenta=bright)

    For brightfield detection method:
    - Left: Brightfield with Canny edges and all candidate circles
    - Right: Fluorescence with validated circles (green) and rejected (red)

    Parameters
    ----------
    fluorescence_frame : np.ndarray
        Single fluorescence image
    detections : List[GUVDetection]
        Validated/detected vesicles for this frame
    output_path : Path
        Where to save the debug image
    config : TrackingConfig
        Configuration with threshold parameters
    title : str
        Title for the plot
    brightfield_frame : Optional[np.ndarray]
        Brightfield image (required for brightfield detection debug)
    candidate_detections : Optional[List[GUVDetection]]
        All candidate circles before validation (for brightfield method)
    """
    if config.detection_method == "brightfield" and brightfield_frame is not None:
        _generate_debug_image_brightfield(
            fluorescence_frame,
            brightfield_frame,
            detections,
            candidate_detections or [],
            output_path,
            config,
            title,
        )
    else:
        _generate_debug_image_fluorescence(
            fluorescence_frame,
            detections,
            output_path,
            config,
            title,
        )


def _generate_debug_image_fluorescence(
    fluorescence_frame: np.ndarray,
    detections: List[GUVDetection],
    output_path: Path,
    config: TrackingConfig,
    title: str,
) -> None:
    """Generate debug image for fluorescence-based detection (original method)."""
    # Calculate Otsu threshold
    threshold = threshold_otsu(fluorescence_frame)
    effective_threshold = threshold * config.dark_threshold_factor

    # Normalize image to 0-1
    img_min = fluorescence_frame.min()
    img_max = fluorescence_frame.max()
    if img_max > img_min:
        img_norm = (fluorescence_frame - img_min) / (img_max - img_min)
    else:
        img_norm = np.zeros_like(fluorescence_frame, dtype=float)

    # Normalize threshold to same scale
    thresh_norm = (effective_threshold - img_min) / (img_max - img_min)

    # Create HSV image where hue indicates position relative to threshold
    height, width = fluorescence_frame.shape
    hsv = np.zeros((height, width, 3), dtype=float)

    # Hue: transition from cyan (0.5) to magenta (0.85) based on intensity
    transition_width = 0.1
    relative_pos = (img_norm - thresh_norm) / transition_width
    hue = 0.5 + 0.35 * (1 / (1 + np.exp(-relative_pos * 5)))
    hsv[:, :, 0] = hue
    hsv[:, :, 1] = 0.7
    hsv[:, :, 2] = 0.3 + 0.7 * img_norm

    rgb = hsv_to_rgb(hsv)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Original fluorescence with detections
    axes[0].imshow(fluorescence_frame, cmap="gray")
    axes[0].set_title("Fluorescence + Detections")

    for det in detections:
        circle = Circle(
            (det.x_px, det.y_px),
            det.radius_px,
            fill=False,
            color="yellow",
            linewidth=1.5,
        )
        axes[0].add_patch(circle)

    axes[0].axis("off")

    # Right: Hue-based threshold overlay
    axes[1].imshow(rgb)
    axes[1].set_title(
        f"Otsu Threshold Overlay\n"
        f"(Otsu={threshold:.0f}, ×{config.dark_threshold_factor}={effective_threshold:.0f})\n"
        f"Cyan=below threshold, Magenta=above"
    )

    for det in detections:
        circle = Circle(
            (det.x_px, det.y_px),
            det.radius_px,
            fill=False,
            color="yellow",
            linewidth=1.5,
        )
        axes[1].add_patch(circle)

    axes[1].axis("off")

    plt.suptitle(f"{title} ({len(detections)} detections)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _generate_debug_image_brightfield(
    fluorescence_frame: np.ndarray,
    brightfield_frame: np.ndarray,
    validated: List[GUVDetection],
    candidates: List[GUVDetection],
    output_path: Path,
    config: TrackingConfig,
    title: str,
) -> None:
    """Generate debug image for brightfield-based detection."""
    # Normalize brightfield for edge detection
    bf_min = brightfield_frame.min()
    bf_max = brightfield_frame.max()
    if bf_max > bf_min:
        bf_norm = (brightfield_frame - bf_min) / (bf_max - bf_min)
    else:
        bf_norm = np.zeros_like(brightfield_frame, dtype=float)

    # Get Canny edges
    edges = canny(bf_norm, sigma=config.canny_sigma)

    # Create RGB overlay for brightfield with edges
    bf_rgb = np.stack([bf_norm, bf_norm, bf_norm], axis=-1)
    # Overlay edges in cyan
    bf_rgb[edges, 0] = 0.0
    bf_rgb[edges, 1] = 1.0
    bf_rgb[edges, 2] = 1.0

    # Get validated detection IDs for comparison
    validated_coords = {(det.x_px, det.y_px, det.radius_px) for det in validated}

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Brightfield with edges and all candidate circles
    axes[0].imshow(bf_rgb)
    axes[0].set_title(
        f"Brightfield + Canny Edges (σ={config.canny_sigma})\n"
        f"Yellow=candidates ({len(candidates)})"
    )

    for det in candidates:
        circle = Circle(
            (det.x_px, det.y_px),
            det.radius_px,
            fill=False,
            color="yellow",
            linewidth=1.0,
            alpha=0.7,
        )
        axes[0].add_patch(circle)

    axes[0].axis("off")

    # Right: Fluorescence with validated (green) vs rejected (red)
    axes[1].imshow(fluorescence_frame, cmap="gray")

    # Calculate threshold for display
    threshold = threshold_otsu(fluorescence_frame)
    darkness_thresh = threshold * config.interior_darkness_threshold

    axes[1].set_title(
        f"Fluorescence Validation\n"
        f"Green=validated ({len(validated)}), Red=rejected ({len(candidates) - len(validated)})\n"
        f"Interior darkness threshold: {darkness_thresh:.0f}"
    )

    # Draw rejected circles (candidates not in validated) in red
    for det in candidates:
        coord = (det.x_px, det.y_px, det.radius_px)
        if coord not in validated_coords:
            circle = Circle(
                (det.x_px, det.y_px),
                det.radius_px,
                fill=False,
                color="red",
                linewidth=1.5,
                alpha=0.7,
            )
            axes[1].add_patch(circle)

    # Draw validated circles in green
    for det in validated:
        circle = Circle(
            (det.x_px, det.y_px),
            det.radius_px,
            fill=False,
            color="lime",
            linewidth=2.0,
        )
        axes[1].add_patch(circle)

    axes[1].axis("off")

    plt.suptitle(f"{title} - Brightfield Detection")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def link_detections(
    all_detections: List[List[GUVDetection]],
    pixel_size_um: float,
    config: TrackingConfig,
) -> List[GUVTrack]:
    """
    Link detections across frames into tracks.

    Parameters
    ----------
    all_detections : List[List[GUVDetection]]
        Per-frame detections
    pixel_size_um : float
        Pixel size in micrometers
    config : TrackingConfig
        Configuration with linking parameters

    Returns
    -------
    tracks : List[GUVTrack]
        Linked tracks
    """
    # Build DataFrame for trackpy
    rows = []
    for frame_idx, frame_dets in enumerate(all_detections):
        for det in frame_dets:
            rows.append(
                {
                    "frame": frame_idx,
                    "x": det.x_px,
                    "y": det.y_px,
                    "radius": det.radius_px,
                    "interior_mean": det.interior_mean,
                    "exterior_mean": det.exterior_mean,
                }
            )

    if not rows:
        return []

    df = pd.DataFrame(rows)

    # Calculate search range based on typical vesicle size
    avg_diameter_um = (config.min_diameter_um + config.max_diameter_um) / 2
    avg_diameter_px = avg_diameter_um / pixel_size_um
    search_range = config.search_range_factor * avg_diameter_px

    # Link trajectories
    tp.quiet()
    try:
        linked = tp.link(df, search_range=search_range, memory=config.link_memory)
    except Exception as e:
        if "Subnetwork" in str(e):
            # Too many points close together - try with smaller search range
            try:
                linked = tp.link(
                    df, search_range=search_range * 0.5, memory=config.link_memory
                )
            except Exception:
                # Give up and return empty
                return []
        else:
            raise

    # Convert to GUVTrack objects
    tracks_dict: Dict[int, GUVTrack] = {}

    for _, row in linked.iterrows():
        track_id = int(row["particle"])
        if track_id not in tracks_dict:
            tracks_dict[track_id] = GUVTrack(track_id=track_id)

        det = GUVDetection(
            x_px=float(row["x"]),
            y_px=float(row["y"]),
            radius_px=float(row["radius"]),
            frame=int(row["frame"]),
            interior_mean=float(row["interior_mean"]),
            exterior_mean=float(row["exterior_mean"]),
        )
        tracks_dict[track_id].detections.append(det)

    # Sort detections by frame within each track
    for track in tracks_dict.values():
        track.detections.sort(key=lambda d: d.frame)

    return list(tracks_dict.values())


def filter_tracks(
    tracks: List[GUVTrack],
    n_frames: int,
    image_shape: Tuple[int, int],
    pixel_size_um: float,
    config: TrackingConfig,
) -> Tuple[List[GUVTrack], Dict[str, int]]:
    """
    Filter tracks to keep only high-quality vesicles.

    Filters:
    - Require vesicle present in at least min_track_fraction of frames
    - Exclude if center within 1 radius of image border at any point
    - Exclude if jump > max_jump_factor × diameter between consecutive frames

    Parameters
    ----------
    tracks : List[GUVTrack]
        Input tracks
    n_frames : int
        Total number of frames
    image_shape : Tuple[int, int]
        (height, width) of images
    pixel_size_um : float
        Pixel size in micrometers
    config : TrackingConfig
        Configuration

    Returns
    -------
    filtered_tracks : List[GUVTrack]
        Tracks passing all filters
    exclusion_stats : dict
        Counts of exclusions by reason
    """
    height, width = image_shape
    avg_diameter_um = (config.min_diameter_um + config.max_diameter_um) / 2
    avg_diameter_px = avg_diameter_um / pixel_size_um
    max_jump_px = config.max_jump_factor * avg_diameter_px
    min_detections = int(n_frames * config.min_track_fraction)

    exclusion_stats = {
        "excluded_incomplete": 0,
        "excluded_border": 0,
        "excluded_jump": 0,
    }

    filtered = []

    for track in tracks:
        # Check completeness (require at least min_track_fraction of frames)
        if len(track.detections) < min_detections:
            exclusion_stats["excluded_incomplete"] += 1
            continue

        # Check border proximity
        border_violation = False
        for det in track.detections:
            margin = det.radius_px
            if (
                det.x_px < margin
                or det.x_px > width - margin
                or det.y_px < margin
                or det.y_px > height - margin
            ):
                border_violation = True
                break

        if border_violation:
            exclusion_stats["excluded_border"] += 1
            continue

        # Check for large jumps
        jump_violation = False
        for i in range(1, len(track.detections)):
            prev = track.detections[i - 1]
            curr = track.detections[i]
            dx = curr.x_px - prev.x_px
            dy = curr.y_px - prev.y_px
            dist = np.sqrt(dx**2 + dy**2)
            if dist > max_jump_px:
                jump_violation = True
                break

        if jump_violation:
            exclusion_stats["excluded_jump"] += 1
            continue

        filtered.append(track)

    return filtered, exclusion_stats


def measure_membrane_fluorescence(
    tracks: List[GUVTrack],
    fluorescence_stack: np.ndarray,
    pixel_size_um: float,
    time_interval_s: Optional[float],
    config: TrackingConfig,
) -> pd.DataFrame:
    """
    Measure fluorescence in annulus at membrane for each tracked vesicle.

    Parameters
    ----------
    tracks : List[GUVTrack]
        Filtered tracks
    fluorescence_stack : np.ndarray
        Shape (n_frames, height, width) fluorescence images
    pixel_size_um : float
        Pixel size in micrometers
    time_interval_s : Optional[float]
        Time between frames in seconds (None if unknown)
    config : TrackingConfig
        Configuration with annulus width

    Returns
    -------
    measurements : pd.DataFrame
        Columns: vesicle_id, frame, time_seconds, x_um, y_um, radius_um,
                 membrane_fluorescence_mean
    """
    rows = []
    image_shape = fluorescence_stack.shape[1:]

    for track in tracks:
        for det in track.detections:
            # Get the fluorescence frame
            fl_frame = fluorescence_stack[det.frame]

            # Create membrane annulus mask
            inner_radius = det.radius_px - config.annulus_width_px
            outer_radius = det.radius_px + config.annulus_width_px
            inner_radius = max(inner_radius, 1)

            membrane_mask = _create_annulus_mask(
                (det.x_px, det.y_px), inner_radius, outer_radius, image_shape
            )

            # Measure mean fluorescence in membrane region
            membrane_mean = (
                np.mean(fl_frame[membrane_mask]) if np.any(membrane_mask) else 0.0
            )

            # Calculate time
            time_s = det.frame * time_interval_s if time_interval_s else det.frame

            rows.append(
                {
                    "vesicle_id": track.track_id,
                    "frame": det.frame,
                    "time_seconds": time_s,
                    "x_um": det.x_px * pixel_size_um,
                    "y_um": det.y_px * pixel_size_um,
                    "radius_um": det.radius_px * pixel_size_um,
                    "membrane_fluorescence_mean": float(membrane_mean),
                }
            )

    return pd.DataFrame(rows)


def process_lif_file(
    lif_path: Path,
    config: TrackingConfig,
    series_index: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Full processing pipeline for a single .lif file.

    Detection method is controlled by config.detection_method:
    - "brightfield": Hough circle detection on brightfield, validated by fluorescence
    - "fluorescence": Dark region detection in fluorescence (original method)

    Parameters
    ----------
    lif_path : Path
        Path to .lif file
    config : TrackingConfig
        Tracking configuration
    series_index : Optional[int]
        Which series to process (None = auto-select longest)

    Returns
    -------
    measurements : pd.DataFrame
        All membrane fluorescence measurements
    summary : dict
        Processing summary statistics
    """
    # Load data
    fluorescence, brightfield, metadata = load_lif_series(
        lif_path, config, series_index
    )

    pixel_size_um = metadata["pixel_size_um"]
    n_frames = metadata["n_frames"]
    image_shape = metadata["dimensions"]
    time_interval_s = metadata["time_interval_s"]

    # Detect vesicles in each frame
    all_detections: List[List[GUVDetection]] = []
    total_detected = 0

    for frame_idx in range(n_frames):
        fl_frame = fluorescence[frame_idx]
        bf_frame = brightfield[frame_idx]

        # Skip frames that are all zeros (corrupted)
        if fl_frame.max() == 0:
            all_detections.append([])
            continue

        # Choose detection method based on config
        if config.detection_method == "brightfield":
            # Brightfield circle detection with fluorescence validation
            detections = detect_vesicles_brightfield_validated(
                fl_frame, bf_frame, pixel_size_um, config, frame_idx
            )
        else:
            # Fallback: dark region detection in fluorescence
            detections = detect_vesicles_dark_regions(
                fl_frame, bf_frame, pixel_size_um, config, frame_idx
            )

        all_detections.append(detections)
        total_detected += len(detections)

    # Link detections into tracks
    tracks = link_detections(all_detections, pixel_size_um, config)

    # Count valid frames (frames with at least one detection)
    # This ensures corrupt/empty frames don't penalize track completeness
    n_valid_frames = sum(1 for dets in all_detections if len(dets) > 0)

    # Filter tracks using valid frame count for the completeness threshold
    filtered_tracks, exclusion_stats = filter_tracks(
        tracks, n_valid_frames, image_shape, pixel_size_um, config
    )

    # Measure membrane fluorescence
    measurements = measure_membrane_fluorescence(
        filtered_tracks, fluorescence, pixel_size_um, time_interval_s, config
    )

    summary = {
        "n_vesicles_detected": total_detected // n_frames if n_frames > 0 else 0,
        "n_vesicles_tracked": len(filtered_tracks),
        "n_frames": n_frames,
        "pixel_size_um": pixel_size_um,
        "detection_method": config.detection_method,
        **exclusion_stats,
    }

    return measurements, summary


def generate_debug_images_for_lif(
    lif_path: Path,
    output_dir: Path,
    config: TrackingConfig,
    frame_indices: Optional[List[int]] = None,
    series_index: Optional[int] = None,
) -> None:
    """
    Generate debug images for specified frames of a LIF file.

    Parameters
    ----------
    lif_path : Path
        Path to .lif file
    output_dir : Path
        Directory to save debug images
    config : TrackingConfig
        Tracking configuration
    frame_indices : Optional[List[int]]
        Which frames to generate debug images for. Default: [0, mid, last]
    series_index : Optional[int]
        Which series to process (None = auto-select longest)
    """
    # Load data
    fluorescence, brightfield, metadata = load_lif_series(
        lif_path, config, series_index
    )

    pixel_size_um = metadata["pixel_size_um"]
    n_frames = metadata["n_frames"]

    # Default to first, middle, and last frames
    if frame_indices is None:
        frame_indices = [0, n_frames // 2, n_frames - 1]
        frame_indices = list(set(frame_indices))  # Remove duplicates for short videos

    output_dir.mkdir(parents=True, exist_ok=True)
    lif_name = lif_path.stem

    for frame_idx in frame_indices:
        if frame_idx >= n_frames:
            continue

        fl_frame = fluorescence[frame_idx]
        bf_frame = brightfield[frame_idx]

        # Skip corrupted frames
        if fl_frame.max() == 0:
            continue

        # Detect vesicles based on configured method
        candidates: Optional[List[GUVDetection]] = None

        if config.detection_method == "brightfield":
            # Get candidates and validated separately for debug visualization
            candidates = detect_circles_brightfield(
                bf_frame, pixel_size_um, config, frame_idx
            )
            detections = validate_with_fluorescence(candidates, fl_frame, config)
        else:
            # Fluorescence-based detection
            detections = detect_vesicles_dark_regions(
                fl_frame, bf_frame, pixel_size_um, config, frame_idx
            )

        # Generate debug image
        output_path = output_dir / f"debug_{lif_name}_frame{frame_idx:03d}.png"
        generate_debug_image(
            fl_frame,
            detections,
            output_path,
            config,
            title=f"{lif_name} - Frame {frame_idx}",
            brightfield_frame=bf_frame,
            candidate_detections=candidates,
        )

    print(f"  Debug images saved to {output_dir}")
