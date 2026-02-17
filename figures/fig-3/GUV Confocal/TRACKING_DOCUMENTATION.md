# GUV Membrane Fluorescence Tracking - How It Works

This document explains the detection and tracking pipeline for analyzing GUV (Giant Unilamellar Vesicle) membrane fluorescence from confocal microscopy data.

## Overview

The pipeline:
1. Loads `.lif` files containing time-series images with 2 channels
2. Detects vesicles as circles in brightfield using Hough transform
3. Validates candidates by requiring dark interiors in fluorescence
4. Links detections across frames using trackpy
5. Filters tracks based on quality criteria
6. Measures membrane fluorescence in an annulus around each vesicle

---

## Channel Assignment

```
Channel 0 = Fluorescence (used for validation + membrane measurement)
Channel 1 = Brightfield (used for circle detection)
```

The fluorescence channel contains a membrane-bound dye. Vesicle interiors appear **dark** because they exclude the dye, which is used to validate brightfield circle candidates.

---

## Detection Algorithm (Current Default)

### Step 1: Brightfield Circle Detection

```python
edges = canny(brightfield_norm, sigma=canny_sigma)
hough_res = hough_circle(edges, hough_radii)
accums, cx, cy, radii = hough_circle_peaks(...)
```

- Detects circular membrane boundaries directly from brightfield
- Robust when vesicles are touching or clustered
- Current cap: up to 500 circle candidates per frame

### Step 2: Fluorescence Interior Validation

```python
threshold = threshold_otsu(fluorescence_frame)
darkness_threshold = threshold * interior_darkness_threshold
interior_mean = mean(fluorescence[interior_mask])
accept if interior_mean < darkness_threshold
```

- Keeps circles whose interiors are sufficiently dark in fluorescence
- Requires the outside solution ring (just beyond membrane) to be brighter than the interior in a majority of angular sectors
- Optionally enforces membrane brightness contrast (disabled by default because membrane itself may be dim in some frames)
- Rejects brightfield false positives that do not look like in-focus GUVs

### Step 3: Overlap Suppression

- Applies greedy non-maximum suppression to remove near-duplicate overlapping circles
- Runs once on brightfield candidates and again after fluorescence validation
- Uses circle IoU plus near-concentric radius checks to collapse duplicate picks of the same GUV

---

## Detection Algorithm (Legacy Fluorescence Mode)

### Step 1: Thresholding

```python
threshold = threshold_otsu(fluorescence_frame)
dark_mask = fluorescence_frame < threshold * dark_threshold_factor
```

- Uses **Otsu's method** to find automatic threshold
- Multiplies by `dark_threshold_factor` (default: 0.7) to find darker regions
- Anything below `threshold × 0.7` is considered "dark" (potential vesicle interior)

### Step 2: Mask Cleanup

```python
dark_mask = remove_small_objects(dark_mask, max_size=int(min_area * 0.5))
dark_mask = remove_small_holes(dark_mask, max_size=int(min_area * 0.3))
```

- Removes small noise objects (< 50% of minimum vesicle area)
- Fills small holes (< 30% of minimum vesicle area)

### Step 3: Connected Component Analysis

```python
labels = measure.label(dark_mask)
regions = measure.regionprops(labels)
```

Each connected dark region becomes a candidate vesicle.

### Step 4: Filtering by Size and Shape

For each region:

| Filter | Condition | Current Settings |
|--------|-----------|------------------|
| **Min area** | `region.area > π × (min_diameter/2)²` | min_diameter = 5 µm |
| **Max area** | `region.area < π × (max_diameter/2)²` | max_diameter = 50 µm |
| **Shape** | `region.eccentricity < max_eccentricity` | max_eccentricity = 0.7 |

Eccentricity: 0 = perfect circle, 1 = line. Default 0.7 allows slightly elliptical vesicles.

### Step 5: Extract Centroid and Radius

```python
y, x = region.centroid
radius = sqrt(region.area / π)  # Equivalent circular radius
```

---

## Tracking (Linking Across Frames)

Uses **trackpy** library to link detections across frames.

### Parameters

| Parameter | Description | Current Value |
|-----------|-------------|---------------|
| `search_range` | Max distance (px) to link between frames | 0.5 × avg_diameter_px |
| `memory` | Allow gaps of N frames in a track | 5 frames |

### How It Works

1. Builds DataFrame with all detections (x, y, frame, radius)
2. `trackpy.link()` finds nearest-neighbor matches between frames
3. Each linked trajectory gets a unique `particle` ID

---

## Track Filtering

After linking, tracks are filtered based on quality criteria.

### Filter 1: Completeness (min_track_fraction)

```python
min_detections = n_valid_frames × min_track_fraction
if len(track.detections) < min_detections:
    # EXCLUDED
```

- `n_valid_frames` = frames where at least 1 vesicle was detected (excludes corrupt frames)
- Current: `min_track_fraction = 0.25` → vesicle must be detected in ≥25% of valid frames

### Filter 2: Border Proximity

```python
for each detection:
    if x < radius OR x > (width - radius) OR y < radius OR y > (height - radius):
        # EXCLUDED - too close to border
```

Excludes any vesicle that touches the image edge at any point in its track.

### Filter 3: Large Jumps

```python
max_jump_px = max_jump_factor × avg_diameter_px  # = 0.5 × diameter

for consecutive frames:
    distance = sqrt(dx² + dy²)
    if distance > max_jump_px:
        # EXCLUDED - suspicious identity swap
```

Rejects tracks where the vesicle "jumps" more than half its diameter between frames (likely a tracking error where two vesicles got swapped).

---

## Membrane Fluorescence Measurement

For each tracked vesicle at each frame:

```python
inner_radius = vesicle_radius - annulus_width_px
outer_radius = vesicle_radius + annulus_width_px
membrane_mask = annulus(center, inner_radius, outer_radius)
membrane_mean = mean(fluorescence[membrane_mask])
```

Current: `annulus_width_px = 3` → samples fluorescence 3 pixels inside and outside the detected radius.

---

## Current Configuration

```python
TrackingConfig(
    fluorescence_channel=0,
    brightfield_channel=1,
    min_diameter_um=5.0,        # Lowered from 10 to detect smaller vesicles
    max_diameter_um=50.0,
    annulus_width_px=3,
    detection_method="brightfield",
    dark_threshold_factor=0.7,  # Used in legacy fluorescence mode
    max_eccentricity=0.7,       # Used in legacy fluorescence mode
    search_range_factor=0.5,    # Link within 0.5 × diameter
    max_jump_factor=0.5,        # Reject if jumps > 0.5 × diameter
    min_track_fraction=0.25,    # Must be in ≥25% of frames
    link_memory=5,              # Allow 5-frame gaps
    hough_num_peaks=500,        # Up to 500 brightfield circle candidates/frame
    require_membrane_contrast_check=False,  # Default: rely on outside ring sectors
    fluorescence_min_bright_sector_fraction=0.5,
    candidate_overlap_iou_threshold=0.75,
    validated_overlap_iou_threshold=0.75,
)
```

---

## Potential Failure Modes

### 1. Detection Fails: Vesicle Interior Not Dark Enough

**Symptom**: Vesicles visible in image but not detected.

**Cause**: The fluorescence inside the vesicle isn't below `threshold × 0.7`.

**Possible fixes**:
- Increase `dark_threshold_factor` (e.g., 0.8 or 0.9) to be less strict
- Check if the membrane fluorescence is too bright relative to interior

### 2. Detection Fails: Vesicles Too Small

**Symptom**: Small vesicles not detected.

**Cause**: Vesicle area below `min_diameter_um` threshold.

**Fix**: Lower `min_diameter_um` (currently 5.0 µm).

### 3. Detection Fails: Vesicles Too Elliptical

**Symptom**: Elongated vesicles not detected.

**Cause**: Eccentricity > 0.7.

**Fix**: Increase `max_eccentricity` (max 1.0).

### 4. Detection Fails: Connected Regions

**Symptom**: Multiple touching vesicles detected as one giant region (rejected for being too large).

**Cause**: Dark interiors merge into one connected component.

**Fix**: Would need watershed segmentation or edge-based separation.

### 5. Tracking Fails: Vesicles Move Too Much

**Symptom**: Detections exist but tracks are short/fragmented.

**Cause**: `search_range` too small for the actual movement.

**Fix**: Increase `search_range_factor` (e.g., 0.7 or 1.0).

### 6. Tracking Fails: Too Many Vesicles Too Close Together

**Symptom**: trackpy throws "Subnetwork" error or links incorrectly.

**Cause**: Ambiguous linking when vesicles are dense.

**Fix**: The code already falls back to smaller search range, but may need manual intervention.

### 6b. Detection Truncation: Candidate Cap Reached

**Symptom**: Candidate detections repeatedly hit the maximum count per frame.

**Cause**: `hough_num_peaks` too low for dense fields.

**Fix**: Increase `hough_num_peaks` (current default: 500).

### 6c. Duplicate Overlap Picks

**Symptom**: Same GUV appears as multiple similar circles.

**Cause**: Hough transform can return near-concentric peaks for one vesicle.

**Fix**: Tighten overlap suppression thresholds (`candidate_overlap_iou_threshold`, `validated_overlap_iou_threshold`) if needed.

### 7. Tracks Filtered Out: Inconsistent Detection

**Symptom**: Vesicles detected in some frames but not others → track length < 25% of frames.

**Cause**: Detection threshold varies as fluorescence changes over time.

**Possible fixes**:
- Lower `min_track_fraction` further
- Use frame-by-frame adaptive thresholding
- Use a fixed absolute threshold instead of Otsu

### 8. Tracks Filtered Out: Border Exclusion

**Symptom**: Vesicles near edges excluded.

**Cause**: Vesicle center within 1 radius of image border.

**Fix**: No good fix - these vesicles genuinely have incomplete data.

---

## Sample 2 Troubleshooting

If Sample 2 is failing while others work, possible causes:

1. **Different imaging conditions**: Check if Sample 2 has different contrast/brightness levels that affect Otsu thresholding

2. **Different vesicle sizes**: Vesicles may be smaller (fixed by lowering min_diameter) or larger than expected

3. **Different vesicle density**: Dense packing may cause connected regions that exceed max_diameter

4. **Different fluorescence dynamics**: If membrane gets very bright, the "dark interior" may not be dark enough relative to the new threshold

5. **Channel swap**: Verify that channel 0 is actually fluorescence for this sample

### Diagnostic Suggestions

1. **Check raw Otsu threshold per frame**:
   ```python
   for frame in range(n_frames):
       threshold = threshold_otsu(fluorescence[frame])
       print(f"Frame {frame}: Otsu threshold = {threshold}")
   ```

2. **Check detection counts per frame**:
   ```python
   for frame_idx, dets in enumerate(all_detections):
       print(f"Frame {frame_idx}: {len(dets)} detections")
   ```

3. **Visualize the dark mask**:
   ```python
   threshold = threshold_otsu(fl_frame)
   dark_mask = fl_frame < threshold * 0.7
   plt.imshow(dark_mask)
   ```

4. **Check region properties before filtering**:
   ```python
   for region in regions:
       print(f"Area: {region.area}, Eccentricity: {region.eccentricity}")
   ```

---

## File Structure

```
figures/fig-3/GUV Confocal/
├── guv-plotting.py              # Main analysis script
├── Sample 1/
│   ├── After DNA brush addition.lif
│   ├── After cyclodextrin addition.lif
│   ├── results/
│   │   ├── guv_tracking_results.h5
│   │   ├── summary_stats.csv
│   │   └── debug/               # Debug images with detection overlay
│   ├── snapshots/               # Images at plot tick timepoints
│   │   ├── after_DNA_0_minutes_fluorescence.png
│   │   ├── after_DNA_0_minutes_brightfield.png
│   │   ├── after_DNA_16_minutes_fluorescence.png
│   │   └── ...
│   ├── membrane_fluorescence_timecourse.svg
│   └── condition_comparison.svg
├── Sample 2/
│   └── ...                      # Same structure as Sample 1
└── TRACKING_DOCUMENTATION.md    # This file

utils/
└── guv_tracking.py              # Core tracking utilities
```

---

## Output Files

### Plots

1. **membrane_fluorescence_timecourse.svg**: Time series of membrane fluorescence
   - Individual vesicle traces (thin, semi-transparent)
   - Mean trace (thick line)
   - Two segments: After DNA Brush, After Cyclodextrin
   - Y-axis ticks positioned outside the axis

2. **condition_comparison.svg**: Final fluorescence comparison
   - Scatter plot with jitter
   - Mean bars

### Snapshots

Saved at x-axis tick timepoints (0, mid, max for each condition):

- `after_DNA_[t]_minutes_fluorescence.png`: Fluorescence channel (green colormap)
- `after_DNA_[t]_minutes_brightfield.png`: Brightfield channel (greyscale)
- `after_cyclodextrin_[t]_minutes_fluorescence.png`
- `after_cyclodextrin_[t]_minutes_brightfield.png`

---

## Detection Methods

The tracking code supports two detection methods, configurable via `TrackingConfig.detection_method`:

### 1. Brightfield-based detection (default: `"brightfield"`)

Detects circles in brightfield using Hough transform, then validates by checking fluorescence interior is dark.

**Advantages:**
- Better for clustered vesicles (membranes visible even when touching)

**Disadvantages:**
- More sensitive to noise in brightfield images
- Requires parameter tuning (Canny sigma, Hough threshold)

### 2. Fluorescence-based detection (`"fluorescence"`)

Detects vesicles by finding dark regions in the fluorescence channel.

**Advantages:**
- Works well when vesicle interiors clearly exclude the dye
- Robust for well-separated vesicles

**Disadvantages:**
- Can fail when vesicles are clustered (dark interiors merge)

**Configuration parameters:**
```python
canny_sigma: float = 2.0           # Edge detection smoothing
hough_min_distance: int = 20       # Min distance between circles
hough_threshold: float = 0.3       # Accumulator threshold
hough_num_peaks: int = 500         # Max circles per frame
interior_darkness_threshold: float = 0.8  # Validation threshold
```
