"""Quick diagnostic: check edge fraction distribution for dark discs."""

from pathlib import Path
import numpy as np
from utils.guv_tracking import (
    TrackingConfig,
    detect_dark_discs,
    load_lif_series,
    validate_discs_with_brightfield,
)
from skimage.feature import canny

config = TrackingConfig(
    fluorescence_channel=0,
    brightfield_channel=1,
    min_diameter_um=5.0,
    max_diameter_um=50.0,
    detection_method="consensus",
    fluorescence_blur_sigma=5.0,
    bf_edge_min_fraction=0.0,  # Accept all to see distribution
    bf_edge_annulus_width_px=3,
    bf_edge_canny_sigma=2.0,
    bf_darkness_check=False,  # Disable to see all scores
)

sample_dir = Path("Sample 2")
lif_path = sample_dir / "After DNA brush addition.lif"
fl, bf, meta = load_lif_series(lif_path, config)
pixel_size = meta["pixel_size_um"]

# Check frame 21
frame_idx = 21
fl_frame = fl[frame_idx]
bf_frame = bf[frame_idx]

discs = detect_dark_discs(fl_frame, pixel_size, config, frame_idx)
print(f"Dark discs found: {len(discs)}")

# Get all edge scores (threshold=0 so all pass)
validated, scores = validate_discs_with_brightfield(discs, bf_frame, config)
print(f"Validated (no threshold): {len(validated)}")

scores_arr = np.array(scores)
print(f"\nEdge fraction stats:")
print(f"  min:    {scores_arr.min():.4f}")
print(f"  max:    {scores_arr.max():.4f}")
print(f"  mean:   {scores_arr.mean():.4f}")
print(f"  median: {np.median(scores_arr):.4f}")
print(f"  p25:    {np.percentile(scores_arr, 25):.4f}")
print(f"  p75:    {np.percentile(scores_arr, 75):.4f}")

# Show histogram
for threshold in [0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]:
    n_pass = int((scores_arr >= threshold).sum())
    print(f"  >= {threshold:.2f}: {n_pass}/{len(scores_arr)}")

# Also check with darkness filter
config2 = TrackingConfig(
    fluorescence_channel=0,
    brightfield_channel=1,
    min_diameter_um=5.0,
    max_diameter_um=50.0,
    detection_method="consensus",
    fluorescence_blur_sigma=5.0,
    bf_edge_min_fraction=0.0,
    bf_edge_annulus_width_px=3,
    bf_edge_canny_sigma=2.0,
    bf_darkness_check=True,
)
validated2, scores2 = validate_discs_with_brightfield(discs, bf_frame, config2)
print(f"\nWith darkness check: {len(validated2)}/{len(discs)} pass")
if scores2:
    s2 = np.array(scores2)
    print(f"  Edge fraction range: {s2.min():.4f} - {s2.max():.4f}")
    for threshold in [0.01, 0.02, 0.05, 0.08, 0.10, 0.15]:
        n_pass = int((s2 >= threshold).sum())
        print(f"  >= {threshold:.2f}: {n_pass}/{len(scores2)}")
