"""Generate debug images for all sample folders using consensus detection.

Produces 4-panel debug images (FL mask, dark discs, BF edges, validated)
for the first, middle, and last frames of each .lif file in each sample.
"""

from pathlib import Path

from utils.guv_tracking import (
    TrackingConfig,
    generate_debug_images_for_lif,
)

CONFIG = TrackingConfig(
    fluorescence_channel=0,
    brightfield_channel=1,
    min_diameter_um=5.0,
    max_diameter_um=50.0,
    fluorescence_blur_sigma=5.0,
    detection_method="consensus",
    bf_edge_min_fraction=0.05,
    bf_edge_annulus_width_px=3,
    bf_edge_canny_sigma=2.0,
    bf_darkness_check=False,
)

FILES_TO_PROCESS = [
    "After DNA brush addition.lif",
    "After cyclodextrin addition.lif",
]

base_dir = Path(__file__).parent

sample_folders = sorted(
    d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("Sample")
)

for sample_dir in sample_folders:
    print(f"\n{'=' * 60}")
    print(f"Processing {sample_dir.name}")
    print("=" * 60)

    debug_dir = sample_dir / "results" / "debug"

    for lif_name in FILES_TO_PROCESS:
        # Case-insensitive file lookup
        lif_path = None
        for f in sample_dir.iterdir():
            if f.name.lower() == lif_name.lower():
                lif_path = f
                break

        if lif_path is None:
            print(f"  {lif_name} not found, skipping")
            continue

        print(f"  Generating debug images for {lif_path.name}...")
        generate_debug_images_for_lif(lif_path, debug_dir, CONFIG)

print("\nDone!")
