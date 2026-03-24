"""
Plot GUV microscopy frames grouped by condition (DNA brush / cyclodextrin).
Each condition gets one SVG with 3 rows (one per run) and columns per timepoint.
DNA plot: max 150 mm wide. Cyclodextrin plot: same image scale.
"""

import os
import re

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.image import imread

matplotlib.rcParams.update(
    {
        "font.size": 8,
        "font.family": "sans-serif",
        "svg.fonttype": "none",
    }
)

# ---------------------------------------------------------------------------
# Layout constants (all in inches)
# ---------------------------------------------------------------------------
MM = 1 / 25.4
GAP = 2 * MM          # whitespace between adjacent images
ROW_LABEL_W = 12 * MM  # width reserved for "Run N" rotated label
TOP_LABEL_H = 5 * MM   # height above each image row for "t = N" labels

DNA_MAX_W = 150 * MM   # maximum total figure width for the DNA plot
DNA_MAX_COLS = 5       # widest row in DNA plot (Run 3, 5 frames)

# Image display size: derived so DNA plot fits exactly in DNA_MAX_W
IMG_W = (DNA_MAX_W - ROW_LABEL_W - (DNA_MAX_COLS - 1) * GAP) / DNA_MAX_COLS
IMG_H = IMG_W  # all source images are square

# ---------------------------------------------------------------------------
# Parse files
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
data: dict[str, dict[int, list[tuple[int, str]]]] = {
    "dna": {},
    "cyclodextrin": {},
}

for run_dir in sorted(os.listdir(BASE)):
    run_path = os.path.join(BASE, run_dir)
    if not os.path.isdir(run_path):
        continue
    m = re.match(r"run-(\d+)$", run_dir)
    if not m:
        continue
    run_num = int(m.group(1))

    for fname in sorted(os.listdir(run_path)):
        if not fname.lower().endswith(".png"):
            continue
        fl = fname.lower()
        if "dna" in fl:
            condition = "dna"
        elif "cyclodextrin" in fl:
            condition = "cyclodextrin"
        else:
            continue

        fm = re.search(r"frame(\d+)", fname, re.IGNORECASE)
        if not fm:
            continue
        frame = int(fm.group(1))
        fpath = os.path.join(run_path, fname)
        data[condition].setdefault(run_num, []).append((frame, fpath))

# Sort each run by frame number
for cond in data:
    for run in data[cond]:
        data[cond][run].sort()

# ---------------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------------


def make_plot(condition: str, out_path: str) -> None:
    runs = sorted(data[condition].keys())
    n_rows = len(runs)
    max_cols = max(len(data[condition][r]) for r in runs)

    fig_w = ROW_LABEL_W + max_cols * IMG_W + (max_cols - 1) * GAP
    fig_h = n_rows * (TOP_LABEL_H + IMG_H) + (n_rows - 1) * GAP

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    for row_i, run_num in enumerate(runs):
        frames = data[condition][run_num]

        # Vertical position of this row's image block (inches from top of figure)
        row_top = row_i * (TOP_LABEL_H + IMG_H + GAP)
        img_top = row_top + TOP_LABEL_H  # below the label strip
        img_bottom = fig_h - img_top - IMG_H  # matplotlib y=0 at bottom

        # "Run N" label – centred vertically on the image block
        row_center_y = (fig_h - img_top - IMG_H / 2) / fig_h
        fig.text(
            (ROW_LABEL_W / 2) / fig_w,
            row_center_y,
            f"Run {run_num}",
            ha="center",
            va="center",
            fontsize=8,
            rotation=90,
        )

        for col_i, (frame, fpath) in enumerate(frames):
            img_left = ROW_LABEL_W + col_i * (IMG_W + GAP)

            # Axes rectangle in figure fractions [left, bottom, width, height]
            ax = fig.add_axes(
                [
                    img_left / fig_w,
                    img_bottom / fig_h,
                    IMG_W / fig_w,
                    IMG_H / fig_h,
                ]
            )
            ax.imshow(imread(fpath), aspect="equal")
            ax.axis("off")

            # "t = N" label centred above the image, inside the label strip
            label_x = (img_left + IMG_W / 2) / fig_w
            label_y = (fig_h - row_top - TOP_LABEL_H / 2) / fig_h
            fig.text(
                label_x,
                label_y,
                f"t = {frame}",
                ha="center",
                va="center",
                fontsize=8,
            )

    fig.savefig(out_path, format="svg", dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Generate both plots
# ---------------------------------------------------------------------------
make_plot("dna", os.path.join(BASE, "dna-frames.svg"))
make_plot("cyclodextrin", os.path.join(BASE, "cyclodextrin-frames.svg"))
