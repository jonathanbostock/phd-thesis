"""Extract individual frames from the GUV confocal microscopy TIFF movie."""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def extract_composite_frame(filename, frame_index):
    """
    Extract a composite frame from a multi-channel TIFF file.

    The overlay file has two channels. We'll create a composite where
    the scale bar and annotations are visible.

    Args:
        filename: Path to the TIFF file
        frame_index: Which frame to extract (0-indexed)

    Returns:
        PIL Image object
    """
    img = Image.open(filename)

    # Extract both channels for this timepoint
    frame_position_ch0 = frame_index * 2
    frame_position_ch1 = frame_index * 2 + 1

    try:
        # Get channel 0 (fluorescence)
        img.seek(frame_position_ch0)
        ch0_array = np.array(img)

        # Get channel 1
        img.seek(frame_position_ch1)
        ch1_array = np.array(img)

        # Create RGB composite - use the green palette
        # Take the maximum of both channels to include any overlays
        composite = np.maximum(ch0_array, ch1_array)

        # Create RGB image with green color
        rgb_array = np.zeros(
            (composite.shape[0], composite.shape[1], 3), dtype=np.uint8
        )
        rgb_array[:, :, 1] = composite  # Green channel

        # Make white pixels (scale bar) appear white
        white_mask = composite == 255
        rgb_array[white_mask, 0] = 255  # Red
        rgb_array[white_mask, 1] = 255  # Green
        rgb_array[white_mask, 2] = 255  # Blue

        return Image.fromarray(rgb_array)
    except EOFError:
        print(f"Frame {frame_index} not found in file")
        return None


def save_frame_as_png(frame, output_filename, time_minutes, dpi=300):
    """Save a frame as a PNG image."""
    # Save directly using PIL to preserve the exact appearance
    frame.save(output_filename, dpi=(dpi, dpi))
    print(f"Saved {output_filename} (t={time_minutes} min)")


def main():
    """Extract frames at 0, 15, and 30 minutes."""
    filename = "experiment-1-overlay.tif"

    # First, let's check how many frames are in the file
    img = Image.open(filename)
    n_frames = 0
    try:
        while True:
            img.seek(n_frames)
            n_frames += 1
    except EOFError:
        pass

    print(f"Total frames in TIFF: {n_frames}")
    n_timepoints = n_frames // 2  # Divide by 2 because we have 2 channels
    print(f"Number of timepoints: {n_timepoints}")

    # Assuming the experiment duration and calculate timepoints
    # If we have 36 timepoints (0-35), we need to figure out the time interval
    # Let's assume it's sampled at 1 min intervals (common for GUV experiments)
    # You may need to adjust this based on your actual experimental setup

    time_interval_minutes = 1  # Adjust this if needed

    # Calculate which frames correspond to 0, 15, and 30 minutes
    times_to_extract = [0, 15, 30]  # in minutes

    for time_min in times_to_extract:
        frame_index = int(time_min / time_interval_minutes)

        if frame_index < n_timepoints:
            # Extract composite frame with overlays
            frame = extract_composite_frame(filename, frame_index)

            if frame is not None:
                output_filename = f"frame_t{time_min}min.png"
                save_frame_as_png(frame, output_filename, time_min)
        else:
            print(
                f"Warning: Time {time_min} min (frame {frame_index}) exceeds available timepoints"
            )


if __name__ == "__main__":
    main()
