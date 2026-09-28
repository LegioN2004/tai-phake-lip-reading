"""
Mouth / Lip ROI Cropping Utility (Person 4 & Shared).
Extracts, pads, and crops bounding boxes around detected mouth landmarks.
"""

from typing import Tuple
import numpy as np


def compute_mouth_bbox(
    landmarks: np.ndarray,
    padding: float = 0.2
) -> Tuple[int, int, int, int]:
    """
    Calculate bounding box around mouth landmarks with proportional padding.

    Args:
        landmarks: (N, 2) mouth landmark coordinates.
        padding: Fractional padding factor to expand bounding box.

    Returns:
        (x_min, y_min, x_max, y_max) bounding box coordinates.
    """
    # Skeleton placeholder
    pass


def crop_mouth_roi(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    target_size: Tuple[int, int] = (96, 96)
) -> np.ndarray:
    """
    Crop mouth ROI from frame and resize to target dimension.

    Args:
        image: Source image frame.
        bbox: (x_min, y_min, x_max, y_max) bounding coordinates.
        target_size: (width, height) desired output shape.

    Returns:
        Cropped and resized mouth ROI image.
    """
    # Skeleton placeholder
    pass


if __name__ == "__main__":
    print("Crop module skeleton initialized.")
