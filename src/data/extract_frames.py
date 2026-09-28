"""
Frame Extraction Script (Person 1).
Extracts high-fidelity frames from trimmed videos in read-only mode to data/frames/<speaker>/<digit>/:
- Ensures sequential zero-padded naming (e.g. frame_0001.png)
- Checks frame counts against video stream
- Never modifies raw or trimmed source videos
"""

from pathlib import Path
from typing import Optional


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    image_format: str = "png"
) -> int:
    """
    Extract individual frames from a trimmed video file to disk.

    Args:
        video_path: Path to trimmed input video (read-only).
        output_dir: Destination directory for frame images.
        image_format: Image file extension ('png' or 'jpg').

    Returns:
        Number of frames successfully extracted.
    """
    # Skeleton placeholder - implementation will be filled during Person 1 workflow
    pass


def extract_all_dataset_frames(
    trimmed_dir: Path,
    frames_dir: Path,
    metadata_csv_path: Optional[Path] = None
) -> None:
    """
    Batch frame extraction across all validated trimmed dataset videos.

    Args:
        trimmed_dir: Root directory of trimmed videos (data/trimmed).
        frames_dir: Root directory for extracted frames (data/frames).
        metadata_csv_path: Optional path to metadata.csv to guide extraction.
    """
    # Skeleton placeholder
    pass


if __name__ == "__main__":
    print("Frame extraction module skeleton initialized.")
