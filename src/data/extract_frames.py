"""
Frame Extraction Script (Person 1).
Extracts high-fidelity frames from trimmed videos directly into each digit subfolder
(e.g., data/digit/<speaker>/<digit>/frame_0001.png):
- Ensures sequential zero-padded naming (e.g. frame_0001.png)
- Checks frame counts against video stream
- Never modifies or overwrites the source video files
"""

from pathlib import Path
from typing import Optional


def extract_frames_from_video(
    video_path: Path,
    output_dir: Optional[Path] = None,
    image_format: str = "png"
) -> int:
    """
    Extract individual frames from a trimmed video file directly into its folder.

    Args:
        video_path: Path to trimmed input video (read-only).
        output_dir: Optional output directory (defaults to video's parent folder, e.g. data/digit/s1/d0/).
        image_format: Image file extension ('png' or 'jpg').

    Returns:
        Number of frames successfully extracted.
    """
    # Skeleton placeholder - implementation will be filled during Person 1 workflow
    pass


def extract_all_dataset_frames(
    dataset_dir: Path,
    metadata_csv_path: Optional[Path] = None
) -> None:
    """
    Batch frame extraction across all trimmed dataset videos directly into their digit folders.

    Args:
        dataset_dir: Root directory of dataset (e.g. data/digit).
        metadata_csv_path: Optional path to metadata.csv to guide extraction.
    """
    # Skeleton placeholder
    pass


if __name__ == "__main__":
    print("Frame extraction module skeleton initialized.")
