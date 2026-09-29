import argparse
import csv
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from tqdm import tqdm


# Vocabulary mapping for Tai Phake digits (d0 to d10)
DIGIT_MAPPING: Dict[str, Dict[str, str]] = {
    "d0": {"label": "Zero", "word": "Pau"},
    "d1": {"label": "One", "word": "Nung"},
    "d2": {"label": "Two", "word": "Saung"},
    "d3": {"label": "Three", "word": "Sam"},
    "d4": {"label": "Four", "word": "Si"},
    "d5": {"label": "Five", "word": "Ha"},
    "d6": {"label": "Six", "word": "Hok"},
    "d7": {"label": "Seven", "word": "Chit"},
    "d8": {"label": "Eight", "word": "Pet"},
    "d9": {"label": "Nine", "word": "Kao"},
    "d10": {"label": "Ten", "word": "Sip"},
}


@dataclass
class VideoRecord:
    video_id: str
    speaker_id: str
    digit: str
    digit_label: str
    tai_phake_word: str
    file_name: str
    relative_path: str
    duration_sec: float
    fps: float
    frame_count: int
    width: int
    height: int
    is_valid: bool
    notes: str


def validate_video_file(file_path: Path, base_dir: Path) -> VideoRecord:
    """
    Validate a single trimmed video file in read-only mode.

    Args:
        file_path: Path to the video file.
        base_dir: Base directory to compute relative path from.

    Returns:
        VideoRecord dataclass with technical metrics and validation status.
    """
    rel_path = file_path.relative_to(base_dir)
    parts = rel_path.parts
    speaker_id = parts[0] if len(parts) >= 2 else "unknown"
    digit = parts[1] if len(parts) >= 2 else "unknown"
    video_id = f"{speaker_id}_{digit}"

    digit_meta = DIGIT_MAPPING.get(digit, {"label": "Unknown", "word": "Unknown"})

    if not file_path.exists() or file_path.stat().st_size == 0:
        return VideoRecord(
            video_id=video_id,
            speaker_id=speaker_id,
            digit=digit,
            digit_label=digit_meta["label"],
            tai_phake_word=digit_meta["word"],
            file_name=file_path.name,
            relative_path=str(rel_path).replace("\\", "/"),
            duration_sec=0.0,
            fps=0.0,
            frame_count=0,
            width=0,
            height=0,
            is_valid=False,
            notes="File does not exist or has 0 bytes.",
        )

    cap = cv2.VideoCapture(str(file_path))
    if not cap.isOpened():
        return VideoRecord(
            video_id=video_id,
            speaker_id=speaker_id,
            digit=digit,
            digit_label=digit_meta["label"],
            tai_phake_word=digit_meta["word"],
            file_name=file_path.name,
            relative_path=str(rel_path).replace("\\", "/"),
            duration_sec=0.0,
            fps=0.0,
            frame_count=0,
            width=0,
            height=0,
            is_valid=False,
            notes="OpenCV could not open video stream.",
        )

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = round(frame_count / fps, 2) if fps > 0 else 0.0

    # Verify decodability of first frame
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return VideoRecord(
            video_id=video_id,
            speaker_id=speaker_id,
            digit=digit,
            digit_label=digit_meta["label"],
            tai_phake_word=digit_meta["word"],
            file_name=file_path.name,
            relative_path=str(rel_path).replace("\\", "/"),
            duration_sec=duration_sec,
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
            is_valid=False,
            notes="First frame could not be decoded.",
        )

    return VideoRecord(
        video_id=video_id,
        speaker_id=speaker_id,
        digit=digit,
        digit_label=digit_meta["label"],
        tai_phake_word=digit_meta["word"],
        file_name=file_path.name,
        relative_path=str(rel_path).replace("\\", "/"),
        duration_sec=duration_sec,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        is_valid=True,
        notes="OK",
    )


def validate_all_videos(
    digit_dir: Path,
    output_metadata_csv: Optional[Path] = None
) -> List[VideoRecord]:
    """
    Scans data/digit/ directory, validates all videos, and optionally writes metadata.csv.
    """
    video_files = sorted(digit_dir.rglob("*.mp4"))
    records: List[VideoRecord] = []

    print(f"Discovered {len(video_files)} video(s) for validation.")
    for vf in tqdm(video_files, desc="Validating videos", unit="video"):
        rec = validate_video_file(vf, digit_dir)
        records.append(rec)

    if output_metadata_csv:
        output_metadata_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "video_id", "speaker_id", "digit", "digit_label", "tai_phake_word",
            "file_name", "relative_path", "duration_sec", "fps", "frame_count",
            "width", "height", "is_valid", "notes"
        ]
        with open(output_metadata_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow(asdict(r))
        print(f"Updated metadata written to: {output_metadata_csv}")

    valid_count = sum(1 for r in records if r.is_valid)
    print("\n" + "=" * 60)
    print(" VIDEO VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total Videos Checked : {len(records)}")
    print(f"Valid Videos         : {valid_count}")
    print(f"Corrupted/Invalid    : {len(records) - valid_count}")
    print("=" * 60)

    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Tai Phake digit videos.")
    parser.add_argument("--input-dir", type=str, default="data/digit", help="Path to data/digit directory.")
    parser.add_argument("--output-csv", type=str, default="data/metadata.csv", help="Path to output metadata.csv.")
    args = parser.parse_args()

    validate_all_videos(Path(args.input_dir), Path(args.output_csv) if args.output_csv else None)
