#!/usr/bin/env python3
"""
Tai Phake Visual Speech Recognition (VSR)
Step 1: Complete Data Audit and Manifest Generator

This script performs a rigorous data audit over `data/digit/`:
1. Discovers all speaker directories (s1 through s30) and detects any missing speakers.
2. Discovers all digit directories (d0 through d10) per speaker and detects any missing digits.
3. For each digit folder:
   - Checks whether the video file exists and tests decodability via OpenCV.
   - Audits sequential extracted frames (`frame_*.jpg`).
   - Reads image dimensions via cv2.
   - Detects zero-byte files, unreadable frames, and frame numbering gaps.
4. Generates a pandas DataFrame manifest with columns:
   [video_path, speaker, digit, num_frames, crop_size, split]
5. Saves the manifest to `data/manifest.csv`.
6. Prints a detailed audit report with summary statistics and anomalies.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import pandas as pd


def audit_and_create_manifest(
    data_dir: Path,
    output_manifest_path: Path,
    expected_speaker_count: int = 30,
    expected_digits: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Performs full data audit and generates manifest CSV.
    """
    if expected_digits is None:
        expected_digits = [f"d{i}" for i in range(11)]

    expected_speakers = [f"s{i}" for i in range(1, expected_speaker_count + 1)]

    # 1. Check speakers
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset root directory does not exist: {data_dir}")

    found_speaker_dirs = {
        d.name: d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("s")
    }
    missing_speakers = [s for s in expected_speakers if s not in found_speaker_dirs]

    # Audit tracking metrics
    total_digit_folders_found = 0
    total_videos_found = 0
    total_frames_counted = 0
    missing_speaker_digits: List[str] = []
    folders_zero_frames: List[str] = []
    unopened_videos: List[str] = []
    zero_size_frames: List[str] = []
    frame_gaps: List[str] = []
    num_frames_per_utterance: List[int] = []

    records: List[Dict[str, Any]] = []

    for sp_idx in range(1, expected_speaker_count + 1):
        sp_name = f"s{sp_idx}"
        sp_path = data_dir / sp_name

        if not sp_path.exists() or not sp_path.is_dir():
            for d_name in expected_digits:
                missing_speaker_digits.append(f"{sp_name}/{d_name}")
                records.append({
                    "video_path": str(data_dir / sp_name / d_name / f"{d_name}.mp4"),
                    "speaker": sp_name,
                    "digit": d_name,
                    "num_frames": 0,
                    "crop_size": None,
                    "split": "unsplit",
                })
            continue

        for d_name in expected_digits:
            d_path = sp_path / d_name
            video_file = d_path / f"{d_name}.mp4"
            rel_video_path = f"data/digit/{sp_name}/{d_name}/{d_name}.mp4"

            if not d_path.exists() or not d_path.is_dir():
                missing_speaker_digits.append(f"{sp_name}/{d_name}")
                records.append({
                    "video_path": rel_video_path,
                    "speaker": sp_name,
                    "digit": d_name,
                    "num_frames": 0,
                    "crop_size": None,
                    "split": "unsplit",
                })
                continue

            total_digit_folders_found += 1

            # Check video existence and readability
            if video_file.exists():
                total_videos_found += 1
                try:
                    cap = cv2.VideoCapture(str(video_file))
                    if not cap.isOpened():
                        unopened_videos.append(str(video_file))
                    else:
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            unopened_videos.append(str(video_file))
                    cap.release()
                except Exception:
                    unopened_videos.append(str(video_file))

            # Audit frame files
            frame_files = sorted(
                d_path.glob("frame_*.jpg"),
                key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)) if re.search(r"(\d+)", p.stem) else 0,
            )
            n_frames = len(frame_files)
            total_frames_counted += n_frames
            num_frames_per_utterance.append(n_frames)

            crop_size_str: Optional[str] = None

            if n_frames == 0:
                folders_zero_frames.append(f"{sp_name}/{d_name}")
            else:
                # Inspect frame dimensions from the first frame
                first_frame_path = frame_files[0]
                try:
                    img = cv2.imread(str(first_frame_path))
                    if img is not None:
                        h, w = img.shape[:2]
                        crop_size_str = f"{w}x{h}"
                    else:
                        crop_size_str = None
                except Exception:
                    crop_size_str = None

                # Check zero-size and sequential numbering
                frame_indices: List[int] = []
                for ff in frame_files:
                    try:
                        if ff.stat().st_size == 0:
                            zero_size_frames.append(str(ff))
                    except OSError:
                        zero_size_frames.append(str(ff))

                    match = re.search(r"frame_(\d+)\.jpg$", ff.name)
                    if match:
                        frame_indices.append(int(match.group(1)))

                # Check for gaps: indices should be consecutive starting from 1 to n_frames
                if frame_indices and frame_indices != list(range(1, n_frames + 1)):
                    expected_set = set(range(1, max(frame_indices) + 1))
                    actual_set = set(frame_indices)
                    missing_indices = sorted(expected_set - actual_set)
                    frame_gaps.append(f"{sp_name}/{d_name}: missing frame numbers {missing_indices}")

            records.append({
                "video_path": rel_video_path,
                "speaker": sp_name,
                "digit": d_name,
                "num_frames": n_frames,
                "crop_size": crop_size_str,
                "split": "unsplit",
            })

    # Create DataFrame
    df_manifest = pd.DataFrame(records)

    # Save manifest
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df_manifest.to_csv(output_manifest_path, index=False)

    # Summary statistics
    audit_stats = {
        "total_speakers_found": len(found_speaker_dirs),
        "expected_speakers_count": expected_speaker_count,
        "missing_speakers": missing_speakers,
        "total_digit_folders_found": total_digit_folders_found,
        "expected_digit_folders": expected_speaker_count * len(expected_digits),
        "total_videos_found": total_videos_found,
        "expected_videos": expected_speaker_count * len(expected_digits),
        "total_frames_counted": total_frames_counted,
        "missing_speaker_digits": missing_speaker_digits,
        "folders_zero_frames": folders_zero_frames,
        "unopened_videos": unopened_videos,
        "zero_size_frames": zero_size_frames,
        "frame_gaps": frame_gaps,
        "avg_frames": (sum(num_frames_per_utterance) / len(num_frames_per_utterance)) if num_frames_per_utterance else 0.0,
        "min_frames": min(num_frames_per_utterance) if num_frames_per_utterance else 0,
        "max_frames": max(num_frames_per_utterance) if num_frames_per_utterance else 0,
    }

    return df_manifest, audit_stats


def print_summary_report(stats: Dict[str, Any], output_manifest_path: Path) -> None:
    print("=" * 75)
    print("                   DATA AUDIT & MANIFEST SUMMARY REPORT")
    print("=" * 75)
    print(f"Speakers Found                   : {stats['total_speakers_found']} / {stats['expected_speakers_count']}")
    if stats["missing_speakers"]:
        print(f"  Missing Speakers               : {stats['missing_speakers']}")
    else:
        print("  Missing Speakers               : None (All 30 present)")

    print(f"Digit Folders Found              : {stats['total_digit_folders_found']} / {stats['expected_digit_folders']}")
    print(f"Total Videos (.mp4) Found        : {stats['total_videos_found']} / {stats['expected_videos']}")
    print(f"Total Frames Counted             : {stats['total_frames_counted']:,}")
    print(f"Frames per Utterance (Min/Avg/Max): {stats['min_frames']} / {stats['avg_frames']:.2f} / {stats['max_frames']}")

    print("\n" + "-" * 75)
    print(" INTEGRITY & HEALTH CHECKS:")
    print("-" * 75)

    print(f"• Missing speaker/digit combinations : {len(stats['missing_speaker_digits'])}")
    if stats["missing_speaker_digits"]:
        for item in stats["missing_speaker_digits"]:
            print(f"    - {item}")

    print(f"• Digit folders with zero frames     : {len(stats['folders_zero_frames'])}")
    if stats["folders_zero_frames"]:
        for item in stats["folders_zero_frames"]:
            print(f"    - {item}")

    print(f"• Videos that could not be opened    : {len(stats['unopened_videos'])}")
    if stats["unopened_videos"]:
        for item in stats["unopened_videos"]:
            print(f"    - {item}")

    print(f"• Extracted frames with zero bytes   : {len(stats['zero_size_frames'])}")
    if stats["zero_size_frames"]:
        for item in stats["zero_size_frames"]:
            print(f"    - {item}")

    print(f"• Gaps in frame numbering            : {len(stats['frame_gaps'])}")
    if stats["frame_gaps"]:
        for item in stats["frame_gaps"]:
            print(f"    - {item}")

    print("\n" + "-" * 75)
    print(f" Manifest successfully saved to: {output_manifest_path}")
    print("=" * 75)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Tai Phake digits dataset and build manifest CSV.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/digit",
        help="Path to data/digit directory (default: data/digit).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/manifest.csv",
        help="Path to output manifest CSV (default: data/manifest.csv).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    data_dir = project_root / args.data_dir if not Path(args.data_dir).is_absolute() else Path(args.data_dir)
    output_manifest_path = project_root / args.output_csv if not Path(args.output_csv).is_absolute() else Path(args.output_csv)

    df_manifest, stats = audit_and_create_manifest(data_dir, output_manifest_path)
    print_summary_report(stats, output_manifest_path)

    print("\nManifest First 10 Rows (`data/manifest.csv`):")
    print(df_manifest.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
