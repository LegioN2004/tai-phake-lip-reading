#!/usr/bin/env python3
"""
Tai Phake VSR Dataset Sanitizer & Integrity Checker (Strictly Read-Only)
========================================================================

Validates the structure, naming, completeness, decodability, and integrity
of any Tai Phake dataset folder (videos, raw frames, or cropped/augmented frames).

Checks performed:
1. Speaker folder discovery (handles base speakers, capitalized names like S35, and augmented variants like s1_aug1).
2. Digit folder completeness: verifies all 11 classes (d0 through d10) exist for every speaker.
3. File naming verification: ensures videos are named exactly 'd{i}.mp4' and flags multi-file/mislabeled folders (e.g. d2_002.mp4).
4. Frame sequence continuity & counting (for image datasets).
5. File integrity: 0-byte detection, OpenCV video decodability, PIL image decodability.
6. Duplicate detection: identifies duplicate files via SHA-256 hashes.
7. Generates terminal reports, CSV error inventories, and JSON summaries.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

# ==============================================================================
# CONFIGURATION: TARGET DATASET PATH
# Change this constant to point to ANY dataset folder you want to inspect:
# Examples:
#   "data/experiments/digit_videos_only"
#   "data/experiments/dlib_augmented_cropped_100_80_20_96x96"
#   "data/experiments/extracted_raw_frames"
#   "data/processed"
# ==============================================================================
TARGET_DATASET_DIR: str = "data/experiments/digit_videos_only"

# Target vocabulary
EXPECTED_DIGITS: List[str] = [f"d{i}" for i in range(11)]

# Extensions
VIDEO_EXTS: Set[str] = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IGNORE_NAMES: Set[str] = {".ds_store", "thumbs.db", ".git", ".gitignore", "__pycache__"}

# Optional deep media decoders (graceful degradation if missing)
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    cv2 = None
    HAS_OPENCV = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    Image = None
    HAS_PIL = False


@dataclass
class Issue:
    severity: str        # 'CRITICAL', 'WARNING', 'INFO'
    issue_type: str      # E.g. 'WRONG_NAME', 'MISSING_DIGIT', 'MULTI_FILE', 'EMPTY_FOLDER', 'CORRUPT_FILE', 'DUPLICATE'
    speaker: str
    digit: str
    file_name: str
    path: str
    details: str
    recommendation: str


class DatasetSanitizer:
    def __init__(self, target_dir: str | Path, output_dir: Optional[str | Path] = None):
        self.project_root = self._find_project_root()
        self.target_path = self._resolve_path(target_dir)
        
        # Output directory for logs
        if output_dir:
            self.output_dir = Path(output_dir).resolve()
        else:
            folder_slug = self.target_path.name
            self.output_dir = self.project_root / "results" / "dataset_sanitization_reports" / folder_slug

        self.issues: List[Issue] = []
        self.dataset_type = "UNKNOWN"  # 'VIDEOS' or 'FRAMES'
        
        # File hashes: hash -> list of paths
        self.file_hashes: Dict[str, List[Tuple[str, str, str]]] = {} # hash -> [(speaker, digit, path)]

        # Speaker & digit stats
        self.speaker_dirs: List[Path] = []
        self.stats: Dict[str, Any] = {}

    def _find_project_root(self) -> Path:
        curr = Path.cwd()
        for p in [curr] + list(curr.parents):
            if (p / "data").exists() or (p / "notebooks").exists():
                return p
        return curr

    def _resolve_path(self, p: str | Path) -> Path:
        path_obj = Path(p)
        if not path_obj.is_absolute():
            path_obj = (self.project_root / path_obj).resolve()
        return path_obj

    def detect_dataset_type(self) -> str:
        """Determines whether the directory contains videos or image sequences."""
        if not self.target_path.exists():
            return "NON_EXISTENT"
        
        video_count = 0
        image_count = 0
        
        # Sample inspection up to depth 3
        for root, _, files in os.walk(self.target_path):
            for f in files:
                ext = Path(f).suffix.lower()
                if ext in VIDEO_EXTS:
                    video_count += 1
                elif ext in IMAGE_EXTS:
                    image_count += 1
                if video_count >= 5:
                    return "VIDEOS"
                if image_count >= 15:
                    return "FRAMES"
        
        if video_count > 0:
            return "VIDEOS"
        if image_count > 0:
            return "FRAMES"
        return "EMPTY_OR_UNKNOWN"

    def run(self) -> bool:
        """Runs the entire sanity check."""
        print("=" * 80)
        print("TAI PHAKE VSR DATASET SANITIZER (READ-ONLY)")
        print("=" * 80)
        print(f"Target Directory : {self.target_path}")
        print(f"Project Root     : {self.project_root}")
        print(f"Reports Output   : {self.output_dir}")
        print(f"OpenCV Available : {'YES (Deep Video Decode Enabled)' if HAS_OPENCV else 'NO (Basic File Checks Only)'}")
        print(f"PIL Available    : {'YES (Deep Image Decode Enabled)' if HAS_PIL else 'NO (Basic File Checks Only)'}")
        print("=" * 80)

        if not self.target_path.exists():
            print(f"\n[CRITICAL ERROR] Target path does not exist:\n  {self.target_path}\n")
            return False

        self.dataset_type = self.detect_dataset_type()
        print(f"Detected Dataset Mode: {self.dataset_type}")

        # 1. Discover speakers
        self.discover_speakers()
        print(f"Found {len(self.speaker_dirs)} Speaker/Folder entries.")

        # 2. Check each speaker
        for sp_path in self.speaker_dirs:
            sp_name = sp_path.name
            self.validate_speaker_completeness(sp_path, sp_name)

        # 3. Check duplicate hashes
        self.check_duplicate_hashes()

        # 4. Print summary
        is_clean = self.print_report()

        # 5. Save reports
        self.save_reports()
        return is_clean

    def discover_speakers(self) -> None:
        """Finds all speaker subdirectories."""
        entries = sorted([d for d in self.target_path.iterdir() if d.is_dir() and not d.name.startswith(".")])
        
        # Exclude known non-speaker metadata/results dirs if any
        ignored_names = {"backup", "backup-20-oct", "backup_pre_nung_replaced", "snapshots", "old_stuff"}
        self.speaker_dirs = [d for d in entries if d.name.lower() not in ignored_names]

    def validate_speaker_completeness(self, sp_path: Path, sp_name: str) -> None:
        """Checks that all 11 digits exist for this speaker and inspects their files."""
        existing_subdirs = {d.name: d for d in sp_path.iterdir() if d.is_dir() and not d.name.startswith(".")}
        
        # Check for unexpected non-digit subdirectories
        for name, sub_p in existing_subdirs.items():
            if name not in EXPECTED_DIGITS:
                self.issues.append(Issue(
                    severity="WARNING",
                    issue_type="UNEXPECTED_FOLDER",
                    speaker=sp_name,
                    digit=name,
                    file_name=name,
                    path=str(sub_p),
                    details=f"Unexpected directory '{name}' inside speaker folder.",
                    recommendation="Remove or move this folder outside the speaker directory."
                ))

        # Check for extraneous loose files in speaker folder
        loose_files = [f for f in sp_path.iterdir() if f.is_file() and f.name.lower() not in IGNORE_NAMES]
        for lf in loose_files:
            self.issues.append(Issue(
                severity="WARNING",
                issue_type="LOOSE_FILE",
                speaker=sp_name,
                digit="-",
                file_name=lf.name,
                path=str(lf),
                details=f"Extraneous file directly in speaker root: {lf.name}",
                recommendation="Move or delete loose files; speaker folders should only contain digit subdirectories (d0-d10)."
            ))

        # Check all 11 expected digits
        for digit in EXPECTED_DIGITS:
            if digit not in existing_subdirs:
                self.issues.append(Issue(
                    severity="CRITICAL",
                    issue_type="MISSING_DIGIT",
                    speaker=sp_name,
                    digit=digit,
                    file_name="-",
                    path=str(sp_path / digit),
                    details=f"Missing digit folder '{digit}' for speaker '{sp_name}'.",
                    recommendation=f"Add missing folder '{digit}' with required media."
                ))
                continue

            digit_dir = existing_subdirs[digit]
            if self.dataset_type == "VIDEOS":
                self.validate_video_digit(digit_dir, sp_name, digit)
            elif self.dataset_type == "FRAMES":
                self.validate_frames_digit(digit_dir, sp_name, digit)

    def validate_video_digit(self, digit_dir: Path, sp_name: str, digit: str) -> None:
        """Validates video file inside a digit directory."""
        all_files = [f for f in digit_dir.iterdir() if not f.name.startswith(".")]
        video_files = [f for f in all_files if f.is_file() and f.suffix.lower() in VIDEO_EXTS]
        other_files = [f for f in all_files if f.is_file() and f.suffix.lower() not in VIDEO_EXTS and f.name.lower() not in IGNORE_NAMES]

        # Flag other extraneous files
        for of in other_files:
            self.issues.append(Issue(
                severity="WARNING",
                issue_type="EXTRANEOUS_FILE",
                speaker=sp_name,
                digit=digit,
                file_name=of.name,
                path=str(of),
                details=f"Extraneous non-video file: {of.name}",
                recommendation="Remove extraneous files from digit directory."
            ))

        # Check video file count
        if len(video_files) == 0:
            self.issues.append(Issue(
                severity="CRITICAL",
                issue_type="EMPTY_DIGIT_FOLDER",
                speaker=sp_name,
                digit=digit,
                file_name="-",
                path=str(digit_dir),
                details="No video file found in digit directory.",
                recommendation=f"Place the video clip named '{digit}.mp4' into this directory."
            ))
            return

        if len(video_files) > 1:
            names = [f.name for f in video_files]
            self.issues.append(Issue(
                severity="CRITICAL",
                issue_type="MULTIPLE_VIDEOS",
                speaker=sp_name,
                digit=digit,
                file_name=", ".join(names),
                path=str(digit_dir),
                details=f"Multiple video files found ({len(video_files)}): {names}",
                recommendation=f"Keep only the single correct video and rename it to '{digit}.mp4'; remove others."
            ))

        # Check each video file
        for vid in video_files:
            expected_name = f"{digit}{vid.suffix.lower()}"
            if vid.name.lower() != expected_name:
                self.issues.append(Issue(
                    severity="CRITICAL",
                    issue_type="WRONG_FILENAME",
                    speaker=sp_name,
                    digit=digit,
                    file_name=vid.name,
                    path=str(vid),
                    details=f"Video file named '{vid.name}' instead of expected '{expected_name}'.",
                    recommendation=f"Rename '{vid.name}' to '{expected_name}'."
                ))

            # Check file size
            size_bytes = vid.stat().st_size
            if size_bytes == 0:
                self.issues.append(Issue(
                    severity="CRITICAL",
                    issue_type="ZERO_BYTE_FILE",
                    speaker=sp_name,
                    digit=digit,
                    file_name=vid.name,
                    path=str(vid),
                    details="Video file is 0 bytes (empty).",
                    recommendation="Re-encode or replace this corrupted video clip."
                ))
                continue
            elif size_bytes < 10240: # < 10 KB
                self.issues.append(Issue(
                    severity="WARNING",
                    issue_type="SUSPICIOUS_SMALL_FILE",
                    speaker=sp_name,
                    digit=digit,
                    file_name=vid.name,
                    path=str(vid),
                    details=f"Video file is suspiciously small ({size_bytes} bytes).",
                    recommendation="Verify that this clip plays completely without truncation."
                ))

            # Compute hash for duplicate tracking
            h = self._compute_sha256(vid)
            if h:
                self.file_hashes.setdefault(h, []).append((sp_name, digit, str(vid)))

            # OpenCV deep decode validation if installed
            if HAS_OPENCV:
                self._verify_video_stream(vid, sp_name, digit)

    def _verify_video_stream(self, vid_path: Path, sp_name: str, digit: str) -> None:
        """Deep decodes video stream using OpenCV."""
        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            self.issues.append(Issue(
                severity="CRITICAL",
                issue_type="UNREADABLE_VIDEO",
                speaker=sp_name,
                digit=digit,
                file_name=vid_path.name,
                path=str(vid_path),
                details="OpenCV could not open video stream.",
                recommendation="Re-encode file with ffmpeg (h264/yuv420p)."
            ))
            return

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        decoded_count = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            decoded_count += 1
        cap.release()

        if decoded_count == 0:
            self.issues.append(Issue(
                severity="CRITICAL",
                issue_type="ZERO_DECODED_FRAMES",
                speaker=sp_name,
                digit=digit,
                file_name=vid_path.name,
                path=str(vid_path),
                details="Video opened but 0 frames could be decoded.",
                recommendation="Re-encode or replace the video file."
            ))
        elif decoded_count < 15:
            self.issues.append(Issue(
                severity="WARNING",
                issue_type="VERY_SHORT_VIDEO",
                speaker=sp_name,
                digit=digit,
                file_name=vid_path.name,
                path=str(vid_path),
                details=f"Video clip is very short: only {decoded_count} frames ({decoded_count/fps:.2f}s).",
                recommendation="Ensure the full digit articulation is captured and not clipped prematurely."
            ))

    def validate_frames_digit(self, digit_dir: Path, sp_name: str, digit: str) -> None:
        """Validates sequential frame images inside a digit directory."""
        all_files = [f for f in digit_dir.iterdir() if not f.name.startswith(".")]
        image_files = [f for f in all_files if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
        
        if len(image_files) == 0:
            self.issues.append(Issue(
                severity="CRITICAL",
                issue_type="EMPTY_FRAME_FOLDER",
                speaker=sp_name,
                digit=digit,
                file_name="-",
                path=str(digit_dir),
                details="No frame images found in digit folder.",
                recommendation="Run frame extraction / cropping pipeline for this utterance."
            ))
            return

        if len(image_files) < 15:
            self.issues.append(Issue(
                severity="WARNING",
                issue_type="LOW_FRAME_COUNT",
                speaker=sp_name,
                digit=digit,
                file_name=f"{len(image_files)} frames",
                path=str(digit_dir),
                details=f"Only {len(image_files)} frames found (expected 25-50 frames).",
                recommendation="Check if video was trimmed too aggressively."
            ))

        # Check sequence continuity
        frame_nums: List[int] = []
        for img in image_files:
            m = re.search(r"(\d+)\.[a-zA-Z0-9]+$", img.name)
            if m:
                frame_nums.append(int(m.group(1)))

        if frame_nums:
            frame_nums.sort()
            min_idx = frame_nums[0]
            expected_seq = list(range(min_idx, min_idx + len(frame_nums)))
            if frame_nums != expected_seq:
                missing = sorted(set(expected_seq) - set(frame_nums))
                self.issues.append(Issue(
                    severity="WARNING",
                    issue_type="FRAME_SEQUENCE_GAP",
                    speaker=sp_name,
                    digit=digit,
                    file_name=f"Missing indices: {missing[:5]}",
                    path=str(digit_dir),
                    details=f"Sequence has gaps or missing indices: {missing[:5]}",
                    recommendation="Re-extract frames to ensure contiguous frame numbering."
                ))

        # PIL image decode check if installed
        if HAS_PIL:
            first_img = image_files[0]
            try:
                with Image.open(first_img) as im:
                    im.verify()
                with Image.open(first_img) as im:
                    w, h = im.size
                    if w != h:
                        self.issues.append(Issue(
                            severity="WARNING",
                            issue_type="NON_SQUARE_CROP",
                            speaker=sp_name,
                            digit=digit,
                            file_name=first_img.name,
                            path=str(first_img),
                            details=f"Frame dimensions are not square ({w}x{h}).",
                            recommendation="Check if 96x96 letterboxing crop was applied."
                        ))
            except Exception as e:
                self.issues.append(Issue(
                    severity="CRITICAL",
                    issue_type="CORRUPT_IMAGE",
                    speaker=sp_name,
                    digit=digit,
                    file_name=first_img.name,
                    path=str(first_img),
                    details=f"Image cannot be decoded: {e}",
                    recommendation="Re-export frame image."
                ))

    def _compute_sha256(self, path: Path) -> Optional[str]:
        try:
            h = hashlib.sha256()
            with open(path, "rb") as fp:
                while chunk := fp.read(65536):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return None

    def check_duplicate_hashes(self) -> None:
        """Flags identical media files across different samples."""
        for h, locations in self.file_hashes.items():
            if len(locations) > 1:
                # Group by speaker
                spks = {loc[0] for loc in locations}
                digits = {loc[1] for loc in locations}
                paths = [loc[2] for loc in locations]
                
                # If identical hash occurs in different digits, that is a severe error (copy-pasting wrong digit)
                if len(digits) > 1:
                    self.issues.append(Issue(
                        severity="CRITICAL",
                        issue_type="IDENTICAL_DIGIT_CLONES",
                        speaker=", ".join(sorted(spks)),
                        digit=", ".join(sorted(digits)),
                        file_name=Path(paths[0]).name,
                        path=" | ".join(paths[:3]),
                        details=f"Identical video content (exact SHA-256 match) used for DIFFERENT digits: {digits}",
                        recommendation="Replace copy-pasted duplicate with authentic distinct digit utterance video."
                    ))
                elif len(spks) > 1 and not any("_aug" in s for s in spks):
                    self.issues.append(Issue(
                        severity="WARNING",
                        issue_type="CROSS_SPEAKER_DUPLICATE",
                        speaker=", ".join(sorted(spks)),
                        digit=", ".join(sorted(digits)),
                        file_name=Path(paths[0]).name,
                        path=" | ".join(paths[:3]),
                        details=f"Identical video content used across different base speakers: {spks}",
                        recommendation="Verify if same speaker video was accidentally reused under different speaker IDs."
                    ))

    def print_report(self) -> bool:
        """Prints colorful formatted audit report."""
        criticals = [i for i in self.issues if i.severity == "CRITICAL"]
        warnings = [i for i in self.issues if i.severity == "WARNING"]

        print("\n" + "=" * 80)
        print("DATASET SANITIZATION AUDIT REPORT")
        print("=" * 80)
        print(f"Total Speakers Inspected : {len(self.speaker_dirs)}")
        print(f"Dataset Mode             : {self.dataset_type}")
        print(f"Critical Issues Found    : {len(criticals)}")
        print(f"Warnings Found           : {len(warnings)}")
        print("=" * 80)

        if criticals:
            print("\n❌ CRITICAL ISSUES (MUST FIX BEFORE TRAINING):")
            print("-" * 80)
            print(f"{'SPEAKER':<10} {'DIGIT':<6} {'ISSUE TYPE':<22} {'DETAILS'}")
            print("-" * 80)
            for c in criticals:
                print(f"{c.speaker:<10} {c.digit:<6} {c.issue_type:<22} {c.details}")
                print(f"   └─ Recommendation: {c.recommendation}")
                print(f"   └─ Path: {c.path}\n")

        if warnings:
            print("\n⚠️  WARNINGS & NON-CRITICAL ANOMALIES:")
            print("-" * 80)
            print(f"{'SPEAKER':<10} {'DIGIT':<6} {'ISSUE TYPE':<22} {'DETAILS'}")
            print("-" * 80)
            for w in warnings[:15]:  # limit terminal output to 15
                print(f"{w.speaker:<10} {w.digit:<6} {w.issue_type:<22} {w.details}")
                print(f"   └─ Path: {w.path}")
            if len(warnings) > 15:
                print(f"\n... and {len(warnings) - 15} more warnings (see full CSV report).")

        print("\n" + "=" * 80)
        if not criticals and not warnings:
            print("🎉 VERDICT: PERFECT PASS! Dataset is completely sanitized and ready.")
            print("=" * 80)
            return True
        elif not criticals:
            print("✅ VERDICT: PASS WITH WARNINGS. No critical blockers found.")
            print("=" * 80)
            return True
        else:
            print("❌ VERDICT: FAIL. Please fix the critical issues above before running training.")
            print("=" * 80)
            return False

    def save_reports(self) -> None:
        """Saves reports to CSV and JSON."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. CSV of issues
        csv_path = self.output_dir / "issues_inventory.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=[
                "severity", "issue_type", "speaker", "digit", "file_name", "path", "details", "recommendation"
            ])
            writer.writeheader()
            for iss in self.issues:
                writer.writerow(asdict(iss))

        # 2. JSON Summary
        json_path = self.output_dir / "sanitization_summary.json"
        summary = {
            "target_directory": str(self.target_path),
            "dataset_mode": self.dataset_type,
            "total_speakers": len(self.speaker_dirs),
            "speakers_list": [d.name for d in self.speaker_dirs],
            "total_critical_issues": len([i for i in self.issues if i.severity == "CRITICAL"]),
            "total_warnings": len([i for i in self.issues if i.severity == "WARNING"]),
            "verdict": "FAIL" if any(i.severity == "CRITICAL" for i in self.issues) else ("PASS WITH WARNINGS" if self.issues else "PASS"),
            "issues": [asdict(i) for i in self.issues]
        }
        with open(json_path, "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)

        print(f"\nSaved Reports:")
        print(f"  • CSV Issues Inventory : {csv_path}")
        print(f"  • JSON Audit Summary   : {json_path}\n")


def main():
    target = TARGET_DATASET_DIR
    if len(sys.argv) > 1:
        target = sys.argv[1]

    sanitizer = DatasetSanitizer(target_dir=target)
    success = sanitizer.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
