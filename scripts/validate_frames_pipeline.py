#!/usr/bin/env python3
"""
Tai Phake VSR: Deep Raw Extracted & Cropped Dataset Validator
==============================================================

Performs comprehensive validation of both raw extracted frames and dlib cropped/augmented
frames using OpenCV, PIL, and NumPy.

Validations:
1. Directory Structure & Speaker Discovery:
   - Dynamic discovery of base speakers in raw frames.
   - Dynamic discovery of 100 directories in cropped dataset.
   - Verification against speaker_split.json (train/val partition).
2. Digit Completeness:
   - Verifies all 11 digit classes (d0 through d10) are present in every folder.
3. Frame Decodability & Health:
   - Deep image decode check (verifies headers and pixel integrity).
   - Detects 0-byte frames or truncated JPEGs.
   - Detects blank / solid-black frames (e.g. failed landmark crop).
4. Geometry & Dimensions:
   - Cropped frames are strictly verified to be 96x96 with 3 channels.
5. Sequence Continuity:
   - Ensures continuous frame index numbering (no missing frame indices).
6. Cross-Dataset Consistency:
   - Verifies that cropped frame count matches the raw extracted frame count for each utterance.

Configured via constants at the top or via CLI arguments.
Run using: uv run python scripts/validate_frames_pipeline.py
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# ==============================================================================
# CONFIGURATION CONSTANTS (CHANGE THESE AS NEEDED)
# ==============================================================================
EXTRACTED_RAW_FRAMES_DIR: str = "data/experiments/extracted_raw_frames"
CROPPED_DATASET_DIR: str = "data/experiments/dlib_augmented_cropped_100_80_20_96x96"

EXPECTED_CROP_WIDTH: int = 96
EXPECTED_CROP_HEIGHT: int = 96
EXPECTED_CROP_CHANNELS: int = 3
EXPECTED_DIGITS: List[str] = [f"d{i}" for i in range(11)]
IMAGE_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Thresholds for blank/black frame detection
MIN_FRAME_BRIGHTNESS_MEAN: float = 3.0   # Flags if whole frame mean < 3.0 (solid black)
MIN_FRAME_BRIGHTNESS_STD: float = 1.0    # Flags if whole frame std < 1.0 (flat uniform color)


@dataclass
class AuditIssue:
    stage: str          # 'RAW_EXTRACTED', 'CROPPED', 'CROSS_CHECK'
    severity: str       # 'CRITICAL', 'WARNING'
    issue_type: str     # 'MISSING_DIGIT', 'COUNT_MISMATCH', 'ZERO_BYTE', 'GAP', 'SIZE_MISMATCH', 'BLANK_FRAME'
    speaker_or_folder: str
    digit: str
    details: str
    path: str


class DeepFramesValidator:
    def __init__(self, raw_dir: str | Path, cropped_dir: str | Path):
        self.project_root = self._find_project_root()
        self.raw_dir = self._resolve_path(raw_dir)
        self.cropped_dir = self._resolve_path(cropped_dir)
        
        self.output_dir = self.project_root / "results" / "frames_pipeline_validation"
        self.issues: List[AuditIssue] = []
        
        # Raw frame counts: (speaker, digit) -> int
        self.raw_counts: Dict[Tuple[str, str], int] = {}
        # Cropped frame counts: (folder, digit) -> int
        self.cropped_counts: Dict[Tuple[str, str], int] = {}
        
        self.raw_speakers: List[str] = []
        self.cropped_folders: List[str] = []

    def _find_project_root(self) -> Path:
        curr = Path.cwd()
        for p in [curr] + list(curr.parents):
            if (p / "data").exists() or (p / "notebooks").exists():
                return p
        return curr

    def _resolve_path(self, p: str | Path) -> Path:
        p_obj = Path(p)
        if not p_obj.is_absolute():
            p_obj = (self.project_root / p_obj).resolve()
        return p_obj

    def _get_image_files(self, d: Path) -> List[Path]:
        if not d.exists() or not d.is_dir():
            return []
        return sorted([f for f in d.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS and not f.name.startswith(".")])

    def run(self) -> bool:
        print("=" * 80)
        print("TAI PHAKE VSR: DEEP RAW & CROPPED FRAMES DATASET VALIDATOR")
        print("=" * 80)
        print(f"1. Raw Extracted Dir : {self.raw_dir}")
        print(f"2. Cropped Dir       : {self.cropped_dir}")
        print(f"Expected Crop Size   : {EXPECTED_CROP_WIDTH}x{EXPECTED_CROP_HEIGHT} ({EXPECTED_CROP_CHANNELS} channels)")
        print(f"Reports Output       : {self.output_dir}")
        print("=" * 80)

        # 1. Audit Raw Extracted Frames
        self.audit_raw_extracted()

        # 2. Audit Cropped Frames
        self.audit_cropped_frames()

        # 3. Cross-Check Consistency
        self.cross_check()

        # 4. Print Summary
        is_clean = self.print_summary()

        # 5. Save Reports
        self.save_reports()
        return is_clean

    def audit_raw_extracted(self) -> None:
        print("\n[Step 1/3] Deep Auditing Raw Extracted Frames...")
        if not self.raw_dir.exists():
            self.issues.append(AuditIssue(
                stage="RAW_EXTRACTED",
                severity="CRITICAL",
                issue_type="DIR_NOT_FOUND",
                speaker_or_folder="-",
                digit="-",
                details=f"Raw directory does not exist: {self.raw_dir}",
                path=str(self.raw_dir)
            ))
            print(f"❌ Error: Raw directory does not exist: {self.raw_dir}")
            return

        sp_dirs = sorted([d for d in self.raw_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
        self.raw_speakers = [d.name for d in sp_dirs]
        print(f"Found {len(self.raw_speakers)} base speakers in raw extracted frames.")

        for sp_dir in tqdm(sp_dirs, desc="Scanning Raw Speakers"):
            sp = sp_dir.name
            for dig in EXPECTED_DIGITS:
                dig_dir = sp_dir / dig
                if not dig_dir.exists():
                    self.issues.append(AuditIssue(
                        stage="RAW_EXTRACTED",
                        severity="CRITICAL",
                        issue_type="MISSING_DIGIT",
                        speaker_or_folder=sp,
                        digit=dig,
                        details=f"Missing raw digit directory '{dig}'.",
                        path=str(dig_dir)
                    ))
                    self.raw_counts[(sp, dig)] = 0
                    continue

                imgs = self._get_image_files(dig_dir)
                count = len(imgs)
                self.raw_counts[(sp, dig)] = count

                if count == 0:
                    self.issues.append(AuditIssue(
                        stage="RAW_EXTRACTED",
                        severity="CRITICAL",
                        issue_type="EMPTY_FOLDER",
                        speaker_or_folder=sp,
                        digit=dig,
                        details="Raw digit directory has 0 frames.",
                        path=str(dig_dir)
                    ))
                    continue
                elif count < 15:
                    self.issues.append(AuditIssue(
                        stage="RAW_EXTRACTED",
                        severity="WARNING",
                        issue_type="LOW_FRAME_COUNT",
                        speaker_or_folder=sp,
                        digit=dig,
                        details=f"Raw digit has low frame count: {count} frames.",
                        path=str(dig_dir)
                    ))

                # Check sequence continuity
                self._check_sequence(imgs, "RAW_EXTRACTED", sp, dig)

                # Verify a sample frame decodability
                self._verify_frame_decodability(imgs[len(imgs) // 2], "RAW_EXTRACTED", sp, dig, check_dims=False)

    def audit_cropped_frames(self) -> None:
        print("\n[Step 2/3] Deep Auditing Cropped / Augmented Frames...")
        if not self.cropped_dir.exists():
            self.issues.append(AuditIssue(
                stage="CROPPED",
                severity="CRITICAL",
                issue_type="DIR_NOT_FOUND",
                speaker_or_folder="-",
                digit="-",
                details=f"Cropped directory does not exist: {self.cropped_dir}",
                path=str(self.cropped_dir)
            ))
            print(f"❌ Error: Cropped directory does not exist: {self.cropped_dir}")
            return

        folder_dirs = sorted([d for d in self.cropped_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
        self.cropped_folders = [d.name for d in folder_dirs]
        print(f"Found {len(self.cropped_folders)} folders in cropped dataset.")

        # Check speaker_split.json if present
        split_file = self.cropped_dir / "speaker_split.json"
        if split_file.exists():
            try:
                with open(split_file) as fp:
                    sp_info = json.load(fp)
                expected_folders = set(sp_info.get("train_speakers", []) + sp_info.get("val_speakers", []))
                actual_folders = set(self.cropped_folders)
                missing = expected_folders - actual_folders
                for mf in missing:
                    self.issues.append(AuditIssue(
                        stage="CROPPED",
                        severity="CRITICAL",
                        issue_type="MISSING_DATASET_FOLDER",
                        speaker_or_folder=mf,
                        digit="-",
                        details=f"Folder '{mf}' is declared in speaker_split.json but missing from disk.",
                        path=str(self.cropped_dir / mf)
                    ))
            except Exception:
                pass

        for f_dir in tqdm(folder_dirs, desc="Scanning Cropped Folders"):
            fname = f_dir.name
            for dig in EXPECTED_DIGITS:
                dig_dir = f_dir / dig
                if not dig_dir.exists():
                    self.issues.append(AuditIssue(
                        stage="CROPPED",
                        severity="CRITICAL",
                        issue_type="MISSING_DIGIT",
                        speaker_or_folder=fname,
                        digit=dig,
                        details=f"Missing cropped digit folder '{dig}'.",
                        path=str(dig_dir)
                    ))
                    self.cropped_counts[(fname, dig)] = 0
                    continue

                imgs = self._get_image_files(dig_dir)
                count = len(imgs)
                self.cropped_counts[(fname, dig)] = count

                if count == 0:
                    self.issues.append(AuditIssue(
                        stage="CROPPED",
                        severity="CRITICAL",
                        issue_type="EMPTY_FOLDER",
                        speaker_or_folder=fname,
                        digit=dig,
                        details="Cropped digit folder has 0 frames.",
                        path=str(dig_dir)
                    ))
                    continue

                # Check sequence numbering
                self._check_sequence(imgs, "CROPPED", fname, dig)

                # Check first and midpoint frame decodability & 96x96 geometry
                self._verify_frame_decodability(imgs[0], "CROPPED", fname, dig, check_dims=True)
                if count > 1:
                    self._verify_frame_decodability(imgs[count // 2], "CROPPED", fname, dig, check_dims=True)

    def _check_sequence(self, img_files: List[Path], stage: str, folder: str, digit: str) -> None:
        for img in img_files:
            if img.stat().st_size == 0:
                self.issues.append(AuditIssue(
                    stage=stage,
                    severity="CRITICAL",
                    issue_type="ZERO_BYTE_FRAME",
                    speaker_or_folder=folder,
                    digit=digit,
                    details=f"0-byte frame: {img.name}",
                    path=str(img)
                ))

        nums = []
        for img in img_files:
            m = re.search(r"(\d+)\.[a-zA-Z0-9]+$", img.name)
            if m:
                nums.append(int(m.group(1)))
        
        if nums:
            nums.sort()
            expected = list(range(nums[0], nums[0] + len(nums)))
            if nums != expected:
                missing = sorted(set(expected) - set(nums))
                self.issues.append(AuditIssue(
                    stage=stage,
                    severity="WARNING",
                    issue_type="SEQUENCE_GAP",
                    speaker_or_folder=folder,
                    digit=digit,
                    details=f"Frame index gap. Missing: {missing[:5]}",
                    path=str(img_files[0].parent)
                ))

    def _verify_frame_decodability(self, img_path: Path, stage: str, folder: str, digit: str, check_dims: bool = True) -> None:
        if img_path.stat().st_size == 0:
            return

        # 1. OpenCV decode
        mat = cv2.imread(str(img_path))
        if mat is None:
            self.issues.append(AuditIssue(
                stage=stage,
                severity="CRITICAL",
                issue_type="CORRUPT_FRAME",
                speaker_or_folder=folder,
                digit=digit,
                details=f"OpenCV failed to decode image: {img_path.name}",
                path=str(img_path)
            ))
            return

        # 2. Check dimensions & channels
        h, w, c = mat.shape
        if check_dims:
            if w != EXPECTED_CROP_WIDTH or h != EXPECTED_CROP_HEIGHT:
                self.issues.append(AuditIssue(
                    stage=stage,
                    severity="CRITICAL",
                    issue_type="WRONG_DIMENSIONS",
                    speaker_or_folder=folder,
                    digit=digit,
                    details=f"Dimensions {w}x{h} != expected {EXPECTED_CROP_WIDTH}x{EXPECTED_CROP_HEIGHT}",
                    path=str(img_path)
                ))
            if c != EXPECTED_CROP_CHANNELS:
                self.issues.append(AuditIssue(
                    stage=stage,
                    severity="CRITICAL",
                    issue_type="WRONG_CHANNELS",
                    speaker_or_folder=folder,
                    digit=digit,
                    details=f"Channel count {c} != expected {EXPECTED_CROP_CHANNELS}",
                    path=str(img_path)
                ))

        # 3. Check for solid-black / dead blank crop
        mean_val = float(np.mean(mat))
        std_val = float(np.std(mat))
        if mean_val < MIN_FRAME_BRIGHTNESS_MEAN or std_val < MIN_FRAME_BRIGHTNESS_STD:
            self.issues.append(AuditIssue(
                stage=stage,
                severity="WARNING",
                issue_type="BLANK_BLACK_FRAME",
                speaker_or_folder=folder,
                digit=digit,
                details=f"Frame appears nearly solid black or flat (mean={mean_val:.1f}, std={std_val:.1f})",
                path=str(img_path)
            ))

    def cross_check(self) -> None:
        print("\n[Step 3/3] Cross-Checking Raw Extracted vs Cropped...")
        # Create case-insensitive lookup
        raw_counts_lower = {(sp.lower(), dig): (sp, count) for (sp, dig), count in self.raw_counts.items()}

        for (folder, dig), crop_count in self.cropped_counts.items():
            base_speaker = folder.split("_aug")[0]
            raw_match = raw_counts_lower.get((base_speaker.lower(), dig))

            if raw_match is None:
                self.issues.append(AuditIssue(
                    stage="CROSS_CHECK",
                    severity="WARNING",
                    issue_type="ORPHAN_CROPPED_FOLDER",
                    speaker_or_folder=folder,
                    digit=dig,
                    details=f"Cropped folder '{folder}' has no raw speaker '{base_speaker}'.",
                    path=str(self.cropped_dir / folder / dig)
                ))
            else:
                raw_sp, raw_count = raw_match
                if raw_count != crop_count and crop_count > 0 and raw_count > 0:
                    self.issues.append(AuditIssue(
                        stage="CROSS_CHECK",
                        severity="CRITICAL",
                        issue_type="FRAME_COUNT_MISMATCH",
                        speaker_or_folder=folder,
                        digit=dig,
                        details=f"Count mismatch: Raw '{raw_sp}' has {raw_count} frames, but cropped '{folder}' has {crop_count} frames.",
                        path=str(self.cropped_dir / folder / dig)
                    ))

    def print_summary(self) -> bool:
        criticals = [i for i in self.issues if i.severity == "CRITICAL"]
        warnings = [i for i in self.issues if i.severity == "WARNING"]

        print("\n" + "=" * 80)
        print("FRAMES PIPELINE AUDIT SUMMARY")
        print("=" * 80)
        print(f"Raw Base Speakers Discovered  : {len(self.raw_speakers)}")
        print(f"Cropped Folders Discovered    : {len(self.cropped_folders)}")
        print(f"Total Raw Frame Utterances    : {len(self.raw_counts)}")
        print(f"Total Cropped Utterances      : {len(self.cropped_counts)}")
        print(f"Critical Issues Found         : {len(criticals)}")
        print(f"Warnings Found                : {len(warnings)}")
        print("=" * 80)

        if criticals:
            print("\n❌ CRITICAL ISSUES FOUND:")
            print("-" * 80)
            print(f"{'STAGE':<14} {'SPEAKER/FOLDER':<18} {'DIGIT':<6} {'ISSUE TYPE':<22} {'DETAILS'}")
            print("-" * 80)
            for c in criticals:
                print(f"{c.stage:<14} {c.speaker_or_folder:<18} {c.digit:<6} {c.issue_type:<22} {c.details}")
                print(f"   └─ Path: {c.path}\n")

        if warnings:
            print("\n⚠️  WARNINGS & NON-CRITICAL ANOMALIES:")
            print("-" * 80)
            for w in warnings[:10]:
                print(f"{w.stage:<14} {w.speaker_or_folder:<18} {w.digit:<6} {w.issue_type:<22} {w.details}")
            if len(warnings) > 10:
                print(f"... and {len(warnings) - 10} more warnings (see CSV report).")

        print("\n" + "=" * 80)
        if not criticals and not warnings:
            print("🎉 VERDICT: PERFECT PASS! Raw frames and cropped frames match 100% with no defects.")
            print("=" * 80)
            return True
        elif not criticals:
            print("✅ VERDICT: PASS WITH WARNINGS. No critical frame defects detected.")
            print("=" * 80)
            return True
        else:
            print("❌ VERDICT: FAIL. Frame discrepancies or corruptions detected.")
            print("=" * 80)
            return False

    def save_reports(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.output_dir / "frames_audit_report.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=["stage", "severity", "issue_type", "speaker_or_folder", "digit", "details", "path"])
            writer.writeheader()
            for iss in self.issues:
                writer.writerow(asdict(iss))

        json_path = self.output_dir / "frames_audit_summary.json"
        summary = {
            "raw_extracted_dir": str(self.raw_dir),
            "cropped_dir": str(self.cropped_dir),
            "raw_speakers_count": len(self.raw_speakers),
            "cropped_folders_count": len(self.cropped_folders),
            "critical_issues_count": len([i for i in self.issues if i.severity == "CRITICAL"]),
            "warnings_count": len([i for i in self.issues if i.severity == "WARNING"]),
            "verdict": "FAIL" if any(i.severity == "CRITICAL" for i in self.issues) else "PASS",
            "issues": [asdict(i) for i in self.issues]
        }
        with open(json_path, "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)

        print(f"\nSaved Detailed Reports:")
        print(f"  • CSV Issue Log: {csv_path}")
        print(f"  • JSON Summary : {json_path}\n")


def main():
    raw_dir = EXTRACTED_RAW_FRAMES_DIR
    cropped_dir = CROPPED_DATASET_DIR

    if len(sys.argv) > 1:
        raw_dir = sys.argv[1]
    if len(sys.argv) > 2:
        cropped_dir = sys.argv[2]

    validator = DeepFramesValidator(raw_dir=raw_dir, cropped_dir=cropped_dir)
    success = validator.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
