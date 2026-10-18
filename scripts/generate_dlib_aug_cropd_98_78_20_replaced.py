#!/usr/bin/env python3
"""
Pipeline: End-to-End Frame Extraction, Dlib Method D (Letterboxed 96x96) Lip ROI Cropping,
and 98-Folder (78 Train + 20 Val) Augmented Dataset Generation for Replaced Videos.

Tai Phake Visual Speech Recognition (VSR) Project
Dataset: data/experiments/dlib_aug_cropd_98_78_20_96x96_replaced
Results: results/dlib_aug_cropd_98_78_20_96x96_replaced
"""

import os
import sys
import re
import json
import time
import shutil
import hashlib
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple, Optional, Any

import cv2
import dlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==============================================================================
# Configuration & Constants
# ==============================================================================
DEFAULT_SEED = 42
random.seed(DEFAULT_SEED)
np.random.seed(DEFAULT_SEED)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

VIDEO_INPUT_DIR = DATA_ROOT / "experiments" / "digit_videos_only"
EXTRACTED_RAW_FRAMES_DIR = DATA_ROOT / "experiments" / "extracted_raw_frames_replaced"
DATASET_NAME = "dlib_aug_cropd_98_78_20_96x96_replaced"
OUTPUT_DATASET_DIR = DATA_ROOT / "experiments" / DATASET_NAME
RESULTS_DIR = PROJECT_ROOT / "results" / DATASET_NAME
DLIB_PREDICTOR_PATH = PROJECT_ROOT / "models" / "dlib" / "shape_predictor_68_face_landmarks.dat"

# Method D Cropping Parameters
TARGET_WIDTH = 96
TARGET_HEIGHT = 96
PADDING_X = 15      # Horizontal margin in px
PADDING_Y = 10      # Vertical margin in px
BORDER_VALUE = 0    # Black letterbox canvas fill
JPEG_QUALITY = 95

# 98-Folder 78/20 Split Configuration
TARGET_TRAIN_FOLDERS = 78
TARGET_VAL_FOLDERS = 20
TOTAL_TARGET_FOLDERS = TARGET_TRAIN_FOLDERS + TARGET_VAL_FOLDERS  # 98

# Base Speakers: 22 Train, 6 Validation (s27 and s29 excluded)
TRAIN_BASE_SPEAKERS = [f"s{i}" for i in range(1, 23)]  # s1 through s22
VAL_BASE_SPEAKERS = ["s23", "s24", "s25", "s26", "s28", "s30"]


def natural_sort_key(s: str) -> int:
    m = re.search(r"\d+", s)
    return int(m.group(0)) if m else 0


def build_speaker_folder_mapping(
    base_speakers: List[str],
    target_count: int,
    prefix_aug: str = "aug",
    custom_four_folders: Optional[List[str]] = None
) -> Dict[str, List[str]]:
    """
    Maps base speakers to exact folder directories (original + augmentations)
    to achieve the exact target folder count.
    """
    n_base = len(base_speakers)
    assert target_count >= n_base, f"Target count {target_count} < base speakers {n_base}"
    extra_folders_needed = target_count - n_base
    base_augs_per_spk = extra_folders_needed // n_base
    remainder = extra_folders_needed % n_base

    mapping: Dict[str, List[str]] = {}
    four_folder_set = set(custom_four_folders) if custom_four_folders else set()

    for idx, spk in enumerate(base_speakers):
        dirs = [spk]
        if custom_four_folders is not None:
            num_augs = base_augs_per_spk + (1 if spk in four_folder_set else 0)
        else:
            num_augs = base_augs_per_spk + (1 if idx < remainder else 0)

        for a_idx in range(1, num_augs + 1):
            dirs.append(f"{spk}_{prefix_aug}{a_idx}")
        mapping[spk] = dirs

    total_dirs = sum(len(v) for v in mapping.values())
    assert total_dirs == target_count, f"Generated {total_dirs} != target {target_count}"
    return mapping


# ==============================================================================
# Helper: Dlib Letterbox Cropping & Landmark Extraction
# ==============================================================================
def extract_method_d_crop(
    img: np.ndarray,
    center_x: float,
    center_y: float,
    raw_width: float,
    raw_height: float,
    pad_x: int = PADDING_X,
    pad_y: int = PADDING_Y,
    canvas_size: int = TARGET_WIDTH,
    border_val: int = BORDER_VALUE
) -> Tuple[np.ndarray, int, int, int, int]:
    """
    Extracts an aspect-preserving letterboxed mouth ROI from full frame image:
    1. Bounding box computation with padding
    2. Proportional aspect-preserving scaling
    3. Neutral centered letterboxing on canvas_size x canvas_size
    """
    h_img, w_img = img.shape[:2]
    min_x = center_x - raw_width / 2.0
    max_x = center_x + raw_width / 2.0
    min_y = center_y - raw_height / 2.0
    max_y = center_y + raw_height / 2.0

    x1 = max(0, int(round(min_x - pad_x)))
    y1 = max(0, int(round(min_y - pad_y)))
    x2 = min(w_img, int(round(max_x + pad_x)))
    y2 = min(h_img, int(round(max_y + pad_y)))

    roi = img[y1:y2, x1:x2]
    h_roi, w_roi = roi.shape[:2]

    if h_roi == 0 or w_roi == 0:
        return np.full((canvas_size, canvas_size, 3), border_val, dtype=np.uint8), 0, 0, 0, 0

    scale = canvas_size / max(w_roi, h_roi)
    new_w = min(canvas_size, max(1, int(round(w_roi * scale))))
    new_h = min(canvas_size, max(1, int(round(h_roi * scale))))

    resized_roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((canvas_size, canvas_size, 3), border_val, dtype=np.uint8)
    offset_x = (canvas_size - new_w) // 2
    offset_y = (canvas_size - new_h) // 2

    canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized_roi
    pad_h = canvas_size - new_w
    pad_v = canvas_size - new_h

    return canvas, new_w, new_h, pad_h, pad_v


# Worker global detector instance
_GLOBAL_DETECTOR = None
_GLOBAL_PREDICTOR = None

def init_dlib_worker(predictor_path: str):
    global _GLOBAL_DETECTOR, _GLOBAL_PREDICTOR
    _GLOBAL_DETECTOR = dlib.get_frontal_face_detector()
    _GLOBAL_PREDICTOR = dlib.shape_predictor(predictor_path)


# ==============================================================================
# Augmentation Functions
# ==============================================================================
def get_geometric_rng(base_seed: int, speaker: str, digit: str, variant_idx: int) -> np.random.Generator:
    key = f"{base_seed}_{speaker}_{digit}_aug{variant_idx}_geom".encode("utf-8")
    seed_uint32 = int.from_bytes(hashlib.sha256(key).digest()[:4], byteorder="little")
    return np.random.default_rng(seed_uint32)


def get_photometric_rng(base_seed: int, speaker: str, digit: str, frame: str, variant_idx: int) -> np.random.Generator:
    key = f"{base_seed}_{speaker}_{digit}_{frame}_aug{variant_idx}_photo".encode("utf-8")
    seed_uint32 = int.from_bytes(hashlib.sha256(key).digest()[:4], byteorder="little")
    return np.random.default_rng(seed_uint32)


def sample_geometric_transform(
    rng: np.random.Generator,
    width: int = TARGET_WIDTH,
    height: int = TARGET_HEIGHT
) -> np.ndarray:
    tx = float(rng.uniform(-3.0, 3.0))
    ty = float(rng.uniform(-3.0, 3.0))
    angle = float(rng.uniform(-3.0, 3.0))
    scale = float(rng.uniform(0.97, 1.03))

    center = (width / 2.0, height / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    return M


def apply_conservative_augmentation(
    img: np.ndarray,
    M: np.ndarray,
    rng: np.random.Generator
) -> np.ndarray:
    h, w = img.shape[:2]

    # 1. Geometric warp with reflection border
    transformed = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    # 2. Photometric: Contrast (0.92 - 1.08) and Brightness (-10 to +10)
    alpha = float(rng.uniform(0.92, 1.08))
    beta = float(rng.uniform(-10.0, 10.0))
    photo = np.clip(alpha * transformed.astype(np.float32) + beta, 0.0, 255.0).astype(np.uint8)

    # 3. Gaussian sensor noise (50% probability, sigma 1.0 - 3.5)
    if rng.random() < 0.50:
        sigma = float(rng.uniform(1.0, 3.5))
        noise = rng.normal(0.0, sigma, photo.shape)
        photo = np.clip(photo.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)

    # 4. Mild Gaussian blur (30% probability, 3x3 kernel, sigma 0.5)
    if rng.random() < 0.30:
        photo = cv2.GaussianBlur(photo, (3, 3), 0.5)

    return photo


# ==============================================================================
# Worker: Video Frame Extraction
# ==============================================================================
def extract_single_video_task(args: Tuple[str, str, str, str, str]) -> Dict[str, Any]:
    speaker, digit, video_path_str, raw_out_dir_str, quality_str = args
    video_path = Path(video_path_str)
    out_dir = Path(raw_out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "speaker": speaker,
            "digit": digit,
            "video_path": video_path_str,
            "status": "FAILED",
            "extracted_count": 0,
            "error": "Failed to open video file"
        }

    frame_count = 0
    save_quality = int(quality_str)
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame_count += 1
        out_frame_path = out_dir / f"frame_{frame_count:04d}.jpg"
        cv2.imwrite(str(out_frame_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), save_quality])

    cap.release()
    return {
        "speaker": speaker,
        "digit": digit,
        "video_path": video_path_str,
        "status": "SUCCESS" if frame_count > 0 else "FAILED",
        "extracted_count": frame_count,
        "error": None if frame_count > 0 else "0 frames read"
    }


# ==============================================================================
# Worker: Dlib Cropping & Augmentation for One Base Speaker
# ==============================================================================
def process_single_speaker_task(args: Tuple[str, List[str], str, str, str, int]) -> Dict[str, Any]:
    base_speaker, target_folders, split_name, raw_frames_dir_str, output_root_str, base_seed = args
    global _GLOBAL_DETECTOR, _GLOBAL_PREDICTOR

    raw_frames_dir = Path(raw_frames_dir_str) / base_speaker
    output_root_dir = Path(output_root_str)

    manifest_records = []
    geom_records = []

    if not raw_frames_dir.exists():
        return {
            "base_speaker": base_speaker,
            "manifest_records": [],
            "geom_records": [],
            "status": "SKIPPED_NO_RAW_DIR"
        }

    # Discover all digits present for this speaker
    digit_dirs = sorted([d for d in raw_frames_dir.iterdir() if d.is_dir() and not d.name.startswith(".")], key=lambda x: natural_sort_key(x.name))
    if not digit_dirs:
        return {
            "base_speaker": base_speaker,
            "manifest_records": [],
            "geom_records": [],
            "status": "SKIPPED_NO_DIGITS"
        }

    aug_variants = [f for f in target_folders if f != base_speaker]

    # Pre-sample geometric transform once per (digit, variant)
    geo_transforms = {}
    for dig_dir in digit_dirs:
        dig = dig_dir.name
        for v_folder in aug_variants:
            m = re.search(r"aug(\d+)", v_folder)
            v_idx = int(m.group(1)) if m else 1
            geo_rng = get_geometric_rng(base_seed, base_speaker, dig, v_idx)
            geo_transforms[(dig, v_folder)] = sample_geometric_transform(geo_rng, TARGET_WIDTH, TARGET_HEIGHT)

    # Process each digit sequentially
    for dig_dir in digit_dirs:
        dig = dig_dir.name
        frame_files = sorted([f for f in dig_dir.glob("*.jpg") if not f.name.startswith(".")], key=lambda x: natural_sort_key(x.name))
        if not frame_files:
            continue

        # Tracking state across frames of this utterance
        last_rect = None
        last_geom = None

        for fpath in frame_files:
            fname = fpath.name
            raw_img = cv2.imread(str(fpath))
            if raw_img is None:
                continue

            # 1. Dlib landmark detection
            geom = None
            if _GLOBAL_DETECTOR is not None and _GLOBAL_PREDICTOR is not None:
                gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
                faces = _GLOBAL_DETECTOR(gray, 0)
                rect = None
                if faces:
                    rect = faces[0]
                    last_rect = rect
                elif last_rect is not None:
                    rect = last_rect

                if rect is not None:
                    try:
                        shape = _GLOBAL_PREDICTOR(gray, rect)
                        pts = np.array([[shape.part(i).x, shape.part(i).y] for i in range(48, 68)], dtype=np.float32)
                        min_x, min_y = pts.min(axis=0)
                        max_x, max_y = pts.max(axis=0)
                        geom = {
                            "center_x": float((min_x + max_x) / 2.0),
                            "center_y": float((min_y + max_y) / 2.0),
                            "raw_width": float(max_x - min_x),
                            "raw_height": float(max_y - min_y)
                        }
                        last_geom = geom
                    except Exception:
                        geom = last_geom
                else:
                    geom = last_geom

            # 2. Extract Method D letterbox crop
            if geom is not None:
                crop_img, sw, sh, ph, pv = extract_method_d_crop(
                    raw_img, geom["center_x"], geom["center_y"], geom["raw_width"], geom["raw_height"],
                    PADDING_X, PADDING_Y, TARGET_WIDTH, BORDER_VALUE
                )
            else:
                # Fallback center crop if detection completely missed
                h_i, w_i = raw_img.shape[:2]
                cx, cy = w_i / 2.0, h_i / 2.0
                crop_img, sw, sh, ph, pv = extract_method_d_crop(
                    raw_img, cx, cy, 100.0, 50.0,
                    PADDING_X, PADDING_Y, TARGET_WIDTH, BORDER_VALUE
                )

            # 3. Save original crop
            dst_orig_dir = output_root_dir / base_speaker / dig
            dst_orig_dir.mkdir(parents=True, exist_ok=True)
            dst_orig_file = dst_orig_dir / fname
            cv2.imwrite(str(dst_orig_file), crop_img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])

            manifest_records.append({
                "split": split_name,
                "base_speaker": base_speaker,
                "folder_name": base_speaker,
                "is_augmented": False,
                "variant": "original",
                "digit": dig,
                "frame": fname,
                "output_path": str(dst_orig_file),
                "width": TARGET_WIDTH,
                "height": TARGET_HEIGHT
            })

            geom_records.append({
                "speaker": base_speaker,
                "digit": dig,
                "frame": fname,
                "status": "SUCCESS" if geom is not None else "FALLBACK",
                "center_x": geom["center_x"] if geom else 0.0,
                "center_y": geom["center_y"] if geom else 0.0,
                "raw_width": geom["raw_width"] if geom else 0.0,
                "raw_height": geom["raw_height"] if geom else 0.0,
                "scaled_w": sw,
                "scaled_h": sh,
                "pad_h": ph,
                "pad_v": pv,
                "output_w": TARGET_WIDTH,
                "output_h": TARGET_HEIGHT
            })

            # 4. Generate and save augmented variants
            for v_folder in aug_variants:
                m = re.search(r"aug(\d+)", v_folder)
                v_idx = int(m.group(1)) if m else 1
                M = geo_transforms.get((dig, v_folder))
                photo_rng = get_photometric_rng(base_seed, base_speaker, dig, fname, v_idx)

                aug_img = apply_conservative_augmentation(crop_img, M, photo_rng)
                dst_aug_dir = output_root_dir / v_folder / dig
                dst_aug_dir.mkdir(parents=True, exist_ok=True)
                dst_aug_file = dst_aug_dir / fname
                cv2.imwrite(str(dst_aug_file), aug_img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])

                manifest_records.append({
                    "split": split_name,
                    "base_speaker": base_speaker,
                    "folder_name": v_folder,
                    "is_augmented": True,
                    "variant": f"aug{v_idx}",
                    "digit": dig,
                    "frame": fname,
                    "output_path": str(dst_aug_file),
                    "width": TARGET_WIDTH,
                    "height": TARGET_HEIGHT
                })

    return {
        "base_speaker": base_speaker,
        "manifest_records": manifest_records,
        "geom_records": geom_records,
        "status": "SUCCESS"
    }


# ==============================================================================
# Main Pipeline Orchestrator
# ==============================================================================
def main():
    print("=" * 80)
    print("      TAI PHAKE VSR: END-TO-END 98-FOLDER DATASET GENERATION PIPELINE")
    print("=" * 80)
    print(f"Video Input Source : {VIDEO_INPUT_DIR}")
    print(f"Extracted Raw Dirs : {EXTRACTED_RAW_FRAMES_DIR}")
    print(f"Target Dataset Dir : {OUTPUT_DATASET_DIR}")
    print(f"Results Directory  : {RESULTS_DIR}")
    print(f"Dlib Model File    : {DLIB_PREDICTOR_PATH}")
    print(f"Cropping Method    : Method D Letterbox ({TARGET_WIDTH}x{TARGET_HEIGHT})")
    print(f"Split Mode         : 78 Train + 20 Val = 98 Folders Total")
    print("-" * 80)

    # 1. Build and verify Speaker Folder Mappings
    # Train: 22 base speakers -> 78 folders (12 spk with 4 folders, 10 spk with 3 folders)
    train_mapping = build_speaker_folder_mapping(TRAIN_BASE_SPEAKERS, TARGET_TRAIN_FOLDERS)

    # Val: 6 base speakers -> 20 folders (s25, s26 with 4 folders, s23, s24, s28, s30 with 3 folders)
    val_mapping = build_speaker_folder_mapping(
        VAL_BASE_SPEAKERS,
        TARGET_VAL_FOLDERS,
        custom_four_folders=["s25", "s26"]
    )

    all_train_folders = sorted([f for dirs in train_mapping.values() for f in dirs], key=natural_sort_key)
    all_val_folders = sorted([f for dirs in val_mapping.values() for f in dirs], key=natural_sort_key)
    all_98_folders = sorted(all_train_folders + all_val_folders, key=natural_sort_key)

    # Verification assertions
    assert len(TRAIN_BASE_SPEAKERS) == 22, f"Expected 22 train base speakers, got {len(TRAIN_BASE_SPEAKERS)}"
    assert len(VAL_BASE_SPEAKERS) == 6, f"Expected 6 val base speakers, got {len(VAL_BASE_SPEAKERS)}"
    assert len(set(TRAIN_BASE_SPEAKERS) & set(VAL_BASE_SPEAKERS)) == 0, "Zero speaker overlap"
    assert len(all_train_folders) == 78, f"Expected 78 train folders, got {len(all_train_folders)}"
    assert len(all_val_folders) == 20, f"Expected 20 val folders, got {len(all_val_folders)}"
    assert len(all_98_folders) == 98, f"Expected 98 total folders, got {len(all_98_folders)}"

    print("Speaker & Folder Allocation Verified:")
    print(f"  Train Folders ({len(all_train_folders)}): 12 spk x 4 ({TRAIN_BASE_SPEAKERS[:12]}) + 10 spk x 3 ({TRAIN_BASE_SPEAKERS[12:]})")
    print(f"  Val Folders   ({len(all_val_folders)}): 2 spk x 4 (['s25', 's26']) + 4 spk x 3 (['s23', 's24', 's28', 's30'])")
    print(f"  Total Folders : {len(all_98_folders)} directories")
    print("-" * 80)

    # Create directories
    EXTRACTED_RAW_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save speaker_split.json
    speaker_split_data = {
        "dataset_name": DATASET_NAME,
        "dataset_mode": "98_78_20_replaced",
        "random_seed": DEFAULT_SEED,
        "num_total_folders": 98,
        "num_train_folders": len(all_train_folders),
        "num_base_train_speakers": len(TRAIN_BASE_SPEAKERS),
        "num_val_folders": len(all_val_folders),
        "num_base_val_speakers": len(VAL_BASE_SPEAKERS),
        "num_test_folders": 0,
        "num_base_test_speakers": 0,
        "base_train_speakers": TRAIN_BASE_SPEAKERS,
        "train_speakers": all_train_folders,
        "train_speaker_mapping": train_mapping,
        "base_val_speakers": VAL_BASE_SPEAKERS,
        "val_speakers": all_val_folders,
        "val_speaker_mapping": val_mapping,
        "base_test_speakers": [],
        "test_speakers": []
    }
    with open(OUTPUT_DATASET_DIR / "speaker_split.json", "w") as f:
        json.dump(speaker_split_data, f, indent=2)
    with open(RESULTS_DIR / "speaker_split.json", "w") as f:
        json.dump(speaker_split_data, f, indent=2)

    # ==========================================================================
    # Step 1: Scan and Extract Video Frames
    # ==========================================================================
    print("\n[STEP 1/3] Scanning Videos and Extracting Sequential JPEG Frames...")
    extraction_tasks = []
    all_speakers = sorted([d.name for d in VIDEO_INPUT_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")], key=natural_sort_key)

    for spk in all_speakers:
        spk_dir = VIDEO_INPUT_DIR / spk
        for d_idx in range(11):
            dig = f"d{d_idx}"
            dig_dir = spk_dir / dig
            if not dig_dir.exists():
                print(f"  [Notice] Directory missing: {spk}/{dig} (skipping gracefully)")
                continue

            mp4s = sorted([f for f in dig_dir.iterdir() if f.name.endswith(".mp4") and not f.name.startswith(".")])
            if not mp4s:
                print(f"  [Notice] No .mp4 file in {spk}/{dig} (skipping gracefully)")
                continue

            # Pick target video (prefer exact match <dig>.mp4)
            target_vid = None
            for v in mp4s:
                if v.name == f"{dig}.mp4":
                    target_vid = v
                    break
            if target_vid is None:
                target_vid = mp4s[0]

            raw_out_dir = EXTRACTED_RAW_FRAMES_DIR / spk / dig
            extraction_tasks.append((spk, dig, str(target_vid), str(raw_out_dir), str(JPEG_QUALITY)))

    print(f"Total video clips scheduled for extraction: {len(extraction_tasks)}")

    extraction_results = []
    num_workers = min(10, os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for res in tqdm(executor.map(extract_single_video_task, extraction_tasks), total=len(extraction_tasks), desc="Extracting Video Frames"):
            extraction_results.append(res)

    df_extract = pd.DataFrame(extraction_results)
    extract_report_path = EXTRACTED_RAW_FRAMES_DIR / "extraction_report.csv"
    df_extract.to_csv(extract_report_path, index=False)
    print(f"Extraction Report saved to: {extract_report_path}")
    total_raw_frames = df_extract["extracted_count"].sum()
    print(f"Total Raw Frames Extracted: {total_raw_frames:,} across {len(df_extract)} clips")

    # ==========================================================================
    # Step 2 & 3: Dlib Landmark Cropping & Augmentation Expansion
    # ==========================================================================
    print("\n[STEP 2 & 3/3] Running Dlib Method D Cropping & Lip-Reading Safe Augmentation...")
    speaker_tasks = []

    # Train speakers
    for spk in TRAIN_BASE_SPEAKERS:
        spk_folders = train_mapping[spk]
        speaker_tasks.append((spk, spk_folders, "train", str(EXTRACTED_RAW_FRAMES_DIR), str(OUTPUT_DATASET_DIR), DEFAULT_SEED))

    # Val speakers
    for spk in VAL_BASE_SPEAKERS:
        spk_folders = val_mapping[spk]
        speaker_tasks.append((spk, spk_folders, "val", str(EXTRACTED_RAW_FRAMES_DIR), str(OUTPUT_DATASET_DIR), DEFAULT_SEED))

    print(f"Total base speakers scheduled: {len(speaker_tasks)} (22 Train + 6 Val)")

    all_manifest_records = []
    all_geom_records = []

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=init_dlib_worker,
        initargs=(str(DLIB_PREDICTOR_PATH),)
    ) as executor:
        for res in tqdm(executor.map(process_single_speaker_task, speaker_tasks), total=len(speaker_tasks), desc="Processing Speakers"):
            all_manifest_records.extend(res["manifest_records"])
            all_geom_records.extend(res["geom_records"])

    # ==========================================================================
    # Step 4: Manifests, Geometry, and Summary Artifacts
    # ==========================================================================
    print("\n[STEP 4/4] Writing Manifests, Geometry Logs, and Verification Artifacts...")

    df_manifest = pd.DataFrame(all_manifest_records)
    manifest_out1 = OUTPUT_DATASET_DIR / "augmentation_manifest.csv"
    manifest_out2 = RESULTS_DIR / "augmentation_manifest.csv"
    df_manifest.to_csv(manifest_out1, index=False)
    df_manifest.to_csv(manifest_out2, index=False)
    print(f"Augmentation Manifest saved: {manifest_out1}")

    df_geom = pd.DataFrame(all_geom_records)
    geom_out = RESULTS_DIR / "dlib_augmented_cropping_geometry.csv"
    df_geom.to_csv(geom_out, index=False)
    print(f"Geometry Logs saved: {geom_out}")

    # Summary Statistics
    train_manifest = df_manifest[df_manifest["split"] == "train"]
    val_manifest = df_manifest[df_manifest["split"] == "val"]

    summary_data = {
        "dataset_name": DATASET_NAME,
        "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_speakers_generated": len(df_manifest["folder_name"].unique()),
        "target_total_speakers": TOTAL_TARGET_FOLDERS,
        "train_folders": len(train_manifest["folder_name"].unique()),
        "target_train_folders": TARGET_TRAIN_FOLDERS,
        "val_folders": len(val_manifest["folder_name"].unique()),
        "target_val_folders": TARGET_VAL_FOLDERS,
        "total_frames_generated": len(df_manifest),
        "train_frames": len(train_manifest),
        "val_frames": len(val_manifest),
        "total_original_frames": len(df_manifest[~df_manifest["is_augmented"]]),
        "total_augmented_frames": len(df_manifest[df_manifest["is_augmented"]]),
        "dlib_landmark_success_rate": float((df_geom["status"] == "SUCCESS").mean()) if len(df_geom) else 0.0,
        "target_resolution": f"{TARGET_WIDTH}x{TARGET_HEIGHT}",
        "zero_leakage_verified": len(set(train_manifest["base_speaker"].unique()) & set(val_manifest["base_speaker"].unique())) == 0
    }

    summary_out1 = OUTPUT_DATASET_DIR / "augmentation_summary.json"
    summary_out2 = RESULTS_DIR / "augmentation_summary.json"
    with open(summary_out1, "w") as f:
        json.dump(summary_data, f, indent=2)
    with open(summary_out2, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Summary JSON saved: {summary_out1}")

    # Generate Preview Grid
    print("Rendering sample preview grid...")
    render_preview_grid(OUTPUT_DATASET_DIR, RESULTS_DIR / "augmented_samples_preview.png")

    print("\n" + "=" * 80)
    print("      DATASET GENERATION PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Total Folders Generated : {summary_data['total_speakers_generated']} / {TOTAL_TARGET_FOLDERS}")
    print(f"Train Folders           : {summary_data['train_folders']} (Base: {len(TRAIN_BASE_SPEAKERS)})")
    print(f"Val Folders             : {summary_data['val_folders']} (Base: {len(VAL_BASE_SPEAKERS)})")
    print(f"Total Frames            : {summary_data['total_frames_generated']:,}")
    print(f"Dlib Success Rate       : {summary_data['dlib_landmark_success_rate']*100:.2f}%")
    print(f"Zero Leakage Verified   : {summary_data['zero_leakage_verified']}")
    print("=" * 80)


def render_preview_grid(dataset_dir: Path, output_img_path: Path):
    """
    Renders a 4x6 grid showing original crops vs aug1, aug2, aug3 variants.
    """
    sample_speakers = ["s1", "s5", "s12", "s25"]
    sample_digits = ["d0", "d1", "d4", "d7", "d10"]

    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    fig.patch.set_facecolor("#181824")

    for row_idx, spk in enumerate(sample_speakers):
        for col_idx, dig in enumerate(sample_digits):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor("#12121c")

            # Try to find a frame
            img_path = dataset_dir / spk / dig / "frame_0015.jpg"
            if not img_path.exists():
                # Fallback to any frame
                frames = list((dataset_dir / spk / dig).glob("*.jpg")) if (dataset_dir / spk / dig).exists() else []
                img_path = frames[0] if frames else None

            if img_path and img_path.exists():
                bgr = cv2.imread(str(img_path))
                if bgr is not None:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    ax.imshow(rgb)
                    ax.set_title(f"{spk} - {dig}", color="white", fontsize=11, fontweight="bold")
            ax.axis("off")

    plt.suptitle("Tai Phake VSR - Dlib Method D 96x96 Preview Grid (Replaced Videos)", color="white", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    output_img_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_img_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"Preview image saved: {output_img_path}")


if __name__ == "__main__":
    main()
