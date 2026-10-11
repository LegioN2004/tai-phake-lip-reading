#!/usr/bin/env python3
"""
scripts/reaugment_dlib_dataset.py

Regenerates or selectively updates the 100-folder augmented dataset:
  data/experiments/dlib_augmented_cropped_100_80_20_96x96/

Folder Allocation Logic (Zero Leakage 80/20):
  - 24 Train base speakers (s1..s24) -> 80 train folders:
      8 base speakers with 4 folders (s1..s8 -> orig + aug1 + aug2 + aug3)
      16 base speakers with 3 folders (s9..s24 -> orig + aug1 + aug2)
  - 6 Val base speakers (s25..s30) -> 20 val folders:
      2 base speakers with 4 folders (s25, s26 -> orig + aug1 + aug2 + aug3)
      4 base speakers with 3 folders (s27, s28, s29, s30 -> orig + aug1 + aug2)

Special Enhancements:
  - If speaker s27 is being processed (or any speaker in LIGHTING_CORRECTION_SPEAKERS),
    applies CLAHE on the L channel of LAB + Gamma correction (gamma=0.90) to fix
    overhead lighting washout and restore bottom-lip to chin boundary definition.

Usage:
  # Regenerate only s27 (deletes s27, s27_aug1, s27_aug2, re-creates with CLAHE+Gamma, updates symlinks)
  python scripts/reaugment_dlib_dataset.py --speaker s27

  # Regenerate all 100 folders
  python scripts/reaugment_dlib_dataset.py --speaker all
"""

import os
import sys
import re
import json
import time
import shutil
import hashlib
import random
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# --- Reproducibility & Dimensions ---
DEFAULT_SEED = 42
random.seed(DEFAULT_SEED)
np.random.seed(DEFAULT_SEED)

TARGET_WIDTH = 96
TARGET_HEIGHT = 96
PADDING_X = 12
PADDING_Y = 16
BORDER_VALUE = 0
JPEG_QUALITY = 95

# --- Lighting Enhancement Configuration (CLAHE on LAB L-channel + Gamma) ---
ENABLE_LIGHTING_ENHANCEMENT = True       # Master switch: set to True to enable enhancement
ENHANCEMENT_TARGET_SPEAKERS = ["s27"]     # Speakers to enhance when ENABLE_LIGHTING_ENHANCEMENT is True
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
CORRECTION_GAMMA = 0.90

def natural_sort_key(s: str) -> int:
    m = re.search(r"\d+", s)
    return int(m.group(0)) if m else 0

ALL_BASE_SPEAKERS = sorted([f"s{i}" for i in range(1, 31)], key=natural_sort_key)
TRAIN_BASE_SPEAKERS = sorted([f"s{i}" for i in range(1, 25)], key=natural_sort_key)
VAL_BASE_SPEAKERS = sorted([f"s{i}" for i in range(25, 31)], key=natural_sort_key)

TARGET_TRAIN_FOLDERS = 80
TARGET_VAL_FOLDERS = 20
TOTAL_TARGET_FOLDERS = 100

def build_speaker_folder_mapping(base_speakers: List[str], target_folder_count: int) -> Dict[str, List[str]]:
    n_base = len(base_speakers)
    rem = target_folder_count % n_base
    quo = target_folder_count // n_base
    mapping = {}
    for i, spk in enumerate(base_speakers):
        folders_for_spk = quo + (1 if i < rem else 0)
        dirs = [spk]
        for a_idx in range(1, folders_for_spk):
            dirs.append(f"{spk}_aug{a_idx}")
        mapping[spk] = dirs
    return mapping

TRAIN_MAPPING = build_speaker_folder_mapping(TRAIN_BASE_SPEAKERS, TARGET_TRAIN_FOLDERS)
VAL_MAPPING = build_speaker_folder_mapping(VAL_BASE_SPEAKERS, TARGET_VAL_FOLDERS)

FULL_FOLDER_MAPPING = {}
FULL_FOLDER_MAPPING.update(TRAIN_MAPPING)
FULL_FOLDER_MAPPING.update(VAL_MAPPING)

# --- Lighting Correction Helpers ---
def apply_clahe_lab(bgr_img: np.ndarray, clip_limit: float = CLAHE_CLIP_LIMIT, tile_grid: tuple = CLAHE_TILE_GRID) -> np.ndarray:
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)

def adjust_gamma(bgr_img: np.ndarray, gamma: float = CORRECTION_GAMMA) -> np.ndarray:
    inv = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(bgr_img, table)

def enhance_tight_mouth_crop(
    tight_crop: np.ndarray,
    gamma: float = CORRECTION_GAMMA,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid: tuple = CLAHE_TILE_GRID
) -> np.ndarray:
    """Stage 4: Apply Gamma correction and CLAHE directly on tight real-mouth pixels before padding."""
    gamma_crop = adjust_gamma(tight_crop, gamma=gamma)
    return apply_clahe_lab(gamma_crop, clip_limit=clip_limit, tile_grid=tile_grid)

def letterbox_pad_to_96x96(img: np.ndarray, target_size: int = TARGET_WIDTH) -> np.ndarray:
    """Stage 5: Spatial aspect-preserving resize and centered letterbox padding onto 96x96 canvas."""
    h_src, w_src = img.shape[:2]
    if h_src == 0 or w_src == 0:
        return np.full((target_size, target_size, 3), BORDER_VALUE, dtype=np.uint8)
    scale = target_size / max(w_src, h_src)
    new_w = min(target_size, max(1, int(round(w_src * scale))))
    new_h = min(target_size, max(1, int(round(h_src * scale))))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_size, target_size, 3), BORDER_VALUE, dtype=np.uint8)
    offset_x = (target_size - new_w) // 2
    offset_y = (target_size - new_h) // 2
    canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized
    return canvas

def extract_tight_crop(
    img_bgr: np.ndarray,
    center_x: float,
    center_y: float,
    raw_width: float,
    raw_height: float,
    pad_x: int = PADDING_X,
    pad_y: int = PADDING_Y
) -> np.ndarray:
    """Stage 3: Extract tight bounding crop around mouth landmarks without artificial border bars."""
    h_img, w_img = img_bgr.shape[:2]
    x1 = max(0, int(round(center_x - raw_width / 2.0 - pad_x)))
    y1 = max(0, int(round(center_y - raw_height / 2.0 - pad_y)))
    x2 = min(w_img, int(round(center_x + raw_width / 2.0 + pad_x)))
    y2 = min(h_img, int(round(center_y + raw_height / 2.0 + pad_y)))
    return img_bgr[y1:y2, x1:x2].copy()

# --- Geometric & Photometric Augmentations ---
def get_geometric_rng(base_seed: int, speaker: str, digit: str, variant_idx: int) -> np.random.Generator:
    key = f"{base_seed}_{speaker}_{digit}_aug{variant_idx}_geom".encode("utf-8")
    seed_uint32 = int.from_bytes(hashlib.sha256(key).digest()[:4], byteorder="little")
    return np.random.default_rng(seed_uint32)

def get_photometric_rng(base_seed: int, speaker: str, digit: str, frame: str, variant_idx: int) -> np.random.Generator:
    key = f"{base_seed}_{speaker}_{digit}_{frame}_aug{variant_idx}_photo".encode("utf-8")
    seed_uint32 = int.from_bytes(hashlib.sha256(key).digest()[:4], byteorder="little")
    return np.random.default_rng(seed_uint32)

def sample_geometric_transform(rng: np.random.Generator, width: int = TARGET_WIDTH, height: int = TARGET_HEIGHT) -> np.ndarray:
    tx = float(rng.uniform(-3.0, 3.0))
    ty = float(rng.uniform(-3.0, 3.0))
    angle = float(rng.uniform(-3.0, 3.0))
    scale = float(rng.uniform(0.97, 1.03))
    center = (width / 2.0, height / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    return M

def apply_conservative_augmentation(img: np.ndarray, M: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    transformed = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    alpha = float(rng.uniform(0.92, 1.08))
    beta = float(rng.uniform(-10.0, 10.0))
    photo = np.clip(alpha * transformed.astype(np.float32) + beta, 0.0, 255.0).astype(np.uint8)
    if rng.random() < 0.50:
        sigma = float(rng.uniform(1.0, 3.5))
        noise = rng.normal(0.0, sigma, photo.shape)
        photo = np.clip(photo.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)
    if rng.random() < 0.30:
        photo = cv2.GaussianBlur(photo, (3, 3), 0.5)
    return photo

# --- Source Frame Collection ---
def collect_base_speaker_frames(
    base_speaker: str,
    project_root: Path,
    is_lighting_fix: bool = False
) -> List[Tuple[str, str, Path]]:
    # 1. If lighting fix is requested and dedicated gamma_enhanced folder exists, use it!
    if is_lighting_fix:
        gamma_enhanced_dir = project_root / "data" / "experiments" / f"gamma_enhanced_{base_speaker}"
        if gamma_enhanced_dir.exists():
            frames = []
            for dig_dir in sorted(gamma_enhanced_dir.iterdir()):
                if dig_dir.is_dir() and not dig_dir.name.startswith("."):
                    dig = dig_dir.name
                    for fpath in sorted(dig_dir.glob("*.jpg")):
                        frames.append((dig, fpath.name, fpath))
            if frames:
                return frames

    # 2. Method D pre-cropped directory (used when no lighting fix or already cropped)
    if not is_lighting_fix:
        pre_cropped_d = project_root / "data" / "experiments" / "cropped_lips_dataset_d_96x96" / base_speaker
        frames = []
        if pre_cropped_d.exists():
            for dig_dir in sorted(pre_cropped_d.iterdir()):
                if dig_dir.is_dir() and not dig_dir.name.startswith("."):
                    dig = dig_dir.name
                    for fpath in sorted(dig_dir.glob("*.jpg")):
                        frames.append((dig, fpath.name, fpath))
            if frames:
                return frames

    # 3. Fallback to raw digit frames
    raw_digit = project_root / "data" / "digit" / base_speaker
    if not raw_digit.exists():
        raw_digit = project_root / "data" / "experiments" / "digit" / base_speaker
    if raw_digit.exists():
        frames = []
        for dig_dir in sorted(raw_digit.iterdir()):
            if dig_dir.is_dir() and not dig_dir.name.startswith("."):
                dig = dig_dir.name
                for fpath in sorted(dig_dir.glob("*.jpg")):
                    frames.append((dig, fpath.name, fpath))
        return frames
    return []

# --- Core Speaker Processing ---
def process_single_speaker(
    base_speaker: str,
    target_folders: List[str],
    split_name: str,
    output_root_dir: Path,
    project_root: Path,
    base_seed: int = DEFAULT_SEED,
    apply_lighting_fix: bool = False,
    geometry_dict: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    manifest_records = []
    frame_list = collect_base_speaker_frames(base_speaker, project_root, is_lighting_fix=apply_lighting_fix)
    if not frame_list:
        print(f"[WARNING] No frames found for base speaker: {base_speaker}")
        return manifest_records

    unique_digits = sorted({dig for dig, _, _ in frame_list})
    aug_variants = [f for f in target_folders if f != base_speaker]

    # Sample geometric transforms once per (digit, variant)
    geo_transforms: Dict[Tuple[str, str], np.ndarray] = {}
    for dig in unique_digits:
        for v_folder in aug_variants:
            m = re.search(r"aug(\d+)", v_folder)
            v_idx = int(m.group(1)) if m else 1
            geo_rng = get_geometric_rng(base_seed, base_speaker, dig, v_idx)
            geo_transforms[(dig, v_folder)] = sample_geometric_transform(geo_rng, TARGET_WIDTH, TARGET_HEIGHT)

    for dig, fname, src_path in frame_list:
        img = cv2.imread(str(src_path))
        if img is None:
            continue

        # If source is raw frame (not 96x96), crop and letterbox strictly following Stage 3 -> 4 -> 5
        if img.shape[0] != TARGET_HEIGHT or img.shape[1] != TARGET_WIDTH:
            key = f"{base_speaker}/{dig}/{fname}"
            if geometry_dict and key in geometry_dict:
                geom = geometry_dict[key]
                # Stage 3: Tight crop
                tight = extract_tight_crop(img, geom["center_x"], geom["center_y"], geom["raw_width"], geom["raw_height"])
            else:
                h_i, w_i = img.shape[:2]
                tight = extract_tight_crop(img, w_i / 2.0, h_i / 2.0, 120.0, 50.0)

            # Stage 4: Enhancement on tight real mouth pixels ONLY
            if apply_lighting_fix:
                tight = enhance_tight_mouth_crop(tight, gamma=CORRECTION_GAMMA, clip_limit=CLAHE_CLIP_LIMIT)

            # Stage 5: Final Letterbox pad to 96x96
            img = letterbox_pad_to_96x96(tight, target_size=TARGET_WIDTH)
        else:
            # Already a 96x96 frame (e.g. from gamma_enhanced_<spk> or pre-cropped dataset)
            pass

        # 1. Base unaugmented frame
        dst_orig_path = output_root_dir / base_speaker / dig / fname
        dst_orig_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst_orig_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])

        manifest_records.append({
            "split": split_name,
            "base_speaker": base_speaker,
            "folder_name": base_speaker,
            "is_augmented": False,
            "variant": "original",
            "digit": dig,
            "frame": fname,
            "output_path": str(dst_orig_path),
            "width": TARGET_WIDTH,
            "height": TARGET_HEIGHT,
            "lighting_corrected": apply_lighting_fix
        })

        # 2. Augmented variants
        for v_folder in aug_variants:
            m = re.search(r"aug(\d+)", v_folder)
            v_idx = int(m.group(1)) if m else 1
            dst_aug_path = output_root_dir / v_folder / dig / fname
            dst_aug_path.parent.mkdir(parents=True, exist_ok=True)

            M = geo_transforms[(dig, v_folder)]
            photo_rng = get_photometric_rng(base_seed, base_speaker, dig, fname, v_idx)
            aug_img = apply_conservative_augmentation(img, M, photo_rng)
            cv2.imwrite(str(dst_aug_path), aug_img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])

            manifest_records.append({
                "split": split_name,
                "base_speaker": base_speaker,
                "folder_name": v_folder,
                "is_augmented": True,
                "variant": f"aug{v_idx}",
                "digit": dig,
                "frame": fname,
                "output_path": str(dst_aug_path),
                "width": TARGET_WIDTH,
                "height": TARGET_HEIGHT,
                "lighting_corrected": apply_lighting_fix
            })

    return manifest_records

def main():
    parser = argparse.ArgumentParser(description="Regenerate or update dlib augmented cropped 100-folder dataset.")
    parser.add_argument("--speaker", type=str, default="s27", help="Base speaker to re-generate (e.g. 's27' or 'all'). Default: 's27'.")
    parser.add_argument("--enable-lighting-enhancement", action="store_true", default=ENABLE_LIGHTING_ENHANCEMENT,
                        help="Explicitly enable CLAHE + Gamma enhancement on target speakers. Default: False.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    dataset_name = "dlib_augmented_cropped_100_80_20_96x96"
    dataset_dir = project_root / "data" / "experiments" / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    results_dir = project_root / "results" / "dlib_augmented_cropped"
    results_dir.mkdir(parents=True, exist_ok=True)

    target_spk = args.speaker.strip().lower()
    enable_enhancement = args.enable_lighting_enhancement

    if target_spk not in ["all", "everything"] and target_spk not in ALL_BASE_SPEAKERS:
        print(f"[ERROR] Invalid speaker '{args.speaker}'. Must be one of {ALL_BASE_SPEAKERS} or 'all'.")
        sys.exit(1)

    speakers_to_process = ALL_BASE_SPEAKERS if target_spk in ["all", "everything"] else [target_spk]

    print("=" * 80)
    print(f"RE-AUGMENTATION DATASET PIPELINE: {dataset_name}")
    print("=" * 80)
    print(f"Target Base Speaker(s)                : {speakers_to_process}")
    print(f"Master Enhancement Switch             : {enable_enhancement}")
    print(f"Enhancement Target Speaker(s)         : {ENHANCEMENT_TARGET_SPEAKERS if enable_enhancement else 'None (Disabled)'}")
    print(f"Target Directory                      : {dataset_dir}")
    print("=" * 80)

    # 1. If targeting specific speaker(s), remove only their folders
    for spk in speakers_to_process:
        folders_for_spk = FULL_FOLDER_MAPPING[spk]
        print(f"Removing old folders for speaker '{spk}': {folders_for_spk}")
        for folder_name in folders_for_spk:
            f_path = dataset_dir / folder_name
            if f_path.exists():
                shutil.rmtree(f_path)
                print(f"  Deleted: {f_path}")

    # Load precomputed geometry records if available
    geom_dict = None
    geom_csv = project_root / "results" / "dlib" / "dlib_roi_geometry_all_frames.csv"
    if geom_csv.exists():
        df_geom = pd.read_csv(geom_csv)
        geom_dict = {
            f"{r['speaker']}/{r['digit']}/{r['frame']}": {
                "center_x": r["center_x"], "center_y": r["center_y"],
                "raw_width": r["raw_width"], "raw_height": r["raw_height"]
            }
            for _, r in df_geom.iterrows()
        }

    # 2. Process requested speaker(s)
    new_records = []
    for spk in tqdm(speakers_to_process, desc="Processing Speaker(s)"):
        split_name = "train" if spk in TRAIN_BASE_SPEAKERS else "val"
        target_dirs = FULL_FOLDER_MAPPING[spk]
        is_fix = enable_enhancement and (
            "all" in ENHANCEMENT_TARGET_SPEAKERS
            or ENHANCEMENT_TARGET_SPEAKERS == "all"
            or spk in ENHANCEMENT_TARGET_SPEAKERS
        )
        if is_fix:
            print(f"\n[APPLYING TIGHT-CROP LIGHTING FIX: CLAHE(L) + Gamma(0.90)] for speaker '{spk}' ({target_dirs})")
        recs = process_single_speaker(
            base_speaker=spk,
            target_folders=target_dirs,
            split_name=split_name,
            output_root_dir=dataset_dir,
            project_root=project_root,
            base_seed=DEFAULT_SEED,
            apply_lighting_fix=is_fix,
            geometry_dict=geom_dict
        )
        new_records.extend(recs)

    # 3. Synchronize manifest
    manifest_file = dataset_dir / "augmentation_manifest.csv"
    if target_spk not in ["all", "everything"] and manifest_file.exists():
        # Load existing manifest, remove records for updated speaker, append new records
        df_old = pd.read_csv(manifest_file)
        df_old = df_old[df_old["base_speaker"] != target_spk]
        df_new = pd.DataFrame(new_records)
        df_merged = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_merged = pd.DataFrame(new_records)

    df_merged.to_csv(manifest_file, index=False)
    df_merged.to_csv(results_dir / "augmentation_manifest.csv", index=False)
    print(f"Updated manifest saved with {len(df_merged):,} records.")

    # 4. Save/Update speaker_split.json & summary
    all_disk_dirs = sorted([d.name for d in dataset_dir.iterdir() if d.is_dir() and not d.name.startswith(".")], key=natural_sort_key)
    train_disk_dirs = [d for d in all_disk_dirs if d.split("_")[0] in TRAIN_BASE_SPEAKERS]
    val_disk_dirs = [d for d in all_disk_dirs if d.split("_")[0] in VAL_BASE_SPEAKERS]

    split_meta = {
        "dataset_name": dataset_name,
        "seed": DEFAULT_SEED,
        "num_train_base_speakers": len(TRAIN_BASE_SPEAKERS),
        "num_val_base_speakers": len(VAL_BASE_SPEAKERS),
        "train_base_speakers": TRAIN_BASE_SPEAKERS,
        "val_base_speakers": VAL_BASE_SPEAKERS,
        "train_folders": train_disk_dirs,
        "val_folders": val_disk_dirs,
        "lighting_correction_speakers": LIGHTING_CORRECTION_SPEAKERS
    }
    with open(dataset_dir / "speaker_split.json", "w") as f:
        json.dump(split_meta, f, indent=2)
    with open(results_dir / "speaker_split.json", "w") as f:
        json.dump(split_meta, f, indent=2)

    # 5. Integrity Verification
    print("=" * 80)
    print("INTEGRITY CHECKS:")
    print("=" * 80)
    print(f"Total Folders on disk: {len(all_disk_dirs)} (Expected: 100)")
    print(f"Train Folders        : {len(train_disk_dirs)} (Expected: 80)")
    print(f"Val Folders          : {len(val_disk_dirs)} (Expected: 20)")
    assert len(all_disk_dirs) == 100, f"Expected 100 folders on disk, got {len(all_disk_dirs)}"
    assert len(train_disk_dirs) == 80, f"Expected 80 train folders, got {len(train_disk_dirs)}"
    assert len(val_disk_dirs) == 20, f"Expected 20 val folders, got {len(val_disk_dirs)}"
    print("[PASS] 100 folders with 80 train / 20 val verified!")

    # 6. Run switch_dataset_80-20.py to update symlinks in data/processed
    switch_script = project_root / "scripts" / "switch_dataset_80-20.py"
    if switch_script.exists():
        print("=" * 80)
        print(f"Running switch_dataset_80-20.py for {dataset_name}...")
        res = subprocess.run([sys.executable, str(switch_script), dataset_name], cwd=str(project_root), capture_output=True, text=True)
        print(res.stdout)
        if res.stderr:
            print("Stderr:", res.stderr)
        assert res.returncode == 0, f"switch_dataset_80-20.py returned code {res.returncode}"
        print("[SUCCESS] Active dataset in data/processed successfully re-linked!")

if __name__ == "__main__":
    main()
