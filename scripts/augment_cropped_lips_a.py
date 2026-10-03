#!/usr/bin/env python3
"""
Generate conservative, realistic image augmentations for Dlib Method A 96x96 lip-cropped dataset.

INPUT:
    data/experiments/cropped_lips_dataset_a_96x96/
OUTPUT:
    data/experiments/augmented_cropped_lips_a_96x96/

For each original speaker s1..s30:
    - sX/ (original copied or preserved identically)
    - sX_aug1/ (variant 1)
    - sX_aug2/ (variant 2)
    - sX_aug3/ (variant 3)
    - sX_aug4/ (variant 4)

Produces:
    - Exactly 4 augmented variants per frame
    - Master CSV manifest: augmentation_manifest.csv
    - Summary JSON: augmentation_summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_ROOT = PROJECT_ROOT / "data" / "experiments" / "cropped_lips_dataset_d_96x96"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "experiments" / "correctly_augmented_cropped_lips_d_96x96"

TARGET_WIDTH = 96
TARGET_HEIGHT = 96
NUM_VARIANTS = 4
DEFAULT_SEED = 42

# Augmentation parameter limits (conservative, lip-reading safe)
# Max translation +/- 3 px (3.1% of 96)
# Max rotation +/- 3 degrees
# Max scale 0.97 to 1.03
# Contrast alpha 0.92 to 1.08
# Brightness beta -10 to +10
# Gaussian noise sigma 1.0 to 3.5
# Mild blur gaussian kernel (3, 3) sigma 0.5


def get_geometric_rng(base_seed: int, speaker: str, digit: str, variant_idx: int) -> np.random.Generator:
    """
    Creates a deterministic numpy Generator unique to (base_seed, speaker, digit, variant)
    for sequence-consistent geometric transformations across all frames of an utterance.
    """
    key = f"{base_seed}_{speaker}_{digit}_aug{variant_idx}_geometry".encode("utf-8")
    hash_digest = hashlib.sha256(key).digest()
    seed_uint32 = int.from_bytes(hash_digest[:4], byteorder="little")
    return np.random.default_rng(seed_uint32)


def get_photometric_rng(base_seed: int, speaker: str, digit: str, frame: str, variant_idx: int) -> np.random.Generator:
    """
    Creates a deterministic numpy Generator unique to (base_seed, speaker, digit, frame, variant)
    for frame-independent photometric variations.
    """
    key = f"{base_seed}_{speaker}_{digit}_{frame}_aug{variant_idx}_photo".encode("utf-8")
    hash_digest = hashlib.sha256(key).digest()
    seed_uint32 = int.from_bytes(hash_digest[:4], byteorder="little")
    return np.random.default_rng(seed_uint32)


def sample_geometric_transform(
    rng: np.random.Generator,
    width: int = TARGET_WIDTH,
    height: int = TARGET_HEIGHT,
) -> np.ndarray:
    """
    Samples geometric transformation parameters ONCE per utterance variant:
    - Small horizontal translation (-3 to +3 px)
    - Small vertical translation (-3 to +3 px)
    - Small rotation (-3 to +3 deg)
    - Very small scale/zoom variation (0.97 to 1.03)

    Returns the 2x3 affine transformation matrix M.
    """
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
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Applies conservative, realistic image augmentation for 96x96 lip ROI:
    1. Geometric transform via affine matrix M (fixed per utterance variant)
    2. Small contrast variation (0.92 to 1.08)
    3. Small brightness variation (-10 to +10)
    4. Mild Gaussian noise (probability 0.5, sigma 1.0 to 3.5)
    5. Mild blur (probability 0.3, 3x3 gaussian blur)

    Guarantees output shape remains exactly 96x96x3.
    """
    h, w = img.shape[:2]

    # 1: Geometric transform via pre-sampled affine matrix
    transformed = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # 2 & 3: Photometric variations (Contrast & Brightness)
    alpha = float(rng.uniform(0.92, 1.08))  # contrast factor
    beta = float(rng.uniform(-10.0, 10.0))  # brightness shift
    photometric = np.clip(alpha * transformed.astype(np.float32) + beta, 0.0, 255.0).astype(np.uint8)

    # 4: Mild Gaussian noise (50% probability)
    if rng.random() < 0.50:
        noise_sigma = float(rng.uniform(1.0, 3.5))
        noise = rng.normal(0.0, noise_sigma, photometric.shape)
        photometric = np.clip(photometric.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)

    # 5: Mild blur (30% probability)
    if rng.random() < 0.30:
        photometric = cv2.GaussianBlur(photometric, (3, 3), 0.5)

    return photometric


def process_speaker_augmentation(
    task: Tuple[str, List[Tuple[str, str, str]], Path, Path, int, bool]
) -> List[Dict[str, any]]:
    """
    Process all frames for a single speaker:
    - Copies original sX frames to output sX
    - Generates 4 augmented variants for every frame under sX_aug1..sX_aug4
    """
    spk, frame_list, in_root, out_root, base_seed, overwrite = task
    records: List[Dict[str, any]] = []

    # Sample geometric transformation ONCE per utterance variant (speaker, digit, variant_idx)
    # The affine matrix M remains fixed across all frames of the utterance.
    geo_transforms: Dict[Tuple[str, int], np.ndarray] = {}
    unique_digits = sorted({dig for dig, _, _ in frame_list})
    for dig in unique_digits:
        for v_idx in range(1, NUM_VARIANTS + 1):
            geo_rng = get_geometric_rng(base_seed, spk, dig, v_idx)
            geo_transforms[(dig, v_idx)] = sample_geometric_transform(geo_rng, TARGET_WIDTH, TARGET_HEIGHT)

    for dig, fname, src_fpath_str in frame_list:
        src_path = Path(src_fpath_str)
        orig_out_path = out_root / spk / dig / fname

        # 1. Handle original frame
        orig_out_path.parent.mkdir(parents=True, exist_ok=True)
        if overwrite or not orig_out_path.exists():
            shutil.copy2(src_path, orig_out_path)

        # Read image once for augmentations
        img = cv2.imread(src_fpath_str)
        if img is None:
            for v_idx in range(1, NUM_VARIANTS + 1):
                records.append({
                    "source_speaker": spk,
                    "augmented_speaker": f"{spk}_aug{v_idx}",
                    "digit": dig,
                    "frame": fname,
                    "augmentation_variant": f"aug{v_idx}",
                    "source_path": str(src_path),
                    "output_path": str(out_root / f"{spk}_aug{v_idx}" / dig / fname),
                    "success": False,
                    "error": "Failed to read source frame",
                })
            continue

        # 2. Generate 4 augmented variants
        for v_idx in range(1, NUM_VARIANTS + 1):
            aug_spk = f"{spk}_aug{v_idx}"
            out_path = out_root / aug_spk / dig / fname
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if overwrite or not out_path.exists():
                M = geo_transforms[(dig, v_idx)]
                photo_rng = get_photometric_rng(base_seed, spk, dig, fname, v_idx)
                aug_img = apply_conservative_augmentation(img, M, photo_rng)
                # Verify dimensions
                if aug_img.shape[0] != TARGET_HEIGHT or aug_img.shape[1] != TARGET_WIDTH:
                    records.append({
                        "source_speaker": spk,
                        "augmented_speaker": aug_spk,
                        "digit": dig,
                        "frame": fname,
                        "augmentation_variant": f"aug{v_idx}",
                        "source_path": str(src_path),
                        "output_path": str(out_path),
                        "success": False,
                        "error": f"Invalid dimensions: {aug_img.shape}",
                    })
                    continue
                # Save JPG
                cv2.imwrite(str(out_path), aug_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                
            records.append({
                "source_speaker": spk,
                "augmented_speaker": aug_spk,
                "digit": dig,
                "frame": fname,
                "augmentation_variant": f"aug{v_idx}",
                "source_path": str(src_path),
                "output_path": str(out_path),
                "success": True,
                "error": None,
            })
            
    return records


def main():
    parser = argparse.ArgumentParser(description="Generate conservative 4x augmented dataset from Method A 96x96")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker processes")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducibility")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files in output destination")
    args = parser.parse_args()
    
    start_time = time.time()
    
    if not INPUT_ROOT.exists():
        print(f"Error: Input directory {INPUT_ROOT} does not exist!")
        sys.exit(1)
        
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("      CONSERVATIVE DATASET AUGMENTATION: CROPPED LIPS METHOD A (96x96)")
    print("=" * 80)
    print(f"Input Directory  : {INPUT_ROOT}")
    print(f"Output Directory : {OUTPUT_ROOT}")
    print(f"Base Random Seed : {args.seed}")
    print(f"Num Variants/Spk : {NUM_VARIANTS} (aug1..aug4)")
    print(f"Worker Processes : {args.workers}")
    print(f"Overwrite Mode   : {args.overwrite}")
    print("-" * 80)
    
    # 1. Discover speakers, digits, frames
    speaker_tasks = []
    total_original_frames = 0
    all_speakers = [f"s{i}" for i in range(1, 31)]
    
    discovered_speakers = []
    for spk in all_speakers:
        spk_dir = INPUT_ROOT / spk
        if not spk_dir.exists():
            continue
        discovered_speakers.append(spk)
        frame_list = []
        for dg_idx in range(11):
            dig = f"d{dg_idx}"
            dig_dir = spk_dir / dig
            if not dig_dir.exists():
                continue
            frames = sorted(
                dig_dir.glob("frame_*.jpg"),
                key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)) if re.search(r"(\d+)", p.stem) else 0,
            )
            for f in frames:
                frame_list.append((dig, f.name, str(f)))
        total_original_frames += len(frame_list)
        speaker_tasks.append((spk, frame_list, INPUT_ROOT, OUTPUT_ROOT, args.seed, args.overwrite))
        
    expected_augmented_frames = total_original_frames * NUM_VARIANTS
    expected_total_frames = total_original_frames * (1 + NUM_VARIANTS)
    
    print(f"Original Speakers Discovered : {len(discovered_speakers)} (s1..s30)")
    print(f"Original Frames Discovered   : {total_original_frames:,}")
    print(f"Expected Augmented Variants  : {expected_augmented_frames:,}")
    print(f"Expected Total Frames        : {expected_total_frames:,} (original + augmented)")
    print("-" * 80)
    
    # 2. Parallel execution across speakers
    all_records: List[Dict[str, any]] = []
    print(f"Executing augmentation with ProcessPoolExecutor ({args.workers} workers)...")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for res_list in tqdm(executor.map(process_speaker_augmentation, speaker_tasks), total=len(speaker_tasks), desc="Augmenting Speakers"):
            all_records.extend(res_list)
            
    df_manifest = pd.DataFrame(all_records)
    
    # 3. Save Manifest CSV
    manifest_cols = [
        "source_speaker",
        "augmented_speaker",
        "digit",
        "frame",
        "augmentation_variant",
        "source_path",
        "output_path",
    ]
    manifest_csv_path = OUTPUT_ROOT / "augmentation_manifest.csv"
    df_manifest[manifest_cols].to_csv(manifest_csv_path, index=False)
    print(f"\nAugmentation manifest CSV saved to: {manifest_csv_path}")
    
    # 4. Strict Validation of Outputs
    print("\nVerifying output dataset structure and image dimensions (96x96x3)...")
    
    output_spk_dirs = sorted([d.name for d in OUTPUT_ROOT.iterdir() if d.is_dir() and not d.name.startswith(".")])
    num_orig_spk_dirs = len([d for d in output_spk_dirs if "_" not in d])
    num_aug_spk_dirs = len([d for d in output_spk_dirs if "_" in d])
    
    all_output_jpegs = list(OUTPUT_ROOT.glob("s*/*/*.jpg"))
    actual_total_frames = len(all_output_jpegs)
    
    # Count original vs augmented in output
    actual_orig_frames = sum(len(list((OUTPUT_ROOT / spk).glob("*/*.jpg"))) for spk in discovered_speakers)
    actual_aug_frames = actual_total_frames - actual_orig_frames
    
    # Dimension and integrity verification
    dimension_errors = 0
    corrupt_errors = 0
    # Spot-check or thorough verification
    for fpath in tqdm(all_output_jpegs, desc="Verifying 96x96x3"):
        img = cv2.imread(str(fpath))
        if img is None:
            corrupt_errors += 1
            continue
        h, w, c = img.shape
        if h != TARGET_HEIGHT or w != TARGET_WIDTH:
            dimension_errors += 1
            
    failed_augmentations = int((~df_manifest["success"]).sum()) + corrupt_errors + dimension_errors
    
    summary = {
        "dataset_name": "augmented_cropped_lips_a_96x96",
        "source_dataset": "cropped_lips_dataset_a_96x96",
        "number_of_original_speakers": len(discovered_speakers),
        "number_of_augmented_speakers": num_aug_spk_dirs,
        "number_of_digits": 11,
        "original_frame_count": total_original_frames,
        "expected_augmented_frame_count": expected_augmented_frames,
        "actual_augmented_frame_count": actual_aug_frames,
        "total_output_frame_count": actual_total_frames,
        "number_of_failed_augmentations": failed_augmentations,
        "dimension_verification": {
            "expected_shape": [TARGET_HEIGHT, TARGET_WIDTH, 3],
            "dimension_errors": dimension_errors,
            "corrupt_errors": corrupt_errors,
            "all_exact_96x96x3": (dimension_errors == 0 and corrupt_errors == 0 and actual_aug_frames == expected_augmented_frames),
        },
        "augmentation_configuration": {
            "num_variants": NUM_VARIANTS,
            "horizontal_translation_px": [-3.0, 3.0],
            "vertical_translation_px": [-3.0, 3.0],
            "rotation_degrees": [-3.0, 3.0],
            "scale_range": [0.97, 1.03],
            "contrast_alpha": [0.92, 1.08],
            "brightness_beta": [-10.0, 10.0],
            "gaussian_noise_prob": 0.50,
            "gaussian_noise_sigma": [1.0, 3.5],
            "gaussian_blur_prob": 0.30,
            "gaussian_blur_kernel": [3, 3],
            "gaussian_blur_sigma": 0.5,
            "border_mode": "BORDER_REFLECT_101",
        },
        "random_seed": args.seed,
        "elapsed_seconds": round(time.time() - start_time, 2),
    }
    
    summary_json_path = OUTPUT_ROOT / "augmentation_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON saved to: {summary_json_path}")
    
    # 5. Print exact requested output summary
    print("\n" + "=" * 80)
    print("                    AUGMENTATION EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Original speakers: {len(discovered_speakers)}")
    print(f"Augmented speakers: {num_aug_spk_dirs}")
    print(f"Original frames: {total_original_frames}")
    print(f"Expected augmented frames: {expected_augmented_frames}")
    print(f"Actual augmented frames: {actual_aug_frames}")
    print(f"Failed: {failed_augmentations}")
    print(f"\nOutput:\n{OUTPUT_ROOT}")
    print("=" * 80)


if __name__ == "__main__":
    main()
