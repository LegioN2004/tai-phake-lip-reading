#!/usr/bin/env python3
"""
scripts/preprocess_gamma_enhanced_speaker.py

Mouth Preprocessing Pipeline strictly following the 6-stage order of operations:
  1. Extract frames: Read input video frame-by-frame from data/digit.
  2. Per-frame dlib landmarks: Detect facial/mouth landmarks per frame (indices 48–67).
  3. Tight dynamic mouth crop: Extract tight bounding crop around mouth landmarks (real mouth pixels, NO padding bars).
  4. Enhancement (Gamma + CLAHE on mouth crop only):
     Apply Gamma correction and LAB L-channel CLAHE directly on the tight crop.
     NEVER run after padding/letterboxing to avoid artificial black-border distortion.
  5. Final Letterbox / Pad to 96x96:
     Proportionally resize and pad the enhanced crop to exactly 96x96 canvas.
  6. Fallback handling:
     If landmark detection fails on a frame, reuse temporal bounding box from adjacent frames or precomputed geometry.

Usage:
  # Process default speaker (configured via TARGET_SPEAKER constant below):
  python scripts/preprocess_gamma_enhanced_speaker.py

  # Process any other speaker via CLI:
  python scripts/preprocess_gamma_enhanced_speaker.py --speaker s12
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import dlib
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==============================================================================
# 1. SPEAKER CONFIGURATION CONSTANT
# Change this variable to process any other speaker (e.g., 's1', 's4', 's27', 's30')
# ==============================================================================
TARGET_SPEAKER = "s27"
# ==============================================================================

# ==============================================================================
# 2. ENHANCEMENT PARAMETERS (Gamma + CLAHE on Tight Real-Pixel Mouth Crop)
# ==============================================================================
GAMMA = 0.90                    # Darkens overexposed regions to recover bottom-lip to chin boundary
CLAHE_CLIP_LIMIT = 2.0          # CLAHE contrast limit on LAB L-channel
CLAHE_TILE_GRID = (8, 8)        # CLAHE local grid size

# ==============================================================================
# 3. GEOMETRY & LETTERBOX PARAMETERS
# ==============================================================================
TARGET_WIDTH = 96
TARGET_HEIGHT = 96
PAD_X = 15                      # Horizontal margin around mouth landmarks in px
PAD_Y = 10                      # Vertical margin around mouth landmarks in px
BORDER_VALUE = 0                # Black neutral background for letterboxing
JPEG_QUALITY = 95


def natural_sort_key(s: str) -> int:
    m = re.search(r"\d+", s)
    return int(m.group(0)) if m else 0


def adjust_gamma(bgr_img: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """Darken bright overexposed regions (gamma < 1.0) to recover boundary contrast."""
    inv = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(bgr_img, table)


def apply_clahe_lab(
    bgr_img: np.ndarray,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid: Tuple[int, int] = CLAHE_TILE_GRID
) -> np.ndarray:
    """
    Applies CLAHE strictly to the L channel of LAB color space.
    Preserves natural color tones while equalizing local intensity variations.
    """
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)


def enhance_tight_mouth_crop(
    tight_crop: np.ndarray,
    gamma: float = GAMMA,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid: Tuple[int, int] = CLAHE_TILE_GRID
) -> np.ndarray:
    """
    STAGE 4: Apply Gamma correction and CLAHE directly on tight real-mouth pixels.
    CRITICAL: Must be run on real pixels before any black-border padding / letterboxing.
    """
    # 1. Darken overexposed lower lip and chin highlight
    gamma_crop = adjust_gamma(tight_crop, gamma=gamma)
    # 2. Local contrast equalization on LAB L-channel
    enhanced_crop = apply_clahe_lab(gamma_crop, clip_limit=clip_limit, tile_grid=tile_grid)
    return enhanced_crop


def letterbox_pad_to_canvas(
    img: np.ndarray,
    target_w: int = TARGET_WIDTH,
    target_h: int = TARGET_HEIGHT,
    border_val: int = BORDER_VALUE
) -> Tuple[np.ndarray, int, int, int, int]:
    """
    STAGE 5: Aspect-preserving resize and centered letterbox padding onto 96x96 canvas.
    This is strictly a spatial pad/resize step, never an enhancement step.
    """
    h_src, w_src = img.shape[:2]
    if h_src == 0 or w_src == 0:
        return np.full((target_h, target_w, 3), border_val, dtype=np.uint8), 0, 0, 0, 0

    scale = min(target_w / w_src, target_h / h_src)
    new_w = min(target_w, max(1, int(round(w_src * scale))))
    new_h = min(target_h, max(1, int(round(h_src * scale))))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((target_h, target_w, 3), border_val, dtype=np.uint8)
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2
    canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized

    pad_h = target_w - new_w
    pad_v = target_h - new_h
    return canvas, new_w, new_h, pad_h, pad_v


class DlibMouthDetector:
    """Wrapper around Dlib 68-point landmark predictor for mouth ROI extraction."""
    def __init__(self, predictor_path: Path):
        self.detector = dlib.get_frontal_face_detector()
        if predictor_path.exists():
            self.predictor = dlib.shape_predictor(str(predictor_path))
        else:
            raise FileNotFoundError(f"Dlib shape predictor file not found at: {predictor_path}")

    def detect_mouth_box(self, img_bgr: np.ndarray) -> Optional[Dict[str, float]]:
        """
        STAGE 2: Detect mouth landmarks (indices 48–67) and compute centroid and bounding box.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)
        if not faces:
            return None

        # Take primary face
        face = faces[0]
        shape = self.predictor(gray, face)
        pts = np.array([[shape.part(i).x, shape.part(i).y] for i in range(48, 68)], dtype=np.float32)
        min_x, min_y = pts.min(axis=0)
        max_x, max_y = pts.max(axis=0)
        raw_w = float(max_x - min_x)
        raw_h = float(max_y - min_y)
        cx = float((min_x + max_x) / 2.0)
        cy = float((min_y + max_y) / 2.0)
        return {
            "center_x": cx,
            "center_y": cy,
            "raw_width": raw_w,
            "raw_height": raw_h,
            "min_x": float(min_x),
            "max_x": float(max_x),
            "min_y": float(min_y),
            "max_y": float(max_y),
        }


def extract_tight_crop(
    img_bgr: np.ndarray,
    center_x: float,
    center_y: float,
    raw_width: float,
    raw_height: float,
    pad_x: int = PAD_X,
    pad_y: int = PAD_Y
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    STAGE 3: Extract tight bounding crop around mouth landmarks without letterbox bars.
    """
    h_img, w_img = img_bgr.shape[:2]
    x1 = max(0, int(round(center_x - raw_width / 2.0 - pad_x)))
    y1 = max(0, int(round(center_y - raw_height / 2.0 - pad_y)))
    x2 = min(w_img, int(round(center_x + raw_width / 2.0 + pad_x)))
    y2 = min(h_img, int(round(center_y + raw_height / 2.0 + pad_y)))

    tight_crop = img_bgr[y1:y2, x1:x2].copy()
    return tight_crop, (x1, y1, x2, y2)


def process_speaker_utterance(
    speaker: str,
    digit: str,
    frame_paths: List[Path],
    output_digit_dir: Path,
    dlib_detector: DlibMouthDetector,
    geometry_dict: Optional[Dict[str, Any]] = None,
    gamma: float = GAMMA,
    clip_limit: float = CLAHE_CLIP_LIMIT,
) -> Tuple[List[Dict[str, Any]], List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """
    Processes all frames for a single utterance following the 6-stage pipeline.
    """
    records = []
    sample_triplets = []

    # Sort frames naturally (frame_0001.jpg, frame_0002.jpg, ...)
    sorted_frames = sorted(frame_paths, key=lambda p: natural_sort_key(p.name))

    # STAGE 2 + 6: Detect landmarks with temporal fallback tracking
    detected_boxes: List[Optional[Dict[str, float]]] = []
    for fp in sorted_frames:
        frame_key = f"{speaker}/{digit}/{fp.name}"
        img = cv2.imread(str(fp))
        if img is None:
            detected_boxes.append(None)
            continue

        box = dlib_detector.detect_mouth_box(img)
        # Check precomputed geometry fallback if real-time detection failed
        if box is None and geometry_dict and frame_key in geometry_dict:
            box = geometry_dict[frame_key]
        detected_boxes.append(box)

    # STAGE 6: Forward/Backward temporal smoothing fallback for any missing frames
    last_valid_box = None
    for i in range(len(detected_boxes)):
        if detected_boxes[i] is not None:
            last_valid_box = detected_boxes[i]
        elif last_valid_box is not None:
            detected_boxes[i] = last_valid_box

    # Backward pass if the first few frames were missing
    if any(b is None for b in detected_boxes):
        first_valid_box = next((b for b in detected_boxes if b is not None), None)
        if first_valid_box is not None:
            for i in range(len(detected_boxes)):
                if detected_boxes[i] is None:
                    detected_boxes[i] = first_valid_box

    # Execute Stages 3 -> 4 -> 5
    for i, fp in enumerate(sorted_frames):
        img_raw = cv2.imread(str(fp))
        if img_raw is None:
            continue

        box = detected_boxes[i]
        h_img, w_img = img_raw.shape[:2]

        if box is None:
            # Complete emergency fallback: center crop
            cx, cy = w_img / 2.0, h_img / 2.0
            rw, rh = 120.0, 50.0
            fallback_used = "EMERGENCY_CENTER_CROP"
        else:
            cx, cy = box["center_x"], box["center_y"]
            rw, rh = box["raw_width"], box["raw_height"]
            fallback_used = "NONE"

        # STAGE 3: Tight dynamic mouth crop (real pixels only)
        tight_crop, (x1, y1, x2, y2) = extract_tight_crop(img_raw, cx, cy, rw, rh, PAD_X, PAD_Y)

        # STAGE 4: Enhancement (Gamma + CLAHE on mouth crop only)
        enhanced_tight_crop = enhance_tight_mouth_crop(
            tight_crop,
            gamma=gamma,
            clip_limit=clip_limit,
            tile_grid=CLAHE_TILE_GRID
        )

        # STAGE 5: Final Letterbox / Pad to 96x96
        canvas_96x96, new_w, new_h, pad_h, pad_v = letterbox_pad_to_canvas(
            enhanced_tight_crop,
            target_w=TARGET_WIDTH,
            target_h=TARGET_HEIGHT,
            border_val=BORDER_VALUE
        )

        # Save result to output directory
        dst_path = output_digit_dir / fp.name
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst_path), canvas_96x96, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])

        record = {
            "speaker": speaker,
            "digit": digit,
            "frame": fp.name,
            "source_path": str(fp),
            "output_path": str(dst_path),
            "raw_w": rw,
            "raw_h": rh,
            "center_x": cx,
            "center_y": cy,
            "tight_crop_box": [x1, y1, x2, y2],
            "scaled_w": new_w,
            "scaled_h": new_h,
            "pad_horizontal": pad_h,
            "pad_vertical": pad_v,
            "gamma": gamma,
            "clahe_clip_limit": clip_limit,
            "fallback_used": fallback_used,
        }
        records.append(record)

        if i % 10 == 0 and len(sample_triplets) < 6:
            sample_triplets.append((tight_crop, enhanced_tight_crop, canvas_96x96))

    return records, sample_triplets


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess mouth crops with strict Crop -> Enhance -> Letterbox pipeline."
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default=TARGET_SPEAKER,
        help=f"Target speaker to process (e.g. 's27', 's12'). Default: '{TARGET_SPEAKER}'."
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=GAMMA,
        help=f"Gamma correction parameter. Default: {GAMMA}."
    )
    parser.add_argument(
        "--clahe-clip",
        type=float,
        default=CLAHE_CLIP_LIMIT,
        help=f"CLAHE clip limit on LAB L-channel. Default: {CLAHE_CLIP_LIMIT}."
    )
    args = parser.parse_args()

    speaker = args.speaker.strip().lower()
    gamma_val = args.gamma
    clahe_clip = args.clahe_clip

    project_root = Path(__file__).resolve().parent.parent
    raw_digit_root = project_root / "data" / "digit" / speaker
    if not raw_digit_root.exists():
        # Fallback path if data/digit is under data/experiments/digit
        raw_digit_root = project_root / "data" / "experiments" / "digit" / speaker

    if not raw_digit_root.exists():
        print(f"[ERROR] Raw frame directory for speaker '{speaker}' not found at: {raw_digit_root}")
        sys.exit(1)

    # Destination directory name: gamma_enhanced_<speaker>
    output_dir_name = f"gamma_enhanced_{speaker}"
    output_dataset_dir = project_root / "data" / "experiments" / output_dir_name
    output_dataset_dir.mkdir(parents=True, exist_ok=True)

    # Symlink at data/gamma_enhanced_<speaker> for direct access convenience
    direct_data_link = project_root / "data" / output_dir_name
    if not direct_data_link.exists():
        try:
            direct_data_link.symlink_to(output_dataset_dir.resolve())
        except Exception:
            pass

    results_dir = project_root / "results" / "dlib_augmented_cropped"
    results_dir.mkdir(parents=True, exist_ok=True)

    predictor_path = project_root / "models" / "dlib" / "shape_predictor_68_face_landmarks.dat"
    geometry_csv = project_root / "results" / "dlib" / "dlib_roi_geometry_all_frames.csv"

    print("=" * 80)
    print("      MOUTH PREPROCESSING PIPELINE: TIGHT CROP -> ENHANCE -> LETTERBOX")
    print("=" * 80)
    print(f"Target Speaker         : {speaker}")
    print(f"Input Directory        : {raw_digit_root}")
    print(f"Output Dataset Folder  : {output_dataset_dir}")
    print(f"Enhancement Settings   : Gamma = {gamma_val}, CLAHE Clip = {clahe_clip}, Grid = {CLAHE_TILE_GRID}")
    print(f"Final Geometry         : {TARGET_WIDTH}x{TARGET_HEIGHT} letterbox pad (Border value = {BORDER_VALUE})")
    print(f"Predictor Model        : {predictor_path}")
    print("=" * 80)

    # Load Dlib detector
    dlib_detector = DlibMouthDetector(predictor_path)

    # Load geometry dictionary fallback if available
    geom_dict = None
    if geometry_csv.exists():
        df_geom = pd.read_csv(geometry_csv)
        geom_dict = {
            f"{r['speaker']}/{r['digit']}/{r['frame']}": {
                "center_x": r["center_x"], "center_y": r["center_y"],
                "raw_width": r["raw_width"], "raw_height": r["raw_height"]
            }
            for _, r in df_geom.iterrows()
        }
        print(f"Loaded {len(geom_dict):,} precomputed fallback geometry records.")

    all_records = []
    collected_samples = []

    # Iterate over all 11 digits (d0 .. d10)
    digit_dirs = sorted([d for d in raw_digit_root.iterdir() if d.is_dir() and not d.name.startswith(".")], key=lambda d: natural_sort_key(d.name))
    print(f"Found {len(digit_dirs)} digit folders to process: {[d.name for d in digit_dirs]}")

    t0 = time.time()
    for dig_dir in tqdm(digit_dirs, desc=f"Processing {speaker}"):
        dig_name = dig_dir.name
        frame_files = list(dig_dir.glob("*.jpg")) + list(dig_dir.glob("*.png"))
        out_dig_dir = output_dataset_dir / dig_name

        recs, samples = process_speaker_utterance(
            speaker=speaker,
            digit=dig_name,
            frame_paths=frame_files,
            output_digit_dir=out_dig_dir,
            dlib_detector=dlib_detector,
            geometry_dict=geom_dict,
            gamma=gamma_val,
            clip_limit=clahe_clip,
        )
        all_records.extend(recs)
        if len(collected_samples) < 5 and samples:
            collected_samples.append((dig_name, samples[0]))

    elapsed = time.time() - t0
    print(f"\nProcessing complete in {elapsed:.2f} seconds! Processed {len(all_records):,} frames.")

    # Save manifest CSV & JSON metadata
    df_manifest = pd.DataFrame(all_records)
    manifest_path = output_dataset_dir / "preprocessing_manifest.csv"
    df_manifest.to_csv(manifest_path, index=False)
    print(f"Manifest saved to: {manifest_path}")

    summary_info = {
        "speaker": speaker,
        "total_frames": len(all_records),
        "digits_processed": [d.name for d in digit_dirs],
        "output_directory": str(output_dataset_dir),
        "gamma": gamma_val,
        "clahe_clip_limit": clahe_clip,
        "clahe_tile_grid": list(CLAHE_TILE_GRID),
        "canvas_size": [TARGET_WIDTH, TARGET_HEIGHT],
        "pad_x": PAD_X,
        "pad_y": PAD_Y,
        "elapsed_seconds": round(elapsed, 2),
        "pipeline_order": [
            "1. Extract frames (raw video)",
            "2. Per-frame dlib landmarks (48-67)",
            "3. Tight dynamic mouth crop (real pixels only)",
            "4. Enhancement: Gamma + CLAHE on tight real mouth pixels",
            "5. Final Letterbox / Pad to 96x96 (spatial canvas embed)",
            "6. Fallback handling (temporal tracking & geometry index)"
        ]
    }
    with open(output_dataset_dir / "summary.json", "w") as f:
        json.dump(summary_info, f, indent=2)

    # Generate visual validation inspection grid
    if collected_samples:
        fig, axes = plt.subplots(len(collected_samples), 3, figsize=(11, 3.2 * len(collected_samples)))
        if len(collected_samples) == 1:
            axes = np.array([axes])

        for row_idx, (d_name, (tight_crop, enhanced_crop, final_lb)) in enumerate(collected_samples):
            axes[row_idx, 0].imshow(cv2.cvtColor(tight_crop, cv2.COLOR_BGR2RGB))
            axes[row_idx, 0].set_title(f"{d_name}: 1. Tight Crop (Real Pixels)", fontsize=10)
            axes[row_idx, 0].axis("off")

            axes[row_idx, 1].imshow(cv2.cvtColor(enhanced_crop, cv2.COLOR_BGR2RGB))
            axes[row_idx, 1].set_title(f"{d_name}: 2. Enhanced (Gamma+CLAHE)", fontsize=10, fontweight="bold", color="darkgreen")
            axes[row_idx, 1].axis("off")

            axes[row_idx, 2].imshow(cv2.cvtColor(final_lb, cv2.COLOR_BGR2RGB))
            axes[row_idx, 2].set_title(f"{d_name}: 3. Final 96x96 Letterbox", fontsize=10)
            axes[row_idx, 2].axis("off")

        plt.suptitle(
            f"Preprocessed Pipeline: Tight Crop -> Gamma({gamma_val})+CLAHE({clahe_clip}) -> Letterbox 96x96\nSpeaker: {speaker}",
            fontsize=13,
            fontweight="bold"
        )
        plt.tight_layout()
        preview_plot_path = results_dir / f"{output_dir_name}_inspection_grid.png"
        plt.savefig(preview_plot_path, dpi=160)
        plt.close()
        print(f"Visual inspection grid saved to: {preview_plot_path}")

    print("=" * 80)
    print(f"[SUCCESS] {output_dir_name} created successfully with {len(all_records)} frames!")
    print("=" * 80)


if __name__ == "__main__":
    main()
