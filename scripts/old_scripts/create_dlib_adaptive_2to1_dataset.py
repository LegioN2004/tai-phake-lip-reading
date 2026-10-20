#!/usr/bin/env python3
"""
Create Dlib Adaptive 2:1 Lip-Cropping Experiment Dataset (128x64).
Processes all 30 speakers (s1..s30) across all 11 digits (d0..d10) from:
    data/experiments/digit/
into:
    data/experiments/cropped_lips_dataset_e_dlib_adaptive_2to1_128x64/

Generates:
    - Master CSV: crop_geometry.csv
    - Summary metrics: crop_metrics.json
    - Visual inspection multi-panel grids in inspection_samples/
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import dlib
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_ROOT = PROJECT_ROOT / "data" / "experiments" / "digit"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "experiments" / "cropped_lips_dataset_e_dlib_adaptive_2to1_128x64"
MODEL_PATH = PROJECT_ROOT / "models" / "dlib" / "shape_predictor_68_face_landmarks.dat"
INSPECTION_DIR = OUTPUT_ROOT / "inspection_samples"

TARGET_WIDTH = 128
TARGET_HEIGHT = 64
TARGET_ASPECT_RATIO = 2.0  # 2:1
MARGIN_FACTOR = 0.20  # 20% proportional surrounding context around mouth landmarks

# Global detector and predictor handles for worker processes
_detector = None
_predictor = None


def init_worker(model_path_str: str) -> None:
    global _detector, _predictor
    _detector = dlib.get_frontal_face_detector()
    _predictor = dlib.shape_predictor(model_path_str)


def crop_adaptive_2to1(
    img: np.ndarray,
    mouth_x_min: float,
    mouth_y_min: float,
    mouth_x_max: float,
    mouth_y_max: float,
    margin_factor: float = MARGIN_FACTOR,
) -> Tuple[np.ndarray, Dict[str, any]]:
    """
    Computes an adaptive 2:1 crop around the mouth landmarks.
    Enlarges the crop if necessary so that the mouth + margin is never cut off.
    Pads image with border reflection / edge replication if extending outside image bounds.
    Resizes output strictly to 128x64.
    """
    img_h, img_w = img.shape[:2]
    
    mouth_w = mouth_x_max - mouth_x_min
    mouth_h = mouth_y_max - mouth_y_min
    
    center_x = (mouth_x_min + mouth_x_max) / 2.0
    center_y = (mouth_y_min + mouth_y_max) / 2.0
    
    # Base bounding box with context margin
    target_w = mouth_w * (1.0 + 2.0 * margin_factor)
    target_h = mouth_h * (1.0 + 2.0 * margin_factor)
    
    # Adaptive 2:1 requirement:
    # crop_w = 2.0 * crop_h
    # Must fully contain target_w and target_h:
    # crop_h >= target_h and crop_w >= target_w => crop_h >= target_w / 2.0
    # Therefore, crop_h = max(target_h, target_w / 2.0)
    enlarged_crop = target_h > (target_w / 2.0)
    crop_h = max(target_h, target_w / 2.0)
    crop_w = 2.0 * crop_h
    
    # Compute float bounding box centered at mouth center
    x1_f = center_x - crop_w / 2.0
    x2_f = center_x + crop_w / 2.0
    y1_f = center_y - crop_h / 2.0
    y2_f = center_y + crop_h / 2.0
    
    # Integer box coordinates
    x1 = int(round(x1_f))
    y1 = int(round(y1_f))
    x2 = int(round(x2_f))
    y2 = int(round(y2_f))
    
    # Ensure exact 2:1 discrete dimensions (even width)
    actual_crop_w = x2 - x1
    actual_crop_h = y2 - y1
    if actual_crop_w % 2 != 0:
        x2 += 1
        actual_crop_w = x2 - x1
    if actual_crop_h * 2 != actual_crop_w:
        # adjust y2
        y2 = y1 + (actual_crop_w // 2)
        actual_crop_h = y2 - y1
        
    actual_ar = float(actual_crop_w) / float(actual_crop_h) if actual_crop_h > 0 else 0.0
    
    # Boundary check and padding if needed
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - img_w)
    pad_bottom = max(0, y2 - img_h)
    
    padding_used = (pad_left > 0) or (pad_top > 0) or (pad_right > 0) or (pad_bottom > 0)
    
    if padding_used:
        # Pad using BORDER_REFLECT_101 for seamless natural appearance
        padded_img = cv2.copyMakeBorder(
            img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_REFLECT_101,
        )
        crop_x1 = x1 + pad_left
        crop_y1 = y1 + pad_top
        crop_x2 = x2 + pad_left
        crop_y2 = y2 + pad_top
        cropped_patch = padded_img[crop_y1:crop_y2, crop_x1:crop_x2]
    else:
        cropped_patch = img[y1:y2, x1:x2]
        
    # Resize to exactly 128x64
    resized_patch = cv2.resize(cropped_patch, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
    
    meta = {
        "crop_x1": x1,
        "crop_y1": y1,
        "crop_x2": x2,
        "crop_y2": y2,
        "crop_width": actual_crop_w,
        "crop_height": actual_crop_h,
        "crop_aspect_ratio": actual_ar,
        "padding_used": padding_used,
        "enlarged_crop": enlarged_crop,
    }
    return resized_patch, meta


def process_single_frame(task: Tuple[str, str, str, str, str]) -> Dict[str, any]:
    """
    Worker task:
    Loads original JPG, detects face & mouth landmarks, calculates adaptive 2:1 crop,
    resizes to 128x64, saves JPG to output destination, and returns metadata.
    """
    global _detector, _predictor
    spk, dig, fname, in_path_str, out_path_str = task
    
    img = cv2.imread(in_path_str)
    if img is None:
        return {
            "speaker": spk,
            "digit": dig,
            "frame": fname,
            "original_width": np.nan,
            "original_height": np.nan,
            "mouth_x_min": np.nan,
            "mouth_y_min": np.nan,
            "mouth_x_max": np.nan,
            "mouth_y_max": np.nan,
            "crop_x1": np.nan,
            "crop_y1": np.nan,
            "crop_x2": np.nan,
            "crop_y2": np.nan,
            "crop_width": np.nan,
            "crop_height": np.nan,
            "crop_aspect_ratio": np.nan,
            "padding_used": False,
            "enlarged_crop": False,
            "detected": False,
            "error": "Failed to read input image",
        }
        
    img_h, img_w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    
    faces = _detector(gray_eq, 0)
    if len(faces) == 0:
        faces = _detector(gray_eq, 1)
        
    if len(faces) == 0:
        return {
            "speaker": spk,
            "digit": dig,
            "frame": fname,
            "original_width": img_w,
            "original_height": img_h,
            "mouth_x_min": np.nan,
            "mouth_y_min": np.nan,
            "mouth_x_max": np.nan,
            "mouth_y_max": np.nan,
            "crop_x1": np.nan,
            "crop_y1": np.nan,
            "crop_x2": np.nan,
            "crop_y2": np.nan,
            "crop_width": np.nan,
            "crop_height": np.nan,
            "crop_aspect_ratio": np.nan,
            "padding_used": False,
            "enlarged_crop": False,
            "detected": False,
            "error": "No face detected by Dlib",
        }
        
    largest_face = max(faces, key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top()))
    
    try:
        shape = _predictor(gray, largest_face)
        # Landmarks 48 through 67 (20 mouth points)
        mouth_pts = np.array([[shape.part(i).x, shape.part(i).y] for i in range(48, 68)], dtype=np.float32)
    except Exception as e:
        return {
            "speaker": spk,
            "digit": dig,
            "frame": fname,
            "original_width": img_w,
            "original_height": img_h,
            "mouth_x_min": np.nan,
            "mouth_y_min": np.nan,
            "mouth_x_max": np.nan,
            "mouth_y_max": np.nan,
            "crop_x1": np.nan,
            "crop_y1": np.nan,
            "crop_x2": np.nan,
            "crop_y2": np.nan,
            "crop_width": np.nan,
            "crop_height": np.nan,
            "crop_aspect_ratio": np.nan,
            "padding_used": False,
            "enlarged_crop": False,
            "detected": False,
            "error": f"Landmark error: {str(e)}",
        }
        
    mouth_x_min = float(np.min(mouth_pts[:, 0]))
    mouth_y_min = float(np.min(mouth_pts[:, 1]))
    mouth_x_max = float(np.max(mouth_pts[:, 0]))
    mouth_y_max = float(np.max(mouth_pts[:, 1]))
    
    cropped_patch, crop_meta = crop_adaptive_2to1(
        img,
        mouth_x_min,
        mouth_y_min,
        mouth_x_max,
        mouth_y_max,
        margin_factor=MARGIN_FACTOR,
    )
    
    # Save output frame
    out_path = Path(out_path_str)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path_str, cropped_patch, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    
    return {
        "speaker": spk,
        "digit": dig,
        "frame": fname,
        "original_width": img_w,
        "original_height": img_h,
        "mouth_x_min": mouth_x_min,
        "mouth_y_min": mouth_y_min,
        "mouth_x_max": mouth_x_max,
        "mouth_y_max": mouth_y_max,
        "crop_x1": crop_meta["crop_x1"],
        "crop_y1": crop_meta["crop_y1"],
        "crop_x2": crop_meta["crop_x2"],
        "crop_y2": crop_meta["crop_y2"],
        "crop_width": crop_meta["crop_width"],
        "crop_height": crop_meta["crop_height"],
        "crop_aspect_ratio": crop_meta["crop_aspect_ratio"],
        "padding_used": crop_meta["padding_used"],
        "enlarged_crop": crop_meta["enlarged_crop"],
        "detected": True,
        "error": None,
    }


def create_visual_inspection_samples(df: pd.DataFrame, num_samples: int = 12) -> None:
    """
    Creates side-by-side visual inspection images:
    Original Frame -> Detected Mouth / Crop Rectangle -> Final 128x64 Crop.
    Selects diverse samples across multiple speakers and digit classes.
    """
    INSPECTION_DIR.mkdir(parents=True, exist_ok=True)
    
    # Pick a balanced selection of speakers and digits
    sample_queries = [
        ("s1", "d0"),
        ("s2", "d2"),
        ("s4", "d4"),
        ("s7", "d6"),
        ("s10", "d8"),
        ("s12", "d10"),
        ("s16", "d1"),
        ("s19", "d3"),
        ("s21", "d5"),
        ("s24", "d7"),
        ("s28", "d9"),
        ("s30", "d0"),
    ]
    
    sample_rows = []
    for spk, dig in sample_queries:
        subset = df[(df["speaker"] == spk) & (df["digit"] == dig) & (df["detected"] == True)]
        if not subset.empty:
            # Pick a middle frame where mouth articulation is active
            mid_idx = len(subset) // 2
            sample_rows.append(subset.iloc[mid_idx])
            
    summary_panels = []
    
    for i, row in enumerate(sample_rows):
        spk = row["speaker"]
        dig = row["digit"]
        fname = row["frame"]
        
        orig_img_path = INPUT_ROOT / spk / dig / fname
        crop_img_path = OUTPUT_ROOT / spk / dig / fname
        
        orig_img = cv2.imread(str(orig_img_path))
        crop_img = cv2.imread(str(crop_img_path))
        
        if orig_img is None or crop_img is None:
            continue
            
        # Draw detected mouth and adaptive crop rectangle on a copy of original image
        annotated = orig_img.copy()
        
        # Mouth bounding box (Green)
        mx1, my1, mx2, my2 = int(row["mouth_x_min"]), int(row["mouth_y_min"]), int(row["mouth_x_max"]), int(row["mouth_y_max"])
        cv2.rectangle(annotated, (mx1, my1), (mx2, my2), (0, 255, 0), 2)
        
        # 2:1 Crop bounding box (Cyan)
        cx1, cy1, cx2, cy2 = int(row["crop_x1"]), int(row["crop_y1"]), int(row["crop_x2"]), int(row["crop_y2"])
        cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), (255, 255, 0), 3)
        
        # Add labels
        label_text = f"{spk} {dig} {fname} | Mouth:[{mx2-mx1}x{my2-my1}] Crop:[{cx2-cx1}x{cy2-cy1}] AR={row['crop_aspect_ratio']:.2f}"
        cv2.putText(annotated, label_text, (cx1 - 20, max(30, cy1 - 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Create a zoomed-in visualization around the face/mouth for clarity
        # Define zoom window around mouth center
        zoom_margin = 160
        zx1 = max(0, cx1 - zoom_margin)
        zy1 = max(0, cy1 - zoom_margin)
        zx2 = min(annotated.shape[1], cx2 + zoom_margin)
        zy2 = min(annotated.shape[0], cy2 + zoom_margin)
        annotated_zoom = annotated[zy1:zy2, zx1:zx2]
        
        # Scale zoom view to height 300
        target_disp_h = 300
        zoom_scale = target_disp_h / annotated_zoom.shape[0]
        disp_zoom = cv2.resize(annotated_zoom, (int(annotated_zoom.shape[1] * zoom_scale), target_disp_h))
        
        # Scale final crop (128x64) up with nearest neighbor or bilinear for visual inspection display (e.g. 300x150)
        disp_crop = cv2.resize(crop_img, (300, 150), interpolation=cv2.INTER_NEAREST)
        # Pad disp_crop to height 300 with black background
        disp_crop_padded = np.zeros((target_disp_h, 300, 3), dtype=np.uint8)
        disp_crop_padded[75:225, :300] = disp_crop
        cv2.putText(disp_crop_padded, f"Final 128x64 Crop", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(disp_crop_padded, f"Aspect Ratio: 2.00", (20, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Assemble side-by-side inspection row
        # 1. Full frame scaled down
        disp_full = cv2.resize(orig_img, (int(orig_img.shape[1] * (target_disp_h / orig_img.shape[0])), target_disp_h))
        cv2.putText(disp_full, f"Original Frame ({orig_img.shape[1]}x{orig_img.shape[0]})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        panel = np.hstack([disp_full, disp_zoom, disp_crop_padded])
        
        out_inspection_path = INSPECTION_DIR / f"inspection_{spk}_{dig}_{fname}"
        cv2.imwrite(str(out_inspection_path), panel)
        summary_panels.append(panel)
        
    print(f"Visual inspection samples generated in: {INSPECTION_DIR} ({len(sample_rows)} samples)")


def main():
    parser = argparse.ArgumentParser(description="Create Dlib Adaptive 2:1 Lip-Cropped Dataset (128x64)")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker processes")
    args = parser.parse_args()
    
    start_time = time.time()
    
    if not INPUT_ROOT.exists():
        print(f"Error: Input directory {INPUT_ROOT} does not exist!")
        sys.exit(1)
        
    if not MODEL_PATH.exists():
        print(f"Error: Dlib model {MODEL_PATH} does not exist!")
        sys.exit(1)
        
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("      DLIB ADAPTIVE 2:1 LIP-CROPPING DATASET GENERATION (128x64)")
    print("=" * 80)
    print(f"Input Directory  : {INPUT_ROOT}")
    print(f"Output Directory : {OUTPUT_ROOT}")
    print(f"Dlib Model       : {MODEL_PATH}")
    print(f"Target Output Res: {TARGET_WIDTH} x {TARGET_HEIGHT} (Aspect Ratio {TARGET_ASPECT_RATIO}:1)")
    print(f"Context Margin   : {int(MARGIN_FACTOR * 100)}% surrounding padding around mouth landmarks")
    print(f"Worker Processes : {args.workers}")
    print("-" * 80)
    
    # 1. Gather all tasks in strict speaker/digit order
    tasks: List[Tuple[str, str, str, str, str]] = []
    for sp_idx in range(1, 31):
        spk = f"s{sp_idx}"
        spk_dir = INPUT_ROOT / spk
        if not spk_dir.exists():
            continue
        for dg_idx in range(11):
            dig = f"d{dg_idx}"
            dig_dir = spk_dir / dig
            if not dig_dir.exists():
                continue
            frame_files = sorted(
                dig_dir.glob("frame_*.jpg"),
                key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)) if re.search(r"(\d+)", p.stem) else 0,
            )
            for f in frame_files:
                out_path = OUTPUT_ROOT / spk / dig / f.name
                tasks.append((spk, dig, f.name, str(f), str(out_path)))
                
    total_input_frames = len(tasks)
    print(f"Total input frames discovered: {total_input_frames:,} across 30 speakers and 330 utterances.")
    
    # 2. Execute parallel processing
    results = []
    print(f"\nProcessing frames with ProcessPoolExecutor ({args.workers} workers)...")
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(str(MODEL_PATH),)) as executor:
        for res in tqdm(executor.map(process_single_frame, tasks, chunksize=50), total=total_input_frames, desc="Cropping Frames"):
            results.append(res)
            
    df = pd.DataFrame(results)
    
    # Ensure ordered columns matching specification
    columns_order = [
        "speaker",
        "digit",
        "frame",
        "original_width",
        "original_height",
        "mouth_x_min",
        "mouth_y_min",
        "mouth_x_max",
        "mouth_y_max",
        "crop_x1",
        "crop_y1",
        "crop_x2",
        "crop_y2",
        "crop_width",
        "crop_height",
        "crop_aspect_ratio",
        "padding_used",
        "enlarged_crop",
        "detected",
        "error",
    ]
    df = df[columns_order]
    
    # 3. Save master CSV
    csv_path = OUTPUT_ROOT / "crop_geometry.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nMaster geometry CSV saved to: {csv_path}")
    
    # 4. Compute metrics
    successful_df = df[df["detected"] == True]
    failed_df = df[df["detected"] == False]
    
    num_successful = int(len(successful_df))
    num_failed = int(len(failed_df))
    detection_rate = float(num_successful / total_input_frames * 100) if total_input_frames > 0 else 0.0
    
    # Output file verification
    print("\nVerifying all output files...")
    output_files = list(OUTPUT_ROOT.glob("s*/*/*.jpg"))
    total_output_frames = len(output_files)
    
    # Verify exact dimensions (128 x 64 x 3)
    dimension_errors = 0
    channel_errors = 0
    for out_img_path in tqdm(output_files, desc="Verifying Output Dimensions"):
        img = cv2.imread(str(out_img_path))
        if img is None:
            dimension_errors += 1
            continue
        h, w, c = img.shape
        if w != TARGET_WIDTH or h != TARGET_HEIGHT:
            dimension_errors += 1
        if c != 3:
            channel_errors += 1
            
    w_series = successful_df["crop_width"]
    h_series = successful_df["crop_height"]
    ar_series = successful_df["crop_aspect_ratio"]
    
    def get_stats_dict(series: pd.Series) -> Dict[str, float]:
        return {
            "min": float(series.min()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()),
            "max": float(series.max()),
            "p5": float(np.percentile(series, 5)),
            "p25": float(np.percentile(series, 25)),
            "p75": float(np.percentile(series, 75)),
            "p95": float(np.percentile(series, 95)),
        }
        
    crop_w_stats = get_stats_dict(w_series)
    crop_h_stats = get_stats_dict(h_series)
    crop_ar_stats = get_stats_dict(ar_series)
    
    padding_count = int(successful_df["padding_used"].sum())
    enlarged_count = int(successful_df["enlarged_crop"].sum())
    
    metrics = {
        "dataset_name": "cropped_lips_dataset_e_dlib_adaptive_2to1_128x64",
        "target_width": TARGET_WIDTH,
        "target_height": TARGET_HEIGHT,
        "target_aspect_ratio": TARGET_ASPECT_RATIO,
        "margin_factor": MARGIN_FACTOR,
        "total_input_frames": total_input_frames,
        "successful_detections": num_successful,
        "failed_detections": num_failed,
        "detection_success_rate_percent": round(detection_rate, 4),
        "total_output_frames": total_output_frames,
        "dimension_verification": {
            "expected_shape": [TARGET_HEIGHT, TARGET_WIDTH, 3],
            "dimension_errors": dimension_errors,
            "channel_errors": channel_errors,
            "all_exact_128x64x3": (dimension_errors == 0 and channel_errors == 0 and total_output_frames == total_input_frames),
        },
        "crop_width_statistics": crop_w_stats,
        "crop_height_statistics": crop_h_stats,
        "crop_aspect_ratio_statistics": crop_ar_stats,
        "padding_count": padding_count,
        "padding_rate_percent": round(float(padding_count / num_successful * 100), 2) if num_successful > 0 else 0.0,
        "enlarged_crop_count": enlarged_count,
        "enlarged_crop_rate_percent": round(float(enlarged_count / num_successful * 100), 2) if num_successful > 0 else 0.0,
        "elapsed_seconds": round(time.time() - start_time, 2),
    }
    
    # 5. Save JSON metrics
    metrics_path = OUTPUT_ROOT / "crop_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics JSON saved to: {metrics_path}")
    
    # 6. Generate visual inspection samples
    print("\nCreating visual inspection grids...")
    create_visual_inspection_samples(df, num_samples=12)
    
    # 7. Print concise processing summary
    print("\n" + "=" * 80)
    print("                    PROCESSING & VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"Output Dataset Location : {OUTPUT_ROOT}")
    print(f"Total Input Frames      : {total_input_frames:,}")
    print(f"Successful Detections   : {num_successful:,} ({detection_rate:.2f}%)")
    print(f"Failed Detections       : {num_failed}")
    print(f"Total Output Frames     : {total_output_frames:,}")
    print(f"Output Size Verification: {TARGET_WIDTH} x {TARGET_HEIGHT} x 3 -> {'100% VERIFIED' if dimension_errors == 0 else f'ERRORS: {dimension_errors}'}")
    print(f"Padding Used Frames     : {padding_count:,} ({metrics['padding_rate_percent']}%)")
    print(f"Enlarged Crop Frames    : {enlarged_count:,} ({metrics['enlarged_crop_rate_percent']}%)")
    print("-" * 80)
    print("Crop Geometry Statistics (Pre-Resize):")
    print(f"  Crop Width  (px) : Mean={crop_w_stats['mean']:.2f}, Median={crop_w_stats['median']:.1f}, Min={crop_w_stats['min']:.1f}, Max={crop_w_stats['max']:.1f}")
    print(f"  Crop Height (px) : Mean={crop_h_stats['mean']:.2f}, Median={crop_h_stats['median']:.1f}, Min={crop_h_stats['min']:.1f}, Max={crop_h_stats['max']:.1f}")
    print(f"  Aspect Ratio     : Mean={crop_ar_stats['mean']:.4f}, Median={crop_ar_stats['median']:.4f}, Min={crop_ar_stats['min']:.4f}, Max={crop_ar_stats['max']:.4f}")
    print(f"Elapsed Processing Time : {metrics['elapsed_seconds']:.2f} seconds")
    print("=" * 80)


if __name__ == "__main__":
    main()
