#!/usr/bin/env python3
"""
MediaPipe Face Mesh Lip ROI Extraction Pipeline (112x112).

Extracts mouth region crops at 112x112 resolution using MediaPipe FaceLandmarker
on the raw Tai Phake digit dataset (30 speakers x 11 digits = 330 utterances, 13,610 frames).

Dataset output:
    data/experiments/mediapipe_lips_dataset_112x112/
    ├── s1/
    │   ├── d0/
    │   │   ├── frame_0001.jpg
    │   │   └── ...
    └── ...

Preview output:
    results/mediapipe_preview_112x112/
    ├── mediapipe_crop_report.csv
    ├── contact_sheet_*.png
    ├── comparison_*.png
    ├── comparison_96_vs_112_*.png
    └── failed_detections/
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceLandmarksConnections

# 40 canonical lip landmark indices from MediaPipe Face Mesh
LIP_CONNECTIONS = FaceLandmarksConnections.FACE_LANDMARKS_LIPS
LIP_LANDMARK_INDICES = sorted(list(set([c.start for c in LIP_CONNECTIONS] + [c.end for c in LIP_CONNECTIONS])))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "mediapipe" / "face_landmarker.task"
MODEL_DOWNLOAD_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

DEFAULT_INPUT_ROOT = PROJECT_ROOT / "data" / "digit"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "experiments" / "mediapipe_lips_dataset_112x112"
EXISTING_96_ROOT = PROJECT_ROOT / "data" / "experiments" / "mediapipe_lips_dataset_96x96"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "mediapipe_preview_112x112"

TARGET_SIZE = 112
PAD_X = 15
PAD_Y = 10

REPRESENTATIVE_UTTERANCES = [
    ("s1", "d0"),
    ("s1", "d5"),
    ("s1", "d10"),
    ("s15", "d0"),
    ("s15", "d5"),
    ("s15", "d10"),
    ("s30", "d0"),
    ("s30", "d5"),
    ("s30", "d10"),
]


def ensure_model_exists(model_path: Path) -> Path:
    """Verifies that face_landmarker.task exists, or downloads it automatically."""
    if not model_path.exists():
        print(f"MediaPipe model asset not found at {model_path}. Downloading...")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_DOWNLOAD_URL, str(model_path))
        print(f"Downloaded model to {model_path} ({model_path.stat().st_size / (1024 * 1024):.2f} MB)")
    return model_path


def init_detector(model_path: Path) -> vision.FaceLandmarker:
    """Initializes MediaPipe FaceLandmarker with CPU delegate for universal compatibility."""
    ensure_model_exists(model_path)
    base_options = python.BaseOptions(
        model_asset_path=str(model_path),
        delegate=python.BaseOptions.Delegate.CPU
    )
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    return vision.FaceLandmarker.create_from_options(options)


def detect_mouth_bbox(
    img_bgr: np.ndarray,
    detector: vision.FaceLandmarker,
    pad_x: int = PAD_X,
    pad_y: int = PAD_Y,
) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[np.ndarray], Optional[str]]:
    """
    Detects face mesh landmarks and extracts mouth bounding box with margin.
    Returns:
        bbox: (x1, y1, x2, y2) or None
        lip_pts: (N, 2) numpy array of pixel coordinates or None
        error_msg: None if successful, else error string
    """
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    
    try:
        detection_result = detector.detect(mp_img)
    except Exception as e:
        return None, None, f"Detection exception: {e}"
    
    if not detection_result.face_landmarks:
        return None, None, "No face detected"
    
    landmarks = detection_result.face_landmarks[0]
    
    # Extract lip landmark coordinates in pixel space
    lip_pts = np.array(
        [[landmarks[idx].x * w, landmarks[idx].y * h] for idx in LIP_LANDMARK_INDICES],
        dtype=np.float32
    )
    
    x_min = np.min(lip_pts[:, 0])
    x_max = np.max(lip_pts[:, 0])
    y_min = np.min(lip_pts[:, 1])
    y_max = np.max(lip_pts[:, 1])
    
    x1 = max(0, int(round(x_min - pad_x)))
    y1 = max(0, int(round(y_min - pad_y)))
    x2 = min(w, int(round(x_max + pad_x)))
    y2 = min(h, int(round(y_max + pad_y)))
    
    if (x2 - x1) <= 0 or (y2 - y1) <= 0:
        return None, lip_pts, f"Invalid crop dimensions: ({x1}, {y1}) to ({x2}, {y2})"
    
    return (x1, y1, x2, y2), lip_pts, None


def extract_and_resize_lip(
    img_bgr: np.ndarray,
    bbox: Tuple[int, int, int, int],
    target_size: int = TARGET_SIZE,
) -> np.ndarray:
    """Crops the bounding box from original frame and resizes directly to target_size x target_size."""
    x1, y1, x2, y2 = bbox
    crop = img_bgr[y1:y2, x1:x2]
    resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return resized


def generate_contact_sheets(
    output_root: Path,
    results_dir: Path,
    num_samples: int = 12,
    representative_list: List[Tuple[str, str]] = REPRESENTATIVE_UTTERANCES,
    target_size: int = TARGET_SIZE,
) -> List[Path]:
    """Generates multi-frame contact sheets for representative utterances."""
    results_dir.mkdir(parents=True, exist_ok=True)
    contact_sheet_paths = []
    
    for spk, dig in representative_list:
        utt_dir = output_root / spk / dig
        if not utt_dir.exists():
            continue
        
        frame_files = sorted(list(utt_dir.glob("*.jpg")))
        if not frame_files:
            continue
        
        n_frames = len(frame_files)
        indices = np.linspace(0, n_frames - 1, min(num_samples, n_frames), dtype=int)
        sampled_files = [frame_files[i] for i in indices]
        
        n_cols = 6
        n_rows = int(np.ceil(len(sampled_files) / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.3, n_rows * 2.6))
        fig.suptitle(f"MediaPipe {target_size}x{target_size} Lip Crops: {spk} / {dig} ({n_frames} frames)",
                     fontsize=14, fontweight="bold", y=0.98)
        
        if n_rows == 1:
            axes = np.array([axes])
        axes_flat = axes.flatten()
        
        for ax_idx, (frame_idx, fpath) in enumerate(zip(indices, sampled_files)):
            ax = axes_flat[ax_idx]
            img = cv2.imread(str(fpath))
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb)
            ax.set_title(f"frame {frame_idx + 1:04d}", fontsize=9)
            ax.axis("off")
        
        for ax_idx in range(len(sampled_files), len(axes_flat)):
            axes_flat[ax_idx].axis("off")
            
        plt.tight_layout()
        out_path = results_dir / f"contact_sheet_{spk}_{dig}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        contact_sheet_paths.append(out_path)
    
    return contact_sheet_paths


def generate_comparisons(
    input_root: Path,
    output_root: Path,
    report_df: pd.DataFrame,
    results_dir: Path,
    existing_96_root: Optional[Path] = EXISTING_96_ROOT,
    samples: Optional[List[Tuple[str, str, str]]] = None,
    target_size: int = TARGET_SIZE,
) -> Tuple[List[Path], List[Path]]:
    """
    Generates:
    1. Original Frame vs MediaPipe 112x112 Crop.
    2. Direct Side-by-Side Comparison: Original Frame | 96x96 MediaPipe Crop | 112x112 MediaPipe Crop.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    if samples is None:
        samples = [
            ("s1", "d0", "frame_0001.jpg"),
            ("s1", "d5", "frame_0015.jpg"),
            ("s1", "d10", "frame_0020.jpg"),
            ("s15", "d0", "frame_0001.jpg"),
            ("s15", "d5", "frame_0015.jpg"),
            ("s30", "d0", "frame_0001.jpg"),
            ("s30", "d5", "frame_0015.jpg"),
        ]
    
    comp_paths = []
    comp_96_vs_112_paths = []
    
    for spk, dig, fname in samples:
        raw_path = input_root / spk / dig / fname
        crop_path = output_root / spk / dig / fname
        crop_96_path = (existing_96_root / spk / dig / fname) if existing_96_root else None
        
        if not raw_path.exists() or not crop_path.exists():
            continue
        
        raw_img = cv2.imread(str(raw_path))
        crop_img = cv2.imread(str(crop_path))
        if raw_img is None or crop_img is None:
            continue
        
        # Query report for bbox
        row = report_df[(report_df["speaker"] == spk) & (report_df["digit"] == dig) & (report_df["frame"] == fname)]
        if row.empty or not row.iloc[0]["detected"]:
            continue
        
        r = row.iloc[0]
        x1, y1, x2, y2 = int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])
        
        # 1. Standard Comparison Figure (Full raw | Face ROI with BBox | 112x112 crop)
        annotated_raw = raw_img.copy()
        cv2.rectangle(annotated_raw, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            annotated_raw,
            f"MediaPipe Mouth ({x2-x1}x{y2-y1})",
            (max(10, x1), max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        
        zoom_margin = 130
        zx1, zy1 = max(0, x1 - zoom_margin), max(0, y1 - zoom_margin)
        zx2, zy2 = min(raw_img.shape[1], x2 + zoom_margin), min(raw_img.shape[0], y2 + zoom_margin)
        zoomed_face = annotated_raw[zy1:zy2, zx1:zx2]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 6), gridspec_kw={"width_ratios": [1.2, 1.2, 1.0]})
        fig.suptitle(f"MediaPipe 112x112 Extraction: {spk} / {dig} / {fname}", fontsize=14, fontweight="bold")
        
        axes[0].imshow(cv2.cvtColor(annotated_raw, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"Original Frame ({raw_img.shape[1]}x{raw_img.shape[0]})", fontsize=11)
        axes[0].axis("off")
        
        axes[1].imshow(cv2.cvtColor(zoomed_face, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"Face ROI with Bounding Box", fontsize=11)
        axes[1].axis("off")
        
        axes[2].imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f"MediaPipe Mouth Crop ({target_size}x{target_size} RGB)", fontsize=11)
        axes[2].axis("off")
        
        plt.tight_layout()
        out_path = results_dir / f"comparison_{spk}_{dig}_{Path(fname).stem}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        comp_paths.append(out_path)
        
        # 2. Side-by-side 96x96 vs 112x112 resolution comparison
        if crop_96_path and crop_96_path.exists():
            crop_96_img = cv2.imread(str(crop_96_path))
            if crop_96_img is not None:
                fig2, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(13, 5), gridspec_kw={"width_ratios": [1.2, 1.0, 1.0]})
                fig2.suptitle(f"Resolution Comparison (96x96 vs 112x112): {spk} / {dig} / {fname}",
                              fontsize=13, fontweight="bold")
                
                ax_a.imshow(cv2.cvtColor(zoomed_face, cv2.COLOR_BGR2RGB))
                ax_a.set_title("Source Face ROI", fontsize=11)
                ax_a.axis("off")
                
                ax_b.imshow(cv2.cvtColor(crop_96_img, cv2.COLOR_BGR2RGB))
                ax_b.set_title("MediaPipe 96x96 Crop", fontsize=11)
                ax_b.axis("off")
                
                ax_c.imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
                ax_c.set_title("MediaPipe 112x112 Crop", fontsize=11)
                ax_c.axis("off")
                
                plt.tight_layout()
                comp_res_path = results_dir / f"comparison_96_vs_112_{spk}_{dig}_{Path(fname).stem}.png"
                fig2.savefig(comp_res_path, dpi=150, bbox_inches="tight")
                plt.close(fig2)
                comp_96_vs_112_paths.append(comp_res_path)
    
    return comp_paths, comp_96_vs_112_paths


def process_dataset(
    input_root: Path,
    output_root: Path,
    results_dir: Path,
    model_path: Path,
    target_size: int = TARGET_SIZE,
    pad_x: int = PAD_X,
    pad_y: int = PAD_Y,
    speakers: Optional[List[str]] = None,
    digits: Optional[List[str]] = None,
    overwrite: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """Runs MediaPipe landmark detection, 112x112 mouth cropping, and report generation."""
    detector = init_detector(model_path)
    output_root.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    failed_dir = results_dir / "failed_detections"
    
    if speakers is None:
        speakers = sorted([p.name for p in input_root.iterdir() if p.is_dir() and p.name.startswith("s")],
                          key=lambda x: int(x[1:]) if x[1:].isdigit() else 999)
    if digits is None:
        digits = [f"d{i}" for i in range(11)]
    
    tasks = []
    utterance_set = set()
    for spk in speakers:
        for dig in digits:
            folder = input_root / spk / dig
            if not folder.is_dir():
                continue
            frame_files = sorted(folder.glob("*.jpg"))
            if frame_files:
                utterance_set.add((spk, dig))
            for fpath in frame_files:
                tasks.append((spk, dig, fpath.name, fpath))
    
    print(f"Total frame tasks identified: {len(tasks)} across {len(speakers)} speakers and {len(utterance_set)} utterances.")
    
    records = []
    success_count = 0
    fail_count = 0
    start_time = time.time()
    
    for spk, dig, fname, fpath in tqdm(tasks, desc=f"MediaPipe {target_size}x{target_size} Extraction", unit="frame"):
        out_dir = output_root / spk / dig
        out_path = out_dir / fname
        
        if out_path.exists() and not overwrite:
            success_count += 1
            records.append({
                "speaker": spk,
                "digit": dig,
                "frame": fname,
                "detected": True,
                "x1": np.nan,
                "y1": np.nan,
                "x2": np.nan,
                "y2": np.nan,
                "crop_width": target_size,
                "crop_height": target_size,
                "error": "Skipped (already exists)",
            })
            continue
        
        img = cv2.imread(str(fpath))
        if img is None:
            fail_count += 1
            records.append({
                "speaker": spk,
                "digit": dig,
                "frame": fname,
                "detected": False,
                "x1": np.nan,
                "y1": np.nan,
                "x2": np.nan,
                "y2": np.nan,
                "crop_width": np.nan,
                "crop_height": np.nan,
                "error": "Failed to read image",
            })
            continue
        
        bbox, lip_pts, err = detect_mouth_bbox(img, detector, pad_x=pad_x, pad_y=pad_y)
        
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            crop_w = x2 - x1
            crop_h = y2 - y1
            
            crop_resized = extract_and_resize_lip(img, bbox, target_size=target_size)
            
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), crop_resized)
            
            success_count += 1
            records.append({
                "speaker": spk,
                "digit": dig,
                "frame": fname,
                "detected": True,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "crop_width": crop_w,
                "crop_height": crop_h,
                "error": None,
            })
        else:
            fail_count += 1
            failed_dir.mkdir(parents=True, exist_ok=True)
            failed_vis_path = failed_dir / f"{spk}_{dig}_{fname}"
            annotated_fail = img.copy()
            cv2.putText(
                annotated_fail,
                f"FAILED: {err}",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imwrite(str(failed_vis_path), annotated_fail)
            
            records.append({
                "speaker": spk,
                "digit": dig,
                "frame": fname,
                "detected": False,
                "x1": np.nan,
                "y1": np.nan,
                "x2": np.nan,
                "y2": np.nan,
                "crop_width": np.nan,
                "crop_height": np.nan,
                "error": err,
            })
    
    elapsed = time.time() - start_time
    total_processed = len(tasks)
    success_rate = (success_count / total_processed * 100.0) if total_processed > 0 else 0.0
    
    report_df = pd.DataFrame(records)
    
    report_csv_output = output_root / "mediapipe_crop_report.csv"
    report_csv_results = results_dir / "mediapipe_crop_report.csv"
    report_df.to_csv(report_csv_output, index=False)
    report_df.to_csv(report_csv_results, index=False)
    
    summary = {
        "total_frames_processed": total_processed,
        "successful_detections": success_count,
        "failed_detections": fail_count,
        "detection_success_rate": success_rate,
        "utterances_processed": len(utterance_set),
        "elapsed_seconds": elapsed,
        "fps": total_processed / elapsed if elapsed > 0 else 0.0,
        "output_resolution": f"{target_size}x{target_size}",
        "output_directory": str(output_root),
        "results_directory": str(results_dir),
    }
    
    return report_df, summary


def main():
    parser = argparse.ArgumentParser(description="MediaPipe Face Mesh 112x112 Lip Extraction Pipeline")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT, help="Root path to raw frames")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Output dataset path")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Results & preview directory")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Path to face_landmarker.task")
    parser.add_argument("--target-size", type=int, default=TARGET_SIZE, help="Target crop size (default 112)")
    parser.add_argument("--pad-x", type=int, default=PAD_X, help="Horizontal padding around lips (px)")
    parser.add_argument("--pad-y", type=int, default=PAD_Y, help="Vertical padding around lips (px)")
    parser.add_argument("--speakers", nargs="+", default=None, help="Optional subset of speakers (e.g. s1 s2)")
    parser.add_argument("--digits", nargs="+", default=None, help="Optional subset of digits (e.g. d0 d5)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cropped frames")
    args = parser.parse_args()
    
    # Fallback input path if data/digit doesn't exist directly
    if not args.input_root.exists():
        fallback = PROJECT_ROOT / "data" / "experiments" / "digit"
        if fallback.exists():
            args.input_root = fallback
    
    print("=" * 75)
    print(f"MediaPipe Face Mesh Lip Extraction Pipeline ({args.target_size}x{args.target_size})")
    print("=" * 75)
    print(f"Input Directory:       {args.input_root.resolve()}")
    print(f"Output Dataset:       {args.output_root.resolve()}")
    print(f"Results Directory:    {args.results_dir.resolve()}")
    print(f"Model Path:           {args.model_path.resolve()}")
    print(f"Target Resolution:    {args.target_size}x{args.target_size} (RGB)")
    print(f"Padding Margin:       pad_x={args.pad_x}px, pad_y={args.pad_y}px")
    print(f"Overwrite:            {args.overwrite}")
    print("=" * 75)
    
    report_df, summary = process_dataset(
        input_root=args.input_root,
        output_root=args.output_root,
        results_dir=args.results_dir,
        model_path=args.model_path,
        target_size=args.target_size,
        pad_x=args.pad_x,
        pad_y=args.pad_y,
        speakers=args.speakers,
        digits=args.digits,
        overwrite=args.overwrite,
    )
    
    print(f"\nGenerating Contact Sheets for Representative Utterances ({args.target_size}x{args.target_size})...")
    cs_paths = generate_contact_sheets(
        output_root=args.output_root,
        results_dir=args.results_dir,
        num_samples=12,
        representative_list=REPRESENTATIVE_UTTERANCES,
        target_size=args.target_size,
    )
    print(f"Generated {len(cs_paths)} contact sheets in {args.results_dir}")
    
    print(f"\nGenerating Comparison Visualizations (Original vs 112x112 vs 96x96)...")
    comp_paths, comp_res_paths = generate_comparisons(
        input_root=args.input_root,
        output_root=args.output_root,
        report_df=report_df,
        results_dir=args.results_dir,
        existing_96_root=EXISTING_96_ROOT,
        target_size=args.target_size,
    )
    print(f"Generated {len(comp_paths)} original-vs-crop figures and {len(comp_res_paths)} 96-vs-112 resolution figures in {args.results_dir}")
    
    print("\n" + "=" * 75)
    print(f"MediaPipe {args.target_size}x{args.target_size} preprocessing complete")
    print(f"Total frames processed:          {summary['total_frames_processed']}")
    print(f"Successful MediaPipe detections: {summary['successful_detections']}")
    print(f"Failed detections:               {summary['failed_detections']}")
    print(f"Detection success percentage:    {summary['detection_success_rate']:.2f}%")
    print(f"Number of utterances processed:  {summary['utterances_processed']}")
    print(f"Output dataset path:             {summary['output_directory']}")
    print(f"Preview path:                    {summary['results_directory']}")
    print(f"Processing Speed:                {summary['fps']:.1f} fps ({summary['elapsed_seconds']:.1f}s)")
    print("=" * 75)


if __name__ == "__main__":
    main()
