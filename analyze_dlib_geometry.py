#!/usr/bin/env python3
"""
Full Dlib Lip ROI Geometry Analysis across the cleaned Tai Phake digits dataset.
Analyzes all 330 utterances and all 13,610 frames using multiprocessing.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_ROOT = Path("data/digit")
MODEL_PATH = Path("models/dlib/shape_predictor_68_face_landmarks.dat")
RESULTS_DIR = Path("results/dlib")
PAD_X = 15
PAD_Y = 10

# Global detector and predictor handles for worker processes
_detector = None
_predictor = None


def init_worker(model_path_str: str) -> None:
    global _detector, _predictor
    _detector = dlib.get_frontal_face_detector()
    _predictor = dlib.shape_predictor(model_path_str)


def process_frame(frame_info: Tuple[str, str, str, str]) -> Dict[str, any]:
    global _detector, _predictor
    spk, dig, fname, fpath_str = frame_info
    
    img = cv2.imread(fpath_str)
    if img is None:
        return {
            "speaker": spk,
            "digit": dig,
            "frame": fname,
            "path": fpath_str,
            "detected": False,
            "roi_width": np.nan,
            "roi_height": np.nan,
            "raw_width": np.nan,
            "raw_height": np.nan,
            "aspect_ratio": np.nan,
            "raw_aspect_ratio": np.nan,
            "center_x": np.nan,
            "center_y": np.nan,
            "x1": np.nan,
            "y1": np.nan,
            "x2": np.nan,
            "y2": np.nan,
            "error": "Unreadable image",
        }

    img_h, img_w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)

    # Fast detection with upsample=0 first
    faces = _detector(gray_eq, 0)
    if len(faces) == 0:
        # Fallback to upsample=1
        faces = _detector(gray_eq, 1)

    if len(faces) == 0:
        return {
            "speaker": spk,
            "digit": dig,
            "frame": fname,
            "path": fpath_str,
            "detected": False,
            "roi_width": np.nan,
            "roi_height": np.nan,
            "raw_width": np.nan,
            "raw_height": np.nan,
            "aspect_ratio": np.nan,
            "raw_aspect_ratio": np.nan,
            "center_x": np.nan,
            "center_y": np.nan,
            "x1": np.nan,
            "y1": np.nan,
            "x2": np.nan,
            "y2": np.nan,
            "error": "No face detected",
        }

    largest_face = max(faces, key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top()))

    try:
        shape = _predictor(gray, largest_face)
        pts = np.array([[shape.part(i).x, shape.part(i).y] for i in range(48, 68)], dtype=np.int32)
    except Exception as e:
        return {
            "speaker": spk,
            "digit": dig,
            "frame": fname,
            "path": fpath_str,
            "detected": False,
            "roi_width": np.nan,
            "roi_height": np.nan,
            "raw_width": np.nan,
            "raw_height": np.nan,
            "aspect_ratio": np.nan,
            "raw_aspect_ratio": np.nan,
            "center_x": np.nan,
            "center_y": np.nan,
            "x1": np.nan,
            "y1": np.nan,
            "x2": np.nan,
            "y2": np.nan,
            "error": f"Landmark error: {str(e)}",
        }

    min_x, min_y = np.min(pts, axis=0)
    max_x, max_y = np.max(pts, axis=0)
    raw_w = int(max_x - min_x)
    raw_h = int(max_y - min_y)

    x1 = max(0, int(min_x - PAD_X))
    y1 = max(0, int(min_y - PAD_Y))
    x2 = min(img_w, int(max_x + PAD_X))
    y2 = min(img_h, int(max_y + PAD_Y))

    roi_w = int(x2 - x1)
    roi_h = int(y2 - y1)

    if roi_w <= 0 or roi_h <= 0:
        return {
            "speaker": spk,
            "digit": dig,
            "frame": fname,
            "path": fpath_str,
            "detected": False,
            "roi_width": np.nan,
            "roi_height": np.nan,
            "raw_width": np.nan,
            "raw_height": np.nan,
            "aspect_ratio": np.nan,
            "raw_aspect_ratio": np.nan,
            "center_x": np.nan,
            "center_y": np.nan,
            "x1": np.nan,
            "y1": np.nan,
            "x2": np.nan,
            "y2": np.nan,
            "error": "Invalid ROI bbox coordinates",
        }

    aspect_ratio = float(roi_w) / float(roi_h)
    raw_ar = (float(raw_w) / float(raw_h)) if raw_h > 0 else np.nan
    cx = float(x1 + x2) / 2.0
    cy = float(y1 + y2) / 2.0

    return {
        "speaker": spk,
        "digit": dig,
        "frame": fname,
        "path": fpath_str,
        "detected": True,
        "roi_width": roi_w,
        "roi_height": roi_h,
        "raw_width": raw_w,
        "raw_height": raw_h,
        "aspect_ratio": aspect_ratio,
        "raw_aspect_ratio": raw_ar,
        "center_x": cx,
        "center_y": cy,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "error": None,
    }


def main():
    parser = argparse.ArgumentParser(description="Full Dlib Lip ROI Geometry Analysis")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker processes")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Discover all frames in order
    frame_tasks: List[Tuple[str, str, str, str]] = []
    for sp_idx in range(1, 31):
        spk = f"s{sp_idx}"
        for dg_idx in range(11):
            dig = f"d{dg_idx}"
            dig_dir = DATA_ROOT / spk / dig
            if not dig_dir.exists():
                continue
            frames = sorted(
                dig_dir.glob("frame_*.jpg"),
                key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)) if re.search(r"(\d+)", p.stem) else 0,
            )
            for f in frames:
                frame_tasks.append((spk, dig, f.name, str(f)))

    print(f"Total frames gathered for Dlib analysis: {len(frame_tasks)} across 330 utterances.")

    # 2. Multiprocessing execution
    results = []
    print(f"Executing with {args.workers} worker processes...")
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(str(MODEL_PATH),)) as executor:
        for res in tqdm(executor.map(process_frame, frame_tasks, chunksize=50), total=len(frame_tasks), desc="Analyzing Frames"):
            results.append(res)

    df = pd.DataFrame(results)

    # Save master per-frame CSV
    csv_path = RESULTS_DIR / "dlib_roi_geometry_all_frames.csv"
    df.to_csv(csv_path, index=False)
    print(f"Master per-frame CSV saved to: {csv_path}")

    # 3. Overall Statistics
    total_frames = len(df)
    successful = df[df["detected"] == True]
    failed = df[df["detected"] == False]

    print("\n" + "=" * 75)
    print("                    DLIB LIP ROI GEOMETRY ANALYSIS REPORT")
    print("=" * 75)
    print(f"Total Frames Analyzed     : {total_frames:,}")
    print(f"Successful Detections     : {len(successful):,} ({len(successful)/total_frames*100:.2f}%)")
    print(f"Failed Detections         : {len(failed):,} ({len(failed)/total_frames*100:.2f}%)")

    w = successful["roi_width"]
    h = successful["roi_height"]
    ar = successful["aspect_ratio"]
    raw_ar = successful["raw_aspect_ratio"]

    def calc_stats(series: pd.Series) -> Dict[str, float]:
        return {
            "min": float(series.min()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()),
            "max": float(series.max()),
        }

    w_stats = calc_stats(w)
    h_stats = calc_stats(h)
    ar_stats = calc_stats(ar)
    raw_ar_stats = calc_stats(raw_ar)

    percentiles = [5, 25, 50, 75, 95]
    ar_pct = np.percentile(ar, percentiles)
    raw_ar_pct = np.percentile(raw_ar.dropna(), percentiles)

    print("\n1. OVERALL METRIC STATISTICS (Padded ROI: Pad_X=15, Pad_Y=10):")
    print("-" * 75)
    print(f"Width  (px) : min={w_stats['min']:.1f}, mean={w_stats['mean']:.2f}, median={w_stats['median']:.1f}, std={w_stats['std']:.2f}, max={w_stats['max']:.1f}")
    print(f"Height (px) : min={h_stats['min']:.1f}, mean={h_stats['mean']:.2f}, median={h_stats['median']:.1f}, std={h_stats['std']:.2f}, max={h_stats['max']:.1f}")
    print(f"Aspect Ratio: min={ar_stats['min']:.2f}, mean={ar_stats['mean']:.2f}, median={ar_stats['median']:.2f}, std={ar_stats['std']:.2f}, max={ar_stats['max']:.2f}")
    print(f"  AR Percentiles: P5={ar_pct[0]:.2f}, P25={ar_pct[1]:.2f}, P50={ar_pct[2]:.2f}, P75={ar_pct[3]:.2f}, P95={ar_pct[4]:.2f}")
    print(f"Raw AR (unpadded): mean={raw_ar_stats['mean']:.2f}, median={raw_ar_stats['median']:.2f}, P5={raw_ar_pct[0]:.2f}, P95={raw_ar_pct[4]:.2f}")

    # 4. Per-speaker statistics
    speaker_stats = (
        successful.groupby("speaker")
        .agg(
            frames=("frame", "count"),
            w_mean=("roi_width", "mean"),
            w_std=("roi_width", "std"),
            h_mean=("roi_height", "mean"),
            h_std=("roi_height", "std"),
            ar_mean=("aspect_ratio", "mean"),
            ar_median=("aspect_ratio", "median"),
            ar_std=("aspect_ratio", "std"),
        )
        .reindex([f"s{i}" for i in range(1, 31)])
    )
    spk_csv_path = RESULTS_DIR / "dlib_roi_per_speaker_stats.csv"
    speaker_stats.to_csv(spk_csv_path)
    print(f"\nPer-speaker stats saved to: {spk_csv_path}")

    # 5. Per-digit statistics
    digit_stats = (
        successful.groupby("digit")
        .agg(
            frames=("frame", "count"),
            w_mean=("roi_width", "mean"),
            w_std=("roi_width", "std"),
            h_mean=("roi_height", "mean"),
            h_std=("roi_height", "std"),
            ar_mean=("aspect_ratio", "mean"),
            ar_median=("aspect_ratio", "median"),
            ar_std=("aspect_ratio", "std"),
        )
        .reindex([f"d{i}" for i in range(11)])
    )
    dig_csv_path = RESULTS_DIR / "dlib_roi_per_digit_stats.csv"
    digit_stats.to_csv(dig_csv_path)
    print(f"Per-digit stats saved to: {dig_csv_path}")

    # 6. Temporal Stability Analysis per Utterance (Speaker x Digit)
    temporal_stability = (
        successful.groupby(["speaker", "digit"])
        .agg(
            frames=("frame", "count"),
            w_mean=("roi_width", "mean"),
            w_std=("roi_width", "std"),
            h_mean=("roi_height", "mean"),
            h_std=("roi_height", "std"),
            ar_mean=("aspect_ratio", "mean"),
            ar_std=("aspect_ratio", "std"),
            cx_mean=("center_x", "mean"),
            cx_std=("center_x", "std"),
            cy_mean=("center_y", "mean"),
            cy_std=("center_y", "std"),
        )
        .reset_index()
    )
    temp_csv_path = RESULTS_DIR / "dlib_roi_temporal_stability.csv"
    temporal_stability.to_csv(temp_csv_path, index=False)
    print(f"Temporal stability stats saved to: {temp_csv_path}")

    print("\n2. TEMPORAL STABILITY SUMMARY ACROSS 330 UTTERANCES:")
    print("-" * 75)
    print(f"Average within-utterance ROI Width  std : {temporal_stability['w_std'].mean():.2f} px")
    print(f"Average within-utterance ROI Height std : {temporal_stability['h_std'].mean():.2f} px")
    print(f"Average within-utterance Aspect Ratio std : {temporal_stability['ar_std'].mean():.2f}")
    print(f"Average within-utterance Center X std   : {temporal_stability['cx_std'].mean():.2f} px")
    print(f"Average within-utterance Center Y std   : {temporal_stability['cy_std'].mean():.2f} px")

    # 7. Generate Visual Plots
    print("\nGenerating visual distribution plots...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # (a) Aspect Ratio Distribution
    ax_ar = axes[0, 0]
    ax_ar.hist(ar, bins=60, color="#1f77b4", edgecolor="black", alpha=0.75, density=True)
    ax_ar.axvline(ar_stats["mean"], color="red", linestyle="--", linewidth=2, label=f"Mean: {ar_stats['mean']:.2f}")
    ax_ar.axvline(ar_stats["median"], color="green", linestyle="-", linewidth=2, label=f"Median: {ar_stats['median']:.2f}")
    ax_ar.axvline(1.0, color="purple", linestyle=":", linewidth=2, label="Square (1.0)")
    ax_ar.set_title("Padded Mouth ROI Aspect Ratio (Width / Height)", fontsize=13, fontweight="bold")
    ax_ar.set_xlabel("Aspect Ratio", fontsize=11)
    ax_ar.set_ylabel("Density", fontsize=11)
    ax_ar.legend(fontsize=10)

    # (b) Width vs Height Distributions
    ax_dim = axes[0, 1]
    ax_dim.hist(w, bins=50, color="#2ca02c", edgecolor="black", alpha=0.6, label="ROI Width")
    ax_dim.hist(h, bins=50, color="#ff7f0e", edgecolor="black", alpha=0.6, label="ROI Height")
    ax_dim.set_title("Mouth ROI Dimensions Distribution (Pixels)", fontsize=13, fontweight="bold")
    ax_dim.set_xlabel("Pixels", fontsize=11)
    ax_dim.set_ylabel("Frame Count", fontsize=11)
    ax_dim.legend(fontsize=10)

    # (c) ROI Center (X, Y) Spatial Distribution
    ax_cnt = axes[1, 0]
    hb = ax_cnt.hexbin(successful["center_x"], successful["center_y"], gridsize=40, cmap="viridis", mincnt=1)
    fig.colorbar(hb, ax=ax_cnt, label="Frame Count")
    ax_cnt.set_title("Mouth ROI Center Coordinates Distribution", fontsize=13, fontweight="bold")
    ax_cnt.set_xlabel("Center X (px)", fontsize=11)
    ax_cnt.set_ylabel("Center Y (px)", fontsize=11)
    ax_cnt.invert_yaxis()  # Image coordinate system

    # (d) Per-Speaker Aspect Ratio Boxplot
    ax_box = axes[1, 1]
    speaker_data = [successful[successful["speaker"] == f"s{i}"]["aspect_ratio"].values for i in range(1, 31)]
    ax_box.boxplot(speaker_data, tick_labels=[f"s{i}" for i in range(1, 31)], showfliers=False)
    ax_box.axhline(ar_stats["median"], color="red", linestyle="--", linewidth=1.5, label=f"Global Median ({ar_stats['median']:.2f})")
    ax_box.axhline(1.0, color="purple", linestyle=":", linewidth=1.5, label="Square 1.0")
    ax_box.set_title("Aspect Ratio Variation Across All 30 Speakers", fontsize=13, fontweight="bold")
    ax_box.set_xlabel("Speaker", fontsize=11)
    ax_box.set_ylabel("Aspect Ratio", fontsize=11)
    ax_box.tick_params(axis="x", rotation=90)
    ax_box.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    plots_path = RESULTS_DIR / "dlib_roi_geometry_distributions.png"
    plt.savefig(plots_path, dpi=200)
    plt.close()
    print(f"Distribution plots saved to: {plots_path}")


if __name__ == "__main__":
    main()
