#!/usr/bin/env python3
"""
Small Preprocessing Comparison Experiment for Dlib Lip ROI:
Compares Method A, Method B, and Method C on 5 representative speakers x 11 digits x 10 evenly spaced frames.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_ROOT = Path("data/digit")
MODEL_PATH = Path("models/dlib/shape_predictor_68_face_landmarks.dat")
EXP_ROOT = Path("results/dlib/comparison_experiment")

# Output directories for each method
DIR_A = EXP_ROOT / "method_a_direct_96x96"
DIR_B = EXP_ROOT / "method_b_square_96x96"
DIR_C = EXP_ROOT / "method_c_aspect_preserving_128x64"
DIR_SHEETS = EXP_ROOT / "contact_sheets"
DIR_SEQS = EXP_ROOT / "temporal_sequences"

# 5 representative speakers across gender/recording variations and all 11 digits
SPEAKERS = ["s1", "s6", "s12", "s18", "s25"]
DIGITS = [f"d{i}" for i in range(11)]
FRAMES_PER_UTTERANCE = 10

# Measured dataset aspect ratio: median = 1.98 -> 2.0:1 (e.g., 128x64)
TARGET_W_C = 128
TARGET_H_C = 64

PAD_X = 15
PAD_Y = 10


def init_dlib():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(MODEL_PATH))
    return detector, predictor


def extract_landmarks(img, detector, predictor):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    faces = detector(gray_eq, 0)
    if not faces:
        faces = detector(gray_eq, 1)
    if not faces:
        return None, None
    largest_face = max(faces, key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top()))
    shape = predictor(gray, largest_face)
    pts = np.array([[shape.part(i).x, shape.part(i).y] for i in range(48, 68)], dtype=np.int32)
    return pts, largest_face


def crop_method_a(img, pts):
    """Method A: Current direct resize of padded mouth bbox to 96x96."""
    h_img, w_img = img.shape[:2]
    min_x, min_y = np.min(pts, axis=0)
    max_x, max_y = np.max(pts, axis=0)

    x1 = max(0, int(min_x - PAD_X))
    y1 = max(0, int(min_y - PAD_Y))
    x2 = min(w_img, int(max_x + PAD_X))
    y2 = min(h_img, int(max_y + PAD_Y))

    crop = img[y1:y2, x1:x2]
    resized = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA)
    return resized, (x1, y1, x2, y2), (x2 - x1), (y2 - y1)


def crop_method_b(img, pts):
    """Method B: Square source crop centered on mouth, resized isotropically to 96x96."""
    h_img, w_img = img.shape[:2]
    min_x, min_y = np.min(pts, axis=0)
    max_x, max_y = np.max(pts, axis=0)

    # Base mouth width & height with current padding
    w_mouth = (max_x - min_x) + 2 * PAD_X
    h_mouth = (max_y - min_y) + 2 * PAD_Y

    # Mouth center
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0

    # Square dimension: enough to fully cover the padded mouth width
    # With a small margin (e.g. max(w_mouth, h_mouth) * 1.05)
    side = int(round(max(w_mouth, h_mouth) * 1.05))

    x1 = int(round(cx - side / 2.0))
    y1 = int(round(cy - side / 2.0))
    x2 = x1 + side
    y2 = y1 + side

    # Safe padding if square extends beyond image bounds
    pad_top = max(0, -y1)
    pad_left = max(0, -x1)
    pad_bottom = max(0, y2 - h_img)
    pad_right = max(0, x2 - w_img)

    if pad_top > 0 or pad_left > 0 or pad_bottom > 0 or pad_right > 0:
        img_padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT)
        crop = img_padded[y1 + pad_top : y2 + pad_top, x1 + pad_left : x2 + pad_left]
    else:
        crop = img[y1:y2, x1:x2]

    resized = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA)
    return resized, (x1, y1, x2, y2), side, side


def crop_method_c(img, pts):
    """Method C: Aspect-preserving rectangular crop (2.0:1) resized to 128x64."""
    h_img, w_img = img.shape[:2]
    min_x, min_y = np.min(pts, axis=0)
    max_x, max_y = np.max(pts, axis=0)

    w_mouth = (max_x - min_x) + 2 * PAD_X
    h_mouth = (max_y - min_y) + 2 * PAD_Y

    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0

    # Target aspect ratio is 2.0 (Width / Height)
    target_ar = 2.0

    # Ensure box bounds both w_mouth and h_mouth while maintaining 2.0:1
    if w_mouth / h_mouth >= target_ar:
        box_w = int(round(w_mouth * 1.05))
        box_h = int(round(box_w / target_ar))
    else:
        box_h = int(round(h_mouth * 1.05))
        box_w = int(round(box_h * target_ar))

    x1 = int(round(cx - box_w / 2.0))
    y1 = int(round(cy - box_h / 2.0))
    x2 = x1 + box_w
    y2 = y1 + box_h

    pad_top = max(0, -y1)
    pad_left = max(0, -x1)
    pad_bottom = max(0, y2 - h_img)
    pad_right = max(0, x2 - w_img)

    if pad_top > 0 or pad_left > 0 or pad_bottom > 0 or pad_right > 0:
        img_padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT)
        crop = img_padded[y1 + pad_top : y2 + pad_top, x1 + pad_left : x2 + pad_left]
    else:
        crop = img[y1:y2, x1:x2]

    resized = cv2.resize(crop, (TARGET_W_C, TARGET_H_C), interpolation=cv2.INTER_AREA)
    return resized, (x1, y1, x2, y2), box_w, box_h


def select_evenly_spaced_frames(frame_list: List[Path], n: int = 10) -> List[Path]:
    total = len(frame_list)
    if total <= n:
        return frame_list
    indices = np.linspace(0, total - 1, n, dtype=int)
    return [frame_list[i] for i in indices]


def main():
    for d in [DIR_A, DIR_B, DIR_C, DIR_SHEETS, DIR_SEQS]:
        d.mkdir(parents=True, exist_ok=True)

    detector, predictor = init_dlib()

    experiment_records = []

    print(f"Running comparison on {len(SPEAKERS)} speakers x {len(DIGITS)} digits = 55 utterances (10 frames each)...")

    temporal_eval_samples = [("s1", "d0"), ("s6", "d5"), ("s18", "d8"), ("s25", "d2")]

    for spk in tqdm(SPEAKERS, desc="Speakers"):
        for dig in DIGITS:
            dig_dir = DATA_ROOT / spk / dig
            frame_files = sorted(
                dig_dir.glob("frame_*.jpg"),
                key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)) if re.search(r"(\d+)", p.stem) else 0,
            )
            sampled_frames = select_evenly_spaced_frames(frame_files, FRAMES_PER_UTTERANCE)

            seq_frames_a = []
            seq_frames_b = []
            seq_frames_c = []

            for frame_path in sampled_frames:
                img = cv2.imread(str(frame_path))
                if img is None:
                    continue

                pts, face_rect = extract_landmarks(img, detector, predictor)
                if pts is None:
                    experiment_records.append({
                        "speaker": spk,
                        "digit": dig,
                        "frame": frame_path.name,
                        "detected": False,
                    })
                    continue

                # Run Method A, B, C
                crop_a, bbox_a, w_a, h_a = crop_method_a(img, pts)
                crop_b, bbox_b, w_b, h_b = crop_method_b(img, pts)
                crop_c, bbox_c, w_c, h_c = crop_method_c(img, pts)

                # Save cropped images
                sub_dir_a = DIR_A / spk / dig
                sub_dir_b = DIR_B / spk / dig
                sub_dir_c = DIR_C / spk / dig
                sub_dir_a.mkdir(parents=True, exist_ok=True)
                sub_dir_b.mkdir(parents=True, exist_ok=True)
                sub_dir_c.mkdir(parents=True, exist_ok=True)

                cv2.imwrite(str(sub_dir_a / frame_path.name), crop_a)
                cv2.imwrite(str(sub_dir_b / frame_path.name), crop_b)
                cv2.imwrite(str(sub_dir_c / frame_path.name), crop_c)

                seq_frames_a.append(crop_a)
                seq_frames_b.append(crop_b)
                seq_frames_c.append(crop_c)

                source_ar = float(w_a) / float(h_a)

                experiment_records.append({
                    "speaker": spk,
                    "digit": dig,
                    "frame": frame_path.name,
                    "detected": True,
                    "source_w": w_a,
                    "source_h": h_a,
                    "source_ar": source_ar,
                    "method_a_dim": "96x96",
                    "method_a_distortion": f"sx={96/w_a:.2f}, sy={96/h_a:.2f} (stretching ratio={((96/h_a)/(96/w_a)):.2f})",
                    "method_b_dim": "96x96",
                    "method_b_distortion": "None (isotropic)",
                    "method_c_dim": f"{TARGET_W_C}x{TARGET_H_C}",
                    "method_c_distortion": "None (aspect-preserving 2.0:1)",
                })

            # Generate Temporal Sequence Strip for selected representative utterances
            if (spk, dig) in temporal_eval_samples and len(seq_frames_a) == 10:
                fig, axes = plt.subplots(3, 10, figsize=(20, 6.5))
                for t in range(10):
                    # Method A (row 0)
                    axes[0, t].imshow(cv2.cvtColor(seq_frames_a[t], cv2.COLOR_BGR2RGB))
                    axes[0, t].set_title(f"t={t+1}", fontsize=10)
                    axes[0, t].axis("off")

                    # Method B (row 1)
                    axes[1, t].imshow(cv2.cvtColor(seq_frames_b[t], cv2.COLOR_BGR2RGB))
                    axes[1, t].axis("off")

                    # Method C (row 2)
                    axes[2, t].imshow(cv2.cvtColor(seq_frames_c[t], cv2.COLOR_BGR2RGB))
                    axes[2, t].axis("off")

                axes[0, 0].set_ylabel("Method A\nDirect 96x96\n(Stretched)", fontsize=11, fontweight="bold")
                axes[1, 0].set_ylabel("Method B\nSquare ROI 96x96\n(Isotropic)", fontsize=11, fontweight="bold")
                axes[2, 0].set_ylabel("Method C\nRect 128x64\n(2.0:1 Isotropic)", fontsize=11, fontweight="bold")
                plt.suptitle(f"Temporal Stability Comparison: Utterance {spk}/{dig} (10 Frames)", fontsize=14, fontweight="bold")
                plt.tight_layout()
                plt.savefig(DIR_SEQS / f"temporal_strip_{spk}_{dig}.png", dpi=180)
                plt.close()

    # Generate Contact Sheets showing A vs B vs C side-by-side for random diverse frames
    print("Generating side-by-side contact sheets...")
    df_exp = pd.DataFrame(experiment_records)
    df_exp.to_csv(EXP_ROOT / "comparison_experiment_metrics.csv", index=False)

    valid_samples = df_exp[df_exp["detected"] == True].sample(n=6, random_state=42)
    fig, axes = plt.subplots(6, 3, figsize=(11, 16))

    for row_idx, (_, r) in enumerate(valid_samples.iterrows()):
        spk = r["speaker"]
        dig = r["digit"]
        fname = r["frame"]

        img_a = cv2.imread(str(DIR_A / spk / dig / fname))
        img_b = cv2.imread(str(DIR_B / spk / dig / fname))
        img_c = cv2.imread(str(DIR_C / spk / dig / fname))

        axes[row_idx, 0].imshow(cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB))
        axes[row_idx, 0].set_title(f"A: Direct 96x96 (AR={r['source_ar']:.2f})", fontsize=10)
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB))
        axes[row_idx, 1].set_title("B: Square ROI 96x96", fontsize=10)
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB))
        axes[row_idx, 2].set_title(f"C: Rect 128x64 (2.0:1)", fontsize=10)
        axes[row_idx, 2].axis("off")

    plt.suptitle("Side-by-Side Visual Comparison: Method A vs B vs C", fontsize=14, fontweight="bold")
    plt.tight_layout()
    contact_sheet_path = DIR_SHEETS / "method_comparison_contact_sheet.png"
    plt.savefig(contact_sheet_path, dpi=200)
    plt.close()
    print(f"Contact sheet saved to: {contact_sheet_path}")
    print(f"Comparison experiment complete! Results saved under: {EXP_ROOT}")


if __name__ == "__main__":
    main()
