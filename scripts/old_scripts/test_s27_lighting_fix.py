#!/usr/bin/env python3
"""
scripts/test_s27_lighting_fix.py

Tests CLAHE (on L channel of LAB) + Gamma correction on speaker s27 frames
to restore overexposed bottom-lip boundary contrast.
Saves a 2x5 visual comparison figure to results/dlib_augmented_cropped/s27_lip_recovery_test.png.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_clahe_lab(bgr_img: np.ndarray, clip_limit: float = 2.0, tile_grid: tuple = (8, 8)) -> np.ndarray:
    """Apply CLAHE to the L channel of LAB — preserves color balance, corrects local exposure."""
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

def adjust_gamma(bgr_img: np.ndarray, gamma: float = 0.90) -> np.ndarray:
    """Darken bright regions (gamma < 1.0) to recover boundary contrast."""
    inv = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(bgr_img, table)

def enhance_lip_exposure(bgr_img: np.ndarray, clip_limit: float = 2.0, gamma: float = 0.90) -> np.ndarray:
    """Combines CLAHE on LAB L-channel followed by Gamma correction."""
    return adjust_gamma(apply_clahe_lab(bgr_img, clip_limit=clip_limit), gamma=gamma)

def main():
    project_root = Path(__file__).resolve().parent.parent
    
    # Try pre-cropped Method D folder first, fallback to raw digit folder
    sample_dir = project_root / "data" / "experiments" / "cropped_lips_dataset_d_96x96" / "s27" / "d4"
    if not sample_dir.exists():
        sample_dir = project_root / "data" / "digit" / "s27" / "d4"
    if not sample_dir.exists():
        sample_dir = project_root / "data" / "experiments" / "digit" / "s27" / "d4"
        
    if not sample_dir.exists():
        print(f"[ERROR] Could not find s27/d4 frames directory. Checked {sample_dir}")
        sys.exit(1)

    all_frames = sorted(sample_dir.glob("*.jpg"))
    if not all_frames:
        all_frames = sorted(sample_dir.glob("*.png"))

    if len(all_frames) < 5:
        print(f"[ERROR] Found only {len(all_frames)} frames in {sample_dir}")
        sys.exit(1)

    # Pick 5 uniformly spaced sample frames
    indices = np.linspace(0, len(all_frames) - 1, 5, dtype=int)
    sample_frames = [all_frames[i] for i in indices]

    out_dir = project_root / "results" / "dlib_augmented_cropped"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_plot_path = out_dir / "s27_lip_recovery_test.png"

    fig, axes = plt.subplots(2, 5, figsize=(18, 7))

    for i, fp in enumerate(sample_frames):
        bgr = cv2.imread(str(fp))
        if bgr.shape[:2] != (96, 96):
            bgr = cv2.resize(bgr, (96, 96), interpolation=cv2.INTER_AREA)

        corrected = enhance_lip_exposure(bgr, clip_limit=2.0, gamma=0.90)

        rgb_before = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_after = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)

        axes[0, i].imshow(rgb_before)
        axes[0, i].set_title(f"Before: Frame {fp.stem}", fontsize=11)
        axes[0, i].axis("off")

        axes[1, i].imshow(rgb_after)
        axes[1, i].set_title("After CLAHE + Gamma (0.9)", fontsize=11, fontweight="bold", color="darkgreen")
        axes[1, i].axis("off")

    plt.suptitle("Speaker s27 Lip Recovery Test (Overhead Lighting & Chin Separation Fix)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_plot_path, dpi=200)
    plt.close()

    print("=" * 80)
    print(f"[SUCCESS] Visual comparison test generated!")
    print(f"Comparison plot saved to: {out_plot_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
