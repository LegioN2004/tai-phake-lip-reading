#!/usr/bin/env python3
"""
Switch the active dataset inside data/processed/ (train, val, test).

Usage:
    python3 scripts/switch_dataset.py [DATASET_NAME]

Example:
    python3 scripts/switch_dataset.py cropped_lips_dataset_a_96x96
    or simply edit DATASET_NAME below and run:
    python3 scripts/switch_dataset.py
"""

import sys
import shutil
from pathlib import Path

# ==============================================================================
# 1. EDIT THIS VARIABLE OR PASS IT AS A COMMAND LINE ARGUMENT
# ==============================================================================
DATASET_NAME = "augmented_100spk_dataset_d"
# ==============================================================================

# Fixed canonical speakers
VAL_SPEAKERS = {"s12", "s19", "s21", "s29"}
TEST_SPEAKERS = {"s9", "s10", "s16", "s18", "s24", "s28"}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "data" / "experiments"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_DIR = PROCESSED_DIR / "train"
VAL_DIR = PROCESSED_DIR / "val"
TEST_DIR = PROCESSED_DIR / "test"


def clear_split_dir(d: Path):
    """Safely clear symlinks/subdirs/files from a split folder."""
    d.mkdir(parents=True, exist_ok=True)
    for item in d.iterdir():
        if item.name.startswith("."):
            continue
        if item.is_symlink() or item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def main():
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else DATASET_NAME
    source_dir = EXPERIMENTS_DIR / dataset_name

    # Fallback to data/<dataset_name> if not directly in data/experiments
    if not source_dir.exists():
        fallback = PROJECT_ROOT / "data" / dataset_name
        if fallback.exists():
            source_dir = fallback
        else:
            print(f"Error: Dataset folder not found at '{source_dir}' or '{fallback}'")
            print("Available datasets in data/experiments:")
            for p in sorted(EXPERIMENTS_DIR.iterdir()):
                if p.is_dir() and not p.name.startswith("."):
                    print(f"  - {p.name}")
            sys.exit(1)

    print(f"Switching active dataset to: {dataset_name}")
    print(f"Source: {source_dir}")

    # 1. Clear existing splits
    print("Clearing data/processed/{train, val, test}...")
    for split_dir in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        clear_split_dir(split_dir)

    # Clean any loose symlinks at the root of data/processed
    for item in PROCESSED_DIR.iterdir():
        if item.is_symlink() and item.name not in {"train", "val", "test"}:
            item.unlink()

    # 2. Discover speakers in source directory
    speaker_dirs = sorted([
        d for d in source_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ], key=lambda x: x.name)

    if not speaker_dirs:
        print(f"Error: No speaker subfolders found in {source_dir}")
        sys.exit(1)

    counts = {"train": 0, "val": 0, "test": 0}

    # 3. Create instantaneous symlinks per speaker
    for spk_path in speaker_dirs:
        spk_name = spk_path.name

        # Extract base speaker ID (e.g. 's4_aug1' -> base 's4')
        base_spk = spk_name.split("_")[0]

        if base_spk in TEST_SPEAKERS:
            target_link = TEST_DIR / spk_name
            target_link.symlink_to(spk_path.resolve())
            counts["test"] += 1
        elif base_spk in VAL_SPEAKERS:
            target_link = VAL_DIR / spk_name
            target_link.symlink_to(spk_path.resolve())
            counts["val"] += 1
        else:
            target_link = TRAIN_DIR / spk_name
            target_link.symlink_to(spk_path.resolve())
            counts["train"] += 1

    print("\nDataset successfully linked!")
    print(f"  Train: {counts['train']} speakers -> {TRAIN_DIR}")
    print(f"  Val:   {counts['val']} speakers (s12, s19, s21, s29) -> {VAL_DIR}")
    print(f"  Test:  {counts['test']} speakers (s9, s10, s16, s18, s24, s28) -> {TEST_DIR}")
    print("\nYou can now open 05_model_training_baseline.ipynb or 06_pretrained_phase1_baseline.ipynb and click 'Run All'!")


if __name__ == "__main__":
    main()
