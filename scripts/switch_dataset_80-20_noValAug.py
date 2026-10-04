#!/usr/bin/env python3
"""
Switch the active dataset inside data/processed/ into an 80/20 base-speaker split
WITHOUT validation augmentations (original-only validation):
    - 24 base speakers -> Train (including original + all their augmented folders)
    - 6 base speakers  -> Validation (ORIGINAL un-augmented folders ONLY)
    - 0 speakers       -> Test (ignored, kept completely empty)

ONE BASE SPEAKER = ONE SPLIT.
Validation speakers have strictly ZERO augmented folders linked into data/processed/val/.
All augmented validation folders (e.g. s25_aug1, s25_aug2...) are skipped.
Training speakers retain both original and all augmented folders.
Zero speaker leakage between train and validation.

Usage:
    python3 scripts/switch_dataset_80-20_noValAug.py [DATASET_NAME]

Example:
    python3 scripts/switch_dataset_80-20_noValAug.py correctly_augmented_cropped_lips_d_96x96
    or simply edit DATASET_NAME below and run:
    python3 scripts/switch_dataset_80-20_noValAug.py
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

# ==============================================================================
# 1. EDIT THIS VARIABLE OR PASS IT AS A COMMAND LINE ARGUMENT
# ==============================================================================
DATASET_NAME = "correctly_augmented_cropped_lips_d_96x96"
# ==============================================================================

# ==============================================================================
# 2. 80/20 BASE SPEAKER SPLIT DEFINITIONS (30 BASE SPEAKERS TOTAL)
# Exactly 6 base speakers for validation (20% of 30 base speakers).
# The remaining 24 base speakers (80% of 30 base speakers) automatically go to training.
# Easily customizable: edit VAL_SPEAKERS below to adjust the split.
# ==============================================================================
VAL_SPEAKERS = {"s25", "s26", "s27", "s28", "s29", "s30"}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "data" / "experiments"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_DIR = PROCESSED_DIR / "train"
VAL_DIR = PROCESSED_DIR / "val"
TEST_DIR = PROCESSED_DIR / "test"


def clear_split_dir(d: Path) -> None:
    """Safely and completely clear all symlinks, directories, and files from a folder."""
    d.mkdir(parents=True, exist_ok=True)
    for item in d.iterdir():
        if item.name.startswith("."):
            continue
        if item.is_symlink() or item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def natural_sort_key(s: str) -> int:
    """Sort speaker identifiers naturally (e.g. s1, s2, ..., s10, s30)."""
    m = re.search(r"\d+", s)
    return int(m.group(0)) if m else 0


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
            if EXPERIMENTS_DIR.exists():
                for p in sorted(EXPERIMENTS_DIR.iterdir()):
                    if p.is_dir() and not p.name.startswith("."):
                        print(f"  - {p.name}")
            sys.exit(1)

    print("=" * 80)
    print("  SWITCH ACTIVE DATASET: 80/20 BASE SPEAKER SPLIT [ORIGINAL-ONLY VALIDATION]")
    print("=" * 80)
    print(f"Active Dataset : {dataset_name}")
    print(f"Source Path    : {source_dir}")
    print("-" * 80)

    # 1. Completely clear all three processed folders first
    print("Clearing data/processed/{train, val, test}...")
    clear_split_dir(TRAIN_DIR)
    clear_split_dir(VAL_DIR)
    clear_split_dir(TEST_DIR)

    # Clean any loose symlinks at the root of data/processed
    if PROCESSED_DIR.exists():
        for item in PROCESSED_DIR.iterdir():
            if item.is_symlink() and item.name not in {"train", "val", "test"}:
                item.unlink()

    # Recreate all three split directories
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Discover all speaker directories in source dataset
    speaker_dirs = sorted([
        d for d in source_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ], key=lambda x: natural_sort_key(x.name))

    if not speaker_dirs:
        print(f"Error: No speaker subfolders found in {source_dir}")
        sys.exit(1)

    # Discover all unique base speakers (portion before '_')
    all_base_speakers = sorted(
        list({d.name.split("_")[0] for d in speaker_dirs}),
        key=natural_sort_key,
    )

    # Verify exactly 30 base speakers exist (s1 through s30)
    if len(all_base_speakers) != 30:
        print(f"Error: Expected 30 base speakers in dataset, but discovered {len(all_base_speakers)}.")
        print(f"Discovered base speakers: {all_base_speakers}")
        sys.exit(1)

    missing_val = VAL_SPEAKERS - set(all_base_speakers)
    if missing_val:
        print(f"Error: {len(missing_val)} configured validation speakers not found in dataset:")
        print(f"  Missing: {sorted(missing_val, key=natural_sort_key)}")
        sys.exit(1)

    train_base_speakers = sorted(
        [s for s in all_base_speakers if s not in VAL_SPEAKERS],
        key=natural_sort_key,
    )
    val_base_speakers = sorted(
        [s for s in all_base_speakers if s in VAL_SPEAKERS],
        key=natural_sort_key,
    )

    counts = {
        "train_dirs": 0,
        "val_dirs": 0,
        "skipped_val_augs": 0,
    }
    skipped_val_aug_names = []

    # 3. Create symlinks per speaker
    # Train: original + all augmented folders for train base speakers
    # Val: ONLY original un-augmented folders for validation base speakers
    for spk_path in speaker_dirs:
        spk_name = spk_path.name
        base_spk = spk_name.split("_")[0]

        if base_spk in VAL_SPEAKERS:
            # Check if this is an original or augmented directory
            if "_" not in spk_name:
                target_link = VAL_DIR / spk_name
                target_link.symlink_to(spk_path.resolve())
                counts["val_dirs"] += 1
            else:
                counts["skipped_val_augs"] += 1
                skipped_val_aug_names.append(spk_name)
        else:
            target_link = TRAIN_DIR / spk_name
            target_link.symlink_to(spk_path.resolve())
            counts["train_dirs"] += 1

    # 4. Strict Validation Checks
    print("\nExecuting strict zero-leakage validation checks...")

    # Check 1: Exactly 30 base speakers discovered
    assert len(all_base_speakers) == 30, (
        f"Validation failed: Expected 30 base speakers, found {len(all_base_speakers)}"
    )

    # Check 2: Exactly 6 base speakers in validation
    assert len(val_base_speakers) == 6, (
        f"Validation failed: Expected 6 base val speakers, got {len(val_base_speakers)}"
    )

    # Check 3: Exactly 24 base speakers in training
    assert len(train_base_speakers) == 24, (
        f"Validation failed: Expected 24 base train speakers, got {len(train_base_speakers)}"
    )

    # Check 4: No base speaker exists in both splits
    overlap = set(train_base_speakers).intersection(set(val_base_speakers))
    assert len(overlap) == 0, f"Validation failed: Speaker leakage detected! Overlap: {overlap}"

    # Check 5: Every directory belonging to a train base speaker is in train; val has only original
    train_dir_names = [d.name for d in TRAIN_DIR.iterdir() if not d.name.startswith(".")]
    val_dir_names = [d.name for d in VAL_DIR.iterdir() if not d.name.startswith(".")]

    train_dir_bases = {d.split("_")[0] for d in train_dir_names}
    val_dir_bases = {d.split("_")[0] for d in val_dir_names}

    assert train_dir_bases == set(train_base_speakers), (
        f"Mismatch between train directory base speakers and expected train base speakers"
    )
    assert val_dir_bases == set(val_base_speakers), (
        f"Mismatch between val directory base speakers and expected val base speakers"
    )

    # Check 6: Strictly NO augmented folders in validation
    val_augs = [d for d in val_dir_names if "_" in d]
    assert len(val_augs) == 0, (
        f"Validation failed: Augmented folders found in validation directory: {val_augs}"
    )

    # Check 7: Exactly 6 validation directories (one original per val speaker)
    assert len(val_dir_names) == len(val_base_speakers) == 6, (
        f"Validation failed: Expected exactly 6 validation directories, found {len(val_dir_names)}: {val_dir_names}"
    )

    # Check 8: Zero cross-leakage of base or augmented folders
    leaked_val_augs = [d for d in train_dir_names if d.split("_")[0] in VAL_SPEAKERS]
    assert len(leaked_val_augs) == 0, (
        f"Validation failed: Validation speaker variants leaked into training: {leaked_val_augs}"
    )

    leaked_train_augs = [d for d in val_dir_names if d.split("_")[0] not in VAL_SPEAKERS]
    assert len(leaked_train_augs) == 0, (
        f"Validation failed: Training speaker variants leaked into validation: {leaked_train_augs}"
    )

    # Check 9: data/processed/test/ is completely empty
    test_items = [item.name for item in TEST_DIR.iterdir() if not item.name.startswith(".")]
    assert len(test_items) == 0, (
        f"Validation failed: Test directory is not empty! Contains: {test_items}"
    )

    print("[SUCCESS] All strict zero-leakage assertions passed successfully!")
    print(f"  [x] Train base speakers: 24 ({len(train_dir_names)} directories with aug)")
    print(f"  [x] Val base speakers:   6 ({len(val_dir_names)} original directories)")
    print(f"  [x] Skipped validation augmented folders: {counts['skipped_val_augs']}")
    print(f"  [x] Test directory:      0 items (cleanly empty)")

    # 5. Output Summary
    print("\n" + "=" * 80)
    print("Training base speakers (24):")
    print(", ".join(train_base_speakers))
    print("\nValidation base speakers (6) [Original Only]:")
    print(", ".join(val_base_speakers))
    if skipped_val_aug_names:
        print("\nSkipped validation augmented folders (not copied):")
        print(", ".join(sorted(skipped_val_aug_names, key=natural_sort_key)))
    print("=" * 80)

    print("\nDataset successfully linked!")
    print("\nBase speakers:")
    print(f"  Total: {len(all_base_speakers)}")
    print(f"  Train: {len(train_base_speakers)}")
    print(f"  Val:   {len(val_base_speakers)}")
    print(f"  Test:  0")

    print("\nSpeaker directories:")
    print(f"  Train: {counts['train_dirs']}")
    print(f"  Val:   {counts['val_dirs']}")
    print(f"  Test:  {len(test_items)}")


if __name__ == "__main__":
    main()
