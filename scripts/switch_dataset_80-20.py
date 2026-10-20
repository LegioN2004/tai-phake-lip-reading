#!/usr/bin/env python3
"""
Switch the active dataset inside data/processed/ into an 80/20 split:
    - Exactly 80 speaker folders -> data/processed/train (augmented or normal)
    - Exactly 20 speaker folders -> data/processed/val   (augmented or normal)
    - Exactly 0 items            -> data/processed/test  (kept completely clean and empty)

Strictly 100 speaker directories in total.
Zero data leakage: all directories/variants for any base speaker stay strictly in ONE split.
No hardcoded speakers: dynamically discovers, partitions, and verifies the 80/20 split.

Usage:
    python3 scripts/switch_dataset_80-20.py [DATASET_NAME]

Example:
    python3 scripts/switch_dataset_80-20.py dlib_augmented_cropped_100_80_20_96x96
    or simply:
    python3 scripts/switch_dataset_80-20.py
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

# ==============================================================================
# 1. DEFAULT DATASET NAME (OVERRIDABLE VIA CLI ARGUMENT)
# ==============================================================================
DATASET_NAME = "dlib_augmented_cropped_100_80_20_96x96"

# ==============================================================================
# 2. STRICT 80/20 FOLDER CONSTRAINTS (EXACTLY 100 IN TOTAL)
# ==============================================================================
TARGET_TRAIN_FOLDERS = 80
TARGET_VAL_FOLDERS = 20
TOTAL_TARGET_FOLDERS = TARGET_TRAIN_FOLDERS + TARGET_VAL_FOLDERS  # Strictly 100

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


def sort_speaker_variants(dirs: List[str]) -> List[str]:
    """Sort variants for a base speaker: original first (no '_aug'), then aug1, aug2, ..."""
    def var_key(name: str) -> int:
        if "_" not in name:
            return 0
        m = re.search(r"aug(\d+)", name, re.IGNORECASE)
        return int(m.group(1)) if m else 99
    return sorted(dirs, key=var_key)


def compute_80_20_split(
    speaker_dirs: List[Path],
) -> Tuple[List[str], List[str], List[str], List[str], Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Dynamically partitions discovered speaker directories into:
      - exactly 80 train folders
      - exactly 20 val folders
      - strictly 100 in total
    with zero data leakage between base speakers, and NO hardcoded speaker names.
    """
    all_dir_names = [d.name for d in speaker_dirs]

    # Group all directories by base speaker (prefix before '_')
    base_to_dirs: Dict[str, List[str]] = {}
    for dname in all_dir_names:
        base = dname.split("_")[0]
        base_to_dirs.setdefault(base, []).append(dname)

    # Sort variants for each base speaker (original first, then aug1, aug2, ...)
    for base in base_to_dirs:
        base_to_dirs[base] = sort_speaker_variants(base_to_dirs[base])

    all_base_speakers = sorted(base_to_dirs.keys(), key=natural_sort_key)
    num_base = len(all_base_speakers)

    if num_base < 2:
        raise ValueError(f"Need at least 2 base speakers for train/val split, found {num_base}.")

    # 80/20 base speaker split without any hardcoded names
    num_train_base = round(num_base * 0.8)
    num_val_base = num_base - num_train_base

    train_base_speakers = all_base_speakers[:num_train_base]
    val_base_speakers = all_base_speakers[num_train_base:]

    # Allocate folders to reach exactly TARGET_TRAIN_FOLDERS (80)
    extra_train = TARGET_TRAIN_FOLDERS - num_train_base
    base_extra_train = extra_train // num_train_base
    rem_train = extra_train % num_train_base

    train_mapping: Dict[str, List[str]] = {}
    train_folders: List[str] = []
    for idx, spk in enumerate(train_base_speakers):
        target_k = 1 + base_extra_train + (1 if idx < rem_train else 0)
        avail = base_to_dirs[spk]
        chosen = avail[:target_k]
        train_mapping[spk] = chosen
        train_folders.extend(chosen)

    # Allocate folders to reach exactly TARGET_VAL_FOLDERS (20)
    extra_val = TARGET_VAL_FOLDERS - num_val_base
    base_extra_val = extra_val // num_val_base
    rem_val = extra_val % num_val_base

    val_mapping: Dict[str, List[str]] = {}
    val_folders: List[str] = []
    for idx, spk in enumerate(val_base_speakers):
        target_k = 1 + base_extra_val + (1 if idx < rem_val else 0)
        avail = base_to_dirs[spk]
        chosen = avail[:target_k]
        val_mapping[spk] = chosen
        val_folders.extend(chosen)

    return (
        train_base_speakers,
        val_base_speakers,
        train_folders,
        val_folders,
        train_mapping,
        val_mapping,
    )


def resolve_split(
    source_dir: Path,
    speaker_dirs: List[Path],
) -> Tuple[List[str], List[str], List[str], List[str], Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Determines the 80 train and 20 val folder lists:
    1. Checks if a valid, non-stale speaker_split.json exists in source_dir.
    2. Otherwise, dynamically partitions directories using compute_80_20_split.
    """
    actual_name_map = {d.name.lower(): d.name for d in speaker_dirs}
    all_dir_names = set(actual_name_map.values())
    split_meta_file = source_dir / "speaker_split.json"

    if split_meta_file.exists():
        try:
            with open(split_meta_file, "r") as f:
                split_meta = json.load(f)

            raw_train = split_meta.get("train_speakers", [])
            raw_val = split_meta.get("val_speakers", [])

            # Case-resilient directory name mapping
            meta_train = [actual_name_map.get(s.lower(), s) for s in raw_train]
            meta_val = [actual_name_map.get(s.lower(), s) for s in raw_val]

            # Check counts: must be exactly 80 train and 20 val
            if len(meta_train) == TARGET_TRAIN_FOLDERS and len(meta_val) == TARGET_VAL_FOLDERS:
                # Check that all folders actually exist in source_dir
                if all(s in all_dir_names for s in meta_train) and all(s in all_dir_names for s in meta_val):
                    train_bases = {s.split("_")[0] for s in meta_train}
                    val_bases = {s.split("_")[0] for s in meta_val}
                    # Zero leakage check
                    if len(train_bases & val_bases) == 0:
                        print("Loaded verified split configuration from existing speaker_split.json.")
                        train_mapping = {
                            b: [actual_name_map.get(f.lower(), f) for f in dirs]
                            for b, dirs in split_meta.get("train_speaker_mapping", {}).items()
                        }
                        val_mapping = {
                            b: [actual_name_map.get(f.lower(), f) for f in dirs]
                            for b, dirs in split_meta.get("val_speaker_mapping", {}).items()
                        }
                        return (
                            sorted(train_bases, key=natural_sort_key),
                            sorted(val_bases, key=natural_sort_key),
                            meta_train,
                            meta_val,
                            train_mapping,
                            val_mapping,
                        )
                    else:
                        print("Warning: speaker_split.json contained leakage! Recomputing split dynamically...")
                else:
                    print("Warning: speaker_split.json contained missing directories! Recomputing split dynamically...")
            else:
                print(f"Warning: speaker_split.json had counts ({len(meta_train)}/{len(meta_val)}) != (80/20). Recomputing...")
        except Exception as e:
            print(f"Warning: Could not parse {split_meta_file}: {e}. Recomputing dynamically...")

    print("Computing dynamic 80/20 split directly from dataset directories (zero hardcoding)...")
    return compute_80_20_split(speaker_dirs)


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
    print("      SWITCH ACTIVE DATASET: STRICT 80/20 SPLIT (80 TRAIN / 20 VAL = 100 TOTAL)")
    print("=" * 80)
    print(f"Active Dataset : {dataset_name}")
    print(f"Source Path    : {source_dir}")
    print("-" * 80)

    # 1. Discover all speaker directories in source dataset
    speaker_dirs = sorted([
        d for d in source_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ], key=lambda x: natural_sort_key(x.name))

    if not speaker_dirs:
        print(f"Error: No speaker subfolders found in {source_dir}")
        sys.exit(1)

    if len(speaker_dirs) < TOTAL_TARGET_FOLDERS:
        print(
            f"Error: Dataset has {len(speaker_dirs)} directories, but switching requires "
            f"at least {TOTAL_TARGET_FOLDERS} folders (80 train + 20 val)."
        )
        sys.exit(1)

    # 2. Resolve or compute the strict 80/20 split (zero hardcoding)
    (
        train_base_speakers,
        val_base_speakers,
        train_folders,
        val_folders,
        train_mapping,
        val_mapping,
    ) = resolve_split(source_dir, speaker_dirs)

    # 3. Completely clear all three processed folders
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

    # 4. Create symlinks for Train (strictly 80 folders)
    counts = {"train_dirs": 0, "val_dirs": 0}
    for spk_name in train_folders:
        src = source_dir / spk_name
        dst = TRAIN_DIR / spk_name
        dst.symlink_to(src.resolve())
        counts["train_dirs"] += 1

    # 5. Create symlinks for Val (strictly 20 folders)
    for spk_name in val_folders:
        src = source_dir / spk_name
        dst = VAL_DIR / spk_name
        dst.symlink_to(src.resolve())
        counts["val_dirs"] += 1

    # 6. Save / update speaker_split.json metadata for dataset
    speaker_split_data = {
        "dataset_name": dataset_name,
        "dataset_mode": "100_80_20",
        "num_total_folders": TOTAL_TARGET_FOLDERS,
        "num_train_folders": len(train_folders),
        "num_base_train_speakers": len(train_base_speakers),
        "num_val_folders": len(val_folders),
        "num_base_val_speakers": len(val_base_speakers),
        "base_train_speakers": train_base_speakers,
        "train_speakers": train_folders,
        "train_speaker_mapping": train_mapping,
        "base_val_speakers": val_base_speakers,
        "val_speakers": val_folders,
        "val_speaker_mapping": val_mapping,
        "base_test_speakers": [],
        "test_speakers": [],
    }

    split_file = source_dir / "speaker_split.json"
    with open(split_file, "w") as f:
        json.dump(speaker_split_data, f, indent=2)

    # Also mirror to results if results dir exists
    results_dir = PROJECT_ROOT / "results" / dataset_name
    if results_dir.exists():
        with open(results_dir / "speaker_split.json", "w") as f:
            json.dump(speaker_split_data, f, indent=2)

    # 7. Strict Zero-Leakage & 100-Folder Validation Checks
    print("\nExecuting strict zero-leakage and 100-folder validation checks...")

    train_dir_names = [d.name for d in TRAIN_DIR.iterdir() if not d.name.startswith(".")]
    val_dir_names = [d.name for d in VAL_DIR.iterdir() if not d.name.startswith(".")]
    test_items = [d.name for d in TEST_DIR.iterdir() if not d.name.startswith(".")]

    # Check 1: Exact folder counts
    assert len(train_dir_names) == TARGET_TRAIN_FOLDERS == 80, (
        f"Validation failed: Expected exactly 80 train directories, found {len(train_dir_names)}"
    )
    assert len(val_dir_names) == TARGET_VAL_FOLDERS == 20, (
        f"Validation failed: Expected exactly 20 val directories, found {len(val_dir_names)}"
    )
    assert len(test_items) == 0, (
        f"Validation failed: Test directory is not empty! Contains: {test_items}"
    )

    # Check 2: Total count strictly 100
    total_processed_folders = len(train_dir_names) + len(val_dir_names)
    assert total_processed_folders == TOTAL_TARGET_FOLDERS == 100, (
        f"Validation failed: Total folders must strictly be 100, got {total_processed_folders}"
    )

    # Check 3: Base speaker extraction
    train_dir_bases = {d.split("_")[0] for d in train_dir_names}
    val_dir_bases = {d.split("_")[0] for d in val_dir_names}

    # Check 4: Zero leakage between base speakers
    overlap = train_dir_bases.intersection(val_dir_bases)
    assert len(overlap) == 0, (
        f"Validation failed: Speaker leakage detected! Overlapping base speakers: {overlap}"
    )

    # Check 5: All symlinks are valid and resolve to existing directories
    for item in TRAIN_DIR.iterdir():
        assert item.resolve().exists(), f"Broken symlink in train: {item}"
    for item in VAL_DIR.iterdir():
        assert item.resolve().exists(), f"Broken symlink in val: {item}"

    print("[SUCCESS] All strict zero-leakage and 100-folder assertions passed successfully!")

    # 8. Output Summary
    print("\n" + "=" * 80)
    print(f"Training base speakers ({len(train_base_speakers)}):")
    print(", ".join(train_base_speakers))
    print(f"\nValidation base speakers ({len(val_base_speakers)}):")
    print(", ".join(val_base_speakers))
    print("=" * 80)

    print("\nDataset successfully linked!")
    print(f"  Train Folders : {counts['train_dirs']} (target 80)")
    print(f"  Val Folders   : {counts['val_dirs']} (target 20)")
    print(f"  Test Folders  : {len(test_items)} (target 0)")
    print(f"  Total Folders : {counts['train_dirs'] + counts['val_dirs']} (strictly 100 in total)")
    print("  Zero Leakage  : VERIFIED (0 overlapping speakers)\n")


if __name__ == "__main__":
    main()
