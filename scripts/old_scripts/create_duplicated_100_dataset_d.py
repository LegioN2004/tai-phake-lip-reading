#!/usr/bin/env python3
"""
Create a 100-speaker duplicated dataset from Method D 96x96 cropped lips.

Split Configuration (Canonical Seed 42):
- Test speakers (3): s1, s4, s21 (unaugmented, withheld from training)
- Validation speakers (3): s8, s9, s24 (unaugmented, withheld from training)
- Train base speakers (24): s2, s3, s5, s6, s7, s10, s11, s12, s13, s14, s15,
                            s16, s17, s18, s19, s20, s22, s23, s25, s26, s27,
                            s28, s29, s30

Duplication Strategy:
- 24 base train speakers are duplicated without image augmentation to reach exactly
  100 train speaker directories:
  - 4 speakers receive 4 duplicate directories (_dup1.._dup4) -> 4 * 5 = 20 directories
  - 20 speakers receive 3 duplicate directories (_dup1.._dup3) -> 20 * 4 = 80 directories
  - Total train speaker directories = 100
- Validation and Test speakers are NEVER duplicated or augmented.
- Total speaker directories in output = 106 (100 train + 3 val + 3 test).

Input:
    data/experiments/cropped_lips_dataset_d_96x96/
Output:
    data/experiments/duplicated_100_cropped_lips_d_96x96/
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "data" / "experiments" / "cropped_lips_dataset_d_96x96"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "experiments" / "duplicated_100_cropped_lips_d_96x96"

TARGET_TRAIN_SPEAKERS = 100
SEED = 42

# Canonical split (matching results/dlib/speaker_split.json)
VAL_SPEAKERS: List[str] = ["s8", "s9", "s24"]
TEST_SPEAKERS: List[str] = ["s1", "s4", "s21"]
TRAIN_BASE_SPEAKERS: List[str] = [
    "s2", "s3", "s5", "s6", "s7", "s10", "s11", "s12", "s13", "s14", "s15",
    "s16", "s17", "s18", "s19", "s20", "s22", "s23", "s25", "s26", "s27",
    "s28", "s29", "s30",
]


def build_speaker_mapping(
    train_base: List[str],
    target_train_count: int = TARGET_TRAIN_SPEAKERS,
) -> Dict[str, List[str]]:
    """
    Builds a mapping from each base speaker to all directory names it will produce.
    Base train speakers are assigned duplicates until target_train_count is reached.
    """
    num_base = len(train_base)
    duplicates_needed = target_train_count - num_base
    if duplicates_needed < 0:
        raise ValueError(f"Base speakers ({num_base}) exceeds target ({target_train_count})")

    base_dups_per_spk = duplicates_needed // num_base
    extra_dups = duplicates_needed % num_base

    mapping: Dict[str, List[str]] = {}
    for idx, spk in enumerate(train_base):
        # 1 original
        dirs = [spk]
        # Number of duplicates for this speaker
        num_dups = base_dups_per_spk + (1 if idx < extra_dups else 0)
        for d_idx in range(1, num_dups + 1):
            dirs.append(f"{spk}_dup{d_idx}")
        mapping[spk] = dirs

    total_train_dirs = sum(len(v) for v in mapping.values())
    assert total_train_dirs == target_train_count, (
        f"Expected {target_train_count} train directories, got {total_train_dirs}"
    )
    return mapping


def copy_file_task(task: Tuple[Path, Path]) -> None:
    src_path, dst_path = task
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate duplicated 100-train-speaker dataset from cropped lips dataset D (96x96)"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Input cropped lips directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output duplicated dataset directory",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of worker threads for parallel file copying",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination files if they exist",
    )
    args = parser.parse_args()

    start_time = time.time()
    input_root = args.input_dir.resolve()
    output_root = args.output_dir.resolve()

    if not input_root.exists():
        print(f"Error: Input directory {input_root} does not exist!")
        sys.exit(1)

    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("      DATASET DUPLICATION: CROPPED LIPS METHOD D (96x96) -> 100 TRAIN SPK")
    print("=" * 80)
    print(f"Input Directory       : {input_root}")
    print(f"Output Directory      : {output_root}")
    print(f"Random Seed           : {SEED}")
    print(f"Validation Speakers   : {len(VAL_SPEAKERS)} {VAL_SPEAKERS}")
    print(f"Test Speakers         : {len(TEST_SPEAKERS)} {TEST_SPEAKERS}")
    print(f"Base Train Speakers   : {len(TRAIN_BASE_SPEAKERS)} {TRAIN_BASE_SPEAKERS}")
    print(f"Target Train Speakers : {TARGET_TRAIN_SPEAKERS}")
    print(f"Workers               : {args.workers}")
    print(f"Overwrite Mode        : {args.overwrite}")
    print("-" * 80)

    # 1. Leakage Verification on Specification
    set_train_base = set(TRAIN_BASE_SPEAKERS)
    set_val = set(VAL_SPEAKERS)
    set_test = set(TEST_SPEAKERS)

    assert len(set_train_base & set_val) == 0, "LEAKAGE: Overlap between Train and Val!"
    assert len(set_train_base & set_test) == 0, "LEAKAGE: Overlap between Train and Test!"
    assert len(set_val & set_test) == 0, "LEAKAGE: Overlap between Val and Test!"
    assert len(set_train_base) == 24, "Expected 24 base train speakers"
    assert len(set_val) == 3, "Expected 3 val speakers"
    assert len(set_test) == 3, "Expected 3 test speakers"

    # 2. Build Train Duplication Plan
    train_speaker_map = build_speaker_mapping(TRAIN_BASE_SPEAKERS, TARGET_TRAIN_SPEAKERS)
    all_train_speakers: List[str] = []
    for spk in TRAIN_BASE_SPEAKERS:
        all_train_speakers.extend(train_speaker_map[spk])

    # Ensure no leakage in generated names
    for spk_dir in all_train_speakers:
        base = spk_dir.split("_")[0]
        assert base not in set_val, f"LEAKAGE: Val speaker {base} in train dirs: {spk_dir}"
        assert base not in set_test, f"LEAKAGE: Test speaker {base} in train dirs: {spk_dir}"

    print(f"Generated {len(all_train_speakers)} train speaker designations (24 original + 76 duplicated).")
    print(f"Total output speaker folders will be: {len(all_train_speakers) + len(VAL_SPEAKERS) + len(TEST_SPEAKERS)}")
    print("-" * 80)

    # 3. Discover Source Frames
    copy_tasks: List[Tuple[Path, Path]] = []
    manifest_records: List[Dict[str, any]] = []

    # Helper to scan a speaker directory
    def scan_speaker_frames(spk: str) -> List[Tuple[str, str, Path]]:
        spk_dir = input_root / spk
        if not spk_dir.exists():
            raise FileNotFoundError(f"Missing source speaker directory: {spk_dir}")
        found = []
        for d in sorted(spk_dir.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            dig = d.name
            for f in sorted(d.glob("frame_*.jpg")):
                found.append((dig, f.name, f))
        return found

    # A. Process Validation Speakers (1x copy, unaugmented)
    print("Indexing Validation Speakers...")
    for spk in VAL_SPEAKERS:
        frames = scan_speaker_frames(spk)
        for dig, fname, src_path in frames:
            dst_path = output_root / spk / dig / fname
            if args.overwrite or not dst_path.exists():
                copy_tasks.append((src_path, dst_path))
            manifest_records.append({
                "speaker": spk,
                "base_speaker": spk,
                "digit": dig,
                "frame": fname,
                "split": "val",
                "is_duplicate": False,
                "source_path": str(src_path),
                "output_path": str(dst_path),
            })

    # B. Process Test Speakers (1x copy, unaugmented)
    print("Indexing Test Speakers...")
    for spk in TEST_SPEAKERS:
        frames = scan_speaker_frames(spk)
        for dig, fname, src_path in frames:
            dst_path = output_root / spk / dig / fname
            if args.overwrite or not dst_path.exists():
                copy_tasks.append((src_path, dst_path))
            manifest_records.append({
                "speaker": spk,
                "base_speaker": spk,
                "digit": dig,
                "frame": fname,
                "split": "test",
                "is_duplicate": False,
                "source_path": str(src_path),
                "output_path": str(dst_path),
            })

    # C. Process Train Speakers (Originals + Duplicates)
    print("Indexing Train Speakers (24 base -> 100 duplicated)...")
    for base_spk in TRAIN_BASE_SPEAKERS:
        frames = scan_speaker_frames(base_spk)
        target_dirs = train_speaker_map[base_spk]
        for out_spk in target_dirs:
            is_dup = (out_spk != base_spk)
            for dig, fname, src_path in frames:
                dst_path = output_root / out_spk / dig / fname
                if args.overwrite or not dst_path.exists():
                    copy_tasks.append((src_path, dst_path))
                manifest_records.append({
                    "speaker": out_spk,
                    "base_speaker": base_spk,
                    "digit": dig,
                    "frame": fname,
                    "split": "train",
                    "is_duplicate": is_dup,
                    "source_path": str(src_path),
                    "output_path": str(dst_path),
                })

    print(f"Total manifest records : {len(manifest_records):,}")
    print(f"Files to copy          : {len(copy_tasks):,}")
    print("-" * 80)

    # 4. Copy files in parallel
    if copy_tasks:
        print(f"Copying files with {args.workers} workers...")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            list(tqdm(executor.map(copy_file_task, copy_tasks), total=len(copy_tasks), desc="Copying frames"))
    else:
        print("All output files already exist, skipping copy (use --overwrite to force re-copy).")

    # 5. Save Manifest CSV
    df_manifest = pd.DataFrame(manifest_records)
    manifest_csv_path = output_root / "duplicated_100_manifest.csv"
    df_manifest.to_csv(manifest_csv_path, index=False)
    print(f"\nManifest saved to: {manifest_csv_path}")

    # 6. Save Split Information JSON
    split_info = {
        "dataset_name": "duplicated_100_cropped_lips_d_96x96",
        "source_dataset": "cropped_lips_dataset_d_96x96",
        "seed": SEED,
        "num_base_train_speakers": len(TRAIN_BASE_SPEAKERS),
        "num_train_speakers": len(all_train_speakers),
        "num_val_speakers": len(VAL_SPEAKERS),
        "num_test_speakers": len(TEST_SPEAKERS),
        "total_speakers": len(all_train_speakers) + len(VAL_SPEAKERS) + len(TEST_SPEAKERS),
        "base_train_speakers": TRAIN_BASE_SPEAKERS,
        "train_speakers": all_train_speakers,
        "val_speakers": VAL_SPEAKERS,
        "test_speakers": TEST_SPEAKERS,
        "utterance_counts": {
            "train": int(len(df_manifest[df_manifest["split"] == "train"].groupby(["speaker", "digit"]))),
            "val": int(len(df_manifest[df_manifest["split"] == "val"].groupby(["speaker", "digit"]))),
            "test": int(len(df_manifest[df_manifest["split"] == "test"].groupby(["speaker", "digit"]))),
        },
        "frame_counts": {
            "train": int((df_manifest["split"] == "train").sum()),
            "val": int((df_manifest["split"] == "val").sum()),
            "test": int((df_manifest["split"] == "test").sum()),
            "total": int(len(df_manifest)),
        },
        "leakage_verified": True,
        "elapsed_seconds": round(time.time() - start_time, 2),
    }

    split_json_path = output_root / "speaker_split.json"
    with open(split_json_path, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"Speaker split JSON saved to: {split_json_path}")

    # 7. Strict Post-Generation Assertions
    print("\nRunning verification assertions...")
    output_dirs = sorted([d.name for d in output_root.iterdir() if d.is_dir() and not d.name.startswith(".")])
    assert len(output_dirs) == 106, f"Expected 106 speaker folders, found {len(output_dirs)}"
    
    val_in_output = [d for d in output_dirs if d in set_val]
    test_in_output = [d for d in output_dirs if d in set_test]
    train_in_output = [d for d in output_dirs if d not in set_val and d not in set_test]

    assert len(val_in_output) == 3, f"Expected 3 val folders, found {len(val_in_output)}"
    assert len(test_in_output) == 3, f"Expected 3 test folders, found {len(test_in_output)}"
    assert len(train_in_output) == 100, f"Expected 100 train folders, found {len(train_in_output)}"

    # Ensure no val/test base names exist in train folders
    for d in train_in_output:
        base = d.split("_")[0]
        assert base not in set_val, f"Leakage: val speaker {base} found in train folder {d}"
        assert base not in set_test, f"Leakage: test speaker {base} found in train folder {d}"

    print("ALL VERIFICATION CHECKS PASSED!")
    print("=" * 80)
    print(f"Summary:")
    print(f"  Train speaker directories : {len(train_in_output)} (from 24 base)")
    print(f"  Validation directories    : {len(val_in_output)} (unaugmented)")
    print(f"  Test directories          : {len(test_in_output)} (unaugmented)")
    print(f"  Total output directories  : {len(output_dirs)}")
    print(f"  Total frames copied       : {len(df_manifest):,}")
    print("=" * 80)


if __name__ == "__main__":
    main()
