#!/usr/bin/env python3
"""
Extract and flatten video files from speaker-digit subfolders into a single directory.

Input hierarchy:
    digit_videos_only/
        ├── s1/
        │   ├── d0/ -> d0.mp4
        │   ├── d1/ -> d1.mp4
        │   └── ...
        └── sN/
            └── d10/ -> d10.mp4

Output flat directory (no subfolders):
    all_videos_outside/
        ├── s1d0.mp4
        ├── s1d1.mp4
        ├── ...
        ├── s2d1.mp4
        └── sNd10.mp4

Usage:
    python scripts/extract_all_videos.py
    python scripts/extract_all_videos.py --src <source_dir> --dst <output_dir>
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path


def natural_sort_key(s: str) -> list[int | str]:
    """Sort strings containing numbers in natural human order (e.g. s1, s2, s10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def find_video_for_digit(digit_dir: Path, digit_name: str) -> Path | None:
    """
    Find the most appropriate video file inside a digit directory.
    Handles duplicate tags '(1)', spaces, 'untitled-' prefixes, or single video files.
    """
    mp4_files = [
        f for f in digit_dir.iterdir()
        if f.is_file() and not f.name.startswith(".") and f.suffix.lower() == ".mp4"
    ]

    if not mp4_files:
        return None

    # Priority 1: Exact match <digit>.mp4 (e.g. d0.mp4)
    exact = [f for f in mp4_files if f.name == f"{digit_name}.mp4"]
    if exact:
        return exact[0]

    # Priority 2: Match after removing spaces (e.g. 'd2 .mp4')
    space_match = [f for f in mp4_files if f.name.replace(" ", "") == f"{digit_name}.mp4"]
    if space_match:
        return space_match[0]

    # Priority 3: 'untitled-<digit>.mp4'
    untitled = [f for f in mp4_files if f.name == f"untitled-{digit_name}.mp4"]
    if untitled:
        return untitled[0]

    # Priority 4: Duplicate file ending in (1)
    dup = [f for f in mp4_files if f.name == f"{digit_name}(1).mp4"]
    if dup:
        return dup[0]

    # Priority 5: If there is only one mp4 file in the folder (e.g. s50.mp4 inside s16/d1)
    if len(mp4_files) == 1:
        return mp4_files[0]

    # Fallback: Sort by name length and pick the shortest
    mp4_files.sort(key=lambda x: len(x.name))
    return mp4_files[0]


def resolve_paths(src_arg: str | None, dst_arg: str | None) -> tuple[Path, Path]:
    """Resolve source and destination directories with smart defaults."""
    cwd = Path.cwd().resolve()
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    # Determine source directory
    if src_arg:
        source_dir = Path(src_arg).resolve()
    elif cwd.name == "digit_videos_only":
        source_dir = cwd
    elif (cwd / "data" / "experiments" / "digit_videos_only").is_dir():
        source_dir = (cwd / "data" / "experiments" / "digit_videos_only").resolve()
    elif (cwd / "digit_videos_only").is_dir():
        source_dir = (cwd / "digit_videos_only").resolve()
    elif (repo_root / "data" / "experiments" / "digit_videos_only").is_dir():
        source_dir = (repo_root / "data" / "experiments" / "digit_videos_only").resolve()
    else:
        source_dir = cwd

    # Determine destination directory (same level as source_dir by default)
    if dst_arg:
        output_dir = Path(dst_arg).resolve()
    else:
        output_dir = source_dir.parent / "all_videos_outside"

    return source_dir, output_dir


def extract_videos(
    source_dir: Path,
    output_dir: Path,
    mode: str = "copy",
    dry_run: bool = False,
) -> None:
    """
    Extract videos from source_dir (organized by speaker/digit) to output_dir
    with naming convention sNdN.mp4.
    """
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    print("=" * 70)
    print("Tai Phake Lip Reading - Video Flattening & Renaming Pipeline")
    print("=" * 70)
    print(f"Source Directory:      {source_dir}")
    print(f"Destination Directory: {output_dir}")
    print(f"Transfer Mode:         {mode}")
    print(f"Dry Run:               {dry_run}")
    print("-" * 70)

    # Discover speaker folders
    speaker_dirs = sorted(
        [p for p in source_dir.iterdir() if p.is_dir() and not p.name.startswith(".")],
        key=lambda p: natural_sort_key(p.name),
    )

    if not speaker_dirs:
        print(f"Error: No speaker subdirectories found in {source_dir}")
        sys.exit(1)

    print(f"Found {len(speaker_dirs)} speaker folders: {', '.join(p.name for p in speaker_dirs)}")

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    extracted_count = 0
    missing_count = 0
    items_to_process = []

    for sp_dir in speaker_dirs:
        sp_name = sp_dir.name
        digit_dirs = sorted(
            [d for d in sp_dir.iterdir() if d.is_dir() and not d.name.startswith(".")],
            key=lambda d: natural_sort_key(d.name),
        )

        for d_dir in digit_dirs:
            d_name = d_dir.name
            target_filename = f"{sp_name}{d_name}.mp4"
            target_path = output_dir / target_filename

            source_file = find_video_for_digit(d_dir, d_name)
            if source_file is None:
                print(f"[MISSING] No video found for {sp_name}/{d_name}")
                missing_count += 1
            else:
                items_to_process.append((source_file, target_path, sp_name, d_name, target_filename))

    print(f"\nPrepared {len(items_to_process)} videos to process (Missing: {missing_count}).\n")

    for src_file, dst_file, sp_name, d_name, target_filename in items_to_process:
        action_desc = f"{mode.upper()}: {src_file.parent.name}/{src_file.name} -> {target_filename}"
        if dry_run:
            print(f"[DRY-RUN] Would {action_desc}")
            extracted_count += 1
            continue

        if mode == "copy":
            shutil.copy2(src_file, dst_file)
        elif mode == "symlink":
            if dst_file.exists() or dst_file.is_symlink():
                dst_file.unlink()
            dst_file.symlink_to(src_file.resolve())
        elif mode == "move":
            shutil.move(str(src_file), str(dst_file))
        else:
            raise ValueError(f"Unknown mode: {mode}")

        extracted_count += 1

    print("\n" + "=" * 70)
    print("Execution Summary:")
    print(f"  Total Processed: {extracted_count}")
    print(f"  Total Missing:   {missing_count}")
    print(f"  Target Folder:   {output_dir}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Extract and flatten videos from sN/dN/ structure to a single folder named sNdN.mp4"
    )
    parser.add_argument(
        "--src",
        type=str,
        default=None,
        help="Path to source 'digit_videos_only' directory (auto-detected if omitted)",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default=None,
        help="Path to destination 'all_videos_outside' directory (defaults to sibling of source)",
    )
    parser.add_argument(
        "--mode",
        choices=["copy", "symlink", "move"],
        default="copy",
        help="Method to extract files ('copy' [default], 'symlink', or 'move')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate the extraction without creating or copying files",
    )

    args = parser.parse_args()
    source_dir, output_dir = resolve_paths(args.src, args.dst)
    extract_videos(source_dir, output_dir, mode=args.mode, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
