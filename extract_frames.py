"""
Tai Phake Visual Speech Recognition (VSR)
Frame Extraction Pipeline - Person 1 Task

Extracts sequential frames from raw/trimmed videos in `data/digit/`
and saves them directly into the corresponding speaker/digit directory
alongside the source video.

Directory Structure:
    data/digit/
    ├── s1/
    │   ├── d0/
    │   │   ├── d0.mp4
    │   │   ├── frame_0001.jpg
    │   │   ├── frame_0002.jpg
    │   │   └── ...
"""

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from tqdm import tqdm


# ==============================================================================
# Logging Configuration
# ==============================================================================
def setup_logger(verbose: bool = False) -> logging.Logger:
    """Configures console logging with clean formatting."""
    logger = logging.getLogger("frame_extractor")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Avoid duplicate handlers if re-initialized
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger


# ==============================================================================
# Data Structures
# ==============================================================================
@dataclass
class VideoExtractionResult:
    """Stores extraction outcome and metrics for an individual video."""
    video_path: str
    speaker_id: str
    digit_id: str
    status: str  # 'SUCCESS', 'SKIPPED', 'FAILED'
    original_fps: float = 0.0
    total_video_frames: int = 0
    extracted_frames: int = 0
    error_message: Optional[str] = None


@dataclass
class ExtractionSummary:
    """Aggregates batch processing metrics across all videos."""
    total_videos_found: int = 0
    videos_processed: int = 0
    videos_skipped: int = 0
    videos_failed: int = 0
    total_frames_extracted: int = 0
    details: List[VideoExtractionResult] = field(default_factory=list)


# ==============================================================================
# Core Extraction Logic
# ==============================================================================
class FrameExtractor:
    """
    Handles robust, sequential frame extraction from video files.
    """

    SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    def __init__(
        self,
        input_dir: Path,
        step: int = 1,
        target_fps: Optional[float] = None,
        overwrite: bool = False,
        jpeg_quality: int = 95,
        padding: int = 4,
        dry_run: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Args:
            input_dir: Root directory containing speaker folders (e.g., data/digit).
            step: Extract every N-th frame (e.g., step=2 extracts every 2nd frame).
            target_fps: Desired sampling rate (FPS). Overrides step if specified.
            overwrite: If True, existing frames in target folder will be overwritten.
            jpeg_quality: Quality factor for JPG compression (1 to 100).
            padding: Zero-padding width for frame numbering (default: 4 -> frame_0001.jpg).
            dry_run: If True, scans and validates without saving any frames to disk.
            logger: Custom logger instance.
        """
        self.input_dir = Path(input_dir).resolve()
        self.step = max(1, step)
        self.target_fps = target_fps
        self.overwrite = overwrite
        self.jpeg_quality = max(1, min(100, jpeg_quality))
        self.padding = padding
        self.dry_run = dry_run
        self.logger = logger or setup_logger()

    def discover_videos(
        self,
        speaker_filter: Optional[str] = None,
        digit_filter: Optional[str] = None
    ) -> List[Path]:
        """
        Recursively scans input_dir for valid video files matching optional filters.
        """
        if not self.input_dir.exists():
            self.logger.error(f"Input directory does not exist: {self.input_dir}")
            return []

        videos: List[Path] = []
        for file_path in sorted(self.input_dir.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                # Path parsing: expects .../data/digit/<speaker>/<digit>/<video.mp4>
                rel_parts = file_path.relative_to(self.input_dir).parts
                if len(rel_parts) >= 2:
                    spk = rel_parts[0]
                    dig = rel_parts[1]
                    if speaker_filter and spk != speaker_filter:
                        continue
                    if digit_filter and dig != digit_filter:
                        continue
                videos.append(file_path)

        return videos

    def is_already_extracted(self, output_dir: Path) -> bool:
        """
        Checks whether sequential frame files already exist in the target folder.
        """
        existing_frames = list(output_dir.glob("frame_*.jpg"))
        return len(existing_frames) > 0

    def calculate_sampling_indices(
        self,
        total_frames: int,
        original_fps: float
    ) -> List[int]:
        """
        Determines the list of 0-based frame indices to extract based on step or target_fps.
        """
        if total_frames <= 0:
            return []

        if self.target_fps is not None and self.target_fps > 0:
            if original_fps <= 0:
                # Fallback to step if FPS cannot be determined
                return list(range(0, total_frames, self.step))
            
            # Resample at target_fps
            step_float = original_fps / self.target_fps
            if step_float <= 1.0:
                # Target FPS is >= original FPS, take all frames
                return list(range(total_frames))
            
            indices: List[int] = []
            curr = 0.0
            while int(round(curr)) < total_frames:
                idx = int(round(curr))
                if not indices or idx != indices[-1]:
                    indices.append(idx)
                curr += step_float
            return indices

        # Standard step-based sampling
        return list(range(0, total_frames, self.step))

    def extract_video(self, video_path: Path) -> VideoExtractionResult:
        """
        Extracts frames from a single video and saves them into the same directory.
        Original video remains completely untouched.
        """
        output_dir = video_path.parent
        rel_parts = video_path.relative_to(self.input_dir).parts
        speaker_id = rel_parts[0] if len(rel_parts) >= 2 else "unknown"
        digit_id = rel_parts[1] if len(rel_parts) >= 2 else "unknown"

        # Check existing frames
        if not self.overwrite and self.is_already_extracted(output_dir):
            existing_count = len(list(output_dir.glob("frame_*.jpg")))
            return VideoExtractionResult(
                video_path=str(video_path),
                speaker_id=speaker_id,
                digit_id=digit_id,
                status="SKIPPED",
                extracted_frames=existing_count,
                error_message=f"Frames already exist ({existing_count} frames found). Use --overwrite to replace."
            )

        # Open video stream
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return VideoExtractionResult(
                video_path=str(video_path),
                speaker_id=speaker_id,
                digit_id=digit_id,
                status="FAILED",
                error_message="Could not open video file (corrupted or unsupported codec)."
            )

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Check decodability by attempting to read frame 0
        ret, first_frame = cap.read()
        if not ret or first_frame is None:
            cap.release()
            return VideoExtractionResult(
                video_path=str(video_path),
                speaker_id=speaker_id,
                digit_id=digit_id,
                status="FAILED",
                original_fps=fps,
                total_video_frames=total_frames,
                error_message="Video file opened but first frame could not be decoded."
            )

        # Reset capture position to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Compute which frames to extract
        target_indices = set(self.calculate_sampling_indices(total_frames, fps))
        if not target_indices:
            # Fallback if total_frames was inaccurate: sample everything
            target_indices = None

        extracted_count = 0
        frame_idx = 0
        saved_seq = 1

        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Check if this frame should be extracted
                should_extract = (
                    target_indices is None or 
                    frame_idx in target_indices or
                    (target_indices is None and frame_idx % self.step == 0)
                )

                if should_extract:
                    if not self.dry_run:
                        filename = f"frame_{saved_seq:0{self.padding}d}.jpg"
                        out_path = output_dir / filename
                        success = cv2.imwrite(str(out_path), frame, encode_params)
                        if not success:
                            raise IOError(f"Failed to write image: {out_path}")
                    saved_seq += 1
                    extracted_count += 1

                frame_idx += 1

        except Exception as e:
            cap.release()
            return VideoExtractionResult(
                video_path=str(video_path),
                speaker_id=speaker_id,
                digit_id=digit_id,
                status="FAILED",
                original_fps=fps,
                total_video_frames=total_frames,
                extracted_frames=extracted_count,
                error_message=str(e)
            )

        cap.release()

        return VideoExtractionResult(
            video_path=str(video_path),
            speaker_id=speaker_id,
            digit_id=digit_id,
            status="SUCCESS",
            original_fps=fps,
            total_video_frames=frame_idx,
            extracted_frames=extracted_count
        )

    def run(
        self,
        speaker_filter: Optional[str] = None,
        digit_filter: Optional[str] = None
    ) -> ExtractionSummary:
        """
        Executes frame extraction across all discovered videos.
        """
        videos = self.discover_videos(speaker_filter=speaker_filter, digit_filter=digit_filter)
        summary = ExtractionSummary(total_videos_found=len(videos))

        if not videos:
            self.logger.warning("No video files found matching the criteria.")
            return summary

        self.logger.info(f"Discovered {len(videos)} video(s) for extraction.")
        if self.dry_run:
            self.logger.info("DRY-RUN MODE ENABLED: No files will be written to disk.")

        pbar = tqdm(videos, desc="Extracting frames", unit="video")
        for video_path in pbar:
            rel_display = f"{video_path.parent.parent.name}/{video_path.parent.name}/{video_path.name}"
            pbar.set_postfix_str(rel_display)

            res = self.extract_video(video_path)
            summary.details.append(res)

            if res.status == "SUCCESS":
                summary.videos_processed += 1
                summary.total_frames_extracted += res.extracted_frames
            elif res.status == "SKIPPED":
                summary.videos_skipped += 1
            elif res.status == "FAILED":
                summary.videos_failed += 1
                self.logger.error(f"Failed on {rel_display}: {res.error_message}")

        return summary


# ==============================================================================
# CLI Entry Point
# ==============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Person 1: Lossless in-situ frame extraction for Tai Phake VSR dataset."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/digit",
        help="Path to root digit directory (default: data/digit)."
    )
    parser.add_argument(
        "--step", "-s",
        type=int,
        default=1,
        help="Extract every N-th frame (default: 1 = extract all frames)."
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help="Target frame rate to resample frames (optional, overrides --step)."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing frames if already extracted."
    )
    parser.add_argument(
        "--quality", "-q",
        type=int,
        default=95,
        help="JPG compression quality 1-100 (default: 95)."
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default=None,
        help="Filter by specific speaker (e.g. 's1')."
    )
    parser.add_argument(
        "--digit",
        type=str,
        default=None,
        help="Filter by specific digit (e.g. 'd0')."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and test decodability without writing frames to disk."
    )
    parser.add_argument(
        "--save-report",
        type=str,
        default=None,
        help="Path to save execution summary report as JSON (e.g. results/extract_report.json)."
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable detailed logging output."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger(verbose=args.verbose)

    extractor = FrameExtractor(
        input_dir=Path(args.input_dir),
        step=args.step,
        target_fps=args.target_fps,
        overwrite=args.overwrite,
        jpeg_quality=args.quality,
        dry_run=args.dry_run,
        logger=logger,
    )

    summary = extractor.run(
        speaker_filter=args.speaker,
        digit_filter=args.digit
    )

    # Print summary report
    print("\n" + "=" * 60)
    print(" FRAME EXTRACTION SUMMARY REPORT")
    print("=" * 60)
    print(f"Total Videos Found       : {summary.total_videos_found}")
    print(f"Videos Processed         : {summary.videos_processed}")
    print(f"Videos Skipped (Existing): {summary.videos_skipped}")
    print(f"Videos Failed            : {summary.videos_failed}")
    print(f"Total Frames Extracted   : {summary.total_frames_extracted}")
    if summary.videos_processed > 0:
        avg_frames = summary.total_frames_extracted / summary.videos_processed
        print(f"Average Frames / Video   : {avg_frames:.1f}")
    print("=" * 60)

    # Optional report export
    if args.save_report:
        report_path = Path(args.save_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_data = {
            "total_videos_found": summary.total_videos_found,
            "videos_processed": summary.videos_processed,
            "videos_skipped": summary.videos_skipped,
            "videos_failed": summary.videos_failed,
            "total_frames_extracted": summary.total_frames_extracted,
            "details": [asdict(d) for d in summary.details],
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        logger.info(f"Summary report saved to: {report_path}")

    # Return non-zero exit code if any failures occurred
    if summary.videos_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
