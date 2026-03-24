#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from fake_news_skillrl.dataset import normalize_fakett_file, normalize_jsonl_file


def _print_frame_extraction_summary(samples: list[object]) -> None:
    status_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()

    for sample in samples:
        metadata = getattr(sample, "metadata", {}) or {}
        status = str(metadata.get("frame_extraction_status", "unknown"))
        reason = str(metadata.get("frame_extraction_reason", "unknown"))
        status_counts[status] += 1
        reason_counts[reason] += 1

    if not status_counts:
        return

    print("Frame extraction summary:")
    for status, count in sorted(status_counts.items()):
        print(f"  status[{status}] = {count}")
    for reason, count in sorted(reason_counts.items()):
        print(f"  reason[{reason}] = {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize fake-news samples into the local schema.")
    parser.add_argument(
        "--dataset-format",
        default="generic",
        choices=["generic", "fakett"],
        help="Input dataset format.",
    )
    parser.add_argument("--input", required=True, help="Path to raw JSONL samples.")
    parser.add_argument("--output", required=True, help="Path to normalized JSONL output.")
    parser.add_argument("--video-dir", default="~/datasets/fakett/video", help="Directory containing Fakett videos.")
    parser.add_argument("--frames-dir", default=None, help="Optional output directory for extracted frames.")
    parser.add_argument("--fps", type=float, default=2.0, help="Sampling rate for extracted frames, in frames per second.")
    parser.add_argument("--max-frames", type=int, default=16, help="Maximum number of sampled frames to keep per video.")
    args = parser.parse_args()

    if args.frames_dir is not None:
        Path(args.frames_dir).mkdir(parents=True, exist_ok=True)

    if args.dataset_format == "fakett":
        samples = normalize_fakett_file(
            input_path=args.input,
            output_path=args.output,
            video_dir=args.video_dir,
            frames_output_dir=args.frames_dir,
            fps=args.fps,
            max_frames=args.max_frames,
        )
    else:
        samples = normalize_jsonl_file(args.input, args.output)
    print(f"Normalized {len(samples)} samples into {Path(args.output)}")
    if args.dataset_format == "fakett":
        _print_frame_extraction_summary(samples)


if __name__ == "__main__":
    main()
