#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from fake_news_skillrl.dataset import normalize_fakett_file, normalize_jsonl_file


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
    parser.add_argument("--num-frames", type=int, default=2, help="Number of frames to sample per video.")
    args = parser.parse_args()

    if args.dataset_format == "fakett":
        samples = normalize_fakett_file(
            input_path=args.input,
            output_path=args.output,
            video_dir=args.video_dir,
            frames_output_dir=args.frames_dir,
            num_frames=args.num_frames,
        )
    else:
        samples = normalize_jsonl_file(args.input, args.output)
    print(f"Normalized {len(samples)} samples into {Path(args.output)}")


if __name__ == "__main__":
    main()
