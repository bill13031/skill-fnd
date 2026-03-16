#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from fake_news_skillrl.dataset import normalize_jsonl_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize fake-news samples into the local schema.")
    parser.add_argument("--input", required=True, help="Path to raw JSONL samples.")
    parser.add_argument("--output", required=True, help="Path to normalized JSONL output.")
    args = parser.parse_args()

    samples = normalize_jsonl_file(args.input, args.output)
    print(f"Normalized {len(samples)} samples into {Path(args.output)}")


if __name__ == "__main__":
    main()
