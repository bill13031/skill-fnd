#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from fake_news_skillrl.dataset import load_normalized_samples
from fake_news_skillrl.trainer import SFTDataBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a lightweight SFT data preparation pass.")
    parser.add_argument("--input", required=True, help="Normalized JSONL input path.")
    args = parser.parse_args()

    samples = load_normalized_samples(args.input)
    rows = SFTDataBuilder().build(samples)
    summary = {
        "num_samples": len(samples),
        "num_sft_rows": len(rows),
        "example_sample_id": rows[0]["sample_id"] if rows else None,
    }
    print(json.dumps(summary, indent=2))
    print(f"SFT scaffold complete for {Path(args.input)}")


if __name__ == "__main__":
    main()
