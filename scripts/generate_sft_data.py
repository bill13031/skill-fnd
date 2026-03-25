#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from fake_news_skillrl.agent import build_agent_pair
from fake_news_skillrl.dataset import load_normalized_samples
from fake_news_skillrl.io_utils import dump_jsonl
from fake_news_skillrl.trainer import SFTDataBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SFT trajectories from normalized fake-news samples.")
    parser.add_argument("--input", required=True, help="Normalized JSONL input path.")
    parser.add_argument("--output", required=True, help="JSONL output path for SFT trajectories.")
    parser.add_argument("--agent-type", default="heuristic", choices=["heuristic", "qwen_vl", "transformers", "openai_sdk", "aliyun_sdk"])
    parser.add_argument("--model-name", default=None, help="Local or HF model name for the VL agent.")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.02)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    samples = load_normalized_samples(args.input)
    analyzer_agent, worker_agent = build_agent_pair(
        agent_type=args.agent_type,
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        trust_remote_code=args.trust_remote_code,
        attach_frames_first_step_only=False,
    )
    builder = SFTDataBuilder(analyzer_agent=analyzer_agent, worker_agent=worker_agent)
    rows = builder.build(samples)
    dump_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} SFT rows to {Path(args.output)}")


if __name__ == "__main__":
    main()
