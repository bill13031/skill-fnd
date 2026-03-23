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

from fake_news_skillrl.agent import build_agent_pair
from fake_news_skillrl.dataset import load_normalized_samples
from fake_news_skillrl.env import FakeNewsEnv, FakeNewsEnvConfig
from fake_news_skillrl.memory import SkillsOnlyMemory
from fake_news_skillrl.trainer import RolloutTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a fake-news agent rollout.")
    parser.add_argument("--input", required=True, help="Normalized JSONL input path.")
    parser.add_argument("--skill-bank", required=True, help="Path to skill bank JSON.")
    parser.add_argument("--agent-type", default="heuristic", choices=["heuristic", "qwen_vl", "transformers"])
    parser.add_argument("--model-name", default=None, help="Local or HF model name for the VL agent.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on the number of samples to evaluate.")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.02)
    parser.add_argument(
        "--attach-frames-first-step-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Attach frame images only on the first reasoning step for VL agents.",
    )
    parser.add_argument(
        "--max-reasoning-steps-before-forced-verdict",
        type=int,
        default=2,
        help="At verdict stage, force a heuristic fallback verdict after this many completed collaboration steps if the agent still has not produced one.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    samples = load_normalized_samples(args.input)
    if args.max_samples is not None:
        samples = samples[: max(0, args.max_samples)]
    memory = SkillsOnlyMemory(args.skill_bank)
    env = FakeNewsEnv(config=FakeNewsEnvConfig(), memory=memory)
    analyzer_agent, worker_agent = build_agent_pair(
        agent_type=args.agent_type,
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        trust_remote_code=args.trust_remote_code,
        attach_frames_first_step_only=args.attach_frames_first_step_only,
    )
    results = RolloutTrainer(
        env=env,
        analyzer_agent=analyzer_agent,
        worker_agent=worker_agent,
        max_reasoning_steps_before_forced_verdict=args.max_reasoning_steps_before_forced_verdict,
    ).run(samples)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
