#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from fake_news_skillrl.agent import build_agent
from fake_news_skillrl.dataset import normalize_fakett_file, normalize_jsonl_file
from fake_news_skillrl.io_utils import dump_jsonl
from fake_news_skillrl.schema import FakeNewsSample


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


def _build_event_extraction_prompt(sample: FakeNewsSample) -> str:
    lines = [
        "You are preparing a fact-checking dataset.",
        "Read the post information and attached frames, then extract the main real-world event or factual incident the post is trying to report.",
        "Do not judge whether it is fake or real.",
        "",
        "## Post Information",
        f"Post text: {sample.post_text}",
    ]
    if sample.transcript.strip():
        lines.append(f"Transcript: {sample.transcript}")
    if sample.ocr_text.strip():
        lines.append(f"OCR text: {sample.ocr_text}")
    lines.extend(
        [
            f"Attached frames: {len(sample.frames)}",
            "",
            "## Current Stage",
            "event_extraction",
            "",
            "Return exactly one plain-text line starting with 'Event: '.",
        ]
    )
    return "\n".join(lines)


def _build_frame_description_prompt(sample: FakeNewsSample, frame_index: int) -> str:
    return "\n".join(
        [
            "You are preparing a fact-checking dataset.",
            "For this single frame, do two things: describe only what is visibly shown, and transcribe any clearly readable text.",
            "Do not infer whether the post is fake or real.",
            "",
            "## Post Information",
            f"Post text: {sample.post_text}",
            f"Frame index: {frame_index}",
            "",
            "## Current Stage",
            "frame_description",
            "",
            "Return exactly two plain-text lines in this format:",
            "Frame description: ...",
            "OCR: ...",
            "If no readable text is visible, write 'OCR: [none]'.",
        ]
    )


def _extract_prefixed_value(text: str, prefix: str) -> str:
    for line in text.splitlines():
        cleaned = line.strip()
        if cleaned.lower().startswith(prefix.lower()):
            return cleaned[len(prefix):].strip()
    cleaned = text.strip()
    if cleaned.lower().startswith(prefix.lower()):
        return cleaned[len(prefix):].strip()
    return ""


def _enrich_samples_with_agent(
    samples: list[FakeNewsSample],
    agent_type: str,
    model_name: str | None,
    max_new_tokens: int,
    temperature: float,
    repetition_penalty: float,
    trust_remote_code: bool,
) -> None:
    agent = build_agent(
        agent_type=agent_type,
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        trust_remote_code=trust_remote_code,
        attach_frames_first_step_only=False,
    )

    for sample in samples:
        event_output = agent.next_action(sample, [], _build_event_extraction_prompt(sample))
        extracted_event = _extract_prefixed_value(event_output, "Event:")
        if extracted_event:
            sample.event_text = extracted_event
            sample.metadata["event"] = extracted_event
            sample.metadata["event_extraction_source"] = f"agent:{agent_type}"
            sample.metadata["event_extraction_model"] = model_name or getattr(agent, "model_name", agent_type)

        frame_ocr_texts: list[str] = []
        for frame_index, frame in enumerate(sample.frames):
            frame_only_sample = FakeNewsSample(
                sample_id=sample.sample_id,
                post_text=sample.post_text,
                transcript=sample.transcript,
                ocr_text=sample.ocr_text,
                event_text=sample.event_text,
                metadata=dict(sample.metadata),
                frames=[frame],
                label=sample.label,
                gold_evidence=list(sample.gold_evidence),
                split=sample.split,
                data_source=sample.data_source,
            )
            frame_output = agent.next_action(
                frame_only_sample,
                [],
                _build_frame_description_prompt(sample, frame_index),
            )
            described = _extract_prefixed_value(frame_output, "Frame description:")
            ocr_text = _extract_prefixed_value(frame_output, "OCR:")
            if described:
                frame.description = described
            if ocr_text and ocr_text.lower() != "[none]":
                frame.ocr_text = ocr_text
                frame_ocr_texts.append(ocr_text)
        sample.metadata["frame_description_source"] = f"agent:{agent_type}"
        sample.metadata["frame_description_model"] = model_name or getattr(agent, "model_name", agent_type)
        sample.metadata["ocr_source"] = f"agent:{agent_type}"
        sample.metadata["ocr_model"] = model_name or getattr(agent, "model_name", agent_type)
        if frame_ocr_texts:
            sample.ocr_text = "\n".join(frame_ocr_texts)


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
    parser.add_argument(
        "--enrich-with-agent",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use an agent/backend during dataset preparation to extract event_text and frame descriptions.",
    )
    parser.add_argument(
        "--agent-type",
        default="heuristic",
        choices=["heuristic", "qwen_vl", "transformers", "openai_sdk", "aliyun_sdk"],
        help="Agent backend used for optional dataset enrichment.",
    )
    parser.add_argument("--model-name", default=None, help="Optional model name for dataset enrichment.")
    parser.add_argument("--max-new-tokens", type=int, default=192, help="Generation length for optional dataset enrichment.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for optional dataset enrichment.")
    parser.add_argument("--repetition-penalty", type=float, default=1.02, help="Repetition penalty for optional dataset enrichment.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code for transformers-based enrichment.")
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

    if args.enrich_with_agent:
        _enrich_samples_with_agent(
            samples=samples,
            agent_type=args.agent_type,
            model_name=args.model_name,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            trust_remote_code=args.trust_remote_code,
        )
        dump_jsonl(args.output, [asdict(sample) for sample in samples])

    print(f"Normalized {len(samples)} samples into {Path(args.output)}")
    if args.dataset_format == "fakett":
        _print_frame_extraction_summary(samples)


if __name__ == "__main__":
    main()
