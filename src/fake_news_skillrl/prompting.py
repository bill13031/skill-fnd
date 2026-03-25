from __future__ import annotations

from typing import List

from .schema import FakeNewsSample


CONTROLLED_STAGES = [
    "preliminary_analysis",
    "worker_skill",
    "verdict",
]


def build_skill_section(skill_prompt: str) -> str:
    if not skill_prompt.strip():
        return ""
    return f"## Retrieved Skills\n{skill_prompt.strip()}\n\n"


def _post_inputs_block(sample: FakeNewsSample, include_frames: bool = True) -> str:
    lines = [f"Post text: {sample.post_text}"]
    if sample.transcript.strip():
        lines.append(f"Transcript: {sample.transcript}")
    if sample.ocr_text.strip():
        lines.append(f"OCR text: {sample.ocr_text}")
    if include_frames:
        described_frames = [
            f"- Frame {frame.frame_id}: {frame.description.strip()}"
            for frame in sample.frames
            if frame.description.strip()
        ]
        if described_frames:
            lines.append("Frame descriptions:")
            lines.extend(described_frames)
        else:
            lines.append(f"Attached frames: {len(sample.frames)}")
    return "\n".join(lines)


def _saved_event_block(sample: FakeNewsSample) -> str:
    event_text = sample.event_text.strip()
    if not event_text:
        return ""
    return f"## Extracted Event\nEvent: {event_text}\n\n"


def _preliminary_analysis_block(stage_outputs: dict[str, str]) -> str:
    preliminary_analysis = stage_outputs.get("preliminary_analysis", "").strip()
    if not preliminary_analysis:
        return ""
    return f"## Preliminary Analysis\n{preliminary_analysis}\n\n"


def _worker_skill_block(stage_outputs: dict[str, str]) -> str:
    worker_skill = stage_outputs.get("worker_skill", "").strip()
    if not worker_skill:
        return ""
    return f"## Worker Skill\n{worker_skill}\n\n"


def _stage_header(stage: str, step_index: int, max_steps: int) -> str:
    return f"## Current Stage\n{stage} ({step_index} of {max_steps})\n\n"


def _preliminary_analysis_context(
    sample: FakeNewsSample,
    stage: str,
    step_index: int,
    max_steps: int,
) -> str:
    return (
        _saved_event_block(sample)
        + "## Post Context Reminder\n"
        f"{_post_inputs_block(sample, include_frames=False)}\n\n"
    )


def _worker_skill_context(
    sample: FakeNewsSample,
    stage_outputs: dict[str, str],
    stage: str,
    step_index: int,
    max_steps: int,
    skill_prompt: str,
) -> str:
    return (
        _saved_event_block(sample)
        + _preliminary_analysis_block(stage_outputs)
        + build_skill_section(skill_prompt)
    )


def _verdict_context(
    sample: FakeNewsSample,
    stage_outputs: dict[str, str],
    stage: str,
    step_index: int,
    max_steps: int,
) -> str:
    return (
        _saved_event_block(sample)
        + _preliminary_analysis_block(stage_outputs)
        + _worker_skill_block(stage_outputs)
        + "## Post Context Reminder\n"
        f"{_post_inputs_block(sample, include_frames=False)}\n\n"
    )


def build_stage_prompt(
    sample: FakeNewsSample,
    stage: str,
    stage_outputs: dict[str, str],
    step_index: int,
    max_steps: int,
    skill_prompt: str = "",
) -> str:
    if stage == "preliminary_analysis":
        return (
            "You are a professional supervisor of social media platform.\n"
            "You're provided with the post's text, frames of videos.\n"
            "Your job is to decide whether the post should be allowed to publish.\n"
            "Harmful, non-factual, misleading posts are not allowed.\n"
            "Think and make an aassessment from multiple perspective.\n"
            "At this stage, reason about the extracted event itself rather than treating the raw post text or visuals as proof.\n"
            "You may use the provided inputs as context, but do not treat them as automatically valid evidence.\n"
            "You may use your own general world knowledge, historical knowledge, and common-sense reasoning.\n"
            "Do not give the final verdict yet.\n\n"
            + _preliminary_analysis_context(sample, stage, step_index, max_steps)
            + "Write exactly two plain-text lines in this format:\n"
            + "Preliminary reasoning: a short reasoning passage about whether the extracted event seems credible, doubtful, misleading, or fabricated.\n"
            + "Need: what verification skill or principle would significantly affect the conclusion, keep it short and concrete.\n"
        )

    if stage == "worker_skill":
        return (
            "Your job is skill support for the teammate.\n"
            "Read the extracted event and preliminary analysis, inspect the skills below, and select one short skill that will help the Analyzer fact-check this event.\n"
            "Prefer an existing retrieved skill when it fits. Otherwise create one short reusable skill.\n"
            "Do not decide fake or real yourself.\n\n"
            + _worker_skill_context(sample, stage_outputs, stage, step_index, max_steps, skill_prompt)
            + "Return exactly one short plain-text line starting with 'Skill: '.\n"
            + "Focus on the rule that will most improve the Analyzer's later fact-checking in this case.\n"
            + "Prefer a concrete verification principle over generic advice.\n"
            + "Do not suggest external search, outside records, or tools that are unavailable in this workflow.\n"
        )

    if stage == "verdict":
        return (
            "You are a professional fact-checker.\n"
            "Use the extracted event, your preliminary analysis, and the provided skill to fact-check the post and decide whether it is fake or real.\n"
            "An extraordinary factual claim is not verified just because the visuals are on-topic; stylized, composite, or generic imagery does not count as documentary proof.\n"
            "Use the provided inputs as primary evidence, but also use your own general world knowledge, historical knowledge, and common-sense reasoning when relevant.\n"
            "Do not label a claim fake solely because the post package does not fully prove it.\n"
            "For historical, military, political, or otherwise public-event claims, distinguish between 'not fully verified by this post' and 'likely false or misleading'.\n"
            "If the post presents an extraordinary real-world claim and the overall evidence plus background knowledge still points to weak credibility, prefer fake.\n\n"
            + _verdict_context(sample, stage_outputs, stage, step_index, max_steps)
            + "Return exactly one final verdict block and nothing else.\n"
            + 'Format: <verdict>{"label":"fake|real","rationale":"..."}</verdict>\n'
            + "Keep the rationale short, concrete, and grounded in the extracted event, preliminary analysis, Worker skill, and any relevant background knowledge.\n"
        )

    raise ValueError(f"Unsupported controlled stage: {stage}")
