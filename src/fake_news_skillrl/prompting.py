from __future__ import annotations

from typing import Iterable, List

from .schema import FakeNewsSample


CONTROLLED_STAGES = [
    "visual_understanding",
    "claim_extraction",
    "consistency_check",
    "skill_application",
    "verdict",
]


def build_skill_section(skill_prompt: str) -> str:
    if not skill_prompt.strip():
        return ""
    return f"## Retrieved Skills\n{skill_prompt.strip()}\n\n"


def _inputs_block(sample: FakeNewsSample) -> str:
    lines = [f"Post text: {sample.post_text}"]
    if sample.transcript.strip():
        lines.append(f"Transcript: {sample.transcript}")
    if sample.ocr_text.strip():
        lines.append(f"OCR text: {sample.ocr_text}")
    lines.append(f"Attached frames: {len(sample.frames)}")
    return "\n".join(lines)


def _history_block(stage_outputs: dict[str, str]) -> str:
    lines: List[str] = []
    if stage_outputs.get("visual_understanding"):
        lines.append(f"Visual understanding: {stage_outputs['visual_understanding']}")
    if stage_outputs.get("claim_extraction"):
        lines.append(f"Claim extraction: {stage_outputs['claim_extraction']}")
    if stage_outputs.get("consistency_check"):
        lines.append(f"Consistency check: {stage_outputs['consistency_check']}")
    if stage_outputs.get("skill_application"):
        lines.append(f"Skill application: {stage_outputs['skill_application']}")
    return "\n".join(lines) if lines else "None yet."


def build_stage_prompt(
    sample: FakeNewsSample,
    stage: str,
    stage_outputs: dict[str, str],
    step_index: int,
    max_steps: int,
    skill_prompt: str = "",
) -> str:
    header = (
        "You are a short-video content credibility analyst.\n"
        "Judge whether the post is misleading or non-factual using only the provided post information and attached frames.\n"
        "Allow harmless humor, metaphor, excitement, or exaggeration when the post is not making a concrete misleading factual claim.\n\n"
    )
    shared = (
        "## Post Information\n"
        f"{_inputs_block(sample)}\n\n"
        "## Completed Stages\n"
        f"{_history_block(stage_outputs)}\n\n"
        f"## Current Stage\n{stage} ({step_index} of {max_steps})\n\n"
    )

    if stage == "visual_understanding":
        return (
            header
            + shared
            + "Describe only what is visibly shown in the attached frames.\n"
            + "Do not judge whether the post is true or false yet.\n"
            + "Output one or two plain sentences only.\n"
        )

    if stage == "claim_extraction":
        return (
            header
            + shared
            + "State the main concrete factual claim the post is making.\n"
            + "Output one short plain sentence only.\n"
        )

    if stage == "consistency_check":
        return (
            header
            + shared
            + "Compare the claimed event with the visible frames and provided text.\n"
            + "Say whether the provided inputs support, contradict, or fail to verify the claim.\n"
            + "Output one short plain sentence only.\n"
        )

    if stage == "skill_application":
        return (
            header
            + shared
            + build_skill_section(skill_prompt)
            + "Apply one retrieved skill if it fits.\n"
            + "If none fits well, create one short reusable skill for this case.\n"
            + "Output one short plain sentence only.\n"
        )

    if stage == "verdict":
        return (
            header
            + shared
            + "Return exactly one final verdict block and nothing else.\n"
            + 'Format: <verdict>{"label":"fake|real","rationale":"..."}</verdict>\n'
            + "Keep the rationale short, concrete, and grounded in the completed stages.\n"
        )

    raise ValueError(f"Unsupported controlled stage: {stage}")
