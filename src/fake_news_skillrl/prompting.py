from __future__ import annotations

from typing import List

from .schema import FakeNewsSample


CONTROLLED_STAGES = [
    "analyzer_report",
    "worker_skill",
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
    if stage_outputs.get("analyzer_report"):
        lines.append(f"Analyzer report:\n{stage_outputs['analyzer_report']}")
    if stage_outputs.get("worker_skill"):
        lines.append(f"Worker skill:\n{stage_outputs['worker_skill']}")
    return "\n\n".join(lines) if lines else "None yet."


def build_stage_prompt(
    sample: FakeNewsSample,
    stage: str,
    stage_outputs: dict[str, str],
    step_index: int,
    max_steps: int,
    skill_prompt: str = "",
) -> str:
    shared = (
        "## Post Information\n"
        f"{_inputs_block(sample)}\n\n"
        "## Collaboration History\n"
        f"{_history_block(stage_outputs)}\n\n"
        f"## Current Stage\n{stage} ({step_index} of {max_steps})\n\n"
    )

    if stage == "analyzer_report":
        return (
            "You are Analyzer.\n"
            "Read the post text and attached frames, then prepare a concise case report for your Worker teammate.\n"
            "Your job is to understand the post, identify the main claim, make a preliminary fake/real judgment, and explain what kind of skill would help most.\n"
            "Do not give the final fake/real verdict yet.\n\n"
            + shared
            + "Write exactly four plain-text lines in this format:\n"
            + "Visual: what is visibly shown in the frames.\n"
            + "Claim: the main concrete factual claim the post makes.\n"
            + "Preliminary judgment: likely fake or likely real, with a short reason.\n"
            + "Need: what verification skill or principle would help judge this case more reliably using only the provided inputs.\n"
            + "Keep each line short, concrete, and grounded in the provided inputs.\n"
        )

    if stage == "worker_skill":
        return (
            "You are Worker.\n"
            "Your job is skill management for the Analyzer teammate.\n"
            "Read the Analyzer report, including the preliminary judgment, inspect the retrieved skills below, and return one short skill the Analyzer should use.\n"
            "Prefer an existing retrieved skill when it fits. Otherwise create one short reusable skill.\n"
            "Do not decide fake or real yourself.\n\n"
            + shared
            + build_skill_section(skill_prompt)
            + "Return exactly one short plain-text line starting with 'Skill: '.\n"
            + "Focus on the rule that will most improve or correct the Analyzer's preliminary judgment in this case.\n"
        )

    if stage == "verdict":
        return (
            "You are Analyzer.\n"
            "Use your preliminary reasoning plus the Worker-provided skill to improve or confirm your judgment and decide whether the post is fake or real.\n"
            "An extraordinary factual claim is not verified just because the visuals are on-topic; stylized, composite, or generic imagery does not count as documentary proof.\n"
            "If the post presents an extraordinary real-world claim without credible verification in the provided inputs, prefer fake.\n\n"
            + shared
            + "Return exactly one final verdict block and nothing else.\n"
            + 'Format: <verdict>{"label":"fake|real","rationale":"..."}</verdict>\n'
            + "Keep the rationale short, concrete, and grounded in the Analyzer report plus Worker skill.\n"
        )

    raise ValueError(f"Unsupported controlled stage: {stage}")
