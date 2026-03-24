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
            "You are a short-video fact-checker.\n"
            "Read the post text and attached frames to determine whether the content is authentic and creadible (real) or fake and misleading (fake), then prepare a concise case report for your assistant teammate.\n"
            "You need to understand the post, identify the main claim, make a preliminary fake/real judgment, and explain what kind of skill would help most.\n"
            "You may use your own general world knowledge, historical knowledge, and common-sense reasoning.\n"
            "Do not treat missing proof inside the post package as automatic evidence that the claim is false.\n"
            "Do not give the final fake/real verdict yet.\n\n"
            + shared
            + "Write exactly four plain-text lines in this format:\n"
            + "Visual: what is visibly shown in the frames.\n"
            + "Claim: the main concrete factual claim the post makes.\n"
            + "Preliminary reasoning: a reasoning passage and conclusion.\n"
            + "Need: what verification skill or principle would significantly affect the conclusion, keep it short and concrete.\n"
        )

    if stage == "worker_skill":
        return (
            "You are an assistant of your fact-cheker teammate.\n"
            "Your job is skill management for the fact-cheker teammate.\n"
            "Read the report and need from the fact-checker, inspect the skills below, and select one short skill that you think the fact-checker should use.\n"
            "Prefer an existing retrieved skill when it fits. Otherwise create one short reusable skill.\n"
            "Do not decide fake or real yourself.\n\n"
            + shared
            + build_skill_section(skill_prompt)
            + "Return exactly one short plain-text line starting with 'Skill: '.\n"
            + "Focus on the rule that will most improve or correct the fact-checker's preliminary judgment in this case.\n"
            + "Prefer a concrete verification principle over generic advice.\n"
            + "Do not suggest external search, outside records, or tools that are unavailable in this workflow.\n"
        )

    if stage == "verdict":
        return (
            "You are a short-video fact-checker.\n"
            "Use your preliminary reasoning plus the provided skill to improve or confirm your reasoning and judgment.\n"
            "An extraordinary factual claim is not verified just because the visuals are on-topic; stylized, composite, or generic imagery does not count as documentary proof.\n"
            "Use the provided inputs as primary evidence, but also use your own general world knowledge, historical knowledge, and common-sense reasoning when relevant.\n"
            "Do not label a claim fake solely because the post package does not fully prove it.\n"
            "For historical, military, political, or otherwise public-event claims, distinguish between 'not fully verified by this post' and 'likely false or misleading'.\n"
            "If the post presents an extraordinary real-world claim and the overall evidence plus background knowledge still points to weak credibility, prefer fake.\n\n"
            + shared
            + "Return exactly one final verdict block and nothing else.\n"
            + 'Format: <verdict>{"label":"fake|real","rationale":"..."}</verdict>\n'
            + "Keep the rationale short, concrete, and grounded in the Analyzer report, Worker skill, and any relevant background knowledge.\n"
        )

    raise ValueError(f"Unsupported controlled stage: {stage}")
