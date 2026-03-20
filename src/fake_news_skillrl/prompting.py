from __future__ import annotations

from typing import Iterable, List, Sequence

from .schema import FakeNewsSample


def available_case_lines(sample: FakeNewsSample) -> List[str]:
    lines = ["- post_text"]
    if sample.transcript.strip():
        lines.append("- transcript")
    if sample.ocr_text.strip():
        lines.append("- ocr_text")
    if sample.frames:
        lines.append(f"- attached video frames ({len(sample.frames)})")
    return lines


def build_skill_section(skill_prompt: str) -> str:
    if not skill_prompt.strip():
        return ""
    return f"## Retrieved Skills\n{skill_prompt.strip()}\n\n"


def build_initial_prompt(sample: FakeNewsSample, skill_prompt: str = "") -> str:
    case_info = "\n".join(available_case_lines(sample))
    transcript_block = f"Transcript: {sample.transcript}\n" if sample.transcript.strip() else ""
    ocr_block = f"OCR text: {sample.ocr_text}\n" if sample.ocr_text.strip() else ""
    return (
        "You are a short-video content credibility analyst.\n"
        "Your job is to judge whether the post contains misleading or non-factual content using only the provided post information and attached frames.\n"
        "Do not punish harmless humor, metaphor, excitement, or obvious exaggeration unless the post is making a concrete misleading factual claim.\n\n"
        f"{build_skill_section(skill_prompt)}"
        "## Post Information\n"
        f"Post text: {sample.post_text}\n"
        f"{transcript_block}"
        f"{ocr_block}"
        f"Attached frames: {len(sample.frames)}\n\n"
        "## Provided Inputs\n"
        f"{case_info}\n\n"
        "## Action Rules\n"
        "1. Use exactly one action per turn.\n"
        "2. All provided inputs are already shown to you. Do not ask to inspect or reveal anything else.\n"
        "3. Follow this staged policy:\n"
        "   <visual_understanding>describe only what is visibly shown in the attached frames, without judging truth yet</visual_understanding>\n"
        "   <create>state a claim decomposition, suspicion, or working hypothesis</create>\n"
        "   <check>state what in the provided inputs supports or contradicts a claim</check>\n"
        "   <use_skill>state which retrieved skill or principle you are applying</use_skill>\n"
        "4. Finish with:\n"
        '   <verdict>{"label":"fake|real","rationale":"..."}</verdict>\n'
        "5. Label meaning:\n"
        "   - fake: the post contains misleading or non-factual content presented as true or documentary.\n"
        "   - real: the post is factual, benign, or expressive without making a misleading factual claim.\n"
        "6. First ground yourself in the visuals, then reason from the caption and any retrieved skill.\n"
        "7. For visual_understanding/create/check/use_skill, write one short plain sentence only.\n"
        "8. Do not use bullet points, markdown, numbering, or extra commentary.\n"
        "9. Always close the XML tag you start.\n"
        "10. Keep the rationale short, concrete, and grounded in the provided inputs.\n"
    )


def build_step_prompt(
    sample: FakeNewsSample,
    visible_evidence: str,
    inspected_items: Iterable[str],
    allowed_actions: Sequence[str],
    step_index: int,
    max_steps: int,
    skill_prompt: str = "",
) -> str:
    history = "\n".join(f"- {item}" for item in inspected_items) if inspected_items else "none"
    allowed = ", ".join(allowed_actions)
    return (
        f"{build_initial_prompt(sample, skill_prompt)}\n"
        f"## Step\n{step_index} of {max_steps}\n"
        "## Prior Action History\n"
        f"{history}\n\n"
        "## Stage Policy\n"
        f"Allowed action(s) this turn: {allowed}\n"
        "Do not repeat the same action type on consecutive turns.\n"
        "Follow the staged policy strictly: visual_understanding -> create -> check -> use_skill or verdict -> verdict.\n\n"
        "## Input Reminder\n"
        f"{visible_evidence.strip() or 'All provided inputs are already shown above.'}\n"
    )
