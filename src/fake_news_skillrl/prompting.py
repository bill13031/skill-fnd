from __future__ import annotations

from typing import Dict, Iterable, List

from .schema import FakeNewsSample


def summarize_metadata(metadata: Dict[str, object]) -> str:
    parts: List[str] = []
    for key, value in metadata.items():
        if key == "engagement" and isinstance(value, dict):
            engagement = ", ".join(f"{k}={v}" for k, v in sorted(value.items()))
            parts.append(f"engagement({engagement})")
        else:
            parts.append(f"{key}={value}")
    return "; ".join(parts)


def available_evidence_lines(sample: FakeNewsSample) -> List[str]:
    lines = [
        "- post_text",
        "- transcript",
        "- ocr_text",
        "- metadata",
    ]
    for index, frame in enumerate(sample.frames):
        lines.append(f"- frame:{index} ({frame.description or frame.path})")
    return lines


def build_skill_section(skill_prompt: str) -> str:
    if not skill_prompt.strip():
        return ""
    return f"## Retrieved Skills\n{skill_prompt.strip()}\n\n"


def build_initial_prompt(sample: FakeNewsSample, skill_prompt: str = "") -> str:
    evidence = "\n".join(available_evidence_lines(sample))
    metadata_summary = summarize_metadata(sample.metadata)
    return (
        "You are a short-video content credibility analyst.\n"
        "Your job is to judge whether the post contains misleading or non-factual content using only the provided evidence package.\n"
        "Do not punish harmless humor, metaphor, excitement, or obvious exaggeration unless the post is making a concrete misleading factual claim.\n\n"
        f"{build_skill_section(skill_prompt)}"
        "## Case Summary\n"
        f"Post text: {sample.post_text}\n"
        f"Transcript: {sample.transcript or '[not available]'}\n"
        f"OCR text: {sample.ocr_text or '[not available]'}\n"
        f"Metadata: {metadata_summary}\n\n"
        "## Available Evidence\n"
        f"{evidence}\n\n"
        "## Action Rules\n"
        "1. Use exactly one action per turn.\n"
        "2. All available evidence is already provided to you. Do not ask to inspect or reveal evidence.\n"
        "3. Use intermediate reasoning-control actions when helpful:\n"
        "   <create>state a claim decomposition, suspicion, or working hypothesis</create>\n"
        "   <check>state what evidence supports or contradicts a claim within the provided case package</check>\n"
        "   <use_skill>state which retrieved skill or principle you are applying</use_skill>\n"
        "4. Finish with:\n"
        '   <verdict>{"label":"fake|real","rationale":"...","evidence":["..."]}</verdict>\n'
        "5. Label meaning:\n"
        "   - fake: the post contains misleading or non-factual content presented as true or documentary.\n"
        "   - real: the post is factual, benign, or expressive without making a misleading factual claim.\n"
        "6. Cite concrete evidence in the final evidence list.\n"
    )


def build_step_prompt(
    sample: FakeNewsSample,
    visible_evidence: str,
    inspected_items: Iterable[str],
    step_index: int,
    max_steps: int,
    skill_prompt: str = "",
) -> str:
    history = "\n".join(f"- {item}" for item in inspected_items) if inspected_items else "none"
    return (
        f"{build_initial_prompt(sample, skill_prompt)}\n"
        f"## Step\n{step_index} of {max_steps}\n"
        "## Prior Action History\n"
        f"{history}\n\n"
        "Use create/check/use_skill to organize reasoning, then finish with a verdict.\n\n"
        "## Evidence Reminder\n"
        f"{visible_evidence.strip() or 'All core evidence is already shown above.'}\n"
    )
