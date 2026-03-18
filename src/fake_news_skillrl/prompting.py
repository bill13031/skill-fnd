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
        "You are an investigative fake-news verification agent.\n"
        "Decide whether the post is factual using only the provided evidence package.\n\n"
        f"{build_skill_section(skill_prompt)}"
        "## Case Summary\n"
        f"Post text: {sample.post_text}\n"
        f"Metadata: {metadata_summary}\n\n"
        "## Available Evidence\n"
        f"{evidence}\n\n"
        "## Action Rules\n"
        "1. Use exactly one action per turn.\n"
        "2. Do not inspect the same evidence item twice.\n"
        "3. Make progress toward a verdict. After inspecting a few key evidence sources, issue a verdict instead of looping.\n"
        "4. Inspect evidence with one of:\n"
        "   <inspect>post_text</inspect>\n"
        "   <inspect>transcript</inspect>\n"
        "   <inspect>ocr_text</inspect>\n"
        "   <inspect>metadata</inspect>\n"
        "   <inspect>frame:0</inspect> (replace 0 with a valid frame index)\n"
        "5. Finish with:\n"
        '   <verdict>{"label":"fake|real|unverified","rationale":"...","evidence":["..."]}</verdict>\n'
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
    history = ", ".join(inspected_items) if inspected_items else "none"
    return (
        f"{build_initial_prompt(sample, skill_prompt)}\n"
        f"## Step\n{step_index} of {max_steps}\n"
        f"Inspected so far: {history}\n\n"
        "If the inspected history already covers the main evidence sources, your next action should usually be a verdict.\n\n"
        "## Visible Evidence\n"
        f"{visible_evidence.strip() or 'No evidence inspected yet.'}\n"
    )
