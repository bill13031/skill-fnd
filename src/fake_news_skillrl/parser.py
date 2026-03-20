from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional


CREATE_RE = re.compile(r"<create>(.*?)</create>", re.IGNORECASE | re.DOTALL)
CHECK_RE = re.compile(r"<check>(.*?)</check>", re.IGNORECASE | re.DOTALL)
USE_SKILL_RE = re.compile(r"<use_skill>(.*?)</use_skill>", re.IGNORECASE | re.DOTALL)
VERDICT_RE = re.compile(r"<verdict>(.*?)</verdict>", re.IGNORECASE | re.DOTALL)


@dataclass(slots=True)
class ParsedAction:
    raw_action: str
    action_type: str
    payload: Dict[str, Any]
    is_valid: bool
    error: Optional[str] = None


def _parse_text_action_payload(
    action_type: str,
    content: str,
    raw_action: str,
) -> ParsedAction:
    content = content.strip()
    if not content:
        return ParsedAction(raw_action, action_type, {}, False, f"{action_type} action content is required.")
    return ParsedAction(raw_action, action_type, {"content": content}, True)


def _parse_intermediate_action(
    stripped: str,
    action_type: str,
    pattern: re.Pattern[str],
    raw_action: str,
) -> ParsedAction:
    match = pattern.fullmatch(stripped)
    if match is not None:
        return _parse_text_action_payload(action_type, match.group(1), raw_action)

    opening_tag = f"<{action_type}>"
    closing_tag = f"</{action_type}>"
    if not stripped.startswith(opening_tag):
        return ParsedAction(raw_action, action_type, {}, False, f"{action_type.title()} action must start with <{action_type}>.")

    content = stripped[len(opening_tag) :]
    if closing_tag in content:
        content, tail = content.split(closing_tag, 1)
        lowered_tail = tail.lower()
        if any(tag in lowered_tail for tag in ("<create>", "<check>", "<use_skill>", "<verdict>")):
            return ParsedAction(
                raw_action=raw_action,
                action_type="invalid",
                payload={},
                is_valid=False,
                error="Cannot mix intermediate actions and verdict actions.",
            )
    return _parse_text_action_payload(action_type, content, raw_action)


def _parse_verdict_payload(raw_payload: str, raw_action: str) -> ParsedAction:
    try:
        verdict_payload = json.loads(raw_payload.strip())
    except json.JSONDecodeError as exc:
        return ParsedAction(raw_action, "verdict", {}, False, f"Malformed verdict JSON: {exc}")

    label = str(verdict_payload.get("label", "")).lower()
    rationale = str(verdict_payload.get("rationale", "")).strip()
    if label not in {"fake", "real"}:
        return ParsedAction(raw_action, "verdict", {}, False, "Unsupported verdict label.")
    if not rationale:
        return ParsedAction(raw_action, "verdict", {}, False, "Verdict rationale is required.")

    return ParsedAction(
        raw_action=raw_action,
        action_type="verdict",
        payload={"label": label, "rationale": rationale},
        is_valid=True,
    )


def parse_action(action: str, max_frame_index: int) -> ParsedAction:
    del max_frame_index
    stripped = action.strip()
    if not stripped:
        return ParsedAction(
            raw_action=action,
            action_type="invalid",
            payload={},
            is_valid=False,
            error="Exactly one create, check, use_skill, or verdict block is required.",
        )

    if stripped.startswith("<verdict>"):
        match = VERDICT_RE.fullmatch(stripped)
        if match is None:
            return ParsedAction(action, "verdict", {}, False, "Verdict action must be a single complete <verdict> block.")
        return _parse_verdict_payload(match.group(1), action)

    if stripped.startswith("<create>"):
        return _parse_intermediate_action(stripped, "create", CREATE_RE, action)

    if stripped.startswith("<check>"):
        return _parse_intermediate_action(stripped, "check", CHECK_RE, action)

    if stripped.startswith("<use_skill>"):
        return _parse_intermediate_action(stripped, "use_skill", USE_SKILL_RE, action)

    create_matches = CREATE_RE.findall(stripped)
    check_matches = CHECK_RE.findall(stripped)
    use_skill_matches = USE_SKILL_RE.findall(stripped)
    verdict_matches = VERDICT_RE.findall(stripped)
    non_verdict_count = len(create_matches) + len(check_matches) + len(use_skill_matches)
    if verdict_matches and non_verdict_count:
        return ParsedAction(
            raw_action=action,
            action_type="invalid",
            payload={},
            is_valid=False,
            error="Cannot mix intermediate actions and verdict actions.",
        )
    return ParsedAction(
        raw_action=action,
        action_type="invalid",
        payload={},
        is_valid=False,
        error="Exactly one create, check, use_skill, or verdict block is required.",
    )
