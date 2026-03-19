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


def _parse_verdict_payload(raw_payload: str, raw_action: str) -> ParsedAction:
    try:
        verdict_payload = json.loads(raw_payload.strip())
    except json.JSONDecodeError as exc:
        return ParsedAction(raw_action, "verdict", {}, False, f"Malformed verdict JSON: {exc}")

    label = str(verdict_payload.get("label", "")).lower()
    rationale = str(verdict_payload.get("rationale", "")).strip()
    evidence = verdict_payload.get("evidence", [])
    if label not in {"fake", "real"}:
        return ParsedAction(raw_action, "verdict", {}, False, "Unsupported verdict label.")
    if not rationale:
        return ParsedAction(raw_action, "verdict", {}, False, "Verdict rationale is required.")
    if not isinstance(evidence, list):
        return ParsedAction(raw_action, "verdict", {}, False, "Verdict evidence must be a list.")
    evidence_list = [str(item).strip() for item in evidence if str(item).strip()]

    return ParsedAction(
        raw_action=raw_action,
        action_type="verdict",
        payload={"label": label, "rationale": rationale, "evidence": evidence_list},
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
        match = CREATE_RE.fullmatch(stripped)
        if match is None:
            return ParsedAction(action, "create", {}, False, "Create action must be a single complete <create> block.")
        return _parse_text_action_payload("create", match.group(1), action)

    if stripped.startswith("<check>"):
        match = CHECK_RE.fullmatch(stripped)
        if match is None:
            return ParsedAction(action, "check", {}, False, "Check action must be a single complete <check> block.")
        return _parse_text_action_payload("check", match.group(1), action)

    if stripped.startswith("<use_skill>"):
        match = USE_SKILL_RE.fullmatch(stripped)
        if match is None:
            return ParsedAction(action, "use_skill", {}, False, "Use_skill action must be a single complete <use_skill> block.")
        return _parse_text_action_payload("use_skill", match.group(1), action)

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
