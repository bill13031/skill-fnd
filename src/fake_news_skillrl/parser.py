from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional


INSPECT_RE = re.compile(r"<inspect>(.*?)</inspect>", re.IGNORECASE | re.DOTALL)
VERDICT_RE = re.compile(r"<verdict>(.*?)</verdict>", re.IGNORECASE | re.DOTALL)


@dataclass(slots=True)
class ParsedAction:
    raw_action: str
    action_type: str
    payload: Dict[str, Any]
    is_valid: bool
    error: Optional[str] = None


def _parse_inspect_payload(target: str, raw_action: str, max_frame_index: int) -> ParsedAction:
    target = target.strip()
    if target.startswith("frame:"):
        _, _, index_str = target.partition(":")
        if not index_str.isdigit():
            return ParsedAction(raw_action, "inspect", {}, False, "Frame index must be numeric.")
        index = int(index_str)
        if index < 0 or index > max_frame_index:
            return ParsedAction(raw_action, "inspect", {}, False, "Frame index out of range.")
        return ParsedAction(raw_action, "inspect", {"target": f"frame:{index}", "frame_index": index}, True)
    if target not in {"post_text", "transcript", "ocr_text", "metadata"}:
        return ParsedAction(raw_action, "inspect", {}, False, "Unsupported inspect target.")
    return ParsedAction(raw_action, "inspect", {"target": target}, True)


def _parse_verdict_payload(raw_payload: str, raw_action: str) -> ParsedAction:
    try:
        verdict_payload = json.loads(raw_payload.strip())
    except json.JSONDecodeError as exc:
        return ParsedAction(raw_action, "verdict", {}, False, f"Malformed verdict JSON: {exc}")

    label = str(verdict_payload.get("label", "")).lower()
    rationale = str(verdict_payload.get("rationale", "")).strip()
    evidence = verdict_payload.get("evidence", [])
    if label not in {"fake", "real", "unverified"}:
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
    stripped = action.strip()
    if not stripped:
        return ParsedAction(
            raw_action=action,
            action_type="invalid",
            payload={},
            is_valid=False,
            error="Exactly one inspect or verdict block is required.",
        )

    if stripped.startswith("<verdict>"):
        match = VERDICT_RE.fullmatch(stripped)
        if match is None:
            return ParsedAction(action, "verdict", {}, False, "Verdict action must be a single complete <verdict> block.")
        return _parse_verdict_payload(match.group(1), action)

    if stripped.startswith("<inspect>"):
        match = INSPECT_RE.fullmatch(stripped)
        if match is None:
            return ParsedAction(action, "inspect", {}, False, "Inspect action must be a single complete <inspect> block.")
        return _parse_inspect_payload(match.group(1), action, max_frame_index)

    inspect_matches = INSPECT_RE.findall(stripped)
    verdict_matches = VERDICT_RE.findall(stripped)
    if inspect_matches and verdict_matches:
        return ParsedAction(
            raw_action=action,
            action_type="invalid",
            payload={},
            is_valid=False,
            error="Cannot mix inspect and verdict actions.",
        )
    return ParsedAction(
        raw_action=action,
        action_type="invalid",
        payload={},
        is_valid=False,
        error="Exactly one inspect or verdict block is required.",
    )
