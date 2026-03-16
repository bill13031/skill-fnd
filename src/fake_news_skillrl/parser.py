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


def parse_action(action: str, max_frame_index: int) -> ParsedAction:
    inspect_matches = INSPECT_RE.findall(action)
    verdict_matches = VERDICT_RE.findall(action)

    if len(inspect_matches) + len(verdict_matches) != 1:
        return ParsedAction(
            raw_action=action,
            action_type="invalid",
            payload={},
            is_valid=False,
            error="Exactly one inspect or verdict block is required.",
        )

    if inspect_matches and verdict_matches:
        return ParsedAction(
            raw_action=action,
            action_type="invalid",
            payload={},
            is_valid=False,
            error="Cannot mix inspect and verdict actions.",
        )

    if inspect_matches:
        target = inspect_matches[0].strip()
        if target.startswith("frame:"):
            _, _, index_str = target.partition(":")
            if not index_str.isdigit():
                return ParsedAction(action, "inspect", {}, False, "Frame index must be numeric.")
            index = int(index_str)
            if index < 0 or index > max_frame_index:
                return ParsedAction(action, "inspect", {}, False, "Frame index out of range.")
            return ParsedAction(action, "inspect", {"target": f"frame:{index}", "frame_index": index}, True)
        if target not in {"post_text", "transcript", "ocr_text", "metadata"}:
            return ParsedAction(action, "inspect", {}, False, "Unsupported inspect target.")
        return ParsedAction(action, "inspect", {"target": target}, True)

    try:
        verdict_payload = json.loads(verdict_matches[0].strip())
    except json.JSONDecodeError as exc:
        return ParsedAction(action, "verdict", {}, False, f"Malformed verdict JSON: {exc}")

    label = str(verdict_payload.get("label", "")).lower()
    rationale = str(verdict_payload.get("rationale", "")).strip()
    evidence = verdict_payload.get("evidence", [])
    if label not in {"fake", "real", "unverified"}:
        return ParsedAction(action, "verdict", {}, False, "Unsupported verdict label.")
    if not rationale:
        return ParsedAction(action, "verdict", {}, False, "Verdict rationale is required.")
    if not isinstance(evidence, list):
        return ParsedAction(action, "verdict", {}, False, "Verdict evidence must be a list.")
    evidence_list = [str(item).strip() for item in evidence if str(item).strip()]

    return ParsedAction(
        raw_action=action,
        action_type="verdict",
        payload={"label": label, "rationale": rationale, "evidence": evidence_list},
        is_valid=True,
    )
