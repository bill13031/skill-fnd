from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from .io_utils import load_json


@dataclass(slots=True)
class RetrievedSkills:
    general_skills: List[Dict[str, Any]]
    task_specific_skills: List[Dict[str, Any]]
    mistakes_to_avoid: List[Dict[str, Any]]
    task_type: str
    retrieval_mode: str = "template"


class SkillsOnlyMemory:
    def __init__(self, skills_json_path: str | Path, retrieval_mode: str = "template") -> None:
        if retrieval_mode != "template":
            raise ValueError("Only template retrieval is implemented in this standalone project.")
        self.skills = load_json(skills_json_path)
        self.retrieval_mode = retrieval_mode

    def _detect_task_type(self, task_description: str) -> str:
        goal = task_description.lower()
        task_specific = self.skills.get("task_specific_skills", {})
        known_types = list(task_specific.keys())
        for task_type in known_types:
            tokens = task_type.replace("_", " ").split()
            if all(token in goal for token in tokens):
                return task_type

        keyword_map = {
            "manipulated_media": ["deepfake", "edited", "manipulated", "fake video"],
            "misleading_caption": ["caption", "headline", "claim", "cures", "breaking"],
            "out_of_context": ["archive", "old footage", "yesterday", "reused footage"],
            "source_credibility": ["official", "source", "authority", "office"],
            "temporal_inconsistency": ["date", "timestamp", "archive", "timeline"],
        }
        for task_type, keywords in keyword_map.items():
            if any(keyword in goal for keyword in keywords):
                return task_type
        return known_types[0] if known_types else "unknown"

    def retrieve(self, task_description: str, top_k: int = 4, task_specific_top_k: int = 3) -> Dict[str, Any]:
        task_type = self._detect_task_type(task_description)
        general = self.skills.get("general_skills", [])[:top_k]
        task_specific = self.skills.get("task_specific_skills", {}).get(task_type, [])[:task_specific_top_k]
        mistakes = self.skills.get("common_mistakes", [])[:5]
        return {
            "general_skills": general,
            "task_specific_skills": task_specific,
            "mistakes_to_avoid": mistakes,
            "task_type": task_type,
            "retrieval_mode": self.retrieval_mode,
        }

    def format_for_prompt(self, retrieved_memories: Dict[str, Any]) -> str:
        sections: List[str] = []

        task_specific = retrieved_memories.get("task_specific_skills", [])
        if task_specific:
            task_type = str(retrieved_memories.get("task_type", "task")).replace("_", " ").title()
            lines = [f"### {task_type} Skills"]
            for skill in task_specific:
                principle = skill.get("principle", "")
                when = skill.get("when_to_apply", "")
                lines.append(f"- {skill.get('title', '')}: {principle} When: {when}")
            sections.append("\n".join(lines))

        general = retrieved_memories.get("general_skills", [])
        if general:
            lines = ["### General Principles"]
            for skill in general:
                lines.append(f"- {skill.get('title', '')}: {skill.get('principle', '')}")
            sections.append("\n".join(lines))

        mistakes = retrieved_memories.get("mistakes_to_avoid", [])
        if mistakes:
            lines = ["### Mistakes To Avoid"]
            for mistake in mistakes:
                lines.append(f"- {mistake.get('description', '')}")
            sections.append("\n".join(lines))

        return "\n\n".join(section for section in sections if section.strip())
