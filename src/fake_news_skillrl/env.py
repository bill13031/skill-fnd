from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from .memory import SkillsOnlyMemory
from .parser import ParsedAction, parse_action
from .prompting import build_step_prompt
from .schema import FakeNewsSample


@dataclass(slots=True)
class FakeNewsEnvConfig:
    max_steps: int = 4
    allow_unverified_label: bool = True
    require_evidence_before_verdict: bool = False
    invalid_action_penalty: float = -0.2
    correct_label_reward: float = 1.0
    wrong_label_penalty: float = -1.0
    evidence_match_reward: float = 0.2


@dataclass(slots=True)
class EpisodeState:
    sample: FakeNewsSample
    step_index: int = 0
    done: bool = False
    visible_evidence: List[str] = field(default_factory=list)
    inspected_items: List[str] = field(default_factory=list)
    final_verdict: Dict[str, Any] | None = None
    invalid_action_count: int = 0
    skill_prompt: str = ""


class FakeNewsEnv:
    def __init__(self, config: FakeNewsEnvConfig | None = None, memory: SkillsOnlyMemory | None = None) -> None:
        self.config = config or FakeNewsEnvConfig()
        self.memory = memory
        self._states: List[EpisodeState] = []

    def reset(self, samples: Sequence[FakeNewsSample]) -> List[str]:
        self._states = []
        observations: List[str] = []
        for sample in samples:
            skill_prompt = ""
            if self.memory is not None:
                retrieved = self.memory.retrieve(task_description=sample.task_description)
                skill_prompt = self.memory.format_for_prompt(retrieved)
            state = EpisodeState(sample=sample, skill_prompt=skill_prompt)
            self._states.append(state)
            observations.append(self._render_observation(state))
        return observations

    def step(self, actions: Sequence[str]) -> tuple[List[str], List[float], List[bool], List[Dict[str, Any]]]:
        if len(actions) != len(self._states):
            raise ValueError("Action count must match active episode count.")

        next_obs: List[str] = []
        rewards: List[float] = []
        dones: List[bool] = []
        infos: List[Dict[str, Any]] = []

        for state, action in zip(self._states, actions):
            parsed = parse_action(action, max_frame_index=max(0, len(state.sample.frames) - 1))
            reward, info = self._apply_action(state, parsed)
            if not state.done and state.step_index >= self.config.max_steps:
                state.done = True
                info.setdefault("termination_reason", "max_steps")
            next_obs.append(self._render_observation(state))
            rewards.append(reward)
            dones.append(state.done)
            infos.append(info)
        return next_obs, rewards, dones, infos

    def success_evaluator(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for state in self._states:
            label_correct = False
            evidence_match_rate = 0.0
            predicted_label = None
            if state.final_verdict is not None:
                predicted_label = state.final_verdict["label"]
                label_correct = predicted_label == state.sample.label
                evidence_match_rate = self._evidence_match_rate(
                    state.final_verdict.get("evidence", []),
                    state.sample.gold_evidence,
                )
            results.append(
                {
                    "sample_id": state.sample.sample_id,
                    "predicted_label": predicted_label,
                    "gold_label": state.sample.label,
                    "label_correct": label_correct,
                    "evidence_match_rate": evidence_match_rate,
                    "invalid_action_count": state.invalid_action_count,
                    "won": label_correct,
                }
            )
        return results

    def _render_observation(self, state: EpisodeState) -> str:
        visible = "\n\n".join(state.visible_evidence)
        return build_step_prompt(
            sample=state.sample,
            visible_evidence=visible,
            inspected_items=state.inspected_items,
            step_index=min(state.step_index + 1, self.config.max_steps),
            max_steps=self.config.max_steps,
            skill_prompt=state.skill_prompt,
        )

    def _apply_action(self, state: EpisodeState, parsed: ParsedAction) -> tuple[float, Dict[str, Any]]:
        if state.done:
            return 0.0, {"is_action_valid": False, "error": "Episode already complete.", "won": False}

        state.step_index += 1
        if not parsed.is_valid:
            state.invalid_action_count += 1
            return self.config.invalid_action_penalty, {
                "is_action_valid": False,
                "error": parsed.error,
                "won": False,
            }

        if parsed.action_type == "inspect":
            target = parsed.payload["target"]
            if target in state.inspected_items:
                state.invalid_action_count += 1
                return self.config.invalid_action_penalty, {
                    "is_action_valid": False,
                    "inspection_target": target,
                    "error": "Repeated inspection is not allowed.",
                    "won": False,
                }
            state.inspected_items.append(target)
            state.visible_evidence.append(self._inspection_text(state.sample, target))
            return 0.0, {"is_action_valid": True, "inspection_target": target, "won": False}

        if self.config.require_evidence_before_verdict and not state.inspected_items:
            state.invalid_action_count += 1
            return self.config.invalid_action_penalty, {
                "is_action_valid": False,
                "error": "At least one evidence inspection is required before verdict.",
                "won": False,
            }

        state.final_verdict = parsed.payload
        state.done = True
        reward = self._score_verdict(state)
        return reward, {
            "is_action_valid": True,
            "predicted_label": parsed.payload["label"],
            "evidence_match_rate": self._evidence_match_rate(parsed.payload.get("evidence", []), state.sample.gold_evidence),
            "won": parsed.payload["label"] == state.sample.label,
        }

    def _score_verdict(self, state: EpisodeState) -> float:
        assert state.final_verdict is not None
        label = state.final_verdict["label"]
        if label == "unverified" and not self.config.allow_unverified_label:
            return self.config.wrong_label_penalty

        reward = self.config.correct_label_reward if label == state.sample.label else self.config.wrong_label_penalty
        reward += self.config.evidence_match_reward * self._evidence_match_rate(
            state.final_verdict.get("evidence", []),
            state.sample.gold_evidence,
        )
        return reward

    def _inspection_text(self, sample: FakeNewsSample, target: str) -> str:
        if target == "post_text":
            return f"[post_text]\n{sample.post_text}"
        if target == "transcript":
            return f"[transcript]\n{sample.transcript}"
        if target == "ocr_text":
            return f"[ocr_text]\n{sample.ocr_text}"
        if target == "metadata":
            return f"[metadata]\n{sample.metadata}"
        if target.startswith("frame:"):
            index = int(target.split(":", 1)[1])
            frame = sample.frames[index]
            return f"[frame:{index}]\npath={frame.path}\ndescription={frame.description}"
        raise ValueError(f"Unsupported inspect target: {target}")

    @staticmethod
    def _evidence_match_rate(predicted_evidence: Sequence[str], gold_evidence: Sequence[str]) -> float:
        if not gold_evidence:
            return 0.0
        if not predicted_evidence:
            return 0.0
        normalized_gold = [item.lower() for item in gold_evidence]
        hits = 0
        for predicted in predicted_evidence:
            text = predicted.lower()
            if any(gold in text or text in gold for gold in normalized_gold):
                hits += 1
        return hits / max(1, len(gold_evidence))
