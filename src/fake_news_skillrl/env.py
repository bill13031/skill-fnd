from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from .memory import SkillsOnlyMemory
from .parser import ParsedAction, parse_action
from .prompting import build_step_prompt
from .schema import FakeNewsSample


@dataclass(slots=True)
class FakeNewsEnvConfig:
    max_steps: int = 5
    require_evidence_before_verdict: bool = False
    invalid_action_penalty: float = -0.2
    correct_label_reward: float = 1.0
    wrong_label_penalty: float = -1.0


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
            state = EpisodeState(
                sample=sample,
                skill_prompt=skill_prompt,
                visible_evidence=self._default_visible_evidence(sample),
            )
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
            predicted_label = None
            if state.final_verdict is not None:
                predicted_label = state.final_verdict["label"]
                label_correct = predicted_label == state.sample.label
            results.append(
                {
                    "sample_id": state.sample.sample_id,
                    "predicted_label": predicted_label,
                    "gold_label": state.sample.label,
                    "label_correct": label_correct,
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
            allowed_actions=self._allowed_action_types(state),
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

        allowed_actions = self._allowed_action_types(state)
        if parsed.action_type not in allowed_actions:
            state.invalid_action_count += 1
            return self.config.invalid_action_penalty, {
                "is_action_valid": False,
                "error": f"Action '{parsed.action_type}' is not allowed now. Allowed action(s): {', '.join(allowed_actions)}.",
                "won": False,
            }

        if parsed.action_type in {"visual_understanding", "create", "check", "use_skill"}:
            if state.inspected_items:
                last_action_type = state.inspected_items[-1].split(":", 1)[0]
                if last_action_type == parsed.action_type:
                    state.invalid_action_count += 1
                    return self.config.invalid_action_penalty, {
                        "is_action_valid": False,
                        "error": "Do not repeat the same action type on consecutive turns.",
                        "won": False,
                    }
            content = parsed.payload["content"]
            state.inspected_items.append(f"{parsed.action_type}: {content}")
            return 0.0, {
                "is_action_valid": True,
                "reasoning_action": parsed.action_type,
                "won": False,
            }

        if self.config.require_evidence_before_verdict and not state.inspected_items:
            state.invalid_action_count += 1
            return self.config.invalid_action_penalty, {
                "is_action_valid": False,
                "error": "At least one reasoning action is required before verdict.",
                "won": False,
            }

        state.final_verdict = parsed.payload
        state.done = True
        reward = self._score_verdict(state)
        return reward, {
            "is_action_valid": True,
            "predicted_label": parsed.payload["label"],
            "won": parsed.payload["label"] == state.sample.label,
        }

    def _score_verdict(self, state: EpisodeState) -> float:
        assert state.final_verdict is not None
        label = state.final_verdict["label"]
        return self.config.correct_label_reward if label == state.sample.label else self.config.wrong_label_penalty

    def _default_visible_evidence(self, sample: FakeNewsSample) -> List[str]:
        evidence = [f"[post_text]\n{sample.post_text}"]
        if sample.transcript.strip():
            evidence.append(f"[transcript]\n{sample.transcript}")
        if sample.ocr_text.strip():
            evidence.append(f"[ocr_text]\n{sample.ocr_text}")
        if sample.frames:
            evidence.append(f"[attached_frames]\n{len(sample.frames)} frame image(s) are attached directly to the model input.")
        return evidence

    @staticmethod
    def _allowed_action_types(state: EpisodeState) -> List[str]:
        history_types = [item.split(":", 1)[0] for item in state.inspected_items]
        if not history_types:
            return ["visual_understanding"]
        if history_types == ["visual_understanding"]:
            return ["create"]
        if history_types == ["visual_understanding", "create"]:
            return ["check"]
        if history_types == ["visual_understanding", "create", "check"]:
            return ["use_skill", "verdict"]
        if history_types == ["visual_understanding", "create", "check", "use_skill"]:
            return ["verdict"]
        return ["verdict"]
