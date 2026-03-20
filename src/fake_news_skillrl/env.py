from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from .memory import SkillsOnlyMemory
from .parser import parse_action
from .prompting import CONTROLLED_STAGES, build_stage_prompt
from .schema import FakeNewsSample


@dataclass(slots=True)
class FakeNewsEnvConfig:
    max_steps: int = 3
    invalid_action_penalty: float = -0.2
    correct_label_reward: float = 1.0
    wrong_label_penalty: float = -1.0


@dataclass(slots=True)
class EpisodeState:
    sample: FakeNewsSample
    step_index: int = 0
    done: bool = False
    stage_outputs: Dict[str, str] = field(default_factory=dict)
    final_verdict: Dict[str, Any] | None = None
    invalid_action_count: int = 0
    retrieved_skill_prompt: str = ""


class FakeNewsEnv:
    def __init__(self, config: FakeNewsEnvConfig | None = None, memory: SkillsOnlyMemory | None = None) -> None:
        self.config = config or FakeNewsEnvConfig()
        self.memory = memory
        self._states: List[EpisodeState] = []

    def reset(self, samples: Sequence[FakeNewsSample]) -> List[str]:
        self._states = [EpisodeState(sample=sample) for sample in samples]
        return [self._render_observation(state) for state in self._states]

    def step(self, actions: Sequence[str]) -> tuple[List[str], List[float], List[bool], List[Dict[str, Any]]]:
        if len(actions) != len(self._states):
            raise ValueError("Action count must match active episode count.")

        next_obs: List[str] = []
        rewards: List[float] = []
        dones: List[bool] = []
        infos: List[Dict[str, Any]] = []

        for state, action in zip(self._states, actions):
            reward, info = self._apply_action(state, action)
            next_obs.append(self._render_observation(state))
            rewards.append(reward)
            dones.append(state.done)
            infos.append(info)
        return next_obs, rewards, dones, infos

    def success_evaluator(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for state in self._states:
            predicted_label = None
            label_correct = False
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

    def _current_stage(self, state: EpisodeState) -> str:
        for stage in CONTROLLED_STAGES[:-1]:
            if stage not in state.stage_outputs:
                return stage
        return "verdict"

    def _current_role(self, state: EpisodeState) -> str:
        return "worker" if self._current_stage(state) == "worker_skill" else "analyzer"

    def _render_observation(self, state: EpisodeState) -> str:
        stage = self._current_stage(state)
        if stage == "worker_skill":
            state.retrieved_skill_prompt = self._retrieve_skills_for_state(state)
        prompt = build_stage_prompt(
            sample=state.sample,
            stage=stage,
            stage_outputs=state.stage_outputs,
            step_index=min(state.step_index + 1, self.config.max_steps),
            max_steps=self.config.max_steps,
            skill_prompt=state.retrieved_skill_prompt if stage == "worker_skill" else "",
        )
        return f"{prompt}\n## Current Role\n{self._current_role(state)}\n"

    def _retrieve_skills_for_state(self, state: EpisodeState) -> str:
        if self.memory is None:
            return ""
        retrieval_query = "\n".join(
            [
                state.sample.task_description,
                state.stage_outputs.get("analyzer_report", ""),
            ]
        )
        retrieved = self.memory.retrieve(task_description=retrieval_query)
        return self.memory.format_for_prompt(retrieved)

    def _apply_action(self, state: EpisodeState, action: str) -> tuple[float, Dict[str, Any]]:
        if state.done:
            return 0.0, {"is_action_valid": False, "error": "Episode already complete.", "won": False}

        stage = self._current_stage(state)
        role = self._current_role(state)
        state.step_index += 1

        if stage == "verdict":
            parsed = parse_action(action, max_frame_index=max(0, len(state.sample.frames) - 1))
            if not parsed.is_valid or parsed.action_type != "verdict":
                state.invalid_action_count += 1
                if state.step_index >= self.config.max_steps:
                    state.done = True
                return self.config.invalid_action_penalty, {
                    "is_action_valid": False,
                    "error": parsed.error if not parsed.is_valid else "Verdict stage requires a valid <verdict> block.",
                    "won": False,
                    **({"termination_reason": "max_steps"} if state.done else {}),
                }

            state.final_verdict = parsed.payload
            state.done = True
            reward = self.config.correct_label_reward if parsed.payload["label"] == state.sample.label else self.config.wrong_label_penalty
            return reward, {
                "is_action_valid": True,
                "predicted_label": parsed.payload["label"],
                "role": role,
                "stage": stage,
                "won": parsed.payload["label"] == state.sample.label,
            }

        normalized = self._normalize_stage_output(action, stage)
        if not normalized:
            state.invalid_action_count += 1
            if state.step_index >= self.config.max_steps:
                state.done = True
            return self.config.invalid_action_penalty, {
                "is_action_valid": False,
                "error": f"{stage} stage requires non-empty output.",
                "role": role,
                "won": False,
                **({"termination_reason": "max_steps"} if state.done else {}),
            }

        state.stage_outputs[stage] = normalized
        return 0.0, {
            "is_action_valid": True,
            "stage": stage,
            "role": role,
            "reasoning_action": stage,
            "won": False,
        }

    def _normalize_stage_output(self, action: str, stage: str) -> str:
        cleaned = (
            action.strip()
            .replace("<|im_end|>", "")
            .replace("<|endoftext|>", "")
            .strip()
        )
        if not cleaned:
            return ""

        parsed = parse_action(cleaned, max_frame_index=0)
        if parsed.is_valid:
            if parsed.action_type == "verdict":
                return ""
            return str(parsed.payload.get("content", "")).strip()

        json_like = cleaned
        if cleaned.startswith("```"):
            json_like = cleaned.strip("`").strip()
        if json_like.startswith("{") and json_like.endswith("}"):
            try:
                payload = json.loads(json_like)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                normalized = self._normalize_json_stage_output(payload, stage)
                if normalized:
                    return normalized

        for prefix in ("analyzer_report:", "worker_skill:", "visual_understanding:", "skill:"):
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                break
        return cleaned

    @staticmethod
    def _normalize_json_stage_output(payload: Dict[str, Any], stage: str) -> str:
        if stage == "analyzer_report":
            visual = str(payload.get("visual") or payload.get("visual_understanding") or "").strip()
            claim = str(payload.get("claim") or payload.get("claim_extraction") or "").strip()
            need = str(payload.get("need") or payload.get("request") or payload.get("skill_request") or "").strip()
            lines = []
            if visual:
                lines.append(f"Visual: {visual}")
            if claim:
                lines.append(f"Claim: {claim}")
            if need:
                lines.append(f"Need: {need}")
            return "\n".join(lines).strip()
        if stage == "worker_skill":
            skill = str(payload.get("skill") or payload.get("response") or payload.get("principle") or "").strip()
            if skill:
                return f"Skill: {skill}"
        return ""
