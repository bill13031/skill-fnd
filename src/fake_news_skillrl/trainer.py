from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence

from .agent import BaseFakeNewsAgent, HeuristicFakeNewsAgent
from .env import FakeNewsEnv
from .metrics import compute_classification_metrics
from .prompting import build_stage_prompt
from .schema import FakeNewsSample


@dataclass(slots=True)
class EpisodeTrace:
    sample_id: str
    observations: List[str]
    actions: List[str]
    rewards: List[float]
    infos: List[Dict[str, object]]
    agent_debug: List[Dict[str, object]]


class SFTDataBuilder:
    def __init__(self, agent: BaseFakeNewsAgent | None = None) -> None:
        self.agent = agent or HeuristicFakeNewsAgent()

    def build(self, samples: Iterable[FakeNewsSample]) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for sample in samples:
            completed_stages: List[str] = []
            stage_outputs: Dict[str, str] = {}
            trajectory: List[Dict[str, str]] = []
            for step_index, stage in enumerate(["visual_understanding", "claim_extraction", "consistency_check", "skill_application"], start=1):
                observation = build_stage_prompt(
                    sample=sample,
                    stage=stage,
                    stage_outputs=stage_outputs,
                    step_index=step_index,
                    max_steps=5,
                    skill_prompt="",
                )
                action = self.agent.next_action(sample, completed_stages, observation)
                trajectory.append({"role": "assistant", "content": action})
                stage_outputs[stage] = action
                completed_stages.append(stage)
            verdict_observation = build_stage_prompt(
                sample=sample,
                stage="verdict",
                stage_outputs=stage_outputs,
                step_index=5,
                max_steps=5,
                skill_prompt="",
            )
            action = self.agent.next_action(sample, completed_stages, verdict_observation)
            trajectory.append({"role": "assistant", "content": action})
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "prompt": sample.task_description,
                    "messages": trajectory,
                    "label": sample.label,
                    "gold_evidence": sample.gold_evidence,
                    "data_source": sample.data_source,
                }
            )
        return rows


class RolloutTrainer:
    def __init__(
        self,
        env: FakeNewsEnv,
        agent: BaseFakeNewsAgent | None = None,
        max_reasoning_steps_before_forced_verdict: int = 4,
    ) -> None:
        self.env = env
        self.agent = agent or HeuristicFakeNewsAgent()
        self.max_reasoning_steps_before_forced_verdict = max_reasoning_steps_before_forced_verdict
        self._forced_verdict_agent = HeuristicFakeNewsAgent(model_name="forced-verdict")

    def run(self, samples: Sequence[FakeNewsSample]) -> Dict[str, object]:
        traces: List[EpisodeTrace] = []
        observations = self.env.reset(samples)

        active = [True for _ in samples]
        completed_stages = [[] for _ in samples]

        while any(active):
            current_observations = list(observations)
            actions: List[str] = []
            agent_debug_by_index: List[Dict[str, object]] = []
            for is_active, sample, completed, observation in zip(active, samples, completed_stages, observations):
                if not is_active:
                    actions.append("<create>inactive</create>")
                    agent_debug_by_index.append({})
                    continue
                action = self.agent.next_action(sample, completed, observation)
                agent_debug = getattr(self.agent, "get_last_debug", lambda: {})()
                if (
                    self.max_reasoning_steps_before_forced_verdict >= 0
                    and len(completed) >= self.max_reasoning_steps_before_forced_verdict
                    and not action.startswith("<verdict>")
                ):
                    action = self._forced_verdict_agent._verdict_action(sample)
                    agent_debug = {
                        **agent_debug,
                        "forced_verdict_used": True,
                        "forced_verdict_action": action,
                        "forced_verdict_reason": (
                            "reached_max_reasoning_steps_before_forced_verdict"
                        ),
                    }
                else:
                    agent_debug = {
                        **agent_debug,
                        "forced_verdict_used": False,
                    }
                actions.append(action)
                agent_debug_by_index.append(agent_debug)

            observations, rewards, dones, infos = self.env.step(actions)
            for index, action in enumerate(actions):
                if len(traces) <= index:
                    traces.append(
                        EpisodeTrace(
                            sample_id=samples[index].sample_id,
                            observations=[],
                            actions=[],
                            rewards=[],
                            infos=[],
                            agent_debug=[],
                        )
                    )
                if active[index]:
                    traces[index].observations.append(current_observations[index])
                    traces[index].actions.append(action)
                    traces[index].rewards.append(rewards[index])
                    traces[index].infos.append(infos[index])
                    traces[index].agent_debug.append(agent_debug_by_index[index])
                    if infos[index].get("is_action_valid"):
                        stage = str(infos[index].get("stage", "")).strip()
                        if stage:
                            completed_stages[index].append(stage)
                active[index] = not dones[index]

        success = self.env.success_evaluator()
        metrics = compute_classification_metrics(success)
        return {
            "metrics": metrics,
            "success": success,
            "traces": [asdict(trace) for trace in traces],
        }
