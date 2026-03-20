from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence

from .agent import BaseFakeNewsAgent, HeuristicFakeNewsAgent
from .env import FakeNewsEnv
from .metrics import compute_classification_metrics
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
            inspected: List[str] = []
            trajectory: List[Dict[str, str]] = []
            observation = sample.task_description
            while True:
                action = self.agent.next_action(sample, inspected, observation)
                trajectory.append({"role": "assistant", "content": action})
                if action.startswith("<visual_understanding>"):
                    inspected.append(f"visual_understanding: {action}")
                    continue
                if action.startswith("<create>"):
                    inspected.append(f"create: {action}")
                    continue
                if action.startswith("<check>"):
                    inspected.append(f"check: {action}")
                    continue
                if action.startswith("<use_skill>"):
                    inspected.append(f"use_skill: {action}")
                    continue
                break
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
        inspected_items = [[] for _ in samples]

        while any(active):
            current_observations = list(observations)
            actions: List[str] = []
            agent_debug_by_index: List[Dict[str, object]] = []
            for is_active, sample, inspected, observation in zip(active, samples, inspected_items, observations):
                if not is_active:
                    actions.append("<create>inactive</create>")
                    agent_debug_by_index.append({})
                    continue
                action = self.agent.next_action(sample, inspected, observation)
                agent_debug = getattr(self.agent, "get_last_debug", lambda: {})()
                if (
                    self.max_reasoning_steps_before_forced_verdict >= 0
                    and len(inspected) >= self.max_reasoning_steps_before_forced_verdict
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
                        if action.startswith("<visual_understanding>"):
                            inspected_items[index].append(f"visual_understanding: {action}")
                        elif action.startswith("<create>"):
                            inspected_items[index].append(f"create: {action}")
                        elif action.startswith("<check>"):
                            inspected_items[index].append(f"check: {action}")
                        elif action.startswith("<use_skill>"):
                            inspected_items[index].append(f"use_skill: {action}")
                active[index] = not dones[index]

        success = self.env.success_evaluator()
        metrics = compute_classification_metrics(success)
        return {
            "metrics": metrics,
            "success": success,
            "traces": [asdict(trace) for trace in traces],
        }
