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
    actions: List[str]
    rewards: List[float]
    infos: List[Dict[str, object]]


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
    def __init__(self, env: FakeNewsEnv, agent: BaseFakeNewsAgent | None = None) -> None:
        self.env = env
        self.agent = agent or HeuristicFakeNewsAgent()

    def run(self, samples: Sequence[FakeNewsSample]) -> Dict[str, object]:
        traces: List[EpisodeTrace] = []
        observations = self.env.reset(samples)

        active = [True for _ in samples]
        inspected_items = [[] for _ in samples]

        while any(active):
            actions: List[str] = []
            for is_active, sample, inspected, observation in zip(active, samples, inspected_items, observations):
                if not is_active:
                    actions.append("<create>inactive</create>")
                    continue
                action = self.agent.next_action(sample, inspected, observation)
                actions.append(action)

            observations, rewards, dones, infos = self.env.step(actions)
            for index, action in enumerate(actions):
                if len(traces) <= index:
                    traces.append(
                        EpisodeTrace(
                            sample_id=samples[index].sample_id,
                            actions=[],
                            rewards=[],
                            infos=[],
                        )
                    )
                if active[index]:
                    traces[index].actions.append(action)
                    traces[index].rewards.append(rewards[index])
                    traces[index].infos.append(infos[index])
                    if infos[index].get("is_action_valid"):
                        if action.startswith("<create>"):
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
