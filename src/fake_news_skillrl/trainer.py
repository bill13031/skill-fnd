from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from .agent import BaseFakeNewsAgent, HeuristicFakeNewsAgent, repair_verdict_output
from .env import FakeNewsEnv
from .metrics import compute_classification_metrics
from .prompting import build_stage_prompt
from .schema import FakeNewsSample


@dataclass(slots=True)
class EpisodeTrace:
    sample_id: str
    steps: List[Dict[str, object]]


class SFTDataBuilder:
    def __init__(
        self,
        analyzer_agent: BaseFakeNewsAgent | None = None,
        worker_agent: BaseFakeNewsAgent | None = None,
    ) -> None:
        self.analyzer_agent = analyzer_agent or HeuristicFakeNewsAgent(model_name="heuristic-analyzer")
        self.worker_agent = worker_agent or HeuristicFakeNewsAgent(model_name="heuristic-worker")

    def build(self, samples: Iterable[FakeNewsSample]) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for sample in samples:
            stage_outputs: Dict[str, str] = {}
            trajectory: List[Dict[str, str]] = []

            analyzer_observation = build_stage_prompt(
                sample=sample,
                stage="event_extraction",
                stage_outputs=stage_outputs,
                step_index=1,
                max_steps=4,
                skill_prompt="",
            ) + "\n## Current Role\nanalyzer\n"
            event_output = self.analyzer_agent.next_action(sample, [], analyzer_observation)
            trajectory.append({"role": "assistant", "content": event_output})
            stage_outputs["event_extraction"] = event_output

            preliminary_observation = build_stage_prompt(
                sample=sample,
                stage="preliminary_analysis",
                stage_outputs=stage_outputs,
                step_index=2,
                max_steps=4,
                skill_prompt="",
            ) + "\n## Current Role\nanalyzer\n"
            preliminary_output = self.analyzer_agent.next_action(sample, ["event_extraction"], preliminary_observation)
            trajectory.append({"role": "assistant", "content": preliminary_output})
            stage_outputs["preliminary_analysis"] = preliminary_output

            worker_observation = build_stage_prompt(
                sample=sample,
                stage="worker_skill",
                stage_outputs=stage_outputs,
                step_index=3,
                max_steps=4,
                skill_prompt="",
            ) + "\n## Current Role\nworker\n"
            worker_output = self.worker_agent.next_action(sample, ["event_extraction", "preliminary_analysis"], worker_observation)
            trajectory.append({"role": "assistant", "content": worker_output})
            stage_outputs["worker_skill"] = worker_output

            verdict_observation = build_stage_prompt(
                sample=sample,
                stage="verdict",
                stage_outputs=stage_outputs,
                step_index=4,
                max_steps=4,
                skill_prompt="",
            ) + "\n## Current Role\nanalyzer\n"
            verdict_output = self.analyzer_agent.next_action(
                sample,
                ["event_extraction", "preliminary_analysis", "worker_skill"],
                verdict_observation,
            )
            trajectory.append({"role": "assistant", "content": verdict_output})

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
        analyzer_agent: BaseFakeNewsAgent | None = None,
        worker_agent: BaseFakeNewsAgent | None = None,
        max_reasoning_steps_before_forced_verdict: int = 2,
    ) -> None:
        self.env = env
        self.analyzer_agent = analyzer_agent or HeuristicFakeNewsAgent(model_name="heuristic-analyzer")
        self.worker_agent = worker_agent or HeuristicFakeNewsAgent(model_name="heuristic-worker")
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
            for index, (is_active, sample, completed, observation) in enumerate(zip(active, samples, completed_stages, observations)):
                if not is_active:
                    actions.append("inactive")
                    agent_debug_by_index.append({})
                    continue

                stage = self.env._current_stage(self.env._states[index])
                agent = self.worker_agent if stage == "worker_skill" else self.analyzer_agent
                action = agent.next_action(sample, completed, observation)
                agent_debug = getattr(agent, "get_last_debug", lambda: {})()

                if (
                    stage == "verdict"
                    and self.max_reasoning_steps_before_forced_verdict >= 0
                    and len(completed) >= self.max_reasoning_steps_before_forced_verdict
                    and not action.startswith("<verdict>")
                ):
                    repaired_action = repair_verdict_output(action)
                    if repaired_action is not None:
                        action = repaired_action
                        agent_debug = {
                            **agent_debug,
                            "forced_verdict_used": False,
                            "verdict_repair_used": True,
                            "verdict_repair_action": repaired_action,
                            "verdict_repair_reason": "wrapped_or_extracted_bare_json_verdict",
                        }
                    else:
                        action = self._forced_verdict_agent._verdict_action(sample)
                        agent_debug = {
                            **agent_debug,
                            "forced_verdict_used": True,
                            "forced_verdict_action": action,
                            "forced_verdict_reason": "reached_verdict_stage_without_repairable_valid_verdict",
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
                            steps=[],
                        )
                    )
                if active[index]:
                    traces[index].steps.append(
                        {
                            "step": len(traces[index].steps) + 1,
                            "role": infos[index].get("role"),
                            "stage": infos[index].get("stage"),
                            "prompt": current_observations[index],
                            "model_generation": action,
                            "reward": rewards[index],
                            "info": infos[index],
                            "agent_debug": agent_debug_by_index[index],
                        }
                    )
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
            "traces": [
                {
                    "sample_id": trace.sample_id,
                    "steps": trace.steps,
                }
                for trace in traces
            ],
        }
