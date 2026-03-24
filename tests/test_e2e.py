from fake_news_skillrl.agent import AnalyzerAgent, HeuristicFakeNewsAgent, WorkerAgent, build_agent, build_agent_pair
from fake_news_skillrl.dataset import load_normalized_samples
from fake_news_skillrl.env import FakeNewsEnv
from fake_news_skillrl.memory import SkillsOnlyMemory
from fake_news_skillrl.trainer import RolloutTrainer, SFTDataBuilder


def test_sft_builder_and_rollout_trainer_end_to_end():
    samples = load_normalized_samples("data/raw/smoke_samples.jsonl")
    sft_rows = SFTDataBuilder().build(samples)
    assert len(sft_rows) == len(samples)

    memory = SkillsOnlyMemory("memory_data/fake_news/claude_style_skills.json")
    trainer = RolloutTrainer(
        env=FakeNewsEnv(memory=memory),
        analyzer_agent=HeuristicFakeNewsAgent(model_name="heuristic-analyzer"),
        worker_agent=HeuristicFakeNewsAgent(model_name="heuristic-worker"),
    )
    results = trainer.run(samples)

    assert "metrics" in results
    assert "label_accuracy" in results["metrics"]
    assert "unverified_f1" not in results["metrics"]
    assert "evidence_match_rate" not in results["metrics"]
    assert len(results["success"]) == len(samples)
    assert results["traces"][0]["steps"]
    assert results["traces"][0]["steps"][0]["prompt"]
    assert results["traces"][0]["steps"][0]["agent_debug"]


def test_build_agent_heuristic_factory():
    agent = build_agent(agent_type="heuristic")
    assert isinstance(agent, HeuristicFakeNewsAgent)


def test_build_agent_pair_returns_explicit_roles():
    analyzer_agent, worker_agent = build_agent_pair(agent_type="heuristic")
    assert isinstance(analyzer_agent, AnalyzerAgent)
    assert isinstance(worker_agent, WorkerAgent)


class _LoopingAgent:
    model_name = "looping-test"

    def next_action(self, sample, inspected_items, observation):
        del sample, observation
        if not inspected_items:
            return (
                "Preliminary reasoning: The extracted event needs a stronger credibility check before final judgment.\n"
                "Need: Need a verification skill."
            )
        return "still not a verdict"


def test_rollout_trainer_forces_verdict_after_reasoning_limit():
    samples = load_normalized_samples("data/raw/smoke_samples.jsonl")[:1]
    memory = SkillsOnlyMemory("memory_data/fake_news/claude_style_skills.json")
    trainer = RolloutTrainer(
        env=FakeNewsEnv(memory=memory),
        analyzer_agent=_LoopingAgent(),
        worker_agent=_LoopingAgent(),
        max_reasoning_steps_before_forced_verdict=2,
    )
    results = trainer.run(samples)

    assert results["traces"][0]["steps"][0]["model_generation"].startswith("Preliminary reasoning:")
    assert results["traces"][0]["steps"][2]["model_generation"].startswith("<verdict>")
