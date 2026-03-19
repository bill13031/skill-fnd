from fake_news_skillrl.agent import HeuristicFakeNewsAgent, build_agent
from fake_news_skillrl.dataset import load_normalized_samples
from fake_news_skillrl.env import FakeNewsEnv
from fake_news_skillrl.memory import SkillsOnlyMemory
from fake_news_skillrl.trainer import RolloutTrainer, SFTDataBuilder


def test_sft_builder_and_rollout_trainer_end_to_end():
    samples = load_normalized_samples("data/raw/smoke_samples.jsonl")
    sft_rows = SFTDataBuilder().build(samples)
    assert len(sft_rows) == len(samples)

    memory = SkillsOnlyMemory("memory_data/fake_news/claude_style_skills.json")
    trainer = RolloutTrainer(env=FakeNewsEnv(memory=memory), agent=HeuristicFakeNewsAgent())
    results = trainer.run(samples)

    assert "metrics" in results
    assert "label_accuracy" in results["metrics"]
    assert "unverified_f1" not in results["metrics"]
    assert len(results["success"]) == len(samples)
    assert results["traces"][0]["observations"]
    assert results["traces"][0]["agent_debug"]


def test_build_agent_heuristic_factory():
    agent = build_agent(agent_type="heuristic")
    assert isinstance(agent, HeuristicFakeNewsAgent)


class _LoopingAgent:
    model_name = "looping-test"

    def next_action(self, sample, inspected_items, observation):
        del sample, inspected_items, observation
        return "<create>keep reasoning forever</create>"


def test_rollout_trainer_forces_verdict_after_reasoning_limit():
    samples = load_normalized_samples("data/raw/smoke_samples.jsonl")[:1]
    memory = SkillsOnlyMemory("memory_data/fake_news/claude_style_skills.json")
    trainer = RolloutTrainer(
        env=FakeNewsEnv(memory=memory),
        agent=_LoopingAgent(),
        max_reasoning_steps_before_forced_verdict=1,
    )
    results = trainer.run(samples)

    assert results["traces"][0]["actions"][0].startswith("<create>")
    assert results["traces"][0]["actions"][1].startswith("<verdict>")
